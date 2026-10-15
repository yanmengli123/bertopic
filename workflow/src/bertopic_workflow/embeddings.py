"""Single-space, aligned embeddings with token-aware pooling and resumable cache."""
from __future__ import annotations

from email.utils import parsedate_to_datetime
import hashlib
import os
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import requests

from .common import WorkflowError, json_bytes, read_json, sha256, write_json


def normalize(vectors, dimension=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or (dimension and vectors.shape[1] != dimension) or not np.isfinite(vectors).all():
        raise WorkflowError('EMBEDDING_SHAPE', '向量数量、维度或数值无效。')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise WorkflowError('EMBEDDING_ZERO', '嵌入中出现零向量。')
    return vectors / norms


def split_text(text, tokenizer, limit):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) <= limit:
        return [(text, max(1, len(encoded.ids)))]
    pieces, start = [], 0
    for last in range(limit, len(encoded.ids) + limit, limit):
        last = min(last, len(encoded.ids))
        end = len(text) if last == len(encoded.ids) else encoded.offsets[last - 1][1]
        # Re-tokenizing a substring can change tokenization at the boundaries.
        part = text[start:end]
        count = len(tokenizer.encode(part, add_special_tokens=False).ids)
        if count > limit:
            # Split by characters until the tokenizer confirms the bound.
            middle = len(part) // 2
            if middle == 0:
                raise WorkflowError('TOKEN_LIMIT', '单个字符超出模型 token 限制。')
            pieces.extend(split_text(part[:middle], tokenizer, limit))
            pieces.extend(split_text(part[middle:], tokenizer, limit))
        elif part:
            pieces.append((part, max(1, count)))
        start = end
        if last == len(encoded.ids):
            break
    if ''.join(p for p, _ in pieces) != text:
        raise WorkflowError('TOKEN_ALIGNMENT', '文本分块未完整覆盖原文。')
    return pieces


class Cache:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, data BLOB NOT NULL, sha256 TEXT NOT NULL)')
        self.connection.commit()

    def get(self, key):
        row = self.connection.execute('SELECT dimension,data,sha256 FROM vectors WHERE key=?', (key,)).fetchone()
        if not row:
            return None
        vector = np.frombuffer(row[1], dtype='<f4').copy()
        if vector.shape != (row[0],) or not np.isfinite(vector).all() or hashlib.sha256(row[1]).hexdigest() != row[2]:
            raise WorkflowError('CACHE_INVALID', '缓存向量损坏，请使用新的缓存目录。')
        return vector

    def put(self, key, vector):
        vector = np.asarray(vector, dtype='<f4')
        data = vector.tobytes()
        self.connection.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)', (key, len(vector), data, hashlib.sha256(data).hexdigest()))
        self.connection.commit()

    def close(self):
        self.connection.close()


def retry_seconds(value, attempt):
    try:
        wait = float(value)
    except (TypeError, ValueError):
        try:
            wait = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            wait = 2**attempt
    return min(55, max(0, wait))


def request_vectors(options, texts, check_cancel, session=None):
    secret = os.environ.get(options.api_key_env)
    if not secret:
        raise WorkflowError('CREDENTIAL_MISSING', '远程嵌入需要宿主注入凭据环境变量。', details={'environment': options.api_key_env})
    client = session or requests.Session()
    try:
        for attempt in range(options.retries + 1):
            check_cancel()
            try:
                response = client.post(options.base_url.rstrip('/') + '/embeddings',
                    headers={'Authorization': 'Bearer ' + secret},
                    json={'model': options.model, 'input': texts, 'encoding_format': 'float'},
                    timeout=(10, options.timeout_seconds), allow_redirects=False)
            except requests.RequestException:
                if attempt == options.retries:
                    raise WorkflowError('API_NETWORK', '嵌入网络请求失败，未记录凭据或服务端响应。') from None
                response = None
            if response is not None and response.status_code == 200:
                try:
                    payload = response.json()
                    ordered = sorted(payload['data'], key=lambda x: x['index'])
                    if [x['index'] for x in ordered] != list(range(len(texts))):
                        raise ValueError('indices')
                    if payload.get('model', options.model) != options.model:
                        raise ValueError('model')
                    vectors = np.asarray([x['embedding'] for x in ordered], dtype=np.float32)
                    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
                        raise ValueError('vectors')
                    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
                        raise ValueError('zero vector')
                    if options.dimension and vectors.shape[1] != options.dimension:
                        raise ValueError('dimension')
                    return vectors
                except (KeyError, ValueError, TypeError):
                    raise WorkflowError('API_RESPONSE', '服务端响应的索引、模型或向量不符合约定。') from None
            status = response.status_code if response is not None else None
            if status is not None and status not in {429, 500, 502, 503, 504}:
                raise WorkflowError('API_HTTP', '嵌入接口返回不可重试错误。', details={'status': status})
            if attempt == options.retries:
                raise WorkflowError('API_RETRIES', '嵌入接口重试次数已耗尽。', details={'status': status})
            wait = retry_seconds(response.headers.get('Retry-After') if response is not None else None, attempt + 1)
            until = time.monotonic() + wait
            while time.monotonic() < until:
                check_cancel()
                time.sleep(min(.25, max(0, until - time.monotonic())))
    finally:
        if session is None:
            client.close()


def embed(config, ingest_dir, destination, check_cancel=lambda: None, progress=lambda **kwargs: None):
    options = config.embedding
    corpus = read_json(Path(ingest_dir) / 'corpus.json')
    ids, hashes, texts = ([r[field] for r in corpus] for field in ['record_id', 'text_sha256', 'text'])
    dimension = options.dimension or (1024 if options.model in {'BAAI/bge-m3', 'BAAI/bge-large-zh-v1.5'} else None)
    if options.backend == 'precomputed':
        metadata = read_json(options.metadata_path)
        if metadata.get('record_ids') != ids or metadata.get('text_sha256') != hashes or metadata.get('model') != options.model:
            raise WorkflowError('EMBEDDING_ALIGNMENT', '预计算向量必须逐条匹配 record_ids、text_sha256 和 model。')
        vectors = np.load(options.vectors_path, allow_pickle=False)
        if len(vectors) != len(corpus):
            raise WorkflowError('EMBEDDING_ALIGNMENT', '预计算向量行数不匹配。')
        vectors = normalize(vectors, dimension)
        chunks = [None] * len(corpus)
        namespace = {'backend': 'precomputed', 'model': options.model,
                     'vectors_sha256': sha256(options.vectors_path), 'metadata_sha256': sha256(options.metadata_path)}
        hits, requests_made = 0, 0
    else:
        from tokenizers import Tokenizer
        tokenizer_path = options.tokenizer_path
        if not tokenizer_path and options.backend == 'local' and (Path(options.model) / 'tokenizer.json').is_file():
            tokenizer_path = str(Path(options.model) / 'tokenizer.json')
        if not tokenizer_path:
            from huggingface_hub import hf_hub_download
            try:
                tokenizer_path = hf_hub_download(options.tokenizer_id or options.model, 'tokenizer.json',
                    revision=options.model_revision, cache_dir=str(Path(config.cache_root) / 'tokenizers'),
                    local_files_only=options.offline or not options.allow_model_download)
            except Exception:
                raise WorkflowError('TOKENIZER_MISSING', '无法取得 tokenizer.json；可配置本地 tokenizer_path。') from None
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        limit = options.max_tokens or (7200 if options.model == 'BAAI/bge-m3' else 480)
        namespace = {'backend': options.backend, 'endpoint': options.base_url, 'model': options.model,
                     'revision': options.model_revision, 'dimension': dimension,
                     'tokenizer_sha256': sha256(tokenizer_path), 'max_tokens': limit,
                     'pooling': 'token_weighted_raw_mean_then_l2_v1'}
        namespace_key = hashlib.sha256(json_bytes(namespace)).hexdigest()
        unique = list(dict.fromkeys(texts))
        pooled, chunk_counts, hits, requests_made = {}, {}, 0, 0
        local_model = None
        cache = Cache(Path(config.cache_root) / 'embedding_cache.sqlite3')
        try:
            for first_doc in range(0, len(unique), options.batch_size):
                check_cancel()
                pending = []
                batch_docs = unique[first_doc:first_doc + options.batch_size]
                for text in batch_docs:
                    parts = split_text(text, tokenizer, limit)
                    chunk_counts[text] = len(parts)
                    cache_key = hashlib.sha256((namespace_key + '\0' + text).encode()).hexdigest()
                    cached = cache.get(cache_key)
                    if cached is not None:
                        pooled[text] = cached
                        hits += 1
                    else:
                        if options.offline and options.backend != 'local':
                            raise WorkflowError('OFFLINE_CACHE_MISS', '离线缓存缺少当前模型或文本的向量。')
                        pending.append((text, parts, cache_key))
                flat = [part for _, parts, _ in pending for part in parts]
                chunk_vectors = []
                for first in range(0, len(flat), options.batch_size):
                        check_cancel()
                        batch = [p[0] for p in flat[first:first + options.batch_size]]
                        if options.backend == 'local':
                            if local_model is None:
                                try:
                                    from sentence_transformers import SentenceTransformer
                                    local_model = SentenceTransformer(options.model, device=options.device,
                                        revision=options.model_revision, trust_remote_code=False,
                                        local_files_only=options.offline or not options.allow_model_download,
                                        cache_folder=str(Path(config.cache_root) / 'local_models'))
                                except Exception:
                                    raise WorkflowError('LOCAL_MODEL', '本地模型加载失败；检查 local 依赖、设备和模型缓存。') from None
                            if limit + 2 > local_model.max_seq_length:
                                raise WorkflowError('TOKEN_LIMIT', 'max_tokens 高于本地模型最大序列长度，请降低该配置。')
                            chunk_vectors.extend(local_model.encode(batch, batch_size=options.batch_size,
                                                   convert_to_numpy=True, show_progress_bar=False))
                        else:
                            chunk_vectors.extend(request_vectors(options, batch, check_cancel))
                            requests_made += 1
                chunk_vectors = np.asarray(chunk_vectors, dtype=np.float32)
                offset = 0
                for text, parts, cache_key in pending:
                    vector = normalize(np.average(chunk_vectors[offset:offset + len(parts)], axis=0,
                                                  weights=[p[1] for p in parts])[None, :], dimension)[0]
                    offset += len(parts)
                    if dimension is None:
                        dimension = len(vector)
                    if len(vector) != dimension:
                        raise WorkflowError('EMBEDDING_SHAPE', '当前任务出现不同维度向量，禁止混合。')
                    cache.put(cache_key, vector)
                    pooled[text] = vector
                for text in batch_docs:
                    if dimension is None:
                        dimension = len(pooled[text])
                    if len(pooled[text]) != dimension:
                        raise WorkflowError('EMBEDDING_SHAPE', '缓存中出现不同维度向量，禁止混合。')
                progress(completed=first_doc + len(batch_docs), total=len(unique), cache_hits=hits, requests=requests_made)
            vectors = normalize(np.stack([pooled[t] for t in texts]), dimension)
            chunks = [chunk_counts[t] for t in texts]
        finally:
            cache.close()
    check_cancel()
    np.save(Path(destination) / 'embeddings.npy', vectors, allow_pickle=False)
    metadata = {'record_ids': ids, 'text_sha256': hashes, 'model': options.model,
                'dimension': vectors.shape[1], 'records': len(corpus), 'normalized': True,
                'namespace': namespace, 'chunk_counts': chunks, 'cache_hits': hits,
                'requests_this_attempt': requests_made, 'vectors_sha256': sha256(Path(destination) / 'embeddings.npy')}
    write_json(Path(destination) / 'embedding_metadata.json', metadata)
    return {'records': len(corpus), 'dimension': vectors.shape[1], 'cache_hits': hits, 'requests': requests_made}
