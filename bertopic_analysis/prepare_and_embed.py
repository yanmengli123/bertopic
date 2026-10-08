"""Normalize four bibliographic exports and cache SiliconFlow embeddings.

All input rows remain in the corpus, including repeated WoS records.
Credentials are read exclusively from SILICONFLOW_API_KEY.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CACHE = ROOT / "cache"
ENDPOINT = "https://api.siliconflow.cn/v1/embeddings"
MODELS = {"joint_bge_m3": "BAAI/bge-m3", "chinese_bge_zh": "BAAI/bge-large-zh-v1.5"}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def clean(value):
    return re.sub(r"\s+", " ", str(value if pd.notna(value) else "")).strip()


def clean_text(value):
    text = html.unescape(clean(value))
    text = re.sub(r"<正>|</?[A-Za-z][^>]*>", " ", text)
    # Remove only an explicit copyright footer, never scientific vocabulary.
    text = re.sub(r"\s+(?:Copyright\s+(?:©|\(C\)|\d{4})|©\s*\d{4}|\(C\)\s*\d{4}).*$", "", text, flags=re.I)
    return clean(text)


def normalized_title(value):
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", clean(value)).casefold())


def normalized_doi(value):
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", clean(value).casefold()).rstrip(".")
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else ""


def prepare(input_dir: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cnki_file = input_dir / "CNKI.xls"
    cnki = pd.read_html(io.StringIO(cnki_file.read_text(encoding="utf-8-sig")))[0]
    cnki.columns = cnki.iloc[0].astype(str)
    cnki = cnki.iloc[1:].fillna("")
    wanfang = pd.read_excel(input_dir / "WanFangdata.xlsx", header=1, dtype=str).fillna("")
    pubmed = pd.read_csv(input_dir / "pubmed_merged_bertopic.csv", dtype=str, keep_default_na=False)
    wos = pd.read_csv(input_dir / "wos.csv", dtype=str, keep_default_na=False)
    specs = [
        ("CNKI", cnki, "CNKI.xls", {"title": "Title-题名", "abstract": "Summary-摘要", "year": "Year-年", "date": "PubTime-发表时间", "doi": "DOI-DOI", "journal": "Source-文献来源", "authors": "Author-作者", "document_type": "SrcDatabase-来源库"}, 2),
        ("WanFang", wanfang, "WanFangdata.xlsx", {"title": "题名", "abstract": "摘要", "year": "年", "doi": "DOI", "journal": "刊名", "authors": "作者"}, 3),
        ("PubMed", pubmed, "pubmed_merged_bertopic.csv", {"title": "title", "abstract": "abstract", "year": "year", "doi": "doi", "journal": "journal", "authors": "authors", "document_type": "publication_types", "pmid": "pmid"}, 2),
        ("WoS", wos, "wos.csv", {"title": "Article Title", "abstract": "Abstract", "year": "Publication Year", "doi": "DOI", "journal": "Source Title", "authors": "Authors", "document_type": "Document Type", "wos_id": "UT (Unique WOS ID)"}, 2),
    ]
    rows = []
    for source, frame, filename, mapping, first_row in specs:
        for position, (_, source_row) in enumerate(frame.iterrows()):
            raw = {name: clean(source_row.get(column, "")) for name, column in mapping.items()}
            title = clean_text(raw["title"])
            abstract = clean_text(raw["abstract"])
            if not title and not abstract:
                raise ValueError(f"Empty document: {source} row {position + first_row}")
            text = title + ((" " if title.endswith((".", "!", "?", "。", "！", "？")) else ". ") + abstract if abstract else "")
            year_match = re.search(r"\b(?:18|19|20)\d{2}\b", raw.get("year", ""))
            year_source = mapping.get("year", "")
            if not year_match and source == "CNKI":
                year_match = re.search(r"\b(?:18|19|20)\d{2}\b", raw.get("date", ""))
                year_source = "PubTime-发表时间"
            year = int(year_match.group()) if year_match else None
            zh_chars = len(re.findall(r"[\u3400-\u9fff]", text))
            latin_chars = len(re.findall(r"[A-Za-z]", text))
            language = "zh" if zh_chars / max(1, zh_chars + latin_chars) > 0.20 else "en"
            rice = bool(re.search(r"\b(?:rice|oryza)\b|水稻|稻米|稻谷", text, re.I))
            endosperm = bool(re.search(r"\b(?:endosperm|aleurone)\b|胚乳|糊粉", text, re.I))
            rows.append({
                "record_id": f"{source}_{position + 1:06d}", "source": source,
                "source_file": filename, "source_row": position + first_row,
                "title": title, "abstract": abstract, "text": text,
                "has_abstract": int(bool(abstract)), "year": year,
                "year_raw": raw.get("year", ""), "year_source": year_source,
                "year_review_required": int(year is None or year > 2026),
                "language_of_text": language, "doi": normalized_doi(raw.get("doi", "")),
                "journal": raw.get("journal", ""), "authors": raw.get("authors", ""),
                "document_type": raw.get("document_type", "期刊" if source == "WanFang" else ""),
                "wos_id": raw.get("wos_id", ""), "pmid": raw.get("pmid", ""),
                "rice_endosperm_term_match": int(rice and endosperm),
                "title_key": normalized_title(title),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
    df = pd.DataFrame(rows)
    for key in ("doi", "title_key", "text_sha256", "wos_id"):
        counts = df.loc[df[key].ne(""), key].value_counts()
        df[f"{key}_record_count"] = df[key].map(counts).fillna(0).astype(int)
    assert len(df) == 22549, f"Unexpected corpus size: {len(df)}"
    assert df.record_id.is_unique and df.text.str.len().gt(0).all()
    assert df.loc[df.source.eq("WoS")].shape[0] == 20092
    df.to_json(OUT / "corpus.json", orient="records", force_ascii=False, indent=None)
    source_summary = df.groupby("source").agg(records=("record_id", "size"), with_abstract=("has_abstract", "sum"), chinese_texts=("language_of_text", lambda x: int(x.eq("zh").sum())), year_min=("year", "min"), year_max=("year", "max")).reset_index()
    source_summary["missing_abstract"] = source_summary.records - source_summary.with_abstract
    sources = {filename: hashlib.sha256((input_dir / filename).read_bytes()).hexdigest() for _, _, filename, _, _ in specs}
    manifest = {
        "rows": len(df), "sources": json.loads(source_summary.to_json(orient="records")),
        "source_sha256": sources, "input_directory": str(input_dir),
        "unique_texts": int(df.text_sha256.nunique()), "missing_abstracts": int(df.has_abstract.eq(0).sum()),
        "chinese_texts": int(df.language_of_text.eq("zh").sum()),
        "wos_rows": 20092, "unique_wos_ids": int(df.loc[df.source.eq("WoS"), "wos_id"].nunique()),
        "all_records_retained": True, "deduplication": "NONE; duplicates are flagged and retained in modeling and counts",
        "scope_filter": "NONE; all supplied subjects retained; term-match flag is diagnostic only",
        "year_review_records": df.loc[df.year_review_required.eq(1), ["record_id", "title", "year", "year_raw"]].to_dict("records"),
        "text_cleaning": "Decode HTML entities, remove HTML/<正> markers and explicit copyright footers, normalize whitespace. Retain scientific punctuation, case, numbers and gene identifiers.",
        "text_rule": "Title + abstract; title only for records without abstract",
        "language_rule": "Actual text: Chinese characters / (Chinese + Latin letters) > 0.20 => zh. Article language metadata is not used.",
        "model_plan": {"joint": "BAAI/bge-m3, entire four-database corpus", "chinese_validation": "BAAI/bge-large-zh-v1.5, Chinese text subset; separate vector space"},
        "analysis_date": "2026-10-08",
    }
    write_json(OUT / "data_manifest.json", manifest)
    print(json.dumps({"stage": "prepared", "rows": len(df), "unique_texts": manifest["unique_texts"], "chinese_texts": manifest["chinese_texts"]}), flush=True)
    return df


class EmbeddingCache:
    def __init__(self):
        self.connection = sqlite3.connect(CACHE / "embeddings.sqlite3")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, model TEXT, vector BLOB, dimension INTEGER)")
        self.connection.commit()

    @staticmethod
    def key(model, text):
        return hashlib.sha256((model + "\0" + text).encode("utf-8")).hexdigest()

    def load(self, model):
        return {key: np.frombuffer(vector, dtype="<f4").copy() for key, vector in self.connection.execute("SELECT cache_key, vector FROM embeddings WHERE model=?", (model,))}

    def save(self, model, texts, vectors):
        self.connection.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)", [(self.key(model, text), model, np.asarray(vector, dtype="<f4").tobytes(), len(vector)) for text, vector in zip(texts, vectors)])
        self.connection.commit()


def tokenizer_for(model):
    destination = CACHE / (model.replace("/", "_") + "_tokenizer.json")
    if not destination.exists():
        response = requests.get(f"https://huggingface.co/{model}/resolve/main/tokenizer.json", timeout=(15, 90))
        response.raise_for_status()
        # Tokenizer data contains no credentials and is fetched from the model's author.
        destination.write_bytes(response.content)
    tokenizer = Tokenizer.from_file(str(destination))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def chunks_for(text, tokenizer, max_tokens):
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) <= max_tokens:
        return [(text, max(1, len(encoding.ids)))]
    chunks = []
    start = 0
    for last in range(max_tokens, len(encoding.ids) + max_tokens, max_tokens):
        last = min(last, len(encoding.ids))
        end = len(text) if last == len(encoding.ids) else encoding.offsets[last - 1][1]
        piece = text[start:end]
        count = len(tokenizer.encode(piece, add_special_tokens=False).ids)
        if count > max_tokens + 8:
            raise ValueError("Unexpected token expansion while chunking")
        chunks.append((piece, max(1, count)))
        start = end
        if last == len(encoding.ids):
            break
    assert "".join(piece for piece, _ in chunks) == text
    return chunks


class APIClient:
    def __init__(self):
        self.key = os.environ.get("SILICONFLOW_API_KEY", "")
        if not self.key:
            raise RuntimeError("Set SILICONFLOW_API_KEY in the environment; do not put the key in files.")
        self.lock = threading.Lock()
        self.cooldown = 0.0
        self.requests = 0
        self.retries = 0
        self.tokens = 0

    def request(self, model, texts):
        attempt = 0
        while attempt < 12:
            with self.lock:
                pause = max(0, self.cooldown - time.monotonic())
            if pause:
                time.sleep(min(pause, 55))
                continue
            attempt += 1
            try:
                response = requests.post(ENDPOINT, headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}, json={"model": model, "input": texts, "encoding_format": "float"}, timeout=(15, 150))
            except requests.RequestException as error:
                with self.lock:
                    self.retries += 1
                time.sleep(min(2 ** attempt, 30))
                if attempt == 12:
                    raise RuntimeError("SiliconFlow network request failed after retries") from error
                continue
            with self.lock:
                self.requests += 1
            if response.status_code == 200:
                data = response.json()
                ordered = sorted(data["data"], key=lambda item: item["index"])
                if len(ordered) != len(texts) or [item["index"] for item in ordered] != list(range(len(texts))):
                    raise RuntimeError("Embedding response indices do not match inputs")
                vectors = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
                if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
                    raise RuntimeError(f"Invalid embedding response shape: {vectors.shape}")
                if data.get("model", model) != model:
                    raise RuntimeError("Server returned an unexpected embedding model")
                with self.lock:
                    self.tokens += data.get("usage", {}).get("total_tokens", 0)
                return vectors
            if response.status_code in (429, 500, 502, 503, 504):
                wait = min(max(float(response.headers.get("Retry-After", 0) or 0), 2 ** min(attempt + 1, 5)), 55)
                with self.lock:
                    self.retries += 1
                    self.cooldown = max(self.cooldown, time.monotonic() + wait)
                print(json.dumps({"stage": "api_retry", "status": response.status_code, "wait_seconds": wait}), flush=True)
                continue
            # Never log headers or credential-bearing objects.
            message = response.text[:300].replace(self.key, "[REDACTED]")
            raise RuntimeError(f"SiliconFlow {model} HTTP {response.status_code}: {message}")
        raise RuntimeError("SiliconFlow retry budget exceeded; cached vectors are retained")


def embed_dataframe(df, model, name, workers=4, batch_size=32):
    fingerprint = hashlib.sha256((model + "\0" + "\n".join(df.text_sha256)).encode()).hexdigest()
    matrix_path = OUT / f"{name}_embeddings.npy"
    metadata_path = OUT / f"{name}_embedding_metadata.json"
    if matrix_path.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] == fingerprint:
            matrix = np.load(matrix_path)
            assert matrix.shape == (len(df), 1024) and np.isfinite(matrix).all()
            print(json.dumps({"stage": "embedding_matrix_reused", "model": model, "rows": len(df)}), flush=True)
            return matrix
    tokenizer = tokenizer_for(model)
    limit = 8000 if model.endswith("bge-m3") else 480
    plans = [chunks_for(text, tokenizer, limit) for text in df.text]
    texts = list(dict.fromkeys(piece for plan in plans for piece, _ in plan))
    cache = EmbeddingCache()
    loaded = cache.load(model)
    missing = [text for text in texts if cache.key(model, text) not in loaded]
    client = APIClient()
    batches = [missing[i:i + batch_size] for i in range(0, len(missing), batch_size)]
    started = time.monotonic()
    done = len(texts) - len(missing)
    def progress(status="running"):
        result = {"stage": "embeddings", "status": status, "model": model, "rows": len(df), "unique_inputs": len(texts), "completed": done, "requests": client.requests, "retries": client.retries, "tokens": client.tokens, "seconds": round(time.monotonic() - started, 1)}
        write_json(OUT / "progress.json", result)
        return result
    print(json.dumps(progress()), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit a bounded window to keep retries responsive and memory bounded.
        iterator = iter(batches)
        pending = {}
        for _ in range(workers):
            batch = next(iterator, None)
            if batch is not None:
                pending[pool.submit(client.request, model, batch)] = batch
        while pending:
            completed, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                batch = pending.pop(future)
                vectors = future.result()
                cache.save(model, batch, vectors)
                for text, vector in zip(batch, vectors):
                    loaded[cache.key(model, text)] = vector
                done += len(batch)
                progress()
                if done % 256 < batch_size or done == len(texts):
                    print(json.dumps(progress()), flush=True)
                next_batch = next(iterator, None)
                if next_batch is not None:
                    pending[pool.submit(client.request, model, next_batch)] = next_batch
    matrix = np.empty((len(df), 1024), dtype=np.float32)
    for index, plan in enumerate(plans):
        vectors = np.stack([loaded[cache.key(model, text)] for text, _ in plan])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
        weights = np.asarray([weight for _, weight in plan], dtype=np.float32)
        matrix[index] = np.average(vectors, axis=0, weights=weights)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(1e-12)
    assert np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
    np.save(matrix_path, matrix)
    metadata = {"fingerprint": fingerprint, "model": model, "endpoint": ENDPOINT, "rows": len(df), "dimensions": 1024, "unique_inputs": len(texts), "chunked_documents": sum(len(plan) > 1 for plan in plans), "maximum_chunks": max(map(len, plans)), "token_limit_per_chunk": limit, "chunk_aggregation": "Token-count weighted mean of L2-normalized chunk vectors, followed by L2 normalization; every text character retained", "api_requests_this_run": client.requests, "api_retries_this_run": client.retries, "api_tokens_this_run": client.tokens, "cached_inputs_at_start": len(texts) - len(missing), "record_ids": df.record_id.tolist()}
    write_json(metadata_path, metadata)
    write_json(OUT / f"{name}_chunk_counts.json", [{"record_id": record_id, "chunks": len(plan), "tokens": sum(weight for _, weight in plan)} for record_id, plan in zip(df.record_id, plans)])
    print(json.dumps(progress("complete")), flush=True)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT.parent)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    df = prepare(args.input_dir)
    if args.prepare_only:
        return
    embed_dataframe(df, MODELS["joint_bge_m3"], "joint_bge_m3", args.workers, args.batch_size)
    chinese = df.loc[df.language_of_text.eq("zh")].copy()
    embed_dataframe(chinese, MODELS["chinese_bge_zh"], "chinese_bge_zh", args.workers, args.batch_size)
    write_json(OUT / "progress.json", {"stage": "all_embeddings_complete", "status": "complete", "joint_rows": len(df), "chinese_rows": len(chinese)})


if __name__ == "__main__":
    main()
