"""Independent numeric and artifact reconciliation; no training or API calls."""
from collections import Counter
import csv
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from .common import WorkflowError, csv_cell, read_json, sha256, verify_entries


def require(condition, description):
    if not condition:
        raise WorkflowError('VALIDATION', description)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(v for k, v in attrs if k in {'src', 'href'} and v)


def verify_content(run_dir, config):
    run_dir = Path(run_dir)
    docs = read_json(run_dir / 'model/document_topics.json')
    corpus = read_json(run_dir / 'ingest/corpus.json')
    ingest = read_json(run_dir / 'ingest/input_manifest.json')
    metadata = read_json(run_dir / 'embed/embedding_metadata.json')
    metrics = read_json(run_dir / 'model/metrics.json')
    summary = read_json(run_dir / 'model/topic_summary.json')
    keywords = read_json(run_dir / 'model/topic_keywords.json')
    require(all(np.isfinite(r['ctfidf_weight']) and r['ctfidf_weight'] >= 0 for r in keywords), '关键词权重无效。')
    vectors = np.load(run_dir / 'embed/embeddings.npy', allow_pickle=False, mmap_mode='r')
    ids = [r['record_id'] for r in corpus]
    require(len(ids) == len(set(ids)) == len(docs) == len(vectors), '文献、ID 与向量记录数不一致。')
    require([r['record_id'] for r in docs] == metadata['record_ids'] == ids, '文献与向量 ID 顺序不一致。')
    require([r['text_sha256'] for r in docs] == metadata['text_sha256'] == [r['text_sha256'] for r in corpus], '文本指纹不一致。')
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5), '嵌入包含非法值或未归一化。')
    require(ingest['read_records'] == len(corpus) + ingest['quarantined_records'] + ingest['deduplicated_records'], '导入记录流失或重复计数。')
    require(ingest['source_counts'] == dict(Counter(r['source'] for r in docs)), '数据库计数不一致。')
    labels = np.array([r['topic'] for r in docs])
    observed = Counter(int(t) for t in labels if t >= 0)
    topic_ids = read_json(run_dir / 'model/topic_ids.json')
    require(topic_ids == sorted(observed), '主题 ID 与归属记录不一致。')
    require({r['topic']: r['count'] for r in summary} == dict(observed), '主题规模不一致。')
    require(metrics['outliers'] == int((labels == -1).sum()) and metrics['topics'] == len(observed), '主题或离群指标不一致。')
    centroids = np.load(run_dir / 'model/topic_centroids.npy', allow_pickle=False)
    similarity = np.load(run_dir / 'model/topic_similarity.npy', allow_pickle=False)
    if topic_ids:
        recomputed = np.stack([vectors[labels == t].mean(axis=0) for t in topic_ids])
        recomputed /= np.linalg.norm(recomputed, axis=1, keepdims=True)
        require(np.allclose(centroids, recomputed, atol=2e-5), '主题中心与原始向量不一致。')
        require(np.allclose(similarity, centroids @ centroids.T, atol=2e-5), '主题相似度计算错误。')
        require(np.allclose(np.diag(similarity), 1, atol=2e-5), '主题相似度对角线无效。')
    yearly = read_json(run_dir / 'model/topics_by_year.json')
    require({(r['year'], r['topic']): r['count'] for r in yearly} ==
            dict(Counter((r['year'], r['topic']) for r in docs if r['year'] is not None)), '逐年记录数不一致。')
    source_rows = read_json(run_dir / 'model/topics_by_source.json')
    require({(r['source'], r['topic']): r['count'] for r in source_rows} ==
            dict(Counter((r['source'], r['topic']) for r in docs)), '分库主题数不一致。')
    source_totals = Counter(r['source'] for r in docs)
    require(all(np.isclose(r['share_within_source'], r['count'] / source_totals[r['source']]) for r in source_rows), '分库比例分母错误。')
    samples = read_json(run_dir / 'model/silhouette_samples.json')
    if samples:
        from sklearn.metrics import silhouette_samples
        positions = {r['record_id']: i for i, r in enumerate(docs)}
        indices = [positions[r['record_id']] for r in samples]
        values = silhouette_samples(vectors[indices], labels[indices], metric='cosine')
        require(np.allclose(values, [r['silhouette'] for r in samples], atol=2e-5), '轮廓系数未通过独立重算。')
        require(np.isclose(values.mean(), metrics['silhouette_cosine'], atol=2e-5), '轮廓均值不一致。')
    # Recompute NPMI from per-document token sets, independently of the model's sparse matrix.
    from .modeling import Tokenizer
    tokenizer = Tokenizer(config.model.stopwords, config.model.domain_terms)
    words = {t: [r['keyword'] for r in keywords if r['topic'] == t and r['rank'] <= 10] for t in topic_ids}
    relevant = {w for terms in words.values() for w in terms}
    presence = {w: 0 for w in relevant}
    for i, doc in enumerate(docs):
        tokens = tokenizer(doc['text'])
        terms = set(tokens) | {a + ' ' + b for a, b in zip(tokens, tokens[1:], strict=False)}
        for word in terms & relevant:
            presence[word] |= 1 << i
    for row in summary:
        terms = words[row['topic']]
        scores = []
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                joint = (presence[a] & presence[b]).bit_count() / len(docs)
                if joint == 0:
                    scores.append(-1.)
                elif joint == 1:
                    scores.append(0.)
                else:
                    pa, pb = presence[a].bit_count() / len(docs), presence[b].bit_count() / len(docs)
                    scores.append(np.log(joint / (pa * pb)) / -np.log(joint))
        require((not scores and row['npmi'] is None) or
                (bool(scores) and np.isclose(np.mean(scores), row['npmi'], atol=1e-6)), 'NPMI 未通过文档共现独立重算。')
    catalog = read_json(run_dir / 'report/figures_catalog.json')
    require([r['number'] for r in catalog['charts']] == list(range(1, 16)), '15 类图的状态清单不完整。')
    for item in catalog['charts']:
        require(item['status'] in {'generated', 'not_applicable'}, '图状态无效。')
        require(item['files'] if item['status'] == 'generated' else item['reason'], '缺少图产物或不适用原因。')
        for relative in item['files']:
            require((run_dir / 'report' / relative).is_file(), '图库引用文件缺失。')
    report = run_dir / 'report'
    def evidence(n):
        path = report / f'evidence/fig{n:02d}.json'
        return read_json(path) if path.exists() else None
    if data := evidence(1):
        require(data['topic_ids'] == topic_ids and data['counts'] == [observed[t] for t in topic_ids], '总览气泡规模不一致。')
    if data := evidence(5):
        shown = data['displayed_record_ids']
        require(len(shown) == len(set(shown)) and set(shown) <= set(ids), '文档分布图重复或错配文献。')
        require(data['sampling'] != 'none' or set(shown) == set(ids), '完整分布图遗漏文献。')
    if data := evidence(8):
        eligible = [r for r in docs if r['year'] is not None and r['year'] <= config.report.year_end
                    and (config.report.year_start is None or r['year'] >= config.report.year_start)]
        totals = Counter(r['year'] for r in eligible)
        require(data['annual_denominators'] == [totals[y] for y in data['years']], '趋势图分母不一致。')
        require(np.allclose(np.sum(data['shares'], axis=0) + data['remainder'], 1), '年度构成未合计为 100%。')
    if data := evidence(10):
        require(np.allclose(data['cosine_similarity'], similarity, atol=1e-6), '相似度图与模型矩阵不一致。')
    if data := evidence(11):
        expected = [(a, b) for i, a in enumerate(topic_ids) for b in topic_ids[i + 1:]
                    if similarity[i, topic_ids.index(b)] >= data['threshold']]
        require([(r['source'], r['target']) for r in data['edges']] == expected, '网络边与精确阈值不一致。')
    if data := evidence(12):
        require(data['cumulative_denominator'] == sum(observed.values()), '帕累托分母不是已归类记录数。')
    if data := evidence(15):
        require(sum(r['count'] for r in data['flows']) == data['included_records'], '桑基图流量不守恒。')
        require(data['included_records'] + data['excluded_records'] == len(docs), '桑基图时间范围计数错误。')
    local_links = 0
    for file in report.rglob('*.html'):
        parser = Links()
        parser.feed(file.read_text(encoding='utf-8'))
        for target in parser.targets:
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = (file.parent / unquote(parsed.path)).resolve()
            require(path.is_relative_to(run_dir.resolve()) and path.is_file(), '离线报告链接失效或越界。')
            local_links += 1
    csv_files = 0
    for phase in ['ingest', 'model']:
        for file in (run_dir / phase).glob('*.csv'):
            expected = read_json(file.with_suffix('.json'))
            with file.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                actual = list(reader)
                require(len(reader.fieldnames or []) == len(set(reader.fieldnames or [])), 'CSV 存在重复列名。')
            require(len(actual) == len(expected), 'CSV 导出记录数不一致。')
            for row, original in zip(actual, expected, strict=True):
                for field in reader.fieldnames or []:
                    value = csv_cell(original.get(field))
                    require(row[field] == ('' if value is None else str(value)), 'CSV 值与安全转义后的 JSON 不一致。')
            csv_files += 1
    return {'status': 'passed', 'records': len(docs), 'topics': len(topic_ids),
            'outliers': metrics['outliers'], 'charts': 15, 'local_links': local_links, 'csv_files': csv_files,
            'independent_checks': ['record_alignment', 'import_reconciliation', 'centroid_cosine',
                                   'source_and_year_totals', 'silhouette_recalculation', 'document_NPMI',
                                   'chart_evidence', 'offline_links']}


def verify_release(run_dir, *, semantic=True):
    from .config import WorkflowConfig
    run_dir = Path(run_dir).resolve()
    result = read_json(run_dir / 'result.json')
    require(result.get('status') == 'completed', '任务未完成，不能当作最终结果。')
    require(sha256(run_dir / 'release_manifest.json') == result['manifest_sha256'], '最终发布清单被修改。')
    manifest = read_json(run_dir / 'release_manifest.json')
    verify_entries(run_dir, manifest['artifacts'])
    for phase in ['ingest', 'embed', 'model', 'report']:
        directory = run_dir / phase
        require(directory.is_dir() and directory.resolve() == directory and not directory.is_symlink(), '阶段路径不安全。')
    config = WorkflowConfig.model_validate(read_json(run_dir / 'config.json'))
    content = verify_content(run_dir, config) if semantic else {'status': 'passed'}
    return {**content, 'files_verified': len(manifest['artifacts']), 'manifest_sha256': result['manifest_sha256']}
