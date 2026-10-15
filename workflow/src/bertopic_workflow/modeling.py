"""BERTopic candidate comparison and diagnostics on one declared vector space."""
from __future__ import annotations

from collections import Counter
import logging
import os
from pathlib import Path
import re

import numpy as np

from .common import WorkflowError, read_json, write_json, write_table


class Tokenizer:
    def __init__(self, stopwords=(), domain_terms=()):
        import jieba
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        jieba.setLogLevel(logging.ERROR)
        self.segmenter = jieba.Tokenizer()
        for term in domain_terms:
            self.segmenter.add_word(term)
        chinese = '的 了 和 与 及 在 对 为 是 中 由 于 以 将 等 并 其 该 这 那 本文 本研究 研究 结果 表明 通过 进行 方法 分析 根据 因此 同时 以及 其中 对于 关于 这些 那些 一个 一种 我们 结论 目的 摘要 作者 文献 年 月 日'.split()
        self.stopwords = set(ENGLISH_STOP_WORDS) | set(chinese) | set(stopwords)
        self.pattern = re.compile(r'[\u3400-\u9fff]+|[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*')

    def __call__(self, text):
        tokens = []
        for part in self.pattern.findall(text):
            words = self.segmenter.lcut(part) if re.match(r'[\u3400-\u9fff]', part) else [part.casefold()]
            tokens.extend(w for w in words if len(w) >= 2 and w not in self.stopwords)
        return tokens


class FixedReduction:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def fit(self, x, y=None):
        if len(x) != len(self.coordinates):
            raise WorkflowError('ALIGNMENT', '降维坐标与文档不匹配。')
        return self

    def transform(self, x):
        self.fit(x)
        return self.coordinates


def coherence(model, texts, top_n=10):
    binary = model.vectorizer_model.transform(texts).astype(np.float64)
    binary.data.fill(1)
    vocabulary = model.vectorizer_model.vocabulary_
    scores = {}
    for topic, pairs in model.get_topics().items():
        if topic < 0:
            continue
        ids = [vocabulary[word] for word, weight in pairs[:top_n] if word in vocabulary and weight > 0]
        if len(ids) < 2:
            scores[topic] = None
            continue
        subset = binary[:, ids]
        counts = np.asarray(subset.sum(axis=0)).ravel()
        together = (subset.T @ subset).toarray()
        values = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                joint = together[a, b] / len(texts)
                values.append(-1 if joint <= 0 else 0 if joint >= 1 else
                              float(np.log(joint / (counts[a] * counts[b] / len(texts)**2)) / -np.log(joint)))
        scores[topic] = float(np.mean(values)) if values else None
    return scores


def evaluate(model, vectors, texts, sample_size, seed):
    from sklearn.metrics import silhouette_samples
    labels = np.asarray(model.topics_, dtype=int)
    assigned = np.flatnonzero(labels >= 0)
    ids = sorted(set(labels) - {-1})
    rng = np.random.default_rng(seed)
    sample = assigned if len(assigned) <= sample_size else np.sort(rng.choice(assigned, sample_size, replace=False))
    silhouette, sampled_values = None, np.zeros(0)
    if 1 < len(set(labels[sample])) < len(sample):
        sampled_values = silhouette_samples(vectors[sample], labels[sample], metric='cosine')
        silhouette = float(sampled_values.mean())
    else:
        sample = np.zeros(0, dtype=int)
    npmi = coherence(model, texts)
    words = [w for t in ids for w, weight in model.get_topic(t)[:10] if weight > 0]
    valid_npmi = [v for v in npmi.values() if v is not None]
    metrics = {'topics': len(ids), 'records': len(labels), 'outliers': int((labels == -1).sum()),
               'outlier_rate': float((labels == -1).mean()), 'silhouette_cosine': silhouette,
               'silhouette_sample_records': len(sample), 'npmi_mean': float(np.mean(valid_npmi)) if valid_npmi else None,
               'keyword_diversity_top10': len(set(words)) / len(words) if words else None}
    # A ranking among this run's candidates; never an enterprise acceptance score.
    metrics['selection_score'] = (.30 * (1 - metrics['outlier_rate']) + .25 * ((silhouette or 0) + 1) / 2
                                  + .25 * ((metrics['npmi_mean'] or 0) + 1) / 2
                                  + .20 * (metrics['keyword_diversity_top10'] or 0)) if ids else -1
    return metrics, npmi, sample, sampled_values


def model_corpus(config, ingest_dir, embed_dir, destination, check_cancel=lambda: None, progress=lambda **kwargs: None):
    # Provided embeddings do not require activating the optional transformer runtime.
    os.environ.setdefault('USE_TORCH', '0')
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    from bertopic import BERTopic
    from bertopic.backend import BaseEmbedder
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from umap import UMAP

    destination = Path(destination)
    corpus = read_json(Path(ingest_dir) / 'corpus.json')
    metadata = read_json(Path(embed_dir) / 'embedding_metadata.json')
    vectors = np.load(Path(embed_dir) / 'embeddings.npy', allow_pickle=False)
    if metadata['record_ids'] != [r['record_id'] for r in corpus] or metadata['text_sha256'] != [r['text_sha256'] for r in corpus]:
        raise WorkflowError('ALIGNMENT', '建模前文献、文本和向量顺序不一致。')
    texts, n, options = [r['text'] for r in corpus], len(corpus), config.model
    sizes = [size for size in options.min_cluster_sizes if size <= n]
    if not sizes or options.min_samples >= n:
        raise WorkflowError('MODEL_PARAMETERS', '记录数不足以使用当前聚类参数。', details={'records': n})
    applied = {'n_neighbors': min(options.n_neighbors, n - 1), 'n_components': min(options.n_components, n - 2),
               'min_cluster_sizes': sizes, 'skipped_cluster_sizes': [s for s in options.min_cluster_sizes if s > n],
               'min_samples': options.min_samples, 'seed': options.seed, 'umap_init': 'random'}
    def reduce(seed):
        return UMAP(n_neighbors=applied['n_neighbors'], n_components=applied['n_components'],
                    init='random', min_dist=0, metric='cosine', random_state=seed,
                    n_jobs=1, low_memory=True).fit_transform(vectors)
    check_cancel()
    progress(action='reduction', records=n)
    coordinates = reduce(options.seed)
    np.save(destination / 'coordinates.npy', coordinates, allow_pickle=False)
    candidates, best = [], None
    tokenizer = Tokenizer(options.stopwords, options.domain_terms)
    for index, size in enumerate(sizes):
        check_cancel()
        vectorizer = CountVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False,
                                     ngram_range=(1, 2), max_features=options.max_features)
        clusterer = HDBSCAN(min_cluster_size=size, min_samples=options.min_samples, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True, core_dist_n_jobs=1)
        model = BERTopic(embedding_model=BaseEmbedder(), umap_model=FixedReduction(coordinates),
                        hdbscan_model=clusterer, vectorizer_model=vectorizer,
                        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                        top_n_words=options.top_n_words, calculate_probabilities=False, verbose=False)
        try:
            model.fit_transform(texts, vectors)
        except ValueError as error:
            if 'vocabulary' in str(error).casefold():
                raise WorkflowError('VOCABULARY_EMPTY', '分词后没有有效词项，请调整停用词或输入内容。') from None
            raise
        metrics, npmi, sample, values = evaluate(model, vectors, texts, options.silhouette_sample, options.seed)
        metrics['min_cluster_size'] = size
        candidates.append(metrics)
        if best is None or metrics['selection_score'] > best[1]['selection_score']:
            best = model, metrics, npmi, sample, values
        progress(action='candidate', completed=index + 1, total=len(sizes), topics=metrics['topics'])
    check_cancel()
    model, metrics, npmi, sample, sample_values = best
    labels = np.asarray(model.topics_, dtype=int)
    topic_ids = sorted(int(t) for t in set(labels) - {-1})
    if len(topic_ids) > options.max_topics:
        raise WorkflowError('MODEL_LIMIT', '主题数超过 max_topics，请调整聚类粒度或提高资源上限。')
    centroids = np.stack([vectors[labels == t].mean(axis=0) for t in topic_ids]) if topic_ids else np.empty((0, vectors.shape[1]), dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise WorkflowError('CENTROID_ZERO', '主题中心为零向量，不能计算余弦相似度。')
    centroids /= np.maximum(norms, 1e-12)
    similarity = centroids @ centroids.T
    np.save(destination / 'topic_centroids.npy', centroids, allow_pickle=False)
    np.save(destination / 'topic_similarity.npy', similarity, allow_pickle=False)
    if len(topic_ids) >= 2:
        distance = np.clip(1 - similarity, 0, 2)
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)
        np.save(destination / 'hierarchy.npy', linkage(squareform(distance, checks=True), method='average'), allow_pickle=False)
    names = {-1: '离群文献'}
    keywords, summary, representatives = [], [], []
    top_words = {t: [w for w, weight in model.get_topic(t)[:10] if weight > 0] for t in topic_ids}
    word_topic_counts = Counter(w for values in top_words.values() for w in set(values))
    # Avoid allocating a corpus-size × topic-count matrix.
    nearest_indices = np.full(n, -1, dtype=int)
    nearest_values = np.full(n, np.nan, dtype=np.float32)
    if topic_ids:
        for first in range(0, n, 4096):
            block = vectors[first:first + 4096] @ centroids.T
            best_columns = block.argmax(axis=1)
            nearest_indices[first:first + len(block)] = best_columns
            nearest_values[first:first + len(block)] = block[np.arange(len(block)), best_columns]
    assignment_rows = []
    for t in topic_ids:
        pairs = [(w, float(v)) for w, v in model.get_topic(t) if w and v > 0]
        names[t] = ' / '.join(w for w, _ in pairs[:3])
        positions = np.flatnonzero(labels == t)
        column = topic_ids.index(t)
        weights = [w for w, _ in pairs[:10]]
        summary.append({'topic': t, 'label': names[t], 'count': len(positions), 'share_all_records': len(positions) / n,
                        'npmi': npmi.get(t), 'centroid_cohesion': float((vectors[positions] @ centroids[column]).mean()),
                        'keyword_exclusivity': sum(word_topic_counts[w] == 1 for w in weights) / len(weights) if weights else None})
        keywords.extend({'topic': t, 'rank': rank, 'keyword': word, 'ctfidf_weight': weight} for rank, (word, weight) in enumerate(pairs, 1))
        seen = set()
        topic_cosines = vectors[positions] @ centroids[column]
        for rank_index in np.argsort(-topic_cosines, kind='stable'):
            i = positions[rank_index]
            row = corpus[int(i)]
            if row['text_sha256'] in seen:
                continue
            seen.add(row['text_sha256'])
            representatives.append({'topic': t, 'rank': len(seen), 'record_id': row['record_id'],
                                    'title': row['title'], 'abstract': row['abstract'], 'source': row['source'],
                                    'year': row['year'], 'doi': row['doi'], 'centroid_cosine': topic_cosines[rank_index]})
            if len(seen) >= options.representative_count:
                break
    strengths = np.asarray(model.probabilities_)
    for i, row in enumerate(corpus):
        best_index = int(nearest_indices[i]) if topic_ids else None
        assignment_rows.append({**row, 'topic': int(labels[i]), 'topic_label': names[int(labels[i])],
                                'membership_strength': float(strengths[i]),
                                'nearest_topic': topic_ids[best_index] if best_index is not None else None,
                                'nearest_cosine': float(nearest_values[i]) if best_index is not None else None,
                                'plot_x': float(coordinates[i, 0]), 'plot_y': float(coordinates[i, 1])})
    annual_counts = Counter((r['year'], r['topic']) for r in assignment_rows if r['year'] is not None)
    annual_totals = Counter(r['year'] for r in assignment_rows if r['year'] is not None)
    annual = [{'year': year, 'topic': t, 'count': count, 'share_of_year': count / annual_totals[year]}
              for (year, t), count in sorted(annual_counts.items())]
    sources = Counter(r['source'] for r in assignment_rows)
    source_counts = Counter((r['source'], r['topic']) for r in assignment_rows)
    source_rows = [{'source': source, 'topic': t, 'count': count, 'share_within_source': count / sources[source]}
                   for (source, t), count in sorted(source_counts.items())]
    stability, seed_rows = [], []
    for index, seed in enumerate(options.stability_seeds):
        check_cancel()
        alt_labels = HDBSCAN(min_cluster_size=metrics['min_cluster_size'], min_samples=options.min_samples,
                             metric='euclidean', core_dist_n_jobs=1).fit_predict(reduce(seed))
        both = (labels >= 0) & (alt_labels >= 0)
        stability.append({'seed': seed, 'topics': len(set(alt_labels) - {-1}),
                          'outlier_rate': float((alt_labels == -1).mean()),
                          'ari_all': float(adjusted_rand_score(labels, alt_labels)),
                          'nmi_all': float(normalized_mutual_info_score(labels, alt_labels)),
                          'both_assigned': int(both.sum()),
                          'ari_both_assigned': float(adjusted_rand_score(labels[both], alt_labels[both])) if both.sum() > 1 else None})
        seed_rows.extend({'record_id': r['record_id'], 'seed': seed, 'primary_topic': int(a), 'comparison_topic': int(b)}
                         for r, a, b in zip(corpus, labels, alt_labels, strict=True))
        progress(action='stability', completed=index + 1, total=len(options.stability_seeds))
    model.vectorizer_model.vocabulary_ = {w: int(i) for w, i in model.vectorizer_model.vocabulary_.items()}
    model.save(str(destination / 'portable_model'), serialization='safetensors', save_embedding_model=False, save_ctfidf=True)
    tables = {'document_topics': assignment_rows, 'topic_summary': sorted(summary, key=lambda r: (-r['count'], r['topic'])),
              'topic_keywords': keywords, 'representative_documents': representatives, 'topics_by_year': annual,
              'topics_by_source': source_rows, 'parameter_comparison': candidates, 'seed_assignments': seed_rows,
              'silhouette_samples': [{'record_id': corpus[int(i)]['record_id'], 'topic': int(labels[i]), 'silhouette': float(v)} for i, v in zip(sample, sample_values, strict=True)]}
    for name, rows in tables.items():
        write_table(destination, name, rows)
    write_json(destination / 'topic_ids.json', topic_ids)
    write_json(destination / 'topic_labels.json', names)
    write_json(destination / 'stability.json', stability)
    write_json(destination / 'applied_parameters.json', applied)
    metrics.update(embedding_model=config.embedding.model, seed=options.seed, stability_runs=len(stability),
                   selection_rule='0.30 coverage + 0.25 normalized silhouette + 0.25 normalized NPMI + 0.20 keyword diversity; candidate ranking only',
                   npmi_definition='Document cooccurrence; absent pairs=-1; ubiquitous pairs=0; fewer than two terms=null',
                   portable_inference='Nearest topic embedding similarity; UMAP/HDBSCAN are not serialized')
    write_json(destination / 'metrics.json', metrics)
    return metrics
