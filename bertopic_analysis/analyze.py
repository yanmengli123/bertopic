"""Fit and evaluate the joint BERTopic model using precomputed API embeddings."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time
import warnings

os.environ.setdefault("NUMBA_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# API embeddings are supplied explicitly. Disable unused local transformer
# backends so optional imports cannot activate the machine's GPU stack.
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from umap import UMAP

from prepare_and_embed import ROOT, OUT, write_json
from text_utils import tokenize


class CachedReducer:
    """Reuse exactly the same UMAP coordinates while comparing HDBSCAN settings."""
    def __init__(self, coordinates):
        self.coordinates = coordinates
    def fit(self, x, y=None):
        if len(x) != len(self.coordinates):
            raise ValueError("Cached reduction row mismatch")
        return self
    def transform(self, x):
        if len(x) != len(self.coordinates):
            raise ValueError("Cached reduction row mismatch")
        return self.coordinates


def save_table(frame, directory, name):
    # Analytical intermediates are exported to final CSV with artifact-tool by export_tables.mjs.
    frame.to_json(directory / f"{name}.json", orient="records", force_ascii=False, double_precision=10)
    write_json(directory / f"{name}.schema.json", {"columns": list(frame.columns), "rows": len(frame)})


def npmi_scores(model, documents, top_n=10):
    binary = model.vectorizer_model.transform(documents).astype(np.float64)
    binary.data.fill(1)
    vocabulary = model.vectorizer_model.vocabulary_
    scores = {}
    for topic, words in model.get_topics().items():
        if topic == -1:
            continue
        ids = [vocabulary[word] for word, weight in words[:top_n] if word in vocabulary and weight > 0]
        if len(ids) < 2:
            scores[int(topic)] = -1.0
            continue
        submatrix = binary[:, ids]
        counts = np.asarray(submatrix.sum(axis=0)).ravel()
        cooccurrences = (submatrix.T @ submatrix).toarray()
        pairs = []
        n = len(documents)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                both = cooccurrences[a, b]
                if both <= 0:
                    pairs.append(-1.0)
                elif both >= n:
                    pairs.append(0.0)
                else:
                    p_ab = both / n
                    pairs.append(float(np.log(p_ab / ((counts[a] / n) * (counts[b] / n))) / -np.log(p_ab)))
        scores[int(topic)] = float(np.mean(pairs))
    return scores


def evaluate(model, embeddings, documents):
    topics = np.asarray(model.topics_)
    assigned = topics >= 0
    topic_ids = np.unique(topics[assigned])
    silhouette = None
    if len(topic_ids) >= 2 and assigned.sum() > len(topic_ids):
        positions = np.flatnonzero(assigned)
        rng = np.random.default_rng(42)
        if len(positions) > 1500:
            positions = rng.choice(positions, 1500, replace=False)
        sample_labels = topics[positions]
        if 1 < len(np.unique(sample_labels)) < len(positions):
            silhouette = float(silhouette_score(embeddings[positions], sample_labels, metric="cosine"))
    words = [word for topic, pairs in model.get_topics().items() if topic >= 0 for word, weight in pairs[:10] if weight > 0]
    coherence = npmi_scores(model, documents)
    result = {
        "topics": int(len(topic_ids)), "documents": len(documents),
        "outliers": int((~assigned).sum()), "outlier_rate": float((~assigned).mean()),
        "coverage": float(assigned.mean()), "semantic_silhouette_sample1500": silhouette,
        "topic_diversity_top10": len(set(words)) / max(1, len(words)),
        "mean_npmi_top10": float(np.mean(list(coherence.values()))) if coherence else None,
        "mean_membership_strength_assigned": float(np.asarray(model.probabilities_)[assigned].mean()) if assigned.any() else None,
    }
    # Explicit heuristic, not an established statistical threshold or significance test.
    result["selection_score"] = float(0.30 * result["coverage"] + 0.25 * ((silhouette or 0) + 1) / 2 + 0.25 * ((result["mean_npmi_top10"] or 0) + 1) / 2 + 0.20 * result["topic_diversity_top10"])
    return result, coherence


def reduced_coordinates(embeddings, directory, neighbors=30):
    fingerprint = hashlib.sha256(embeddings.tobytes()).hexdigest()
    metadata_path = directory / "reduction_metadata.json"
    reducer_path = directory / "umap.pkl"
    coordinates_path = directory / "umap_5d.npy"
    if metadata_path.exists() and reducer_path.exists() and coordinates_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["fingerprint"] == fingerprint and metadata["neighbors"] == neighbors:
            with reducer_path.open("rb") as stream:
                reducer = pickle.load(stream)
            return reducer, np.load(coordinates_path)
    print(json.dumps({"stage": "umap_fit", "rows": len(embeddings), "neighbors": neighbors}), flush=True)
    reducer = UMAP(n_neighbors=neighbors, n_components=5, min_dist=0.0, metric="cosine", random_state=42, n_jobs=1, low_memory=True, verbose=True)
    coordinates = reducer.fit_transform(embeddings)
    assert coordinates.shape == (len(embeddings), 5) and np.isfinite(coordinates).all()
    np.save(coordinates_path, coordinates)
    with reducer_path.open("wb") as stream:
        pickle.dump(reducer, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(metadata_path, {"fingerprint": fingerprint, "neighbors": neighbors, "components": 5, "random_state": 42})
    return reducer, coordinates


def build_outputs(model, df, embeddings, coordinates, coherence, directory):
    topic_ids = sorted(topic for topic in model.get_topics() if topic >= 0)
    topics = np.asarray(model.topics_)
    centroids = np.stack([embeddings[topics == topic].mean(axis=0) for topic in topic_ids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-12)
    np.save(directory / "topic_centroids.npy", centroids)
    similarities = embeddings @ centroids.T
    best = similarities.argmax(axis=1)
    assignments = df.copy()
    assignments["topic"] = topics
    assignments["cluster_membership_strength"] = np.asarray(model.probabilities_)
    assignments["nearest_semantic_topic"] = np.asarray(topic_ids)[best]
    assignments["nearest_topic_cosine"] = similarities[np.arange(len(df)), best]
    assignments["umap_1"] = coordinates[:, 0]
    assignments["umap_2"] = coordinates[:, 1]
    info_rows = []
    keyword_rows = []
    representative_rows = []
    labels = {-1: "离群文献 / Outliers"}
    for column, topic in enumerate(topic_ids):
        positions = np.flatnonzero(topics == topic)
        subset = assignments.iloc[positions]
        pairs = model.get_topic(topic)
        words = [word for word, weight in pairs if weight > 0]
        label = " / ".join(words[:3])
        labels[topic] = label
        info_rows.append({"topic": topic, "label": label, "count": len(positions), "share_all_documents": len(positions) / len(df), "keywords": "; ".join(words), "npmi_top10": coherence.get(topic), "mean_membership_strength": float(subset.cluster_membership_strength.mean()), "mean_centroid_cosine": float(similarities[positions, column].mean()), "chinese_documents": int(subset.language_of_text.eq("zh").sum()), "year_min": subset.year.min(), "year_max": subset.year.max(), "title_only_documents": int(subset.has_abstract.eq(0).sum())})
        for rank, (word, weight) in enumerate(pairs, 1):
            keyword_rows.append({"topic": topic, "rank": rank, "keyword": word, "ctfidf_weight": float(weight)})
        # Deduplicate examples only; every repeated record still contributes to fitting and counts.
        ranked = positions[np.argsort(-similarities[positions, column])]
        seen = set()
        for position in ranked:
            record = assignments.iloc[position]
            if record.text_sha256 in seen:
                continue
            seen.add(record.text_sha256)
            representative_rows.append({"topic": topic, "rank": len(seen), "record_id": record.record_id, "source": record.source, "year": record.year, "title": record.title, "abstract": record.abstract, "doi": record.doi, "centroid_cosine": float(similarities[position, column]), "cluster_membership_strength": float(record.cluster_membership_strength)})
            if len(seen) == 5:
                break
    assignments["topic_label"] = assignments.topic.map(labels)
    info = pd.DataFrame(info_rows).sort_values("count", ascending=False)
    keyword_frame = pd.DataFrame(keyword_rows)
    representatives = pd.DataFrame(representative_rows)
    annual = assignments.dropna(subset=["year"]).groupby(["year", "topic"]).size().rename("count").reset_index()
    annual["year"] = annual.year.astype(int)
    annual_totals = assignments.dropna(subset=["year"]).groupby("year").size()
    annual["share_of_year_records"] = annual["count"] / annual.year.map(annual_totals)
    annual["topic_label"] = annual.topic.map(labels)
    annual["year_review_required"] = (annual.year > 2026).astype(int)
    source_topics = assignments.groupby(["source", "topic"]).size().rename("count").reset_index()
    source_topics["share_within_source"] = source_topics["count"] / source_topics.source.map(assignments.source.value_counts())
    source_topics["share_within_topic"] = source_topics["count"] / source_topics.topic.map(assignments.topic.value_counts())
    source_topics["topic_label"] = source_topics.topic.map(labels)
    language_topics = assignments.groupby(["language_of_text", "topic"]).size().rename("count").reset_index()
    language_topics["share_within_language"] = language_topics["count"] / language_topics.language_of_text.map(assignments.language_of_text.value_counts())
    language_topics["topic_label"] = language_topics.topic.map(labels)
    cosine = centroids @ centroids.T
    np.save(directory / "topic_similarity.npy", cosine)
    similarity_rows = [{"topic_a": topic_ids[a], "topic_b": topic_ids[b], "cosine_similarity": float(cosine[a, b]), "label_a": labels[topic_ids[a]], "label_b": labels[topic_ids[b]]} for a in range(len(topic_ids)) for b in range(a + 1, len(topic_ids))]
    similarity_frame = pd.DataFrame(similarity_rows).sort_values("cosine_similarity", ascending=False)
    if len(topic_ids) > 1:
        distance = np.clip(1 - cosine, 0, 2)
        np.fill_diagonal(distance, 0)
        hierarchy = linkage(squareform(distance, checks=False), method="average")
        np.save(directory / "hierarchy_linkage.npy", hierarchy)
    for name, frame in {"document_topics": assignments, "topic_summary": info, "topic_keywords": keyword_frame, "representative_documents": representatives, "topics_by_year": annual, "topics_by_source": source_topics, "topics_by_language": language_topics, "topic_similarity": similarity_frame}.items():
        save_table(frame, directory, name)
    write_json(directory / "topic_ids.json", topic_ids)
    write_json(directory / "topic_labels.json", {str(k): value for k, value in labels.items()})
    assert len(assignments) == len(df) and int(assignments.topic.value_counts().sum()) == len(df)
    assert int(source_topics["count"].sum()) == len(df)
    assert len(assignments.loc[assignments.source.eq("WoS")]) == len(df.loc[df.source.eq("WoS")])
    return assignments, info


def analyze_corpus(name, df, embeddings, sizes, neighbors, stability=False, dynamic=False, selection_size=None):
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    documents = df.text.tolist()
    assert embeddings.shape == (len(df), 1024) and np.isfinite(embeddings).all()
    reducer, coordinates = reduced_coordinates(embeddings, directory, neighbors)
    candidates = []
    best_model = None
    best_metrics = None
    best_coherence = None
    for size in sizes:
        print(json.dumps({"stage": "bertopic_candidate", "analysis": name, "min_cluster_size": size}), flush=True)
        vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, ngram_range=(1, 2), min_df=1, max_features=50000 if len(df) > 1000 else 12000)
        clusterer = HDBSCAN(min_cluster_size=size, min_samples=10 if len(df) > 1000 else 3, metric="euclidean", cluster_selection_method="eom", prediction_data=True, gen_min_span_tree=True, core_dist_n_jobs=8)
        model = BERTopic(embedding_model=BaseEmbedder(), umap_model=CachedReducer(coordinates), hdbscan_model=clusterer, vectorizer_model=vectorizer, ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True), top_n_words=20, calculate_probabilities=False, verbose=True)
        model.fit_transform(documents, embeddings)
        metrics, coherence = evaluate(model, embeddings, documents)
        metrics.update(min_cluster_size=size, min_samples=clusterer.min_samples, n_neighbors=neighbors, analysis=name)
        candidates.append(metrics)
        write_json(directory / "candidate_metrics.json", candidates)
        print(json.dumps({"stage": "candidate_metrics", **metrics}), flush=True)
        eligible = metrics["topics"] >= 2 and (selection_size is None or size == selection_size)
        if eligible and (best_metrics is None or metrics["selection_score"] > best_metrics["selection_score"]):
            best_model, best_metrics, best_coherence = model, metrics, coherence
    if best_model is None:
        raise RuntimeError(f"No valid topic model for {name}")
    # sklearn 1.8 can use numpy integer vocabulary indices; BERTopic's portable
    # JSON writer requires Python integers. This does not change token mapping.
    best_model.vectorizer_model.vocabulary_ = {term: int(index) for term, index in best_model.vectorizer_model.vocabulary_.items()}
    best_model.umap_model = reducer
    model_path = directory / "bertopic_model.pkl"
    best_model.save(str(model_path), serialization="pickle", save_embedding_model=False)
    # A portable representation model supplements the complete local pickle model.
    best_model.save(str(directory / "portable_model"), serialization="safetensors", save_embedding_model=False, save_ctfidf=True)
    assignments, summary = build_outputs(best_model, df, embeddings, coordinates, best_coherence, directory)
    save_table(pd.DataFrame(candidates), directory, "parameter_comparison")
    if stability:
        print(json.dumps({"stage": "stability", "analysis": name}), flush=True)
        alternative_reducer = UMAP(n_neighbors=neighbors, n_components=5, min_dist=0.0, metric="cosine", random_state=7, n_jobs=1, low_memory=True)
        alt_coords = alternative_reducer.fit_transform(embeddings)
        alternative_clusterer = HDBSCAN(min_cluster_size=best_metrics["min_cluster_size"], min_samples=best_metrics["min_samples"], metric="euclidean", prediction_data=True, core_dist_n_jobs=8)
        alt_labels = alternative_clusterer.fit_predict(alt_coords)
        original = assignments.topic.to_numpy()
        both_assigned = (original >= 0) & (alt_labels >= 0)
        stability_result = {"seed_primary": 42, "seed_comparison": 7, "ari_all_including_noise": float(adjusted_rand_score(original, alt_labels)), "nmi_all_including_noise": float(normalized_mutual_info_score(original, alt_labels)), "ari_both_assigned": float(adjusted_rand_score(original[both_assigned], alt_labels[both_assigned])) if both_assigned.any() else None, "both_assigned_documents": int(both_assigned.sum()), "alternative_topics": int(len(set(alt_labels) - {-1})), "alternative_outlier_rate": float((alt_labels == -1).mean())}
        write_json(directory / "stability.json", stability_result)
        save_table(pd.DataFrame({"record_id": df.record_id, "topic_seed42": original, "topic_seed7": alt_labels}), directory, "seed_stability_assignments")
    if dynamic:
        eligible = df.year.notna() & df.year.le(2026)
        dynamic_df = df.loc[eligible]
        timestamps = ((dynamic_df.year.astype(int) // 5) * 5).tolist()
        print(json.dumps({"stage": "topics_over_time", "analysis": name, "bins": len(set(timestamps))}), flush=True)
        time_topics = best_model.topics_over_time(dynamic_df.text.tolist(), timestamps, topics=assignments.loc[eligible, "topic"].tolist(), global_tuning=True, evolution_tuning=True)
        time_topics["bin_definition"] = "Five-year interval starting at Timestamp; 2025 bin contains incomplete 2026 and excludes >2026"
        save_table(time_topics, directory, "topic_keywords_over_time")
    best_metrics["selection_rule"] = "Fixed min_cluster_size=15 for matched Chinese embedding-model comparison; candidate scores are diagnostic only" if selection_size else "0.30 coverage + 0.25 normalized semantic silhouette + 0.25 normalized NPMI + 0.20 topic diversity; heuristic used only among three candidates"
    best_metrics["embedding_model"] = "BAAI/bge-large-zh-v1.5" if name == "chinese_bge_zh" else "BAAI/bge-m3"
    best_metrics["seed"] = 42
    best_metrics["deduplication"] = "NONE"
    write_json(directory / "selected_model_metrics.json", best_metrics)
    print(json.dumps({"stage": "analysis_complete", "analysis": name, **best_metrics}), flush=True)
    return assignments, best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-only", action="store_true")
    parser.add_argument("--chinese-only", action="store_true", help="Refresh Chinese comparisons while retaining an already completed joint fit")
    args = parser.parse_args()
    df = pd.read_json(OUT / "corpus.json", dtype={"doi": str, "pmid": str, "wos_id": str})
    df["year"] = pd.to_numeric(df.year, errors="coerce")
    def validated_matrix(frame, model, name):
        metadata = json.loads((OUT / f"{name}_embedding_metadata.json").read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256((model + "\0" + "\n".join(frame.text_sha256)).encode()).hexdigest()
        if metadata["fingerprint"] != fingerprint or metadata["record_ids"] != frame.record_id.tolist():
            raise RuntimeError(f"{name}: corpus text/order changed; regenerate the matching embeddings first")
        matrix = np.load(OUT / f"{name}_embeddings.npy")
        if matrix.shape != (len(frame), 1024) or not np.isfinite(matrix).all():
            raise RuntimeError(f"{name}: invalid embedding matrix")
        return matrix
    embeddings = validated_matrix(df, "BAAI/bge-m3", "joint_bge_m3")
    if args.chinese_only:
        metrics = json.loads((OUT / "joint_bge_m3/selected_model_metrics.json").read_text(encoding="utf-8"))
        existing = pd.read_json(OUT / "joint_bge_m3/document_topics.json")
        if existing.record_id.tolist() != df.record_id.tolist() or metrics["documents"] != len(df):
            raise RuntimeError("Existing joint model does not match the corpus")
    else:
        joint, metrics = analyze_corpus("joint_bge_m3", df, embeddings, [30, 60, 100], 30, stability=True, dynamic=True)
    summaries = {"joint": metrics}
    if not args.joint_only:
        mask = df.language_of_text.eq("zh").to_numpy()
        chinese = df.loc[mask]
        zh_embeddings = validated_matrix(chinese, "BAAI/bge-large-zh-v1.5", "chinese_bge_zh")
        zh, zh_metrics = analyze_corpus("chinese_bge_zh", chinese, zh_embeddings, [10, 15, 25], 15, selection_size=15)
        m3_zh, m3_zh_metrics = analyze_corpus("chinese_bge_m3_comparison", chinese, embeddings[mask], [10, 15, 25], 15, selection_size=15)
        comparison = {"chinese_documents": len(chinese), "bge_zh": zh_metrics, "bge_m3": m3_zh_metrics, "ari_all": float(adjusted_rand_score(zh.topic, m3_zh.topic)), "nmi_all": float(normalized_mutual_info_score(zh.topic, m3_zh.topic)), "interpretation": "Separate fits in separate model vector spaces; topic numbers do not correspond across models. ARI/NMI are label-permutation invariant agreement diagnostics."}
        write_json(OUT / "chinese_model_comparison.json", comparison)
        comparison_assignments = chinese[["record_id", "title", "source"]].reset_index(drop=True)
        comparison_assignments["topic_bge_zh"] = zh.topic.to_numpy()
        comparison_assignments["topic_bge_m3"] = m3_zh.topic.to_numpy()
        save_table(comparison_assignments, OUT, "chinese_model_assignments")
        summaries.update(chinese_bge_zh=zh_metrics, chinese_bge_m3=m3_zh_metrics)
    versions = {name: importlib.metadata.version(name) for name in ["bertopic", "umap-learn", "hdbscan", "numpy", "pandas", "scikit-learn", "scipy", "jieba", "plotly", "requests", "tokenizers", "numba"]}
    write_json(OUT / "runtime_versions.json", versions)
    write_json(OUT / "analysis_summary.json", summaries)
    write_json(OUT / "progress.json", {"stage": "modeling_complete", "status": "complete", "metrics": summaries})


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*force_all_finite.*")
    main()
