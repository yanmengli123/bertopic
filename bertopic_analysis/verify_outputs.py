"""Independent reconciliation of saved analytical deliverables against inputs."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main():
    manifest = json.loads((OUT / "data_manifest.json").read_text(encoding="utf-8"))
    checks = {}
    checks["original_file_hashes"] = {name: hashlib.sha256((Path(manifest["input_directory"]) / name).read_bytes()).hexdigest() == expected for name, expected in manifest["source_sha256"].items()}
    assert all(checks["original_file_hashes"].values()), "An input was changed"
    corpus = json.loads((OUT / "corpus.json").read_text(encoding="utf-8"))
    expected_counts = {item["source"]: item["records"] for item in manifest["sources"]}
    csv_files = []
    for file in OUT.rglob("*.csv"):
        with file.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert all(None not in row for row in rows), f"Misquoted CSV row: {file.name}"
            assert len(reader.fieldnames) == len(set(reader.fieldnames)), f"Duplicate header: {file.name}"
        json_file = file.with_suffix(".json")
        expected = corpus if file.name == "merged_four_databases.csv" else json.loads(json_file.read_text(encoding="utf-8"))
        assert len(rows) == len(expected), f"Lost rows: {file.name}"
        for actual, original in zip(rows, expected):
            for field, value in original.items():
                if value is None:
                    assert actual[field] == ""
                elif isinstance(value, bool):
                    assert actual[field] == str(value).lower()
                elif isinstance(value, (int, float)):
                    assert np.isclose(float(actual[field]), value, rtol=1e-12, atol=1e-12)
                else:
                    assert actual[field] == value, f"Changed text/ID: {file.name}/{field}"
        csv_files.append({"file": str(file.relative_to(OUT)), "rows": len(rows), "columns": len(reader.fieldnames)})
    checks["csv_roundtrip"] = csv_files
    data = pd.read_json(OUT / "joint_bge_m3/document_topics.json", dtype={"doi": str, "pmid": str, "wos_id": str})
    assert len(data) == 22549 == manifest["rows"]
    assert data.source.value_counts().to_dict() == expected_counts
    assert data.record_id.is_unique
    assert data.record_id.tolist() == [row["record_id"] for row in corpus]
    assert data.topic.notna().all() and data.cluster_membership_strength.between(0, 1).all()
    summary = pd.read_json(OUT / "joint_bge_m3/topic_summary.json")
    observed_counts = data.loc[data.topic.ge(0)].topic.value_counts().sort_index().to_dict()
    assert summary.set_index("topic")["count"].sort_index().to_dict() == observed_counts
    assert int(summary["count"].sum()) + int(data.topic.eq(-1).sum()) == len(data)
    checks["joint_records"] = len(data)
    checks["source_counts"] = expected_counts
    checks["topics"] = len(summary)
    checks["outlier_records"] = int(data.topic.eq(-1).sum())
    wos = data.loc[data.source.eq("WoS")]
    assert len(wos) == 20092 and wos.wos_id.nunique() == 19092
    checks["wos_retained"] = {"records": len(wos), "unique_ids": int(wos.wos_id.nunique())}
    for name, rows in [("joint_bge_m3", 22549), ("chinese_bge_zh", 715)]:
        vectors = np.load(OUT / f"{name}_embeddings.npy", mmap_mode="r")
        assert vectors.shape == (rows, 1024) and np.isfinite(vectors).all()
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    checks["embedding_shapes_and_norms"] = "passed"
    for name, rows in [("joint_bge_m3", 22549), ("chinese_bge_zh", 715), ("chinese_bge_m3_comparison", 715)]:
        directory = OUT / name
        assignments = pd.read_json(directory / "document_topics.json")
        assert len(assignments) == rows
        annual = pd.read_json(directory / "topics_by_year.json")
        by_source = pd.read_json(directory / "topics_by_source.json")
        assert int(annual["count"].sum()) == int(assignments.year.notna().sum())
        assert int(by_source["count"].sum()) == rows
        assert np.allclose(by_source.groupby("source").share_within_source.sum(), 1)
        assert (directory / "bertopic_model.pkl").is_file()
        for file in ["config.json", "topics.json", "topic_embeddings.safetensors", "ctfidf.safetensors", "ctfidf_config.json"]:
            assert (directory / "portable_model" / file).is_file(), f"Missing portable model file: {file}"
    comparison = pd.read_json(OUT / "chinese_model_assignments.json")
    assert len(comparison) == 715 and comparison.record_id.is_unique
    assert set(comparison.record_id) == set(data.loc[data.language_of_text.eq("zh"), "record_id"])
    representatives = pd.read_json(OUT / "joint_bge_m3/representative_documents.json")
    lookup = data.set_index("record_id")
    assert (lookup.loc[representatives.record_id, "topic"].to_numpy() == representatives.topic.to_numpy()).all()
    assert data.loc[data.year.gt(2026)].shape[0] == 1
    assert (OUT / "analysis_report.html").is_file()
    checks["status"] = "passed"
    (OUT / "validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "records": len(data), "topics": len(summary), "csv_files": len(csv_files), "original_inputs_unchanged": True}))


if __name__ == "__main__":
    main()
