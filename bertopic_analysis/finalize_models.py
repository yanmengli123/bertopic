"""Persist explanatory labels and verify that saved models can be loaded."""
from __future__ import annotations
import json
import os
from pathlib import Path
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
import numpy as np
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from text_utils import tokenize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main():
    results = {}
    for name in ["joint_bge_m3", "chinese_bge_zh", "chinese_bge_m3_comparison"]:
        directory = OUT / name
        labels = {int(key): value for key, value in json.loads((directory / "topic_labels.json").read_text(encoding="utf-8")).items()}
        model = BERTopic.load(str(directory / "bertopic_model.pkl"), embedding_model=BaseEmbedder())
        records = json.loads((directory / "document_topics.json").read_text(encoding="utf-8"))
        expected = np.asarray([record["topic"] for record in records])
        assert np.array_equal(expected, model.topics_)
        assert int(model.get_topic_info()["Count"].sum()) == len(records)
        model.set_topic_labels(labels)
        model.save(str(directory / "bertopic_model.pkl"), serialization="pickle", save_embedding_model=False)
        model.save(str(directory / "portable_model"), serialization="safetensors", save_embedding_model=False, save_ctfidf=True)
        portable = BERTopic.load(str(directory / "portable_model"), embedding_model=BaseEmbedder())
        # Portable config cannot serialize a Python tokenizer function. Restore it
        # explicitly when continuing c-TF-IDF analysis with the portable model.
        portable.vectorizer_model.set_params(tokenizer=tokenize, token_pattern=None, lowercase=False)
        assert set(portable.get_topics()) == set(model.get_topics())
        assert portable.c_tf_idf_.shape == model.c_tf_idf_.shape
        assert np.allclose(portable.c_tf_idf_.data, model.c_tf_idf_.data)
        results[name] = {"full_model_loaded": True, "portable_model_loaded": True, "documents": len(records), "topics_excluding_noise": len(set(expected) - {-1}), "custom_labels_saved": True, "portable_tokenizer_restore": "from text_utils import tokenize; model.vectorizer_model.set_params(tokenizer=tokenize,token_pattern=None,lowercase=False)"}
        print(json.dumps({"stage": "saved_model_verified", "analysis": name, "records": len(records)}), flush=True)
        del model, portable, records
    (OUT / "model_validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
