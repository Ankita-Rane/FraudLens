"""Unit checks for the dataset-neutral evidence contract."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import (  # noqa: E402
    build_alert_records,
    build_unified_evidence,
    read_jsonl,
    write_jsonl,
)


class EvidenceTests(unittest.TestCase):
    def test_alert_records_keep_features_nested(self) -> None:
        features = pd.DataFrame({"amount": [5.0, 900.0], "category": ["food", "travel"]})
        records = build_alert_records(
            dataset_id="Example",
            features=features,
            y_true=np.array([0, 1]),
            scores=np.array([0.2, 0.91]),
            threshold=0.8,
            model_name="test-model",
            record_ids=["row-0", "row-1"],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_id"], "row-1")
        self.assertEqual(records[0]["features"]["category"], "travel")

    def test_unifier_strips_truth_from_llm_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for key, label in (("ulb", "ULB"), ("sparkov", "Sparkov")):
                artifact_dir = root / "artifacts" / key
                artifact_dir.mkdir(parents=True)
                records = [{
                    "schema_version": "1.0",
                    "evidence_id": f"model-output:{key}:one",
                    "evidence_type": "fraud_model_alert",
                    "dataset_id": label,
                    "record_id": "one",
                    "event_time": None,
                    "model_name": "test-model",
                    "fraud_probability": 0.9,
                    "decision_threshold": 0.7,
                    "fraud_alert": 1,
                    "actual_class": 1,
                    "features": {"x": 1},
                }]
                write_jsonl(records, artifact_dir / "llm_alert_evidence.jsonl")
                pd.DataFrame([{
                    "dataset": label,
                    "model": "test-model",
                    "split": "test",
                    "threshold_rule": "validation_tuned",
                    "threshold": 0.7,
                    "average_precision": 0.8,
                }]).to_csv(artifact_dir / "test_metrics.csv", index=False)
                (artifact_dir / "run_manifest.json").write_text(json.dumps({
                    "dataset": label,
                    "selected_model": "test-model",
                    "selected_threshold": 0.7,
                    "primary_metric": "average_precision",
                }))

            bundle = build_unified_evidence(root, max_alerts_per_dataset=1)
            self.assertEqual({item["dataset_id"] for item in bundle["datasets"]}, {"ULB", "Sparkov"})
            for dataset in bundle["datasets"]:
                self.assertNotIn("actual_class", dataset["top_alerts"][0])

            all_records = read_jsonl(root / "artifacts" / "unified" / "unified_alerts.jsonl")
            self.assertEqual(len(all_records), 2)
            self.assertIn("actual_class", all_records[0])


if __name__ == "__main__":
    unittest.main()
