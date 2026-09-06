"""Checks for question-aware ULB/Sparkov evidence routing and answers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_answer import build_evidence_answer  # noqa: E402
from src.historical import load_historical_profiles  # noqa: E402
from src.model_evidence_retrieval import scope_evidence_for_question  # noqa: E402


def historical_bundle() -> dict:
    profiles = load_historical_profiles(PROJECT_ROOT)
    return {
        "combination_policy": "historical only",
        "interpretation_notes": {
            dataset["dataset_id"]: dataset["interpretation_limit"]
            for dataset in profiles["datasets"]
        },
        "cross_dataset_test_comparison": [],
        "historical_profiles": profiles,
        "datasets": [{
            "dataset_id": dataset["dataset_id"],
            "model_name": "Not generated",
            "selected_threshold": None,
            "primary_metric": "average_precision",
            "test_metrics": [],
            "alert_count": 0,
            "top_alerts": [],
        } for dataset in profiles["datasets"]],
    }


class ModelEvidenceRetrievalTests(unittest.TestCase):
    def test_comparison_retrieves_prevalence_from_both_datasets(self) -> None:
        scoped, routing = scope_evidence_for_question(
            historical_bundle(),
            "Compare historical fraud rates in ULB and Sparkov",
        )
        self.assertEqual(set(routing["evidence_ids"]), {
            "historical:ulb:prevalence",
            "historical:sparkov:prevalence",
        })
        answer, citations = build_evidence_answer("compare", scoped)
        self.assertIn("0.1663%", answer)
        self.assertIn("0.5753%", answer)
        self.assertEqual(set(citations), set(routing["evidence_ids"]))

    def test_explicit_sparkov_question_excludes_ulb(self) -> None:
        _, routing = scope_evidence_for_question(
            historical_bundle(),
            "Which Sparkov categories and hours had the highest fraud rates?",
        )
        self.assertTrue(routing["evidence_ids"])
        self.assertTrue(all(
            evidence_id.startswith("historical:sparkov:")
            for evidence_id in routing["evidence_ids"]
        ))


if __name__ == "__main__":
    unittest.main()
