"""Cohort feasibility tests keep narrative and guardrail outcomes distinct."""

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.plan_evaluation_cohorts import cohort_feasibility  # noqa: E402


class EvaluationCohortTests(unittest.TestCase):
    def test_cohorts_are_disjoint_and_exhaustive(self):
        predictions = pd.DataFrame({
            "actual": [1, 0, 1, 0],
            "fraud_probability": [0.9, 0.7, 0.3, 0.1],
        })
        result = cohort_feasibility(
            predictions, detector_threshold=0.2, release_gates=(0.5,)
        )[0]
        self.assertEqual(result["narrative_comparison_total"], 2)
        self.assertEqual(result["score_gate_challenge_total"], 1)
        self.assertEqual(result["true_negative_control"], 1)
        self.assertEqual(result["false_negative_control"], 0)

    def test_release_gate_must_exceed_detector_threshold(self):
        predictions = pd.DataFrame({
            "actual": [0], "fraud_probability": [0.1]
        })
        with self.assertRaises(ValueError):
            cohort_feasibility(
                predictions, detector_threshold=0.2, release_gates=(0.2,)
            )


if __name__ == "__main__":
    unittest.main()
