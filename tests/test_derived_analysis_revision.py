"""Regression tests for the versioned post-hoc derived-analysis correction."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.derived_analysis_revision import (  # noqa: E402
    evaluate_run_corrected,
    paired_configuration_analysis_corrected,
)
from tests.test_abc_evaluation import run  # noqa: E402


class DerivedAnalysisRevisionTests(unittest.TestCase):
    def test_candidate_json_validity_is_independent_of_blocked_release(self):
        blocked = run("C")
        blocked["status"] = "blocked"
        blocked["parsed_answer"] = None
        blocked["candidate_response_text"] = '{"answer": "valid candidate"}'
        metrics = evaluate_run_corrected(blocked)
        self.assertEqual(metrics["response_json_valid"], 1.0)
        self.assertEqual(metrics["released_parsed_answer_available"], 0.0)

    def test_invalid_raw_candidate_is_not_rescued_by_release_field(self):
        inconsistent = run("B")
        inconsistent["candidate_response_text"] = "not-json"
        metrics = evaluate_run_corrected(inconsistent)
        self.assertEqual(metrics["response_json_valid"], 0.0)
        self.assertEqual(metrics["released_parsed_answer_available"], 1.0)

    def test_suppressed_run_without_candidate_is_not_json_valid(self):
        suppressed = run("C")
        suppressed["candidate_response_text"] = None
        suppressed["parsed_answer"] = None
        self.assertEqual(evaluate_run_corrected(suppressed)["response_json_valid"], 0.0)

    def test_holm_support_uses_primary_adjusted_p_value_only(self):
        runs = []
        for sample_index in range(1, 31):
            for configuration in ("A", "B", "C"):
                item = run(configuration)
                item["request_id"] = f"{sample_index}-{configuration}"
                item["sample_id"] = f"CASE-{sample_index}"
                item["candidate_response_text"] = json.dumps(item["parsed_answer"])
                item["trace"]["cpu_time_ms"] = (
                    sample_index if configuration == "A" else sample_index + 100
                )
                runs.append(item)
        analysis = paired_configuration_analysis_corrected(
            runs, metric_keys=("cpu_time_ms",)
        )
        row = next(
            value for value in analysis["results"] if value["comparison"] == "B-A"
        )
        self.assertEqual(row["primary_p_value"], row["wilcoxon_p_value"])
        self.assertEqual(
            row["holm_supported"], row["holm_adjusted_p_value"] < analysis["holm_alpha"]
        )
        self.assertIn(
            "paired-t p-values do not determine support",
            analysis["significance_decision_rule"],
        )


if __name__ == "__main__":
    unittest.main()
