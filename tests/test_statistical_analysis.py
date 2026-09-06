"""Tests for paired A/B/C inferential analysis."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_analysis import _holm_adjust, paired_configuration_analysis  # noqa: E402
from tests.test_abc_evaluation import run  # noqa: E402


class StatisticalAnalysisTests(unittest.TestCase):
    def test_holm_is_per_dataset_and_retains_global_sensitivity(self):
        rows = [
            {"dataset_id": "ULB", "primary_p_value": 0.01},
            {"dataset_id": "ULB", "primary_p_value": 0.04},
            {"dataset_id": "Sparkov", "primary_p_value": 0.02},
            {"dataset_id": "Sparkov", "primary_p_value": 0.03},
        ]
        _holm_adjust(rows)
        self.assertEqual([row["holm_adjusted_p_value"] for row in rows], [0.02, 0.04, 0.04, 0.04])
        self.assertEqual(
            [row["holm_adjusted_p_value_global_sensitivity"] for row in rows],
            [0.04, 0.06, 0.06, 0.06],
        )

    def test_holm_preserves_tiny_nonzero_p_values(self):
        rows = [{"dataset_id": "ULB", "primary_p_value": 1e-14}]
        _holm_adjust(rows)
        self.assertGreater(rows[0]["holm_adjusted_p_value"], 0.0)
        self.assertEqual(rows[0]["holm_adjusted_p_value"], 1e-14)

    def test_analysis_keeps_datasets_and_pair_directions_explicit(self):
        runs = []
        for sample_id in ("CASE-1", "CASE-2"):
            for repeat in (1, 2):
                for configuration, latency in (("A", 30), ("B", 25), ("C", 20)):
                    item = run(configuration)
                    item["request_id"] = f"{sample_id}-{configuration}-{repeat}"
                    item["sample_id"] = sample_id
                    item["trace"]["repeat"] = repeat
                    item["trace"]["end_to_end_latency_ms"] = latency
                    runs.append(item)
        analysis = paired_configuration_analysis(
            runs, metric_keys=("processing_latency_ms",)
        )
        rows = analysis["results"]
        self.assertEqual({row["comparison"] for row in rows}, {"B-A", "C-B"})
        self.assertTrue(all(row["dataset_id"] == "Sparkov" for row in rows))
        self.assertTrue(all(row["pair_count"] == 2 for row in rows))

    def test_repeats_do_not_inflate_alert_pair_count(self):
        runs = []
        for repeat in (1, 2, 3):
            for configuration in ("A", "B", "C"):
                item = run(configuration)
                item["request_id"] = f"{configuration}-{repeat}"
                item["trace"]["repeat"] = repeat
                runs.append(item)
        analysis = paired_configuration_analysis(
            runs, metric_keys=("processing_latency_ms",)
        )
        self.assertTrue(all(row["pair_count"] == 1 for row in analysis["results"]))

    def test_batches_with_one_study_id_are_analysed_together(self):
        runs = []
        for batch, sample_id in (("batch-1", "CASE-1"), ("batch-2", "CASE-2")):
            for configuration in ("A", "B", "C"):
                item = run(configuration)
                item["request_id"] = f"{batch}-{configuration}"
                item["sample_id"] = sample_id
                item["trace"]["experiment_id"] = batch
                item["trace"]["study_id"] = "study-1"
                runs.append(item)
        analysis = paired_configuration_analysis(
            runs, metric_keys=("processing_latency_ms",)
        )
        self.assertTrue(all(row["pair_count"] == 2 for row in analysis["results"]))


if __name__ == "__main__":
    unittest.main()
