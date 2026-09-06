"""Checks for the generated training-only historical evidence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.historical import load_historical_profiles  # noqa: E402


class HistoricalProfileTests(unittest.TestCase):
    def test_profiles_have_expected_training_counts(self) -> None:
        profiles = load_historical_profiles(PROJECT_ROOT)
        by_dataset = {item["dataset_id"]: item for item in profiles["datasets"]}
        ulb_prevalence = next(
            record for record in by_dataset["ULB"]["records"]
            if record["topic"] == "historical_prevalence"
        )
        sparkov_prevalence = next(
            record for record in by_dataset["Sparkov"]["records"]
            if record["topic"] == "historical_prevalence"
        )
        self.assertEqual(ulb_prevalence["facts"]["fraud"], 302)
        self.assertEqual(sparkov_prevalence["facts"]["fraud"], 5_968)
        self.assertTrue(all(
            record["evidence_id"].startswith("historical:")
            for dataset in profiles["datasets"]
            for record in dataset["records"]
        ))


if __name__ == "__main__":
    unittest.main()
