"""Input manifest integrity test."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_audit import audit_input_artifacts  # noqa: E402


REQUIRED_LOCAL_INPUTS = (
    PROJECT_ROOT / "datasets/ULB_creditCard.csv",
    PROJECT_ROOT / "datasets/sparkov/fraudTrain.csv",
    PROJECT_ROOT / "datasets/sparkov/fraudTest.csv",
    PROJECT_ROOT / "datasets/ulb/followup_transaction_samples.csv",
    PROJECT_ROOT / "datasets/sparkov/followup_transaction_samples.csv",
)


@unittest.skipUnless(
    all(path.exists() for path in REQUIRED_LOCAL_INPUTS),
    "requires locally downloaded source datasets and reconstructed fixtures",
)
class ArtifactAuditTests(unittest.TestCase):
    def test_registered_inputs_are_present_and_unchanged(self):
        result = audit_input_artifacts(PROJECT_ROOT)
        self.assertEqual(result["status"], "passed", result["errors"])
        self.assertTrue(all(check["inside_datasets"] for check in result["checks"]))


if __name__ == "__main__":
    unittest.main()
