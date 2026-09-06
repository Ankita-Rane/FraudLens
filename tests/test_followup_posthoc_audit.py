"""Tests for independent post-hoc evidence verification."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.followup_posthoc_analysis import POSTHOC_ANALYSIS_ID  # noqa: E402
from tools.audit_followup_posthoc_evidence import audit_evidence  # noqa: E402


class FollowupPosthocEvidenceAuditTests(unittest.TestCase):
    def test_generated_evidence_bundle_passes_independent_audit(self) -> None:
        directory = (
            PROJECT_ROOT / "artifacts/results/followup/posthoc" / POSTHOC_ANALYSIS_ID
        )
        result = audit_evidence(directory)
        self.assertEqual(result["status"], "passed", result["errors"])
        self.assertEqual(result["check_count"], result["checks_passed"])
        self.assertEqual(
            result["observed_fidelity_unavailable_reasons"],
            {"candidate_invalid_json": 818, "model_drivers_missing": 29},
        )
        self.assertEqual(
            result["observed_disposition_counts"],
            {
                "generated_blocked": 796,
                "generated_released": 1661,
                "pre_generation_suppressed": 843,
            },
        )


if __name__ == "__main__":
    unittest.main()
