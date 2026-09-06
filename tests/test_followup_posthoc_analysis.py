"""Tests for the locked-study post-hoc descriptive evidence pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.followup_conditions import FOLLOWUP_CONDITION_IDS  # noqa: E402
from src.followup_posthoc_analysis import (  # noqa: E402
    EXPECTED_BATCH_COUNT,
    EXPECTED_CALL_COUNT,
    EXPECTED_CASE_COUNT,
    build_fidelity_rows,
    build_gate_case_rows,
    build_latency_rows,
    classify_disposition,
    classify_fidelity_availability,
    fidelity_summary,
    gate_summary,
    latency_summary,
    load_v4_locked_runs,
)


SOURCE_RESULTS = (
    PROJECT_ROOT
    / "artifacts/results/followup/"
    / "FOLLOWUP-RETRIEVAL-20260902-02-CORRECTED-V4.json"
)


def locked_run_archive_available() -> bool:
    """Return whether every V4-selected manifest and raw-run set is available."""
    try:
        payload = json.loads(SOURCE_RESULTS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    sources = payload.get("source_manifests") or []
    if len(sources) != EXPECTED_BATCH_COUNT:
        return False
    for source in sources:
        manifest_path = PROJECT_ROOT / str(source.get("manifest_path", ""))
        if not manifest_path.is_file():
            return False
        expected = int(source.get("run_count", -1))
        if len(list((manifest_path.parent / "raw_runs").glob("*.json"))) != expected:
            return False
    return True


def synthetic_run(
    condition: str,
    *,
    status: str = "completed",
    candidate: str | None = None,
    output_tokens: int = 10,
    score: float = 0.7,
) -> dict:
    base = "A" if condition == "A_direct" else "B" if condition.startswith("B_") else "C"
    candidate_text = candidate if candidate is not None else json.dumps({
        "model_drivers": ["V1"]
    })
    if status == "suppressed":
        candidate_text = None
        output_tokens = 0
    try:
        parsed_answer = json.loads(candidate_text) if candidate_text else None
    except json.JSONDecodeError:
        parsed_answer = None
    return {
        "request_id": f"REQ-{condition}",
        "sample_id": "CASE-1",
        "condition_id": condition,
        "configuration": base,
        "status": status,
        "candidate_response_text": candidate_text,
        "parsed_answer": parsed_answer,
        "model_evidence": {
            "datasets": [{
                "dataset_id": "ULB",
                "top_alerts": [{
                    "fraud_probability": score,
                    "local_attribution": {"top_features": [{"feature": "V1"}]},
                }],
            }]
        },
        "trace": {
            "condition_id": condition,
            "output_tokens": output_tokens,
            "model_score_generation_gate": 0.5,
            "end_to_end_latency_ms": 100.0,
            "generation_latency_ms": 90.0,
            "retrieval_latency_ms": 5.0,
            "validation_latency_ms": 1.0,
            "cpu_time_ms": 10.0,
            "memory_rss_delta_bytes": 1024,
        },
    }


class PosthocUnitTests(unittest.TestCase):
    def test_gate_rows_require_and_collapse_one_six_condition_case(self) -> None:
        runs = [synthetic_run(condition) for condition in FOLLOWUP_CONDITION_IDS]
        rows = build_gate_case_rows(runs)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["guardrailed_generation_eligible"])
        self.assertEqual(gate_summary(rows)[0]["at_or_above_gate_count"], 1)

    def test_fidelity_reasons_follow_the_actual_candidate_contract(self) -> None:
        valid = synthetic_run("A_direct")
        invalid = synthetic_run("B_lexical", candidate="not-json")
        missing_drivers = synthetic_run("B_dense", candidate='{"claims": []}')
        suppressed = synthetic_run("C_lexical", status="suppressed")
        self.assertEqual(classify_fidelity_availability(valid)[0], "available")
        self.assertEqual(
            classify_fidelity_availability(invalid)[0], "candidate_invalid_json"
        )
        self.assertEqual(
            classify_fidelity_availability(missing_drivers)[0], "model_drivers_missing"
        )
        self.assertEqual(
            classify_fidelity_availability(suppressed)[0], "not_generated"
        )

    def test_disposition_is_mutually_exclusive_and_generation_latency_is_nullable(self) -> None:
        released = synthetic_run("A_direct")
        blocked = synthetic_run("C_dense", status="blocked")
        suppressed = synthetic_run("D_hybrid", status="suppressed")
        self.assertEqual(classify_disposition(released), "generated_released")
        self.assertEqual(classify_disposition(blocked), "generated_blocked")
        self.assertEqual(classify_disposition(suppressed), "pre_generation_suppressed")
        rows = build_latency_rows([released, blocked, suppressed])
        self.assertIsNone(rows[-1]["generation_latency_ms"])
        summary = latency_summary(rows)
        self.assertEqual(sum(summary["disposition_counts"].values()), 3)


@unittest.skipUnless(
    locked_run_archive_available(),
    "requires the separately retained immutable 3,300-run archive",
)
class LockedStudyPosthocContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runs, cls.provenance = load_v4_locked_runs(PROJECT_ROOT, SOURCE_RESULTS)

    def test_v4_selects_exact_locked_design(self) -> None:
        self.assertEqual(len(self.runs), EXPECTED_CALL_COUNT)
        self.assertEqual(self.provenance["accepted_manifest_count"], EXPECTED_BATCH_COUNT)
        self.assertEqual(self.provenance["case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(
            self.provenance["condition_ids"], list(FOLLOWUP_CONDITION_IDS)
        )

    def test_gate_denominators_match_the_immutable_cases(self) -> None:
        rows = build_gate_case_rows(self.runs)
        summaries = {row["dataset_id"]: row for row in gate_summary(rows)}
        self.assertEqual(len(rows), EXPECTED_CASE_COUNT)
        self.assertEqual(summaries["ULB"]["case_count"], 150)
        self.assertEqual(summaries["ULB"]["at_or_above_gate_count"], 68)
        self.assertEqual(summaries["ULB"]["below_gate_count"], 82)
        self.assertEqual(summaries["Sparkov"]["case_count"], 400)
        self.assertEqual(summaries["Sparkov"]["at_or_above_gate_count"], 201)
        self.assertEqual(summaries["Sparkov"]["below_gate_count"], 199)

    def test_fidelity_denominators_match_the_immutable_records(self) -> None:
        summary = fidelity_summary(build_fidelity_rows(self.runs))
        self.assertEqual(summary["all_run_count"], EXPECTED_CALL_COUNT)
        self.assertEqual(summary["generated_count"], 2457)
        self.assertEqual(summary["not_generated_count"], 843)
        self.assertEqual(summary["fidelity_available_count"], 1610)
        self.assertEqual(summary["fidelity_unavailable_count"], 847)

    def test_dispositions_partition_all_immutable_records(self) -> None:
        summary = latency_summary(build_latency_rows(self.runs))
        self.assertEqual(summary["all_run_count"], EXPECTED_CALL_COUNT)
        self.assertEqual(
            sum(summary["disposition_counts"].values()), EXPECTED_CALL_COUNT
        )
        self.assertEqual(
            summary["disposition_counts"]["pre_generation_suppressed"], 843
        )


if __name__ == "__main__":
    unittest.main()
