"""Metric-definition and paired-design tests."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc_evaluation import (  # noqa: E402
    TRACE_FIELDS,
    ClaimAnnotation,
    aggregate_runs,
    evaluate_run,
    explanation_consistency,
    validate_paired_design,
)


def run(configuration="C"):
    payload = {
        "request_id": f"request-{configuration}",
        "configuration": configuration,
        "sample_id": "CASE",
        "question": "Assess it",
        "status": "completed",
        "answer": "Review the transaction.",
        "parsed_answer": {"answer": "Review", "model_drivers": ["amount"]},
        "candidate_response_text": '{"answer":"Review"}',
        "evidence_ids": ["model:1", "policy:1"],
        "retrieval_query": "Assess CASE under the applicable fraud policy.",
        "retrieved_policy": [{"evidence_id": "policy:1"}],
        "model_evidence": {"datasets": [{
            "dataset_id": "Sparkov",
            "top_alerts": [{"local_attribution": {
                "top_features": [{"feature": "amount"}],
            }}],
        }]},
        "guardrail_checks": {"schema": True, "citation": True},
        "guardrail_violations": [],
        "trace": {field: 1 for field in TRACE_FIELDS},
    }
    payload["trace"].update({
        "model_name": "test-model",
        "engine_name": "test-engine",
        "experiment_id": "experiment",
        "question_id": "q1",
        "repeat": 1,
    })
    event_specs = (
        ("model_evidence_loaded", "completed", ["model:1"], {"sample_id": "CASE"}),
        ("input_safety_assessed", "completed", [], {"findings": []}),
        (
            "policy_retrieval_skipped" if configuration == "A" else "policy_retrieval",
            "completed",
            [] if configuration == "A" else ["policy:1"],
            {"top_k": None, "result_count": 0, "latency_ms": 0.0}
            if configuration == "A"
            else {"top_k": 1, "result_count": 1, "latency_ms": 1.0},
        ),
        (
            "generation_completed", "completed", ["model:1", "policy:1"],
            {"model_name": "test-model", "input_tokens": 10, "output_tokens": 5},
        ),
        ("guardrail_check", "passed", ["model:1"], {"check_name": "schema"}),
        (
            "output_disposition", "released_for_analyst_review",
            ["model:1", "policy:1"], {"reasons": []},
        ),
        (
            "claim_ledger", "unavailable", [],
            {"claim_count": 0, "reason": "no_parseable_structured_claims"},
        ),
    )
    payload["audit_events"] = [
        {
            "event_id": str(index), "event_type": event_type, "status": status,
            "occurred_at_utc": "2026-01-01T00:00:00Z",
            "evidence_ids": evidence_ids, "details": details,
        }
        for index, (event_type, status, evidence_ids, details) in enumerate(event_specs)
    ]
    payload["requires_human_review"] = False
    return payload


class EvaluationTests(unittest.TestCase):
    def test_claim_metrics_remain_pending_without_annotation(self):
        metrics = evaluate_run(run())
        self.assertIsNone(metrics["hallucination_rate"])
        self.assertEqual(
            metrics["annotation_status"], "deferred_to_future_work_by_dec020"
        )

    def test_automatic_citation_metrics_do_not_claim_semantic_correctness(self):
        item = run("B")
        item["evidence_ids"] = ["model:1", "policy:1"]
        item["retrieved_policy"] = [{"evidence_id": "policy:1"}]
        item["parsed_answer"]["claims"] = [
            {"statement": "One", "citations": ["model:1"]},
            {"statement": "Two", "citations": ["policy:1", "invented:1"]},
        ]
        metrics = evaluate_run(item)
        self.assertEqual(metrics["response_json_valid"], 1.0)
        self.assertEqual(metrics["claim_citation_coverage"], 1.0)
        self.assertEqual(metrics["citation_id_validity"], 0.666667)
        self.assertEqual(metrics["policy_citation_coverage"], 0.5)
        self.assertIsNone(metrics["citation_correctness"])

    def test_claim_metrics_use_explicit_annotation_counts(self):
        metrics = evaluate_run(run(), ClaimAnnotation(
            material_claims=4,
            unsupported_claims=1,
            citations_assessed=3,
            correct_citations=2,
            policy_dependent_claims=2,
            grounded_policy_claims=2,
        ))
        self.assertEqual(metrics["hallucination_rate"], 0.25)
        self.assertEqual(metrics["citation_correctness"], 0.666667)
        self.assertEqual(metrics["policy_grounding_completeness"], 1.0)
        self.assertEqual(metrics["factual_grounding"], 0.75)

    def test_explanation_fidelity_matches_frozen_model_drivers(self):
        metrics = evaluate_run(run("B"))
        self.assertEqual(metrics["explanation_fidelity"], 1.0)

    def test_paired_design_requires_all_three_conditions(self):
        complete = validate_paired_design([run("A"), run("B"), run("C")])
        incomplete = validate_paired_design([run("A"), run("B")])
        self.assertTrue(complete["paired"])
        self.assertFalse(incomplete["paired"])

    def test_consistency_is_one_for_identical_answers(self):
        self.assertEqual(explanation_consistency(["Same answer", "Same answer"]), 1.0)

    def test_aggregate_does_not_compare_answers_to_different_questions(self):
        first = run("A")
        second = run("A")
        first["question"] = "Question one"
        second["question"] = "Question two"
        first["trace"]["question_id"] = "q1"
        second["trace"]["question_id"] = "q2"
        summary = aggregate_runs([first, second])[0]
        self.assertIsNone(summary["explanation_consistency"])
        self.assertEqual(summary["consistency_repeated_cell_count"], 0)

    def test_not_called_trace_does_not_require_prompt_hash(self):
        suppressed = run("C")
        suppressed["trace"]["model_name"] = "not-called:input-safety-block"
        suppressed["trace"]["prompt_sha256"] = None
        metrics = evaluate_run(suppressed)
        self.assertEqual(metrics["audit_trace_completeness"], 1.0)

    def test_suppressed_run_has_no_candidate_quality_but_zero_released_quality(self):
        suppressed = run("C")
        suppressed["status"] = "suppressed"
        suppressed["candidate_response_text"] = None
        suppressed["parsed_answer"] = None
        suppressed["trace"]["output_tokens"] = 0
        metrics = evaluate_run(suppressed)
        for key in (
            "candidate_response_json_valid",
            "candidate_explanation_fidelity",
            "candidate_claim_citation_coverage",
            "candidate_citation_id_validity",
            "candidate_policy_citation_coverage",
        ):
            self.assertIsNone(metrics[key], key)
        self.assertEqual(metrics["released_response_json_valid"], 0.0)
        self.assertEqual(metrics["released_claim_citation_coverage"], 0.0)

    def test_schema_only_event_is_not_counted_as_reconstructable(self):
        item = run("C")
        guardrail = next(
            event for event in item["audit_events"]
            if event["event_type"] == "guardrail_check"
        )
        guardrail["evidence_ids"] = []
        guardrail["details"] = {}
        metrics = evaluate_run(item)
        self.assertLess(metrics["audit_trace_completeness"], 1.0)
        self.assertEqual(metrics["audit_event_completeness"], 0.857143)
        self.assertIn("guardrail_check", metrics["audit_unreconstructable_event_types"])

    def test_explicit_empty_outcome_is_a_valid_evidential_basis(self):
        item = run("A")
        metrics = evaluate_run(item)
        self.assertEqual(metrics["audit_trace_completeness"], 1.0)
        self.assertEqual(metrics["audit_unreconstructable_event_types"], [])

    def test_blank_trace_value_is_not_complete(self):
        item = run("A")
        item["trace"]["request_id"] = "  "
        metrics = evaluate_run(item)
        self.assertLess(metrics["trace_field_completeness"], 1.0)

    def test_schema_validity_is_separate_from_other_c_guardrails(self):
        blocked = run("C")
        blocked["status"] = "blocked"
        blocked["parsed_answer"] = None
        blocked["guardrail_checks"] = {
            "valid_json_object": True,
            "schema_contract": True,
            "model_and_policy_citations": False,
        }
        blocked["guardrail_violations"] = [
            "model_and_policy_citations: missing retrieved-policy citation"
        ]
        metrics = evaluate_run(blocked)
        self.assertTrue(metrics["schema_valid"])
        self.assertEqual(metrics["guardrail_violation_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
