"""Contract tests for the three experimental LLM configurations."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_backend import (  # noqa: E402
    Configuration,
    PolicyEvidence,
    prepare_request,
    evaluate_configuration_c_response,
    validate_configuration_c_response,
)


def sample_bundle(probability: float = 0.91) -> dict:
    alert = {
        "evidence_id": "model-output:ulb:one",
        "dataset_id": "ULB",
        "record_id": "one",
        "model_name": "test-model",
        "fraud_probability": probability,
        "decision_threshold": 0.5,
        "fraud_alert": 1,
        "features": {"Amount": 100.0},
    }
    return {
        "combination_policy": "study-level fusion",
        "interpretation_notes": {"ULB": "anonymized", "Sparkov": "synthetic"},
        "cross_dataset_test_comparison": [],
        "datasets": [{
            "dataset_id": "ULB",
            "model_name": "test-model",
            "selected_threshold": 0.5,
            "primary_metric": "average_precision",
            "test_metrics": [],
            "top_alerts": [alert],
        }],
    }


class BackendContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PolicyEvidence(
            evidence_id="policy:manual:section-1",
            title="Fraud manual",
            text="Escalate high-risk transactions for human review.",
            source="fraud_manual.pdf#section-1",
        )

    def test_a_does_not_require_retrieval(self) -> None:
        request = prepare_request(
            configuration=Configuration.A,
            question="What should an analyst review?",
            unified_evidence=sample_bundle(),
        )
        self.assertTrue(request.should_call_llm)
        self.assertNotIn("Fraud manual", request.prompt)
        self.assertIn("Return only one JSON object", request.prompt)
        self.assertIn("plain evidence-ID strings, not citation objects", request.prompt)
        self.assertIn("Keep answer at most 70 words", request.prompt)
        payload = json.loads(request.prompt)
        self.assertEqual(
            payload["response_constraints"]["allowed_evidence_ids"],
            ["model-output:ulb:one"],
        )
        self.assertEqual(
            payload["response_constraints"]["required_citation_prefixes"], []
        )

    def test_a_includes_historical_evidence_ids(self) -> None:
        bundle = sample_bundle()
        bundle["historical_profiles"] = {
            "datasets": [{
                "dataset_id": "ULB",
                "records": [{
                    "evidence_id": "historical:ulb:prevalence",
                    "topic": "historical_prevalence",
                    "facts": {"fraud": 302},
                }],
            }],
        }
        request = prepare_request(
            configuration="A",
            question="What was historical fraud prevalence?",
            unified_evidence=bundle,
        )
        self.assertIn("historical:ulb:prevalence", request.evidence_ids)
        self.assertIn("historical_prevalence", request.prompt)

    def test_b_requires_retrieval(self) -> None:
        with self.assertRaises(ValueError):
            prepare_request(
                configuration="B",
                question="What policy applies?",
                unified_evidence=sample_bundle(),
            )

    def test_c_suppresses_low_confidence_alerts(self) -> None:
        request = prepare_request(
            configuration="C",
            question="Should this be escalated?",
            unified_evidence=sample_bundle(probability=0.6),
            retrieved_documents=[self.policy],
            model_score_gate=0.7,
        )
        self.assertFalse(request.should_call_llm)
        self.assertIsNotNone(request.suppression_reason)

    def test_c_rejects_unknown_citation(self) -> None:
        response = json.dumps({
            "answer": "Escalate.",
            "risk_level": "high",
            "self_reported_confidence": None,
            "citations": ["invented:source"],
            "claims": [{"statement": "Escalate.", "citations": ["invented:source"]}],
            "model_drivers": [],
            "recommended_actions": ["Human review"],
            "limitations": [],
        })
        with self.assertRaises(ValueError):
            validate_configuration_c_response(
                response,
                allowed_evidence_ids=["model-output:ulb:one"],
            )

    def test_c_requires_model_and_policy_citations(self) -> None:
        response = json.dumps({
            "answer": "Escalate under the supplied policy.",
            "risk_level": "high",
            "self_reported_confidence": None,
            "citations": ["policy:manual:section-1"],
            "claims": [{
                "statement": "Escalate under the supplied policy.",
                "citations": ["policy:manual:section-1"],
            }],
            "model_drivers": [],
            "recommended_actions": ["Human review"],
            "limitations": [],
        })
        with self.assertRaises(ValueError):
            validate_configuration_c_response(
                response,
                allowed_evidence_ids=[
                    "model-output:ulb:one",
                    "policy:manual:section-1",
                ],
            )

    def test_c_requires_claim_level_citations_and_scope_disclosure(self) -> None:
        response = json.dumps({
            "answer": "The model alert should receive human review.",
            "risk_level": "high",
            "self_reported_confidence": None,
            "citations": ["model-output:ulb:one", "policy:manual:section-1"],
            "claims": [{
                "statement": "The model alert should receive human review.",
                "citations": ["model-output:ulb:one", "policy:manual:section-1"],
            }],
            "model_drivers": [],
            "recommended_actions": ["Human review"],
            "limitations": ["Public sources are not institution-specific policy."],
        })
        validated = validate_configuration_c_response(
            response,
            allowed_evidence_ids=[
                "model-output:ulb:one", "policy:manual:section-1"
            ],
        )
        self.assertEqual(len(validated["claims"]), 1)

    def test_c_prompt_exposes_exact_constraint_allowlists(self) -> None:
        request = prepare_request(
            configuration="C",
            question="What should an analyst review?",
            unified_evidence=sample_bundle(),
            retrieved_documents=[self.policy],
        )
        constraints = json.loads(request.prompt)["response_constraints"]
        self.assertIn("model-output:ulb:one", constraints["allowed_evidence_ids"])
        self.assertIn("policy:manual:section-1", constraints["allowed_evidence_ids"])
        self.assertEqual(
            constraints["required_citation_prefixes"],
            ["model-output:", "policy:"],
        )
        self.assertEqual(
            constraints["required_limitation_text"],
            "Public sources are not institution-specific policy or legal advice.",
        )

    def test_c_reports_independent_structural_guardrail_outcomes(self) -> None:
        response = json.dumps({
            "answer": "Review it.",
            "risk_level": "high",
            "self_reported_confidence": 0.9,
            "citations": ["invented:source"],
            "claims": [{"statement": "Review it.", "citations": ["invented:source"]}],
            "model_drivers": ["invented_driver"],
            "recommended_actions": ["Review"],
            "limitations": [],
        })
        report = evaluate_configuration_c_response(
            response,
            allowed_evidence_ids=["model-output:ulb:one", "policy:manual:section-1"],
            allowed_model_drivers=["V14"],
        )
        self.assertFalse(report.passed)
        self.assertTrue(report.checks["valid_json_object"])
        self.assertFalse(report.checks["schema_contract"])
        self.assertFalse(report.checks["citation_allowlist"])
        self.assertFalse(report.checks["model_driver_allowlist"])


if __name__ == "__main__":
    unittest.main()
