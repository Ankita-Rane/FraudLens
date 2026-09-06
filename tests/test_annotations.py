"""Annotation records are persistent, pseudonymised, and configuration-hidden."""

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import (  # noqa: E402
    AnnotationRecord,
    ClaimUnitRecord,
    JsonAnnotationRepository,
    JsonClaimUnitRepository,
    annotation_eligibility,
    blinded_annotation_task,
    counts_from_claim_assessments,
    pairwise_annotation_agreement,
)


class AnnotationTests(unittest.TestCase):
    def test_repository_persists_pseudonym_not_reviewer_identity(self):
        code = AnnotationRecord.pseudonymize_annotator("reviewer@example.test")
        record = AnnotationRecord(
            run_id="run-1",
            annotator_code=code,
            material_claims=2,
            unsupported_claims=1,
            citations_assessed=1,
            correct_citations=1,
            policy_dependent_claims=1,
            grounded_policy_claims=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonAnnotationRepository(Path(directory))
            repository.save(record)
            stored = repository.list_for_run("run-1")
        self.assertEqual(stored[0]["annotator_code"], code)
        self.assertNotIn("reviewer@example.test", str(stored))

    def test_blinded_task_hides_configuration_and_status(self):
        task = blinded_annotation_task({
            "request_id": "r1",
            "configuration": "C",
            "status": "completed",
            "answer": "Review it",
            "model_evidence": {"datasets": []},
        })
        self.assertNotIn("configuration", task)
        self.assertNotIn("status", task)
        self.assertEqual(task["model_evidence"], {"datasets": []})

    def test_counts_are_derived_from_atomic_claims(self):
        counts = counts_from_claim_assessments([{
            "claim_text": "A claim",
            "support_status": "unsupported",
            "failure_cause": "citation_fabrication",
            "citation_assessments": [{
                "evidence_id": "policy:1", "correctness": "incorrect"
            }],
            "policy_grounding": "ungrounded",
        }, {
            "claim_text": "Cannot assess",
            "support_status": "not_assessable",
            "failure_cause": "not_applicable",
            "citation_assessments": [],
            "policy_grounding": "not_applicable",
        }])
        self.assertEqual(counts["claims_total"], 2)
        self.assertEqual(counts["material_claims"], 1)
        self.assertEqual(counts["unsupported_claims"], 1)
        self.assertEqual(counts["not_assessable_claims"], 1)
        self.assertEqual(counts["failure_cause_counts"]["citation_fabrication"], 1)

    def test_unsupported_claim_requires_a_coded_failure_cause(self):
        with self.assertRaises(ValueError):
            counts_from_claim_assessments([{
                "support_status": "unsupported",
                "failure_cause": "not_applicable",
                "citation_assessments": [],
                "policy_grounding": "not_applicable",
            }])

    def test_demo_run_is_not_final_annotation_eligible(self):
        eligibility = annotation_eligibility({
            "status": "completed",
            "configuration": "A",
            "model_evidence": {"datasets": []},
            "trace": {
                "engine_name": "deterministic-evidence-demo",
                "experiment_id": "exp",
                "question_id": "q",
                "repeat": 1,
            },
        })
        self.assertFalse(eligibility["eligible"])
        self.assertIn("deterministic_demo_not_an_llm_result", eligibility["reasons"])

    def test_claim_units_are_frozen_and_cannot_be_overwritten(self):
        record = ClaimUnitRecord(
            run_id="run-1",
            creator_code="creator",
            claim_units=({"claim_id": "c1", "claim_text": "One claim."},),
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = JsonClaimUnitRepository(Path(directory))
            repository.save(record)
            stored = repository.get("run-1")
            with self.assertRaises(FileExistsError):
                repository.save(record)
        self.assertEqual(stored["claim_units"][0]["claim_id"], "c1")

    def test_agreement_uses_identical_claim_and_citation_units(self):
        base_claim = {
            "claim_id": "c1",
            "claim_text": "Claim",
            "support_status": "supported",
            "failure_cause": "not_applicable",
            "policy_grounding": "grounded",
            "citation_assessments": [{
                "evidence_id": "policy:1", "correctness": "correct"
            }],
        }
        result = pairwise_annotation_agreement(
            {"claim_assessments": [base_claim]},
            {"claim_assessments": [dict(base_claim)]},
        )
        self.assertEqual(result["support_raw_agreement"], 1.0)
        self.assertEqual(result["citation_correctness_raw_agreement"], 1.0)
