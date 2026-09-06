"""Retrieval metrics require locked source-registry known-item expectations."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_backend import PolicyEvidence  # noqa: E402
from src.retrieval_evaluation import (  # noqa: E402
    LOCKED_KNOWN_ITEM_BASIS,
    RetrievalJudgment,
    evaluate_retrieval,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            PolicyEvidence("policy:1", "One", "x", "x", document_id="one"),
            PolicyEvidence("policy:2", "Two", "y", "y", document_id="two"),
        ]

    def test_unlocked_expectation_returns_null_metrics(self):
        result = evaluate_retrieval(self.results, RetrievalJudgment(
            "q1", "query", frozenset({"one"}), False
        ))
        self.assertIsNone(result["precision_at_k"])

    def test_locked_known_item_calculates_bounded_document_metrics(self):
        result = evaluate_retrieval(self.results, RetrievalJudgment(
            "q1", "query", frozenset({"one"}), LOCKED_KNOWN_ITEM_BASIS
        ))
        self.assertEqual(result["precision_at_k"], 0.5)
        self.assertEqual(result["reciprocal_rank"], 1.0)
        self.assertLessEqual(result["recall_at_k"], 1.0)
        self.assertLessEqual(result["ndcg_at_k"], 1.0)

    def test_precision_uses_fixed_cutoff_when_fewer_documents_return(self):
        result = evaluate_retrieval(
            self.results[:1],
            RetrievalJudgment("q1", "query", frozenset({"one"}), LOCKED_KNOWN_ITEM_BASIS),
            cutoff_k=4,
        )
        self.assertEqual(result["precision_at_k"], 0.25)
        self.assertEqual(result["returned_unique_document_count"], 1)
