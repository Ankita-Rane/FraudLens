"""Checks for the local policy retrieval baseline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (  # noqa: E402
    chunk_policy_text,
    load_policy_directory,
    retrieve_policy_evidence,
)


class RetrievalTests(unittest.TestCase):
    def test_retrieval_prefers_relevant_policy(self) -> None:
        fraud = chunk_policy_text(
            title="Fraud escalation",
            text=("High risk fraud alerts require analyst escalation and card review. " * 8),
            source="fraud.md",
            chunk_size=300,
            overlap=20,
        )
        unrelated = chunk_policy_text(
            title="Office facilities",
            text=("Meeting rooms and office desks require advance reservation. " * 8),
            source="facilities.md",
            chunk_size=300,
            overlap=20,
        )
        results = retrieve_policy_evidence(
            "Should this fraud alert be escalated for analyst review?",
            fraud + unrelated,
            top_k=2,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].title, "Fraud escalation")
        self.assertEqual(results[0].retrieval_rank, 1)
        self.assertIsNotNone(results[0].retrieval_similarity)
        self.assertIsNotNone(results[0].chunk_hash)

    def test_no_documents_returns_no_results(self) -> None:
        self.assertEqual(retrieve_policy_evidence("fraud", []), [])

    def test_html_loader_excludes_navigation_outside_main(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "policy.html"
            path.write_text(
                "<html><body><nav>irrelevant navigation</nav>"
                "<main><h1>Review policy</h1><p>Escalate alerts for human review. "
                + "Evidence must be recorded. " * 30
                + "</p></main><footer>irrelevant footer</footer></body></html>",
                encoding="utf-8",
            )
            chunks = load_policy_directory(Path(temporary_directory))
        text = " ".join(chunk.text for chunk in chunks)
        self.assertIn("Escalate alerts", text)
        self.assertNotIn("irrelevant navigation", text)
        self.assertNotIn("irrelevant footer", text)


if __name__ == "__main__":
    unittest.main()
