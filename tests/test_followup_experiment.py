"""Tests for the six-condition follow-up experiment contracts."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.abc_evaluation import validate_paired_design
from src.followup_conditions import (
    FOLLOWUP_CONDITION_IDS,
    condition_registry_sha256,
    counterbalanced_condition_sequence,
    resolve_condition,
)
from src.retrieval_strategies import (
    DensePolicyRepository,
    HashingSmokeTextEncoder,
    HybridRRFPolicyRepository,
    LexicalPolicyRepository,
)
from tools.build_followup_panels import select_label_stratified_positions
from tools.benchmark_followup_retrieval import _locked_expectations


class FollowupConditionTests(unittest.TestCase):
    def test_registered_conditions_are_orthogonal_and_stable(self) -> None:
        self.assertEqual(len(FOLLOWUP_CONDITION_IDS), 6)
        self.assertEqual(len(set(FOLLOWUP_CONDITION_IDS)), 6)
        self.assertEqual(len(condition_registry_sha256()), 64)
        self.assertEqual(resolve_condition("A_direct").retrieval_strategy, "none")
        self.assertEqual(resolve_condition("D_hybrid").guardrail_profile, "full")

    def test_counterbalancing_rotates_each_condition_to_each_position(self) -> None:
        sequences = [counterbalanced_condition_sequence(index, 1) for index in range(6)]
        self.assertEqual({sequence[0] for sequence in sequences}, set(FOLLOWUP_CONDITION_IDS))
        self.assertTrue(all(set(sequence) == set(FOLLOWUP_CONDITION_IDS) for sequence in sequences))

    def test_six_condition_pairing(self) -> None:
        runs = []
        for condition in FOLLOWUP_CONDITION_IDS:
            runs.append({
                "configuration": resolve_condition(condition).base_configuration,
                "condition_id": condition,
                "sample_id": "CASE-1",
                "question": "Q",
                "trace": {
                    "experiment_id": "EXP",
                    "question_id": "Q1",
                    "repeat": 1,
                    "engine_name": "demo",
                },
            })
        self.assertTrue(validate_paired_design(
            runs, expected_conditions=FOLLOWUP_CONDITION_IDS
        )["paired"])
        runs.pop()
        self.assertFalse(validate_paired_design(
            runs, expected_conditions=FOLLOWUP_CONDITION_IDS
        )["paired"])


class FollowupRetrievalTests(unittest.TestCase):
    def test_dense_and_rrf_return_stable_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "one.txt").write_text(
                "Fraud alerts require human investigation and documented evidence.",
                encoding="utf-8",
            )
            (source / "two.txt").write_text(
                "Model scores are not proof of fraud and require review.",
                encoding="utf-8",
            )
            lexical = LexicalPolicyRepository(source)
            dense = DensePolicyRepository(
                source,
                encoder=HashingSmokeTextEncoder(n_features=128),
                cache_directory=root / "cache",
            )
            hybrid = HybridRRFPolicyRepository(
                lexical=lexical, dense=dense, rrf_k=60, candidate_multiplier=2
            )
            dense_rows = dense.search("fraud alert investigation", top_k=2)
            hybrid_rows = hybrid.search("fraud alert investigation", top_k=2)
            self.assertEqual(len({row.evidence_id for row in dense_rows}), len(dense_rows))
            self.assertTrue(all(row.retrieval_strategy == "dense" for row in dense_rows))
            self.assertTrue(all(row.dense_rank for row in dense_rows))
            self.assertTrue(all(row.retrieval_strategy == "hybrid_rrf" for row in hybrid_rows))
            self.assertTrue(all(row.rrf_score is not None for row in hybrid_rows))
            self.assertEqual(lexical.corpus_sha256, dense.corpus_sha256)


class FollowupPanelTests(unittest.TestCase):
    def test_label_stratified_selection_is_deterministic_and_excludes_rows(self) -> None:
        predictions = pd.DataFrame({
            "row_index": list(range(12)),
            "actual": [1] * 6 + [0] * 6,
            "fraud_probability": [value / 11 for value in range(12)],
        })
        first = select_label_stratified_positions(
            predictions,
            quotas={1: 3, 0: 2},
            source_id_column="row_index",
            excluded_ids={0, 6},
        )
        second = select_label_stratified_positions(
            predictions,
            quotas={1: 3, 0: 2},
            source_id_column="row_index",
            excluded_ids={0, 6},
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertNotIn(0, first)
        self.assertNotIn(6, first)

    def test_impossible_quota_fails_closed(self) -> None:
        predictions = pd.DataFrame({
            "actual": [1, 0], "fraud_probability": [0.9, 0.1]
        })
        with self.assertRaisesRegex(ValueError, "only 1 exist"):
            select_label_stratified_positions(
                predictions,
                quotas={1: 2, 0: 0},
                source_id_column=None,
                excluded_ids=set(),
            )


class RetrievalDiagnosticLockTests(unittest.TestCase):
    def test_generated_manifest_is_hash_bound_to_real_registry(self) -> None:
        import json
        manifest = json.loads((Path(__file__).parents[1] / "datasets/evaluation/followup_retrieval/known_item_diagnostic_locked.json").read_text())
        judgments = _locked_expectations(manifest)
        self.assertEqual(len(judgments), 15)
        self.assertTrue(all(judgment.relevant_document_ids for judgment in judgments))

    def test_unlocked_manifest_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not locked"):
            _locked_expectations({"scientific_status": "draft"})


if __name__ == "__main__":
    unittest.main()
