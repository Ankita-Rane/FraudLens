"""Document-level metrics for a locked researcher-authored known-item diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable, Sequence

from src.llm_backend import PolicyEvidence


@dataclass(frozen=True)
class RetrievalJudgment:
    query_id: str
    query: str
    relevant_document_ids: frozenset[str]
    expectation_basis: str = "unlocked"


LOCKED_KNOWN_ITEM_BASIS = "locked_source_registry_known_item_expectation"


def evaluate_retrieval(
    retrieved: Sequence[PolicyEvidence], judgment: RetrievalJudgment, *, cutoff_k: int | None = None
) -> dict[str, float | int | str | bool | None]:
    """Calculate document-level P@k, recall, reciprocal rank, and binary nDCG."""
    if judgment.expectation_basis != LOCKED_KNOWN_ITEM_BASIS:
        return {
            "query_id": judgment.query_id,
            "diagnostic_eligible": False,
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
            "ndcg_at_k": None,
            "top_1_success": None,
            "warning": "Expected documents are not from a locked source-registry diagnostic; metrics are null.",
        }
    relevant = judgment.relevant_document_ids
    effective_k = cutoff_k if cutoff_k is not None else len(retrieved)
    if effective_k < 1:
        raise ValueError("cutoff_k must be positive.")
    ranked_ids: list[str] = []
    for document in retrieved:
        document_id = document.document_id or document.evidence_id
        if document_id not in ranked_ids:
            ranked_ids.append(document_id)
    hits = [1 if document_id in relevant else 0 for document_id in ranked_ids]
    hit_count = sum(hits)
    precision = hit_count / effective_k
    recall = hit_count / len(relevant) if relevant else None
    first_hit = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    reciprocal_rank = 1 / first_hit if first_hit is not None else 0.0
    dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), effective_k)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else None
    return {
        "query_id": judgment.query_id,
        "diagnostic_eligible": True,
        "k": effective_k,
        "returned_unique_document_count": len(ranked_ids),
        "precision_at_k": round(precision, 6),
        "recall_at_k": round(recall, 6) if recall is not None else None,
        "reciprocal_rank": round(reciprocal_rank, 6),
        "ndcg_at_k": round(ndcg, 6) if ndcg is not None else None,
        "top_1_success": bool(hits and hits[0]),
    }


def macro_average(results: Iterable[dict]) -> dict[str, float | int | None]:
    approved = [result for result in results if result.get("diagnostic_eligible")]
    keys = ("precision_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k", "top_1_success")
    return {
        "eligible_query_count": len(approved),
        **{
            key: (
                round(sum(float(row[key]) for row in approved if row.get(key) is not None)
                      / sum(row.get(key) is not None for row in approved), 6)
                if any(row.get(key) is not None for row in approved)
                else None
            )
            for key in keys
        },
    }
