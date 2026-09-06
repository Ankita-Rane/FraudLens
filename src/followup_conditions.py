"""Registered treatment definitions for the follow-up retrieval experiment.

The legacy A/B/C enum remains unchanged so the original locked study can still be
read with its original implementation.  Follow-up conditions make retrieval and
guardrail factors explicit while mapping onto the legacy prompt/validation modes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable

import json


@dataclass(frozen=True)
class FollowupCondition:
    condition_id: str
    base_configuration: str
    retrieval_strategy: str
    guardrail_profile: str
    analysis_role: str

    def __post_init__(self) -> None:
        if self.base_configuration not in {"A", "B", "C"}:
            raise ValueError("base_configuration must be A, B or C.")
        if self.retrieval_strategy not in {
            "none", "lexical_tfidf", "dense", "hybrid_rrf"
        }:
            raise ValueError("Unknown retrieval strategy.")
        if self.guardrail_profile not in {"off", "full"}:
            raise ValueError("guardrail_profile must be off or full.")
        requires_retrieval = self.base_configuration in {"B", "C"}
        if requires_retrieval == (self.retrieval_strategy == "none"):
            raise ValueError("Base configuration and retrieval strategy disagree.")
        if (self.base_configuration == "C") != (self.guardrail_profile == "full"):
            raise ValueError("Only base Configuration C may use full guardrails.")


FOLLOWUP_CONDITIONS: tuple[FollowupCondition, ...] = (
    FollowupCondition("A_direct", "A", "none", "off", "direct_baseline"),
    FollowupCondition("B_lexical", "B", "lexical_tfidf", "off", "factorial_core"),
    FollowupCondition("B_dense", "B", "dense", "off", "factorial_core"),
    FollowupCondition("C_lexical", "C", "lexical_tfidf", "full", "factorial_core"),
    FollowupCondition("C_dense", "C", "dense", "full", "factorial_core"),
    FollowupCondition("D_hybrid", "C", "hybrid_rrf", "full", "exploratory_hybrid"),
)

FOLLOWUP_CONDITION_IDS: tuple[str, ...] = tuple(
    condition.condition_id for condition in FOLLOWUP_CONDITIONS
)
FOLLOWUP_CONDITION_BY_ID = {
    condition.condition_id: condition for condition in FOLLOWUP_CONDITIONS
}


def condition_registry_payload(
    conditions: Iterable[FollowupCondition] = FOLLOWUP_CONDITIONS,
) -> list[dict[str, str]]:
    return [asdict(condition) for condition in conditions]


def condition_registry_sha256(
    conditions: Iterable[FollowupCondition] = FOLLOWUP_CONDITIONS,
) -> str:
    payload = json.dumps(
        condition_registry_payload(conditions),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def resolve_condition(condition_id: str) -> FollowupCondition:
    try:
        return FOLLOWUP_CONDITION_BY_ID[condition_id]
    except KeyError as error:
        raise ValueError(f"Unregistered follow-up condition: {condition_id}") from error


def counterbalanced_condition_sequence(
    cell_index: int,
    repeat: int,
    condition_ids: tuple[str, ...] = FOLLOWUP_CONDITION_IDS,
) -> tuple[str, ...]:
    """Rotate the starting treatment while preserving a registered cyclic order."""
    if cell_index < 0 or repeat < 1:
        raise ValueError("cell_index must be non-negative and repeat must be positive.")
    if not condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition_ids must be a non-empty unique sequence.")
    offset = (cell_index + repeat - 1) % len(condition_ids)
    return condition_ids[offset:] + condition_ids[:offset]
