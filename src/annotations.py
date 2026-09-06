"""Persistent blinded human annotations for thesis reliability metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import os
import tempfile
import uuid

from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class AnnotationRecord:
    run_id: str
    annotator_code: str
    material_claims: int
    unsupported_claims: int
    citations_assessed: int
    correct_citations: int
    policy_dependent_claims: int
    grounded_policy_claims: int
    claims_total: int | None = None
    not_assessable_claims: int = 0
    claim_assessments: tuple[dict[str, Any], ...] = ()
    configuration_guess: str | None = None
    notes: str = ""
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @staticmethod
    def pseudonymize_annotator(reviewer_code: str) -> str:
        normalized = reviewer_code.strip()
        if not normalized:
            raise ValueError("reviewer_code must not be empty.")
        return sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ClaimUnitRecord:
    run_id: str
    creator_code: str
    claim_units: tuple[dict[str, Any], ...]
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class JsonClaimUnitRepository:
    """Store one frozen, configuration-blind atomic-claim set per run."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def save(self, record: ClaimUnitRecord) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{record.run_id}.json"
        if path.exists():
            raise FileExistsError(
                "Claim units are frozen once created; adjudicate through a new protocol "
                f"version instead of overwriting {path}."
            )
        payload = json.dumps(asdict(record), indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._directory,
            prefix=f".{record.run_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        return path

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self._directory / f"{run_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class JsonAnnotationRepository:
    """Append-only local adapter; suitable for research, not a regulated WORM store."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def save(self, record: AnnotationRecord) -> Path:
        target_directory = self._directory / record.run_id
        target_directory.mkdir(parents=True, exist_ok=True)
        path = target_directory / f"{record.annotation_id}.json"
        payload = json.dumps(asdict(record), indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_directory,
            prefix=f".{record.annotation_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        return path

    def list_for_run(self, run_id: str) -> list[dict[str, Any]]:
        directory = self._directory / run_id
        if not directory.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
        return records


def blinded_annotation_task(
    run: dict[str, Any], claim_units: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Expose evidence and answer without the configuration label or guardrail status."""
    eligibility = annotation_eligibility(run)
    return {
        "task_id": run.get("request_id"),
        "sample_id": run.get("sample_id"),
        "question": run.get("question"),
        "answer": run.get("answer"),
        "model_evidence": run.get("model_evidence"),
        "evidence_ids": run.get("evidence_ids", []),
        "retrieved_policy": run.get("retrieved_policy", []),
        "claim_units": claim_units or [],
        "annotation_eligibility": eligibility,
        "annotation_contract": {
            "unit": "one atomic material claim",
            "support_status": ["supported", "unsupported", "not_assessable"],
            "citation_correctness": ["correct", "incorrect", "not_assessable"],
            "policy_grounding": ["grounded", "ungrounded", "not_applicable"],
            "failure_cause": [
                "retrieval_miss", "grounding_misrepresentation",
                "citation_fabrication", "transaction_evidence_mismatch",
                "attribution_mismatch", "format_or_specification_failure",
                "unsupported_other", "not_applicable",
            ],
            "instruction": (
                "Assess each frozen claim_id against only the supplied model and "
                "retrieved-policy evidence. Do not add, remove, or rewrite claim units."
            ),
        },
        "blinding_note": (
            "Configuration and enforcement status are withheld. Formatting may still "
            "make perfect blinding impossible; measure and report reviewer guesses."
        ),
    }


def annotation_eligibility(run: dict[str, Any]) -> dict[str, Any]:
    """Prevent demo, unpaired, or evidence-incomplete runs entering final annotation."""
    trace = run.get("trace") or {}
    reasons: list[str] = []
    if run.get("status") != "completed":
        reasons.append("run_not_completed")
    if trace.get("engine_name") == "deterministic-evidence-demo":
        reasons.append("deterministic_demo_not_an_llm_result")
    if not trace.get("experiment_id") or not trace.get("question_id") or not trace.get("repeat"):
        reasons.append("missing_registered_experiment_cell_metadata")
    if not run.get("model_evidence"):
        reasons.append("model_evidence_snapshot_missing")
    if run.get("configuration") in {"B", "C"} and not run.get("retrieved_policy"):
        reasons.append("retrieved_policy_snapshot_missing")
    return {"eligible": not reasons, "reasons": reasons}


def counts_from_claim_assessments(
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive metric counts from stored atomic-claim and citation assessments."""
    if not claims:
        raise ValueError("At least one atomic claim assessment is required.")
    failure_causes = (
        "retrieval_miss", "grounding_misrepresentation", "citation_fabrication",
        "transaction_evidence_mismatch", "attribution_mismatch",
        "format_or_specification_failure", "unsupported_other",
    )
    for claim in claims:
        status = claim.get("support_status")
        cause = claim.get("failure_cause", "not_applicable")
        if status == "unsupported" and cause not in failure_causes:
            raise ValueError("Every unsupported claim requires one coded failure_cause.")
        if status != "unsupported" and cause != "not_applicable":
            raise ValueError("Only unsupported claims may have a failure_cause.")
    citations = [
        citation
        for claim in claims
        for citation in claim.get("citation_assessments", [])
        if citation.get("correctness") in {"correct", "incorrect"}
    ]
    policy_claims = [
        claim
        for claim in claims
        if claim.get("policy_grounding") in {"grounded", "ungrounded"}
    ]
    result = {
        "claims_total": len(claims),
        "material_claims": sum(
            claim.get("support_status") in {"supported", "unsupported"}
            for claim in claims
        ),
        "unsupported_claims": sum(
            claim.get("support_status") == "unsupported" for claim in claims
        ),
        "not_assessable_claims": sum(
            claim.get("support_status") == "not_assessable" for claim in claims
        ),
        "citations_assessed": len(citations),
        "correct_citations": sum(
            citation.get("correctness") == "correct" for citation in citations
        ),
        "policy_dependent_claims": len(policy_claims),
        "grounded_policy_claims": sum(
            claim.get("policy_grounding") == "grounded" for claim in policy_claims
        ),
    }
    result["failure_cause_counts"] = {
        cause: sum(claim.get("failure_cause") == cause for claim in claims)
        for cause in failure_causes
    }
    return result


def pairwise_annotation_agreement(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Calculate agreement for two reviewers rating the same frozen claim IDs."""
    first_claims = {
        claim["claim_id"]: claim for claim in first.get("claim_assessments", [])
    }
    second_claims = {
        claim["claim_id"]: claim for claim in second.get("claim_assessments", [])
    }
    if not first_claims or set(first_claims) != set(second_claims):
        raise ValueError("Reviewers must assess the same non-empty frozen claim-id set.")

    ordered_ids = sorted(first_claims)

    def agreement_for(field: str) -> tuple[float, float | None]:
        left = [first_claims[claim_id][field] for claim_id in ordered_ids]
        right = [second_claims[claim_id][field] for claim_id in ordered_ids]
        raw = sum(a == b for a, b in zip(left, right)) / len(left)
        labels = set(left) | set(right)
        kappa = float(cohen_kappa_score(left, right)) if len(labels) > 1 else None
        return round(raw, 6), round(kappa, 6) if kappa is not None else None

    support_raw, support_kappa = agreement_for("support_status")
    policy_raw, policy_kappa = agreement_for("policy_grounding")
    def citation_map(claims: dict[str, dict[str, Any]]) -> dict[tuple[str, str], str]:
        return {
            (claim_id, citation["evidence_id"]): citation["correctness"]
            for claim_id, claim in claims.items()
            for citation in claim.get("citation_assessments", [])
        }

    first_citations = citation_map(first_claims)
    second_citations = citation_map(second_claims)
    if set(first_citations) != set(second_citations):
        raise ValueError("Reviewers must assess the same frozen claim-citation pairs.")
    citation_raw: float | None = None
    citation_kappa: float | None = None
    if first_citations:
        ordered_citations = sorted(first_citations)
        left = [first_citations[key] for key in ordered_citations]
        right = [second_citations[key] for key in ordered_citations]
        citation_raw = round(
            sum(a == b for a, b in zip(left, right)) / len(left), 6
        )
        if len(set(left) | set(right)) > 1:
            citation_kappa = round(float(cohen_kappa_score(left, right)), 6)
    return {
        "claim_count": len(ordered_ids),
        "support_raw_agreement": support_raw,
        "support_cohen_kappa": support_kappa,
        "policy_grounding_raw_agreement": policy_raw,
        "policy_grounding_cohen_kappa": policy_kappa,
        "citation_pair_count": len(first_citations),
        "citation_correctness_raw_agreement": citation_raw,
        "citation_correctness_cohen_kappa": citation_kappa,
        "kappa_note": (
            None if support_kappa is not None or policy_kappa is not None
            else "Kappa is undefined when both reviewers use only one category; report raw agreement."
        ),
    }
