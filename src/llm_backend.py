"""Shared request contracts for Configuration A, B, and C.

This module prepares backend requests and validates Configuration C responses. It is
LLM-provider agnostic: a UI/API layer can pass the resulting prompt to a local model or
hosted inference service without changing the experimental conditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

import json
import time
import uuid


class Configuration(str, Enum):
    A = "A"
    B = "B"
    C = "C"


CONFIGURATION_REGISTRY: dict[str, dict[str, Any]] = {
    "A": {
        "title": "Baseline System",
        "retrieval": False,
        "guardrail_enforcement": False,
        "structured_output_requested": True,
        "structured_output_enforced": False,
        "observability": True,
        "cost_tracking": True,
    },
    "B": {
        "title": "Retrieval-Augmented System",
        "retrieval": True,
        "guardrail_enforcement": False,
        "structured_output_requested": True,
        "structured_output_enforced": False,
        "observability": True,
        "cost_tracking": True,
    },
    "C": {
        "title": "Guardrail-Aware RAG Investigation Assistant",
        "retrieval": True,
        "guardrail_enforcement": True,
        "structured_output_requested": True,
        "structured_output_enforced": True,
        "observability": True,
        "cost_tracking": True,
    },
}


@dataclass(frozen=True)
class PolicyEvidence:
    evidence_id: str
    title: str
    text: str
    source: str
    document_id: str | None = None
    chunk_hash: str | None = None
    retrieval_rank: int | None = None
    retrieval_similarity: float | None = None
    publisher: str | None = None
    jurisdiction: str | None = None
    authority: str | None = None
    snapshot_date: str | None = None
    effective_date: str | None = None
    use_for: tuple[str, ...] = ()
    not_for: tuple[str, ...] = ()
    retrieval_strategy: str | None = None
    lexical_rank: int | None = None
    lexical_similarity: float | None = None
    dense_rank: int | None = None
    dense_similarity: float | None = None
    rrf_score: float | None = None
    rrf_k: int | None = None


@dataclass(frozen=True)
class PreparedRequest:
    request_id: str
    configuration: str
    question: str
    prompt: str
    evidence_ids: list[str]
    allowed_model_drivers: list[str]
    should_call_llm: bool
    suppression_reason: str | None
    started_at_utc: str


def _compact_model_context(
    unified_evidence: dict[str, Any],
    *,
    minimum_probability: float | None = None,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    context = {
        "combination_policy": unified_evidence["combination_policy"],
        "interpretation_notes": unified_evidence["interpretation_notes"],
        "cross_dataset_test_comparison": unified_evidence["cross_dataset_test_comparison"],
        "historical_profiles": unified_evidence.get("historical_profiles"),
        "active_evaluation_sample": unified_evidence.get("active_evaluation_sample"),
        "datasets": [],
    }
    alert_evidence_ids: list[str] = []
    for dataset in unified_evidence["datasets"]:
        alerts = dataset["top_alerts"]
        if minimum_probability is not None:
            alerts = [
                alert for alert in alerts
                if alert["fraud_probability"] >= minimum_probability
            ]
        alert_evidence_ids.extend(alert["evidence_id"] for alert in alerts)
        context["datasets"].append({
            "dataset_id": dataset["dataset_id"],
            "model_name": dataset["model_name"],
            "selected_threshold": dataset["selected_threshold"],
            "primary_metric": dataset["primary_metric"],
            "test_metrics": dataset["test_metrics"],
            "top_alerts": alerts,
        })
    contextual_evidence_ids: list[str] = [
        row["evidence_id"]
        for dataset in context["datasets"]
        for row in dataset.get("test_metrics", [])
        if row.get("evidence_id")
    ]
    contextual_evidence_ids.extend([
        str(alert["local_attribution"]["evidence_id"])
        for dataset in context["datasets"]
        for alert in dataset.get("top_alerts", [])
        if (alert.get("local_attribution") or {}).get("evidence_id")
    ])
    historical_profiles = context["historical_profiles"]
    if historical_profiles:
        contextual_evidence_ids.extend([
            record["evidence_id"]
            for dataset in historical_profiles.get("datasets", [])
            for record in dataset.get("records", [])
        ])
    model_drivers = sorted({
        str(item.get("feature") or item.get("feature_group"))
        for dataset in context["datasets"]
        for alert in dataset.get("top_alerts", [])
        for item in (alert.get("local_attribution") or {}).get("top_features", [])
        if item.get("feature") or item.get("feature_group")
    })
    return context, alert_evidence_ids, contextual_evidence_ids, model_drivers


def _policy_payload(documents: Iterable[PolicyEvidence]) -> list[dict[str, str]]:
    return [asdict(document) for document in documents]


def prepare_request(
    *,
    configuration: Configuration | str,
    question: str,
    unified_evidence: dict[str, Any],
    retrieved_documents: Iterable[PolicyEvidence] = (),
    model_score_gate: float = 0.70,
) -> PreparedRequest:
    """Prepare one of the three controlled experimental conditions."""
    mode = Configuration(configuration)
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")
    if not 0 <= model_score_gate <= 1:
        raise ValueError("model_score_gate must be between 0 and 1.")

    policy_documents = list(retrieved_documents)
    if mode in {Configuration.B, Configuration.C} and not policy_documents:
        raise ValueError(f"Configuration {mode.value} requires retrieved policy evidence.")

    minimum_probability = model_score_gate if mode is Configuration.C else None
    (
        model_context,
        alert_evidence_ids,
        contextual_evidence_ids,
        allowed_model_drivers,
    ) = _compact_model_context(
        unified_evidence, minimum_probability=minimum_probability
    )
    policy_payload = _policy_payload(policy_documents)
    policy_evidence_ids = [document.evidence_id for document in policy_documents]
    all_evidence_ids = (
        alert_evidence_ids + contextual_evidence_ids + policy_evidence_ids
    )

    if mode is Configuration.C and not alert_evidence_ids:
        return PreparedRequest(
            request_id=str(uuid.uuid4()),
            configuration=mode.value,
            question=question,
            prompt="",
            evidence_ids=all_evidence_ids,
            allowed_model_drivers=allowed_model_drivers,
            should_call_llm=False,
            suppression_reason=(
                "No model alert met the configured fraud-model score gate; generation "
                "was suppressed by Configuration C."
            ),
            started_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    if mode is Configuration.A:
        instruction = (
            "Answer directly from the supplied fraud-model evidence. This is the "
            "baseline condition: do not retrieve policies or add external context. "
            "Explain dataset limitations and never invent meanings for ULB V1-V28."
        )
        citation_rule = (
            "Citations may reference only supplied model evidence IDs; no policy source "
            "is available in this condition."
        )
    elif mode is Configuration.B:
        instruction = (
            "Answer using the supplied fraud-model evidence and retrieved policy text. "
            "This condition uses RAG but does not apply programmatic guardrail checks. "
            "retrieval_similarity measures text relevance and is never transaction risk."
        )
        citation_rule = (
            "Citations should reference supplied model and retrieved-policy evidence IDs. "
            "No programmatic output validator is applied in this condition."
        )
    else:
        instruction = (
            "Answer only from the supplied model and policy evidence. Every material "
            "claim must cite one or more supplied evidence_id values in its own claim "
            "object. If evidence conflicts or is insufficient, state that explicitly. "
            "The sources are public research context, not institution-specific policy "
            "or legal advice. Do not recommend an automatic adverse action."
            " retrieval_similarity measures text relevance and is never transaction risk."
        )
        citation_rule = (
            "Citations must reference supplied model and retrieved-policy evidence IDs. "
            "Configuration C will enforce the response contract after generation."
        )

    output_instruction = (
        "Return only one JSON object with exactly these keys: answer (non-empty string), "
        "risk_level (low|medium|high), self_reported_confidence (null), citations "
        "(non-empty evidence-ID array), claims (non-empty array of objects containing "
        "statement and citations), model_drivers (array containing only supplied "
        "attribution feature or feature-group names), recommended_actions (string array), "
        "and limitations (string array). Every material claim must carry evidence IDs. "
        f"{citation_rule} Self-reported confidence must remain null because this study "
        "has no calibrated LLM answer-confidence model. If no local attribution is "
        "supplied, model_drivers must be an empty array. Use this exact compact shape "
        "with plain evidence-ID strings, not citation objects: "
        '{"answer":"...","risk_level":"medium","self_reported_confidence":null,'
        '"citations":["evidence_id"],"claims":[{"statement":"...",'
        '"citations":["evidence_id"]}],"model_drivers":["feature_name"],'
        '"recommended_actions":["..."],"limitations":["..."]}. '
        "Keep answer at most 70 words; use at most 3 claims, 4 citations, 3 model "
        "drivers, 2 recommended actions, and 2 limitations. Keep every list item "
        "under 25 words. Do not repeat evidence text."
    )
    if mode in {Configuration.B, Configuration.C}:
        output_instruction += (
            " The limitations must state that public sources are not institution-specific "
            "policy or legal advice."
        )

    prompt_payload = {
        "instruction": instruction,
        "question": question,
        "model_evidence": model_context,
        "retrieved_policy_evidence": policy_payload if mode is not Configuration.A else [],
        "response_constraints": {
            "allowed_evidence_ids": all_evidence_ids,
            "required_citation_prefixes": (
                ["model-output:", "policy:"]
                if mode is Configuration.C else []
            ),
            "allowed_model_drivers": allowed_model_drivers,
            "minimum_model_drivers": 1 if allowed_model_drivers else 0,
            "required_limitation_text": (
                "Public sources are not institution-specific policy or legal advice."
                if mode in {Configuration.B, Configuration.C} else None
            ),
        },
        "output_instruction": output_instruction,
    }
    return PreparedRequest(
        request_id=str(uuid.uuid4()),
        configuration=mode.value,
        question=question,
        prompt=json.dumps(prompt_payload, indent=2, ensure_ascii=False),
        evidence_ids=all_evidence_ids,
        allowed_model_drivers=allowed_model_drivers,
        should_call_llm=True,
        suppression_reason=None,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
    )


@dataclass(frozen=True)
class ConfigurationCValidationReport:
    """Independent structural guardrail outcomes for one C candidate response.

    These checks establish format, provenance membership, and response-scope contracts.
    They do not establish semantic entailment between a claim and a cited passage; that
    remains a blinded human-annotation outcome in this study.
    """

    response: dict[str, Any] | None
    checks: dict[str, bool]
    violations: tuple[str, ...]
    validation_scope: str = "structural_provenance_not_semantic_entailment"

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())


def evaluate_configuration_c_response(
    response_text: str,
    *,
    allowed_evidence_ids: Iterable[str],
    allowed_model_drivers: Iterable[str] = (),
) -> ConfigurationCValidationReport:
    """Evaluate every C guardrail independently instead of stopping at first failure."""
    check_names = (
        "valid_json_object",
        "schema_contract",
        "citation_allowlist",
        "model_and_policy_citations",
        "claim_level_citation_structure",
        "model_driver_allowlist",
        "public_source_scope_disclosure",
    )
    checks = {name: False for name in check_names}
    violations: list[str] = []
    try:
        candidate = json.loads(response_text)
    except json.JSONDecodeError as error:
        violations.append(f"valid_json_object: {error.msg}")
        return ConfigurationCValidationReport(None, checks, tuple(violations))
    if not isinstance(candidate, dict):
        violations.append("valid_json_object: response must be a JSON object.")
        return ConfigurationCValidationReport(None, checks, tuple(violations))
    checks["valid_json_object"] = True
    response: dict[str, Any] = candidate

    required = {
        "answer",
        "risk_level",
        "self_reported_confidence",
        "citations",
        "claims",
        "model_drivers",
        "recommended_actions",
        "limitations",
    }
    schema_errors: list[str] = []
    missing = required - response.keys()
    extra = response.keys() - required
    if missing:
        schema_errors.append(f"missing keys {sorted(missing)}")
    if extra:
        schema_errors.append(f"unexpected keys {sorted(extra)}")
    if not isinstance(response.get("answer"), str) or not response.get("answer", "").strip():
        schema_errors.append("answer must be a non-empty string")
    if response.get("risk_level") not in {"low", "medium", "high"}:
        schema_errors.append("risk_level must be low, medium, or high")
    if response.get("self_reported_confidence", object()) is not None:
        schema_errors.append(
            "self_reported_confidence must be null; no calibrated LLM confidence exists"
        )
    for key in ("citations", "model_drivers", "recommended_actions", "limitations"):
        value = response.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            schema_errors.append(f"{key} must be a string array")
    if isinstance(response.get("citations"), list) and not response["citations"]:
        schema_errors.append("citations must not be empty")
    if not isinstance(response.get("claims"), list) or not response.get("claims"):
        schema_errors.append("claims must be a non-empty array")
    checks["schema_contract"] = not schema_errors
    violations.extend(f"schema_contract: {error}" for error in schema_errors)

    allowed = set(allowed_evidence_ids)
    cited_values = response.get("citations", [])
    cited = {value for value in cited_values if isinstance(value, str)}
    unknown = cited - allowed
    checks["citation_allowlist"] = (
        isinstance(cited_values, list)
        and bool(cited_values)
        and len(cited) == len(cited_values)
        and not unknown
    )
    if unknown:
        violations.append(f"citation_allowlist: unsupported citations {sorted(unknown)}")
    elif not checks["citation_allowlist"]:
        violations.append("citation_allowlist: citations must be unique supplied evidence IDs")

    allowed_policy = {value for value in allowed if value.startswith("policy:")}
    allowed_model_outputs = {value for value in allowed if value.startswith("model-output:")}
    model_policy_ok = True
    if allowed_model_outputs and not (cited & allowed_model_outputs):
        model_policy_ok = False
        violations.append("model_and_policy_citations: missing model-output citation")
    if allowed_policy and not (cited & allowed_policy):
        model_policy_ok = False
        violations.append("model_and_policy_citations: missing retrieved-policy citation")
    checks["model_and_policy_citations"] = model_policy_ok

    claims = response.get("claims")
    claim_structure_ok = isinstance(claims, list) and bool(claims)
    claim_citations: set[str] = set()
    if isinstance(claims, list):
        for index, claim in enumerate(claims):
            valid_claim = (
                isinstance(claim, dict)
                and set(claim) == {"statement", "citations"}
                and isinstance(claim.get("statement"), str)
                and bool(claim.get("statement", "").strip())
                and isinstance(claim.get("citations"), list)
                and bool(claim.get("citations"))
                and all(isinstance(value, str) for value in claim.get("citations", []))
            )
            if not valid_claim:
                claim_structure_ok = False
                violations.append(
                    f"claim_level_citation_structure: claims[{index}] is malformed"
                )
                continue
            values = set(claim["citations"])
            if len(values) != len(claim["citations"]):
                claim_structure_ok = False
                violations.append(
                    "claim_level_citation_structure: "
                    f"claims[{index}] citations must be unique"
                )
            claim_citations.update(values)
            claim_unknown = values - allowed
            if claim_unknown:
                claim_structure_ok = False
                violations.append(
                    "claim_level_citation_structure: "
                    f"claims[{index}] has unsupported citations {sorted(claim_unknown)}"
                )
    if not claim_citations.issubset(cited):
        claim_structure_ok = False
        violations.append(
            "claim_level_citation_structure: claim citations must appear at top level"
        )
    checks["claim_level_citation_structure"] = claim_structure_ok

    allowed_drivers = set(allowed_model_drivers)
    reported_driver_values = response.get("model_drivers", [])
    reported_drivers = {
        value for value in reported_driver_values if isinstance(value, str)
    }
    unknown_drivers = reported_drivers - allowed_drivers
    driver_ok = (
        isinstance(reported_driver_values, list)
        and len(reported_drivers) == len(reported_driver_values)
        and not unknown_drivers
        and (bool(reported_drivers) if allowed_drivers else not reported_drivers)
    )
    checks["model_driver_allowlist"] = driver_ok
    if unknown_drivers:
        violations.append(
            f"model_driver_allowlist: unsupported drivers {sorted(unknown_drivers)}"
        )
    elif not driver_ok:
        violations.append(
            "model_driver_allowlist: use supplied drivers, or an empty array when none exist"
        )

    limitations = response.get("limitations")
    scope_text = " ".join(limitations).lower() if isinstance(limitations, list) and all(
        isinstance(value, str) for value in limitations
    ) else ""
    scope_markers = (
        "not institution-specific",
        "not bank policy",
        "public source",
        "not legal advice",
    )
    scope_ok = not allowed_policy or any(marker in scope_text for marker in scope_markers)
    checks["public_source_scope_disclosure"] = scope_ok
    if not scope_ok:
        violations.append(
            "public_source_scope_disclosure: disclose that public sources are not "
            "institution-specific policy or legal advice"
        )
    return ConfigurationCValidationReport(response, checks, tuple(violations))


def validate_configuration_c_response(
    response_text: str,
    *,
    allowed_evidence_ids: Iterable[str],
    allowed_model_drivers: Iterable[str] = (),
) -> dict[str, Any]:
    """Return a validated C response or raise with all structural violations."""
    report = evaluate_configuration_c_response(
        response_text,
        allowed_evidence_ids=allowed_evidence_ids,
        allowed_model_drivers=allowed_model_drivers,
    )
    if not report.passed or report.response is None:
        raise ValueError("; ".join(report.violations) or "Configuration C validation failed.")
    return report.response


def build_observation(
    request: PreparedRequest,
    *,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
    started_monotonic: float | None = None,
) -> dict[str, Any]:
    """Create the Configuration C observability and cost record."""
    if min(input_tokens, output_tokens, input_cost_per_million, output_cost_per_million) < 0:
        raise ValueError("Token counts and prices cannot be negative.")
    elapsed_ms = None
    if started_monotonic is not None:
        elapsed_ms = round((time.monotonic() - started_monotonic) * 1_000, 3)
    cost = (
        input_tokens * input_cost_per_million
        + output_tokens * output_cost_per_million
    ) / 1_000_000
    return {
        "request_id": request.request_id,
        "configuration": request.configuration,
        "model_name": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 8),
        "latency_ms": elapsed_ms,
        "evidence_count": len(request.evidence_ids),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
