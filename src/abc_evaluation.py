"""Quantitative system-level evaluation for paired A/B/C experiments.

Claim-level reliability metrics require a blinded reference annotation. The module
keeps those values explicitly missing until an annotation exists; it never substitutes
token overlap or an LLM self-score and presents it as human-verified correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import mean
from typing import Any, Iterable, Sequence

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TRACE_FIELDS = {
    "request_id",
    "configuration",
    "model_name",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "total_estimated_cost_usd",
    "latency_ms",
    "end_to_end_latency_ms",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "validation_latency_ms",
    "evidence_count",
    "completed_at_utc",
    "status",
    "sample_id",
    "question",
    "cpu_time_ms",
    "memory_rss_delta_bytes",
    "prompt_sha256",
    "engine_name",
    "generation_parameters",
    "experiment_id",
    "study_id",
    "question_id",
    "repeat",
    "run_order",
    "retrieval_query_builder_version",
    "prompt_contract_version",
    "requires_human_review",
    "audit_event_count",
}


METRIC_CATALOG: list[dict[str, str]] = [
    {
        "key": "factual_grounding",
        "category": "Explainability",
        "definition": "Supported assessed material claims divided by assessed material claims.",
        "method": "Blinded claim-level annotation against frozen evidence.",
    },
    {
        "key": "hallucination_rate",
        "category": "Explainability",
        "definition": "Unsupported assessed material claims divided by assessed material claims.",
        "method": "Blinded claim-level annotation against frozen evidence.",
    },
    {
        "key": "citation_correctness",
        "category": "Explainability",
        "definition": "Citations that entail the associated claim divided by citations assessed.",
        "method": "Blinded claim-to-source annotation; citation existence alone is insufficient.",
    },
    {
        "key": "policy_grounding_completeness",
        "category": "Explainability",
        "definition": "Policy-dependent claims supported by retrieved context divided by policy-dependent claims.",
        "method": "Blinded claim-level annotation.",
    },
    {
        "key": "claim_assessment_coverage",
        "category": "Explainability",
        "definition": "Assessable material claims divided by all identified material claims.",
        "method": "Blinded atomic-claim annotation; report beside hallucination rate.",
    },
    {
        "key": "explanation_fidelity",
        "category": "Explainability",
        "definition": "F1 overlap between model drivers stated in the response and the frozen local-attribution top features.",
        "method": "Automatic response-to-TreeSHAP evidence comparison; semantic interpretation still requires review.",
    },
    {
        "key": "response_json_valid",
        "category": "Structural governance",
        "definition": "Whether the generated candidate response was parsed as a JSON object.",
        "method": "Automatic parser outcome; parseability does not establish correctness.",
    },
    {
        "key": "claim_citation_coverage",
        "category": "Structural governance",
        "definition": "Parsed claims containing at least one citation divided by parsed claims.",
        "method": "Automatic response-structure inspection; citation presence does not establish entailment.",
    },
    {
        "key": "citation_id_validity",
        "category": "Structural governance",
        "definition": "Cited evidence IDs present in the supplied evidence allowlist divided by cited IDs.",
        "method": "Automatic provenance membership check; a valid ID does not establish semantic support.",
    },
    {
        "key": "policy_citation_coverage",
        "category": "Structural governance",
        "definition": "Parsed claims citing at least one retrieved policy evidence ID divided by parsed claims.",
        "method": "Automatic retrieved-source linkage check; relevance and correctness are future human-validation outcomes.",
    },
    {
        "key": "guardrail_violation_rate",
        "category": "Governance",
        "definition": "Configuration C runs with at least one failed enforced check divided by C runs.",
        "method": "Automatic Configuration C validator telemetry; check-level failure rate is supplementary.",
    },
    {
        "key": "audit_trace_completeness",
        "category": "Governance",
        "definition": (
            "Mean completeness of required trace fields and reconstructable required "
            "lifecycle audit-event types."
        ),
        "method": (
            "Automatic inspection requiring trace values plus an explicit decision and "
            "evidential basis for each applicable lifecycle event type."
        ),
    },
    {
        "key": "explanation_consistency",
        "category": "Governance",
        "definition": "Mean pairwise TF-IDF cosine similarity over repeated answers to an identical case and prompt.",
        "method": "Automatic repeated-run analysis; semantic review remains recommended.",
    },
    {
        "key": "processing_latency_ms",
        "category": "Operational",
        "definition": "End-to-end service latency in milliseconds per alert.",
        "method": "Monotonic timer from request orchestration through validation; report median and p95. Generation and retrieval are also recorded separately.",
    },
    {
        "key": "cost_per_alert_usd",
        "category": "Operational",
        "definition": "Estimated generation cost plus system-triggered review-referral cost per alert.",
        "method": "Recorded tokens and review disposition multiplied by frozen declared assumptions.",
    },
    {
        "key": "resource_utilization",
        "category": "Operational",
        "definition": "Process CPU time and resident-memory change during a run.",
        "method": "Process telemetry; use OS/GPU instrumentation for final experiments.",
    },
    {
        "key": "manual_review_referral_rate",
        "category": "Operational",
        "definition": "Runs programmatically referred for human review divided by runs.",
        "method": "Automatic output-disposition telemetry; this does not measure completed analyst reviews.",
    },
]


@dataclass(frozen=True)
class ClaimAnnotation:
    """Independent assessment counts for one frozen response."""

    material_claims: int
    unsupported_claims: int
    citations_assessed: int
    correct_citations: int
    policy_dependent_claims: int
    grounded_policy_claims: int
    claims_total: int | None = None
    not_assessable_claims: int = 0

    def __post_init__(self) -> None:
        pairs = (
            (self.unsupported_claims, self.material_claims),
            (self.correct_citations, self.citations_assessed),
            (self.grounded_policy_claims, self.policy_dependent_claims),
        )
        if any(numerator < 0 or denominator < 0 or numerator > denominator for numerator, denominator in pairs):
            raise ValueError("Annotation counts must be non-negative and numerator <= denominator.")
        if self.claims_total is not None and (
            self.claims_total < self.material_claims
            or self.not_assessable_claims < 0
            or self.material_claims + self.not_assessable_claims != self.claims_total
        ):
            raise ValueError(
                "claims_total must equal assessed material claims plus not-assessable claims."
            )


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def _is_recorded(value: Any) -> bool:
    """Return whether a trace value is recorded rather than a blank placeholder."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _trace_field_completeness(trace: dict[str, Any]) -> tuple[float, int]:
    applicable = set(TRACE_FIELDS)
    if str(trace.get("model_name", "")).startswith("not-called:"):
        applicable.difference_update({
            "prompt_sha256", "generation_latency_ms", "validation_latency_ms"
        })
    present = sum(field in trace and _is_recorded(trace[field]) for field in applicable)
    return float(_ratio(present, len(applicable)) or 0.0), len(applicable)


def _non_empty_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _valid_audit_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _audit_event_reconstructable(event: dict[str, Any], run: dict[str, Any]) -> bool:
    """Require both a recorded decision and its event-specific evidential basis.

    Evidence IDs are required when the decision is based on supplied model/policy
    evidence. Explicit findings, counts or reasons are the basis for stages where an
    empty outcome (for example, no input-safety finding) is itself the decision.
    """
    required_fields = {
        "event_id", "event_type", "status", "occurred_at_utc", "evidence_ids", "details"
    }
    if not required_fields.issubset(event):
        return False
    if not all(_is_recorded(event[field]) for field in ("event_id", "event_type", "status")):
        return False
    if not _valid_audit_timestamp(event["occurred_at_utc"]):
        return False
    evidence_ids = event["evidence_ids"]
    details = event["details"]
    if not isinstance(evidence_ids, list) or not isinstance(details, dict):
        return False
    if any(not isinstance(item, str) or not item.strip() for item in evidence_ids):
        return False

    event_type = str(event["event_type"])
    status = str(event["status"])
    if event_type == "model_evidence_loaded":
        return (
            status == "completed"
            and _non_empty_strings(evidence_ids)
            and _is_recorded(details.get("sample_id"))
        )
    if event_type == "input_safety_assessed":
        findings = details.get("findings")
        return status in {"completed", "blocked"} and isinstance(findings, list)
    if event_type == "policy_retrieval":
        result_count = details.get("result_count")
        return (
            status == "completed"
            and _is_recorded(run.get("retrieval_query"))
            and isinstance(details.get("top_k"), int)
            and details["top_k"] > 0
            and isinstance(result_count, int)
            and result_count > 0
            and result_count == len(evidence_ids)
            and isinstance(details.get("latency_ms"), (int, float))
            and details["latency_ms"] >= 0
            and _non_empty_strings(evidence_ids)
        )
    if event_type == "policy_retrieval_skipped":
        return (
            status == "completed"
            and run.get("configuration") == "A"
            and details.get("top_k") is None
            and details.get("result_count") == 0
            and isinstance(details.get("latency_ms"), (int, float))
            and details["latency_ms"] >= 0
            and evidence_ids == []
        )
    if event_type == "generation_completed":
        return (
            status == "completed"
            and _non_empty_strings(evidence_ids)
            and _is_recorded(details.get("model_name"))
            and all(
                isinstance(details.get(field), int) and details[field] >= 0
                for field in ("input_tokens", "output_tokens")
            )
            and isinstance(run.get("candidate_response_text"), str)
            and bool(run["candidate_response_text"].strip())
        )
    if event_type == "guardrail_check":
        check_name = details.get("check_name")
        if status not in {"passed", "failed"} or not _is_recorded(check_name):
            return False
        if check_name == "input_safety":
            return isinstance(details.get("violations"), list)
        return _non_empty_strings(evidence_ids)
    if event_type == "claim_ledger":
        claim_count = details.get("claim_count")
        if not isinstance(claim_count, int) or claim_count < 0:
            return False
        if status == "unavailable":
            return claim_count == 0 and _is_recorded(details.get("reason"))
        return status == "completed" and claim_count > 0 and _non_empty_strings(evidence_ids)
    if event_type == "generation_suppressed":
        return (
            status == "completed"
            and _is_recorded(details.get("reason"))
            and isinstance(details.get("requires_human_review"), bool)
            and _non_empty_strings(evidence_ids)
        )
    if event_type == "output_disposition":
        reasons = details.get("reasons")
        if not isinstance(reasons, list):
            return False
        if status == "review_required":
            return bool(reasons) and all(_is_recorded(reason) for reason in reasons)
        if status == "released_for_analyst_review":
            return reasons == [] and _non_empty_strings(evidence_ids)
        if status == "no_alert_no_generation":
            return reasons == []
        return False
    return False


def _required_audit_event_types(run: dict[str, Any]) -> set[str]:
    required_types = {"model_evidence_loaded", "input_safety_assessed", "output_disposition"}
    model_name = str((run.get("trace") or {}).get("model_name", ""))
    status = run.get("status")
    if model_name == "not-called:input-safety-block":
        required_types.add("guardrail_check")
    else:
        required_types.add(
            "policy_retrieval_skipped" if run.get("configuration") == "A"
            else "policy_retrieval"
        )
        if status == "suppressed":
            required_types.update({"guardrail_check", "generation_suppressed"})
        else:
            required_types.update({"generation_completed", "claim_ledger"})
            if run.get("configuration") == "C":
                required_types.add("guardrail_check")
    return required_types


def _audit_event_completeness(
    run: dict[str, Any],
) -> tuple[float, int, list[str]]:
    events = run.get("audit_events") or []
    reconstructable_events = [
        event for event in events
        if isinstance(event, dict) and _audit_event_reconstructable(event, run)
    ]
    event_types = {str(event["event_type"]) for event in reconstructable_events}
    required_types = _required_audit_event_types(run)
    missing_types = sorted(required_types - event_types)
    return (
        float(_ratio(len(required_types & event_types), len(required_types)) or 0.0),
        len(required_types),
        missing_types,
    )


def _explanation_fidelity(run: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    parsed = run.get("parsed_answer")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("model_drivers"), list):
        return None, None, None
    expected: set[str] = set()
    for dataset in (run.get("model_evidence") or {}).get("datasets", []):
        for alert in dataset.get("top_alerts", []):
            for item in (alert.get("local_attribution") or {}).get("top_features", []):
                driver = item.get("feature") or item.get("feature_group")
                if driver:
                    expected.add(str(driver))
    if not expected:
        return None, None, None
    reported = {str(value) for value in parsed["model_drivers"] if isinstance(value, str)}
    overlap = len(expected & reported)
    precision = _ratio(overlap, len(reported)) if reported else 0.0
    recall = _ratio(overlap, len(expected))
    f1 = (
        round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return precision, recall, f1


def _generation_attempted(run: dict[str, Any]) -> bool:
    """True when the model was actually asked to generate a candidate response.

    Runs suppressed by the fraud-model score gate never reach the model: they
    carry no candidate text and zero output tokens.  Scoring those as 0.0 on
    candidate text-quality metrics would record "was never asked to write" as
    "wrote badly", so candidate metrics are reported as None instead and the
    run is excluded from candidate-metric comparisons.  Released metrics are
    unaffected: a suppressed run genuinely released nothing.
    """
    text = run.get("candidate_response_text")
    if isinstance(text, str) and text.strip():
        return True
    trace = run.get("trace") or {}
    return bool(trace.get("output_tokens"))


def _raw_candidate_object(run: dict[str, Any]) -> dict[str, Any] | None:
    text = run.get("candidate_response_text")
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _with_parsed_answer(
    run: dict[str, Any], parsed_answer: dict[str, Any] | None
) -> dict[str, Any]:
    scoped = dict(run)
    scoped["parsed_answer"] = parsed_answer
    return scoped


def _condition_id(run: dict[str, Any]) -> str:
    trace = run.get("trace") or {}
    return str(run.get("condition_id") or trace.get("condition_id") or run.get("configuration"))


def _citation_structure(run: dict[str, Any]) -> tuple[float, float, float]:
    parsed = run.get("parsed_answer")
    claims = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(claims, list) or not claims:
        return 0.0, 0.0, 0.0
    normalized_claims = [claim for claim in claims if isinstance(claim, dict)]
    if not normalized_claims:
        return 0.0, 0.0, 0.0
    citations_by_claim = [
        [citation for citation in claim.get("citations", []) if isinstance(citation, str)]
        for claim in normalized_claims
    ]
    citations = [citation for values in citations_by_claim for citation in values]
    allowed_ids = {str(value) for value in run.get("evidence_ids", [])}
    policy_ids = {
        str(item["evidence_id"])
        for item in run.get("retrieved_policy", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    citation_coverage = float(_ratio(
        sum(bool(values) for values in citations_by_claim), len(normalized_claims)
    ) or 0.0)
    citation_id_validity = float(_ratio(
        sum(citation in allowed_ids for citation in citations), len(citations)
    ) or 0.0)
    policy_citation_coverage = float(_ratio(
        sum(any(citation in policy_ids for citation in values) for values in citations_by_claim),
        len(normalized_claims),
    ) or 0.0)
    return citation_coverage, citation_id_validity, policy_citation_coverage


def evaluate_run(
    run: dict[str, Any], annotation: ClaimAnnotation | None = None
) -> dict[str, Any]:
    """Calculate automatic metrics and attach independently annotated metrics."""
    trace = run.get("trace") or {}
    checks = run.get("guardrail_checks") or {}
    violations = run.get("guardrail_violations") or []
    failed_checks = sum(value is False for value in checks.values())
    trace_completeness, applicable_trace_field_count = _trace_field_completeness(trace)
    (
        event_completeness,
        required_event_type_count,
        unreconstructable_event_types,
    ) = _audit_event_completeness(run)
    fidelity_precision, fidelity_recall, fidelity_f1 = _explanation_fidelity(run)
    citation_coverage, citation_id_validity, policy_citation_coverage = _citation_structure(run)
    generation_attempted = _generation_attempted(run)
    candidate_object = _raw_candidate_object(run)
    candidate_run = _with_parsed_answer(run, candidate_object)
    (
        candidate_fidelity_precision,
        candidate_fidelity_recall,
        candidate_fidelity_f1,
    ) = _explanation_fidelity(candidate_run)
    (
        candidate_citation_coverage,
        candidate_citation_id_validity,
        candidate_policy_citation_coverage,
    ) = _citation_structure(candidate_run)
    is_configuration_c = (
        run.get("configuration") == "C"
        or run.get("guardrail_profile") == "full"
        or trace.get("guardrail_profile") == "full"
    )
    result: dict[str, Any] = {
        "request_id": run.get("request_id"),
        "configuration": _condition_id(run),
        "base_configuration": run.get("configuration"),
        "retrieval_strategy_id": (
            run.get("retrieval_strategy_id") or trace.get("retrieval_strategy_id")
        ),
        "guardrail_profile": run.get("guardrail_profile") or trace.get("guardrail_profile"),
        "sample_id": run.get("sample_id"),
        "dataset_id": next((
            dataset.get("dataset_id")
            for dataset in (run.get("model_evidence") or {}).get("datasets", [])
        ), None),
        "evaluation_cohort": (
            (run.get("model_evidence") or {})
            .get("active_evaluation_sample", {})
            .get("evaluation_cohort", "unclassified")
        ),
        "experiment_id": trace.get("experiment_id"),
        "study_id": trace.get("study_id") or trace.get("experiment_id"),
        "question_id": trace.get("question_id") or run.get("question"),
        "repeat": trace.get("repeat"),
        "engine_name": trace.get("engine_name") or trace.get("model_name"),
        "status": run.get("status"),
        "hallucination_rate": None,
        "factual_grounding": None,
        "citation_correctness": None,
        "policy_grounding_completeness": None,
        "claim_assessment_coverage": None,
        "explanation_fidelity": fidelity_f1,
        "explanation_fidelity_precision": fidelity_precision,
        "explanation_fidelity_recall": fidelity_recall,
        "response_json_valid": float(isinstance(run.get("parsed_answer"), dict)),
        "candidate_response_json_valid": (
            float(candidate_object is not None) if generation_attempted else None
        ),
        "released_response_json_valid": float(isinstance(run.get("parsed_answer"), dict)),
        "candidate_explanation_fidelity": (candidate_fidelity_f1 if generation_attempted else None),
        "candidate_explanation_fidelity_precision": (candidate_fidelity_precision if generation_attempted else None),
        "candidate_explanation_fidelity_recall": (candidate_fidelity_recall if generation_attempted else None),
        "candidate_claim_citation_coverage": (candidate_citation_coverage if generation_attempted else None),
        "candidate_citation_id_validity": (candidate_citation_id_validity if generation_attempted else None),
        "candidate_policy_citation_coverage": (candidate_policy_citation_coverage if generation_attempted else None),
        "released_claim_citation_coverage": citation_coverage,
        "released_citation_id_validity": citation_id_validity,
        "released_policy_citation_coverage": policy_citation_coverage,
        "claim_citation_coverage": citation_coverage,
        "citation_id_validity": citation_id_validity,
        "policy_citation_coverage": policy_citation_coverage,
        "guardrail_violation_rate": (
            float(failed_checks > 0) if is_configuration_c and checks else None
        ),
        "guardrail_check_failure_rate": _ratio(failed_checks, len(checks)),
        "guardrail_violation_count": len(violations),
        "audit_trace_completeness": round(
            (trace_completeness + event_completeness) / 2.0, 6
        ),
        "trace_field_completeness": trace_completeness,
        "audit_event_completeness": event_completeness,
        "audit_trace_applicable_field_count": applicable_trace_field_count,
        "audit_required_event_type_count": required_event_type_count,
        "audit_reconstructable_event_type_count": (
            required_event_type_count - len(unreconstructable_event_types)
        ),
        "audit_unreconstructable_event_types": unreconstructable_event_types,
        "schema_valid": (
            bool(checks.get("valid_json_object") and checks.get("schema_contract"))
            if is_configuration_c else None
        ),
        "processing_latency_ms": trace.get("end_to_end_latency_ms", trace.get("latency_ms")),
        "generation_latency_ms": trace.get("generation_latency_ms", trace.get("latency_ms")),
        "validation_latency_ms": trace.get("validation_latency_ms"),
        "retrieval_latency_ms": trace.get("retrieval_latency_ms"),
        "cost_per_alert_usd": trace.get(
            "total_estimated_cost_usd", trace.get("estimated_cost_usd")
        ),
        "generation_cost_usd": trace.get("estimated_cost_usd"),
        "estimated_human_review_cost_usd": trace.get("estimated_human_review_cost_usd"),
        "estimated_compute_cost_usd": trace.get("estimated_compute_cost_usd"),
        "manual_review_referral": float(bool(run.get("requires_human_review"))),
        "cpu_time_ms": trace.get("cpu_time_ms"),
        "memory_rss_delta_bytes": trace.get("memory_rss_delta_bytes"),
        "annotation_status": "deferred_to_future_work_by_dec020",
    }
    if annotation is not None:
        result.update({
            "hallucination_rate": _ratio(
                annotation.unsupported_claims, annotation.material_claims
            ),
            "factual_grounding": _ratio(
                annotation.material_claims - annotation.unsupported_claims,
                annotation.material_claims,
            ),
            "citation_correctness": _ratio(
                annotation.correct_citations, annotation.citations_assessed
            ),
            "policy_grounding_completeness": _ratio(
                annotation.grounded_policy_claims, annotation.policy_dependent_claims
            ),
            "claim_assessment_coverage": _ratio(
                annotation.material_claims,
                annotation.claims_total
                if annotation.claims_total is not None
                else annotation.material_claims,
            ),
            "annotation_status": "complete",
            "annotation_counts": asdict(annotation),
        })
    return result


def explanation_consistency(answers: Sequence[str]) -> float | None:
    """Return mean pairwise lexical-semantic similarity for repeated outputs."""
    normalized = [answer.strip() for answer in answers if answer and answer.strip()]
    if len(normalized) < 2:
        return None
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(normalized)
    except ValueError:
        return 1.0 if len(set(normalized)) == 1 else 0.0
    similarities = cosine_similarity(matrix)
    upper_triangle = similarities[np.triu_indices(len(normalized), k=1)]
    return round(float(upper_triangle.mean()), 6)


def repeated_explanation_consistency(
    runs: Sequence[dict[str, Any]],
) -> tuple[float | None, int]:
    """Average consistency only within identical case/question repeated-run cells."""
    cells: dict[tuple[Any, ...], list[str]] = {}
    for run in runs:
        trace = run.get("trace") or {}
        key = (
            trace.get("experiment_id"),
            run.get("sample_id"),
            trace.get("question_id") or run.get("question"),
            _condition_id(run),
        )
        cells.setdefault(key, []).append(str(run.get("answer", "")))
    scores = [
        score
        for answers in cells.values()
        if (score := explanation_consistency(answers)) is not None
    ]
    return (round(mean(scores), 6), len(scores)) if scores else (None, 0)


def aggregate_runs(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate comparable telemetry without mixing LLM and demo engines."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        trace = run.get("trace") or {}
        engine = str(trace.get("engine_name") or trace.get("model_name") or "unknown")
        experiment = str(trace.get("experiment_id") or "unregistered")
        key = (_condition_id(run), engine, experiment)
        grouped.setdefault(key, []).append(run)

    summary: list[dict[str, Any]] = []
    for (configuration, engine, experiment_id), items in sorted(grouped.items()):
        evaluated = [evaluate_run(item) for item in items]
        consistency, consistency_cell_count = repeated_explanation_consistency(items)
        latencies = [
            float(item["processing_latency_ms"])
            for item in evaluated
            if item["processing_latency_ms"] is not None
        ]
        costs = [
            float(item["cost_per_alert_usd"])
            for item in evaluated
            if item["cost_per_alert_usd"] is not None
        ]
        fidelities = [
            float(item["explanation_fidelity"])
            for item in evaluated
            if item["explanation_fidelity"] is not None
        ]
        guardrail_indicators = [
            float(item["guardrail_violation_rate"])
            for item in evaluated
            if item["guardrail_violation_rate"] is not None
        ]
        summary.append({
            "configuration": configuration,
            "engine_name": engine,
            "experiment_id": experiment_id,
            "run_count": len(items),
            "completed_count": sum(item.get("status") == "completed" for item in items),
            "blocked_count": sum(item.get("status") == "blocked" for item in items),
            "suppressed_count": sum(item.get("status") == "suppressed" for item in items),
            "latency_median_ms": round(float(np.median(latencies)), 3) if latencies else None,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else None,
            "mean_cost_per_alert_usd": round(mean(costs), 8) if costs else None,
            "mean_explanation_fidelity": round(mean(fidelities), 6) if fidelities else None,
            "guardrail_violation_rate": (
                round(mean(guardrail_indicators), 6) if guardrail_indicators else None
            ),
            "manual_review_referral_rate": round(mean(
                item["manual_review_referral"] for item in evaluated
            ), 6),
            "mean_audit_trace_completeness": round(mean(
                item["audit_trace_completeness"] for item in evaluated
            ), 6),
            "explanation_consistency": consistency,
            "consistency_repeated_cell_count": consistency_cell_count,
            "claim_metrics_status": "deferred_to_future_work_by_dec020",
        })
    return summary


def validate_paired_design(
    runs: Iterable[dict[str, Any]],
    *,
    expected_conditions: Sequence[str] = ("A", "B", "C"),
) -> dict[str, Any]:
    """Check registered-treatment completeness at case/question/repeat level."""
    expected = sorted(str(value) for value in expected_conditions)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_conditions must be a non-empty unique sequence.")
    cells: dict[tuple[Any, ...], list[str]] = {}
    for run in runs:
        trace = run.get("trace") or {}
        key = (
            trace.get("experiment_id"),
            run.get("sample_id"),
            trace.get("question_id") or run.get("question"),
            trace.get("repeat"),
            trace.get("engine_name") or trace.get("model_name"),
        )
        cells.setdefault(key, []).append(_condition_id(run))
    incomplete = [
        {
            "experiment_id": key[0],
            "sample_id": key[1],
            "question_id": key[2],
            "repeat": key[3],
            "engine_name": key[4],
            "present": sorted(configurations),
        }
        for key, configurations in cells.items()
        if sorted(configurations) != expected
    ]
    return {
        "paired": not incomplete and bool(cells),
        "pair_count": len(cells),
        "expected_conditions": expected,
        "incomplete_pairs": incomplete,
    }
