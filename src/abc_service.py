"""SOLID orchestration for controlled Configuration A/B/C investigations.

The service applies dependency inversion through small repository/generator protocols,
keeps each experimental condition in one strategy, and gives every condition the same
sample transaction, question, model evidence, timing, and resource instrumentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import copy
import hashlib
import json
import os
import tempfile
import time
import uuid

import pandas as pd

from src.evidence import load_unified_evidence
from src.llm_backend import (
    Configuration,
    PolicyEvidence,
    PreparedRequest,
    build_observation,
    evaluate_configuration_c_response,
    prepare_request,
)
from src.llm_provider import GenerationResult
from src.retrieval import load_policy_directory, retrieve_policy_evidence
from src.safety import assess_question, blocking_findings, findings_payload


RETRIEVAL_QUERY_BUILDER_VERSION = "1.0"
PROMPT_CONTRACT_VERSION = "3.1"


class EvidenceRepository(Protocol):
    def load(self) -> dict[str, Any]: ...


class PolicyRepository(Protocol):
    def search(self, question: str, *, top_k: int) -> list[PolicyEvidence]: ...


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 250) -> GenerationResult: ...


class RunRepository(Protocol):
    def save(self, payload: dict[str, Any], request_id: str) -> Path: ...


class SampleRepository(Protocol):
    def list(self, *, include_evaluation_labels: bool = False) -> list[dict[str, Any]]: ...

    def get(self, sample_id: str) -> dict[str, Any]: ...


class AttributionRepository(Protocol):
    def get(self, sample_id: str) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class InvestigationCommand:
    configuration: Configuration | str
    question: str
    sample_id: str | None = None
    model_score_gate: float = 0.70
    policy_top_k: int = 4
    max_new_tokens: int = 180
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    analyst_hourly_cost_usd: float = 0.0
    estimated_review_minutes: float = 0.0
    compute_hourly_cost_usd: float = 0.0
    experiment_id: str | None = None
    study_id: str | None = None
    question_id: str | None = None
    repeat: int | None = None
    run_order: int | None = None
    condition_id: str | None = None
    retrieval_strategy_id: str | None = None
    guardrail_profile: str | None = None


@dataclass
class InvestigationResult:
    request_id: str
    configuration: str
    status: str
    question: str
    retrieval_query: str | None
    sample_id: str | None
    answer: str
    candidate_response_text: str | None
    parsed_answer: dict[str, Any] | None
    model_evidence: dict[str, Any] | None
    evidence_ids: list[str]
    retrieved_policy: list[dict[str, Any]]
    guardrail_checks: dict[str, bool]
    guardrail_violations: list[str]
    requires_human_review: bool
    human_review_reasons: list[str]
    audit_events: list[dict[str, Any]]
    trace: dict[str, Any]
    condition_id: str | None = None
    retrieval_strategy_id: str | None = None
    guardrail_profile: str | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_event(
    event_type: str,
    status: str,
    *,
    evidence_ids: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one timestamped, reconstructable processing event."""
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "status": status,
        "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_ids": list(evidence_ids),
        "details": dict(details or {}),
    }


class FileEvidenceRepository:
    """Load the frozen ULB/Sparkov evidence bundle."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    def load(self) -> dict[str, Any]:
        return load_unified_evidence(self._project_root)


class LocalPolicyRepository:
    """Search a frozen, provenance-registered local policy corpus."""

    def __init__(self, source_directory: Path) -> None:
        self._source_directory = source_directory

    def search(self, question: str, *, top_k: int) -> list[PolicyEvidence]:
        documents = load_policy_directory(self._source_directory)
        if not documents:
            raise RuntimeError(
                "The approved policy corpus is empty; Configurations B and C cannot run."
            )
        return retrieve_policy_evidence(question, documents, top_k=top_k)


class CsvSampleRepository:
    """Read frozen UI/evaluation transactions and prevent label leakage."""

    LABEL_COLUMN = "evaluation_only_actual_class"

    def __init__(self, path: Path) -> None:
        self._path = path

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Transaction sample file not found: {path}")
        frame = pd.read_csv(path)
        return json.loads(frame.to_json(orient="records"))

    def list(self, *, include_evaluation_labels: bool = False) -> list[dict[str, Any]]:
        records = self._records(self._path)
        if not include_evaluation_labels:
            for record in records:
                record.pop(self.LABEL_COLUMN, None)
        return records

    def get(self, sample_id: str) -> dict[str, Any]:
        for record in self._records(self._path):
            if record.get("sample_id") == sample_id:
                return record
        raise KeyError(f"Unknown sample_id: {sample_id}")


class CompositeSampleRepository:
    """Expose dataset-specific sample files through one repository contract."""

    def __init__(self, repositories: Sequence[SampleRepository]) -> None:
        self._repositories = tuple(repositories)

    def list(self, *, include_evaluation_labels: bool = False) -> list[dict[str, Any]]:
        return [
            record
            for repository in self._repositories
            for record in repository.list(
                include_evaluation_labels=include_evaluation_labels
            )
        ]

    def get(self, sample_id: str) -> dict[str, Any]:
        for repository in self._repositories:
            try:
                return repository.get(sample_id)
            except KeyError:
                continue
        raise KeyError(f"Unknown sample_id: {sample_id}")


class JsonRunRepository:
    """Persist immutable per-run traces for audit and later evaluation."""

    def __init__(self, output_directory: Path) -> None:
        self._output_directory = output_directory

    def save(self, payload: dict[str, Any], request_id: str) -> Path:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        path = self._output_directory / f"{request_id}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite run record: {path}")
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._output_directory,
            prefix=f".{request_id}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
        return path


class FileAttributionRepository:
    """Load generated, model-specific local attribution evidence for frozen samples."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def get(self, sample_id: str) -> dict[str, Any] | None:
        if not self._path.exists():
            return None
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        for record in payload.get("samples", []):
            if record.get("sample_id") == sample_id:
                return record
        return None


class CompositeAttributionRepository:
    """Search independent dataset attribution stores without merging their schemas."""

    def __init__(self, repositories: Sequence[AttributionRepository]) -> None:
        self._repositories = tuple(repositories)

    def get(self, sample_id: str) -> dict[str, Any] | None:
        for repository in self._repositories:
            if record := repository.get(sample_id):
                return record
        return None


def _sample_evidence_bundle(
    bundle: dict[str, Any],
    sample: dict[str, Any] | None,
    attribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace alert context with one frozen, dataset-specific evaluation transaction."""
    scoped = copy.deepcopy(bundle)
    if sample is None:
        return scoped

    probability = float(sample["expected_fraud_probability"])
    threshold = float(sample["decision_threshold"])
    dataset_id = str(sample.get("dataset_id") or "Sparkov")
    excluded = {
        "sample_id",
        "dataset_id",
        "test_case_type",
        "evaluation_cohort",
        "source_test_row_index",
        "expected_fraud_probability",
        "decision_threshold",
        "expected_alert",
        CsvSampleRepository.LABEL_COLUMN,
    }
    record_id = str(sample["source_test_row_index"])
    dataset_metadata = next(
        (
            dataset
            for dataset in scoped["datasets"]
            if str(dataset.get("dataset_id", "")).lower() == dataset_id.lower()
        ),
        None,
    )
    if dataset_metadata is None:
        raise KeyError(f"No model evidence exists for sample dataset: {dataset_id}")
    alert = {
        "evidence_id": f"model-output:{dataset_id.lower()}:sample:{sample['sample_id']}",
        "evidence_type": "fraud_model_output",
        "dataset_id": dataset_id,
        "record_id": record_id,
        "event_time": sample.get("transaction_time", sample.get("Time")),
        "model_name": dataset_metadata.get("model_name", "Random forest"),
        "fraud_probability": probability,
        "decision_threshold": threshold,
        "fraud_alert": int(sample["expected_alert"]),
        "features": {key: value for key, value in sample.items() if key not in excluded},
    }
    if attribution is not None:
        alert["local_attribution"] = attribution
    for dataset in scoped["datasets"]:
        dataset["top_alerts"] = (
            [alert] if dataset["dataset_id"].lower() == dataset_id.lower() else []
        )
    scoped["datasets"] = [
        dataset
        for dataset in scoped["datasets"]
        if dataset["dataset_id"].lower() == dataset_id.lower()
    ]
    scoped["cross_dataset_test_comparison"] = []
    scoped["historical_profiles"] = None
    scoped["interpretation_notes"] = {
        dataset_id: scoped["interpretation_notes"][dataset_id]
    }
    scoped["active_evaluation_sample"] = {
        "sample_id": sample["sample_id"],
        "dataset_id": dataset_id,
        "test_case_type": sample["test_case_type"],
        "evaluation_cohort": sample.get("evaluation_cohort", "unclassified"),
    }
    return scoped


def build_transaction_aware_retrieval_query(
    question: str, bundle: dict[str, Any]
) -> str:
    """Build a privacy-minimised query from the question and supplied model evidence.

    Exact amounts, coordinates, anonymized ULB components, and direct identifiers are
    excluded. The query adds only coarse score/alert state, interpretable Sparkov
    feature names, merchant category, and validated attribution feature names.
    """
    terms = [question.strip(), "credit card fraud alert human investigation"]
    safe_feature_names = {
        "amt": "transaction amount",
        "distance_km": "merchant distance",
        "hour": "transaction time",
        "category": "merchant category",
        "is_weekend": "weekend transaction",
        "amount": "transaction amount",
        "transaction_time": "transaction time",
        "merchant_category": "merchant category",
        "cardholder_context": "cardholder context",
        "merchant_location": "merchant location and distance",
    }
    for dataset in bundle.get("datasets", []):
        dataset_id = str(dataset.get("dataset_id", "unknown"))
        terms.append(f"dataset {dataset_id}")
        for alert in dataset.get("top_alerts", []):
            probability = float(alert.get("fraud_probability", 0.0))
            terms.append(
                "high model risk score" if probability >= 0.7
                else "moderate model risk score" if probability >= 0.4
                else "low model risk score"
            )
            features = alert.get("features", {})
            if dataset_id.lower() == "sparkov":
                for key, label in safe_feature_names.items():
                    if key in features:
                        terms.append(label)
                category = features.get("category")
                if isinstance(category, str) and category.replace("_", "").isalnum():
                    terms.append(f"merchant category {category.replace('_', ' ')}")
            attribution = alert.get("local_attribution") or {}
            for item in attribution.get("top_features", [])[:5]:
                feature = str(item.get("feature_group", item.get("feature", "")))
                if feature in safe_feature_names:
                    terms.append(f"model driver {safe_feature_names[feature]}")
    return " | ".join(dict.fromkeys(term for term in terms if term))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _policy_corpus_hash(policy: Sequence[PolicyEvidence]) -> str | None:
    hashes = sorted(document.chunk_hash or document.evidence_id for document in policy)
    return _hash_text("|".join(hashes)) if hashes else None


@dataclass(frozen=True)
class StrategyOutcome:
    answer: str
    parsed_answer: dict[str, Any] | None
    checks: dict[str, bool]
    violations: list[str]


class ConfigurationStrategy(Protocol):
    configuration: Configuration
    requires_policy: bool

    def prepare(
        self,
        command: InvestigationCommand,
        bundle: dict[str, Any],
        policy: Sequence[PolicyEvidence],
    ) -> PreparedRequest: ...

    def finalize(self, text: str, request: PreparedRequest) -> StrategyOutcome: ...


class _NaturalLanguageStrategy:
    requires_policy = False

    def prepare(
        self,
        command: InvestigationCommand,
        bundle: dict[str, Any],
        policy: Sequence[PolicyEvidence],
    ) -> PreparedRequest:
        return prepare_request(
            configuration=self.configuration,
            question=command.question,
            unified_evidence=bundle,
            retrieved_documents=policy,
            model_score_gate=command.model_score_gate,
        )

    def finalize(self, text: str, request: PreparedRequest) -> StrategyOutcome:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return StrategyOutcome(
            text,
            parsed if isinstance(parsed, dict) else None,
            {},
            [],
        )


class BaselineStrategy(_NaturalLanguageStrategy):
    configuration = Configuration.A


class RetrievalStrategy(_NaturalLanguageStrategy):
    configuration = Configuration.B
    requires_policy = True


class GuardedStrategy(_NaturalLanguageStrategy):
    configuration = Configuration.C
    requires_policy = True

    def finalize(self, text: str, request: PreparedRequest) -> StrategyOutcome:
        report = evaluate_configuration_c_response(
            text,
            allowed_evidence_ids=request.evidence_ids,
            allowed_model_drivers=request.allowed_model_drivers,
        )
        if report.passed and report.response is not None:
            return StrategyOutcome(
                answer=json.dumps(report.response, indent=2),
                parsed_answer=report.response,
                checks=report.checks,
                violations=[],
            )
        return StrategyOutcome(
            answer=(
                "Response blocked by Configuration C structural controls: "
                + "; ".join(report.violations)
            ),
            parsed_answer=None,
            checks=report.checks,
            violations=list(report.violations),
        )


class DeterministicDemoGenerator:
    """Transparent UI smoke-test engine; not an LLM experimental result."""

    model_name = "deterministic-evidence-demo"
    generation_parameters = {"deterministic": True}

    def generate(self, prompt: str, *, max_new_tokens: int = 180) -> GenerationResult:
        del max_new_tokens
        payload = json.loads(prompt)
        model_records = [
            alert
            for dataset in payload["model_evidence"]["datasets"]
            for alert in dataset.get("top_alerts", [])
        ]
        policy = payload.get("retrieved_policy_evidence", [])
        record = model_records[0] if model_records else None
        probability = float(record["fraud_probability"]) if record else 0.0
        dataset_id = str(record.get("dataset_id", "model")) if record else "model"
        risk = "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"
        summary = (
            f"The {dataset_id} model score is {probability:.3f}. This is model evidence, "
            "not proof of fraud; a human investigator should review the supplied features."
        )
        if policy:
            summary += " Retrieved public governance/regulatory context is available."

        output_instruction = payload.get("output_instruction", "")
        if "Return only one JSON object" in output_instruction:
            model_ids = [row["evidence_id"] for row in model_records]
            policy_ids = [row["evidence_id"] for row in policy]
            citations = (model_ids[:1] + policy_ids[:1])
            drivers = [
                str(item.get("feature") or item.get("feature_group"))
                for row in model_records[:1]
                for item in (row.get("local_attribution") or {}).get("top_features", [])
                if item.get("feature") or item.get("feature_group")
            ][:5]
            text = json.dumps({
                "answer": summary,
                "risk_level": risk,
                "self_reported_confidence": None,
                "citations": citations,
                "claims": [{
                    "statement": summary,
                    "citations": citations,
                }],
                "model_drivers": drivers,
                "recommended_actions": ["Route the alert to a human investigator."],
                "limitations": [
                    f"{dataset_id} dataset limitations apply.",
                    "Public sources are not institution-specific operating policy."
                ],
            })
        else:
            sources = " ".join(f"[{row['evidence_id']}]" for row in policy[:2])
            text = f"{summary} {sources}".strip()
        return GenerationResult(
            text=text,
            model_name=self.model_name,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(text.split())),
        )


class InvestigationService:
    """Orchestrate runs while depending only on abstractions supplied at construction."""

    def __init__(
        self,
        *,
        evidence_repository: EvidenceRepository,
        policy_repository: PolicyRepository,
        sample_repository: SampleRepository,
        run_repository: RunRepository,
        generator: TextGenerator,
        attribution_repository: AttributionRepository | None = None,
        strategies: Mapping[Configuration, ConfigurationStrategy] | None = None,
        policy_repositories: Mapping[str, PolicyRepository] | None = None,
    ) -> None:
        self._evidence_repository = evidence_repository
        self._policy_repository = policy_repository
        self._sample_repository = sample_repository
        self._run_repository = run_repository
        self._generator = generator
        self._attribution_repository = attribution_repository
        self._policy_repositories = dict(policy_repositories or {})
        self._strategies = strategies or {
            Configuration.A: BaselineStrategy(),
            Configuration.B: RetrievalStrategy(),
            Configuration.C: GuardedStrategy(),
        }

    def run(self, command: InvestigationCommand) -> InvestigationResult:
        end_to_end_started = time.monotonic()
        mode = Configuration(command.configuration)
        strategy = self._strategies[mode]
        policy_repository = self._policy_repositories.get(
            command.retrieval_strategy_id or "lexical_tfidf",
            self._policy_repository,
        )
        sample = self._sample_repository.get(command.sample_id) if command.sample_id else None
        attribution = (
            self._attribution_repository.get(command.sample_id)
            if self._attribution_repository is not None and command.sample_id
            else None
        )
        bundle = _sample_evidence_bundle(
            self._evidence_repository.load(), sample, attribution
        )
        audit_events = [_audit_event(
            "model_evidence_loaded",
            "completed",
            evidence_ids=[
                alert["evidence_id"]
                for dataset in bundle.get("datasets", [])
                for alert in dataset.get("top_alerts", [])
                if alert.get("evidence_id")
            ],
            details={"sample_id": command.sample_id},
        )]
        safety_findings = assess_question(command.question)
        blocked_findings = blocking_findings(safety_findings)
        audit_events.append(_audit_event(
            "input_safety_assessed",
            "blocked" if mode is Configuration.C and blocked_findings else "completed",
            details={"findings": findings_payload(safety_findings)},
        ))
        if mode is Configuration.C and blocked_findings:
            return self._blocked_input_result(
                command=command,
                findings=safety_findings,
                started_monotonic=end_to_end_started,
                audit_events=audit_events,
            )
        retrieval_query = (
            build_transaction_aware_retrieval_query(command.question, bundle)
            if strategy.requires_policy
            else None
        )
        retrieval_started = time.monotonic()
        policy = (
            policy_repository.search(
                retrieval_query or command.question, top_k=command.policy_top_k
            )
            if strategy.requires_policy
            else []
        )
        retrieval_latency_ms = round(
            (time.monotonic() - retrieval_started) * 1_000, 3
        ) if strategy.requires_policy else 0.0
        audit_events.append(_audit_event(
            "policy_retrieval" if strategy.requires_policy else "policy_retrieval_skipped",
            "completed",
            evidence_ids=[document.evidence_id for document in policy],
            details={
                "top_k": command.policy_top_k if strategy.requires_policy else None,
                "result_count": len(policy),
                "latency_ms": retrieval_latency_ms,
                "retrieval_strategy_id": (
                    command.retrieval_strategy_id if strategy.requires_policy else "none"
                ),
            },
        ))
        request = strategy.prepare(command, bundle, policy)
        policy_payload = [asdict(document) for document in policy]
        model_evidence_payload = (
            json.loads(request.prompt).get("model_evidence")
            if request.prompt else copy.deepcopy(bundle)
        )

        if not request.should_call_llm:
            active_model_alert = any(
                int(alert.get("fraud_alert", 0)) == 1
                for dataset in bundle.get("datasets", [])
                for alert in dataset.get("top_alerts", [])
            )
            review_reasons = (
                ["model_alert_below_explanation_release_gate"]
                if active_model_alert else []
            )
            audit_events.extend([
                _audit_event(
                    "guardrail_check",
                    "failed" if active_model_alert else "passed",
                    evidence_ids=request.evidence_ids,
                    details={"check_name": "model_score_generation_gate"},
                ),
                _audit_event(
                    "generation_suppressed",
                    "completed",
                    evidence_ids=request.evidence_ids,
                    details={
                        "reason": request.suppression_reason,
                        "requires_human_review": active_model_alert,
                    },
                ),
                _audit_event(
                    "output_disposition",
                    "review_required" if active_model_alert else "no_alert_no_generation",
                    evidence_ids=request.evidence_ids,
                    details={"reasons": review_reasons},
                ),
            ])
            suppression_trace = self._suppression_trace(
                request,
                command,
                retrieval_latency_ms=retrieval_latency_ms,
                end_to_end_started=end_to_end_started,
                policy=policy,
                safety_findings=safety_findings,
                retrieval_query=retrieval_query,
                requires_human_review=active_model_alert,
            )
            suppression_trace.update({
                "guardrail_violation_count": len(review_reasons),
                "requires_human_review": active_model_alert,
                "audit_event_count": len(audit_events),
            })
            result = InvestigationResult(
                request_id=request.request_id,
                configuration=mode.value,
                status="suppressed",
                question=command.question,
                retrieval_query=retrieval_query,
                sample_id=command.sample_id,
                answer=request.suppression_reason or "Generation suppressed.",
                candidate_response_text=None,
                parsed_answer=None,
                model_evidence=model_evidence_payload,
                evidence_ids=request.evidence_ids,
                retrieved_policy=policy_payload,
                guardrail_checks={"model_score_generation_gate": not active_model_alert},
                guardrail_violations=review_reasons,
                requires_human_review=active_model_alert,
                human_review_reasons=review_reasons,
                audit_events=audit_events,
                trace=suppression_trace,
                condition_id=command.condition_id,
                retrieval_strategy_id=command.retrieval_strategy_id,
                guardrail_profile=command.guardrail_profile,
            )
            self._run_repository.save(result.to_dict(), request.request_id)
            return result

        generation_started = time.monotonic()
        started_cpu, started_memory = self._resource_snapshot()
        generation = self._generator.generate(
            request.prompt, max_new_tokens=command.max_new_tokens
        )
        generation_latency_ms = round(
            (time.monotonic() - generation_started) * 1_000, 3
        )
        validation_started = time.monotonic()
        outcome = strategy.finalize(generation.text, request)
        validation_latency_ms = round(
            (time.monotonic() - validation_started) * 1_000, 3
        )
        audit_events.append(_audit_event(
            "generation_completed",
            "completed",
            evidence_ids=request.evidence_ids,
            details={
                "model_name": generation.model_name,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            },
        ))
        for check_name, passed in outcome.checks.items():
            audit_events.append(_audit_event(
                "guardrail_check",
                "passed" if passed else "failed",
                evidence_ids=request.evidence_ids,
                details={"check_name": check_name},
            ))
        claims = (
            outcome.parsed_answer.get("claims", [])
            if isinstance(outcome.parsed_answer, dict) else []
        )
        audit_events.append(_audit_event(
            "claim_ledger",
            "completed" if claims else "unavailable",
            evidence_ids=sorted({
                citation
                for claim in claims if isinstance(claim, dict)
                for citation in claim.get("citations", []) if isinstance(citation, str)
            }),
            details={
                "claim_count": len(claims),
                "reason": None if claims else "no_parseable_structured_claims",
            },
        ))
        requires_human_review = bool(outcome.violations)
        review_reasons = list(outcome.violations)
        audit_events.append(_audit_event(
            "output_disposition",
            "review_required" if requires_human_review else "released_for_analyst_review",
            evidence_ids=request.evidence_ids,
            details={"reasons": review_reasons},
        ))
        trace = build_observation(
            request,
            model_name=generation.model_name,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            input_cost_per_million=command.input_cost_per_million,
            output_cost_per_million=command.output_cost_per_million,
            started_monotonic=generation_started,
        )
        review_cost = (
            command.analyst_hourly_cost_usd
            * command.estimated_review_minutes
            / 60.0
            if requires_human_review
            else 0.0
        )
        end_to_end_latency_ms = round(
            (time.monotonic() - end_to_end_started) * 1_000, 3
        )
        compute_cost = (
            command.compute_hourly_cost_usd * end_to_end_latency_ms / 3_600_000.0
        )
        ending_cpu, ending_memory = self._resource_snapshot()
        trace.update({
            "status": "blocked" if outcome.violations else "completed",
            "sample_id": command.sample_id,
            "question": command.question,
            "cpu_time_ms": round(max(0.0, ending_cpu - started_cpu) * 1000, 3),
            "memory_rss_start_bytes": started_memory,
            "memory_rss_end_bytes": ending_memory,
            "memory_rss_delta_bytes": ending_memory - started_memory,
            "guardrail_check_count": len(outcome.checks),
            "guardrail_violation_count": len(outcome.violations),
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "validation_latency_ms": validation_latency_ms,
            "retrieval_query_hash": _hash_text(retrieval_query) if retrieval_query else None,
            "prompt_sha256": _hash_text(request.prompt),
            "policy_corpus_hash": _policy_corpus_hash(policy),
            "policy_top_k": command.policy_top_k if strategy.requires_policy else None,
            "model_score_generation_gate": command.model_score_gate,
            "max_new_tokens": command.max_new_tokens,
            "experiment_id": command.experiment_id,
            "study_id": command.study_id,
            "question_id": command.question_id,
            "repeat": command.repeat,
            "run_order": command.run_order,
            "input_safety_findings": findings_payload(safety_findings),
            "input_safety_signal_count": len(safety_findings),
            "engine_name": getattr(self._generator, "model_name", type(self._generator).__name__),
            "generation_parameters": getattr(self._generator, "generation_parameters", {}),
            "retrieval_query_builder_version": RETRIEVAL_QUERY_BUILDER_VERSION,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "requires_human_review": requires_human_review,
            "estimated_human_review_cost_usd": round(review_cost, 8),
            "estimated_compute_cost_usd": round(compute_cost, 8),
            "total_estimated_cost_usd": round(
                float(trace["estimated_cost_usd"]) + review_cost + compute_cost, 8
            ),
            "cost_assumptions": {
                "input_cost_per_million": command.input_cost_per_million,
                "output_cost_per_million": command.output_cost_per_million,
                "analyst_hourly_cost_usd": command.analyst_hourly_cost_usd,
                "estimated_review_minutes": command.estimated_review_minutes,
                "compute_hourly_cost_usd": command.compute_hourly_cost_usd,
            },
            "audit_event_count": len(audit_events),
            "condition_id": command.condition_id,
            "retrieval_strategy_id": command.retrieval_strategy_id,
            "guardrail_profile": command.guardrail_profile,
        })
        result = InvestigationResult(
            request_id=request.request_id,
            configuration=mode.value,
            status=trace["status"],
            question=command.question,
            retrieval_query=retrieval_query,
            sample_id=command.sample_id,
            answer=outcome.answer,
            candidate_response_text=generation.text,
            parsed_answer=outcome.parsed_answer,
            model_evidence=model_evidence_payload,
            evidence_ids=request.evidence_ids,
            retrieved_policy=policy_payload,
            guardrail_checks=outcome.checks,
            guardrail_violations=outcome.violations,
            requires_human_review=requires_human_review,
            human_review_reasons=review_reasons,
            audit_events=audit_events,
            trace=trace,
            condition_id=command.condition_id,
            retrieval_strategy_id=command.retrieval_strategy_id,
            guardrail_profile=command.guardrail_profile,
        )
        self._run_repository.save(result.to_dict(), request.request_id)
        return result

    @staticmethod
    def _resource_snapshot() -> tuple[float, int]:
        try:
            import psutil

            process = psutil.Process()
            cpu = process.cpu_times()
            return cpu.user + cpu.system, int(process.memory_info().rss)
        except ImportError:
            return 0.0, 0

    def _suppression_trace(
        self,
        request: PreparedRequest,
        command: InvestigationCommand,
        *,
        retrieval_latency_ms: float,
        end_to_end_started: float,
        policy: Sequence[PolicyEvidence],
        safety_findings: Sequence[Any],
        retrieval_query: str | None,
        requires_human_review: bool,
    ) -> dict[str, Any]:
        end_to_end_latency_ms = round(
            (time.monotonic() - end_to_end_started) * 1_000, 3
        )
        review_cost = (
            command.analyst_hourly_cost_usd * command.estimated_review_minutes / 60.0
            if requires_human_review else 0.0
        )
        compute_cost = (
            command.compute_hourly_cost_usd * end_to_end_latency_ms / 3_600_000.0
        )
        return {
            "request_id": request.request_id,
            "configuration": request.configuration,
            "model_name": "not-called:model-score-gate",
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "estimated_human_review_cost_usd": round(review_cost, 8),
            "estimated_compute_cost_usd": round(compute_cost, 8),
            "total_estimated_cost_usd": round(review_cost + compute_cost, 8),
            "cost_assumptions": {
                "input_cost_per_million": command.input_cost_per_million,
                "output_cost_per_million": command.output_cost_per_million,
                "analyst_hourly_cost_usd": command.analyst_hourly_cost_usd,
                "estimated_review_minutes": command.estimated_review_minutes,
                "compute_hourly_cost_usd": command.compute_hourly_cost_usd,
            },
            "latency_ms": 0.0,
            "evidence_count": len(request.evidence_ids),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "suppressed",
            "sample_id": command.sample_id,
            "question": request.question,
            "cpu_time_ms": 0.0,
            "memory_rss_delta_bytes": 0,
            "guardrail_check_count": 1,
            "guardrail_violation_count": 0,
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieval_query_hash": _hash_text(retrieval_query) if retrieval_query else None,
            "prompt_sha256": None,
            "policy_corpus_hash": _policy_corpus_hash(policy),
            "policy_top_k": command.policy_top_k,
            "model_score_generation_gate": command.model_score_gate,
            "max_new_tokens": command.max_new_tokens,
            "experiment_id": command.experiment_id,
            "study_id": command.study_id,
            "question_id": command.question_id,
            "repeat": command.repeat,
            "run_order": command.run_order,
            "input_safety_findings": findings_payload(safety_findings),
            "input_safety_signal_count": len(safety_findings),
            "engine_name": getattr(self._generator, "model_name", type(self._generator).__name__),
            "generation_parameters": getattr(self._generator, "generation_parameters", {}),
            "retrieval_query_builder_version": RETRIEVAL_QUERY_BUILDER_VERSION,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "condition_id": command.condition_id,
            "retrieval_strategy_id": command.retrieval_strategy_id,
            "guardrail_profile": command.guardrail_profile,
        }

    def _blocked_input_result(
        self,
        *,
        command: InvestigationCommand,
        findings: Sequence[Any],
        started_monotonic: float,
        audit_events: list[dict[str, Any]],
    ) -> InvestigationResult:
        request_id = str(uuid.uuid4())
        violations = [f"{finding.code}: {finding.description}" for finding in findings]
        audit_events.extend([
            _audit_event(
                "guardrail_check",
                "failed",
                details={"check_name": "input_safety", "violations": violations},
            ),
            _audit_event(
                "output_disposition",
                "review_required",
                details={"reasons": violations},
            ),
        ])
        end_to_end_latency_ms = round(
            (time.monotonic() - started_monotonic) * 1_000, 3
        )
        review_cost = (
            command.analyst_hourly_cost_usd * command.estimated_review_minutes / 60.0
        )
        compute_cost = (
            command.compute_hourly_cost_usd * end_to_end_latency_ms / 3_600_000.0
        )
        trace = {
            "request_id": request_id,
            "configuration": Configuration.C.value,
            "model_name": "not-called:input-safety-block",
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "estimated_human_review_cost_usd": round(review_cost, 8),
            "estimated_compute_cost_usd": round(compute_cost, 8),
            "total_estimated_cost_usd": round(
                review_cost + compute_cost, 8
            ),
            "cost_assumptions": {
                "input_cost_per_million": command.input_cost_per_million,
                "output_cost_per_million": command.output_cost_per_million,
                "analyst_hourly_cost_usd": command.analyst_hourly_cost_usd,
                "estimated_review_minutes": command.estimated_review_minutes,
                "compute_hourly_cost_usd": command.compute_hourly_cost_usd,
            },
            "latency_ms": 0.0,
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "retrieval_latency_ms": 0.0,
            "evidence_count": 0,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "blocked",
            "sample_id": command.sample_id,
            "question": command.question,
            "cpu_time_ms": 0.0,
            "memory_rss_delta_bytes": 0,
            "guardrail_check_count": 1,
            "guardrail_violation_count": len(violations),
            "prompt_sha256": None,
            "retrieval_query_hash": None,
            "policy_corpus_hash": None,
            "policy_top_k": command.policy_top_k,
            "model_score_generation_gate": command.model_score_gate,
            "max_new_tokens": command.max_new_tokens,
            "experiment_id": command.experiment_id,
            "study_id": command.study_id,
            "question_id": command.question_id,
            "repeat": command.repeat,
            "run_order": command.run_order,
            "input_safety_findings": findings_payload(findings),
            "input_safety_signal_count": len(findings),
            "engine_name": getattr(self._generator, "model_name", type(self._generator).__name__),
            "generation_parameters": getattr(self._generator, "generation_parameters", {}),
            "retrieval_query_builder_version": RETRIEVAL_QUERY_BUILDER_VERSION,
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "requires_human_review": True,
            "audit_event_count": len(audit_events),
            "condition_id": command.condition_id,
            "retrieval_strategy_id": command.retrieval_strategy_id,
            "guardrail_profile": command.guardrail_profile,
        }
        result = InvestigationResult(
            request_id=request_id,
            configuration=Configuration.C.value,
            status="blocked",
            question=command.question,
            retrieval_query=None,
            sample_id=command.sample_id,
            answer=(
                "Request blocked by Configuration C input controls. Rephrase it as an "
                "evidence-based investigation question without instruction override or "
                "protected-data requests."
            ),
            candidate_response_text=None,
            parsed_answer=None,
            model_evidence=None,
            evidence_ids=[],
            retrieved_policy=[],
            guardrail_checks={"input_safety": False},
            guardrail_violations=violations,
            requires_human_review=True,
            human_review_reasons=violations,
            audit_events=audit_events,
            trace=trace,
            condition_id=command.condition_id,
            retrieval_strategy_id=command.retrieval_strategy_id,
            guardrail_profile=command.guardrail_profile,
        )
        self._run_repository.save(result.to_dict(), request_id)
        return result


def build_default_service(
    project_root: Path,
    generator: TextGenerator,
    *,
    sample_paths: Sequence[Path] | None = None,
    attribution_paths: Sequence[Path] | None = None,
    run_directory: Path | None = None,
    policy_repositories: Mapping[str, PolicyRepository] | None = None,
) -> InvestigationService:
    """Composition root kept outside domain classes for clean dependency injection."""
    sample_paths = sample_paths or (
        project_root / "datasets" / "ulb" / "runtime_transaction_samples.csv",
        project_root / "datasets" / "sparkov" / "runtime_transaction_samples.csv",
    )
    attribution_paths = attribution_paths or (
        project_root / "artifacts" / "ulb" / "runtime_sample_attributions.json",
        project_root / "artifacts" / "sparkov" / "runtime_sample_attributions.json",
    )
    return InvestigationService(
        evidence_repository=FileEvidenceRepository(project_root),
        policy_repository=LocalPolicyRepository(
            project_root / "datasets" / "policy_sources" / "source"
        ),
        sample_repository=CompositeSampleRepository([
            CsvSampleRepository(path) for path in sample_paths
        ]),
        run_repository=JsonRunRepository(
            run_directory or project_root / "artifacts" / "abc_runs"
        ),
        generator=generator,
        attribution_repository=CompositeAttributionRepository([
            FileAttributionRepository(path) for path in attribution_paths
        ]),
        policy_repositories=policy_repositories,
    )
