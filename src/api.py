"""FastAPI adapter for the Node.js A/B/C research interface."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Literal

import asyncio
import contextlib
import json
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.abc_evaluation import (
    METRIC_CATALOG,
    ClaimAnnotation,
    aggregate_runs,
    evaluate_run,
    validate_paired_design,
)
from src.abc_service import (
    CompositeSampleRepository,
    CsvSampleRepository,
    DeterministicDemoGenerator,
    InvestigationCommand,
    build_default_service,
)
from src.annotations import (
    AnnotationRecord,
    ClaimUnitRecord,
    JsonAnnotationRepository,
    JsonClaimUnitRepository,
    annotation_eligibility,
    blinded_annotation_task,
    counts_from_claim_assessments,
    pairwise_annotation_agreement,
)
from src.artifact_audit import audit_input_artifacts
from src.llm_provider import LocalQwenProvider, provider_is_available
from src.statistical_analysis import paired_configuration_analysis
from src.jira_analytics import calculate_jira_metrics
from src.ui_data import eda_and_models, evidence_registry, readiness
from src.ui_jobs import JobManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = PROJECT_ROOT / "artifacts" / "abc_runs"
LOGGER = logging.getLogger(__name__)


class InvestigationRequest(BaseModel):
    configuration: Literal["A", "B", "C"]
    question: str = Field(min_length=1, max_length=2_000)
    sample_id: str | None = None
    engine: Literal["demo", "qwen"] = "demo"
    model_score_gate: float = Field(default=0.70, ge=0.0, le=1.0)
    policy_top_k: int = Field(default=4, ge=1, le=10)
    max_new_tokens: int = Field(default=180, ge=20, le=1_500)
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    analyst_hourly_cost_usd: float = Field(default=0.0, ge=0.0)
    estimated_review_minutes: float = Field(default=0.0, ge=0.0)
    compute_hourly_cost_usd: float = Field(default=0.0, ge=0.0)


class OperationalAlertRequest(BaseModel):
    """One newly scored detector alert; operational routing is always through C."""

    sample_id: str = Field(min_length=1, max_length=300)
    question: str = Field(
        default=(
            "Using only the supplied evidence, explain the model risk drivers and "
            "limitations, identify relevant public governance context, and state the "
            "appropriate human investigation step without treating the score as proof "
            "of fraud."
        ),
        min_length=1,
        max_length=2_000,
    )
    engine: Literal["demo", "qwen"] = "qwen"
    model_score_gate: float = Field(default=0.70, ge=0.0, le=1.0)
    policy_top_k: int = Field(default=4, ge=1, le=10)
    max_new_tokens: int = Field(default=420, ge=20, le=1_500)


class OperationalAlertBatchRequest(BaseModel):
    sample_ids: list[str] = Field(min_length=1, max_length=10)
    engine: Literal["demo", "qwen"] = "qwen"
    model_score_gate: float = Field(default=0.70, ge=0.0, le=1.0)
    policy_top_k: int = Field(default=4, ge=1, le=10)
    max_new_tokens: int = Field(default=420, ge=20, le=1_500)
    confirm_write: bool = False


class JobCreateRequest(BaseModel):
    job_type: str = Field(min_length=1, max_length=100)
    parameters: dict = Field(default_factory=dict)


class ComparisonRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    sample_id: str
    engine: Literal["demo", "qwen"] = "demo"
    repeats: int = Field(default=1, ge=1, le=10)
    model_score_gate: float = Field(default=0.70, ge=0.0, le=1.0)
    policy_top_k: int = Field(default=4, ge=1, le=10)
    max_new_tokens: int = Field(default=180, ge=20, le=1_500)
    input_cost_per_million: float = Field(default=0.0, ge=0.0)
    output_cost_per_million: float = Field(default=0.0, ge=0.0)
    analyst_hourly_cost_usd: float = Field(default=0.0, ge=0.0)
    estimated_review_minutes: float = Field(default=0.0, ge=0.0)
    compute_hourly_cost_usd: float = Field(default=0.0, ge=0.0)


class CitationAssessmentRequest(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=300)
    correctness: Literal["correct", "incorrect", "not_assessable"]
    notes: str = Field(default="", max_length=1_000)


class ClaimAssessmentRequest(BaseModel):
    claim_id: str = Field(min_length=1, max_length=100)
    support_status: Literal["supported", "unsupported", "not_assessable"]
    citation_assessments: list[CitationAssessmentRequest] = Field(default_factory=list)
    policy_grounding: Literal["grounded", "ungrounded", "not_applicable"]
    failure_cause: Literal[
        "retrieval_miss",
        "grounding_misrepresentation",
        "citation_fabrication",
        "transaction_evidence_mismatch",
        "attribution_mismatch",
        "format_or_specification_failure",
        "unsupported_other",
        "not_applicable",
    ] = "not_applicable"
    notes: str = Field(default="", max_length=1_000)


class AnnotationRequest(BaseModel):
    request_id: str
    reviewer_code: str = Field(
        min_length=2, max_length=32, pattern=r"^[A-Za-z0-9_-]+$"
    )
    claim_assessments: list[ClaimAssessmentRequest] = Field(min_length=1)
    configuration_guess: Literal["A", "B", "C", "unknown"] = "unknown"
    notes: str = Field(default="", max_length=2_000)


class ClaimUnitInputRequest(BaseModel):
    claim_text: str = Field(min_length=1, max_length=2_000)
    citation_ids: list[str] = Field(default_factory=list, max_length=50)


class ClaimUnitCreateRequest(BaseModel):
    request_id: str
    creator_code: str = Field(
        min_length=2, max_length=32, pattern=r"^[A-Za-z0-9_-]+$"
    )
    claims: list[ClaimUnitInputRequest] = Field(min_length=1, max_length=100)


@lru_cache(maxsize=2)
def _service(engine: str):
    if engine == "qwen":
        if not provider_is_available():
            raise RuntimeError("Local Qwen dependencies are not installed.")
        generator = LocalQwenProvider()
    else:
        generator = DeterministicDemoGenerator()
    service = build_default_service(PROJECT_ROOT, generator)
    jira_trigger = _jira_trigger()
    if jira_trigger is not None:
        from src.jira_trigger import attach_post_save_trigger

        attach_post_save_trigger(service, jira_trigger)
    return service


@lru_cache(maxsize=1)
def _jira_trigger():
    from src.jira_trigger import build_jira_alert_trigger_from_environment

    return build_jira_alert_trigger_from_environment(PROJECT_ROOT)


@lru_cache(maxsize=1)
def _job_manager() -> JobManager:
    return JobManager(PROJECT_ROOT)


def _automatic_monitor_interval_seconds() -> int:
    raw = os.environ.get("JIRA_AUTO_MONITOR_INTERVAL_MINUTES", "60")
    try:
        minutes = int(raw)
    except ValueError as error:
        raise RuntimeError("JIRA_AUTO_MONITOR_INTERVAL_MINUTES must be an integer.") from error
    if minutes < 5 or minutes > 1_440:
        raise RuntimeError(
            "JIRA_AUTO_MONITOR_INTERVAL_MINUTES must be between 5 and 1440."
        )
    return minutes * 60


async def _automatic_monitor_loop(trigger, interval_seconds: int) -> None:
    """Run until API shutdown; each cycle remains bounded by JiraAgentConfig."""
    while True:
        try:
            await asyncio.to_thread(trigger.monitor_once)
        except Exception as error:
            LOGGER.exception(
                "jira_automatic_monitor_cycle_failed error_type=%s",
                type(error).__name__,
            )
        await asyncio.sleep(interval_seconds)


@contextlib.asynccontextmanager
async def _application_lifespan(_app: FastAPI):
    monitor_task: asyncio.Task | None = None
    if os.environ.get("JIRA_AUTO_MONITOR_ENABLED", "").strip().lower() == "true":
        trigger = _jira_trigger()
        if trigger is None:
            raise RuntimeError(
                "JIRA_AUTO_MONITOR_ENABLED requires JIRA_AUTO_REFERRAL_ENABLED=true."
            )
        interval = _automatic_monitor_interval_seconds()
        monitor_task = asyncio.create_task(_automatic_monitor_loop(trigger, interval))
        LOGGER.info("jira_automatic_monitor_started interval_seconds=%d", interval)
    try:
        yield
    finally:
        if monitor_task is not None:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task
            LOGGER.info("jira_automatic_monitor_stopped")


def _to_command(request: InvestigationRequest) -> InvestigationCommand:
    return InvestigationCommand(
        configuration=request.configuration,
        question=request.question.strip(),
        sample_id=request.sample_id,
        model_score_gate=request.model_score_gate,
        policy_top_k=request.policy_top_k,
        max_new_tokens=request.max_new_tokens,
        input_cost_per_million=request.input_cost_per_million,
        output_cost_per_million=request.output_cost_per_million,
        analyst_hourly_cost_usd=request.analyst_hourly_cost_usd,
        estimated_review_minutes=request.estimated_review_minutes,
        compute_hourly_cost_usd=request.compute_hourly_cost_usd,
    )


def _run_files() -> list[dict]:
    if not RUN_DIRECTORY.exists():
        return []
    runs: list[dict] = []
    for path in sorted(RUN_DIRECTORY.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def _question_id(question: str) -> str:
    normalized = " ".join(question.strip().split()).lower()
    return sha256(normalized.encode("utf-8")).hexdigest()[:16]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fraud Investigation A/B/C API",
        version="1.0.0",
        description="Paired system-level evaluation over frozen ULB/Sparkov evidence.",
        lifespan=_application_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    @app.get("/api/v1/health")
    def health() -> dict:
        policy_registry = (
            PROJECT_ROOT / "datasets" / "policy_sources" / "policy_source_registry.json"
        )
        return {
            "status": "ok",
            "qwen_available": provider_is_available(),
            "unified_evidence_available": (
                PROJECT_ROOT / "artifacts" / "unified" / "unified_model_evidence.json"
            ).exists(),
            "policy_registry_available": policy_registry.exists(),
            "jira_automation": {
                "referral_enabled": (
                    os.environ.get("JIRA_AUTO_REFERRAL_ENABLED", "").strip().lower()
                    == "true"
                ),
                "live_referral_writes": all(
                    os.environ.get(name, "").strip().lower() == "true"
                    for name in (
                        "JIRA_AUTO_REFERRAL_ENABLED",
                        "JIRA_ENABLE_WRITES",
                        "JIRA_AUTO_REFERRAL_APPROVED",
                    )
                ),
                "monitor_enabled": (
                    os.environ.get("JIRA_AUTO_MONITOR_ENABLED", "").strip().lower()
                    == "true"
                ),
                "planner_engine": os.environ.get("JIRA_AUTO_PLANNER_ENGINE", "rule"),
            },
        }

    @app.get("/api/v1/samples")
    def samples() -> dict:
        repository = CompositeSampleRepository([
            CsvSampleRepository(
                PROJECT_ROOT / "datasets" / "ulb" / "runtime_transaction_samples.csv"
            ),
            CsvSampleRepository(
                PROJECT_ROOT / "datasets" / "sparkov" / "runtime_transaction_samples.csv"
            ),
        ])
        return {
            "samples": repository.list(include_evaluation_labels=False),
            "label_leakage_control": (
                "evaluation_only_actual_class is withheld from the UI and every LLM prompt"
            ),
        }

    @app.post("/api/v1/investigations")
    def investigate(request: InvestigationRequest) -> dict:
        try:
            return _service(request.engine).run(_to_command(request)).to_dict()
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/v1/operational-alerts")
    def operational_alert(request: OperationalAlertRequest) -> dict:
        """Process a new detector alert through C and publish its post-save event."""
        try:
            result = _service(request.engine).run(InvestigationCommand(
                configuration="C",
                question=request.question.strip(),
                sample_id=request.sample_id,
                model_score_gate=request.model_score_gate,
                policy_top_k=request.policy_top_k,
                max_new_tokens=request.max_new_tokens,
            ))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "operational_mode": True,
            "configuration_forced": "C",
            "post_persistence_jira_trigger_enabled": (
                os.environ.get("JIRA_AUTO_REFERRAL_ENABLED", "").strip().lower()
                == "true"
            ),
            "result": result.to_dict(),
            "boundary": (
                "A Jira referral is a suspected-fraud research case for human review, "
                "not proof of fraud or an autonomous adverse action."
            ),
        }

    @app.post("/api/v1/comparisons")
    def compare(request: ComparisonRequest) -> dict:
        results: list[dict] = []
        experiment_id = str(uuid.uuid4())
        question_id = _question_id(request.question)
        base_order = ("A", "B", "C")
        try:
            service = _service(request.engine)
            for repeat in range(1, request.repeats + 1):
                offset = (repeat - 1) % len(base_order)
                run_sequence = base_order[offset:] + base_order[:offset]
                for run_order, configuration in enumerate(run_sequence, start=1):
                    command = InvestigationCommand(
                        configuration=configuration,
                        question=request.question.strip(),
                        sample_id=request.sample_id,
                        model_score_gate=request.model_score_gate,
                        policy_top_k=request.policy_top_k,
                        max_new_tokens=request.max_new_tokens,
                        input_cost_per_million=request.input_cost_per_million,
                        output_cost_per_million=request.output_cost_per_million,
                        analyst_hourly_cost_usd=request.analyst_hourly_cost_usd,
                        estimated_review_minutes=request.estimated_review_minutes,
                        compute_hourly_cost_usd=request.compute_hourly_cost_usd,
                        experiment_id=experiment_id,
                        question_id=question_id,
                        repeat=repeat,
                        run_order=run_order,
                    )
                    payload = service.run(command).to_dict()
                    results.append(payload)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "experimental_control": {
                "same_sample_id": request.sample_id,
                "same_question": request.question.strip(),
                "same_engine": request.engine,
                "same_repeats": request.repeats,
                "experiment_id": experiment_id,
                "question_id": question_id,
                "counterbalanced_order": True,
                "ground_truth_withheld_from_generation": True,
            },
            "results": results,
            "automatic_summary": aggregate_runs(results),
            "paired_design": validate_paired_design(results),
            "paired_statistical_analysis": paired_configuration_analysis(results),
        }

    @app.get("/api/v1/evaluations")
    def evaluations() -> dict:
        runs = _run_files()
        return {
            "metric_catalog": METRIC_CATALOG,
            "automatic_summary": aggregate_runs(runs),
            "paired_design": validate_paired_design(runs),
            "paired_statistical_analysis": paired_configuration_analysis(runs),
            "run_count": len(runs),
            "warning": (
                "Hallucination, citation correctness, and grounding completeness remain "
                "null until blinded claim-level annotations are submitted."
            ),
        }

    @app.post("/api/v1/evaluations/annotations")
    def annotate(request: AnnotationRequest) -> dict:
        path = RUN_DIRECTORY / f"{request.request_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Run not found.")
        run = json.loads(path.read_text(encoding="utf-8"))
        eligibility = annotation_eligibility(run)
        if not eligibility["eligible"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Run is not eligible for final blinded annotation.",
                    **eligibility,
                },
            )
        claim_unit_repository = JsonClaimUnitRepository(
            PROJECT_ROOT / "artifacts" / "abc_claim_units"
        )
        claim_unit_record = claim_unit_repository.get(request.request_id)
        if claim_unit_record is None:
            raise HTTPException(
                status_code=422,
                detail="Frozen claim units must be created before reviewer annotation.",
            )
        try:
            frozen_units = {
                unit["claim_id"]: unit["claim_text"]
                for unit in claim_unit_record["claim_units"]
            }
            claim_assessments = [
                {
                    **claim.model_dump(),
                    "claim_text": frozen_units.get(claim.claim_id),
                }
                for claim in request.claim_assessments
            ]
            supplied_claim_ids = [claim["claim_id"] for claim in claim_assessments]
            if len(supplied_claim_ids) != len(set(supplied_claim_ids)):
                raise ValueError("Each frozen claim_id may be assessed only once.")
            if set(supplied_claim_ids) != set(frozen_units):
                raise ValueError(
                    "Reviewer submission must assess every frozen claim_id exactly once."
                )
            frozen_unit_records = {
                unit["claim_id"]: unit for unit in claim_unit_record["claim_units"]
            }
            for claim in claim_assessments:
                assessed_ids = {
                    citation["evidence_id"]
                    for citation in claim["citation_assessments"]
                }
                frozen_ids = set(
                    frozen_unit_records[claim["claim_id"]].get("citation_ids", [])
                )
                if assessed_ids != frozen_ids:
                    raise ValueError(
                        f"{claim['claim_id']} must assess every frozen citation ID exactly once."
                    )
            counts = counts_from_claim_assessments(claim_assessments)
            allowed_evidence_ids = set(run.get("evidence_ids", []))
            assessed_evidence_ids = {
                citation["evidence_id"]
                for claim in claim_assessments
                for citation in claim["citation_assessments"]
            }
            unknown_evidence_ids = assessed_evidence_ids - allowed_evidence_ids
            if unknown_evidence_ids:
                raise ValueError(
                    "Annotation references evidence not supplied to the run: "
                    f"{sorted(unknown_evidence_ids)}"
                )
            annotation = ClaimAnnotation(
                **{
                    key: counts[key]
                    for key in (
                        "material_claims",
                        "unsupported_claims",
                        "citations_assessed",
                        "correct_citations",
                        "policy_dependent_claims",
                        "grounded_policy_claims",
                    )
                },
                claims_total=counts["claims_total"],
                not_assessable_claims=counts["not_assessable_claims"],
            )
            record = AnnotationRecord(
                run_id=request.request_id,
                annotator_code=AnnotationRecord.pseudonymize_annotator(
                    request.reviewer_code
                ),
                **{key: value for key, value in counts.items() if key != "failure_cause_counts"},
                claim_assessments=tuple(claim_assessments),
                configuration_guess=request.configuration_guess,
                notes=request.notes,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        repository = JsonAnnotationRepository(
            PROJECT_ROOT / "artifacts" / "abc_annotations"
        )
        path = repository.save(record)
        return {
            "metrics": evaluate_run(run, annotation),
            "annotation_id": record.annotation_id,
            "stored_relative_path": str(path.relative_to(PROJECT_ROOT)),
            "warning": "One annotation is not adjudicated ground truth.",
        }

    @app.get("/api/v1/evaluations/annotation-tasks")
    def annotation_tasks() -> dict:
        claim_unit_repository = JsonClaimUnitRepository(
            PROJECT_ROOT / "artifacts" / "abc_claim_units"
        )
        tasks = []
        for run in _run_files():
            if not annotation_eligibility(run)["eligible"]:
                continue
            claim_unit_record = claim_unit_repository.get(str(run.get("request_id")))
            if claim_unit_record is None:
                continue
            tasks.append(blinded_annotation_task(
                run, claim_units=claim_unit_record["claim_units"]
            ))
        return {
            "tasks": tasks,
            "configuration_hidden": True,
            "eligible_runs_only": True,
            "warning": (
                "Use two independent reviewers and adjudicate disagreements before "
                "reporting claim-level metrics."
            ),
        }

    @app.post("/api/v1/evaluations/claim-units")
    def create_claim_units(request: ClaimUnitCreateRequest) -> dict:
        path = RUN_DIRECTORY / f"{request.request_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Run not found.")
        run = json.loads(path.read_text(encoding="utf-8"))
        eligibility = annotation_eligibility(run)
        if not eligibility["eligible"]:
            raise HTTPException(status_code=422, detail=eligibility)
        normalized_claims = [
            {
                "claim_text": " ".join(claim.claim_text.split()),
                "citation_ids": list(dict.fromkeys(claim.citation_ids)),
            }
            for claim in request.claims
        ]
        if any(not claim["claim_text"] for claim in normalized_claims):
            raise HTTPException(status_code=422, detail="Claim text must not be empty.")
        allowed_evidence_ids = set(run.get("evidence_ids", []))
        unknown_citations = {
            citation_id
            for claim in normalized_claims
            for citation_id in claim["citation_ids"]
            if citation_id not in allowed_evidence_ids
        }
        if unknown_citations:
            raise HTTPException(
                status_code=422,
                detail=f"Claim units contain unknown evidence IDs: {sorted(unknown_citations)}",
            )
        claim_units = tuple({
            "claim_id": f"claim-{index:03d}-{sha256(claim['claim_text'].encode('utf-8')).hexdigest()[:10]}",
            **claim,
        } for index, claim in enumerate(normalized_claims, start=1))
        record = ClaimUnitRecord(
            run_id=request.request_id,
            creator_code=AnnotationRecord.pseudonymize_annotator(request.creator_code),
            claim_units=claim_units,
        )
        repository = JsonClaimUnitRepository(
            PROJECT_ROOT / "artifacts" / "abc_claim_units"
        )
        try:
            stored = repository.save(record)
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "task_id": request.request_id,
            "claim_units": claim_units,
            "stored_relative_path": str(stored.relative_to(PROJECT_ROOT)),
            "warning": (
                "Claim segmentation is now frozen for both reviewers. The creator must "
                "remain configuration-blind and must not rate the claims."
            ),
        }

    @app.get("/api/v1/evaluations/claim-unit-tasks")
    def claim_unit_tasks() -> dict:
        repository = JsonClaimUnitRepository(
            PROJECT_ROOT / "artifacts" / "abc_claim_units"
        )
        tasks = [
            blinded_annotation_task(run)
            for run in _run_files()
            if annotation_eligibility(run)["eligible"]
            and repository.get(str(run.get("request_id"))) is None
        ]
        return {
            "tasks": tasks,
            "configuration_hidden": True,
            "instruction": (
                "A configuration-blind claim-unit creator freezes atomic claim text; "
                "the two outcome reviewers then rate those identical units independently."
            ),
        }

    @app.get("/api/v1/evaluations/agreement/{request_id}")
    def annotation_agreement(request_id: str) -> dict:
        repository = JsonAnnotationRepository(
            PROJECT_ROOT / "artifacts" / "abc_annotations"
        )
        records = repository.list_for_run(request_id)
        if len(records) < 2:
            raise HTTPException(
                status_code=422,
                detail="At least two independent annotations are required.",
            )
        pairs = []
        try:
            for first, second in combinations(records, 2):
                pairs.append({
                    "first_annotation_id": first["annotation_id"],
                    "second_annotation_id": second["annotation_id"],
                    **pairwise_annotation_agreement(first, second),
                })
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "request_id": request_id,
            "annotation_count": len(records),
            "pairwise_agreement": pairs,
            "warning": (
                "Agreement is descriptive until reviewer independence and the claim-unit "
                "protocol are verified; disagreements still require adjudication."
            ),
        }

    @app.get("/api/v1/artifacts/audit")
    def artifact_audit() -> dict:
        return audit_input_artifacts(PROJECT_ROOT)

    @app.get("/api/v1/readiness")
    @app.get("/api/v1/ui/readiness")
    def ui_readiness() -> dict:
        return readiness(PROJECT_ROOT)

    @app.get("/api/v1/workflows")
    def workflows() -> dict:
        return {
            "job_types": _job_manager().list_specs(),
            "modes": [
                {
                    "id": "replay",
                    "label": "Replay existing evidence",
                    "new_model_calls": 0,
                    "jira_writes": 0,
                    "classification": "primary evidence replay",
                },
                {
                    "id": "fresh-build-pilot",
                    "label": "Fresh local build and deterministic pilot",
                    "new_model_calls": 0,
                    "jira_writes": 0,
                    "classification": "pipeline verification; not thesis evidence",
                },
                {
                    "id": "exact-replication",
                    "label": "Reserve legacy A/B/C replication",
                    "new_model_calls": 180,
                    "jira_writes": 0,
                    "classification": "legacy development scaffold; not thesis evidence",
                },
                {
                    "id": "jira-demo",
                    "label": "Operational Jira demonstration",
                    "new_model_calls": "one C call per alert",
                    "jira_writes": "bounded and confirmed",
                    "classification": "operations prototype",
                },
            ],
            "arbitrary_command_execution": False,
        }

    @app.get("/api/v1/jobs")
    def jobs(limit: int = Query(default=50, ge=1, le=500)) -> dict:
        return {"jobs": _job_manager().list(limit)}

    @app.post("/api/v1/jobs")
    def create_job(request: JobCreateRequest) -> dict:
        try:
            return _job_manager().create(request.job_type, request.parameters)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str) -> dict:
        try:
            return _job_manager().get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

    @app.get("/api/v1/jobs/{job_id}/log")
    def job_log(job_id: str, lines: int = Query(default=120, ge=1, le=2_000)) -> dict:
        try:
            return {"job_id": job_id, "lines": _job_manager().log_tail(job_id, lines=lines)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

    @app.get("/api/v1/jobs/{job_id}/events")
    async def job_events(job_id: str):
        try:
            _job_manager().get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

        async def stream():
            seen = 0
            while True:
                events = _job_manager().events(job_id)
                for event in events[seen:]:
                    yield "data: " + json.dumps(event) + "\n\n"
                seen = len(events)
                current = _job_manager().get(job_id)
                if current["terminal"]:
                    yield "event: end\ndata: " + json.dumps({"status": current["status"]}) + "\n\n"
                    return
                await asyncio.sleep(0.75)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        try:
            return _job_manager().cancel(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found.") from error

    @app.get("/api/v1/eda-models")
    def ui_eda_and_models() -> dict:
        return eda_and_models(PROJECT_ROOT)

    @app.get("/api/v1/eda/{dataset_id}")
    def ui_eda_dataset(dataset_id: str) -> dict:
        match = next(
            (item for item in eda_and_models(PROJECT_ROOT)["datasets"]
             if item["dataset_id"].lower() == dataset_id.lower()),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return match

    @app.get("/api/v1/models/{dataset_id}/builds")
    def ui_model_builds(dataset_id: str) -> dict:
        dataset = ui_eda_dataset(dataset_id)
        return {
            "dataset_id": dataset["dataset_id"],
            "builds": [{
                "build_id": "current",
                "run_manifest": dataset["run_manifest"],
                "test_metrics": dataset["test_metrics"],
            }],
        }

    @app.get("/api/v1/evidence")
    def ui_evidence_registry() -> dict:
        return evidence_registry(PROJECT_ROOT)

    @app.post("/api/v1/operational-alerts/preview")
    def preview_operational_alerts(request: OperationalAlertBatchRequest) -> dict:
        unique_ids = list(dict.fromkeys(request.sample_ids))
        return {
            "sample_ids": unique_ids,
            "requested_count": len(unique_ids),
            "maximum_possible_jira_writes": len(unique_ids),
            "configuration_forced": "C",
            "writes_performed": 0,
            "eligible_after_release_or_detector_only_fallback": True,
            "comparative_experiments_excluded": True,
            "warning": "A Jira Story is a human-review referral, not a fraud determination.",
        }

    @app.post("/api/v1/operational-alerts/batch")
    def batch_operational_alerts(request: OperationalAlertBatchRequest) -> dict:
        if not request.confirm_write:
            raise HTTPException(status_code=422, detail="confirm_write must be true.")
        required = (
            "JIRA_AUTO_REFERRAL_ENABLED", "JIRA_ENABLE_WRITES", "JIRA_AUTO_REFERRAL_APPROVED"
        )
        if not all(os.environ.get(name, "").strip().lower() == "true" for name in required):
            raise HTTPException(status_code=422, detail="Jira write gates are not all enabled.")
        unique_ids = list(dict.fromkeys(request.sample_ids))
        results = []
        try:
            service = _service(request.engine)
            for sample_id in unique_ids:
                result = service.run(InvestigationCommand(
                    configuration="C",
                    question=OperationalAlertRequest(sample_id=sample_id).question,
                    sample_id=sample_id,
                    model_score_gate=request.model_score_gate,
                    policy_top_k=request.policy_top_k,
                    max_new_tokens=request.max_new_tokens,
                ))
                results.append(result.to_dict())
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "configuration_forced": "C",
            "processed_count": len(results),
            "results": results,
            "boundary": (
                "Each eligible alert (released narrative or detector-only fallback) "
                "may create one idempotent human-review Story."
            ),
        }

    @app.get("/api/v1/jira/analytics")
    def jira_analytics() -> dict:
        try:
            from src.jira_agent import AGENT_LABEL
            from src.jira_client import JiraCloudClient, JiraCloudConfig, JiraClientError

            config = JiraCloudConfig.from_environment()
            client = JiraCloudClient(config)
            issues = client.search_issues(
                f'project = "{config.project_key}" AND issuetype = "Story" '
                f'AND labels = "{AGENT_LABEL}" ORDER BY created DESC',
                fields=("summary", "status", "created", "resolutiondate", "labels", "parent"),
                max_results=500,
            )
            changelogs = {
                str(issue.get("key")): client.get_issue_changelog(str(issue.get("key")))
                for issue in issues
            }
            return {"status": "available", **calculate_jira_metrics(issues, changelogs)}
        except (ValueError, RuntimeError, JiraClientError) as error:
            LOGGER.warning("jira_analytics_unavailable error_type=%s", type(error).__name__)
            return {
                "status": "unavailable",
                "reason": str(error).replace(os.environ.get("JIRA_API_TOKEN", "__never__"), "[REDACTED]"),
                "credentials_exposed": False,
            }

    @app.get("/api/v1/jira/issues")
    def jira_issues() -> dict:
        analytics = jira_analytics()
        return {
            "status": analytics.get("status"),
            "issues": analytics.get("issues", []),
            "credentials_exposed": False,
        }

    return app


app = create_app()
