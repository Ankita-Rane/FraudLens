"""Bounded post-Configuration-C Jira referral and monitoring agent.

This module is independent from the locked A/B/C experiment.  It converts only a
successfully released Configuration C packet into a research Jira Story, maintains
separate ULB and Sparkov Epic containers, creates saved JQL filters, and identifies
open/due-date exceptions.  External writes require both write-enabled configuration
and explicit per-call approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid

from src.jira_client import JiraGateway


AGENT_VERSION = "1.1"
ALLOWED_DATASETS = {"ULB", "Sparkov"}
PROHIBITED_LABEL_KEYS = {
    "evaluation_only_actual_class",
    "actual_class",
    "ground_truth",
    "ground_truth_label",
    "is_fraud",
    "fraud_label",
}
AGENT_LABEL = "agentic-referral"
STALE_LABEL = "open-over-7-days"
OVERDUE_LABEL = "overdue"
DUE_SOON_LABEL = "due-soon"
MISSING_DUE_LABEL = "missing-due-date"
LOGGER = logging.getLogger(__name__)


class JiraAgentError(RuntimeError):
    """Raised when a safety, permission, or input contract fails closed."""


class ActionPlanner(Protocol):
    """Planner boundary for a future schema-constrained local model planner."""

    def choose(self, observation: Mapping[str, Any]) -> str: ...


class PlannerTextGenerator(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 120) -> Any: ...


class BoundedRulePlanner:
    """Deterministic reference planner used until a model planner is evaluated."""

    ACTIONS = {"create_story", "flag_issue", "no_action"}

    def choose(self, observation: Mapping[str, Any]) -> str:
        action = str(observation.get("eligible_action", "no_action"))
        return action if action in self.ACTIONS else "no_action"


class StructuredModelPlanner:
    """Schema-constrained model planner whose decisions remain policy-authorized.

    The planner sees only a compact operational observation. It never receives Jira
    credentials, offline labels, raw transaction identities, or unrestricted tools.
    Invalid model output fails closed to ``no_action``.
    """

    ACTIONS = ("create_story", "flag_issue", "no_action")

    def __init__(self, generator: PlannerTextGenerator) -> None:
        self.generator = generator
        self.last_trace: dict[str, Any] = {}

    def choose(self, observation: Mapping[str, Any]) -> str:
        compact = {
            "eligible_action": observation.get("eligible_action"),
            "dataset_id": observation.get("dataset_id"),
            "issue_key": observation.get("issue_key"),
            "eligibility_checks": observation.get("eligibility_checks"),
            "reasons": observation.get("reasons"),
        }
        prompt = json.dumps({
            "role": "bounded Jira referral planner",
            "instruction": (
                "Choose exactly one allowlisted action. Never declare fraud, expose "
                "identity, modify evidence, or override the eligible_action. If the "
                "requested action is unclear, choose no_action. Return JSON only."
            ),
            "allowed_actions": list(self.ACTIONS),
            "observation": compact,
            "response_schema": {"action": "allowlisted string", "rationale": "short string"},
        }, ensure_ascii=False)
        started = time.perf_counter()
        try:
            generation = self.generator.generate(prompt, max_new_tokens=120)
            generated_text = str(generation.text)
            candidate = json.loads(generated_text)
        except Exception as error:
            self.last_trace = {
                "outcome": "no_action",
                "failure_type": type(error).__name__,
                "latency_seconds": round(time.perf_counter() - started, 6),
            }
            LOGGER.warning(
                "jira_model_planner_invalid_output error_type=%s outcome=no_action",
                type(error).__name__,
            )
            return "no_action"
        action = candidate.get("action") if isinstance(candidate, dict) else None
        eligible = compact["eligible_action"]
        self.last_trace = {
            "model_name": getattr(generation, "model_name", None),
            "model_revision": getattr(self.generator, "model_revision", None),
            "input_tokens": getattr(generation, "input_tokens", None),
            "output_tokens": getattr(generation, "output_tokens", None),
            "output_sha256": hashlib.sha256(
                generated_text.encode("utf-8")
            ).hexdigest(),
            "proposed_action": action,
            "eligible_action": eligible,
            "latency_seconds": round(time.perf_counter() - started, 6),
        }
        if action not in self.ACTIONS or action != eligible:
            self.last_trace["outcome"] = "no_action"
            LOGGER.warning(
                "jira_model_planner_unauthorized_decision proposed=%s eligible=%s outcome=no_action",
                action,
                eligible,
            )
            return "no_action"
        self.last_trace["outcome"] = str(action)
        LOGGER.info("jira_model_planner_decision action=%s", action)
        return str(action)


@dataclass(frozen=True)
class JiraAgentConfig:
    project_key: str
    writes_enabled: bool = False
    stale_days: int = 7
    due_soon_days: int = 2
    story_due_days: int = 7
    max_actions_per_run: int = 100
    epic_issue_type: str = "Epic"
    story_issue_type: str = "Story"

    def __post_init__(self) -> None:
        if not self.project_key or not self.project_key.replace("_", "").isalnum():
            raise ValueError("project_key must be an alphanumeric Jira project key.")
        for name in ("stale_days", "due_soon_days", "story_due_days"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.max_actions_per_run < 1:
            raise ValueError("max_actions_per_run must be positive.")


@dataclass(frozen=True)
class EpicSpec:
    dataset_id: str
    summary: str
    description: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class FilterSpec:
    name: str
    description: str
    jql: str
    favourite: bool = True


@dataclass(frozen=True)
class BoardSpec:
    name: str
    source_filter_name: str
    board_type: str = "kanban"


@dataclass(frozen=True)
class StoryDraft:
    request_id: str
    dataset_id: str
    epic_key: str
    alert_evidence_id: str
    alert_label: str
    source_label: str
    fields: dict[str, Any]
    eligibility_checks: dict[str, bool]
    content_mode: str = "released_configuration_c"
    safety_boundary: str = (
        "Research suspected-fraud referral; not proof of fraud or an adverse action."
    )


@dataclass(frozen=True)
class MonitoringFinding:
    issue_key: str
    epic_key: str | None
    created_date: str | None
    due_date: str | None
    status: str | None
    labels_before: tuple[str, ...]
    proposed_labels: tuple[str, ...]
    reasons: tuple[str, ...]
    action: str


@dataclass
class AgentRun:
    agent_run_id: str
    operation: str
    status: str
    dry_run: bool
    started_at_utc: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str | None = None
    completed_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonAgentAuditRepository:
    """Write immutable one-file-per-agent-run audit records atomically."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, run: AgentRun) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{run.agent_run_id}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Jira-agent audit: {path}")
        serialized = json.dumps(run.to_dict(), indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.directory, delete=False, suffix=".tmp"
        ) as handle:
            handle.write(serialized)
            temporary = Path(handle.name)
        os.replace(temporary, path)
        return path


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_label(request_id: str) -> str:
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
    return f"source-request-{digest}"


def _alert_label(evidence_id: str) -> str:
    """Return the stable Jira idempotency key for one detector alert/transaction."""
    digest = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:24]
    return f"alert-evidence-{digest}"


def _dataset_label(dataset_id: str) -> str:
    return f"dataset-{dataset_id.lower()}"


def _adf_text(text: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [{
            "type": "paragraph",
            "content": [{"type": "text", "text": text}],
        }],
    }


def _adf_case_description(sections: Sequence[tuple[str, Sequence[str]]]) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for heading, values in sections:
        content.append({
            "type": "heading",
            "attrs": {"level": 3},
            "content": [{"type": "text", "text": heading}],
        })
        for value in values:
            content.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": str(value)}],
            })
    return {"type": "doc", "version": 1, "content": content}


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def _is_experiment_run(c_run: Mapping[str, Any]) -> bool:
    """True when a run belongs to the locked/comparative A/B/C experiment.

    Mirrors ``JiraAlertRunTrigger.after_save()``'s own pre-check so both the
    automatic post-save trigger and every manual/shared caller of this agent
    apply the same rejection before any Story content is drafted.
    """
    trace = c_run.get("trace") if isinstance(c_run.get("trace"), Mapping) else {}
    return bool(trace.get("experiment_id"))


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def default_epic_specs() -> tuple[EpicSpec, EpicSpec]:
    return (
        EpicSpec(
            dataset_id="ULB",
            summary="ULB Research Alert Investigations",
            description=(
                "Contains analyst-review stories generated from validated Configuration C "
                "packets for the real, anonymised ULB benchmark. These are research cases, "
                "not live-bank fraud incidents."
            ),
            labels=(AGENT_LABEL, "dataset-ulb", "research-benchmark"),
        ),
        EpicSpec(
            dataset_id="Sparkov",
            summary="Sparkov Synthetic Alert Investigations",
            description=(
                "Contains analyst-review stories generated from validated Configuration C "
                "packets for Sparkov synthetic data. These are simulated research cases, "
                "not real transactions."
            ),
            labels=(AGENT_LABEL, "dataset-sparkov", "synthetic-research"),
        ),
    )


def default_filter_specs(project_key: str, stale_days: int = 7) -> tuple[FilterSpec, ...]:
    project = project_key.upper()
    base = f'project = "{project}" AND labels = "{AGENT_LABEL}"'
    return (
        FilterSpec(
            "Agentic referrals - open",
            "All unresolved research referral stories; also backs the Kanban board.",
            f"{base} AND statusCategory != Done ORDER BY Rank ASC",
        ),
        FilterSpec(
            f"Agentic referrals - open over {stale_days} days",
            "Unresolved research referrals whose creation age exceeds the monitoring SLA.",
            f"{base} AND statusCategory != Done AND created <= -{stale_days}d "
            "ORDER BY created ASC",
        ),
        FilterSpec(
            "Agentic referrals - overdue",
            "Unresolved research referrals past their Jira due date.",
            f"{base} AND statusCategory != Done AND duedate < startOfDay() "
            "ORDER BY duedate ASC",
        ),
        FilterSpec(
            "ULB agentic referrals - open",
            "Open ULB research referral stories.",
            f'{base} AND labels = "dataset-ulb" AND statusCategory != Done '
            "ORDER BY duedate ASC",
        ),
        FilterSpec(
            "Sparkov agentic referrals - open",
            "Open Sparkov synthetic research referral stories.",
            f'{base} AND labels = "dataset-sparkov" AND statusCategory != Done '
            "ORDER BY duedate ASC",
        ),
    )


def default_board_spec() -> BoardSpec:
    return BoardSpec(
        name="Fraud Research Investigation Kanban",
        source_filter_name="Agentic referrals - open",
    )


class JiraReferralAgent:
    """Execute an allowlisted, bounded referral and monitoring workflow."""

    def __init__(
        self,
        *,
        gateway: JiraGateway,
        config: JiraAgentConfig,
        audit_repository: JsonAgentAuditRepository,
        planner: ActionPlanner | None = None,
    ) -> None:
        self.gateway = gateway
        self.config = config
        self.audit_repository = audit_repository
        self.planner = planner or BoundedRulePlanner()

    def _start(self, operation: str, approve_write: bool) -> AgentRun:
        run = AgentRun(
            agent_run_id=str(uuid.uuid4()),
            operation=operation,
            status="running",
            dry_run=not (self.config.writes_enabled and approve_write),
            started_at_utc=_now_utc(),
        )
        LOGGER.info(
            "jira_agent_operation_started operation=%s run_id=%s dry_run=%s planner=%s",
            operation,
            run.agent_run_id,
            run.dry_run,
            type(self.planner).__name__,
        )
        return run

    def _finish(self, run: AgentRun, *, status: str, reason: str) -> dict[str, Any]:
        run.status = status
        run.stopped_reason = reason
        run.completed_at_utc = _now_utc()
        path = self.audit_repository.save(run)
        LOGGER.info(
            "jira_agent_operation_completed operation=%s run_id=%s status=%s reason=%s actions=%d",
            run.operation,
            run.agent_run_id,
            status,
            reason,
            len(run.actions),
        )
        payload = run.to_dict()
        payload["audit_path"] = str(path)
        return payload

    def _planner_audit(self, selected_action: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "action": "planner_decision",
            "planner": type(self.planner).__name__,
            "selected_action": selected_action,
        }
        trace = getattr(self.planner, "last_trace", None)
        if isinstance(trace, dict) and trace:
            record["trace"] = dict(trace)
        return record

    def doctor(self) -> dict[str, Any]:
        user = self.gateway.current_user()
        project = self.gateway.get_project(self.config.project_key)
        return {
            "status": "connected",
            "site_account_id": user.get("accountId"),
            "display_name": user.get("displayName"),
            "project_key": self.config.project_key,
            "project_name": project.get("name"),
            "project_id": project.get("id"),
            "writes_enabled": self.config.writes_enabled,
            "secret_values_returned": False,
        }

    def ensure_epics(self, *, approve_write: bool = False) -> dict[str, Any]:
        run = self._start("ensure_epics", approve_write)
        results: dict[str, Any] = {}
        for spec in default_epic_specs():
            jql = (
                f'project = "{self.config.project_key}" AND issuetype = Epic '
                f'AND labels = "{_dataset_label(spec.dataset_id)}" '
                f'AND labels = "{AGENT_LABEL}"'
            )
            existing = self.gateway.search_issues(
                jql, fields=("summary", "labels", "status"), max_results=2
            )
            if len(existing) > 1:
                raise JiraAgentError(
                    f"Multiple {spec.dataset_id} agent Epics exist; resolve the "
                    "ambiguous parent before continuing."
                )
            if existing:
                results[spec.dataset_id] = {
                    "status": "existing",
                    "key": existing[0].get("key"),
                }
                run.actions.append({
                    "action": "find_epic",
                    "dataset_id": spec.dataset_id,
                    "outcome": "existing",
                    "issue_key": existing[0].get("key"),
                })
                continue
            fields = {
                "project": {"key": self.config.project_key},
                "issuetype": {"name": self.config.epic_issue_type},
                "summary": spec.summary,
                "description": _adf_text(spec.description),
                "labels": list(spec.labels),
            }
            if run.dry_run:
                results[spec.dataset_id] = {"status": "draft", "fields": fields}
                outcome = "drafted"
            else:
                created = self.gateway.create_issue(fields)
                results[spec.dataset_id] = {
                    "status": "created",
                    "key": created.get("key"),
                    "id": created.get("id"),
                }
                outcome = "created"
            run.actions.append({
                "action": "create_epic",
                "dataset_id": spec.dataset_id,
                "outcome": outcome,
            })
        run.result = {"epics": results}
        return self._finish(run, status="completed", reason="epic_check_complete")

    def ensure_filters(self, *, approve_write: bool = False) -> dict[str, Any]:
        run = self._start("ensure_filters", approve_write)
        results: list[dict[str, Any]] = []
        for spec in default_filter_specs(self.config.project_key, self.config.stale_days):
            existing = self.gateway.find_filter(spec.name)
            if existing:
                results.append({"name": spec.name, "status": "existing", "id": existing.get("id")})
                run.actions.append({"action": "find_filter", "name": spec.name, "outcome": "existing"})
                continue
            payload = asdict(spec)
            if run.dry_run:
                results.append({"name": spec.name, "status": "draft", "payload": payload})
                outcome = "drafted"
            else:
                created = self.gateway.create_filter(payload)
                results.append({"name": spec.name, "status": "created", "id": created.get("id")})
                outcome = "created"
            run.actions.append({"action": "create_filter", "name": spec.name, "outcome": outcome})
        run.result = {"filters": results}
        return self._finish(run, status="completed", reason="filter_check_complete")

    def ensure_board(
        self,
        *,
        approve_write: bool = False,
        source_filter_id: str | int | None = None,
    ) -> dict[str, Any]:
        """Ensure one project-scoped Kanban board backed by the open-referrals filter.

        ``source_filter_id`` lets a bootstrap invocation reuse the ID returned by the
        immediately preceding filter-creation response. Jira's filter-search index can
        lag behind a successful create, so requiring a second name lookup in the same
        invocation can otherwise fail after the filter has already been written.
        """
        run = self._start("ensure_board", approve_write)
        spec = default_board_spec()
        existing = self.gateway.find_boards(
            project_key=self.config.project_key,
            name=spec.name,
        )
        if len(existing) > 1:
            raise JiraAgentError(
                "Multiple agent Kanban boards have the same name; resolve the ambiguity."
            )
        if existing:
            board = existing[0]
            run.actions.append({
                "action": "find_board",
                "outcome": "existing",
                "board_id": board.get("id"),
            })
            run.result = {"status": "existing", "board": board}
            return self._finish(run, status="completed", reason="existing_board_found")

        if source_filter_id is not None:
            source_filter_id_text = str(source_filter_id)
            if not source_filter_id_text.isdigit() or int(source_filter_id_text) < 1:
                raise JiraAgentError("The source filter ID must be a positive Jira ID.")
            resolved_filter_id: int | None = int(source_filter_id_text)
            run.actions.append({
                "action": "reuse_source_filter_id",
                "filter_name": spec.source_filter_name,
                "filter_id": resolved_filter_id,
                "outcome": "accepted",
            })
        else:
            source_filter = self.gateway.find_filter(spec.source_filter_name)
            resolved_filter_id = (
                int(source_filter["id"])
                if source_filter is not None and str(source_filter.get("id", "")).isdigit()
                else None
            )

        if resolved_filter_id is None:
            if not run.dry_run:
                raise JiraAgentError(
                    "The open-referrals saved filter does not exist; run filter "
                    "bootstrap before creating the Kanban board."
                )
            payload: dict[str, Any] = {
                "name": spec.name,
                "type": spec.board_type,
                "filter_dependency": spec.source_filter_name,
                "location": {
                    "type": "project",
                    "projectKeyOrId": self.config.project_key,
                },
            }
        else:
            payload = {
                "name": spec.name,
                "type": spec.board_type,
                "filterId": resolved_filter_id,
                "location": {
                    "type": "project",
                    "projectKeyOrId": self.config.project_key,
                },
            }
        if run.dry_run:
            status = "draft"
            result_board = payload
            outcome = "drafted"
        else:
            result_board = self.gateway.create_board(payload)
            status = "created"
            outcome = "created"
        run.actions.append({"action": "create_board", "outcome": outcome})
        run.result = {"status": status, "board": result_board}
        return self._finish(run, status="completed", reason="board_check_complete")

    def build_story_draft(
        self, c_run: Mapping[str, Any], *, epic_key: str, today: date | None = None
    ) -> StoryDraft:
        checks = {
            "configuration_c": c_run.get("configuration") == "C",
            "released_status": c_run.get("status") == "completed",
            "structured_response": isinstance(c_run.get("parsed_answer"), dict),
            "guardrails_present": bool(c_run.get("guardrail_checks")),
            "all_guardrails_passed": bool(c_run.get("guardrail_checks")) and all(
                value is True for value in c_run.get("guardrail_checks", {}).values()
            ),
            "no_guardrail_violations": not c_run.get("guardrail_violations"),
            "model_evidence_present": isinstance(c_run.get("model_evidence"), dict),
            "offline_label_absent": not bool(
                _walk_keys(c_run.get("model_evidence")) & PROHIBITED_LABEL_KEYS
            ),
            "epic_key_supplied": bool(re.fullmatch(r"[A-Z][A-Z0-9_]*-\d+", epic_key)),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise JiraAgentError(
                "Configuration C packet is not eligible for a Jira Story: "
                + ", ".join(failed)
            )
        request_id = str(c_run.get("request_id") or "")
        if not request_id:
            raise JiraAgentError("Configuration C packet has no request_id.")
        evidence = c_run["model_evidence"]
        active = evidence.get("active_evaluation_sample") or {}
        dataset_id = str(active.get("dataset_id") or "")
        if dataset_id not in ALLOWED_DATASETS:
            raise JiraAgentError(f"Unsupported or missing dataset_id: {dataset_id!r}")
        dataset_rows = [
            row for row in evidence.get("datasets", [])
            if row.get("dataset_id") == dataset_id
        ]
        if len(dataset_rows) != 1 or len(dataset_rows[0].get("top_alerts", [])) != 1:
            raise JiraAgentError("Expected exactly one active dataset alert in C evidence.")
        dataset = dataset_rows[0]
        alert = dataset["top_alerts"][0]
        try:
            score = float(alert["fraud_probability"])
            threshold = float(alert["decision_threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise JiraAgentError(
                "C evidence has no valid model score and decision threshold."
            ) from error
        checks["active_detector_alert"] = (
            alert.get("fraud_alert") in (1, True, "1") and score >= threshold
        )
        checks["alert_evidence_id_present"] = bool(alert.get("evidence_id"))
        if not checks["active_detector_alert"] or not checks["alert_evidence_id_present"]:
            failed = [
                name
                for name in ("active_detector_alert", "alert_evidence_id_present")
                if not checks[name]
            ]
            raise JiraAgentError(
                "Configuration C packet is not eligible for a Jira Story: "
                + ", ".join(failed)
            )
        parsed = c_run["parsed_answer"]
        limitations = [str(item) for item in parsed.get("limitations", [])]
        if dataset_id == "Sparkov" and not any("synthetic" in item.lower() for item in limitations):
            limitations.append("Sparkov is synthetic; this is not a real transaction.")
        if dataset_id == "ULB" and not any("anonym" in item.lower() for item in limitations):
            limitations.append("ULB is real but anonymised; V1-V28 meanings are unknown.")
        citations = [str(item) for item in parsed.get("citations", [])]
        drivers = [str(item) for item in parsed.get("model_drivers", [])]
        actions = [str(item) for item in parsed.get("recommended_actions", [])]
        alert_evidence_id = str(alert["evidence_id"])
        alert_label = _alert_label(alert_evidence_id)
        source_label = _source_label(request_id)
        due = (today or date.today()) + timedelta(days=self.config.story_due_days)
        sections = (
            ("Research case", (
                f"Case ID: {c_run.get('sample_id')}",
                f"Dataset: {dataset_id}",
                "Classification: model-generated suspected-fraud alert; not proof of fraud.",
            )),
            ("Detector evidence", (
                f"Model: {dataset.get('model_name')}",
                f"Model score: {score:.6f}",
                f"Validation-selected threshold: {threshold:.6f}",
                "Top model drivers: " + (", ".join(drivers) if drivers else "Unavailable"),
            )),
            ("Configuration C investigation packet", (str(parsed.get("answer", "")),)),
            ("Evidence IDs", tuple(citations) or ("No citations supplied",)),
            ("Recommended analyst actions", tuple(actions) or ("Review the supplied evidence",)),
            ("Limitations", tuple(limitations)),
            ("Audit lineage", (
                f"Request ID: {request_id}",
                f"Prompt SHA-256: {(c_run.get('trace') or {}).get('prompt_sha256')}",
                f"Engine: {(c_run.get('trace') or {}).get('engine_name')}",
            )),
        )
        fields = {
            "project": {"key": self.config.project_key},
            "issuetype": {"name": self.config.story_issue_type},
            "parent": {"key": epic_key},
            "summary": (
                f"[{dataset_id} research review] Suspected-fraud alert "
                f"{c_run.get('sample_id')}"
            )[:255],
            "description": _adf_case_description(sections),
            "labels": [
                AGENT_LABEL,
                _dataset_label(dataset_id),
                "research-benchmark",
                alert_label,
                source_label,
            ],
            "duedate": due.isoformat(),
        }
        return StoryDraft(
            request_id=request_id,
            dataset_id=dataset_id,
            epic_key=epic_key,
            alert_evidence_id=alert_evidence_id,
            alert_label=alert_label,
            source_label=source_label,
            fields=fields,
            eligibility_checks=checks,
        )

    def build_fallback_story_draft(
        self, c_run: Mapping[str, Any], *, epic_key: str, today: date | None = None
    ) -> StoryDraft:
        """Build a detector-only case when C cannot safely release a narrative.

        The blocked/suppressed candidate response is deliberately excluded. This path
        exists so an active detector alert is never silently dropped merely because
        explanation generation or output guardrails did not release C content.
        """
        evidence = c_run.get("model_evidence")
        checks = {
            "configuration_c": c_run.get("configuration") == "C",
            "model_evidence_present": isinstance(evidence, dict),
            "offline_label_absent": not bool(
                _walk_keys(evidence) & PROHIBITED_LABEL_KEYS
            ),
            "epic_key_supplied": bool(re.fullmatch(r"[A-Z][A-Z0-9_]*-\d+", epic_key)),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise JiraAgentError(
                "Configuration C packet has no safe detector-only Jira fallback: "
                + ", ".join(failed)
            )
        request_id = str(c_run.get("request_id") or "")
        if not request_id:
            raise JiraAgentError("Configuration C packet has no request_id.")
        active = evidence.get("active_evaluation_sample") or {}
        dataset_id = str(active.get("dataset_id") or "")
        if dataset_id not in ALLOWED_DATASETS:
            raise JiraAgentError(f"Unsupported or missing dataset_id: {dataset_id!r}")
        dataset_rows = [
            row for row in evidence.get("datasets", [])
            if row.get("dataset_id") == dataset_id
        ]
        if len(dataset_rows) != 1 or len(dataset_rows[0].get("top_alerts", [])) != 1:
            raise JiraAgentError("Expected exactly one active dataset alert in C evidence.")
        dataset = dataset_rows[0]
        alert = dataset["top_alerts"][0]
        try:
            score = float(alert["fraud_probability"])
            threshold = float(alert["decision_threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise JiraAgentError(
                "C evidence has no valid model score and decision threshold."
            ) from error
        checks["active_detector_alert"] = (
            alert.get("fraud_alert") in (1, True, "1") and score >= threshold
        )
        checks["alert_evidence_id_present"] = bool(alert.get("evidence_id"))
        if not checks["active_detector_alert"] or not checks["alert_evidence_id_present"]:
            failed = [
                name
                for name in ("active_detector_alert", "alert_evidence_id_present")
                if not checks[name]
            ]
            raise JiraAgentError(
                "Configuration C packet has no safe detector-only Jira fallback: "
                + ", ".join(failed)
            )

        alert_evidence_id = str(alert["evidence_id"])
        alert_label = _alert_label(alert_evidence_id)
        source_label = _source_label(request_id)
        status = str(c_run.get("status") or "unknown")
        reasons = [str(item) for item in c_run.get("guardrail_violations", [])]
        reasons.extend(str(item) for item in c_run.get("human_review_reasons", []))
        reasons = list(dict.fromkeys(reasons))
        limitations = [
            "Configuration C narrative was not released; no blocked candidate text is included.",
            "The detector alert is not proof of fraud; a human analyst must investigate.",
        ]
        if dataset_id == "ULB":
            limitations.append("ULB is real but anonymised; V1-V28 meanings are unknown.")
        else:
            limitations.append("Sparkov is synthetic; this is not a real transaction.")
        sections = (
            ("Research case", (
                f"Case ID: {c_run.get('sample_id')}",
                f"Dataset: {dataset_id}",
                "Classification: model-generated suspected-fraud alert; not proof of fraud.",
            )),
            ("Detector evidence", (
                f"Model: {dataset.get('model_name')}",
                f"Model score: {score:.6f}",
                f"Validation-selected threshold: {threshold:.6f}",
                f"Alert evidence ID: {alert_evidence_id}",
            )),
            ("Configuration C disposition", (
                f"Status: {status}",
                "Release reason(s): " + (", ".join(reasons) if reasons else "not supplied"),
            )),
            ("Required analyst action", (
                "Review the detector evidence manually; do not infer fraud from the score alone.",
            )),
            ("Limitations", tuple(limitations)),
            ("Audit lineage", (
                f"Request ID: {request_id}",
                f"Prompt SHA-256: {(c_run.get('trace') or {}).get('prompt_sha256')}",
                f"Engine: {(c_run.get('trace') or {}).get('engine_name')}",
            )),
        )
        due = (today or date.today()) + timedelta(days=self.config.story_due_days)
        fields = {
            "project": {"key": self.config.project_key},
            "issuetype": {"name": self.config.story_issue_type},
            "parent": {"key": epic_key},
            "summary": (
                f"[{dataset_id} research review] Suspected-fraud alert "
                f"{c_run.get('sample_id')}"
            )[:255],
            "description": _adf_case_description(sections),
            "labels": [
                AGENT_LABEL,
                _dataset_label(dataset_id),
                "research-benchmark",
                "c-narrative-unavailable",
                "human-review-required",
                alert_label,
                source_label,
            ],
            "duedate": due.isoformat(),
        }
        return StoryDraft(
            request_id=request_id,
            dataset_id=dataset_id,
            epic_key=epic_key,
            alert_evidence_id=alert_evidence_id,
            alert_label=alert_label,
            source_label=source_label,
            fields=fields,
            eligibility_checks=checks,
            content_mode="detector_only_fallback",
        )

    def build_referral_draft(
        self,
        c_run: Mapping[str, Any],
        *,
        epic_key: str,
        today: date | None = None,
        allow_guardrail_fallback: bool = False,
    ) -> StoryDraft:
        if _is_experiment_run(c_run):
            raise JiraAgentError(
                "ignored_experiment_run: locked/comparative A/B/C runs cannot "
                "produce external Jira effects."
            )
        try:
            return self.build_story_draft(c_run, epic_key=epic_key, today=today)
        except JiraAgentError:
            if not allow_guardrail_fallback:
                raise
            return self.build_fallback_story_draft(c_run, epic_key=epic_key, today=today)

    def create_story(
        self,
        c_run: Mapping[str, Any],
        *,
        epic_key: str,
        approve_write: bool = False,
        today: date | None = None,
        validate_epic_parent: bool = True,
        allow_guardrail_fallback: bool = False,
    ) -> dict[str, Any]:
        run = self._start("create_story", approve_write)
        if _is_experiment_run(c_run):
            run.actions.append({
                "action": "create_story",
                "outcome": "rejected",
                "reason": "ignored_experiment_run",
            })
            run.result = {"status": "rejected"}
            return self._finish(run, status="stopped", reason="ignored_experiment_run")
        draft = self.build_referral_draft(
            c_run,
            epic_key=epic_key,
            today=today,
            allow_guardrail_fallback=allow_guardrail_fallback,
        )
        if validate_epic_parent:
            parent = self.gateway.get_issue(
                epic_key, fields=("issuetype", "labels", "summary")
            )
            parent_fields = parent.get("fields") or {}
            parent_labels = set(parent_fields.get("labels") or [])
            parent_type = (parent_fields.get("issuetype") or {}).get("name")
            if (
                parent_type != self.config.epic_issue_type
                or AGENT_LABEL not in parent_labels
                or _dataset_label(draft.dataset_id) not in parent_labels
            ):
                raise JiraAgentError(
                    f"{epic_key} is not the agent-managed {draft.dataset_id} Epic."
                )
            run.actions.append({
                "action": "validate_epic_parent",
                "dataset_id": draft.dataset_id,
                "epic_key": epic_key,
                "outcome": "passed",
            })
        action = self.planner.choose({
            "eligible_action": "create_story",
            "dataset_id": draft.dataset_id,
            "eligibility_checks": draft.eligibility_checks,
        })
        run.actions.append(self._planner_audit(action))
        if action != "create_story":
            run.result = {"draft": asdict(draft)}
            return self._finish(run, status="stopped", reason="planner_declined_story")
        duplicate_jql = (
            f'project = "{self.config.project_key}" AND labels = "{draft.alert_label}"'
        )
        duplicates = self.gateway.search_issues(
            duplicate_jql, fields=("summary", "status", "labels"), max_results=2
        )
        duplicate_match = "alert_evidence_id"
        if not duplicates:
            # Compatibility lookup for a Story written by agent version 1.0. Once
            # found, the approved path adds the stable alert label so later C
            # requests for the same transaction cannot create another Story.
            legacy_jql = (
                f'project = "{self.config.project_key}" '
                f'AND labels = "{draft.source_label}"'
            )
            duplicates = self.gateway.search_issues(
                legacy_jql, fields=("summary", "status", "labels"), max_results=2
            )
            duplicate_match = "legacy_source_request" if duplicates else duplicate_match
        if duplicates:
            existing = duplicates[0]
            existing_labels = set((existing.get("fields") or {}).get("labels") or [])
            label_migration_needed = draft.alert_label not in existing_labels
            if label_migration_needed and not run.dry_run:
                self.gateway.update_issue(
                    str(existing["key"]),
                    {"labels": sorted(existing_labels | {draft.alert_label})},
                )
            if label_migration_needed:
                run.actions.append({
                    "action": "migrate_alert_idempotency_label",
                    "outcome": "drafted" if run.dry_run else "updated",
                    "issue_key": existing.get("key"),
                    "alert_label": draft.alert_label,
                })
            run.actions.append({
                "action": "create_story",
                "outcome": "duplicate_prevented",
                "issue_key": existing.get("key"),
                "matched_by": duplicate_match,
            })
            run.result = {"status": "existing", "issue": existing, "draft": asdict(draft)}
            return self._finish(run, status="completed", reason="idempotent_existing_story")
        if run.dry_run:
            run.actions.append({"action": "create_story", "outcome": "drafted"})
            run.result = {"status": "draft", "draft": asdict(draft)}
            return self._finish(run, status="completed", reason="dry_run_no_external_write")
        created = self.gateway.create_issue(draft.fields)
        run.actions.append({
            "action": "create_story",
            "outcome": "created",
            "issue_key": created.get("key"),
        })
        run.result = {
            "status": "created",
            "issue": created,
            "source_request_id": draft.request_id,
            "alert_evidence_id": draft.alert_evidence_id,
        }
        return self._finish(run, status="completed", reason="story_created")

    def monitor_epics(
        self,
        epic_keys: Sequence[str],
        *,
        approve_write: bool = False,
        now: datetime | None = None,
        validate_epic_parents: bool = True,
    ) -> dict[str, Any]:
        if not epic_keys or any(
            not re.fullmatch(r"[A-Z][A-Z0-9_]*-\d+", key) for key in epic_keys
        ):
            raise JiraAgentError("monitor_epics requires valid ULB/Sparkov Epic keys.")
        run = self._start("monitor_epics", approve_write)
        if validate_epic_parents:
            found_datasets: set[str] = set()
            for epic_key in epic_keys:
                epic = self.gateway.get_issue(
                    epic_key, fields=("issuetype", "labels", "summary")
                )
                epic_fields = epic.get("fields") or {}
                labels = set(epic_fields.get("labels") or [])
                issue_type = (epic_fields.get("issuetype") or {}).get("name")
                if issue_type != self.config.epic_issue_type or AGENT_LABEL not in labels:
                    raise JiraAgentError(
                        f"{epic_key} is not an agent-managed Epic."
                    )
                matching = {
                    dataset
                    for dataset in ALLOWED_DATASETS
                    if _dataset_label(dataset) in labels
                }
                if len(matching) != 1:
                    raise JiraAgentError(
                        f"{epic_key} must identify exactly one ULB or Sparkov dataset."
                    )
                found_datasets.update(matching)
            if found_datasets != ALLOWED_DATASETS:
                raise JiraAgentError(
                    "Monitoring requires one ULB Epic and one Sparkov Epic."
                )
        clock = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        quoted = ", ".join(f'"{key}"' for key in epic_keys)
        jql = (
            f'parent in ({quoted}) AND labels = "{AGENT_LABEL}" '
            "AND statusCategory != Done ORDER BY duedate ASC, created ASC"
        )
        issues = self.gateway.search_issues(
            jql,
            fields=("summary", "status", "created", "updated", "duedate", "labels", "parent"),
            max_results=500,
        )
        findings: list[MonitoringFinding] = []
        action_count = 0
        for issue in issues:
            fields = issue.get("fields") or {}
            labels = tuple(str(item) for item in fields.get("labels", []))
            created = _parse_datetime(fields.get("created"))
            due = _parse_date(fields.get("duedate"))
            reasons: list[str] = []
            proposed = set(labels)
            if created and clock - created >= timedelta(days=self.config.stale_days):
                reasons.append(f"open_at_least_{self.config.stale_days}_days")
                proposed.add(STALE_LABEL)
            if due is None:
                reasons.append("missing_due_date")
                proposed.add(MISSING_DUE_LABEL)
            elif due < clock.date():
                reasons.append("past_due_date")
                proposed.add(OVERDUE_LABEL)
            elif due <= clock.date() + timedelta(days=self.config.due_soon_days):
                reasons.append(f"due_within_{self.config.due_soon_days}_days")
                proposed.add(DUE_SOON_LABEL)
            new_labels = tuple(sorted(proposed - set(labels)))
            action = self.planner.choose({
                "eligible_action": "flag_issue" if new_labels else "no_action",
                "issue_key": issue.get("key"),
                "reasons": reasons,
            })
            planner_record = self._planner_audit(action)
            planner_record["issue_key"] = issue.get("key")
            run.actions.append(planner_record)
            if action == "flag_issue" and new_labels:
                if action_count >= self.config.max_actions_per_run:
                    run.stopped_reason = "maximum_action_limit_reached"
                    break
                action_count += 1
                if not run.dry_run:
                    self.gateway.update_issue(
                        str(issue["key"]), {"labels": sorted(proposed)}
                    )
                    self.gateway.add_comment(
                        str(issue["key"]),
                        _adf_text(
                            "Bounded monitoring agent flag: " + ", ".join(reasons)
                            + ". Human review remains required."
                        ),
                    )
                run.actions.append({
                    "action": "flag_issue",
                    "issue_key": issue.get("key"),
                    "outcome": "drafted" if run.dry_run else "updated",
                    "labels_added": list(new_labels),
                    "reasons": reasons,
                })
            finding = MonitoringFinding(
                issue_key=str(issue.get("key")),
                epic_key=(fields.get("parent") or {}).get("key"),
                created_date=created.date().isoformat() if created else None,
                due_date=due.isoformat() if due else None,
                status=(fields.get("status") or {}).get("name"),
                labels_before=labels,
                proposed_labels=new_labels,
                reasons=tuple(reasons),
                action=action,
            )
            findings.append(finding)
        reason = run.stopped_reason or "all_open_epic_issues_checked"
        run.result = {
            "epic_keys": list(epic_keys),
            "open_issue_count": len(issues),
            "flagged_or_proposed_count": sum(bool(item.proposed_labels) for item in findings),
            "missing_due_date_count": sum("missing_due_date" in item.reasons for item in findings),
            "overdue_count": sum("past_due_date" in item.reasons for item in findings),
            "open_over_sla_count": sum(
                f"open_at_least_{self.config.stale_days}_days" in item.reasons
                for item in findings
            ),
            "findings": [asdict(item) for item in findings],
        }
        return self._finish(run, status="completed", reason=reason)
