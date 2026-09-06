"""Event-driven Jira referral trigger for newly persisted operational C alerts.

The trigger is deliberately outside the locked A/B/C experiment runner. It observes
an immutable run only after persistence, routes one unique detector evidence ID to its
dataset Epic, and delegates the bounded create/find decision to ``JiraReferralAgent``.
Jira failures never rewrite or invalidate the scientific run record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import copy
import json
import logging
import os
import re
import tempfile
import uuid

from src.jira_agent import (
    AGENT_VERSION,
    JiraAgentConfig,
    JiraAgentError,
    JiraReferralAgent,
    JsonAgentAuditRepository,
    StructuredModelPlanner,
)
from src.jira_client import JiraCloudClient, JiraCloudConfig, JiraClientError


LOGGER = logging.getLogger(__name__)
ISSUE_KEY_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*-\d+")


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() == "true"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_application_logging(project_root: Path) -> Path:
    """Ensure API-trigger activity is appended to the shared application log."""
    log_path = project_root / "log" / "application.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in root.handlers
    ):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        root.addHandler(handler)
    return log_path


class JsonTriggerAuditRepository:
    """Persist one immutable, privacy-minimised record per trigger decision."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, record: Mapping[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        event_id = str(record.get("event_id") or uuid.uuid4())
        path = self.directory / f"{event_id}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Jira-trigger audit: {path}")
        serialized = json.dumps(dict(record), indent=2, ensure_ascii=False)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.directory, delete=False, suffix=".tmp"
        ) as handle:
            handle.write(serialized)
            temporary = Path(handle.name)
        os.replace(temporary, path)
        return path


class PostSaveRunRepository:
    """Operational adapter that publishes only after the immutable run is saved.

    This decorator keeps event delivery outside ``src/abc_service.py``, whose exact
    bytes are registered by the locked A/B/C protocol. A Jira failure is therefore
    isolated from the scientific persistence contract.
    """

    def __init__(self, repository: Any, hooks: tuple[Any, ...]) -> None:
        self.repository = repository
        self.hooks = hooks

    def save(self, payload: dict[str, Any], request_id: str) -> Path:
        path = self.repository.save(payload, request_id)
        for hook in self.hooks:
            try:
                hook.after_save(copy.deepcopy(payload), path)
            except Exception as error:
                LOGGER.exception(
                    "run_saved_hook_failed hook=%s request_id=%s error_type=%s",
                    type(hook).__name__,
                    request_id,
                    type(error).__name__,
                )
        return path


def attach_post_save_trigger(service: Any, trigger: Any) -> Any:
    """Decorate the API service's repository without changing locked core code."""
    repository = getattr(service, "_run_repository", None)
    if repository is None or not callable(getattr(repository, "save", None)):
        raise TypeError("Investigation service has no compatible run repository.")
    service._run_repository = PostSaveRunRepository(repository, (trigger,))
    return service


@dataclass(frozen=True)
class JiraAutoReferralConfig:
    epic_keys: Mapping[str, str]
    approve_write: bool = False
    approve_monitor_write: bool = False
    exclude_experiment_runs: bool = True

    def __post_init__(self) -> None:
        if set(self.epic_keys) != {"ULB", "Sparkov"}:
            raise ValueError("Automatic referral requires ULB and Sparkov Epic keys.")
        if any(not ISSUE_KEY_PATTERN.fullmatch(key) for key in self.epic_keys.values()):
            raise ValueError("Automatic referral Epic keys must be valid Jira issue keys.")


class JiraAlertRunTrigger:
    """React once to each newly persisted operational Configuration C run."""

    def __init__(
        self,
        *,
        agent: JiraReferralAgent,
        config: JiraAutoReferralConfig,
        audit_repository: JsonTriggerAuditRepository,
    ) -> None:
        self.agent = agent
        self.config = config
        self.audit_repository = audit_repository

    def _record(self, base: dict[str, Any], **values: Any) -> Path:
        record = {**base, **values, "completed_at_utc": _now_utc()}
        path = self.audit_repository.save(record)
        LOGGER.info(
            "jira_alert_trigger_completed request_id=%s disposition=%s audit=%s",
            record.get("request_id"),
            record.get("disposition"),
            path,
        )
        return path

    def after_save(self, payload: Mapping[str, Any], run_path: Path) -> None:
        if payload.get("configuration") != "C":
            return
        trace = payload.get("trace") if isinstance(payload.get("trace"), Mapping) else {}
        base = {
            "event_id": str(uuid.uuid4()),
            "event_type": "configuration_c_run_persisted",
            "started_at_utc": _now_utc(),
            "request_id": payload.get("request_id"),
            "sample_id": payload.get("sample_id"),
            "run_path": str(run_path.resolve()),
            "agent_version": AGENT_VERSION,
        }
        if self.config.exclude_experiment_runs and trace.get("experiment_id"):
            self._record(
                base,
                disposition="ignored_experiment_run",
                reason="Locked/comparative A/B/C runs cannot produce external Jira effects.",
            )
            return

        evidence = payload.get("model_evidence")
        active = (
            evidence.get("active_evaluation_sample", {})
            if isinstance(evidence, Mapping) else {}
        )
        dataset_id = str(active.get("dataset_id") or "")
        epic_key = self.config.epic_keys.get(dataset_id)
        if not epic_key:
            self._record(
                base,
                disposition="ineligible",
                reason=f"Unsupported or missing dataset_id: {dataset_id!r}",
            )
            return

        try:
            draft = self.agent.build_referral_draft(
                payload,
                epic_key=epic_key,
                allow_guardrail_fallback=True,
            )
        except (JiraAgentError, KeyError, TypeError, ValueError) as error:
            self._record(
                base,
                dataset_id=dataset_id,
                epic_key=epic_key,
                disposition="ineligible",
                reason=str(error),
            )
            return

        try:
            result = self.agent.create_story(
                payload,
                epic_key=epic_key,
                approve_write=self.config.approve_write,
                allow_guardrail_fallback=True,
            )
        except (JiraAgentError, JiraClientError, KeyError, TypeError, ValueError) as error:
            self._record(
                base,
                dataset_id=dataset_id,
                epic_key=epic_key,
                alert_evidence_id=draft.alert_evidence_id,
                alert_label=draft.alert_label,
                disposition="failed",
                reason=str(error),
                error_type=type(error).__name__,
            )
            return

        result_payload = result.get("result", {})
        issue = result_payload.get("issue") or {}
        disposition = str(result_payload.get("status") or result.get("status") or "unknown")
        self._record(
            base,
            dataset_id=dataset_id,
            epic_key=epic_key,
            alert_evidence_id=draft.alert_evidence_id,
            alert_label=draft.alert_label,
            content_mode=draft.content_mode,
            disposition=disposition,
            jira_issue_key=issue.get("key"),
            agent_run_id=result.get("agent_run_id"),
            agent_audit_path=result.get("audit_path"),
            dry_run=result.get("dry_run"),
        )

    def monitor_once(self) -> dict[str, Any]:
        """Run one bounded, audited monitoring cycle over both dataset Epics."""
        LOGGER.info("jira_automatic_monitor_cycle_started")
        result = self.agent.monitor_epics(
            [self.config.epic_keys["ULB"], self.config.epic_keys["Sparkov"]],
            approve_write=self.config.approve_monitor_write,
        )
        LOGGER.info(
            "jira_automatic_monitor_cycle_completed agent_run_id=%s dry_run=%s",
            result.get("agent_run_id"),
            result.get("dry_run"),
        )
        return result


def build_jira_alert_trigger_from_environment(
    project_root: Path,
) -> JiraAlertRunTrigger | None:
    """Build the operational trigger only when it is explicitly enabled.

    Live writes require three persistent deployment gates:
    ``JIRA_AUTO_REFERRAL_ENABLED=true``, ``JIRA_ENABLE_WRITES=true``, and
    ``JIRA_AUTO_REFERRAL_APPROVED=true``. With only the first gate, events are
    processed automatically in dry-run mode.
    """
    if not _enabled("JIRA_AUTO_REFERRAL_ENABLED"):
        return None
    configure_application_logging(project_root)
    cloud = JiraCloudConfig.from_environment()
    epic_keys = {
        "ULB": os.environ.get("JIRA_ULB_EPIC_KEY", "").strip().upper(),
        "Sparkov": os.environ.get("JIRA_SPARKOV_EPIC_KEY", "").strip().upper(),
    }
    project_prefix = cloud.project_key + "-"
    if any(not key.startswith(project_prefix) for key in epic_keys.values()):
        raise JiraClientError(
            "JIRA_ULB_EPIC_KEY and JIRA_SPARKOV_EPIC_KEY must belong to JIRA_PROJECT_KEY."
        )
    writes_enabled = _enabled("JIRA_ENABLE_WRITES")
    approve_write = _enabled("JIRA_AUTO_REFERRAL_APPROVED")
    approve_monitor_write = _enabled("JIRA_AUTO_MONITOR_APPROVED")

    planner = None
    planner_engine = os.environ.get("JIRA_AUTO_PLANNER_ENGINE", "rule").strip().lower()
    if planner_engine == "qwen":
        from src.llm_provider import LocalQwenProvider, provider_is_available

        if not provider_is_available():
            raise JiraClientError(
                "JIRA_AUTO_PLANNER_ENGINE=qwen requires requirements-llm.txt."
            )
        planner = StructuredModelPlanner(LocalQwenProvider(
            local_files_only=_enabled("JIRA_AUTO_PLANNER_LOCAL_FILES_ONLY")
        ))
    elif planner_engine != "rule":
        raise JiraClientError("JIRA_AUTO_PLANNER_ENGINE must be rule or qwen.")

    LOGGER.info(
        "jira_alert_trigger_enabled project=%s live_writes=%s planner=%s",
        cloud.project_key,
        writes_enabled and approve_write,
        planner_engine,
    )
    return JiraAlertRunTrigger(
        agent=JiraReferralAgent(
            gateway=JiraCloudClient(cloud),
            config=JiraAgentConfig(
                project_key=cloud.project_key,
                writes_enabled=writes_enabled,
            ),
            audit_repository=JsonAgentAuditRepository(
                project_root / "artifacts" / "jira_agent" / "audit"
            ),
            planner=planner,
        ),
        config=JiraAutoReferralConfig(
            epic_keys=epic_keys,
            approve_write=approve_write,
            approve_monitor_write=approve_monitor_write,
        ),
        audit_repository=JsonTriggerAuditRepository(
            project_root / "artifacts" / "jira_agent" / "trigger_audit"
        ),
    )
