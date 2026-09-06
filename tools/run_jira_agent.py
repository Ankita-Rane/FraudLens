#!/usr/bin/env python3
"""Operate the bounded post-C Jira referral agent.

Writes require all three conditions: Jira credentials, JIRA_ENABLE_WRITES=true, and an
explicit --approve-write flag.  Commands log to log/application.log while omitting
credentials and raw authorization headers.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import json
import logging
import os
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jira_agent import (  # noqa: E402
    JiraAgentConfig,
    JiraAgentError,
    JiraReferralAgent,
    JsonAgentAuditRepository,
    StructuredModelPlanner,
    default_board_spec,
    default_epic_specs,
    default_filter_specs,
)
from src.jira_client import JiraCloudClient, JiraCloudConfig, JiraClientError  # noqa: E402


LOG_PATH = PROJECT_ROOT / "log" / "application.log"
AUDIT_DIRECTORY = PROJECT_ROOT / "artifacts" / "jira_agent" / "audit"


def configure_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == LOG_PATH
        for handler in root_logger.handlers
    ):
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        root_logger.addHandler(handler)
    return logging.getLogger("run_jira_agent")


LOGGER = configure_logging()


class OfflineGateway:
    """No-network gateway used only to build a local Story draft."""

    def current_user(self) -> dict[str, Any]:
        return {"accountId": "offline", "displayName": "Offline dry run"}

    def get_project(self, project_key: str) -> dict[str, Any]:
        return {"id": "offline", "key": project_key, "name": "Offline plan"}

    def get_issue(self, issue_key: str, *, fields=()) -> dict[str, Any]:
        raise AssertionError("OfflineGateway cannot validate a Jira Epic.")

    def search_issues(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    def create_issue(self, fields: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("OfflineGateway cannot create issues.")

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None:
        raise AssertionError("OfflineGateway cannot update issues.")

    def add_comment(self, issue_key: str, body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("OfflineGateway cannot add comments.")

    def find_filter(self, name: str) -> dict[str, Any] | None:
        return None

    def create_filter(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("OfflineGateway cannot create filters.")

    def find_boards(self, *, project_key: str, name: str) -> list[dict[str, Any]]:
        return []

    def create_board(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("OfflineGateway cannot create boards.")


def parser() -> ArgumentParser:
    root = ArgumentParser(description=__doc__)
    root.add_argument(
        "--project-key",
        default=os.environ.get("JIRA_PROJECT_KEY", "FIR"),
        help="Jira project key; defaults to JIRA_PROJECT_KEY or FIR for offline plans.",
    )
    root.add_argument("--stale-days", type=int, default=7)
    root.add_argument("--due-soon-days", type=int, default=2)
    root.add_argument("--story-due-days", type=int, default=7)
    root.add_argument(
        "--planner-engine",
        choices=("rule", "qwen"),
        default="rule",
        help=(
            "Bounded action planner. 'rule' is deterministic; 'qwen' asks the local "
            "schema-constrained model and fails closed on invalid output."
        ),
    )
    root.add_argument("--planner-model-name", default="Qwen/Qwen3-0.6B")
    root.add_argument("--planner-model-revision")
    root.add_argument("--planner-local-files-only", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Print the offline Epic, board-column, and filter plan.")
    commands.add_parser("doctor", help="Verify Jira credentials with a read-only call.")
    commands.add_parser(
        "automation-doctor",
        help="Read-only validation of automatic referral gates and dataset Epic routes.",
    )
    bootstrap = commands.add_parser("bootstrap", help="Ensure the two Epics and saved filters.")
    bootstrap.add_argument("--approve-write", action="store_true")
    board = commands.add_parser(
        "ensure-board",
        help="Ensure only the Kanban board using a verified saved-filter ID.",
    )
    board.add_argument("--source-filter-id", required=True)
    board.add_argument("--approve-write", action="store_true")
    draft = commands.add_parser("draft-story", help="Build a local Jira Story from a C run.")
    draft.add_argument("--run-file", type=Path, required=True)
    draft.add_argument("--epic-key", required=True)
    create = commands.add_parser("create-story", help="Create or find an idempotent Jira Story.")
    create.add_argument("--run-file", type=Path, required=True)
    create.add_argument("--epic-key", required=True)
    create.add_argument("--approve-write", action="store_true")
    monitor = commands.add_parser("monitor", help="Check all unresolved Stories below the two Epics.")
    monitor.add_argument("--ulb-epic-key", required=True)
    monitor.add_argument("--sparkov-epic-key", required=True)
    monitor.add_argument("--approve-write", action="store_true")
    watch = commands.add_parser("watch", help="Run bounded repeated Epic monitoring cycles.")
    watch.add_argument("--ulb-epic-key", required=True)
    watch.add_argument("--sparkov-epic-key", required=True)
    watch.add_argument("--cycles", type=int, default=1)
    watch.add_argument("--interval-minutes", type=int, default=60)
    watch.add_argument("--approve-write", action="store_true")
    return root


def writes_enabled() -> bool:
    return os.environ.get("JIRA_ENABLE_WRITES", "").strip().lower() == "true"


def agent_config(args: Namespace, *, force_dry_run: bool = False) -> JiraAgentConfig:
    return JiraAgentConfig(
        project_key=args.project_key.upper(),
        writes_enabled=False if force_dry_run else writes_enabled(),
        stale_days=args.stale_days,
        due_soon_days=args.due_soon_days,
        story_due_days=args.story_due_days,
    )


def action_planner(args: Namespace):
    if args.planner_engine == "rule":
        return None
    from src.llm_provider import LocalQwenProvider, provider_is_available

    if not provider_is_available():
        raise JiraAgentError(
            "The qwen planner requires the optional packages in requirements-llm.txt."
        )
    generator = LocalQwenProvider(
        model_name=args.planner_model_name,
        revision=args.planner_model_revision,
        local_files_only=args.planner_local_files_only,
    )
    return StructuredModelPlanner(generator)


def connected_agent(args: Namespace, *, planner_required: bool = False) -> JiraReferralAgent:
    cloud = JiraCloudConfig.from_environment()
    if cloud.project_key != args.project_key.upper():
        raise JiraClientError(
            "--project-key and JIRA_PROJECT_KEY must identify the same project."
        )
    return JiraReferralAgent(
        gateway=JiraCloudClient(cloud),
        config=agent_config(args),
        audit_repository=JsonAgentAuditRepository(AUDIT_DIRECTORY),
        planner=action_planner(args) if planner_required else None,
    )


def offline_agent(args: Namespace) -> JiraReferralAgent:
    return JiraReferralAgent(
        gateway=OfflineGateway(),
        config=agent_config(args, force_dry_run=True),
        audit_repository=JsonAgentAuditRepository(AUDIT_DIRECTORY),
        planner=None,
    )


def read_run(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JiraAgentError("Run file must contain one JSON object.")
    return payload


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def offline_plan(args: Namespace) -> dict[str, Any]:
    return {
        "status": "offline_plan",
        "project_key": args.project_key.upper(),
        "jira_site": "Read from JIRA_BASE_URL only during connected commands.",
        "board": {
            "specification": asdict(default_board_spec()),
            "recommended_template": "Kanban",
            "columns": ["New", "Triage", "Investigating", "Waiting for Evidence", "Resolved"],
            "free_plan_boundary": (
                "Use Epic/Story due dates and filters; Jira Premium Plans are not required."
            ),
        },
        "epics": [asdict(item) for item in default_epic_specs()],
        "filters": [
            asdict(item)
            for item in default_filter_specs(args.project_key, args.stale_days)
        ],
        "agent_tasks": [
            "Create a detailed research Story from an eligible Configuration C packet.",
            "Check both Epics and flag open-over-SLA, overdue, due-soon, or missing-due issues.",
        ],
        "write_gate": "JIRA_ENABLE_WRITES=true plus --approve-write",
        "planner": {
            "selected_engine": args.planner_engine,
            "rule": "Deterministic bounded reference planner.",
            "qwen": (
                "Optional local model planner; only an exactly eligible allowlisted "
                "action is accepted, otherwise the action becomes no_action."
            ),
        },
    }


def main() -> None:
    args = parser().parse_args()
    LOGGER.info(
        "jira_agent_command_started command=%s project_key=%s approve_write=%s planner=%s",
        args.command,
        args.project_key.upper(),
        bool(getattr(args, "approve_write", False)),
        args.planner_engine,
    )
    if args.command == "plan":
        print_json(offline_plan(args))
        LOGGER.info("jira_agent_command_completed command=plan")
        return
    if args.command == "draft-story":
        payload = offline_agent(args).create_story(
            read_run(args.run_file),
            epic_key=args.epic_key,
            approve_write=False,
            validate_epic_parent=False,
        )
        print_json(payload)
        LOGGER.info("jira_agent_command_completed command=draft-story")
        return
    agent = connected_agent(
        args,
        planner_required=args.command in {"create-story", "monitor", "watch"},
    )
    if args.command == "doctor":
        print_json(agent.doctor())
    elif args.command == "automation-doctor":
        from src.jira_trigger import build_jira_alert_trigger_from_environment

        trigger = build_jira_alert_trigger_from_environment(PROJECT_ROOT)
        if trigger is None:
            raise JiraAgentError(
                "JIRA_AUTO_REFERRAL_ENABLED must be true for automation-doctor."
            )
        epics: dict[str, dict[str, Any]] = {}
        for dataset_id, epic_key in trigger.config.epic_keys.items():
            issue = trigger.agent.gateway.get_issue(
                epic_key, fields=("issuetype", "labels", "summary")
            )
            fields = issue.get("fields") or {}
            labels = set(fields.get("labels") or [])
            expected_dataset_label = f"dataset-{dataset_id.lower()}"
            valid = (
                (fields.get("issuetype") or {}).get("name")
                == trigger.agent.config.epic_issue_type
                and "agentic-referral" in labels
                and expected_dataset_label in labels
            )
            epics[dataset_id] = {
                "key": epic_key,
                "summary": fields.get("summary"),
                "valid_agent_epic": valid,
                "required_labels_present": valid,
            }
        if not all(item["valid_agent_epic"] for item in epics.values()):
            raise JiraAgentError(
                "Automatic referral Epic validation failed; no writes were attempted."
            )
        print_json({
            "status": "ready",
            "read_only": True,
            "project_key": trigger.agent.config.project_key,
            "epics": epics,
            "referral_mode": (
                "live" if trigger.agent.config.writes_enabled
                and trigger.config.approve_write else "dry_run"
            ),
            "monitor_mode": (
                "live" if trigger.agent.config.writes_enabled
                and trigger.config.approve_monitor_write else "dry_run"
            ),
            "experiment_runs_excluded": trigger.config.exclude_experiment_runs,
            "idempotency": "one Story per detector evidence ID",
        })
    elif args.command == "bootstrap":
        epics = agent.ensure_epics(approve_write=args.approve_write)
        filters = agent.ensure_filters(approve_write=args.approve_write)
        source_filter_id = next(
            (
                item.get("id")
                for item in filters.get("result", {}).get("filters", [])
                if item.get("name") == default_board_spec().source_filter_name
            ),
            None,
        )
        print_json({
            "epics": epics,
            "filters": filters,
            "board": agent.ensure_board(
                approve_write=args.approve_write,
                source_filter_id=source_filter_id,
            ),
        })
    elif args.command == "ensure-board":
        print_json(agent.ensure_board(
            approve_write=args.approve_write,
            source_filter_id=args.source_filter_id,
        ))
    elif args.command == "create-story":
        print_json(agent.create_story(
            read_run(args.run_file),
            epic_key=args.epic_key,
            approve_write=args.approve_write,
        ))
    elif args.command == "monitor":
        print_json(agent.monitor_epics(
            [args.ulb_epic_key, args.sparkov_epic_key],
            approve_write=args.approve_write,
        ))
    elif args.command == "watch":
        if args.cycles < 1 or args.cycles > 1_000:
            raise JiraAgentError("--cycles must be between 1 and 1000.")
        if args.interval_minutes < 1:
            raise JiraAgentError("--interval-minutes must be positive.")
        for cycle in range(1, args.cycles + 1):
            LOGGER.info("jira_monitor_cycle_started cycle=%d total=%d", cycle, args.cycles)
            print_json(agent.monitor_epics(
                [args.ulb_epic_key, args.sparkov_epic_key],
                approve_write=args.approve_write,
            ))
            if cycle < args.cycles:
                time.sleep(args.interval_minutes * 60)
    LOGGER.info("jira_agent_command_completed command=%s", args.command)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, JiraAgentError, JiraClientError, ValueError) as error:
        LOGGER.exception("jira_agent_command_failed error_type=%s", type(error).__name__)
        raise SystemExit(str(error)) from error
