from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import json
import os

from src.jira_agent import (
    AGENT_LABEL,
    JiraAgentConfig,
    JiraAgentError,
    JiraReferralAgent,
    JsonAgentAuditRepository,
    StructuredModelPlanner,
    default_board_spec,
    default_epic_specs,
    default_filter_specs,
)
from src.jira_client import JiraCloudConfig, JiraClientError, normalize_cloud_url
from src.jira_trigger import (
    JiraAlertRunTrigger,
    JiraAutoReferralConfig,
    JsonTriggerAuditRepository,
    PostSaveRunRepository,
)


class FakeJiraGateway:
    def __init__(self) -> None:
        self.issues: list[dict] = []
        self.created_fields: list[dict] = []
        self.updated: list[tuple[str, dict]] = []
        self.comments: list[tuple[str, dict]] = []
        self.filters: list[dict] = []
        self.boards: list[dict] = []
        self.searches: list[str] = []

    def current_user(self) -> dict:
        return {"accountId": "account-1", "displayName": "Research User"}

    def get_project(self, project_key: str) -> dict:
        return {"id": "10000", "key": project_key, "name": "Fraud Research"}

    def get_issue(self, issue_key: str, *, fields=()) -> dict:
        return next(issue for issue in self.issues if issue.get("key") == issue_key)

    def search_issues(self, jql: str, *, fields=(), max_results=500) -> list[dict]:
        self.searches.append(jql)
        if "source-request-" in jql or "alert-evidence-" in jql:
            marker = jql.split('labels = "', 1)[1].split('"', 1)[0]
            return [
                issue for issue in self.issues
                if marker in (issue.get("fields") or {}).get("labels", [])
            ][:max_results]
        if "issuetype = Epic" in jql:
            dataset_label = "dataset-ulb" if "dataset-ulb" in jql else "dataset-sparkov"
            return [
                issue for issue in self.issues
                if issue.get("fields", {}).get("issuetype", {}).get("name") == "Epic"
                and dataset_label in issue.get("fields", {}).get("labels", [])
            ][:max_results]
        if "parent in" in jql:
            return [
                issue for issue in self.issues
                if issue.get("fields", {}).get("issuetype", {}).get("name") != "Epic"
                and AGENT_LABEL in issue.get("fields", {}).get("labels", [])
                if issue.get("fields", {}).get("status", {}).get("statusCategory", {}).get("key")
                != "done"
            ][:max_results]
        return []

    def create_issue(self, fields: dict) -> dict:
        key = f"FIR-{len(self.issues) + 1}"
        self.created_fields.append(fields)
        issue = {"id": str(len(self.created_fields)), "key": key, "fields": fields}
        self.issues.append(issue)
        return {"id": issue["id"], "key": key, "self": f"https://example/{key}"}

    def update_issue(self, issue_key: str, fields: dict) -> None:
        self.updated.append((issue_key, fields))
        for issue in self.issues:
            if issue.get("key") == issue_key:
                issue.setdefault("fields", {}).update(fields)

    def add_comment(self, issue_key: str, body: dict) -> dict:
        self.comments.append((issue_key, body))
        return {"id": str(len(self.comments))}

    def find_filter(self, name: str) -> dict | None:
        return next((item for item in self.filters if item["name"] == name), None)

    def create_filter(self, payload: dict) -> dict:
        record = {**payload, "id": str(len(self.filters) + 1)}
        self.filters.append(record)
        return record

    def find_boards(self, *, project_key: str, name: str) -> list[dict]:
        return [item for item in self.boards if item["name"] == name]

    def create_board(self, payload: dict) -> dict:
        record = {**payload, "id": len(self.boards) + 1}
        self.boards.append(record)
        return record

    def add_agent_epic(self, key: str, dataset_id: str) -> None:
        self.issues.append({
            "key": key,
            "fields": {
                "summary": f"{dataset_id} Epic",
                "issuetype": {"name": "Epic"},
                "labels": [AGENT_LABEL, f"dataset-{dataset_id.lower()}"],
                "status": {
                    "name": "Open",
                    "statusCategory": {"key": "new"},
                },
            },
        })


def valid_c_run(dataset_id: str = "ULB") -> dict:
    dataset_slug = dataset_id.lower()
    limitations = (
        ["ULB is real but anonymised; V1-V28 meanings are unknown."]
        if dataset_id == "ULB"
        else ["Sparkov is synthetic; it is not a real banking transaction."]
    )
    return {
        "request_id": "req-123",
        "configuration": "C",
        "status": "completed",
        "sample_id": f"CASE_{dataset_id}_001",
        "parsed_answer": {
            "answer": "The model alert warrants analyst review; it is not proof of fraud.",
            "risk_level": "high",
            "self_reported_confidence": None,
            "citations": [
                f"model-output:{dataset_slug}:sample:CASE_{dataset_id}_001",
                "policy:source:chunk-1",
            ],
            "claims": [{"statement": "Review is warranted.", "citations": ["policy:source:chunk-1"]}],
            "model_drivers": ["V14" if dataset_id == "ULB" else "amount"],
            "recommended_actions": ["Route to a human analyst."],
            "limitations": limitations,
        },
        "guardrail_checks": {
            "valid_json_object": True,
            "schema_contract": True,
            "citation_allowlist": True,
        },
        "guardrail_violations": [],
        "model_evidence": {
            "active_evaluation_sample": {"dataset_id": dataset_id},
            "datasets": [{
                "dataset_id": dataset_id,
                "model_name": "Random forest",
                "top_alerts": [{
                    "fraud_probability": 0.91,
                    "decision_threshold": 0.41,
                    "fraud_alert": 1,
                    "evidence_id": f"model-output:{dataset_slug}:sample:CASE_{dataset_id}_001",
                }],
            }],
        },
        "trace": {
            "prompt_sha256": "abc123",
            "engine_name": "deterministic-test",
        },
    }


class JiraAgentTests(TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.gateway = FakeJiraGateway()
        self.audit = JsonAgentAuditRepository(Path(self.temp.name) / "audit")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def agent(self, *, writes_enabled: bool = False, max_actions: int = 100) -> JiraReferralAgent:
        return JiraReferralAgent(
            gateway=self.gateway,
            config=JiraAgentConfig(
                project_key="FIR",
                writes_enabled=writes_enabled,
                max_actions_per_run=max_actions,
            ),
            audit_repository=self.audit,
        )

    def test_site_url_discards_welcome_query_and_requires_atlassian_cloud(self) -> None:
        self.assertEqual(
            normalize_cloud_url("https://fraud-research-example.atlassian.net?continue=welcome"),
            "https://fraud-research-example.atlassian.net",
        )
        with self.assertRaises(ValueError):
            normalize_cloud_url("http://fraud-research-example.atlassian.net")
        with self.assertRaises(ValueError):
            normalize_cloud_url("https://attacker.example")

    def test_post_save_repository_emits_event_only_after_persistence(self) -> None:
        order = []

        class Repository:
            def save(self, payload, request_id):
                order.append(("persisted", request_id, payload["configuration"]))
                return Path(self.temp_name) / f"{request_id}.json"

        Repository.temp_name = self.temp.name

        class Hook:
            def after_save(self, payload, run_path):
                order.append(("triggered", payload["request_id"], run_path.name))

        payload = {"request_id": "req-1", "configuration": "C"}
        path = PostSaveRunRepository(Repository(), (Hook(),)).save(payload, "req-1")

        self.assertEqual(path.name, "req-1.json")
        self.assertEqual([item[0] for item in order], ["persisted", "triggered"])

    def test_post_save_trigger_failure_does_not_invalidate_saved_run(self) -> None:
        class Repository:
            def save(self, payload, request_id):
                return Path(self.temp_name) / f"{request_id}.json"

        Repository.temp_name = self.temp.name

        class FailingHook:
            def after_save(self, payload, run_path):
                raise RuntimeError("simulated Jira outage")

        path = PostSaveRunRepository(Repository(), (FailingHook(),)).save(
            {"request_id": "req-2", "configuration": "C"},
            "req-2",
        )
        self.assertEqual(path.name, "req-2.json")

    def test_environment_config_fails_without_credentials(self) -> None:
        names = ("JIRA_BASE_URL", "JIRA_USER_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY")
        original = {name: os.environ.pop(name, None) for name in names}
        try:
            with self.assertRaises(JiraClientError):
                JiraCloudConfig.from_environment()
        finally:
            for name, value in original.items():
                if value is not None:
                    os.environ[name] = value

    def test_only_released_guarded_configuration_c_is_eligible(self) -> None:
        for mutation in (
            {"configuration": "B"},
            {"status": "blocked"},
            {"guardrail_checks": {}},
            {"guardrail_violations": ["citation_allowlist"]},
            {"parsed_answer": None},
        ):
            run = valid_c_run()
            run.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(JiraAgentError):
                self.agent().build_story_draft(run, epic_key="FIR-1")

    def test_offline_evaluation_label_is_rejected(self) -> None:
        run = valid_c_run()
        run["model_evidence"]["active_evaluation_sample"]["evaluation_only_actual_class"] = 1
        with self.assertRaisesRegex(JiraAgentError, "offline_label_absent"):
            self.agent().build_story_draft(run, epic_key="FIR-1")

    def test_non_alert_or_below_threshold_packet_is_rejected(self) -> None:
        for mutation in (
            {"fraud_alert": 0},
            {"fraud_probability": 0.40},
        ):
            run = valid_c_run()
            run["model_evidence"]["datasets"][0]["top_alerts"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                JiraAgentError, "active_detector_alert"
            ):
                self.agent().build_story_draft(run, epic_key="FIR-1")

    def test_story_keeps_ulb_and_sparkov_under_the_supplied_epic(self) -> None:
        ulb = self.agent().build_story_draft(valid_c_run("ULB"), epic_key="FIR-10", today=date(2026, 8, 30))
        sparkov = self.agent().build_story_draft(valid_c_run("Sparkov"), epic_key="FIR-20", today=date(2026, 8, 30))
        self.assertEqual(ulb.fields["parent"]["key"], "FIR-10")
        self.assertIn("dataset-ulb", ulb.fields["labels"])
        self.assertEqual(sparkov.fields["parent"]["key"], "FIR-20")
        self.assertIn("dataset-sparkov", sparkov.fields["labels"])
        self.assertEqual(ulb.fields["duedate"], "2026-09-06")

    def test_story_payload_contains_no_offline_label_or_fraud_certainty(self) -> None:
        draft = self.agent().build_story_draft(valid_c_run(), epic_key="FIR-1")
        serialized = json.dumps(draft.fields).lower()
        self.assertNotIn("evaluation_only_actual_class", serialized)
        self.assertIn("not proof of fraud", serialized)

    def test_write_requires_configuration_and_explicit_approval(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        first = self.agent(writes_enabled=False).create_story(
            valid_c_run(), epic_key="FIR-1", approve_write=True
        )
        second = self.agent(writes_enabled=True).create_story(
            valid_c_run(), epic_key="FIR-1", approve_write=False
        )
        self.assertEqual(first["result"]["status"], "draft")
        self.assertEqual(second["result"]["status"], "draft")
        self.assertEqual(self.gateway.created_fields, [])

    def test_approved_write_creates_story_once_and_prevents_duplicate(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        agent = self.agent(writes_enabled=True)
        first = agent.create_story(valid_c_run(), epic_key="FIR-1", approve_write=True)
        second = agent.create_story(valid_c_run(), epic_key="FIR-1", approve_write=True)
        self.assertEqual(first["result"]["status"], "created")
        self.assertEqual(second["result"]["status"], "existing")
        self.assertEqual(len(self.gateway.created_fields), 1)

    def test_different_c_requests_for_one_alert_create_only_one_story(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        agent = self.agent(writes_enabled=True)
        first_run = valid_c_run()
        second_run = valid_c_run()
        second_run["request_id"] = "req-456"

        first = agent.create_story(first_run, epic_key="FIR-1", approve_write=True)
        second = agent.create_story(second_run, epic_key="FIR-1", approve_write=True)

        self.assertEqual(first["result"]["status"], "created")
        self.assertEqual(second["result"]["status"], "existing")
        self.assertEqual(len(self.gateway.created_fields), 1)
        labels = self.gateway.created_fields[0]["labels"]
        self.assertEqual(
            len([label for label in labels if label.startswith("alert-evidence-")]),
            1,
        )

    def test_persisted_c_alert_automatically_routes_once_to_dataset_epic(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        self.gateway.add_agent_epic("FIR-2", "Sparkov")
        trigger_audit = JsonTriggerAuditRepository(Path(self.temp.name) / "trigger")
        trigger = JiraAlertRunTrigger(
            agent=self.agent(writes_enabled=True),
            config=JiraAutoReferralConfig(
                epic_keys={"ULB": "FIR-1", "Sparkov": "FIR-2"},
                approve_write=True,
            ),
            audit_repository=trigger_audit,
        )
        first = valid_c_run()
        second = valid_c_run()
        second["request_id"] = "another-c-request-for-the-same-alert"

        trigger.after_save(first, Path(self.temp.name) / "first.json")
        trigger.after_save(second, Path(self.temp.name) / "second.json")

        self.assertEqual(len(self.gateway.created_fields), 1)
        self.assertEqual(self.gateway.created_fields[0]["parent"]["key"], "FIR-1")
        audits = [json.loads(path.read_text()) for path in trigger_audit.directory.glob("*.json")]
        self.assertEqual({item["disposition"] for item in audits}, {"created", "existing"})

    def test_automatic_trigger_excludes_comparative_experiment_runs(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        run = valid_c_run()
        run["trace"]["experiment_id"] = "locked-study-batch"
        trigger_audit = JsonTriggerAuditRepository(Path(self.temp.name) / "trigger")
        trigger = JiraAlertRunTrigger(
            agent=self.agent(writes_enabled=True),
            config=JiraAutoReferralConfig(
                epic_keys={"ULB": "FIR-1", "Sparkov": "FIR-2"},
                approve_write=True,
            ),
            audit_repository=trigger_audit,
        )

        trigger.after_save(run, Path(self.temp.name) / "experiment.json")

        self.assertEqual(self.gateway.created_fields, [])
        audit = json.loads(next(trigger_audit.directory.glob("*.json")).read_text())
        self.assertEqual(audit["disposition"], "ignored_experiment_run")

    def test_manual_create_story_path_rejects_experiment_run_released_narrative(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        run = valid_c_run()
        run["trace"]["experiment_id"] = "locked-study-batch"
        agent = self.agent(writes_enabled=True)

        result = agent.create_story(run, epic_key="FIR-1", approve_write=True)

        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["stopped_reason"], "ignored_experiment_run")
        self.assertEqual(self.gateway.created_fields, [])
        self.assertEqual(self.gateway.searches, [])

    def test_manual_create_story_path_rejects_experiment_run_detector_only_fallback(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        run = valid_c_run()
        run.update({
            "status": "blocked",
            "candidate_response_text": "SENSITIVE BLOCKED CANDIDATE MUST NOT LEAK",
            "parsed_answer": None,
            "guardrail_checks": {"schema_contract": False},
            "guardrail_violations": ["schema_contract"],
            "human_review_reasons": ["schema_contract"],
        })
        run["trace"]["experiment_id"] = "locked-study-batch"
        agent = self.agent(writes_enabled=True)

        result = agent.create_story(
            run, epic_key="FIR-1", approve_write=True, allow_guardrail_fallback=True
        )

        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["stopped_reason"], "ignored_experiment_run")
        self.assertEqual(self.gateway.created_fields, [])
        self.assertEqual(self.gateway.searches, [])

    def test_build_referral_draft_rejects_experiment_run_released_narrative(self) -> None:
        run = valid_c_run()
        run["trace"]["experiment_id"] = "locked-study-batch"
        with self.assertRaisesRegex(JiraAgentError, "ignored_experiment_run"):
            self.agent().build_referral_draft(run, epic_key="FIR-1")

    def test_build_referral_draft_rejects_experiment_run_detector_only_fallback(self) -> None:
        run = valid_c_run()
        run.update({
            "status": "blocked",
            "candidate_response_text": "SENSITIVE BLOCKED CANDIDATE MUST NOT LEAK",
            "parsed_answer": None,
            "guardrail_checks": {"schema_contract": False},
            "guardrail_violations": ["schema_contract"],
            "human_review_reasons": ["schema_contract"],
        })
        run["trace"]["experiment_id"] = "locked-study-batch"
        with self.assertRaisesRegex(JiraAgentError, "ignored_experiment_run"):
            self.agent().build_referral_draft(
                run, epic_key="FIR-1", allow_guardrail_fallback=True
            )

    def test_blocked_c_alert_creates_detector_only_story_without_candidate_text(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        run = valid_c_run()
        run.update({
            "status": "blocked",
            "candidate_response_text": "SENSITIVE BLOCKED CANDIDATE MUST NOT LEAK",
            "parsed_answer": None,
            "guardrail_checks": {"schema_contract": False},
            "guardrail_violations": ["schema_contract"],
            "human_review_reasons": ["schema_contract"],
        })
        trigger_audit = JsonTriggerAuditRepository(Path(self.temp.name) / "trigger")
        trigger = JiraAlertRunTrigger(
            agent=self.agent(writes_enabled=True),
            config=JiraAutoReferralConfig(
                epic_keys={"ULB": "FIR-1", "Sparkov": "FIR-2"},
                approve_write=True,
            ),
            audit_repository=trigger_audit,
        )

        trigger.after_save(run, Path(self.temp.name) / "blocked.json")

        self.assertEqual(len(self.gateway.created_fields), 1)
        serialized = json.dumps(self.gateway.created_fields[0])
        self.assertNotIn("SENSITIVE BLOCKED CANDIDATE", serialized)
        self.assertIn("c-narrative-unavailable", self.gateway.created_fields[0]["labels"])
        audit = json.loads(next(trigger_audit.directory.glob("*.json")).read_text())
        self.assertEqual(audit["disposition"], "created")
        self.assertEqual(audit["content_mode"], "detector_only_fallback")

    def test_non_alert_c_packet_never_creates_fallback_story(self) -> None:
        run = valid_c_run()
        run["status"] = "suppressed"
        run["parsed_answer"] = None
        run["model_evidence"]["datasets"][0]["top_alerts"][0]["fraud_alert"] = 0
        trigger_audit = JsonTriggerAuditRepository(Path(self.temp.name) / "trigger")
        trigger = JiraAlertRunTrigger(
            agent=self.agent(writes_enabled=True),
            config=JiraAutoReferralConfig(
                epic_keys={"ULB": "FIR-1", "Sparkov": "FIR-2"},
                approve_write=True,
            ),
            audit_repository=trigger_audit,
        )

        trigger.after_save(run, Path(self.temp.name) / "non-alert.json")

        self.assertEqual(self.gateway.created_fields, [])
        audit = json.loads(next(trigger_audit.directory.glob("*.json")).read_text())
        self.assertEqual(audit["disposition"], "ineligible")

    def test_story_parent_must_match_dataset_epic(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        with self.assertRaisesRegex(JiraAgentError, "Sparkov Epic"):
            self.agent().create_story(valid_c_run("Sparkov"), epic_key="FIR-1")

    def test_epic_specs_explicitly_bound_dataset_provenance(self) -> None:
        specs = default_epic_specs()
        self.assertEqual({item.dataset_id for item in specs}, {"ULB", "Sparkov"})
        sparkov = next(item for item in specs if item.dataset_id == "Sparkov")
        self.assertIn("synthetic", sparkov.description.lower())

    def test_epic_bootstrap_is_dry_run_by_default(self) -> None:
        result = self.agent().ensure_epics(approve_write=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(
            {item["status"] for item in result["result"]["epics"].values()},
            {"draft"},
        )
        self.assertEqual(self.gateway.created_fields, [])

    def test_filter_specs_cover_open_stale_overdue_and_each_dataset(self) -> None:
        filters = default_filter_specs("FIR", 7)
        joined = "\n".join(item.jql for item in filters)
        self.assertIn("statusCategory != Done", joined)
        self.assertIn("created <= -7d", joined)
        self.assertIn("duedate < startOfDay()", joined)
        self.assertIn("dataset-ulb", joined)
        self.assertIn("dataset-sparkov", joined)
        self.assertIn("ORDER BY Rank ASC", filters[0].jql)

    def test_approved_filter_bootstrap_is_idempotent(self) -> None:
        agent = self.agent(writes_enabled=True)
        first = agent.ensure_filters(approve_write=True)
        second = agent.ensure_filters(approve_write=True)
        self.assertEqual({item["status"] for item in first["result"]["filters"]}, {"created"})
        self.assertEqual({item["status"] for item in second["result"]["filters"]}, {"existing"})
        self.assertEqual(len(self.gateway.filters), 5)

    def test_kanban_board_bootstrap_is_filter_backed_and_idempotent(self) -> None:
        agent = self.agent(writes_enabled=True)
        agent.ensure_filters(approve_write=True)
        first = agent.ensure_board(approve_write=True)
        second = agent.ensure_board(approve_write=True)
        self.assertEqual(first["result"]["status"], "created")
        self.assertEqual(second["result"]["status"], "existing")
        self.assertEqual(len(self.gateway.boards), 1)
        self.assertEqual(self.gateway.boards[0]["type"], "kanban")
        self.assertEqual(
            self.gateway.boards[0]["name"], default_board_spec().name
        )

    def test_board_reuses_created_filter_id_during_search_index_lag(self) -> None:
        agent = self.agent(writes_enabled=True)
        filters = agent.ensure_filters(approve_write=True)
        source_filter_id = next(
            item["id"]
            for item in filters["result"]["filters"]
            if item["name"] == default_board_spec().source_filter_name
        )
        self.gateway.find_filter = lambda _name: None

        result = agent.ensure_board(
            approve_write=True,
            source_filter_id=source_filter_id,
        )

        self.assertEqual(result["result"]["status"], "created")
        self.assertEqual(self.gateway.boards[0]["filterId"], int(source_filter_id))
        self.assertEqual(
            result["actions"][0]["action"], "reuse_source_filter_id"
        )

    def test_board_rejects_invalid_supplied_filter_id(self) -> None:
        with self.assertRaisesRegex(JiraAgentError, "positive Jira ID"):
            self.agent(writes_enabled=True).ensure_board(
                approve_write=True,
                source_filter_id="not-an-id",
            )

    def test_monitor_flags_open_age_overdue_due_soon_and_missing_due(self) -> None:
        self.gateway.issues = [
            {"key": "FIR-11", "fields": {
                "parent": {"key": "FIR-1"}, "created": "2026-08-20T10:00:00+00:00",
                "duedate": "2026-08-29", "labels": [AGENT_LABEL],
                "status": {"name": "Triage", "statusCategory": {"key": "indeterminate"}},
            }},
            {"key": "FIR-12", "fields": {
                "parent": {"key": "FIR-2"}, "created": "2026-08-29T10:00:00+00:00",
                "duedate": "2026-09-01", "labels": [AGENT_LABEL],
                "status": {"name": "Investigating", "statusCategory": {"key": "indeterminate"}},
            }},
            {"key": "FIR-13", "fields": {
                "parent": {"key": "FIR-2"}, "created": "2026-08-29T10:00:00+00:00",
                "duedate": None, "labels": [AGENT_LABEL],
                "status": {"name": "New", "statusCategory": {"key": "new"}},
            }},
        ]
        result = self.agent().monitor_epics(
            ["FIR-1", "FIR-2"],
            now=datetime(2026, 8, 30, 12, tzinfo=timezone.utc),
            validate_epic_parents=False,
        )
        summary = result["result"]
        self.assertEqual(summary["open_issue_count"], 3)
        self.assertEqual(summary["open_over_sla_count"], 1)
        self.assertEqual(summary["overdue_count"], 1)
        self.assertEqual(summary["missing_due_date_count"], 1)
        self.assertEqual(len(self.gateway.updated), 0)

    def test_monitor_write_adds_labels_and_audited_comments(self) -> None:
        self.gateway.issues = [{"key": "FIR-11", "fields": {
            "parent": {"key": "FIR-1"}, "created": "2026-08-20T10:00:00+00:00",
            "duedate": "2026-08-29", "labels": [AGENT_LABEL],
            "status": {"name": "Triage", "statusCategory": {"key": "indeterminate"}},
        }}]
        self.agent(writes_enabled=True).monitor_epics(
            ["FIR-1", "FIR-2"], approve_write=True,
            now=datetime(2026, 8, 30, 12, tzinfo=timezone.utc),
            validate_epic_parents=False,
        )
        self.assertEqual(len(self.gateway.updated), 1)
        self.assertEqual(len(self.gateway.comments), 1)
        labels = self.gateway.updated[0][1]["labels"]
        self.assertIn("open-over-7-days", labels)
        self.assertIn("overdue", labels)

    def test_monitor_respects_maximum_action_limit(self) -> None:
        self.gateway.issues = [
            {"key": f"FIR-{index}", "fields": {
                "parent": {"key": "FIR-1"}, "created": "2026-08-01T00:00:00+00:00",
                "duedate": None, "labels": [AGENT_LABEL],
                "status": {"name": "New", "statusCategory": {"key": "new"}},
            }} for index in range(10, 13)
        ]
        result = self.agent(writes_enabled=True, max_actions=1).monitor_epics(
            ["FIR-1", "FIR-2"], approve_write=True,
            now=datetime(2026, 8, 30, tzinfo=timezone.utc),
            validate_epic_parents=False,
        )
        self.assertEqual(len(self.gateway.updated), 1)
        self.assertEqual(result["stopped_reason"], "maximum_action_limit_reached")

    def test_agent_audit_is_append_only(self) -> None:
        result = self.agent().create_story(
            valid_c_run(), epic_key="FIR-1", validate_epic_parent=False
        )
        path = Path(result["audit_path"])
        self.assertTrue(path.is_file())
        with self.assertRaises(FileExistsError):
            self.audit.save(type("Run", (), {"agent_run_id": path.stem, "to_dict": lambda self: {}})())

    def test_monitor_validates_one_epic_for_each_dataset(self) -> None:
        self.gateway.add_agent_epic("FIR-1", "ULB")
        self.gateway.add_agent_epic("FIR-2", "Sparkov")
        result = self.agent().monitor_epics(["FIR-1", "FIR-2"])
        self.assertEqual(result["result"]["open_issue_count"], 0)
        with self.assertRaisesRegex(JiraAgentError, "one ULB Epic"):
            self.agent().monitor_epics(["FIR-1", "FIR-1"])

    def test_resolved_story_is_excluded_from_monitoring(self) -> None:
        self.gateway.issues = [{"key": "FIR-11", "fields": {
            "parent": {"key": "FIR-1"},
            "issuetype": {"name": "Story"},
            "created": "2026-08-01T00:00:00+00:00",
            "duedate": "2026-08-10",
            "labels": [AGENT_LABEL],
            "status": {"name": "Resolved", "statusCategory": {"key": "done"}},
        }}]
        result = self.agent().monitor_epics(
            ["FIR-1", "FIR-2"], validate_epic_parents=False
        )
        self.assertEqual(result["result"]["open_issue_count"], 0)


class FakePlannerGenerator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int = 120):
        self.prompts.append(prompt)
        return type("Generation", (), {"text": self.text})()


class StructuredModelPlannerTests(TestCase):
    def test_accepts_only_the_policy_eligible_action(self) -> None:
        generator = FakePlannerGenerator(
            '{"action":"create_story","rationale":"eligible"}'
        )
        planner = StructuredModelPlanner(generator)
        selected = planner.choose({
            "eligible_action": "create_story",
            "dataset_id": "ULB",
            "secret": "must-not-enter-prompt",
        })
        self.assertEqual(selected, "create_story")
        self.assertNotIn("must-not-enter-prompt", generator.prompts[0])
        self.assertEqual(planner.last_trace["outcome"], "create_story")
        self.assertIn("output_sha256", planner.last_trace)
        self.assertNotIn("rationale", planner.last_trace)

    def test_invalid_or_unauthorized_model_output_fails_closed(self) -> None:
        for output in (
            "not-json",
            '{"action":"delete_issue","rationale":"unsafe"}',
            '{"action":"flag_issue","rationale":"wrong action"}',
        ):
            with self.subTest(output=output):
                planner = StructuredModelPlanner(FakePlannerGenerator(output))
                self.assertEqual(
                    planner.choose({"eligible_action": "create_story"}),
                    "no_action",
                )

    def test_model_provider_failure_fails_closed_without_retry(self) -> None:
        class FailingGenerator:
            calls = 0

            def generate(self, prompt: str, *, max_new_tokens: int = 120):
                self.calls += 1
                raise RuntimeError("model unavailable")

        generator = FailingGenerator()
        planner = StructuredModelPlanner(generator)
        self.assertEqual(
            planner.choose({"eligible_action": "create_story"}), "no_action"
        )
        self.assertEqual(generator.calls, 1)
        self.assertEqual(planner.last_trace["failure_type"], "RuntimeError")
