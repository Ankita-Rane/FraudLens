from __future__ import annotations

from io import BytesIO
from unittest import TestCase
from unittest.mock import patch
from urllib.error import HTTPError

import json

from src.jira_client import JiraClientError, JiraCloudClient, JiraCloudConfig


class FakeResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = b"" if payload is None else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.payload


class JiraClientTests(TestCase):
    def client(self) -> JiraCloudClient:
        return JiraCloudClient(JiraCloudConfig(
            base_url="https://research.atlassian.net",
            email="researcher@example.com",
            api_token="secret-token",
            project_key="FIR",
        ))

    def test_config_repr_does_not_expose_api_token(self) -> None:
        rendered = repr(self.client().config)
        self.assertNotIn("secret-token", rendered)

    @patch("src.jira_client.urlopen")
    def test_create_issue_uses_only_fixed_rest_endpoint(self, mocked) -> None:
        mocked.return_value = FakeResponse({"id": "1", "key": "FIR-1"})
        result = self.client().create_issue({"summary": "Research case"})
        request = mocked.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.full_url, "https://research.atlassian.net/rest/api/3/issue")
        self.assertEqual(json.loads(request.data), {"fields": {"summary": "Research case"}})
        self.assertEqual(result["key"], "FIR-1")

    @patch("src.jira_client.urlopen")
    def test_search_paginates_with_next_page_token(self, mocked) -> None:
        mocked.side_effect = [
            FakeResponse({"issues": [{"key": "FIR-1"}], "nextPageToken": "next"}),
            FakeResponse({"issues": [{"key": "FIR-2"}], "isLast": True}),
        ]
        issues = self.client().search_issues("project = FIR", fields=("summary",))
        self.assertEqual([item["key"] for item in issues], ["FIR-1", "FIR-2"])
        self.assertIn("nextPageToken=next", mocked.call_args_list[1].args[0].full_url)

    @patch("src.jira_client.urlopen")
    def test_empty_success_response_is_supported_for_updates(self, mocked) -> None:
        mocked.return_value = FakeResponse()
        self.client().update_issue("FIR-1", {"labels": ["overdue"]})
        self.assertEqual(mocked.call_args.args[0].method, "PUT")

    @patch("src.jira_client.urlopen")
    def test_http_failure_is_bounded_and_does_not_echo_token(self, mocked) -> None:
        mocked.side_effect = HTTPError(
            "https://research.atlassian.net/rest/api/3/myself",
            403,
            "Forbidden",
            {},
            BytesIO(b'{"errorMessages":["Forbidden"]}'),
        )
        with self.assertRaises(JiraClientError) as raised:
            self.client().current_user()
        self.assertIn("HTTP 403", str(raised.exception))
        self.assertNotIn("secret-token", str(raised.exception))
        self.assertEqual(mocked.call_count, 1)

    @patch("src.jira_client.urlopen")
    def test_filter_lookup_requires_exact_name(self, mocked) -> None:
        mocked.return_value = FakeResponse({"values": [
            {"id": "1", "name": "Agentic referrals"},
            {"id": "2", "name": "Agentic referrals - open"},
        ]})
        result = self.client().find_filter("Agentic referrals - open")
        self.assertEqual(result["id"], "2")

    @patch("src.jira_client.urlopen")
    def test_project_and_issue_reads_use_fixed_endpoints(self, mocked) -> None:
        mocked.side_effect = [
            FakeResponse({"id": "10000", "key": "FIR"}),
            FakeResponse({"id": "10001", "key": "FIR-1", "fields": {}}),
        ]
        self.client().get_project("FIR")
        self.client().get_issue("FIR-1", fields=("labels", "issuetype"))
        self.assertIn("/rest/api/3/project/FIR", mocked.call_args_list[0].args[0].full_url)
        self.assertIn("/rest/api/3/issue/FIR-1", mocked.call_args_list[1].args[0].full_url)

    @patch("src.jira_client.urlopen")
    def test_changelog_read_is_paginated_and_bounded(self, mocked) -> None:
        mocked.side_effect = [
            FakeResponse({"values": [{"id": "1"}], "total": 2}),
            FakeResponse({"values": [{"id": "2"}], "total": 2}),
        ]
        values = self.client().get_issue_changelog("FIR-1")
        self.assertEqual([item["id"] for item in values], ["1", "2"])
        self.assertIn("/rest/api/3/issue/FIR-1/changelog", mocked.call_args.args[0].full_url)

    @patch("src.jira_client.urlopen")
    def test_kanban_board_creation_uses_agile_endpoint(self, mocked) -> None:
        mocked.return_value = FakeResponse({"id": 7, "name": "Research Kanban"})
        result = self.client().create_board({
            "name": "Research Kanban",
            "type": "kanban",
            "filterId": 42,
            "location": {"type": "project", "projectKeyOrId": "FIR"},
        })
        request = mocked.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            request.full_url,
            "https://research.atlassian.net/rest/agile/1.0/board",
        )
        self.assertEqual(result["id"], 7)

    def test_board_adapter_rejects_non_kanban_and_unknown_fields(self) -> None:
        with self.assertRaises(ValueError):
            self.client().create_board({"name": "Unsafe", "type": "scrum"})
        with self.assertRaises(ValueError):
            self.client().create_board({
                "name": "Unsafe", "type": "kanban", "deleteProject": True
            })
