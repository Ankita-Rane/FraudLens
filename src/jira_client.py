"""Minimal, auditable Jira Cloud REST adapter for the post-C agentic layer.

The adapter deliberately exposes only the operations required by the research
prototype.  It has no delete, transition, assignment, project-administration, or
arbitrary-request surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import base64
import json
import logging
import os
import re


LOGGER = logging.getLogger(__name__)
PROJECT_KEY_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
ISSUE_KEY_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*-\d+")


class JiraClientError(RuntimeError):
    """Raised when a constrained Jira operation cannot be completed."""


class JiraGateway(Protocol):
    """Allowlisted Jira operations consumed by the agentic service."""

    def current_user(self) -> dict[str, Any]: ...

    def get_project(self, project_key: str) -> dict[str, Any]: ...

    def get_issue(
        self, issue_key: str, *, fields: Sequence[str] = ()
    ) -> dict[str, Any]: ...

    def get_issue_changelog(
        self, issue_key: str, *, max_results: int = 500
    ) -> list[dict[str, Any]]: ...

    def search_issues(
        self, jql: str, *, fields: Sequence[str] = (), max_results: int = 500
    ) -> list[dict[str, Any]]: ...

    def create_issue(self, fields: dict[str, Any]) -> dict[str, Any]: ...

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None: ...

    def add_comment(self, issue_key: str, body: dict[str, Any]) -> dict[str, Any]: ...

    def find_filter(self, name: str) -> dict[str, Any] | None: ...

    def create_filter(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def find_boards(
        self, *, project_key: str, name: str
    ) -> list[dict[str, Any]]: ...

    def create_board(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def normalize_cloud_url(value: str) -> str:
    """Return an HTTPS Jira Cloud origin and discard welcome/query fragments."""
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("JIRA_BASE_URL must be an absolute HTTPS URL.")
    hostname = (parsed.hostname or "").lower()
    if not hostname.endswith(".atlassian.net"):
        raise ValueError("JIRA_BASE_URL must use an atlassian.net Jira Cloud site.")
    return urlunsplit(("https", parsed.netloc, "", "", "")).rstrip("/")


@dataclass(frozen=True)
class JiraCloudConfig:
    base_url: str
    email: str
    api_token: str = field(repr=False)
    project_key: str

    @classmethod
    def from_environment(cls) -> "JiraCloudConfig":
        missing = [
            name
            for name in (
                "JIRA_BASE_URL",
                "JIRA_USER_EMAIL",
                "JIRA_API_TOKEN",
                "JIRA_PROJECT_KEY",
            )
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            raise JiraClientError(
                "Missing Jira environment variables: " + ", ".join(missing)
            )
        project_key = os.environ["JIRA_PROJECT_KEY"].strip().upper()
        if not project_key.replace("_", "").isalnum():
            raise JiraClientError("JIRA_PROJECT_KEY contains unsupported characters.")
        return cls(
            base_url=normalize_cloud_url(os.environ["JIRA_BASE_URL"]),
            email=os.environ["JIRA_USER_EMAIL"].strip(),
            api_token=os.environ["JIRA_API_TOKEN"].strip(),
            project_key=project_key,
        )


class JiraCloudClient:
    """Small Jira Cloud client with a fixed operation surface and no secret logging."""

    def __init__(self, config: JiraCloudConfig, *, timeout_seconds: float = 30.0) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        credentials = f"{config.email}:{config.api_token}".encode("utf-8")
        self._authorization = "Basic " + base64.b64encode(credentials).decode("ascii")

    def _redact(self, value: str) -> str:
        redacted = value
        for secret in (self.config.api_token, self.config.email, self._authorization):
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        if query:
            url += "?" + urlencode(query, doseq=True)
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = Request(
            url,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": self._authorization,
                "User-Agent": "thesis-bounded-jira-agent/1.0",
            },
        )
        LOGGER.info("jira_request_started method=%s path=%s", method, path)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as error:
            detail = self._redact(
                error.read().decode("utf-8", errors="replace")[:1_000]
            )
            LOGGER.error(
                "jira_request_failed method=%s path=%s http_status=%d",
                method,
                path,
                error.code,
            )
            raise JiraClientError(
                f"Jira {method} {path} failed with HTTP {error.code}: {detail}"
            ) from error
        except URLError as error:
            LOGGER.error(
                "jira_request_connection_failed method=%s path=%s error_type=%s",
                method,
                path,
                type(error.reason).__name__,
            )
            raise JiraClientError(
                f"Jira {method} {path} could not connect: {error.reason}"
            ) from error
        LOGGER.info(
            "jira_request_completed method=%s path=%s response_bytes=%d",
            method,
            path,
            len(raw),
        )
        if not raw:
            return {}
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise JiraClientError(
                f"Jira {method} {path} returned non-JSON content."
            ) from error
        if not isinstance(decoded, dict):
            raise JiraClientError(f"Jira {method} {path} returned an unexpected shape.")
        return decoded

    def current_user(self) -> dict[str, Any]:
        return self._request("GET", "/rest/api/3/myself")

    def get_project(self, project_key: str) -> dict[str, Any]:
        if not PROJECT_KEY_PATTERN.fullmatch(project_key):
            raise ValueError("Invalid Jira project key.")
        return self._request(
            "GET",
            f"/rest/api/3/project/{project_key}",
            query={"expand": "description,lead,projectKeys"},
        )

    def get_issue(
        self, issue_key: str, *, fields: Sequence[str] = ()
    ) -> dict[str, Any]:
        if not ISSUE_KEY_PATTERN.fullmatch(issue_key):
            raise ValueError("Invalid Jira issue key.")
        query = {"fields": ",".join(fields)} if fields else None
        return self._request("GET", f"/rest/api/3/issue/{issue_key}", query=query)

    def get_issue_changelog(
        self, issue_key: str, *, max_results: int = 500
    ) -> list[dict[str, Any]]:
        if not ISSUE_KEY_PATTERN.fullmatch(issue_key):
            raise ValueError("Invalid Jira issue key.")
        if max_results < 1 or max_results > 5_000:
            raise ValueError("max_results must be between 1 and 5000.")
        values: list[dict[str, Any]] = []
        start_at = 0
        while len(values) < max_results:
            page_size = min(100, max_results - len(values))
            page = self._request(
                "GET",
                f"/rest/api/3/issue/{issue_key}/changelog",
                query={"startAt": start_at, "maxResults": page_size},
            )
            page_values = page.get("values", [])
            if not isinstance(page_values, list):
                raise JiraClientError("Jira changelog response has no values array.")
            values.extend(item for item in page_values if isinstance(item, dict))
            start_at += len(page_values)
            total = page.get("total")
            if not page_values or (isinstance(total, int) and start_at >= total):
                break
        return values[:max_results]

    def search_issues(
        self, jql: str, *, fields: Sequence[str] = (), max_results: int = 500
    ) -> list[dict[str, Any]]:
        if max_results < 1:
            raise ValueError("max_results must be positive.")
        issues: list[dict[str, Any]] = []
        next_page_token: str | None = None
        while len(issues) < max_results:
            query: dict[str, Any] = {
                "jql": jql,
                "maxResults": min(100, max_results - len(issues)),
            }
            if fields:
                query["fields"] = ",".join(fields)
            if next_page_token:
                query["nextPageToken"] = next_page_token
            page = self._request("GET", "/rest/api/3/search/jql", query=query)
            page_issues = page.get("issues", [])
            if not isinstance(page_issues, list):
                raise JiraClientError("Jira search response has no issue array.")
            issues.extend(item for item in page_issues if isinstance(item, dict))
            next_page_token = page.get("nextPageToken")
            if page.get("isLast") is True or not next_page_token or not page_issues:
                break
        return issues[:max_results]

    def create_issue(self, fields: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/rest/api/3/issue", payload={"fields": fields})

    def update_issue(self, issue_key: str, fields: dict[str, Any]) -> None:
        if not ISSUE_KEY_PATTERN.fullmatch(issue_key):
            raise ValueError("Invalid Jira issue key.")
        self._request(
            "PUT", f"/rest/api/3/issue/{issue_key}", payload={"fields": fields}
        )

    def add_comment(self, issue_key: str, body: dict[str, Any]) -> dict[str, Any]:
        if not ISSUE_KEY_PATTERN.fullmatch(issue_key):
            raise ValueError("Invalid Jira issue key.")
        return self._request(
            "POST", f"/rest/api/3/issue/{issue_key}/comment", payload={"body": body}
        )

    def find_filter(self, name: str) -> dict[str, Any] | None:
        response = self._request(
            "GET", "/rest/api/3/filter/search", query={"filterName": name, "maxResults": 50}
        )
        values = response.get("values", [])
        return next(
            (item for item in values if item.get("name") == name),
            None,
        )

    def create_filter(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "description", "jql", "favourite"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unsupported Jira filter fields: {sorted(unknown)}")
        return self._request("POST", "/rest/api/3/filter", payload=payload)

    def find_boards(
        self, *, project_key: str, name: str
    ) -> list[dict[str, Any]]:
        response = self._request(
            "GET",
            "/rest/agile/1.0/board",
            query={
                "projectKeyOrId": project_key,
                "type": "kanban",
                "name": name,
                "maxResults": 50,
            },
        )
        values = response.get("values", [])
        if not isinstance(values, list):
            raise JiraClientError("Jira board search response has no values array.")
        return [
            item
            for item in values
            if isinstance(item, dict) and item.get("name") == name
        ]

    def create_board(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {"name", "type", "filterId", "location"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unsupported Jira board fields: {sorted(unknown)}")
        if payload.get("type") != "kanban":
            raise ValueError("The bounded agent may create only a Kanban board.")
        return self._request("POST", "/rest/agile/1.0/board", payload=payload)
