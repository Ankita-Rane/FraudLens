from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.jira_analytics import calculate_jira_metrics


class JiraAnalyticsTests(unittest.TestCase):
    def test_mttr_uses_only_resolved_issues_and_reopens_are_counted(self) -> None:
        issues = [
            {"key": "FIR-1", "fields": {
                "summary": "Resolved", "created": "2026-01-01T00:00:00+00:00",
                "resolutiondate": "2026-01-03T00:00:00+00:00",
                "status": {"name": "Done", "statusCategory": {"key": "done"}},
            }},
            {"key": "FIR-2", "fields": {
                "summary": "Open", "created": "2026-01-01T00:00:00+00:00",
                "resolutiondate": None,
                "status": {"name": "In Progress", "statusCategory": {"key": "indeterminate"}},
            }},
        ]
        changelogs = {
            "FIR-1": [{"created": "2026-01-01T02:00:00+00:00", "items": [
                {"field": "status", "fromString": "To Do", "toString": "In Progress"}
            ]}],
            "FIR-2": [
                {"created": "2026-01-02T00:00:00+00:00", "items": [
                    {"field": "status", "fromString": "To Do", "toString": "Done"}
                ]},
                {"created": "2026-01-03T00:00:00+00:00", "items": [
                    {"field": "status", "fromString": "Done", "toString": "In Progress"}
                ]},
            ],
        }
        metrics = calculate_jira_metrics(
            issues, changelogs, now=datetime(2026, 1, 10, tzinfo=timezone.utc)
        )
        self.assertEqual(metrics["open_count"], 1)
        self.assertEqual(metrics["closed_count"], 1)
        self.assertEqual(metrics["mean_time_to_resolution_hours"], 48.0)
        self.assertEqual(metrics["resolved_duration_sample_size"], 1)
        self.assertEqual(metrics["reopen_event_count"], 1)
        self.assertTrue(metrics["definitions"]["open_age_excluded_from_mttr"])


if __name__ == "__main__":
    unittest.main()
