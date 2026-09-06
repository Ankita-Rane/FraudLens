"""Read-only Jira research-case lifecycle metrics with explicit definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Mapping, Sequence


DONE_NAMES = frozenset({"done", "closed", "resolved", "cancelled", "canceled"})


def _time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        # Jira commonly returns offsets such as +0000.
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _hours(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds() / 3600)


def _done(name: str | None, category: str | None = None) -> bool:
    return str(category or "").lower() == "done" or str(name or "").lower() in DONE_NAMES


def _status_events(histories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for history in histories:
        at = _time(str(history.get("created", "")))
        if at is None:
            continue
        for item in history.get("items", []):
            if str(item.get("field", "")).lower() != "status":
                continue
            events.append({
                "at": at,
                "from": item.get("fromString"),
                "to": item.get("toString"),
                "from_category": item.get("fromStatusCategory"),
                "to_category": item.get("toStatusCategory"),
            })
    return sorted(events, key=lambda item: item["at"])


def calculate_jira_metrics(
    issues: Sequence[Mapping[str, Any]],
    changelogs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    now: datetime | None = None,
    stale_days: int = 7,
) -> dict[str, Any]:
    """Calculate snapshot metrics; open ages are never included in MTTR."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    resolution_hours: list[float] = []
    acknowledgement_hours: list[float] = []
    for issue in issues:
        key = str(issue.get("key", ""))
        fields = issue.get("fields", {}) if isinstance(issue.get("fields"), Mapping) else {}
        created = _time(fields.get("created"))
        resolution = _time(fields.get("resolutiondate"))
        status = fields.get("status", {}) if isinstance(fields.get("status"), Mapping) else {}
        category = status.get("statusCategory", {}) if isinstance(status.get("statusCategory"), Mapping) else {}
        is_closed = bool(resolution) or _done(status.get("name"), category.get("key") or category.get("name"))
        events = _status_events(changelogs.get(key, []))
        acknowledgement = events[0]["at"] if events else None
        if created and acknowledgement:
            acknowledgement_hours.append(_hours(created, acknowledgement))
        inferred_done = next((event["at"] for event in events if _done(event["to"], event["to_category"])), None)
        resolved_at = resolution or inferred_done
        if created and resolved_at and is_closed:
            resolution_hours.append(_hours(created, resolved_at))
        reopen_count = sum(
            _done(event["from"], event["from_category"])
            and not _done(event["to"], event["to_category"])
            for event in events
        )
        age_hours = _hours(created, now) if created and not is_closed else None
        rows.append({
            "key": key,
            "summary": fields.get("summary"),
            "status": status.get("name"),
            "status_category": category.get("name") or category.get("key"),
            "created_at": created.isoformat() if created else None,
            "resolved_at": resolved_at.isoformat() if resolved_at else None,
            "open": not is_closed,
            "open_age_hours": round(age_hours, 3) if age_hours is not None else None,
            "stale": bool(age_hours is not None and age_hours >= stale_days * 24),
            "reopen_count": reopen_count,
        })
    open_rows = [row for row in rows if row["open"]]
    closed_rows = [row for row in rows if not row["open"]]
    new_last_7_days = sum(
        bool(
            _time(row["created_at"])
            and _hours(_time(row["created_at"]), now) <= 7 * 24
        )
        for row in rows
    )
    by_status: dict[str, int] = {}
    for row in rows:
        label = str(row["status"] or "Unknown")
        by_status[label] = by_status.get(label, 0) + 1
    return {
        "issue_count": len(rows),
        "open_count": len(open_rows),
        "closed_count": len(closed_rows),
        "new_last_7_days_count": new_last_7_days,
        "stale_open_count": sum(row["stale"] for row in open_rows),
        "reopened_issue_count": sum(row["reopen_count"] > 0 for row in rows),
        "reopen_event_count": sum(row["reopen_count"] for row in rows),
        "status_counts": by_status,
        "mean_time_to_acknowledgement_hours": round(mean(acknowledgement_hours), 3) if acknowledgement_hours else None,
        "median_time_to_acknowledgement_hours": round(median(acknowledgement_hours), 3) if acknowledgement_hours else None,
        "mean_time_to_resolution_hours": round(mean(resolution_hours), 3) if resolution_hours else None,
        "median_time_to_resolution_hours": round(median(resolution_hours), 3) if resolution_hours else None,
        "resolved_duration_sample_size": len(resolution_hours),
        "issues": rows,
        "definitions": {
            "open": "Current status category is not Done and no resolution timestamp exists.",
            "stale": f"Open for at least {stale_days} calendar days.",
            "new": "Created within the previous seven calendar days.",
            "mtta": "Creation to first recorded status transition.",
            "mttr": "Creation to resolution/first Done transition for resolved issues only.",
            "open_age_excluded_from_mttr": True,
        },
        "interpretation_boundary": (
            "Jira status measures research-case handling, not fraud truth or analyst decision quality."
        ),
    }
