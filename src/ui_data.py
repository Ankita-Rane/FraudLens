"""Read-only adapters from saved research evidence to the UI contract."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import csv
import json
import os


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _hash(path: Path) -> str | None:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def readiness(project_root: Path) -> dict[str, Any]:
    required = {
        "ulb_detector": "artifacts/ulb/selected_pipeline.joblib",
        "sparkov_detector": "artifacts/sparkov/selected_pipeline.joblib",
        "unified_evidence": "artifacts/unified/unified_model_evidence.json",
        "primary_results": "artifacts/results/followup/FOLLOWUP-RETRIEVAL-20260902-02-CORRECTED-V4.json",
        "protocol_lock": "datasets/evaluation/followup_retrieval/protocol_locked.json",
        "historical_profiles": "artifacts/historical/historical_profiles.json",
    }
    checks = {
        key: {"available": (project_root / relative).is_file(), "path": relative}
        for key, relative in required.items()
    }
    jira_required = ("JIRA_BASE_URL", "JIRA_PROJECT_KEY", "JIRA_USER_EMAIL", "JIRA_API_TOKEN")
    return {
        "status": "ready" if all(item["available"] for item in checks.values()) else "partial",
        "checks": checks,
        "jira": {
            "configured": all(os.environ.get(name, "").strip() for name in jira_required),
            "project_key": os.environ.get("JIRA_PROJECT_KEY", "").strip() or None,
            "base_url_host": (
                os.environ.get("JIRA_BASE_URL", "").replace("https://", "").split("/")[0]
                or None
            ),
            "credentials_exposed": False,
        },
        "scientific_boundaries": {
            "datasets_independent": True,
            "labels_withheld_from_generation": True,
            "jira_excluded_from_experiments": True,
        },
    }


def eda_and_models(project_root: Path) -> dict[str, Any]:
    profile_path = project_root / "artifacts/historical/historical_profiles.json"
    profiles = _json(profile_path) if profile_path.is_file() else {"datasets": []}
    datasets: list[dict[str, Any]] = []
    for dataset_id in ("ulb", "sparkov"):
        artifact_dir = project_root / "artifacts" / dataset_id
        manifest_path = artifact_dir / "run_manifest.json"
        manifest = _json(manifest_path) if manifest_path.is_file() else None
        metrics = _csv(artifact_dir / "test_metrics.csv")
        for row in metrics:
            for key, value in list(row.items()):
                if key not in {"dataset", "model", "split", "threshold_rule"}:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
        profile = next(
            (item for item in profiles.get("datasets", []) if str(item.get("dataset_id", "")).lower() == dataset_id),
            None,
        )
        datasets.append({
            "dataset_id": dataset_id.upper() if dataset_id == "ulb" else "Sparkov",
            "profile": profile,
            "run_manifest": manifest,
            "test_metrics": metrics,
            "attributions": {
                panel: {
                    "available": (artifact_dir / f"{panel}_sample_attributions.json").is_file(),
                    "path": f"artifacts/{dataset_id}/{panel}_sample_attributions.json",
                    "sha256": _hash(artifact_dir / f"{panel}_sample_attributions.json"),
                }
                for panel in ("runtime", "candidate", "pilot")
            },
            "interpretation_boundary": (
                "Detector metrics are dataset-specific and must not be pooled across ULB and Sparkov."
            ),
        })
    return {
        "generated_from_saved_artifacts": True,
        "profile_sha256": _hash(profile_path),
        "datasets": datasets,
    }


def evidence_registry(project_root: Path) -> dict[str, Any]:
    entries = [
        ("Original six-condition V4 results", "artifacts/results/followup/FOLLOWUP-RETRIEVAL-20260902-02-CORRECTED-V4.json", "primary evidence"),
        ("Locked six-condition protocol", "datasets/evaluation/followup_retrieval/protocol_locked.json", "protocol"),
        ("Known-item retrieval diagnostic", "datasets/evaluation/followup_retrieval/retrieval_diagnostic_results.json", "retrieval evidence"),
        ("Retrieval diagnostic audit", "datasets/evaluation/followup_retrieval/retrieval_diagnostic_audit.json", "verification"),
        ("Historical profiles", "artifacts/historical/historical_profiles.json", "EDA"),
        ("ULB model manifest", "artifacts/ulb/run_manifest.json", "detector"),
        ("Sparkov model manifest", "artifacts/sparkov/run_manifest.json", "detector"),
        ("Six-condition evidence ledger", "Six_Condition_Evidence_Ledger.html", "projection"),
    ]
    return {
        "entries": [
            {
                "name": name,
                "path": path,
                "classification": classification,
                "available": (project_root / path).is_file(),
                "sha256": _hash(project_root / path),
            }
            for name, path, classification in entries
        ],
        "warning": "HTML and spreadsheets are projections; JSON/CSV/manifests are the evidence sources.",
    }
