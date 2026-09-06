"""Dataset-neutral evidence records for downstream LLM configurations.

ULB and Sparkov use incompatible predictor spaces. This module therefore combines
their outputs as nested, source-labelled evidence rather than joining feature columns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.0"


def _json_value(value: Any) -> Any:
    """Convert pandas/NumPy values into strict JSON-compatible values."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def build_alert_records(
    *,
    dataset_id: str,
    features: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    scores: Sequence[float],
    threshold: float,
    model_name: str,
    record_ids: Sequence[Any] | None = None,
    event_times: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Create source-labelled records for observations above a locked threshold.

    `actual_class` is retained for offline evaluation but is removed from the compact
    LLM context produced by :func:`build_unified_evidence`.
    """
    score_array = np.asarray(scores, dtype=float)
    target_array = np.asarray(y_true, dtype=int)
    if len(features) != len(score_array) or len(features) != len(target_array):
        raise ValueError("Features, labels, and scores must have equal lengths.")

    if record_ids is None:
        record_ids = features.index.tolist()
    if event_times is None:
        event_times = [None] * len(features)
    if len(record_ids) != len(features) or len(event_times) != len(features):
        raise ValueError("Record IDs and event times must align with features.")

    records: list[dict[str, Any]] = []
    for position in np.flatnonzero(score_array >= threshold):
        source_record_id = str(_json_value(record_ids[position]))
        evidence_id = f"model-output:{dataset_id.lower()}:{source_record_id}"
        feature_payload = {
            str(column): _json_value(value)
            for column, value in features.iloc[position].items()
        }
        records.append({
            "schema_version": SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "evidence_type": "fraud_model_alert",
            "dataset_id": dataset_id,
            "record_id": source_record_id,
            "event_time": _json_value(event_times[position]),
            "model_name": model_name,
            "fraud_probability": float(score_array[position]),
            "decision_threshold": float(threshold),
            "fraud_alert": 1,
            "actual_class": int(target_array[position]),
            "features": feature_payload,
        })
    return sorted(records, key=lambda item: item["fraud_probability"], reverse=True)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines artifact."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_dataset_artifacts(project_root: Path, dataset_key: str) -> dict[str, Any]:
    artifact_dir = project_root / "artifacts" / dataset_key
    required = {
        "alerts": artifact_dir / "llm_alert_evidence.jsonl",
        "metrics": artifact_dir / "test_metrics.csv",
        "manifest": artifact_dir / "run_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run both modelling notebooks before building unified evidence. Missing: "
            + ", ".join(missing)
        )

    alerts = read_jsonl(required["alerts"])
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    metrics = pd.read_csv(required["metrics"]).replace({np.nan: None})
    metric_records = metrics.to_dict(orient="records")
    for row in metric_records:
        split = str(row.get("split", "test")).lower()
        threshold_rule = str(row.get("threshold_rule", "unknown")).lower()
        row["evidence_id"] = (
            f"model-metrics:{manifest['dataset'].lower()}:{split}:{threshold_rule}"
        )
    return {
        "dataset_id": manifest["dataset"],
        "model_name": manifest["selected_model"],
        "selected_threshold": manifest["selected_threshold"],
        "primary_metric": manifest["primary_metric"],
        "test_metrics": metric_records,
        "run_manifest": manifest,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


def _llm_safe_alert(record: dict[str, Any]) -> dict[str, Any]:
    """Remove offline truth labels before supplying evidence to an LLM."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"actual_class", "schema_version"}
    }


def build_unified_evidence(
    project_root: Path,
    *,
    max_alerts_per_dataset: int = 25,
) -> dict[str, Any]:
    """Build one LLM-ready bundle while preserving dataset provenance."""
    if max_alerts_per_dataset < 1:
        raise ValueError("max_alerts_per_dataset must be at least 1.")

    datasets = [
        _read_dataset_artifacts(project_root, "ulb"),
        _read_dataset_artifacts(project_root, "sparkov"),
    ]
    historical_path = (
        project_root / "artifacts" / "historical" / "historical_profiles.json"
    )
    historical_profiles = (
        json.loads(historical_path.read_text(encoding="utf-8"))
        if historical_path.exists()
        else None
    )

    llm_datasets: list[dict[str, Any]] = []
    combined_alerts: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        alerts = dataset.pop("alerts")
        combined_alerts.extend(alerts)
        compact_alerts = [
            _llm_safe_alert(record)
            for record in alerts[:max_alerts_per_dataset]
        ]
        llm_datasets.append({**dataset, "top_alerts": compact_alerts})

        tuned_rows = [
            row for row in dataset["test_metrics"]
            if row.get("threshold_rule") == "validation_tuned"
        ]
        if tuned_rows:
            comparison_rows.append(tuned_rows[0])

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Shared model evidence for LLM configurations A, B, and C",
        "combination_policy": (
            "Study-level fusion only. Dataset-specific feature objects remain nested; "
            "ULB and Sparkov rows are never treated as a shared feature matrix."
        ),
        "interpretation_notes": {
            "ULB": (
                "Real anonymized benchmark. V1-V28 have undisclosed semantics and must "
                "not be assigned invented business meanings."
            ),
            "Sparkov": (
                "Synthetic behavioural data. It supports interpretable feature analysis "
                "but not claims of independent real-bank deployment validity."
            ),
        },
        "cross_dataset_test_comparison": comparison_rows,
        "datasets": llm_datasets,
        "historical_profiles": historical_profiles,
    }

    output_dir = project_root / "artifacts" / "unified"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "unified_model_evidence.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    write_jsonl(combined_alerts, output_dir / "unified_alerts.jsonl")
    return bundle


def load_unified_evidence(project_root: Path) -> dict[str, Any]:
    """Load the generated compact bundle."""
    path = project_root / "artifacts" / "unified" / "unified_model_evidence.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Unified evidence not found at {path}. Run tools/build_unified_evidence.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))
