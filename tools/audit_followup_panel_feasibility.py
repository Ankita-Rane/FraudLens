"""Audit whether requested follow-up panels can be built without leakage or reuse."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import argparse
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    PROJECT_ROOT / "datasets" / "evaluation" / "followup_retrieval"
    / "panel_feasibility.json"
)
REQUESTED = {
    "ULB": {"total": 150, "fraud": 97, "non_fraud": 53},
    "Sparkov": {"total": 400, "fraud": 260, "non_fraud": 140},
}


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _threshold(path: Path) -> float:
    frame = pd.read_csv(path)
    rows = frame[frame["threshold_rule"] == "validation_tuned"]
    if len(rows) != 1:
        raise ValueError(f"Expected one validation_tuned threshold in {path}.")
    return float(rows.iloc[0]["threshold"])


def _exposed_source_ids(dataset_key: str) -> set[int]:
    exposed: set[int] = set()
    for panel in ("runtime", "pilot", "candidate", "followup_pilot"):
        path = (
            PROJECT_ROOT / "datasets" / dataset_key
            / f"{panel}_transaction_samples.csv"
        )
        if path.exists():
            values = pd.read_csv(path, usecols=["source_test_row_index"])
            exposed.update(values["source_test_row_index"].astype(int).tolist())
    return exposed


def audit_dataset(
    dataset_key: str,
    dataset_label: str,
    *,
    artifact_directory: Path | None = None,
) -> dict[str, Any]:
    artifact_directory = artifact_directory or PROJECT_ROOT / "artifacts" / dataset_key
    if not artifact_directory.is_absolute():
        artifact_directory = PROJECT_ROOT / artifact_directory
    artifact_directory = artifact_directory.resolve()
    prediction_path = artifact_directory / "test_predictions.csv"
    metric_path = artifact_directory / "test_metrics.csv"
    predictions = pd.read_csv(prediction_path)
    threshold = _threshold(metric_path)
    source_ids = (
        predictions["row_index"].astype(int)
        if "row_index" in predictions else pd.Series(range(len(predictions)), dtype="int64")
    )
    frame = predictions.assign(source_id=source_ids)
    frame["predicted_alert"] = frame["fraud_probability"].astype(float) >= threshold
    frame["stratum"] = "TN"
    frame.loc[frame["actual"].eq(1) & frame["predicted_alert"], "stratum"] = "TP"
    frame.loc[frame["actual"].eq(0) & frame["predicted_alert"], "stratum"] = "FP"
    frame.loc[frame["actual"].eq(1) & ~frame["predicted_alert"], "stratum"] = "FN"
    exposed = _exposed_source_ids(dataset_key)
    fresh = frame[~frame["source_id"].isin(exposed)]
    requested = REQUESTED[dataset_label]
    all_counts = frame["actual"].value_counts().to_dict()
    fresh_counts = fresh["actual"].value_counts().to_dict()
    alert = frame[frame["predicted_alert"]]
    fresh_alert = fresh[fresh["predicted_alert"]]
    transaction_feasible = (
        int(fresh_counts.get(1, 0)) >= requested["fraud"]
        and int(fresh_counts.get(0, 0)) >= requested["non_fraud"]
    )
    alert_feasible = (
        int((fresh_alert["actual"] == 1).sum()) >= requested["fraud"]
        and int((fresh_alert["actual"] == 0).sum()) >= requested["non_fraud"]
    )
    return {
        "dataset_id": dataset_label,
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "prediction_sha256": _sha256(prediction_path),
        "threshold_path": str(metric_path.relative_to(PROJECT_ROOT)),
        "threshold": threshold,
        "requested": requested,
        "heldout_rows": len(frame),
        "heldout_label_counts": {
            "fraud": int(all_counts.get(1, 0)),
            "non_fraud": int(all_counts.get(0, 0)),
        },
        "heldout_confusion_counts": {
            key: int(value) for key, value in frame["stratum"].value_counts().items()
        },
        "heldout_alert_counts": {
            "fraud": int((alert["actual"] == 1).sum()),
            "non_fraud": int((alert["actual"] == 0).sum()),
        },
        "excluded_previously_exposed_source_rows": len(exposed),
        "fresh_label_counts": {
            "fraud": int(fresh_counts.get(1, 0)),
            "non_fraud": int(fresh_counts.get(0, 0)),
        },
        "fresh_confusion_counts": {
            key: int(value) for key, value in fresh["stratum"].value_counts().items()
        },
        "fresh_alert_counts": {
            "fraud": int((fresh_alert["actual"] == 1).sum()),
            "non_fraud": int((fresh_alert["actual"] == 0).sum()),
        },
        "transaction_panel_feasible": transaction_feasible,
        "alert_only_panel_feasible": alert_feasible,
        "status": "feasible" if transaction_feasible else "infeasible_current_artifacts",
    }


def build_audit(
    *,
    ulb_artifact_directory: Path | None = None,
    sparkov_artifact_directory: Path | None = None,
) -> dict[str, Any]:
    datasets = [
        audit_dataset("ulb", "ULB", artifact_directory=ulb_artifact_directory),
        audit_dataset("sparkov", "Sparkov", artifact_directory=sparkov_artifact_directory),
    ]
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "pre_generation_feasibility_only",
        "sampling_frame_pending": "label_stratified_TP_FP_FN_TN_transactions",
        "ground_truth_generation_boundary": (
            "Labels are used only to form/evaluate the panel and must be removed from "
            "generation inputs."
        ),
        "datasets": datasets,
        "overall_status": (
            "feasible_current_artifacts"
            if all(row["transaction_panel_feasible"] for row in datasets)
            else "blocked_requires_new_leakage_safe_evidence"
        ),
        "blocking_reasons": [
            (
                f"{row['dataset_id']} has only {row['fresh_label_counts']['fraud']} "
                f"fresh held-out fraud rows but {row['requested']['fraud']} are requested."
            )
            for row in datasets if not row["transaction_panel_feasible"]
        ],
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ulb-artifact-directory", type=Path, default=None)
    parser.add_argument("--sparkov-artifact-directory", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    payload = build_audit(
        ulb_artifact_directory=args.ulb_artifact_directory,
        sparkov_artifact_directory=args.sparkov_artifact_directory,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
