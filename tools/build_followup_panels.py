"""Build label-free follow-up transaction panels and evaluator-only labels.

The 30-case pilot is non-reportable and validates orchestration only.  The final mode
enforces the confirmed 150/400 and 65/35-like quotas and refuses infeasible inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import joblib
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_pilot_sample_panels import _sparkov_engineered


FINAL_QUOTAS = {
    "ULB": {1: 97, 0: 53},
    "Sparkov": {1: 260, 0: 140},
}
PILOT_QUOTAS = {
    "ULB": {1: 10, 0: 5},
    "Sparkov": {1: 10, 0: 5},
}


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _threshold(path: Path) -> float:
    rows = pd.read_csv(path)
    selected = rows[rows["threshold_rule"] == "validation_tuned"]
    if len(selected) != 1:
        raise ValueError(f"Expected one validation_tuned threshold in {path}.")
    return float(selected.iloc[0]["threshold"])


def _exposed_ids(dataset_key: str, *, include_pilot: bool) -> set[int]:
    names = ["runtime", "pilot", "candidate"]
    if include_pilot:
        names.append("followup_pilot")
    exposed: set[int] = set()
    represented: set[str] = set()
    selector_manifest = (
        PROJECT_ROOT / "datasets" / "evaluation" / "dataset_fixture_selection_manifest.json"
    )
    if selector_manifest.exists():
        manifest = json.loads(selector_manifest.read_text(encoding="utf-8"))
        for fixture in manifest.get("fixtures", []):
            fixture_path = Path(fixture["path"])
            if fixture_path.parent.name != dataset_key:
                continue
            for name in names:
                if fixture_path.name == f"{name}_transaction_samples.csv":
                    exposed.update(
                        int(row["source_test_row_index"])
                        for row in fixture.get("rows", [])
                    )
                    represented.add(name)
    for name in names:
        if name in represented:
            continue
        path = PROJECT_ROOT / "datasets" / dataset_key / f"{name}_transaction_samples.csv"
        if path.exists():
            frame = pd.read_csv(path, usecols=["source_test_row_index"])
            exposed.update(frame["source_test_row_index"].astype(int).tolist())
    return exposed


def select_label_stratified_positions(
    predictions: pd.DataFrame,
    *,
    quotas: dict[int, int],
    source_id_column: str | None,
    excluded_ids: set[int],
) -> list[int]:
    required = {"actual", "fraud_probability"}
    if missing := required - set(predictions):
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["source_position"] = np.arange(len(frame), dtype=int)
    frame["source_id"] = (
        frame[source_id_column].astype(int)
        if source_id_column else frame["source_position"]
    )
    frame = frame[~frame["source_id"].isin(excluded_ids)]
    positions: list[int] = []
    for label, count in sorted(quotas.items(), reverse=True):
        candidates = frame[frame["actual"].astype(int).eq(label)].sort_values(
            ["fraud_probability", "source_id"], kind="stable"
        )
        if len(candidates) < count:
            raise ValueError(
                f"Label {label} needs {count} fresh cases but only {len(candidates)} exist."
            )
        ranks = np.linspace(0, len(candidates) - 1, count).round().astype(int)
        selected = [int(candidates.iloc[rank]["source_position"]) for rank in ranks]
        if len(set(selected)) != count:
            raise AssertionError("Deterministic quantile selection produced duplicates.")
        positions.extend(selected)
    return sorted(positions, key=lambda position: (
        float(predictions.iloc[position]["fraud_probability"]), position
    ))


def _build_dataset(
    dataset_key: str,
    dataset_label: str,
    *,
    mode: str,
    prediction_path: Path,
    metric_path: Path,
    pipeline_path: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    quotas = PILOT_QUOTAS[dataset_label] if mode == "pilot" else FINAL_QUOTAS[dataset_label]
    predictions = pd.read_csv(prediction_path)
    threshold = _threshold(metric_path)
    pipeline = joblib.load(pipeline_path)
    positions = select_label_stratified_positions(
        predictions,
        quotas=quotas,
        source_id_column="row_index" if dataset_label == "ULB" else None,
        excluded_ids=(set() if mode == "pilot" else _exposed_ids(dataset_key, include_pilot=True)),
    )
    if dataset_label == "ULB":
        raw = pd.read_csv(PROJECT_ROOT / "datasets" / "ULB_creditCard.csv")
        source_frame = raw.drop_duplicates().reset_index(drop=True)
    else:
        raw = pd.read_csv(
            PROJECT_ROOT / "datasets" / "sparkov" / "fraudTest.csv", low_memory=False
        )
        source_frame = _sparkov_engineered(raw)
        if "transaction_time" in predictions and not pd.to_datetime(
            predictions["transaction_time"]
        ).reset_index(drop=True).equals(source_frame["transaction_time"].reset_index(drop=True)):
            raise ValueError("Sparkov prediction order does not match the source test file.")

    generation_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    prefix = "FOLLOWUP_PILOT" if mode == "pilot" else "FOLLOWUP"
    for sequence, position in enumerate(positions, start=1):
        prediction = predictions.iloc[position]
        source_id = int(prediction["row_index"]) if dataset_label == "ULB" else int(position)
        source = source_frame.iloc[source_id] if dataset_label == "ULB" else source_frame.iloc[position]
        features = {name: source[name] for name in pipeline.feature_names_in_}
        score = float(pipeline.predict_proba(pd.DataFrame([features]))[0, 1])
        if abs(score - float(prediction["fraud_probability"])) > 1e-8:
            raise ValueError(f"{dataset_label} row does not reproduce its frozen score.")
        sample_id = f"{prefix}_{dataset_label.upper()}_{sequence:03d}"
        actual = int(prediction["actual"])
        predicted = int(score >= threshold)
        stratum = (
            "TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"
        )
        generation_rows.append({
            "sample_id": sample_id,
            "dataset_id": dataset_label,
            "test_case_type": f"followup_{mode}_retrieval_comparison",
            "evaluation_cohort": "label_stratified_transaction_panel",
            "source_test_row_index": source_id,
            **({"transaction_time": source["transaction_time"].isoformat()}
               if dataset_label == "Sparkov" else {}),
            **features,
            "expected_fraud_probability": score,
            "decision_threshold": threshold,
            "expected_alert": predicted,
        })
        label_rows.append({
            "sample_id": sample_id,
            "dataset_id": dataset_label,
            "source_test_row_index": source_id,
            "evaluation_only_actual_class": actual,
            "confusion_stratum": stratum,
        })

    filename = "followup_pilot_transaction_samples.csv" if mode == "pilot" else "followup_transaction_samples.csv"
    output_path = PROJECT_ROOT / "datasets" / dataset_key / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(generation_rows).to_csv(output_path, index=False, float_format="%.10f")
    return output_path, label_rows, {
        "dataset_id": dataset_label,
        "sample_count": len(generation_rows),
        "fraud_label_count": sum(row["evaluation_only_actual_class"] for row in label_rows),
        "non_fraud_label_count": sum(not row["evaluation_only_actual_class"] for row in label_rows),
        "confusion_counts": dict(pd.Series(
            [row["confusion_stratum"] for row in label_rows]
        ).value_counts().sort_index()),
        "generation_panel_path": str(output_path.relative_to(PROJECT_ROOT)),
        "generation_panel_sha256": _sha256(output_path),
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "prediction_sha256": _sha256(prediction_path),
        "pipeline_path": str(pipeline_path.relative_to(PROJECT_ROOT)),
        "pipeline_sha256": _sha256(pipeline_path),
        "threshold": threshold,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "final"), required=True)
    parser.add_argument(
        "--ulb-artifact-directory", type=Path,
        default=PROJECT_ROOT / "artifacts" / "ulb",
    )
    parser.add_argument(
        "--sparkov-artifact-directory", type=Path,
        default=PROJECT_ROOT / "artifacts" / "sparkov",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    rows: list[dict[str, Any]] = []
    panel_records: list[dict[str, Any]] = []
    for key, label, directory in (
        ("ulb", "ULB", args.ulb_artifact_directory),
        ("sparkov", "Sparkov", args.sparkov_artifact_directory),
    ):
        output, labels, record = _build_dataset(
            key,
            label,
            mode=args.mode,
            prediction_path=directory / "test_predictions.csv",
            metric_path=directory / "test_metrics.csv",
            pipeline_path=directory / "selected_pipeline.joblib",
        )
        del output
        rows.extend(labels)
        panel_records.append(record)
    evaluation_directory = (
        PROJECT_ROOT / "datasets" / "evaluation" / "followup_retrieval"
    )
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    stem = "followup_pilot" if args.mode == "pilot" else "followup"
    labels_path = evaluation_directory / f"{stem}_evaluation_labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "non_reportable_30_case_orchestration_pilot"
            if args.mode == "pilot" else "candidate_panel_not_protocol_locked"
        ),
        "mode": args.mode,
        "sampling_frame": "label_stratified_TP_FP_FN_TN_transactions",
        "selection": "deterministic evenly spaced score ranks within offline label strata",
        "ground_truth_withheld_from_generation": True,
        "labels_path": str(labels_path.relative_to(PROJECT_ROOT)),
        "labels_sha256": _sha256(labels_path),
        "panels": panel_records,
    }
    manifest_path = evaluation_directory / f"{stem}_panel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=int), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=int))


if __name__ == "__main__":
    main()
