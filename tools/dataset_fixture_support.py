"""Shared deterministic reconstruction logic for local ULB/Sparkov CSV fixtures."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from tools.build_pilot_sample_panels import _sparkov_engineered


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXTURE_PATHS = (
    "datasets/ulb/candidate_transaction_samples.csv",
    "datasets/ulb/followup_transaction_samples.csv",
    "datasets/ulb/pilot_transaction_samples.csv",
    "datasets/ulb/runtime_transaction_samples.csv",
    "datasets/sparkov/candidate_transaction_samples.csv",
    "datasets/sparkov/followup_transaction_samples.csv",
    "datasets/sparkov/pilot_transaction_samples.csv",
    "datasets/sparkov/runtime_transaction_samples.csv",
)

SELECTOR_COLUMNS = (
    "sample_id",
    "test_case_type",
    "evaluation_cohort",
    "source_test_row_index",
)

DATASET_CONFIG = {
    "ULB": {
        "dataset_key": "ulb",
        "source_path": "datasets/ULB_creditCard.csv",
        "source_preparation": "drop_exact_duplicates_keep_first_then_reset_index",
    },
    "Sparkov": {
        "dataset_key": "sparkov",
        "source_path": "datasets/sparkov/fraudTest.csv",
        "source_preparation": "frozen_sparkov_feature_engineering_in_source_row_order",
    },
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_from_metrics(path: Path) -> float:
    frame = pd.read_csv(path)
    selected = frame[frame["threshold_rule"].eq("validation_tuned")]
    if len(selected) != 1:
        raise ValueError(f"Expected one validation_tuned threshold in {path}.")
    return float(selected.iloc[0]["threshold"])


@lru_cache(maxsize=2)
def source_frame(dataset_id: str) -> pd.DataFrame:
    config = DATASET_CONFIG[dataset_id]
    path = PROJECT_ROOT / config["source_path"]
    if dataset_id == "ULB":
        return pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return _sparkov_engineered(pd.read_csv(path, low_memory=False))


@lru_cache(maxsize=4)
def pipeline(relative_path: str) -> Any:
    return joblib.load(PROJECT_ROOT / relative_path)


@lru_cache(maxsize=4)
def detector_threshold(relative_path: str) -> float:
    return threshold_from_metrics(PROJECT_ROOT / relative_path)


def reconstruct_fixture(specification: dict[str, Any]) -> pd.DataFrame:
    """Rebuild one fixture from source-row selectors and frozen detector artifacts."""
    dataset_id = specification["dataset_id"]
    frame = source_frame(dataset_id)
    model = pipeline(specification["pipeline_path"])
    threshold = detector_threshold(specification["metrics_path"])
    expected_threshold = float(specification["decision_threshold"])
    decimals = int(specification["float_format"].split(".", 1)[1][:-1])
    tolerance = 0.5 * (10 ** -decimals) + 1e-15
    if abs(threshold - expected_threshold) > tolerance:
        raise ValueError(
            f"Threshold drift for {dataset_id}: {threshold} != {expected_threshold}."
        )

    rows: list[dict[str, Any]] = []
    for selector in specification["rows"]:
        source_index = int(selector["source_test_row_index"])
        if not 0 <= source_index < len(frame):
            raise IndexError(
                f"Source index {source_index} is outside {dataset_id} row range."
            )
        source = frame.iloc[source_index]
        features = {name: source[name] for name in model.feature_names_in_}
        score = float(model.predict_proba(pd.DataFrame([features]))[0, 1])
        row: dict[str, Any] = {
            "sample_id": selector["sample_id"],
            "dataset_id": dataset_id,
            "test_case_type": selector["test_case_type"],
            "source_test_row_index": source_index,
        }
        if "evaluation_cohort" in specification["columns"]:
            row["evaluation_cohort"] = selector["evaluation_cohort"]
        if dataset_id == "Sparkov" and "transaction_time" in specification["columns"]:
            timestamp = pd.Timestamp(source["transaction_time"])
            row["transaction_time"] = (
                timestamp.isoformat()
                if specification["timestamp_style"] == "iso_t"
                else timestamp.strftime("%Y-%m-%d %H:%M:%S")
            )
        row.update(features)
        row["expected_fraud_probability"] = score
        row["decision_threshold"] = threshold
        row["expected_alert"] = int(score >= threshold)
        if "evaluation_only_actual_class" in specification["columns"]:
            row["evaluation_only_actual_class"] = int(
                source["Class"] if dataset_id == "ULB" else source["is_fraud"]
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    expected_columns = list(specification["columns"])
    missing = set(expected_columns) - set(result.columns)
    extra = set(result.columns) - set(expected_columns)
    if missing or extra:
        raise ValueError(
            f"Column reconstruction mismatch for {specification['path']}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return result[expected_columns]


def write_fixture(frame: pd.DataFrame, path: Path, specification: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = frame.copy()
    for column, format_specification in specification.get("column_formats", {}).items():
        if format_specification == "d":
            serialized[column] = serialized[column].map(lambda value: str(int(value)))
        else:
            serialized[column] = serialized[column].map(
                lambda value, spec=format_specification: format(float(value), spec)
            )
    serialized.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format=(
            None if specification.get("column_formats") else specification["float_format"]
        ),
    )
