"""Training-only historical summaries for LLM questions and thesis reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


SEED = 42


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _amount_by_class(frame: pd.DataFrame, target: str, amount: str) -> list[dict[str, Any]]:
    summary = frame.groupby(target)[amount].agg(
        transactions="count",
        mean="mean",
        median="median",
        standard_deviation="std",
        minimum="min",
        maximum="max",
    ).reset_index()
    return _records(summary.round(6))


def build_ulb_profile(project_root: Path) -> dict[str, Any]:
    data = pd.read_csv(
        project_root / "datasets" / "ULB_creditCard.csv"
    ).drop_duplicates().reset_index(drop=True)
    X = data.drop(columns="Class")
    y = data["Class"].astype("int8")
    X_development, _, y_development, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )
    X_train, _, y_train, _ = train_test_split(
        X_development,
        y_development,
        test_size=0.20,
        stratify=y_development,
        random_state=SEED,
    )
    training = X_train.copy()
    training["Class"] = y_train

    hourly = (
        training.assign(relative_hour=(training["Time"] // 3_600).astype(int))
        .groupby("relative_hour")["Class"]
        .agg(transactions="count", fraud="sum", fraud_rate="mean")
        .reset_index()
    )
    hourly = hourly[hourly["transactions"] >= 1_000].nlargest(8, "fraud_rate")

    v_columns = [f"V{number}" for number in range(1, 29)]
    medians = training.groupby("Class")[v_columns].median().T
    medians.columns = ["legitimate_median", "fraud_median"]
    medians["absolute_median_difference"] = (
        medians["fraud_median"] - medians["legitimate_median"]
    ).abs()
    medians = medians.nlargest(10, "absolute_median_difference").reset_index(
        names="anonymized_feature"
    )

    return {
        "dataset_id": "ULB",
        "dataset_nature": "real anonymized benchmark",
        "scope": "deduplicated training partition only",
        "interpretation_limit": (
            "V1-V28 semantics are undisclosed; differences are statistical and must not "
            "be assigned invented business meanings."
        ),
        "records": [
            {
                "evidence_id": "historical:ulb:prevalence",
                "topic": "historical_prevalence",
                "facts": {
                    "transactions": int(len(training)),
                    "fraud": int(training["Class"].sum()),
                    "legitimate": int((training["Class"] == 0).sum()),
                    "fraud_rate": float(training["Class"].mean()),
                    "elapsed_hours_min": float(training["Time"].min() / 3_600),
                    "elapsed_hours_max": float(training["Time"].max() / 3_600),
                },
            },
            {
                "evidence_id": "historical:ulb:amount",
                "topic": "amount_by_class",
                "facts": _amount_by_class(training, "Class", "Amount"),
            },
            {
                "evidence_id": "historical:ulb:relative-time",
                "topic": "highest_observed_fraud_rate_relative_hours",
                "facts": _records(hourly.round(8)),
                "caution": "Descriptive relative-time bins; no calendar timestamp is supplied.",
            },
            {
                "evidence_id": "historical:ulb:anonymized-features",
                "topic": "largest_training_median_differences",
                "facts": _records(medians.round(8)),
            },
        ],
    }


def build_sparkov_profile(project_root: Path) -> dict[str, Any]:
    path = project_root / "datasets" / "sparkov" / "fraudTrain.csv"
    data = pd.read_csv(
        path,
        usecols=["trans_date_trans_time", "category", "amt", "dob", "is_fraud"],
        parse_dates=["trans_date_trans_time", "dob"],
    )
    cutoff = data["trans_date_trans_time"].quantile(0.80, interpolation="nearest")
    training = data[data["trans_date_trans_time"] < cutoff].copy()
    training["hour"] = training["trans_date_trans_time"].dt.hour
    training["day_of_week"] = training["trans_date_trans_time"].dt.dayofweek
    training["month"] = training["trans_date_trans_time"].dt.to_period("M").astype(str)
    training["age"] = (
        (training["trans_date_trans_time"] - training["dob"]).dt.days / 365.2425
    ).clip(0, 110)
    training["age_band"] = pd.cut(
        training["age"],
        bins=[0, 25, 35, 45, 55, 65, 110],
        labels=["under_25", "25_34", "35_44", "45_54", "55_64", "65_plus"],
        right=False,
    )

    def grouped_rate(column: str, minimum_transactions: int = 1) -> pd.DataFrame:
        summary = (
            training.groupby(column, observed=True)["is_fraud"]
            .agg(transactions="count", fraud="sum", fraud_rate="mean")
            .reset_index()
        )
        return summary[summary["transactions"] >= minimum_transactions]

    categories = grouped_rate("category", 1_000).nlargest(10, "fraud_rate")
    hours = grouped_rate("hour", 1_000).nlargest(8, "fraud_rate")
    weekdays = grouped_rate("day_of_week", 1_000).sort_values("day_of_week")
    ages = grouped_rate("age_band", 1_000).sort_values("age_band")
    monthly = grouped_rate("month", 1_000).sort_values("month")

    return {
        "dataset_id": "Sparkov",
        "dataset_nature": "synthetic behavioural benchmark",
        "scope": f"chronological training window before {cutoff.isoformat()}",
        "interpretation_limit": (
            "Patterns describe the Sparkov generator and cannot establish independent "
            "real-bank deployment behaviour."
        ),
        "records": [
            {
                "evidence_id": "historical:sparkov:prevalence",
                "topic": "historical_prevalence",
                "facts": {
                    "transactions": int(len(training)),
                    "fraud": int(training["is_fraud"].sum()),
                    "legitimate": int((training["is_fraud"] == 0).sum()),
                    "fraud_rate": float(training["is_fraud"].mean()),
                    "start": training["trans_date_trans_time"].min().isoformat(),
                    "end": training["trans_date_trans_time"].max().isoformat(),
                },
            },
            {
                "evidence_id": "historical:sparkov:amount",
                "topic": "amount_by_class",
                "facts": _amount_by_class(training, "is_fraud", "amt"),
            },
            {
                "evidence_id": "historical:sparkov:category",
                "topic": "highest_observed_fraud_rate_categories",
                "facts": _records(categories.round(8)),
            },
            {
                "evidence_id": "historical:sparkov:hour",
                "topic": "highest_observed_fraud_rate_hours",
                "facts": _records(hours.round(8)),
            },
            {
                "evidence_id": "historical:sparkov:weekday",
                "topic": "fraud_rate_by_weekday_monday_zero",
                "facts": _records(weekdays.round(8)),
            },
            {
                "evidence_id": "historical:sparkov:age",
                "topic": "fraud_rate_by_age_band",
                "facts": _records(ages.round(8)),
            },
            {
                "evidence_id": "historical:sparkov:monthly",
                "topic": "monthly_fraud_trend",
                "facts": _records(monthly.round(8)),
            },
        ],
    }


def build_historical_profiles(project_root: Path) -> dict[str, Any]:
    """Generate and save both training-only historical profiles."""
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Descriptive aggregates use only the locked training partitions. They are "
            "not final-test results and do not contain individual customer identifiers."
        ),
        "datasets": [
            build_ulb_profile(project_root),
            build_sparkov_profile(project_root),
        ],
    }
    output_dir = project_root / "artifacts" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "historical_profiles.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def load_historical_profiles(project_root: Path) -> dict[str, Any]:
    path = project_root / "artifacts" / "historical" / "historical_profiles.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Historical profiles not found at {path}. Run tools/build_historical_profiles.py."
        )
    return json.loads(path.read_text(encoding="utf-8"))
