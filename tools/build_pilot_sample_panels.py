"""Build small, deterministic ULB/Sparkov cohorts for non-reportable Qwen piloting."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_GATE = 0.50
NARRATIVE_PER_DATASET = 4
GATE_CHALLENGE_PER_DATASET = 2


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _threshold(path: Path) -> float:
    frame = pd.read_csv(path)
    row = frame[frame["threshold_rule"] == "validation_tuned"]
    if len(row) != 1:
        raise ValueError(f"Expected one validation_tuned threshold in {path}.")
    return float(row.iloc[0]["threshold"])


def _evenly_spaced_indices(candidates: pd.DataFrame, count: int) -> list[int]:
    if len(candidates) < count:
        raise ValueError(f"Need {count} candidates but found {len(candidates)}.")
    ordered = candidates.sort_values(
        ["fraud_probability", "source_position"], kind="stable"
    )
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [int(ordered.iloc[position]["source_position"]) for position in positions]


def select_pilot_cases(
    predictions: pd.DataFrame,
    detector_threshold: float,
    *,
    release_gate: float = RELEASE_GATE,
    narrative_count: int = NARRATIVE_PER_DATASET,
    gate_challenge_count: int = GATE_CHALLENGE_PER_DATASET,
) -> list[dict[str, Any]]:
    """Select disjoint pilot cases without using labels for generation cohorts."""
    required = {"actual", "fraud_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if not 0 <= detector_threshold < release_gate <= 1:
        raise ValueError(
            "The detector threshold and release gate must satisfy "
            "0 <= detector_threshold < release_gate <= 1."
        )
    if narrative_count < 1 or gate_challenge_count < 1:
        raise ValueError("Pilot cohort counts must be positive.")

    frame = predictions.copy()
    frame["fraud_probability"] = frame["fraud_probability"].astype(float)
    frame["actual"] = frame["actual"].astype(int)
    if not frame["fraud_probability"].between(0, 1).all():
        raise ValueError("Fraud probabilities must be in [0, 1].")
    if not frame["actual"].isin((0, 1)).all():
        raise ValueError("Actual labels must be binary.")
    frame["source_position"] = np.arange(len(frame), dtype=int)
    narrative = frame[frame["fraud_probability"] >= release_gate]
    gate = frame[
        (frame["fraud_probability"] >= detector_threshold)
        & (frame["fraud_probability"] < release_gate)
    ]
    false_negative = frame[
        (frame["fraud_probability"] < detector_threshold) & frame["actual"].eq(1)
    ].sort_values("fraud_probability", ascending=False, kind="stable")
    true_negative = frame[
        (frame["fraud_probability"] < detector_threshold) & frame["actual"].eq(0)
    ].sort_values(["fraud_probability", "source_position"], kind="stable")
    if false_negative.empty or true_negative.empty:
        raise ValueError("Pilot controls require at least one FN and TN.")
    selected: list[dict[str, Any]] = []
    for position in _evenly_spaced_indices(narrative, narrative_count):
        selected.append({"source_position": position, "cohort": "narrative_comparison"})
    for position in _evenly_spaced_indices(gate, gate_challenge_count):
        selected.append({"source_position": position, "cohort": "score_gate_challenge"})
    selected.extend([
        {"source_position": int(false_negative.iloc[0]["source_position"]), "cohort": "false_negative_control"},
        {"source_position": int(true_negative.iloc[0]["source_position"]), "cohort": "true_negative_control"},
    ])
    return selected


def _sparkov_engineered(raw: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(raw["trans_date_trans_time"])
    dob = pd.to_datetime(raw["dob"])
    age = ((timestamp - dob).dt.days / 365.2425).clip(lower=0, upper=110)
    earth_radius_km = 6_371.0088
    lat1, lon1 = np.radians(raw["lat"]), np.radians(raw["long"])
    lat2, lon2 = np.radians(raw["merch_lat"]), np.radians(raw["merch_long"])
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return pd.DataFrame({
        "transaction_time": timestamp,
        "amt": raw["amt"].astype(float),
        "log_amt": np.log1p(raw["amt"]),
        "city_pop": raw["city_pop"].astype(float),
        "lat": raw["lat"].astype(float),
        "long": raw["long"].astype(float),
        "merch_lat": raw["merch_lat"].astype(float),
        "merch_long": raw["merch_long"].astype(float),
        "age": age,
        "distance_km": 2 * earth_radius_km * np.arcsin(np.sqrt(haversine)),
        "hour": timestamp.dt.hour.astype(int),
        "day_of_week": timestamp.dt.dayofweek.astype(int),
        "month": timestamp.dt.month.astype(int),
        "is_weekend": (timestamp.dt.dayofweek >= 5).astype(int),
        "category": raw["category"].astype(str),
        "gender": raw["gender"].astype(str),
        "state": raw["state"].astype(str),
        "is_fraud": raw["is_fraud"].astype(int),
    })


def _build_ulb() -> tuple[Path, dict[str, Any]]:
    raw_path = PROJECT_ROOT / "datasets" / "ULB_creditCard.csv"
    prediction_path = PROJECT_ROOT / "artifacts" / "ulb" / "test_predictions.csv"
    metrics_path = PROJECT_ROOT / "artifacts" / "ulb" / "test_metrics.csv"
    pipeline_path = PROJECT_ROOT / "artifacts" / "ulb" / "selected_pipeline.joblib"
    output = PROJECT_ROOT / "datasets" / "ulb" / "pilot_transaction_samples.csv"
    model_data = pd.read_csv(raw_path).drop_duplicates().reset_index(drop=True)
    predictions = pd.read_csv(prediction_path)
    threshold = _threshold(metrics_path)
    pipeline = joblib.load(pipeline_path)
    rows = []
    for sequence, selected in enumerate(
        select_pilot_cases(predictions, threshold), start=1
    ):
        prediction = predictions.iloc[selected["source_position"]]
        row_index = int(prediction["row_index"])
        features = model_data.loc[row_index].drop(labels=["Class"]).to_dict()
        frame = pd.DataFrame([{name: features[name] for name in pipeline.feature_names_in_}])
        score = float(pipeline.predict_proba(frame)[0, 1])
        if abs(score - float(prediction["fraud_probability"])) > 1e-9:
            raise ValueError("ULB pilot row does not reproduce its frozen prediction.")
        rows.append({
            "sample_id": f"PILOT_ULB_{sequence:03d}",
            "dataset_id": "ULB",
            "test_case_type": f"pilot_{selected['cohort']}",
            "evaluation_cohort": selected["cohort"],
            "source_test_row_index": row_index,
            **features,
            "expected_fraud_probability": score,
            "decision_threshold": threshold,
            "expected_alert": int(score >= threshold),
            "evaluation_only_actual_class": int(prediction["actual"]),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.10f")
    return output, {
        "dataset_id": "ULB", "sample_count": len(rows),
        "source_sha256": _sha256(raw_path), "predictions_sha256": _sha256(prediction_path),
        "pipeline_sha256": _sha256(pipeline_path), "samples_sha256": _sha256(output),
    }


def _build_sparkov() -> tuple[Path, dict[str, Any]]:
    raw_path = PROJECT_ROOT / "datasets" / "sparkov" / "fraudTest.csv"
    prediction_path = PROJECT_ROOT / "artifacts" / "sparkov" / "test_predictions.csv"
    metrics_path = PROJECT_ROOT / "artifacts" / "sparkov" / "test_metrics.csv"
    pipeline_path = PROJECT_ROOT / "artifacts" / "sparkov" / "selected_pipeline.joblib"
    output = PROJECT_ROOT / "datasets" / "sparkov" / "pilot_transaction_samples.csv"
    raw = pd.read_csv(raw_path, low_memory=False)
    engineered = _sparkov_engineered(raw)
    predictions = pd.read_csv(prediction_path)
    prediction_times = pd.to_datetime(predictions["transaction_time"])
    if not prediction_times.reset_index(drop=True).equals(
        engineered["transaction_time"].reset_index(drop=True)
    ):
        raise ValueError("Sparkov prediction rows do not match the official test order.")
    threshold = _threshold(metrics_path)
    pipeline = joblib.load(pipeline_path)
    rows = []
    for sequence, selected in enumerate(
        select_pilot_cases(predictions, threshold), start=1
    ):
        position = selected["source_position"]
        prediction = predictions.iloc[position]
        source = engineered.iloc[position]
        features = {name: source[name] for name in pipeline.feature_names_in_}
        frame = pd.DataFrame([features])
        score = float(pipeline.predict_proba(frame)[0, 1])
        if abs(score - float(prediction["fraud_probability"])) > 1e-9:
            raise ValueError("Sparkov pilot row does not reproduce its frozen prediction.")
        rows.append({
            "sample_id": f"PILOT_SPARKOV_{sequence:03d}",
            "dataset_id": "Sparkov",
            "test_case_type": f"pilot_{selected['cohort']}",
            "evaluation_cohort": selected["cohort"],
            "source_test_row_index": int(position),
            "transaction_time": source["transaction_time"].isoformat(),
            **features,
            "expected_fraud_probability": score,
            "decision_threshold": threshold,
            "expected_alert": int(score >= threshold),
            "evaluation_only_actual_class": int(prediction["actual"]),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.10f")
    return output, {
        "dataset_id": "Sparkov", "sample_count": len(rows),
        "source_sha256": _sha256(raw_path), "predictions_sha256": _sha256(prediction_path),
        "pipeline_sha256": _sha256(pipeline_path), "samples_sha256": _sha256(output),
    }


def main() -> None:
    ulb_path, ulb = _build_ulb()
    sparkov_path, sparkov = _build_sparkov()
    manifest = {
        "schema_version": "1.0",
        "scientific_status": "non_reportable_pilot_only",
        "release_gate_candidate": RELEASE_GATE,
        "selection": (
            "Evenly spaced model-score ranks within narrative and gate cohorts, plus "
            "nearest-threshold FN and lowest-score TN controls; labels do not select "
            "narrative/gate rows and remain evaluation-only."
        ),
        "panels": [
            {**ulb, "path": str(ulb_path.relative_to(PROJECT_ROOT))},
            {**sparkov, "path": str(sparkov_path.relative_to(PROJECT_ROOT))},
        ],
    }
    output = PROJECT_ROOT / "datasets" / "evaluation" / "pilot_panel_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
