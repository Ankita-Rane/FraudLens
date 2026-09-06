"""Quantify feasible final-evaluation cohorts without freezing a thesis result sample."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATES = (0.50, 0.60, 0.70)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _threshold(path: Path) -> float:
    metrics = pd.read_csv(path)
    selected = metrics[metrics["threshold_rule"] == "validation_tuned"]
    if len(selected) != 1:
        raise ValueError(f"Expected one validation_tuned row in {path}.")
    return float(selected.iloc[0]["threshold"])


def cohort_feasibility(
    predictions: pd.DataFrame,
    *,
    detector_threshold: float,
    release_gates: Sequence[float] = DEFAULT_GATES,
) -> list[dict[str, Any]]:
    """Count disjoint narrative, score-gate, FN and TN strata for each gate."""
    required = {"actual", "fraud_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    actual = predictions["actual"].astype(int)
    score = predictions["fraud_probability"].astype(float)
    if not score.between(0, 1).all():
        raise ValueError("Fraud probabilities must be in [0, 1].")

    rows: list[dict[str, Any]] = []
    for gate in release_gates:
        if not detector_threshold < gate <= 1:
            raise ValueError(
                f"Release gate {gate} must exceed detector threshold {detector_threshold}."
            )
        narrative = score >= gate
        gate_challenge = (score >= detector_threshold) & (score < gate)
        detector_negative = score < detector_threshold
        counts = {
            "release_gate": gate,
            "detector_threshold": detector_threshold,
            "narrative_comparison_total": int(narrative.sum()),
            "narrative_true_positive": int((narrative & actual.eq(1)).sum()),
            "narrative_false_positive": int((narrative & actual.eq(0)).sum()),
            "score_gate_challenge_total": int(gate_challenge.sum()),
            "score_gate_true_positive": int((gate_challenge & actual.eq(1)).sum()),
            "score_gate_false_positive": int((gate_challenge & actual.eq(0)).sum()),
            "false_negative_control": int((detector_negative & actual.eq(1)).sum()),
            "true_negative_control": int((detector_negative & actual.eq(0)).sum()),
        }
        if (
            counts["narrative_comparison_total"]
            + counts["score_gate_challenge_total"]
            + counts["false_negative_control"]
            + counts["true_negative_control"]
            != len(predictions)
        ):
            raise AssertionError("Cohort strata do not partition the prediction file.")
        rows.append(counts)
    return rows


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    datasets = []
    for key, label in (("ulb", "ULB"), ("sparkov", "Sparkov")):
        predictions_path = project_root / "artifacts" / key / "test_predictions.csv"
        metrics_path = project_root / "artifacts" / key / "test_metrics.csv"
        threshold = _threshold(metrics_path)
        predictions = pd.read_csv(predictions_path)
        datasets.append({
            "dataset_id": label,
            "prediction_count": len(predictions),
            "predictions_sha256": _file_hash(predictions_path),
            "test_metrics_sha256": _file_hash(metrics_path),
            "gate_candidates": cohort_feasibility(
                predictions,
                detector_threshold=threshold,
            ),
        })
    return {
        "schema_version": "1.0",
        "scientific_status": "design_feasibility_not_final_sample",
        "analysis_unit": (
            "One alert. Question and repeat cells must be aggregated within alert and "
            "configuration or handled with a cluster-aware model."
        ),
        "cohort_roles": {
            "narrative_comparison": (
                "All A/B/C configurations can generate; eligible for paired narrative metrics."
            ),
            "score_gate_challenge": (
                "Detector alert below C release gate; evaluates suppression/referral, not "
                "paired narrative correctness."
            ),
            "detector_controls": (
                "FN/TN cases test safe no-alert behaviour and are not pooled with narrative metrics."
            ),
        },
        "candidate_common_release_gate": 0.50,
        "candidate_gate_warning": (
            "0.50 is a pilot candidate, not a result-selected final value. Freeze the gate "
            "before final outcomes after reviewing feasibility, calibration limits and workload."
        ),
        "label_policy": (
            "Actual labels are used only for offline cohort auditing and stratified reporting; "
            "they remain excluded from generation evidence."
        ),
        "datasets": datasets,
    }


def main() -> None:
    output = PROJECT_ROOT / "datasets" / "evaluation" / "cohort_feasibility.json"
    payload = build_payload()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
