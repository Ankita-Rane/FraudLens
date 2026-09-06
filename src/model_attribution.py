"""Model-faithful local attribution evidence for frozen fraud alerts.

TreeSHAP is used for the fitted random-forest pipelines and checked against the model
probability through the additive identity. The older grouped perturbation method is
retained as a separately labelled sensitivity analysis. Neither method is causal.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

import json
import numpy as np
import pandas as pd


SPARKOV_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "amount": ("amt", "log_amt"),
    "transaction_time": ("hour", "day_of_week", "month", "is_weekend"),
    "merchant_category": ("category",),
    "cardholder_context": ("city_pop", "lat", "long", "age", "gender", "state"),
    "merchant_location": ("merch_lat", "merch_long", "distance_km"),
}


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _clean_transformed_feature_name(name: str) -> str:
    """Remove a ColumnTransformer prefix while retaining the actual input name."""
    return name.split("__", maxsplit=1)[-1]


def canonical_driver_name(name: str, expected_features: Sequence[str]) -> str:
    """Map transformed/one-hot names to the frozen semantic driver vocabulary."""
    clean = _clean_transformed_feature_name(name)
    for group, features in SPARKOV_FEATURE_GROUPS.items():
        for feature in features:
            if clean == feature or clean.startswith(f"{feature}_"):
                return group
    if clean in expected_features:
        return clean
    return clean


def tree_shap_attribution(
    *,
    pipeline: Any,
    sample: Mapping[str, Any],
    top_k: int = 10,
) -> dict[str, Any]:
    """Return additive TreeSHAP evidence for the positive fraud class.

    The fitted preprocessing steps are applied before TreeSHAP, so the explanation is
    tied to the exact model input. This function intentionally supports only pipelines
    whose final estimator is tree-based and exposes binary ``predict_proba`` output.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    try:
        import shap
    except ImportError as error:  # pragma: no cover - covered by dependency audit
        raise RuntimeError("TreeSHAP attribution requires the 'shap' package.") from error

    expected_features = list(getattr(pipeline, "feature_names_in_", []))
    if not expected_features:
        raise ValueError("The fitted pipeline does not expose feature_names_in_.")
    missing = [name for name in expected_features if name not in sample]
    if missing:
        raise ValueError(f"Missing sample features={missing}.")

    frame = pd.DataFrame([{name: sample[name] for name in expected_features}])
    if not hasattr(pipeline, "steps") or len(pipeline.steps) < 2:
        raise ValueError("TreeSHAP requires a fitted preprocessing-plus-model pipeline.")
    preprocessing = pipeline[:-1]
    transformed = preprocessing.transform(frame)
    model = pipeline.steps[-1][1]
    if not hasattr(model, "estimators_") or not hasattr(model, "predict_proba"):
        raise ValueError("The final estimator is not a supported fitted tree classifier.")

    transformed_names = [str(name) for name in preprocessing.get_feature_names_out()]
    explanation = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )(transformed, check_additivity=True)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim != 3 or values.shape[2] < 2:
        raise ValueError("Expected binary-class TreeSHAP output with a fraud class.")
    fraud_values = values[0, :, 1].astype(float)
    fraud_base = float(base_values[0, 1])
    model_score = float(pipeline.predict_proba(frame)[0, 1])
    reconstructed = fraud_base + float(fraud_values.sum())
    additivity_error = abs(model_score - reconstructed)
    if additivity_error > 1e-6:
        raise ValueError(
            "TreeSHAP additivity check failed: "
            f"model={model_score}, reconstructed={reconstructed}."
        )

    raw_records = [
        {
            "transformed_feature": _clean_transformed_feature_name(feature),
            "feature": canonical_driver_name(feature, expected_features),
            "shap_value": round(float(value), 10),
            "absolute_shap_value": round(abs(float(value)), 10),
            "direction": (
                "increases_fraud_probability" if value > 0
                else "decreases_fraud_probability" if value < 0
                else "no_local_contribution"
            ),
        }
        for feature, value in zip(transformed_names, fraud_values)
    ]
    grouped: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        feature = str(record["feature"])
        target = grouped.setdefault(feature, {
            "feature": feature,
            "shap_value": 0.0,
            "transformed_features": [],
        })
        target["shap_value"] += float(record["shap_value"])
        target["transformed_features"].append(record["transformed_feature"])
    records = []
    for target in grouped.values():
        value = float(target["shap_value"])
        records.append({
            "feature": target["feature"],
            "shap_value": round(value, 10),
            "absolute_shap_value": round(abs(value), 10),
            "direction": (
                "increases_fraud_probability" if value > 0
                else "decreases_fraud_probability" if value < 0
                else "no_local_contribution"
            ),
            "transformed_features": sorted(set(target["transformed_features"])),
        })
    records.sort(key=lambda row: row["absolute_shap_value"], reverse=True)
    absolute = np.abs(fraud_values)
    cleaned_names = [_clean_transformed_feature_name(name) for name in transformed_names]
    categorical_mask = np.asarray([
        any(name == feature or name.startswith(f"{feature}_") for feature in (
            "category", "gender", "state"
        ))
        for name in cleaned_names
    ], dtype=bool)
    total_absolute = float(absolute.sum())
    categorical_absolute = float(absolute[categorical_mask].sum())
    ordered_absolute = np.sort(absolute)[::-1]
    tolerance = 1e-12
    probabilities = absolute / total_absolute if total_absolute else np.zeros_like(absolute)
    positive_probabilities = probabilities[probabilities > 0]
    entropy = float(-(positive_probabilities * np.log(positive_probabilities)).sum())
    effective_feature_count = float(np.exp(entropy)) if len(positive_probabilities) else 0.0

    def mass_share(count: int) -> float:
        return float(ordered_absolute[:count].sum() / total_absolute) if total_absolute else 0.0

    return {
        "method": "tree_shap_tree_path_dependent",
        "method_is_shap": True,
        "explained_class": "fraud_class_1",
        "model_fraud_probability": round(model_score, 10),
        "base_value": round(fraud_base, 10),
        "additivity_reconstructed_probability": round(reconstructed, 10),
        "additivity_absolute_error": round(additivity_error, 12),
        "all_feature_count": len(raw_records),
        "transformed_feature_count": len(raw_records),
        "semantic_driver_count": len(records),
        "driver_mapping_version": "canonical-driver-groups-v1",
        "diagnostics": {
            "nonzero_tolerance": tolerance,
            "nonzero_shap_count": int((absolute > tolerance).sum()),
            "total_absolute_shap_mass": round(total_absolute, 10),
            "top_1_absolute_mass_share": round(mass_share(1), 10),
            "top_3_absolute_mass_share": round(mass_share(3), 10),
            "top_10_absolute_mass_share": round(mass_share(10), 10),
            "effective_feature_count": round(effective_feature_count, 10),
            "categorical_nonzero_shap_count": int(
                ((absolute > tolerance) & categorical_mask).sum()
            ),
            "categorical_absolute_shap_mass_share": round(
                categorical_absolute / total_absolute if total_absolute else 0.0, 10
            ),
            "numeric_absolute_shap_mass_share": round(
                (total_absolute - categorical_absolute) / total_absolute
                if total_absolute else 0.0,
                10,
            ),
        },
        "top_features": records[:top_k],
        "limitations": [
            "TreeSHAP explains the fitted random-forest prediction; it does not prove causality or fraud.",
            "Correlated inputs can share or redistribute attribution and must be interpreted cautiously.",
            "ULB V1-V28 are anonymized components and have no recoverable business meaning.",
            "Sparkov is synthetic, so its driver patterns are not assumed to generalize to a bank portfolio.",
        ],
    }


def grouped_reference_perturbation(
    *,
    pipeline: Any,
    sample: Mapping[str, Any],
    reference_profile: Mapping[str, Any],
    feature_groups: Mapping[str, Sequence[str]] = SPARKOV_FEATURE_GROUPS,
    top_k: int = 5,
) -> dict[str, Any]:
    """Return model score changes under auditable group reference substitutions."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    expected_features = list(getattr(pipeline, "feature_names_in_", []))
    if not expected_features:
        raise ValueError("The fitted pipeline does not expose feature_names_in_.")
    missing_sample = [name for name in expected_features if name not in sample]
    missing_reference = [name for name in expected_features if name not in reference_profile]
    if missing_sample or missing_reference:
        raise ValueError(
            f"Missing sample features={missing_sample}; reference features={missing_reference}."
        )

    original = pd.DataFrame([{name: sample[name] for name in expected_features}])
    model_score = float(pipeline.predict_proba(original)[0, 1])
    records: list[dict[str, Any]] = []
    covered: set[str] = set()
    for group, features in feature_groups.items():
        valid_features = [name for name in features if name in expected_features]
        if not valid_features:
            continue
        covered.update(valid_features)
        counterfactual = original.copy()
        for feature in valid_features:
            counterfactual.at[0, feature] = reference_profile[feature]
        counterfactual_score = float(pipeline.predict_proba(counterfactual)[0, 1])
        delta = model_score - counterfactual_score
        records.append({
            "feature_group": group,
            "features": valid_features,
            "score_delta_when_replaced": round(delta, 10),
            "probability_after_reference_replacement": round(counterfactual_score, 10),
            "direction": (
                "observed_group_increases_score" if delta > 0
                else "observed_group_decreases_score" if delta < 0
                else "no_observed_change"
            ),
        })
    uncovered = sorted(set(expected_features) - covered)
    if uncovered:
        raise ValueError(f"Feature groups do not cover model inputs: {uncovered}")
    records.sort(key=lambda row: abs(row["score_delta_when_replaced"]), reverse=True)
    return {
        "method": "grouped_training_reference_perturbation",
        "method_is_shap": False,
        "model_fraud_probability": round(model_score, 10),
        "reference_profile_sha256": stable_payload_hash(reference_profile),
        "top_features": records[:top_k],
        "limitations": [
            "Score deltas are model sensitivity to one grouped reference substitution, not causal effects.",
            "Feature interactions mean group deltas are not additive and must not be summed.",
            "Marginal training medians/modes may not form an observed joint transaction and can create an out-of-distribution reference.",
            "This pilot is not SHAP; final SHAP use requires a frozen training-only background and method validation.",
        ],
    }
