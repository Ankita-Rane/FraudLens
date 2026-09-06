"""Paired A/B/C statistical analysis with explicit small-sample safeguards."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats

from src.abc_evaluation import evaluate_run


DEFAULT_METRICS = (
    "explanation_fidelity",
    "response_json_valid",
    "claim_citation_coverage",
    "citation_id_validity",
    "policy_citation_coverage",
    "audit_trace_completeness",
    "processing_latency_ms",
    "cpu_time_ms",
    "memory_rss_delta_bytes",
    "manual_review_referral",
)

BINARY_METRICS = {
    "guardrail_violation_rate",
    "manual_review_referral",
    "response_json_valid",
    "candidate_response_json_valid",
    "released_response_json_valid",
}

FOLLOWUP_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A_direct", "B_lexical"),
    ("A_direct", "B_dense"),
    ("B_lexical", "B_dense"),
    ("B_lexical", "C_lexical"),
    ("B_dense", "C_dense"),
    ("C_lexical", "C_dense"),
    ("C_dense", "D_hybrid"),
    ("C_lexical", "D_hybrid"),
    ("A_direct", "D_hybrid"),
)

FOLLOWUP_METRICS: tuple[str, ...] = (
    "candidate_explanation_fidelity",
    "candidate_response_json_valid",
    "candidate_claim_citation_coverage",
    "candidate_citation_id_validity",
    "candidate_policy_citation_coverage",
    "released_response_json_valid",
    "released_claim_citation_coverage",
    "released_citation_id_validity",
    "released_policy_citation_coverage",
    "audit_trace_completeness",
    "processing_latency_ms",
    "retrieval_latency_ms",
    "cpu_time_ms",
    "memory_rss_delta_bytes",
    "manual_review_referral",
)


def _alert_level_metric_cells(
    records: Sequence[dict[str, Any]], metric_keys: Sequence[str]
) -> dict[tuple[Any, ...], dict[str, dict[str, float]]]:
    """Aggregate repeated questions/runs so the independent unit remains one alert."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["dataset_id"],
            record["evaluation_cohort"],
            record.get("study_id") or record["experiment_id"],
            record["sample_id"],
            record["engine_name"],
            record["configuration"],
        )
        grouped[key].append(record)

    cells: dict[tuple[Any, ...], dict[str, dict[str, float]]] = defaultdict(dict)
    for key, items in grouped.items():
        dataset, cohort, experiment, sample_id, engine, configuration = key
        aggregated: dict[str, float] = {}
        for metric in metric_keys:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            if not values:
                continue
            aggregated[metric] = max(values) if metric in BINARY_METRICS else float(np.mean(values))
        aggregated["source_cell_count"] = float(len(items))
        cells[(dataset, cohort, experiment, sample_id, engine)][str(configuration)] = aggregated
    return cells


def _p(value: float) -> float:
    """Return a p-value at full float precision.

    Rounding to a fixed number of decimal places collapses any genuinely tiny
    p-value to exactly 0.0, which is not a valid p-value and which also creates
    artificial ties in the Holm adjustment.  p-values are therefore stored
    unrounded; effect sizes and intervals keep their 8-decimal rounding.
    """
    return float(value)


def _holm_step_down(results: list[dict[str, Any]], indices: list[int], field: str) -> None:
    """Holm step-down over one family of tests, writing the adjusted value to `field`."""
    ordered = sorted(indices, key=lambda i: results[i]["primary_p_value"])
    count = len(ordered)
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * results[index]["primary_p_value"])
        running = max(running, adjusted)
        results[index][field] = _p(running)


def _holm_adjust(results: list[dict[str, Any]]) -> None:
    """Control family-wise error with Holm, treating each dataset as one family.

    The locked protocol required Holm adjustment "within preregistered endpoint
    families" but did not enumerate those families operationally.  Each dataset is
    therefore treated as a single family, consistent with the protocol's prohibition
    on dataset pooling and with the prospectively stated dataset-separated analysis.
    This is a disclosed clarification, not a claim that per-dataset families were
    explicitly preregistered.

    Partitioning more finely (by dataset and metric, or by dataset and comparison)
    would raise the number of significant results, so no finer partition is applied.
    The original single global family is retained alongside, as a conservative
    sensitivity analysis, in `holm_adjusted_p_value_global_sensitivity`.
    """
    testable = [
        index for index, row in enumerate(results)
        if row.get("primary_p_value") is not None
    ]

    families: dict[Any, list[int]] = defaultdict(list)
    for index in testable:
        families[results[index].get("dataset_id")].append(index)
    for indices in families.values():
        _holm_step_down(results, indices, "holm_adjusted_p_value")

    _holm_step_down(results, testable, "holm_adjusted_p_value_global_sensitivity")


def _continuous_result(
    reference: Sequence[float], candidate: Sequence[float]
) -> dict[str, Any]:
    differences = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    count = len(differences)
    mean_difference = float(differences.mean())
    if count < 2:
        return {
            "pair_count": count,
            "mean_paired_difference": round(mean_difference, 8),
            "confidence_interval_95": None,
            "paired_t_p_value": None,
            "wilcoxon_p_value": None,
            "primary_p_value": None,
            "inference_status": "insufficient_pairs",
        }
    standard_error = float(differences.std(ddof=1) / sqrt(count))
    if standard_error == 0.0:
        p_value = 1.0 if mean_difference == 0.0 else 0.0
        return {
            "pair_count": count,
            "mean_paired_difference": round(mean_difference, 8),
            "confidence_interval_95": [round(mean_difference, 8)] * 2,
            "paired_t_p_value": p_value,
            "wilcoxon_p_value": p_value,
            "primary_p_value": p_value,
            "inference_status": "exploratory" if count < 20 else "estimable",
        }
    margin = float(stats.t.ppf(0.975, count - 1) * standard_error)
    paired_t = stats.ttest_rel(candidate, reference)
    try:
        wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "pair_count": count,
        "mean_paired_difference": round(mean_difference, 8),
        "confidence_interval_95": [
            round(mean_difference - margin, 8), round(mean_difference + margin, 8)
        ],
        "paired_t_p_value": _p(paired_t.pvalue),
        "wilcoxon_p_value": _p(wilcoxon_p),
        "primary_p_value": _p(wilcoxon_p),
        "inference_status": "exploratory" if count < 20 else "estimable",
    }


def _binary_result(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    reference_values = np.asarray(reference, dtype=int)
    candidate_values = np.asarray(candidate, dtype=int)
    improved = int(np.sum((reference_values == 1) & (candidate_values == 0)))
    worsened = int(np.sum((reference_values == 0) & (candidate_values == 1)))
    discordant = improved + worsened
    p_value = (
        float(stats.binomtest(min(improved, worsened), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    return {
        "pair_count": len(reference),
        "reference_1_candidate_0": improved,
        "reference_0_candidate_1": worsened,
        "discordant_pair_count": discordant,
        "mcnemar_exact_p_value": _p(p_value),
        "primary_p_value": _p(p_value),
        "inference_status": "exploratory" if len(reference) < 20 else "estimable",
    }


def paired_configuration_analysis(
    runs: Iterable[dict[str, Any]],
    *,
    metric_keys: Sequence[str] = DEFAULT_METRICS,
    comparisons: Sequence[tuple[str, str]] = (("A", "B"), ("B", "C")),
    include_factorial_interaction: bool = False,
) -> dict[str, Any]:
    """Compare registered treatments within identical paired evaluation cases."""
    normalized_comparisons = tuple(
        (str(reference), str(candidate)) for reference, candidate in comparisons
    )
    if not normalized_comparisons or any(a == b for a, b in normalized_comparisons):
        raise ValueError("comparisons must contain distinct reference/candidate pairs.")
    records = [evaluate_run(run) for run in runs]
    cells = _alert_level_metric_cells(records, metric_keys)

    results: list[dict[str, Any]] = []
    datasets = sorted({str(key[0]) for key in cells if key[0] is not None})
    for dataset in datasets:
        cohorts = sorted({
            str(key[1] or "unclassified")
            for key in cells if str(key[0]) == dataset
        })
        for cohort in cohorts:
            dataset_cells = [
                value for key, value in cells.items()
                if str(key[0]) == dataset and str(key[1] or "unclassified") == cohort
            ]
            for reference_configuration, candidate_configuration in normalized_comparisons:
                for metric in metric_keys:
                    pairs = [
                        (cell[reference_configuration].get(metric), cell[candidate_configuration].get(metric))
                        for cell in dataset_cells
                        if reference_configuration in cell and candidate_configuration in cell
                    ]
                    complete = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None]
                    if not complete:
                        results.append({
                            "dataset_id": dataset,
                            "evaluation_cohort": cohort,
                            "comparison": f"{candidate_configuration}-{reference_configuration}",
                            "metric": metric,
                            "pair_count": 0,
                            "primary_p_value": None,
                            "inference_status": "metric_unavailable",
                        })
                        continue
                    reference, candidate = zip(*complete)
                    calculation = (
                        _binary_result(reference, candidate)
                        if metric in BINARY_METRICS
                        else _continuous_result(reference, candidate)
                    )
                    results.append({
                        "dataset_id": dataset,
                        "evaluation_cohort": cohort,
                        "comparison": f"{candidate_configuration}-{reference_configuration}",
                        "metric": metric,
                        **calculation,
                    })
            if include_factorial_interaction:
                interaction_conditions = {
                    "B_lexical", "B_dense", "C_lexical", "C_dense"
                }
                for metric in metric_keys:
                    if metric in BINARY_METRICS:
                        continue
                    interaction_values: list[float] = []
                    for cell in dataset_cells:
                        if not interaction_conditions.issubset(cell):
                            continue
                        values = {
                            condition: cell[condition].get(metric)
                            for condition in interaction_conditions
                        }
                        if any(value is None for value in values.values()):
                            continue
                        interaction_values.append(
                            (float(values["C_dense"]) - float(values["C_lexical"]))
                            - (float(values["B_dense"]) - float(values["B_lexical"]))
                        )
                    if not interaction_values:
                        results.append({
                            "dataset_id": dataset,
                            "evaluation_cohort": cohort,
                            "comparison": "retrieval_x_guardrail_interaction",
                            "metric": metric,
                            "pair_count": 0,
                            "primary_p_value": None,
                            "inference_status": "metric_unavailable",
                        })
                        continue
                    calculation = _continuous_result(
                        [0.0] * len(interaction_values), interaction_values
                    )
                    results.append({
                        "dataset_id": dataset,
                        "evaluation_cohort": cohort,
                        "comparison": "retrieval_x_guardrail_interaction",
                        "metric": metric,
                        **calculation,
                    })
    _holm_adjust(results)
    return {
        "design": (
            "paired at alert level within dataset, sample, engine and registered study "
            "(or experiment when no study ID exists); "
            "question/repeat cells are aggregated within alert and configuration, and "
            "cohorts are analysed separately"
        ),
        "continuous_alert_aggregation": "arithmetic mean across available question/repeat cells",
        "binary_alert_aggregation": "any failure/referral across available question/repeat cells",
        "comparisons": [f"{candidate}-{reference}" for reference, candidate in normalized_comparisons],
        "factorial_interaction": include_factorial_interaction,
        "multiplicity_adjustment": (
            "Holm family-wise adjustment, one family per dataset"
        ),
        "multiplicity_family_definition": "dataset_id",
        "multiplicity_disclosure": (
            "The locked protocol required Holm adjustment within preregistered "
            "endpoint families but did not enumerate those families operationally. "
            "Each dataset is treated as one family, consistent with the protocol's "
            "prohibition on dataset pooling and the prospectively stated "
            "dataset-separated analysis. Per-dataset families were not themselves "
            "explicitly preregistered. The originally implemented single global "
            "family is retained as a conservative sensitivity analysis in "
            "holm_adjusted_p_value_global_sensitivity; no finer partition was "
            "computed, because finer partitions increase the number of significant "
            "results and had no preregistered basis."
        ),
        "small_sample_rule": "n<20 is reported as exploratory; n<2 has no continuous inference",
        "results": results,
    }


def paired_followup_analysis(
    runs: Iterable[dict[str, Any]],
    *,
    metric_keys: Sequence[str] = FOLLOWUP_METRICS,
) -> dict[str, Any]:
    """Run the preregisterable six-condition follow-up contrast family."""
    return paired_configuration_analysis(
        runs,
        metric_keys=metric_keys,
        comparisons=FOLLOWUP_COMPARISONS,
        include_factorial_interaction=True,
    )
