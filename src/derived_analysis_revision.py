"""Versioned post-hoc corrections for derived automated-study metrics.

The immutable DEC-021 generation/analysis implementation snapshot remains untouched.
This module re-evaluates saved raw runs without new model calls and records the exact
post-hoc correction separately from the pre-registered implementation.
"""

from __future__ import annotations

from collections import defaultdict
import json
import logging
from typing import Any, Iterable, Sequence

import numpy as np

from src.abc_evaluation import evaluate_run as evaluate_locked_run
from src.statistical_analysis import (
    BINARY_METRICS,
    DEFAULT_METRICS,
    _binary_result,
    _continuous_result,
    _holm_adjust,
)


ANALYSIS_REVISION = "DERIVED-EVAL-20260831-01"
HOLM_ALPHA = 0.05
LOGGER = logging.getLogger(__name__)


def _raw_candidate_json_object(run: dict[str, Any]) -> dict[str, Any] | None:
    candidate = run.get("candidate_response_text")
    if not isinstance(candidate, str):
        return None
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def evaluate_run_corrected(run: dict[str, Any]) -> dict[str, Any]:
    """Apply versioned derived-metric corrections to a locked raw run."""
    result = evaluate_locked_run(run)
    candidate_json = _raw_candidate_json_object(run)
    result["response_json_valid"] = float(candidate_json is not None)
    result["response_json_valid_basis"] = "raw_candidate_response_text"
    result["released_parsed_answer_available"] = float(
        isinstance(run.get("parsed_answer"), dict)
    )
    LOGGER.info(
        "evaluated_run request_id=%s dataset=%s configuration=%s status=%s "
        "candidate_json_valid=%s released_parsed_answer=%s",
        result["request_id"], result["dataset_id"], result["configuration"],
        result["status"], int(result["response_json_valid"]),
        int(result["released_parsed_answer_available"]),
    )
    return result


def _corrected_alert_cells(
    records: Sequence[dict[str, Any]], metric_keys: Sequence[str]
) -> dict[tuple[Any, ...], dict[str, dict[str, float]]]:
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
            aggregated[metric] = (
                max(values) if metric in BINARY_METRICS else float(np.mean(values))
            )
        aggregated["source_cell_count"] = float(len(items))
        cells[(dataset, cohort, experiment, sample_id, engine)][str(configuration)] = aggregated
    return cells


def paired_configuration_analysis_corrected(
    runs: Iterable[dict[str, Any]],
    *,
    metric_keys: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, Any]:
    """Recompute paired tests from corrected per-run metrics and annotate support."""
    records = [evaluate_run_corrected(run) for run in runs]
    cells = _corrected_alert_cells(records, metric_keys)
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
            for reference_configuration, candidate_configuration in (("A", "B"), ("B", "C")):
                for metric in metric_keys:
                    pairs = [
                        (
                            cell[reference_configuration].get(metric),
                            cell[candidate_configuration].get(metric),
                        )
                        for cell in dataset_cells
                        if reference_configuration in cell and candidate_configuration in cell
                    ]
                    complete = [
                        (float(reference), float(candidate))
                        for reference, candidate in pairs
                        if reference is not None and candidate is not None
                    ]
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
    _holm_adjust(results)
    for row in results:
        adjusted = row.get("holm_adjusted_p_value")
        row["holm_supported"] = adjusted < HOLM_ALPHA if adjusted is not None else None
        LOGGER.info(
            "paired_result dataset=%s comparison=%s metric=%s pairs=%s "
            "primary_p=%s holm_p=%s holm_supported=%s",
            row.get("dataset_id"), row.get("comparison"), row.get("metric"),
            row.get("pair_count"), row.get("primary_p_value"), adjusted,
            row.get("holm_supported"),
        )
    return {
        "analysis_revision": ANALYSIS_REVISION,
        "design": (
            "paired at alert level within dataset, sample, engine and registered study; "
            "saved question/repeat cells are aggregated within alert and configuration"
        ),
        "continuous_alert_aggregation": "arithmetic mean across available question/repeat cells",
        "binary_alert_aggregation": "any failure/referral across available question/repeat cells",
        "comparisons": ["B-A", "C-B"],
        "multiplicity_adjustment": "Holm family-wise adjustment over available primary tests",
        "significance_decision_rule": (
            "Supported only when the registered primary p-value, after Holm family-wise "
            f"adjustment, is < {HOLM_ALPHA}; paired-t p-values do not determine support."
        ),
        "holm_alpha": HOLM_ALPHA,
        "small_sample_rule": "n<20 is reported as exploratory; n<2 has no continuous inference",
        "results": results,
    }
