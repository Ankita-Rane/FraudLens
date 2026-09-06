"""Post-hoc descriptive analyses for the locked six-condition follow-up study.

The functions in this module operate only on saved raw-run JSON records.  They do
not call a model, execute retrieval, re-run guardrails, or modify the locked V4
aggregate.  Every classification is explicit so the resulting denominators can be
audited from the immutable records.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import json
import math
import numpy as np

from src.abc_evaluation import evaluate_run
from src.followup_conditions import FOLLOWUP_CONDITION_IDS


EXPECTED_STUDY_ID = "FOLLOWUP-RETRIEVAL-20260902-02"
EXPECTED_CALL_COUNT = 3300
EXPECTED_CASE_COUNT = 550
EXPECTED_BATCH_COUNT = 55
GENERATION_GATE = 0.5
POSTHOC_ANALYSIS_ID = f"{EXPECTED_STUDY_ID}-POSTHOC-DESCRIPTIVE-V1"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    return sha256(path.read_bytes()).hexdigest()


def _dataset_id(run: dict[str, Any]) -> str:
    datasets = (run.get("model_evidence") or {}).get("datasets") or []
    identifiers = {
        str(row.get("dataset_id"))
        for row in datasets
        if isinstance(row, dict) and row.get("dataset_id")
    }
    if len(identifiers) != 1:
        raise ValueError(
            f"Run {run.get('request_id')} has {len(identifiers)} dataset identifiers."
        )
    return next(iter(identifiers))


def _condition_id(run: dict[str, Any]) -> str:
    trace = run.get("trace") or {}
    value = run.get("condition_id") or trace.get("condition_id")
    if value not in FOLLOWUP_CONDITION_IDS:
        raise ValueError(
            f"Run {run.get('request_id')} has unknown condition {value!r}."
        )
    return str(value)


def _detector_score(run: dict[str, Any]) -> float:
    scores = {
        float(alert["fraud_probability"])
        for row in (run.get("model_evidence") or {}).get("datasets", [])
        if isinstance(row, dict)
        for alert in row.get("top_alerts", [])
        if isinstance(alert, dict) and alert.get("fraud_probability") is not None
    }
    if len(scores) != 1:
        raise ValueError(
            f"Run {run.get('request_id')} has {len(scores)} detector scores."
        )
    score = next(iter(scores))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Detector score outside [0, 1]: {score}.")
    return score


def _generation_gate(run: dict[str, Any]) -> float:
    trace = run.get("trace") or {}
    gate = trace.get("model_score_generation_gate")
    if gate is None:
        raise ValueError(f"Run {run.get('request_id')} has no generation gate.")
    value = float(gate)
    if not math.isclose(value, GENERATION_GATE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Unexpected generation gate {value}.")
    return value


def _generation_attempted(run: dict[str, Any]) -> bool:
    candidate = run.get("candidate_response_text")
    if isinstance(candidate, str) and candidate.strip():
        return True
    return bool((run.get("trace") or {}).get("output_tokens"))


def _raw_run_set_sha256(paths: Sequence[Path]) -> str:
    """Hash the accepted run set without depending on an absolute directory path."""
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_v4_locked_runs(
    project_root: Path,
    source_results: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exactly the raw runs referenced by the accepted V4 aggregate.

    Extra failed or superseded experiment directories are deliberately ignored.  The
    V4 source-manifest list is the selection authority, and every referenced manifest
    is hash-checked before its raw records are loaded.
    """
    root = project_root.resolve()
    results_path = source_results.resolve()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if payload.get("scientific_status") != "locked_followup_aggregated_results":
        raise RuntimeError("Source results are not the locked follow-up aggregate.")
    if payload.get("study_id") != EXPECTED_STUDY_ID:
        raise RuntimeError("Source-results study ID is not the accepted follow-up study.")
    if payload.get("raw_call_count") != EXPECTED_CALL_COUNT:
        raise RuntimeError("Source results do not contain the required 3,300 calls.")
    if payload.get("case_count") != EXPECTED_CASE_COUNT:
        raise RuntimeError("Source results do not contain the required 550 cases.")
    if tuple(payload.get("condition_ids") or ()) != tuple(FOLLOWUP_CONDITION_IDS):
        raise RuntimeError("Source-results condition registry differs from the code registry.")

    protocol_path = root / str(payload["protocol_path"])
    if file_sha256(protocol_path) != payload.get("protocol_sha256"):
        raise RuntimeError("Locked protocol hash differs from the V4 record.")
    sources = payload.get("source_manifests") or []
    if len(sources) != EXPECTED_BATCH_COUNT:
        raise RuntimeError("V4 does not reference exactly 55 accepted batch manifests.")

    runs: list[dict[str, Any]] = []
    run_paths: list[Path] = []
    batch_ids: set[str] = set()
    for source in sources:
        manifest_path = root / str(source["manifest_path"])
        if file_sha256(manifest_path) != source.get("manifest_sha256"):
            raise RuntimeError(f"Manifest hash mismatch: {manifest_path}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_id = str(manifest.get("batch_id"))
        if batch_id in batch_ids:
            raise RuntimeError(f"Duplicate accepted batch ID: {batch_id}.")
        batch_ids.add(batch_id)
        if manifest.get("study_id") != EXPECTED_STUDY_ID:
            raise RuntimeError(f"Wrong study ID in {manifest_path}.")
        if manifest.get("scientific_status") != (
            "locked_followup_qwen_raw_outputs_pending_analysis"
        ):
            raise RuntimeError(f"Unaccepted batch status in {manifest_path}.")
        paths = sorted((manifest_path.parent / "raw_runs").glob("*.json"))
        expected = int(source["run_count"])
        if len(paths) != expected or manifest.get("completed_run_count") != expected:
            raise RuntimeError(f"Run-count mismatch for accepted batch {batch_id}.")
        run_paths.extend(paths)
        runs.extend(json.loads(path.read_text(encoding="utf-8")) for path in paths)

    if len(runs) != EXPECTED_CALL_COUNT:
        raise RuntimeError(f"Loaded {len(runs)} runs; expected {EXPECTED_CALL_COUNT}.")
    request_ids = [str(run.get("request_id")) for run in runs]
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("Duplicate request IDs exist in the accepted raw-run set.")
    if any((run.get("trace") or {}).get("study_id") != EXPECTED_STUDY_ID for run in runs):
        raise RuntimeError("At least one accepted raw run has the wrong study ID.")

    cells: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        cells[str(run["sample_id"])].add(_condition_id(run))
    expected_conditions = set(FOLLOWUP_CONDITION_IDS)
    incomplete = {
        sample_id: sorted(expected_conditions - conditions)
        for sample_id, conditions in cells.items()
        if conditions != expected_conditions
    }
    if len(cells) != EXPECTED_CASE_COUNT or incomplete:
        raise RuntimeError(
            f"Paired-design mismatch: cases={len(cells)}, incomplete={incomplete}."
        )

    provenance = {
        "source_results_path": str(results_path.relative_to(root)),
        "source_results_sha256": file_sha256(results_path),
        "protocol_path": str(protocol_path.relative_to(root)),
        "protocol_sha256": file_sha256(protocol_path),
        "accepted_manifest_count": len(sources),
        "accepted_manifest_set_sha256": sha256(
            "\n".join(sorted(str(row["manifest_sha256"]) for row in sources)).encode(
                "utf-8"
            )
        ).hexdigest(),
        "accepted_raw_run_count": len(runs),
        "accepted_raw_run_set_sha256": _raw_run_set_sha256(run_paths),
        "case_count": len(cells),
        "condition_ids": list(FOLLOWUP_CONDITION_IDS),
    }
    return runs, provenance


def build_gate_case_rows(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one verified detector-score row per unique evaluation case."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["sample_id"])].append(run)
    rows: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped.items()):
        conditions = {_condition_id(run) for run in items}
        datasets = {_dataset_id(run) for run in items}
        scores = {_detector_score(run) for run in items}
        gates = {_generation_gate(run) for run in items}
        if conditions != set(FOLLOWUP_CONDITION_IDS) or len(items) != len(
            FOLLOWUP_CONDITION_IDS
        ):
            raise ValueError(f"Case {sample_id} is not a complete six-condition cell.")
        if len(datasets) != 1 or len(scores) != 1 or len(gates) != 1:
            raise ValueError(f"Case-level score metadata differ across {sample_id}.")
        score = next(iter(scores))
        gate = next(iter(gates))
        rows.append({
            "dataset_id": next(iter(datasets)),
            "sample_id": sample_id,
            "detector_score": score,
            "generation_gate": gate,
            "gate_band": "at_or_above_gate" if score >= gate else "below_gate",
            "guardrailed_generation_eligible": score >= gate,
            "condition_count_verified": len(conditions),
        })
    return rows


def classify_fidelity_availability(run: dict[str, Any]) -> tuple[str, float | None]:
    """Classify candidate-fidelity availability using the metric's actual contract."""
    if not _generation_attempted(run):
        return "not_generated", None
    candidate = run.get("candidate_response_text")
    if not isinstance(candidate, str) or not candidate.strip():
        return "candidate_text_missing_despite_generation_trace", None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return "candidate_invalid_json", None
    if not isinstance(parsed, dict):
        return "candidate_json_not_object", None
    if "model_drivers" not in parsed:
        return "model_drivers_missing", None
    if not isinstance(parsed.get("model_drivers"), list):
        return "model_drivers_not_list", None

    expected_features = {
        str(item.get("feature") or item.get("feature_group"))
        for dataset in (run.get("model_evidence") or {}).get("datasets", [])
        if isinstance(dataset, dict)
        for alert in dataset.get("top_alerts", [])
        if isinstance(alert, dict)
        for item in (alert.get("local_attribution") or {}).get("top_features", [])
        if isinstance(item, dict) and (item.get("feature") or item.get("feature_group"))
    }
    if not expected_features:
        return "reference_attribution_missing", None
    fidelity = evaluate_run(run)["candidate_explanation_fidelity"]
    if fidelity is None:
        raise RuntimeError(
            f"Fidelity contract unexpectedly unavailable for {run.get('request_id')}."
        )
    return "available", float(fidelity)


def build_fidelity_rows(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one non-textual fidelity availability record per condition run."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        reason, fidelity = classify_fidelity_availability(run)
        attempted = _generation_attempted(run)
        rows.append({
            "dataset_id": _dataset_id(run),
            "sample_id": str(run["sample_id"]),
            "condition_id": _condition_id(run),
            "request_id": str(run["request_id"]),
            "status": str(run["status"]),
            "generation_attempted": attempted,
            "fidelity_availability": "available" if reason == "available" else "unavailable",
            "fidelity_reason": reason,
            "candidate_explanation_fidelity": fidelity,
        })
    return rows


def classify_disposition(run: dict[str, Any]) -> str:
    """Map a raw run to exactly one computational disposition."""
    attempted = _generation_attempted(run)
    status = run.get("status")
    if status == "suppressed" and not attempted:
        return "pre_generation_suppressed"
    if status == "blocked" and attempted:
        return "generated_blocked"
    if status == "completed" and attempted:
        return "generated_released"
    raise ValueError(
        f"Unclassifiable disposition for {run.get('request_id')}: "
        f"status={status!r}, generation_attempted={attempted}."
    )


def _nonnegative_number(value: Any, field: str, request_id: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"Invalid {field} for {request_id}: {value!r}.")
    return number


def build_latency_rows(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-run latency telemetry with an explicit disposition denominator."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        trace = run.get("trace") or {}
        request_id = run.get("request_id")
        attempted = _generation_attempted(run)
        generation_latency = _nonnegative_number(
            trace.get("generation_latency_ms"), "generation_latency_ms", request_id
        )
        if not attempted:
            generation_latency = None
        rows.append({
            "dataset_id": _dataset_id(run),
            "sample_id": str(run["sample_id"]),
            "condition_id": _condition_id(run),
            "request_id": str(request_id),
            "status": str(run["status"]),
            "disposition": classify_disposition(run),
            "generation_attempted": attempted,
            "processing_latency_ms": _nonnegative_number(
                trace.get("end_to_end_latency_ms", trace.get("latency_ms")),
                "processing_latency_ms",
                request_id,
            ),
            "generation_latency_ms": generation_latency,
            "retrieval_latency_ms": _nonnegative_number(
                trace.get("retrieval_latency_ms"), "retrieval_latency_ms", request_id
            ),
            "validation_latency_ms": _nonnegative_number(
                trace.get("validation_latency_ms"), "validation_latency_ms", request_id
            ),
            "cpu_time_ms": _nonnegative_number(
                trace.get("cpu_time_ms"), "cpu_time_ms", request_id
            ),
            "memory_rss_delta_bytes": (
                float(trace["memory_rss_delta_bytes"])
                if trace.get("memory_rss_delta_bytes") is not None
                else None
            ),
        })
    return rows


def numeric_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    """Return a deterministic descriptive summary without inferential claims."""
    clean = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=float,
    )
    if clean.size == 0:
        return {
            "n": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "n": int(clean.size),
        "mean": round(float(mean(clean)), 6),
        "standard_deviation": (
            round(float(clean.std(ddof=1)), 6) if clean.size > 1 else None
        ),
        "minimum": round(float(clean.min()), 6),
        "p25": round(float(np.percentile(clean, 25)), 6),
        "median": round(float(median(clean)), 6),
        "p75": round(float(np.percentile(clean, 75)), 6),
        "p95": round(float(np.percentile(clean, 95)), 6),
        "maximum": round(float(clean.max()), 6),
    }


def gate_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise case-level detector scores and generation eligibility by dataset."""
    result: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset_id"]) for row in rows}):
        selected = [row for row in rows if row["dataset_id"] == dataset]
        eligible = sum(bool(row["guardrailed_generation_eligible"]) for row in selected)
        result.append({
            "dataset_id": dataset,
            "case_count": len(selected),
            "generation_gate": GENERATION_GATE,
            "at_or_above_gate_count": eligible,
            "below_gate_count": len(selected) - eligible,
            "at_or_above_gate_percent": round(100.0 * eligible / len(selected), 6),
            "below_gate_percent": round(
                100.0 * (len(selected) - eligible) / len(selected), 6
            ),
            "detector_score": numeric_summary(row["detector_score"] for row in selected),
        })
    return result


def fidelity_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise fidelity availability by dataset, condition, and explicit reason."""
    generated = [row for row in rows if row["generation_attempted"]]
    available = [row for row in generated if row["fidelity_reason"] == "available"]
    by_group: list[dict[str, Any]] = []
    groups = sorted({(row["dataset_id"], row["condition_id"]) for row in rows})
    for dataset, condition in groups:
        selected = [
            row
            for row in rows
            if row["dataset_id"] == dataset and row["condition_id"] == condition
        ]
        selected_generated = [row for row in selected if row["generation_attempted"]]
        counts = Counter(row["fidelity_reason"] for row in selected_generated)
        available_count = counts.pop("available", 0)
        by_group.append({
            "dataset_id": dataset,
            "condition_id": condition,
            "run_count": len(selected),
            "generated_count": len(selected_generated),
            "not_generated_count": len(selected) - len(selected_generated),
            "fidelity_available_count": available_count,
            "fidelity_unavailable_count": len(selected_generated) - available_count,
            "fidelity_available_percent_of_generated": round(
                100.0 * available_count / len(selected_generated), 6
            ) if selected_generated else None,
            "unavailable_reason_counts": dict(sorted(counts.items())),
        })
    reason_counts = Counter(row["fidelity_reason"] for row in generated)
    reason_counts.pop("available", None)
    return {
        "all_run_count": len(rows),
        "generated_count": len(generated),
        "not_generated_count": len(rows) - len(generated),
        "fidelity_available_count": len(available),
        "fidelity_unavailable_count": len(generated) - len(available),
        "fidelity_available_percent_of_generated": round(
            100.0 * len(available) / len(generated), 6
        ),
        "unavailable_reason_counts": dict(sorted(reason_counts.items())),
        "by_dataset_condition": by_group,
    }


def latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise latency within disposition and within generated-only denominators."""
    metrics = (
        "processing_latency_ms",
        "generation_latency_ms",
        "retrieval_latency_ms",
        "validation_latency_ms",
        "cpu_time_ms",
        "memory_rss_delta_bytes",
    )
    by_disposition: list[dict[str, Any]] = []
    groups = sorted({
        (row["dataset_id"], row["condition_id"], row["disposition"]) for row in rows
    })
    for dataset, condition, disposition in groups:
        selected = [
            row for row in rows
            if row["dataset_id"] == dataset
            and row["condition_id"] == condition
            and row["disposition"] == disposition
        ]
        by_disposition.append({
            "dataset_id": dataset,
            "condition_id": condition,
            "disposition": disposition,
            "run_count": len(selected),
            **{metric: numeric_summary(row[metric] for row in selected) for metric in metrics},
        })

    generation_only: list[dict[str, Any]] = []
    condition_groups = sorted({(row["dataset_id"], row["condition_id"]) for row in rows})
    for dataset, condition in condition_groups:
        selected = [
            row for row in rows
            if row["dataset_id"] == dataset
            and row["condition_id"] == condition
            and row["generation_attempted"]
        ]
        generation_only.append({
            "dataset_id": dataset,
            "condition_id": condition,
            "generated_count": len(selected),
            "generation_latency_ms": numeric_summary(
                row["generation_latency_ms"] for row in selected
            ),
            "processing_latency_ms": numeric_summary(
                row["processing_latency_ms"] for row in selected
            ),
        })

    disposition_counts = Counter(row["disposition"] for row in rows)
    return {
        "all_run_count": len(rows),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "by_dataset_condition_disposition": by_disposition,
        "generation_only_by_dataset_condition": generation_only,
    }


def build_posthoc_analysis(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build all three planned post-hoc descriptive analyses."""
    gate_rows = build_gate_case_rows(runs)
    fidelity_rows = build_fidelity_rows(runs)
    latency_rows = build_latency_rows(runs)
    payload = {
        "schema_version": "1.0",
        "analysis_id": POSTHOC_ANALYSIS_ID,
        "scientific_status": "post_hoc_descriptive_analysis_no_new_model_calls",
        "study_id": EXPECTED_STUDY_ID,
        "scope_boundary": (
            "Derived only from accepted immutable raw records; no model, retrieval, "
            "guardrail, panel, protocol, or V4 re-execution or alteration."
        ),
        "gate_analysis": {
            "unit": "unique evaluation case within dataset",
            "purpose": "describe exposure to the 0.50 pre-generation gate",
            "claim_boundary": (
                "The gate split is descriptive and is not an estimate of detector "
                "accuracy or a causal guardrail effect."
            ),
            "summary": gate_summary(gate_rows),
        },
        "fidelity_missingness_analysis": {
            "unit": "generated condition record within dataset and condition",
            "claim_boundary": (
                "Availability describes whether the automatic response-to-TreeSHAP "
                "comparison could be computed; it is not a human correctness score."
            ),
            "summary": fidelity_summary(fidelity_rows),
        },
        "latency_by_disposition_analysis": {
            "unit": "condition record within dataset, condition, and disposition",
            "claim_boundary": (
                "These are post-hoc descriptive timings. Disposition groups are "
                "outcomes of the pipeline and are not randomized comparison groups."
            ),
            "summary": latency_summary(latency_rows),
        },
    }
    return {
        "summary": payload,
        "gate_rows": gate_rows,
        "fidelity_rows": fidelity_rows,
        "latency_rows": latency_rows,
    }
