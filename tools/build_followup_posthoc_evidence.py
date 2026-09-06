"""Build hashed post-hoc descriptive evidence from the accepted V4 raw-run set."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import json
import logging
import sys

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.followup_conditions import FOLLOWUP_CONDITION_IDS  # noqa: E402
from src.followup_posthoc_analysis import (  # noqa: E402
    EXPECTED_CALL_COUNT,
    EXPECTED_CASE_COUNT,
    EXPECTED_STUDY_ID,
    GENERATION_GATE,
    POSTHOC_ANALYSIS_ID,
    build_posthoc_analysis,
    file_sha256,
    load_v4_locked_runs,
)


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "artifacts/results/followup/"
    / f"{EXPECTED_STUDY_ID}-CORRECTED-V4.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/results/followup/posthoc" / POSTHOC_ANALYSIS_ID
)
LOG_PATH = PROJECT_ROOT / "log" / "application.log"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _configure_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_followup_posthoc_evidence")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == LOG_PATH.resolve()
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = _configure_logging()


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}.")
    fieldnames = list(materialized[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _flatten_fidelity_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in summary["by_dataset_condition"]:
        base = {
            key: group[key]
            for key in (
                "dataset_id",
                "condition_id",
                "run_count",
                "generated_count",
                "not_generated_count",
                "fidelity_available_count",
                "fidelity_unavailable_count",
                "fidelity_available_percent_of_generated",
            )
        }
        reasons = group["unavailable_reason_counts"]
        if reasons:
            for reason, count in reasons.items():
                rows.append({**base, "unavailable_reason": reason, "reason_count": count})
        else:
            rows.append({**base, "unavailable_reason": "none", "reason_count": 0})
    return rows


def _flatten_latency_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in summary["by_dataset_condition_disposition"]:
        for metric in (
            "processing_latency_ms",
            "generation_latency_ms",
            "retrieval_latency_ms",
            "validation_latency_ms",
            "cpu_time_ms",
            "memory_rss_delta_bytes",
        ):
            values = group[metric]
            rows.append({
                "dataset_id": group["dataset_id"],
                "condition_id": group["condition_id"],
                "disposition": group["disposition"],
                "run_count": group["run_count"],
                "metric": metric,
                **values,
            })
    return rows


def _flatten_generation_only(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in summary["generation_only_by_dataset_condition"]:
        for metric in ("generation_latency_ms", "processing_latency_ms"):
            rows.append({
                "dataset_id": group["dataset_id"],
                "condition_id": group["condition_id"],
                "generated_count": group["generated_count"],
                "metric": metric,
                **group[metric],
            })
    return rows


def _plot_gate_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    datasets = sorted({row["dataset_id"] for row in rows})
    figure, axes = plt.subplots(1, len(datasets), figsize=(11.2, 4.4), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    bins = np.linspace(0.0, 1.0, 21)
    for axis, dataset in zip(axes, datasets, strict=True):
        selected = [row for row in rows if row["dataset_id"] == dataset]
        scores = [row["detector_score"] for row in selected]
        eligible = sum(row["guardrailed_generation_eligible"] for row in selected)
        axis.hist(scores, bins=bins, color="#2F6B9A", edgecolor="white", alpha=0.9)
        axis.axvline(
            GENERATION_GATE,
            color="#B22222",
            linestyle="--",
            linewidth=2,
            label="Generation gate = 0.50",
        )
        axis.set_title(
            f"{dataset}: {eligible} eligible, {len(selected) - eligible} suppressed"
        )
        axis.set_xlabel("Detector fraud probability")
        axis.set_ylabel("Unique evaluation cases")
        axis.set_xlim(0.0, 1.0)
        axis.legend(frameon=False, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Case-level detector scores relative to the pre-generation gate")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_latency(path: Path, latency_summary: dict[str, Any]) -> None:
    rows = latency_summary["by_dataset_condition_disposition"]
    datasets = sorted({row["dataset_id"] for row in rows})
    dispositions = (
        "pre_generation_suppressed",
        "generated_blocked",
        "generated_released",
    )
    colors = {
        "pre_generation_suppressed": "#8C8C8C",
        "generated_blocked": "#D97706",
        "generated_released": "#2F6B9A",
    }
    figure, axes = plt.subplots(1, len(datasets), figsize=(13.0, 5.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets, strict=True):
        selected = [row for row in rows if row["dataset_id"] == dataset]
        labels: list[str] = []
        positions: list[float] = []
        offset = 0
        for condition in FOLLOWUP_CONDITION_IDS:
            condition_rows = {
                row["disposition"]: row
                for row in selected
                if row["condition_id"] == condition
            }
            for index, disposition in enumerate(dispositions):
                row = condition_rows.get(disposition)
                if not row:
                    continue
                median_value = row["processing_latency_ms"]["median"]
                p95_value = row["processing_latency_ms"]["p95"]
                if median_value is None:
                    continue
                x = offset + index * 0.24
                axis.bar(
                    x,
                    median_value,
                    width=0.21,
                    color=colors[disposition],
                    label=disposition.replace("_", " "),
                )
                if p95_value is not None:
                    axis.scatter(x, p95_value, color="black", s=12, zorder=3)
            labels.append(condition.replace("_", "\n"))
            positions.append(offset + 0.24)
            offset += 1.0
        axis.set_title(dataset)
        axis.set_xticks(positions, labels, fontsize=8)
        axis.set_ylabel("Processing latency (ms, log scale)")
        axis.set_yscale("log")
        axis.grid(axis="y", which="both", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle("Median processing latency by computational disposition (dots = p95)")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _safe_notes(summary: dict[str, Any]) -> str:
    gate = {row["dataset_id"]: row for row in summary["gate_analysis"]["summary"]}
    fidelity = summary["fidelity_missingness_analysis"]["summary"]
    disposition = summary["latency_by_disposition_analysis"]["summary"]
    return "\n".join([
        "POST-HOC DESCRIPTIVE EVIDENCE: THESIS REPORTING NOTES",
        "",
        f"Analysis ID: {POSTHOC_ANALYSIS_ID}",
        "No new model calls, retrieval, guardrail execution, or V4 modification occurred.",
        "",
        (
            f"ULB: {gate['ULB']['at_or_above_gate_count']} of "
            f"{gate['ULB']['case_count']} cases were at or above the 0.50 generation "
            f"gate; {gate['ULB']['below_gate_count']} were below it."
        ),
        (
            f"Sparkov: {gate['Sparkov']['at_or_above_gate_count']} of "
            f"{gate['Sparkov']['case_count']} cases were at or above the 0.50 generation "
            f"gate; {gate['Sparkov']['below_gate_count']} were below it."
        ),
        (
            f"Explanation fidelity was computable for "
            f"{fidelity['fidelity_available_count']} of {fidelity['generated_count']} "
            f"generated condition records; {fidelity['fidelity_unavailable_count']} "
            "generated records were unavailable for the explicitly tabulated reasons."
        ),
        (
            "Disposition counts across the 3,300 condition records: "
            + ", ".join(
                f"{name}={count}"
                for name, count in disposition["disposition_counts"].items()
            )
            + "."
        ),
        "",
        "Claim boundaries:",
        "- The gate split is descriptive, not a detector-accuracy estimate.",
        "- Fidelity availability is not a human assessment of correctness.",
        "- Disposition-stratified latency groups are outcomes, not randomized groups.",
        "- Do not add counts from overlapping condition contrasts as independent cases.",
    ]) + "\n"


def main() -> None:
    args = _arguments()
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite post-hoc evidence: {output}")
    LOGGER.info(
        "posthoc_start analysis_id=%s source=%s output=%s",
        POSTHOC_ANALYSIS_ID,
        args.source_results,
        output,
    )
    try:
        runs, provenance = load_v4_locked_runs(PROJECT_ROOT, args.source_results)
        LOGGER.info(
            "posthoc_inputs_verified manifests=%s calls=%s cases=%s raw_set_sha256=%s",
            provenance["accepted_manifest_count"],
            provenance["accepted_raw_run_count"],
            provenance["case_count"],
            provenance["accepted_raw_run_set_sha256"],
        )
        built = build_posthoc_analysis(runs)
        summary = built["summary"]
        output.mkdir(parents=True)

        files = {
            "summary": output / "posthoc_summary.json",
            "gate_case_scores": output / "gate_case_scores.csv",
            "fidelity_availability_records": output / "fidelity_availability_records.csv",
            "fidelity_missingness_summary": output / "fidelity_missingness_summary.csv",
            "latency_records": output / "latency_records.csv",
            "latency_by_disposition_summary": output / "latency_by_disposition_summary.csv",
            "generation_only_latency_summary": output / "generation_only_latency_summary.csv",
            "gate_figure": output / "gate_score_distribution.png",
            "latency_figure": output / "latency_by_disposition.png",
            "reporting_notes": output / "THESIS_REPORTING_NOTES.txt",
        }
        _json_write(files["summary"], summary)
        _csv_write(files["gate_case_scores"], built["gate_rows"])
        _csv_write(files["fidelity_availability_records"], built["fidelity_rows"])
        _csv_write(
            files["fidelity_missingness_summary"],
            _flatten_fidelity_summary(
                summary["fidelity_missingness_analysis"]["summary"]
            ),
        )
        _csv_write(files["latency_records"], built["latency_rows"])
        latency = summary["latency_by_disposition_analysis"]["summary"]
        _csv_write(
            files["latency_by_disposition_summary"],
            _flatten_latency_summary(latency),
        )
        _csv_write(
            files["generation_only_latency_summary"],
            _flatten_generation_only(latency),
        )
        _plot_gate_scores(files["gate_figure"], built["gate_rows"])
        _plot_latency(files["latency_figure"], latency)
        files["reporting_notes"].write_text(_safe_notes(summary), encoding="utf-8")

        output_manifest = [
            {
                "role": role,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for role, path in sorted(files.items())
        ]
        audit = {
            "schema_version": "1.0",
            "analysis_id": POSTHOC_ANALYSIS_ID,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "post_hoc_descriptive_evidence_audit_passed",
            "no_new_model_calls": True,
            "v4_modified": False,
            "source_provenance": provenance,
            "code_provenance": {
                "analysis_module": "src/followup_posthoc_analysis.py",
                "analysis_module_sha256": file_sha256(
                    PROJECT_ROOT / "src/followup_posthoc_analysis.py"
                ),
                "build_tool": "tools/build_followup_posthoc_evidence.py",
                "build_tool_sha256": file_sha256(Path(__file__).resolve()),
            },
            "verified_invariants": {
                "accepted_manifest_count": 55,
                "case_count": EXPECTED_CASE_COUNT,
                "raw_call_count": EXPECTED_CALL_COUNT,
                "condition_count": len(FOLLOWUP_CONDITION_IDS),
                "generated_count": summary["fidelity_missingness_analysis"]["summary"][
                    "generated_count"
                ],
                "not_generated_count": summary["fidelity_missingness_analysis"][
                    "summary"
                ]["not_generated_count"],
                "fidelity_available_count": summary["fidelity_missingness_analysis"][
                    "summary"
                ]["fidelity_available_count"],
                "fidelity_unavailable_count": summary["fidelity_missingness_analysis"][
                    "summary"
                ]["fidelity_unavailable_count"],
            },
            "outputs": output_manifest,
        }
        audit_path = output / "audit_manifest.json"
        if file_sha256(args.source_results.resolve()) != provenance["source_results_sha256"]:
            raise RuntimeError("V4 source changed while the post-hoc evidence was built.")
        _json_write(audit_path, audit)
        LOGGER.info(
            "posthoc_complete output=%s audit_sha256=%s generated=%s "
            "fidelity_available=%s fidelity_unavailable=%s",
            output,
            file_sha256(audit_path),
            audit["verified_invariants"]["generated_count"],
            audit["verified_invariants"]["fidelity_available_count"],
            audit["verified_invariants"]["fidelity_unavailable_count"],
        )
        print(json.dumps({
            "output_directory": str(output.relative_to(PROJECT_ROOT)),
            "analysis_id": POSTHOC_ANALYSIS_ID,
            "audit_manifest": str(audit_path.relative_to(PROJECT_ROOT)),
            "audit_manifest_sha256": file_sha256(audit_path),
            "verified_invariants": audit["verified_invariants"],
        }, indent=2))
    except Exception:
        LOGGER.exception("posthoc_failed analysis_id=%s", POSTHOC_ANALYSIS_ID)
        raise


if __name__ == "__main__":
    main()
