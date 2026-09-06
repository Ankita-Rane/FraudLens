"""Validate and aggregate every locked batch in one follow-up study.

This tool is read-only with respect to raw experiment directories.  It refuses to
publish a study result unless every protocol batch is present exactly once, every
batch is paired across all six conditions, and the total call count matches the
locked protocol.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc_evaluation import aggregate_runs, validate_paired_design  # noqa: E402
from src.followup_conditions import (  # noqa: E402
    FOLLOWUP_CONDITION_IDS,
    condition_registry_sha256,
)
from src.statistical_analysis import paired_followup_analysis  # noqa: E402


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument(
        "--experiments-root", type=Path,
        default=PROJECT_ROOT / "artifacts" / "experiments" / "followup",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="Defaults to artifacts/results/followup/<study-id>.json.",
    )
    return parser.parse_args()


def _load_batch_runs(directory: Path) -> list[dict[str, Any]]:
    paths = sorted((directory / "raw_runs").glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No raw runs found in {directory}.")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def main() -> None:
    args = _arguments()
    protocol_path = args.protocol_lock.resolve()
    experiments_root = args.experiments_root.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("scientific_status") != "locked_followup_protocol":
        raise RuntimeError("Aggregation requires a locked follow-up protocol.")
    if args.study_id != protocol.get("study_id"):
        raise RuntimeError("--study-id differs from the protocol.")
    if protocol.get("condition_registry_sha256") != condition_registry_sha256():
        raise RuntimeError("Condition registry differs from the locked protocol.")
    diagnostic = protocol.get("retrieval_diagnostic", {})
    if diagnostic.get("scientific_status") != "completed_scope_safe_known_item_diagnostic" or not diagnostic.get("audit_sha256"):
        raise RuntimeError("Protocol lacks verified scope-safe retrieval diagnostic evidence.")
    if protocol.get("planned_call_count") != 3300 or protocol.get("repeats") != 1:
        raise RuntimeError("Aggregation requires the exact 3,300-call, one-execution design.")

    eligible_status = "locked_followup_qwen_raw_outputs_pending_analysis"
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    failed_attempts: list[dict[str, Any]] = []
    for path in sorted(experiments_root.glob("*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("study_id") != args.study_id:
            continue
        if payload.get("scientific_status") != eligible_status:
            failed_attempts.append({
                "batch_id": payload.get("batch_id"),
                "experiment_id": payload.get("experiment_id"),
                "scientific_status": payload.get("scientific_status"),
                "manifest_path": str(path.relative_to(PROJECT_ROOT)),
                "manifest_sha256": _sha256(path),
            })
            continue
        batch_id = str(payload.get("batch_id"))
        if batch_id in manifests:
            raise RuntimeError(f"Duplicate completed batch: {batch_id}.")
        manifests[batch_id] = (path, payload)

    expected_batches = {
        str(row["batch_id"]): row for row in protocol.get("batches", [])
    }
    missing = sorted(set(expected_batches) - set(manifests))
    unexpected = sorted(set(manifests) - set(expected_batches))
    if missing or unexpected:
        raise RuntimeError(
            f"Batch coverage mismatch; missing={missing}, unexpected={unexpected}."
        )

    all_runs: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    for batch_id in expected_batches:
        path, manifest = manifests[batch_id]
        expected = expected_batches[batch_id]
        if manifest.get("sample_ids") != expected.get("sample_ids"):
            raise RuntimeError(f"Sample membership/order differs for {batch_id}.")
        if manifest.get("global_cell_offset") != expected.get("cell_offset"):
            raise RuntimeError(f"Counterbalance offset differs for {batch_id}.")
        if manifest.get("inputs", {}).get("protocol_sha256") != _sha256(protocol_path):
            raise RuntimeError(f"Protocol hash differs for {batch_id}.")
        runs = _load_batch_runs(path.parent)
        planned = int(expected["planned_calls"])
        if len(runs) != planned or manifest.get("completed_run_count") != planned:
            raise RuntimeError(f"Run count differs from protocol for {batch_id}.")
        pairing = validate_paired_design(
            runs, expected_conditions=FOLLOWUP_CONDITION_IDS
        )
        if not pairing["paired"]:
            raise RuntimeError(f"Incomplete six-condition cells in {batch_id}.")
        all_runs.extend(runs)
        source_manifests.append({
            "batch_id": batch_id,
            "experiment_id": manifest.get("experiment_id"),
            "manifest_path": str(path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": _sha256(path),
            "run_count": len(runs),
        })

    planned_total = int(protocol.get("planned_call_count", -1))
    if len(all_runs) != planned_total:
        raise RuntimeError(
            f"Study has {len(all_runs)} raw calls but protocol requires {planned_total}."
        )
    request_ids = [str(run.get("request_id")) for run in all_runs]
    if len(request_ids) != len(set(request_ids)):
        raise RuntimeError("Duplicate request IDs exist across study batches.")

    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "locked_followup_aggregated_results",
        "study_id": args.study_id,
        "protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": _sha256(protocol_path),
        "batch_count": len(source_manifests),
        "case_count": int(protocol["sample_count"]),
        "raw_call_count": len(all_runs),
        "condition_ids": list(FOLLOWUP_CONDITION_IDS),
        "ground_truth_withheld_from_generation": True,
        "source_manifests": source_manifests,
        "failed_attempts_excluded": failed_attempts,
        "batch_automatic_summaries": aggregate_runs(all_runs),
        "paired_statistical_analysis": paired_followup_analysis(all_runs),
    }
    output = args.output or (
        PROJECT_ROOT / "artifacts" / "results" / "followup" / f"{args.study_id}.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregated evidence: {output}")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(PROJECT_ROOT)),
        "scientific_status": payload["scientific_status"],
        "batch_count": payload["batch_count"],
        "raw_call_count": payload["raw_call_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
