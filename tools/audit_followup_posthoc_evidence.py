"""Independently verify the post-hoc descriptive evidence bundle and its hashes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.followup_conditions import FOLLOWUP_CONDITION_IDS  # noqa: E402
from src.followup_posthoc_analysis import (  # noqa: E402
    EXPECTED_BATCH_COUNT,
    EXPECTED_CALL_COUNT,
    EXPECTED_CASE_COUNT,
    POSTHOC_ANALYSIS_ID,
    file_sha256,
)


DEFAULT_DIRECTORY = (
    PROJECT_ROOT / "artifacts/results/followup/posthoc" / POSTHOC_ANALYSIS_ID
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_evidence(directory: Path) -> dict[str, Any]:
    """Verify source/code/output hashes and the registered denominator contracts."""
    root = directory.resolve()
    manifest_path = root / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append({
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        })
        if not passed:
            errors.append(f"{name}: observed={observed!r}, expected={expected!r}")

    check(
        "analysis_id",
        manifest.get("analysis_id") == POSTHOC_ANALYSIS_ID,
        manifest.get("analysis_id"),
        POSTHOC_ANALYSIS_ID,
    )
    check(
        "scientific_status",
        manifest.get("scientific_status")
        == "post_hoc_descriptive_evidence_audit_passed",
        manifest.get("scientific_status"),
        "post_hoc_descriptive_evidence_audit_passed",
    )
    check("no_new_model_calls", manifest.get("no_new_model_calls") is True,
          manifest.get("no_new_model_calls"), True)
    check("v4_modified", manifest.get("v4_modified") is False,
          manifest.get("v4_modified"), False)

    provenance = manifest.get("source_provenance") or {}
    source_results = PROJECT_ROOT / str(provenance.get("source_results_path"))
    protocol = PROJECT_ROOT / str(provenance.get("protocol_path"))
    check(
        "source_results_hash",
        source_results.is_file()
        and file_sha256(source_results) == provenance.get("source_results_sha256"),
        file_sha256(source_results) if source_results.is_file() else None,
        provenance.get("source_results_sha256"),
    )
    check(
        "protocol_hash",
        protocol.is_file() and file_sha256(protocol) == provenance.get("protocol_sha256"),
        file_sha256(protocol) if protocol.is_file() else None,
        provenance.get("protocol_sha256"),
    )

    code = manifest.get("code_provenance") or {}
    for role in ("analysis_module", "build_tool"):
        path = PROJECT_ROOT / str(code.get(role))
        expected_hash = code.get(f"{role}_sha256")
        check(
            f"{role}_hash",
            path.is_file() and file_sha256(path) == expected_hash,
            file_sha256(path) if path.is_file() else None,
            expected_hash,
        )

    for output in manifest.get("outputs") or []:
        path = PROJECT_ROOT / str(output.get("path"))
        check(
            f"output_hash:{output.get('role')}",
            path.is_file()
            and path.stat().st_size == output.get("bytes")
            and file_sha256(path) == output.get("sha256"),
            {
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": file_sha256(path) if path.is_file() else None,
            },
            {"exists": True, "bytes": output.get("bytes"), "sha256": output.get("sha256")},
        )

    invariants = manifest.get("verified_invariants") or {}
    check("accepted_manifest_count", invariants.get("accepted_manifest_count") == EXPECTED_BATCH_COUNT,
          invariants.get("accepted_manifest_count"), EXPECTED_BATCH_COUNT)
    check("case_count", invariants.get("case_count") == EXPECTED_CASE_COUNT,
          invariants.get("case_count"), EXPECTED_CASE_COUNT)
    check("raw_call_count", invariants.get("raw_call_count") == EXPECTED_CALL_COUNT,
          invariants.get("raw_call_count"), EXPECTED_CALL_COUNT)
    check("condition_count", invariants.get("condition_count") == len(FOLLOWUP_CONDITION_IDS),
          invariants.get("condition_count"), len(FOLLOWUP_CONDITION_IDS))

    gate_rows = _csv_rows(root / "gate_case_scores.csv")
    gate_counts = Counter(
        (row["dataset_id"], row["gate_band"]) for row in gate_rows
    )
    check("gate_csv_rows", len(gate_rows) == EXPECTED_CASE_COUNT,
          len(gate_rows), EXPECTED_CASE_COUNT)
    check("gate_unique_samples", len({row["sample_id"] for row in gate_rows}) == EXPECTED_CASE_COUNT,
          len({row["sample_id"] for row in gate_rows}), EXPECTED_CASE_COUNT)
    for dataset, eligible, below in (("ULB", 68, 82), ("Sparkov", 201, 199)):
        check(f"{dataset}_gate_eligible",
              gate_counts[(dataset, "at_or_above_gate")] == eligible,
              gate_counts[(dataset, "at_or_above_gate")], eligible)
        check(f"{dataset}_gate_below", gate_counts[(dataset, "below_gate")] == below,
              gate_counts[(dataset, "below_gate")], below)

    fidelity_rows = _csv_rows(root / "fidelity_availability_records.csv")
    generated = [row for row in fidelity_rows if row["generation_attempted"] == "True"]
    available = [row for row in generated if row["fidelity_availability"] == "available"]
    unavailable = [row for row in generated if row["fidelity_availability"] == "unavailable"]
    check("fidelity_csv_rows", len(fidelity_rows) == EXPECTED_CALL_COUNT,
          len(fidelity_rows), EXPECTED_CALL_COUNT)
    check("generated_count", len(generated) == 2457, len(generated), 2457)
    check("not_generated_count", len(fidelity_rows) - len(generated) == 843,
          len(fidelity_rows) - len(generated), 843)
    check("fidelity_available_count", len(available) == 1610, len(available), 1610)
    check("fidelity_unavailable_count", len(unavailable) == 847, len(unavailable), 847)
    unavailable_reasons = Counter(row["fidelity_reason"] for row in unavailable)
    check(
        "fidelity_unavailable_reasons_partition",
        sum(unavailable_reasons.values()) == 847,
        dict(sorted(unavailable_reasons.items())),
        "counts sum to 847",
    )

    latency_rows = _csv_rows(root / "latency_records.csv")
    dispositions = Counter(row["disposition"] for row in latency_rows)
    check("latency_csv_rows", len(latency_rows) == EXPECTED_CALL_COUNT,
          len(latency_rows), EXPECTED_CALL_COUNT)
    check("disposition_partition", sum(dispositions.values()) == EXPECTED_CALL_COUNT,
          sum(dispositions.values()), EXPECTED_CALL_COUNT)
    check("suppressed_disposition", dispositions["pre_generation_suppressed"] == 843,
          dispositions["pre_generation_suppressed"], 843)
    check(
        "suppressed_generation_latency_blank",
        all(
            not row["generation_latency_ms"]
            for row in latency_rows
            if row["disposition"] == "pre_generation_suppressed"
        ),
        "all blank" if all(
            not row["generation_latency_ms"]
            for row in latency_rows
            if row["disposition"] == "pre_generation_suppressed"
        ) else "at least one nonblank",
        "all blank",
    )

    return {
        "schema_version": "1.0",
        "analysis_id": POSTHOC_ANALYSIS_ID,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "evidence_directory": str(root.relative_to(PROJECT_ROOT)),
        "audit_manifest_sha256": file_sha256(manifest_path),
        "check_count": len(checks),
        "checks_passed": sum(row["passed"] for row in checks),
        "errors": errors,
        "checks": checks,
        "observed_fidelity_unavailable_reasons": dict(sorted(unavailable_reasons.items())),
        "observed_disposition_counts": dict(sorted(dispositions.items())),
    }


def main() -> None:
    args = _arguments()
    result = audit_evidence(args.evidence_directory)
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite verification evidence: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "status": result["status"],
        "check_count": result["check_count"],
        "checks_passed": result["checks_passed"],
        "errors": result["errors"],
        "audit_manifest_sha256": result["audit_manifest_sha256"],
        "verification_output": str(args.output.resolve().relative_to(PROJECT_ROOT))
        if args.output else None,
    }, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
