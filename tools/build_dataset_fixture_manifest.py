"""Lock row selectors needed to reconstruct local ULB/Sparkov CSV fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset_fixture_support import (
    DATASET_CONFIG,
    FIXTURE_PATHS,
    SELECTOR_COLUMNS,
    reconstruct_fixture,
    sha256_file,
    write_fixture,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "datasets" / "evaluation" / "dataset_fixture_selection_manifest.json"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fixture_format(relative_path: str) -> tuple[str, str]:
    if relative_path == "datasets/sparkov/runtime_transaction_samples.csv":
        return "%.6f", "space"
    return "%.10f", "iso_t"


def fixture_column_formats(relative_path: str) -> dict[str, str]:
    if relative_path != "datasets/sparkov/runtime_transaction_samples.csv":
        return {}
    return {
        "amt": ".2f",
        "log_amt": ".6f",
        "city_pop": "d",
        "lat": ".4f",
        "long": ".4f",
        "merch_lat": ".6f",
        "merch_long": ".6f",
        "age": ".6f",
        "distance_km": ".6f",
        "expected_fraud_probability": ".6f",
        "decision_threshold": ".6f",
    }


def detector_paths(relative_path: str, dataset_id: str) -> tuple[str, str]:
    dataset_key = DATASET_CONFIG[dataset_id]["dataset_key"]
    base = (
        f"artifacts/followup_detectors/{dataset_key}"
        if dataset_id == "ULB"
        and relative_path.endswith("/followup_transaction_samples.csv")
        else f"artifacts/{dataset_key}"
    )
    return f"{base}/selected_pipeline.joblib", f"{base}/test_metrics.csv"


def build_specification(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot lock missing fixture {relative_path}; build the manifest before removal."
        )
    frame = pd.read_csv(path, dtype={"sample_id": str, "test_case_type": str})
    dataset_values = frame["dataset_id"].drop_duplicates().tolist()
    if len(dataset_values) != 1 or dataset_values[0] not in DATASET_CONFIG:
        raise ValueError(f"Unexpected dataset_id in {relative_path}: {dataset_values}")
    dataset_id = dataset_values[0]
    float_format, timestamp_style = fixture_format(relative_path)
    pipeline_path, metrics_path = detector_paths(relative_path, dataset_id)
    rows = []
    for record in frame.to_dict(orient="records"):
        selector = {
            "sample_id": record["sample_id"],
            "test_case_type": record["test_case_type"],
            "source_test_row_index": int(record["source_test_row_index"]),
        }
        if "evaluation_cohort" in frame.columns:
            selector["evaluation_cohort"] = record["evaluation_cohort"]
        rows.append(selector)
    specification = {
        "path": relative_path,
        "dataset_id": dataset_id,
        "expected_rows": len(frame),
        "expected_sha256": sha256_file(path),
        "columns": frame.columns.tolist(),
        "decision_threshold": float(frame["decision_threshold"].iloc[0]),
        "pipeline_path": pipeline_path,
        "pipeline_sha256": sha256_file(PROJECT_ROOT / pipeline_path),
        "metrics_path": metrics_path,
        "metrics_sha256": sha256_file(PROJECT_ROOT / metrics_path),
        "float_format": float_format,
        "column_formats": fixture_column_formats(relative_path),
        "timestamp_style": timestamp_style,
        "rows": rows,
    }

    rebuilt = reconstruct_fixture(specification)
    temporary = path.with_suffix(path.suffix + ".manifest-check.tmp")
    try:
        write_fixture(rebuilt, temporary, specification)
        rebuilt_hash = sha256_file(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    if rebuilt_hash != specification["expected_sha256"]:
        raise ValueError(
            f"Current reconstruction logic does not reproduce {relative_path}: "
            f"{rebuilt_hash} != {specification['expected_sha256']}"
        )
    return specification


def main() -> None:
    args = arguments()
    datasets: dict[str, Any] = {}
    for dataset_id, config in DATASET_CONFIG.items():
        dataset_record = dict(config)
        path = PROJECT_ROOT / config["source_path"]
        if not path.exists():
            raise FileNotFoundError(path)
        dataset_record["source_sha256"] = sha256_file(path)
        datasets[dataset_id] = dataset_record

    manifest = {
        "schema_version": "1.0",
        "purpose": (
            "Row selectors and serialization rules for byte-identical local fixture "
            "reconstruction without redistributing transaction values."
        ),
        "contains_transaction_values": False,
        "selector_fields": list(SELECTOR_COLUMNS),
        "datasets": datasets,
        "fixtures": [build_specification(path) for path in FIXTURE_PATHS],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fixtures": len(manifest["fixtures"]),
        "rows": sum(item["expected_rows"] for item in manifest["fixtures"]),
        "sha256": sha256_file(output),
    }, indent=2))


if __name__ == "__main__":
    main()
