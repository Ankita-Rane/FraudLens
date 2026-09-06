"""Build checked TreeSHAP evidence for frozen ULB and Sparkov samples."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import argparse
import sys

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_attribution import tree_shap_attribution  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        choices=("runtime", "pilot", "candidate", "followup_pilot", "followup"),
        default="runtime",
    )
    parser.add_argument("--ulb-artifact-directory", type=Path, default=None)
    parser.add_argument("--sparkov-artifact-directory", type=Path, default=None)
    return parser.parse_args()


def _build(
    dataset_key: str,
    dataset_label: str,
    *,
    panel: str,
    artifact_directory: Path | None = None,
) -> Path:
    artifact_directory = artifact_directory or PROJECT_ROOT / "artifacts" / dataset_key
    pipeline_path = artifact_directory / "selected_pipeline.joblib"
    sample_filename = f"{panel}_transaction_samples.csv"
    output_filename = f"{panel}_sample_attributions.json"
    sample_path = PROJECT_ROOT / "datasets" / dataset_key / sample_filename
    output_path = artifact_directory / output_filename

    pipeline = joblib.load(pipeline_path)
    sample_frame = pd.read_csv(sample_path)
    samples = []
    for sample in sample_frame.to_dict(orient="records"):
        attribution = tree_shap_attribution(
            pipeline=pipeline,
            sample=sample,
        )
        samples.append({
            "sample_id": sample["sample_id"],
            "dataset_id": dataset_label,
            "evidence_id": f"model-attribution:{dataset_key}:{sample['sample_id']}",
            **attribution,
        })

    payload = {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_label,
        "panel": panel,
        "model_pipeline_sha256": _sha256_file(pipeline_path),
        "sample_file_sha256": _sha256_file(sample_path),
        "method": "tree_shap_tree_path_dependent",
        "method_scope": "local positive-class attribution with an additive model-score check",
        "samples": samples,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def main() -> None:
    args = _arguments()
    directories = {
        "ulb": args.ulb_artifact_directory,
        "sparkov": args.sparkov_artifact_directory,
    }
    for dataset_key, dataset_label in (("ulb", "ULB"), ("sparkov", "Sparkov")):
        print(_build(
            dataset_key,
            dataset_label,
            panel=args.panel,
            artifact_directory=directories[dataset_key],
        ))


if __name__ == "__main__":
    main()
