"""Build a compact ULB + Sparkov evidence bundle for the LLM layer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import build_unified_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-alerts-per-dataset",
        type=int,
        default=25,
        help="Highest-probability alerts from each dataset placed in compact LLM context.",
    )
    args = parser.parse_args()
    try:
        bundle = build_unified_evidence(
            PROJECT_ROOT,
            max_alerts_per_dataset=args.max_alerts_per_dataset,
        )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    print("Unified evidence written to artifacts/unified/")
    for dataset in bundle["datasets"]:
        print(
            f"- {dataset['dataset_id']}: {dataset['alert_count']} total alerts, "
            f"{len(dataset['top_alerts'])} included in compact context"
        )


if __name__ == "__main__":
    main()
