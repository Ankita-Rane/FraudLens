"""Build training-only ULB and Sparkov historical fraud summaries."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.historical import build_historical_profiles  # noqa: E402


def main() -> None:
    profiles = build_historical_profiles(PROJECT_ROOT)
    print("Historical profiles written to artifacts/historical/historical_profiles.json")
    for dataset in profiles["datasets"]:
        prevalence = next(
            record for record in dataset["records"]
            if record["topic"] == "historical_prevalence"
        )
        print(
            f"- {dataset['dataset_id']}: {prevalence['facts']['transactions']} training "
            f"transactions, {prevalence['facts']['fraud']} fraud"
        )


if __name__ == "__main__":
    main()
