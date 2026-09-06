from __future__ import annotations

import json
from pathlib import Path

from tools.build_followup_panels import _exposed_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "datasets" / "evaluation" / "dataset_fixture_selection_manifest.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_fixture_manifest_contains_only_selectors_not_transaction_values() -> None:
    manifest = _manifest()
    assert manifest["contains_transaction_values"] is False
    assert len(manifest["fixtures"]) == 8
    assert sum(item["expected_rows"] for item in manifest["fixtures"]) == 634
    allowed = {
        "sample_id",
        "test_case_type",
        "evaluation_cohort",
        "source_test_row_index",
    }
    for fixture in manifest["fixtures"]:
        for row in fixture["rows"]:
            assert set(row) <= allowed


def test_followup_exclusions_survive_fixture_csv_removal() -> None:
    manifest = _manifest()
    for dataset_key in ("ulb", "sparkov"):
        expected = {
            int(row["source_test_row_index"])
            for fixture in manifest["fixtures"]
            if Path(fixture["path"]).parent.name == dataset_key
            and Path(fixture["path"]).name
            in {
                "runtime_transaction_samples.csv",
                "pilot_transaction_samples.csv",
                "candidate_transaction_samples.csv",
            }
            for row in fixture["rows"]
        }
        assert _exposed_ids(dataset_key, include_pilot=True) == expected
