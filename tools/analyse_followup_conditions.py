"""Generate dataset-separated six-condition and guardrail conditional analyses."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_GUARDRAIL_CONDITIONS = ("C_lexical", "C_dense", "D_hybrid")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-directory", type=Path, action="append", default=[],
        help="Completed batch directory; repeat this option for a multi-batch study.",
    )
    parser.add_argument("--study-id", default=None)
    parser.add_argument(
        "--experiments-root", type=Path,
        default=PROJECT_ROOT / "artifacts" / "experiments" / "followup",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ulb-attributions", type=Path, required=True)
    parser.add_argument("--sparkov-attributions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _study_directories(root: Path, study_id: str) -> list[Path]:
    directories: list[Path] = []
    for path in sorted(root.glob("*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("study_id") == study_id
            and payload.get("scientific_status")
            == "locked_followup_qwen_raw_outputs_pending_analysis"
        ):
            directories.append(path.parent)
    if not directories:
        raise FileNotFoundError(f"No completed experiment batches for study {study_id}.")
    return directories


def _load_runs(directories: list[Path]) -> list[dict[str, Any]]:
    paths = sorted(
        path
        for directory in directories
        for path in (directory / "raw_runs").glob("*.json")
    )
    if not paths:
        raise FileNotFoundError("No raw runs found under the supplied directories.")
    if len(paths) != len({path.resolve() for path in paths}):
        raise ValueError("The same raw-run path was supplied more than once.")
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _load_diagnostics(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("samples", []):
            result[str(row["sample_id"])] = dict(row.get("diagnostics") or {})
    return result


def _fisher(rows: pd.DataFrame, field: str) -> dict[str, Any] | None:
    if rows["actual"].nunique() < 2:
        return None
    table = pd.crosstab(rows["actual"], rows[field].astype(bool)).reindex(
        index=[0, 1], columns=[False, True], fill_value=0
    )
    odds, p_value = stats.fisher_exact(table.to_numpy())
    return {
        "table_nonfraud_fraud_by_false_true": table.to_numpy().tolist(),
        "odds_ratio": None if not np.isfinite(odds) else round(float(odds), 8),
        "p_value": round(float(p_value), 8),
    }


def _spearman(rows: pd.DataFrame, x: str, y: str) -> dict[str, Any] | None:
    valid = rows[[x, y]].dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return None
    result = stats.spearmanr(valid[x], valid[y])
    return {
        "n": len(valid),
        "rho": round(float(result.statistic), 8),
        "p_value": round(float(result.pvalue), 8),
    }


def build_analysis(
    runs: list[dict[str, Any]],
    labels: pd.DataFrame,
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    label_by_id = labels.set_index("sample_id").to_dict(orient="index")
    for run in runs:
        condition = str(run.get("condition_id") or run.get("configuration"))
        if condition not in FULL_GUARDRAIL_CONDITIONS:
            continue
        sample_id = str(run["sample_id"])
        label = label_by_id.get(sample_id)
        if label is None:
            raise ValueError(f"Missing evaluator label for {sample_id}.")
        dataset = str(label["dataset_id"])
        alerts = [
            alert
            for dataset_row in (run.get("model_evidence") or {}).get("datasets", [])
            for alert in dataset_row.get("top_alerts", [])
        ]
        score = float(alerts[0]["fraud_probability"]) if alerts else None
        threshold = float(alerts[0]["decision_threshold"]) if alerts else None
        checks = run.get("guardrail_checks") or {}
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "dataset_id": dataset,
            "condition_id": condition,
            "actual": int(label["evaluation_only_actual_class"]),
            "confusion_stratum": str(label["confusion_stratum"]),
            "status": run.get("status"),
            "released": int(run.get("status") == "completed"),
            "referred": int(bool(run.get("requires_human_review"))),
            "guardrail_failure_count": sum(value is False for value in checks.values()),
            "detector_score": score,
            "distance_from_threshold": score - threshold if score is not None else None,
            "candidate_json_valid": int(
                isinstance(run.get("candidate_response_text"), str)
                and _json_object(run["candidate_response_text"])
            ),
        }
        for check, passed in checks.items():
            record[f"failed__{check}"] = int(passed is False)
        record.update(diagnostics.get(sample_id, {}))
        records.append(record)
    frame = pd.DataFrame(records)
    results: list[dict[str, Any]] = []
    for (dataset, condition), group in frame.groupby(["dataset_id", "condition_id"]):
        fraud_count = int(group["actual"].sum())
        nonfraud_count = int((group["actual"] == 0).sum())
        smaller = min(fraud_count, nonfraud_count)
        label_inference = (
            "inferential" if smaller >= 20
            else "descriptive_only" if smaller >= 5
            else "case_listing_only"
        )
        alert_group = group[group["confusion_stratum"].isin(["TP", "FP"])]
        tp_count = int((alert_group["confusion_stratum"] == "TP").sum())
        fp_count = int((alert_group["confusion_stratum"] == "FP").sum())
        alert_smaller = min(tp_count, fp_count)
        tp_fp_inference = (
            "inferential" if alert_smaller >= 20
            else "descriptive_only" if alert_smaller >= 5
            else "case_listing_only"
        )
        if dataset == "ULB":
            released = int(group["released"].sum())
            blocked = len(group) - released
            disposition_inference = (
                "inferential" if min(released, blocked) >= 10 else "descriptive_only"
            )
        else:
            disposition_inference = (
                "inferential" if smaller >= 20 else label_inference
            )
        check_fields = sorted(column for column in group if column.startswith("failed__"))
        continuous = [
            "detector_score", "distance_from_threshold", "nonzero_shap_count",
            "top_1_absolute_mass_share", "top_3_absolute_mass_share",
            "top_10_absolute_mass_share", "effective_feature_count",
            "categorical_absolute_shap_mass_share",
        ]
        result = {
            "dataset_id": dataset,
            "condition_id": condition,
            "case_count": len(group),
            "fraud_count": fraud_count,
            "non_fraud_count": nonfraud_count,
            "confusion_counts": {
                str(key): int(value)
                for key, value in group["confusion_stratum"].value_counts().items()
            },
            "status_counts": {
                str(key): int(value) for key, value in group["status"].value_counts().items()
            },
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fraud_nonfraud_analysis_status": label_inference,
            "tp_fp_analysis_status": tp_fp_inference,
            "pass_block_analysis_status": disposition_inference,
            "fraud_nonfraud_fisher": {
                field: test for field in ["released", "referred", "candidate_json_valid", *check_fields]
                if (test := _fisher(group, field)) is not None
            } if label_inference == "inferential" else {},
            "tp_fp_fisher": {
                field: test
                for field in ["released", "referred", "candidate_json_valid", *check_fields]
                if (test := _fisher(alert_group, field)) is not None
            } if tp_fp_inference == "inferential" else {},
            "tp_fp_case_listing": (
                alert_group[[
                    "sample_id", "confusion_stratum", "status", "detector_score",
                    "guardrail_failure_count", "candidate_json_valid",
                ]].to_dict(orient="records")
                if tp_fp_inference != "inferential" else []
            ),
            "guardrail_failure_correlations": {
                field: test for field in continuous
                if field in group and (test := _spearman(
                    group, field, "guardrail_failure_count"
                )) is not None
            },
            "limitations": [
                "Associations are exploratory and do not establish causation.",
                "ULB anonymized V-features do not support categorical business interpretation."
                if dataset == "ULB" else
                "Sparkov is synthetic and categorical associations may not generalize to banks.",
            ],
        }
        results.append(result)
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "conditional_exploratory_analysis",
        "unit": "evaluation transaction within dataset and condition",
        "dataset_pooling": "prohibited",
        "ulb_rules": {
            "tp_fp_inference_minimum_per_stratum": 20,
            "descriptive_minimum_per_stratum": 5,
            "pass_block_minimum_per_group": 10,
        },
        "results": results,
    }


def _json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def main() -> None:
    args = _arguments()
    if bool(args.experiment_directory) == bool(args.study_id):
        raise ValueError(
            "Supply either one-or-more --experiment-directory values or --study-id."
        )
    directories = (
        args.experiment_directory
        if args.experiment_directory
        else _study_directories(args.experiments_root, args.study_id)
    )
    payload = build_analysis(
        _load_runs(directories),
        pd.read_csv(args.labels),
        _load_diagnostics([args.ulb_attributions, args.sparkov_attributions]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "result_groups": len(payload["results"]),
        "scientific_status": payload["scientific_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
