"""Transparent non-LLM answer renderer for chatbot demonstration mode."""

from __future__ import annotations

from typing import Any


def _percentage(value: float) -> str:
    return f"{100 * value:.4f}%"


def _class_row(rows: list[dict[str, Any]], target_value: int) -> dict[str, Any] | None:
    for row in rows:
        class_value = row.get("Class", row.get("is_fraud"))
        if class_value == target_value:
            return row
    return None


def build_evidence_answer(
    question: str,
    scoped_evidence: dict[str, Any],
) -> tuple[str, list[str]]:
    """Summarize retrieved facts without claiming that an LLM generated them."""
    paragraphs: list[str] = []
    cited_ids: list[str] = []
    profiles = scoped_evidence.get("historical_profiles")

    if profiles:
        for dataset in profiles.get("datasets", []):
            dataset_lines: list[str] = []
            for record in dataset.get("records", []):
                evidence_id = record["evidence_id"]
                cited_ids.append(evidence_id)
                topic = record.get("topic")
                facts = record.get("facts")

                if topic == "historical_prevalence":
                    dataset_lines.append(
                        f"The training evidence contains {facts['transactions']:,} "
                        f"transactions and {facts['fraud']:,} labelled fraud cases "
                        f"({_percentage(facts['fraud_rate'])})."
                    )
                elif topic == "amount_by_class":
                    legitimate = _class_row(facts, 0)
                    fraud = _class_row(facts, 1)
                    if legitimate and fraud:
                        dataset_lines.append(
                            "Fraud transaction amounts had mean/median values of "
                            f"{fraud['mean']:,.2f}/{fraud['median']:,.2f}, compared with "
                            f"{legitimate['mean']:,.2f}/{legitimate['median']:,.2f} for "
                            "legitimate transactions."
                        )
                elif topic == "highest_observed_fraud_rate_categories":
                    top = facts[:3]
                    rendered = ", ".join(
                        f"{row['category']} ({_percentage(row['fraud_rate'])})"
                        for row in top
                    )
                    dataset_lines.append(f"The highest observed categories were {rendered}.")
                elif topic == "highest_observed_fraud_rate_hours":
                    top = facts[:3]
                    rendered = ", ".join(
                        f"hour {row['hour']} ({_percentage(row['fraud_rate'])})"
                        for row in top
                    )
                    dataset_lines.append(f"The highest observed hourly rates were {rendered}.")
                elif topic == "fraud_rate_by_weekday_monday_zero":
                    top = sorted(facts, key=lambda row: row["fraud_rate"], reverse=True)[:3]
                    rendered = ", ".join(
                        f"day {row['day_of_week']} ({_percentage(row['fraud_rate'])})"
                        for row in top
                    )
                    dataset_lines.append(f"The highest weekday rates were {rendered} (Monday=0).")
                elif topic == "fraud_rate_by_age_band":
                    top = sorted(facts, key=lambda row: row["fraud_rate"], reverse=True)[:3]
                    rendered = ", ".join(
                        f"{row['age_band']} ({_percentage(row['fraud_rate'])})"
                        for row in top
                    )
                    dataset_lines.append(f"The highest age-band rates were {rendered}.")
                elif topic == "monthly_fraud_trend":
                    if facts:
                        highest = max(facts, key=lambda row: row["fraud_rate"])
                        dataset_lines.append(
                            f"The highest monthly observed rate was in {highest['month']} "
                            f"({_percentage(highest['fraud_rate'])})."
                        )
                elif topic == "largest_training_median_differences":
                    features = ", ".join(row["anonymized_feature"] for row in facts[:5])
                    dataset_lines.append(
                        f"The largest median differences were observed for {features}; "
                        "their business meanings are undisclosed."
                    )
                elif topic == "highest_observed_fraud_rate_relative_hours":
                    top = facts[:3]
                    rendered = ", ".join(
                        f"relative hour {row['relative_hour']} ({_percentage(row['fraud_rate'])})"
                        for row in top
                    )
                    dataset_lines.append(f"The highest relative-hour rates were {rendered}.")

            if dataset_lines:
                paragraphs.append(
                    f"**{dataset['dataset_id']}** — " + " ".join(dataset_lines)
                )
                paragraphs.append(f"Limitation: {dataset['interpretation_limit']}")

    alert_lines: list[str] = []
    for dataset in scoped_evidence.get("datasets", []):
        for alert in dataset.get("top_alerts", []):
            cited_ids.append(alert["evidence_id"])
            selected_features = {
                key: value
                for key, value in alert.get("features", {}).items()
                if key in {"Amount", "amt", "category", "hour", "age", "distance_km"}
            }
            alert_lines.append(
                f"{dataset['dataset_id']} record `{alert['record_id']}` has fraud "
                f"probability {alert['fraud_probability']:.4f}; selected context: "
                f"{selected_features}."
            )
    if alert_lines:
        paragraphs.append("**Highest-score model alerts retrieved:** " + " ".join(alert_lines))

    metric_lines: list[str] = []
    for dataset in scoped_evidence.get("datasets", []):
        tuned_rows = [
            row for row in dataset.get("test_metrics", [])
            if row.get("threshold_rule") == "validation_tuned"
        ]
        if not tuned_rows:
            continue
        row = tuned_rows[0]
        if row.get("evidence_id"):
            cited_ids.append(row["evidence_id"])
        available_metrics = []
        for key, label in (
            ("average_precision", "average precision"),
            ("roc_auc", "ROC-AUC"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f2", "F2"),
            ("mcc", "MCC"),
        ):
            if row.get(key) is not None:
                available_metrics.append(f"{label}={row[key]:.4f}")
        if available_metrics:
            metric_lines.append(
                f"{dataset['dataset_id']} {row.get('split', 'test')}: "
                + ", ".join(available_metrics)
            )
    if metric_lines:
        paragraphs.append("**Locked test performance:** " + "; ".join(metric_lines) + ".")

    if not paragraphs:
        paragraphs.append(
            "I could not find a matching fact in the available ULB/Sparkov evidence. "
            "Try asking about prevalence, amounts, time, categories, model performance, "
            "or high-risk alerts."
        )

    paragraphs.append(
        "This is a deterministic evidence summary for UI demonstration, not a direct "
        "LLM-generated Configuration A result. Select Local Qwen for the experimental "
        "direct-LLM condition."
    )
    return "\n\n".join(paragraphs), list(dict.fromkeys(cited_ids))
