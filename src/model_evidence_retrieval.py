"""Question-aware routing over ULB and Sparkov model evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOPIC_HINTS = {
    "historical_prevalence": "fraud prevalence rate percentage count frequency how many",
    "amount_by_class": "amount value spending price legitimate fraudulent median mean",
    "highest_observed_fraud_rate_categories": "merchant category categories shopping grocery",
    "highest_observed_fraud_rate_hours": "hour time overnight night daytime",
    "fraud_rate_by_weekday_monday_zero": "weekday day monday tuesday wednesday thursday friday weekend",
    "fraud_rate_by_age_band": "age demographic older younger cardholder",
    "monthly_fraud_trend": "month monthly trend drift over time historical",
    "highest_observed_fraud_rate_relative_hours": "relative hour time two days temporal",
    "largest_training_median_differences": "feature variables V1 V2 anonymized difference important",
}

ALERT_TERMS = {
    "alert", "alerts", "case", "cases", "transaction", "transactions",
    "highest risk", "high risk", "probability", "flagged", "investigate",
}
METRIC_TERMS = {
    "model performance", "metric", "metrics", "precision", "recall", "f1", "f2",
    "average precision", "pr auc", "roc", "mcc", "accuracy", "calibration",
}


def _intent_topics(question: str) -> set[str]:
    text = question.lower()
    topics: set[str] = set()
    if any(term in text for term in ("amount", "spend", "value", "price")):
        topics.add("amount_by_class")
    if any(term in text for term in ("category", "categories", "merchant type")):
        topics.add("highest_observed_fraud_rate_categories")
    if any(term in text for term in ("hour", "time of day", "overnight", "night")):
        topics.update({
            "highest_observed_fraud_rate_hours",
            "highest_observed_fraud_rate_relative_hours",
        })
    if any(term in text for term in ("weekday", "day of week", "weekend")):
        topics.add("fraud_rate_by_weekday_monday_zero")
    if any(term in text for term in ("age", "younger", "older", "demographic")):
        topics.add("fraud_rate_by_age_band")
    if any(term in text for term in ("month", "monthly", "trend", "drift", "over time")):
        topics.add("monthly_fraud_trend")
    if any(term in text for term in ("feature", "variable", "v1", "v2", "anonymized")):
        topics.add("largest_training_median_differences")
    if not topics and any(term in text for term in (
        "prevalence", "fraud rate", "percentage", "percent", "how many",
        "frequency", "historical fraud",
    )):
        topics.add("historical_prevalence")
    return topics


def _record_text(dataset_id: str, record: dict[str, Any]) -> str:
    topic = record.get("topic", "").replace("_", " ")
    hints = TOPIC_HINTS.get(record.get("topic", ""), "")
    return " ".join([
        dataset_id,
        topic,
        hints,
        json.dumps(record.get("facts"), ensure_ascii=False),
        str(record.get("caution", "")),
    ])


def _select_records(
    question: str,
    dataset_id: str,
    records: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    intent_topics = _intent_topics(question)
    intent_matches = [
        record for record in records
        if record.get("topic") in intent_topics
    ]
    if intent_matches:
        return intent_matches[:top_k]
    documents = [_record_text(dataset_id, record) for record in records]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents)
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix).ravel()
    ranked_indices = np.argsort(scores)[::-1]
    selected = [records[index] for index in ranked_indices[:top_k] if scores[index] > 0]
    if not selected:
        prevalence = [
            record for record in records
            if record.get("topic") == "historical_prevalence"
        ]
        return prevalence[:1]
    return selected


def scope_evidence_for_question(
    unified_evidence: dict[str, Any],
    question: str,
    *,
    historical_records_per_dataset: int = 3,
    alerts_per_dataset: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a source-preserving evidence subset relevant to one question."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    scoped = deepcopy(unified_evidence)
    question_lower = question.lower()
    mentioned_datasets = {
        dataset_id
        for dataset_id in ("ULB", "Sparkov")
        if dataset_id.lower() in question_lower
    }
    include_alerts = any(term in question_lower for term in ALERT_TERMS)
    include_metrics = any(term in question_lower for term in METRIC_TERMS)

    selected_ids: list[str] = []
    profiles = scoped.get("historical_profiles")
    if profiles:
        for dataset in profiles.get("datasets", []):
            if mentioned_datasets and dataset["dataset_id"] not in mentioned_datasets:
                dataset["records"] = []
                continue
            selected = _select_records(
                question,
                dataset["dataset_id"],
                dataset.get("records", []),
                historical_records_per_dataset,
            )
            dataset["records"] = selected
            selected_ids.extend(record["evidence_id"] for record in selected)

    for dataset in scoped["datasets"]:
        dataset_is_requested = (
            not mentioned_datasets or dataset["dataset_id"] in mentioned_datasets
        )
        if include_alerts and dataset_is_requested:
            dataset["top_alerts"] = dataset.get("top_alerts", [])[:alerts_per_dataset]
            selected_ids.extend(
                alert["evidence_id"] for alert in dataset["top_alerts"]
            )
        else:
            dataset["top_alerts"] = []
        if not include_metrics or not dataset_is_requested:
            dataset["test_metrics"] = []
        else:
            selected_ids.extend(
                row["evidence_id"]
                for row in dataset.get("test_metrics", [])
                if row.get("evidence_id")
            )

    if not include_metrics:
        scoped["cross_dataset_test_comparison"] = []

    retrieval = {
        "question": question,
        "historical_records_per_dataset": historical_records_per_dataset,
        "alerts_per_dataset": alerts_per_dataset if include_alerts else 0,
        "included_alerts": include_alerts,
        "included_metrics": include_metrics,
        "evidence_ids": selected_ids,
    }
    scoped["question_evidence_routing"] = retrieval
    return scoped, retrieval
