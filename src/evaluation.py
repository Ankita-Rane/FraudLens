"""Leakage-safe evaluation helpers shared by the ULB and Sparkov workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def positive_class_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities or decision scores."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise TypeError("Model must implement predict_proba or decision_function.")


def select_fbeta_threshold(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    beta: float = 2.0,
) -> dict[str, float]:
    """Select a decision threshold on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return {"threshold": 0.5, "validation_fbeta": 0.0}

    beta_sq = beta**2
    denominator = beta_sq * precision[:-1] + recall[:-1]
    scores = np.divide(
        (1 + beta_sq) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(scores))
    return {
        "threshold": float(thresholds[best_index]),
        "validation_fbeta": float(scores[best_index]),
    }


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    beta: float = 2.0,
) -> dict[str, float | int]:
    """Compute imbalance-aware metrics at a pre-declared threshold."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": int(y_true_array.size),
        "fraud_n": int(y_true_array.sum()),
        "fraud_rate": float(y_true_array.mean()),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true_array, y_score)),
        "roc_auc": float(roc_auc_score(y_true_array, y_score)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        f"f{beta:g}": float(
            fbeta_score(y_true_array, y_pred, beta=beta, zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "mcc": float(matthews_corrcoef(y_true_array, y_pred)),
        "brier": float(brier_score_loss(y_true_array, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluation_rows(
    *,
    dataset: str,
    model_name: str,
    split: str,
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    tuned_threshold: float,
    beta: float = 2.0,
) -> pd.DataFrame:
    """Return comparable default- and tuned-threshold evaluation rows."""
    rows: list[dict[str, Any]] = []
    for threshold_name, threshold in (
        ("default_0.5", 0.5),
        ("validation_tuned", tuned_threshold),
    ):
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model_name,
            "split": split,
            "threshold_rule": threshold_name,
        }
        row.update(binary_metrics(y_true, y_score, threshold=threshold, beta=beta))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_diagnostics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    title: str,
) -> plt.Figure:
    """Plot PR, ROC, and calibration diagnostics without selecting a threshold."""
    y_true_array = np.asarray(y_true, dtype=int)
    precision, recall, _ = precision_recall_curve(y_true_array, y_score)
    fpr, tpr, _ = roc_curve(y_true_array, y_score)
    prob_true, prob_pred = calibration_curve(
        y_true_array,
        y_score,
        n_bins=10,
        strategy="quantile",
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(recall, precision)
    axes[0].axhline(y_true_array.mean(), linestyle="--", color="grey")
    axes[0].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[1].set(title="ROC", xlabel="False-positive rate", ylabel="True-positive rate")

    axes[2].plot(prob_pred, prob_true, marker="o")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[2].set(
        title="Calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed fraud rate",
    )

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_validation_comparison(
    validation_results: pd.DataFrame,
    *,
    beta: float = 2.0,
    title: str,
) -> plt.Figure:
    """Compare candidate models using validation data only."""
    metric_columns = ["average_precision", "precision", "recall", f"f{beta:g}"]
    tuned = (
        validation_results.loc[
            validation_results["threshold_rule"] == "validation_tuned",
            ["model", *metric_columns],
        ]
        .set_index("model")
        .sort_values("average_precision", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.75 * len(tuned))))
    y_positions = np.arange(len(tuned))
    bar_height = 0.18
    colours = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    display_names = ["PR-AUC", "Precision", "Recall", f"F{beta:g}"]
    for index, (metric, label, colour) in enumerate(
        zip(metric_columns, display_names, colours)
    ):
        offset = (index - 1.5) * bar_height
        ax.barh(
            y_positions + offset,
            tuned[metric],
            height=bar_height,
            label=label,
            color=colour,
        )
    ax.set(
        yticks=y_positions,
        yticklabels=tuned.index,
        xlim=(0, 1),
        xlabel="Validation metric",
        title=title,
    )
    ax.legend(ncols=4, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    selected_threshold: float,
    beta: float = 2.0,
    title: str,
) -> plt.Figure:
    """Visualize precision, recall, and F-beta over validation thresholds."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    beta_sq = beta**2
    denominator = beta_sq * precision[:-1] + recall[:-1]
    fbeta = np.divide(
        (1 + beta_sq) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    if len(thresholds) > 2_000:
        plot_index = np.unique(np.linspace(0, len(thresholds) - 1, 2_000).astype(int))
    else:
        plot_index = np.arange(len(thresholds))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(thresholds[plot_index], precision[:-1][plot_index], label="Precision")
    ax.plot(thresholds[plot_index], recall[:-1][plot_index], label="Recall")
    ax.plot(thresholds[plot_index], fbeta[plot_index], label=f"F{beta:g}", linewidth=2.5)
    ax.axvline(
        selected_threshold,
        color="#E45756",
        linestyle="--",
        label=f"Selected threshold = {selected_threshold:.4f}",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Decision threshold",
        ylabel="Metric value",
        title=title,
    )
    ax.legend(ncols=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_confusion_summary(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    title: str,
) -> plt.Figure:
    """Plot count and row-normalized confusion matrices at a fixed threshold."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])
    normalized = matrix / matrix.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, values, panel_title, value_format in (
        (axes[0], matrix, "Transaction counts", "d"),
        (axes[1], normalized, "Row-normalized rates", ".1%"),
    ):
        image = axis.imshow(values, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    format(values[row, column], value_format),
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            xlabel="Predicted class",
            ylabel="Actual class",
            title=panel_title,
        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a JSON artifact with NumPy scalar support."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")
