# %% [markdown]
# # ULB/Worldline Credit-Card Fraud: EDA and Predictive Modelling
#
# **Dataset role:** real, anonymized public benchmark.
#
# This notebook implements a thesis-safe workflow:
#
# - integrity checks are performed before modelling;
# - exact duplicates are handled before splitting;
# - the test set is never balanced, augmented, or used for feature selection;
# - target-informed EDA uses training data only;
# - model selection and threshold selection use validation data only;
# - the selected model is evaluated once on the untouched test set;
# - PR-AUC is the primary selection metric.

# %%
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import platform
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import imblearn
from IPython import get_ipython
from IPython.display import display

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


SEED = 42
TEST_SIZE = float(os.getenv("ULB_TEST_SIZE", "0.20"))
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.20
PRIMARY_METRIC = "average_precision"
FBETA = 2.0

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda value: f"{value:,.5f}")


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        ulb_data = candidate / "datasets" / "ULB_creditCard.csv"
        if ulb_data.exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Run from thesis_code or one of its subdirectories.")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
DATA_PATH = PROJECT_ROOT / "datasets" / "ULB_creditCard.csv"
ARTIFACT_DIR = Path(
    os.getenv("ULB_ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts" / "ulb"))
).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation import (  # noqa: E402
    evaluation_rows,
    plot_confusion_summary,
    plot_diagnostics,
    plot_threshold_tradeoff,
    plot_validation_comparison,
    positive_class_scores,
    save_json,
    select_fbeta_threshold,
)
from src.evidence import build_alert_records, write_jsonl  # noqa: E402

print({
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scikit_learn": sklearn.__version__,
    "imbalanced_learn": imblearn.__version__,
    "project_root": str(PROJECT_ROOT),
})

# %% [markdown]
# ## 1. Provenance, integrity, and dataset card
#
# Whole-dataset access in this section is restricted to schema and integrity checks.
# No modelling choice is made from test-label patterns.

# %%
data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
raw = pd.read_csv(DATA_PATH)

expected_columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
assert raw.columns.tolist() == expected_columns, "Unexpected ULB schema."
assert raw.shape[1] == 31
assert set(raw["Class"].unique()) == {0, 1}
assert raw.isna().sum().sum() == 0
assert np.isfinite(raw.select_dtypes(include="number")).all().all()

dataset_card = pd.Series({
    "source": "ULB/Worldline credit-card fraud benchmark",
    "rows": len(raw),
    "predictors": raw.shape[1] - 1,
    "legitimate": int((raw["Class"] == 0).sum()),
    "fraud": int((raw["Class"] == 1).sum()),
    "fraud_rate_percent": 100 * raw["Class"].mean(),
    "missing_cells": int(raw.isna().sum().sum()),
    "redundant_exact_duplicates": int(raw.duplicated().sum()),
    "sha256": data_hash,
})
dataset_card.to_frame("value")

# %%
raw.sample(5, random_state=SEED)

# %% [markdown]
# ## 2. Class imbalance and integrity-only EDA

# %%
class_summary = (
    raw["Class"]
    .value_counts()
    .sort_index()
    .rename(index={0: "Legitimate", 1: "Fraud"})
    .to_frame("count")
)
class_summary["percent"] = 100 * class_summary["count"] / len(raw)
display(class_summary)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].pie(
    class_summary["count"],
    labels=class_summary.index,
    colors=["#4C78A8", "#E45756"],
    autopct=lambda value: f"{value:.3f}%",
    startangle=90,
    wedgeprops={"width": 0.42, "edgecolor": "white"},
)
axes[0].set_title("ULB class composition")
class_summary["count"].plot.bar(ax=axes[1], color=["#4C78A8", "#E45756"])
axes[1].set(title="ULB class counts", xlabel="Class", ylabel="Transactions")
class_summary["count"].plot.bar(ax=axes[2], logy=True, color=["#4C78A8", "#E45756"])
axes[2].set(title="ULB class counts (log scale)", xlabel="Class", ylabel="Transactions")
for axis in axes[1:]:
    axis.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()

# %%
duplicate_rows = raw[raw.duplicated(keep=False)].copy()
duplicate_audit = pd.Series({
    "rows_in_duplicate_groups": len(duplicate_rows),
    "redundant_rows": int(raw.duplicated().sum()),
    "fraud_rows_in_duplicate_groups": int(duplicate_rows["Class"].sum()),
    "redundant_fraud_rows": int(raw.loc[raw.duplicated(), "Class"].sum()),
})
duplicate_audit.to_frame("value")

# %% [markdown]
# ## 3. Lock the modelling partitions
#
# The primary analysis retains one row per exact duplicate group. This prevents exact
# copies from crossing partitions. A sensitivity run retaining duplicate groups intact
# should be reported separately.

# %%
model_data = raw.drop_duplicates().reset_index(drop=True)
X = model_data.drop(columns="Class")
y = model_data["Class"].astype("int8")

X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED,
)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=VALIDATION_SIZE_WITHIN_DEVELOPMENT,
    stratify=y_development,
    random_state=SEED,
)

split_manifest = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "rows": [len(X_train), len(X_validation), len(X_test)],
    "fraud": [int(y_train.sum()), int(y_validation.sum()), int(y_test.sum())],
    "fraud_rate": [y_train.mean(), y_validation.mean(), y_test.mean()],
})
display(split_manifest)

assert set(X_train.index).isdisjoint(X_validation.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_validation.index).isdisjoint(X_test.index)

# %% [markdown]
# ## 4. Training-only detailed EDA

# %%
train_eda = X_train.copy()
train_eda["Class"] = y_train

amount_summary = train_eda.groupby("Class")["Amount"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
display(amount_summary)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.ecdfplot(data=train_eda, x="Amount", hue="Class", ax=axes[0])
axes[0].set(xscale="symlog", title="Training amount ECDF", xlabel="Amount (symlog)")
sns.histplot(
    data=train_eda.assign(log_amount=np.log1p(train_eda["Amount"])),
    x="log_amount",
    hue="Class",
    bins=80,
    stat="density",
    common_norm=False,
    element="step",
    ax=axes[1],
)
axes[1].set(title="Training log(1 + Amount) distribution")
fig.tight_layout()
plt.show()

# A class-balanced visualization sample keeps the fraud shape visible without
# changing any model input or evaluation prevalence.
amount_plot_frame = pd.concat([
    train_eda.loc[train_eda["Class"] == 1],
    train_eda.loc[train_eda["Class"] == 0].sample(
        n=min(5_000, int((train_eda["Class"] == 0).sum())),
        random_state=SEED,
    ),
]).assign(log_amount=lambda frame: np.log1p(frame["Amount"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.violinplot(
    data=amount_plot_frame,
    x="Class",
    y="log_amount",
    inner="quartile",
    cut=0,
    palette=["#4C78A8", "#E45756"],
    hue="Class",
    legend=False,
    ax=axes[0],
)
axes[0].set(title="Amount shape by class (visualization sample)", ylabel="log(1 + Amount)")
sns.boxenplot(
    data=amount_plot_frame,
    x="Class",
    y="log_amount",
    palette=["#4C78A8", "#E45756"],
    hue="Class",
    legend=False,
    ax=axes[1],
)
axes[1].set(title="Amount tails by class", ylabel="log(1 + Amount)")
fig.tight_layout()
plt.show()

# %%
time_eda = train_eda.assign(
    relative_hour=train_eda["Time"] / 3600,
    relative_day=(train_eda["Time"] // 86400).astype(int),
)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(
    data=time_eda,
    x="relative_hour",
    hue="Class",
    bins=48,
    stat="density",
    common_norm=False,
    element="step",
    ax=axes[0],
)
axes[0].set(title="Training transaction time by class", xlabel="Hours since first transaction")
hour_bins = pd.cut(time_eda["relative_hour"], bins=24)
hourly = time_eda.groupby(hour_bins, observed=True)["Class"].agg(["count", "sum", "mean"])
bin_midpoints = [interval.mid for interval in hourly.index]
axes[1].plot(bin_midpoints, hourly["mean"].values)
axes[1].set_xticks(bin_midpoints[::2])
axes[1].set_xticklabels([f"{m:.0f}h" for m in bin_midpoints[::2]])
axes[1].set(
    title="Observed training fraud rate by time bin",
    xlabel="Relative-time bin (hour, bin midpoint)",
    ylabel="Fraud rate",
)
fig.tight_layout()
plt.show()

time_eda["relative_hour_of_day"] = (time_eda["Time"] // 3600).astype(int) % 24
temporal_grid = (
    time_eda.groupby(["relative_day", "relative_hour_of_day"], observed=True)["Class"]
    .agg(transactions="count", fraud="sum", fraud_rate="mean")
    .reset_index()
)
fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
sns.heatmap(
    temporal_grid.pivot(index="relative_day", columns="relative_hour_of_day", values="transactions"),
    cmap="Blues",
    ax=axes[0],
)
axes[0].set(title="Training transaction density by relative day/hour", ylabel="Relative day")
sns.heatmap(
    temporal_grid.pivot(index="relative_day", columns="relative_hour_of_day", values="fraud_rate"),
    cmap="magma",
    robust=True,
    ax=axes[1],
)
axes[1].set(title="Observed training fraud rate by relative day/hour", xlabel="Hour of day", ylabel="Relative day")
fig.tight_layout()
plt.show()

# This recreates the original notebook's ANOVA ranking without its leakage:
# the score is fitted on the locked training split only and is not used to peek
# at validation or test labels.
anova_scores, _ = f_classif(X_train, y_train)
anova_ranking = (
    pd.Series(anova_scores, index=X_train.columns, name="ANOVA F-score")
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .sort_values(ascending=False)
)
display(anova_ranking.head(20).to_frame())

fig, ax = plt.subplots(figsize=(10, 6))
anova_ranking.head(20).sort_values().plot.barh(color=sns.color_palette("viridis", 20), ax=ax)
ax.set(title="Top ULB ANOVA signals — training split only", xlabel="ANOVA F-score")
fig.tight_layout()
plt.show()

# PCA is used only as a two-dimensional visualization of the anonymized feature
# space; it is not the fitted representation used by the predictive models.
pca_legitimate = X_train.loc[y_train == 0].sample(
    n=min(12_000, int((y_train == 0).sum())),
    random_state=SEED,
)
pca_frame = pd.concat([pca_legitimate, X_train.loc[y_train == 1]])
pca_labels = y_train.loc[pca_frame.index]
pca_scaler = StandardScaler().fit(X_train)
pca_model = PCA(n_components=2, random_state=SEED).fit(pca_scaler.transform(X_train))
pca_coordinates = pca_model.transform(pca_scaler.transform(pca_frame))
pca_plot = pd.DataFrame({
    "PC1": pca_coordinates[:, 0],
    "PC2": pca_coordinates[:, 1],
    "Class": pca_labels.map({0: "Legitimate", 1: "Fraud"}).to_numpy(),
})
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_plot,
    x="PC1",
    y="PC2",
    hue="Class",
    hue_order=["Legitimate", "Fraud"],
    palette={"Legitimate": "#4C78A8", "Fraud": "#E45756"},
    alpha=0.55,
    s=18,
)
plt.title("ULB PCA projection (training visualization sample)")
plt.tight_layout()
plt.show()

# %%
v_columns = [f"V{i}" for i in range(1, 29)]
feature_summary = train_eda.groupby("Class")[v_columns].median().T
feature_summary.columns = ["legitimate_median", "fraud_median"]
feature_summary["absolute_median_difference"] = (
    feature_summary["fraud_median"] - feature_summary["legitimate_median"]
).abs()
feature_summary = feature_summary.sort_values("absolute_median_difference", ascending=False)
display(feature_summary.head(15))

top_features = feature_summary.head(12).index.tolist()
plot_frame = train_eda.sample(min(50_000, len(train_eda)), random_state=SEED)
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
for feature, axis in zip(top_features, axes.flat):
    sns.boxplot(
        data=plot_frame,
        x="Class",
        y=feature,
        showfliers=False,
        ax=axis,
    )
    axis.set_title(f"{feature}: training distribution")
fig.tight_layout()
plt.show()

# %%
training_correlations = train_eda.drop(columns="Class").corr(method="spearman")
plt.figure(figsize=(13, 11))
sns.heatmap(training_correlations, cmap="vlag", center=0, vmin=-1, vmax=1)
plt.title("ULB training predictors: Spearman correlation")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Training-only class-balancing experiment
#
# The original notebook combined random under-sampling and SMOTE. Here that idea is
# retained as a controlled candidate pipeline, but samplers see **training rows only**.
# Validation and test retain their natural fraud prevalence. Scaling precedes SMOTE so
# neighbour distances are not dominated by large-scale `Time` or `Amount` values.

# %%
balance_preview = ImbalancedPipeline([
    ("under", RandomUnderSampler(sampling_strategy=0.10, random_state=SEED)),
    ("smote", SMOTE(sampling_strategy=0.50, random_state=SEED, k_neighbors=5)),
])
_, y_train_hybrid_preview = balance_preview.fit_resample(X_train, y_train)

balance_counts = pd.DataFrame({
    "Original training": y_train.value_counts().sort_index(),
    "Under-sampling + SMOTE": pd.Series(y_train_hybrid_preview).value_counts().sort_index(),
}).rename(index={0: "Legitimate", 1: "Fraud"})
display(balance_counts)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
balance_counts.plot.bar(ax=axes[0], color=["#4C78A8", "#E45756"])
axes[0].set(title="ULB training counts before/after balancing", ylabel="Transactions")
(balance_counts / balance_counts.sum()).plot.bar(ax=axes[1], color=["#4C78A8", "#E45756"])
axes[1].set(title="ULB training class proportions", ylabel="Within-sample proportion")
for axis in axes:
    axis.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Candidate models
#
# Weighted baselines and a leakage-safe hybrid-resampling ablation use all predictors.
# No sampler is ever applied to validation or test data.

# %%
numeric_columns = X_train.columns.tolist()

models = {
    "Dummy prior": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="prior")),
    ]),
    "Logistic regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=SEED,
        )),
    ]),
    "Logistic + hybrid resampling": ImbalancedPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("under", RandomUnderSampler(sampling_strategy=0.10, random_state=SEED)),
        ("smote", SMOTE(sampling_strategy=0.50, random_state=SEED, k_neighbors=5)),
        ("model", LogisticRegression(
            max_iter=2_000,
            random_state=SEED,
        )),
    ]),
    "Random forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        )),
    ]),
    "Histogram gradient boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            class_weight="balanced",
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        )),
    ]),
}

# %% [markdown]
# ## 7. Validation-only model and threshold selection

# %%
trained_models = {}
validation_rows = []
thresholds = {}
validation_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    trained_models[model_name] = model

    validation_score = positive_class_scores(model, X_validation)
    validation_scores[model_name] = validation_score
    threshold_result = select_fbeta_threshold(
        y_validation,
        validation_score,
        beta=FBETA,
    )
    thresholds[model_name] = threshold_result["threshold"]
    validation_rows.append(
        evaluation_rows(
            dataset="ULB",
            model_name=model_name,
            split="validation",
            y_true=y_validation,
            y_score=validation_score,
            tuned_threshold=threshold_result["threshold"],
            beta=FBETA,
        )
    )

validation_results = pd.concat(validation_rows, ignore_index=True)
display(validation_results.sort_values([PRIMARY_METRIC, "model"], ascending=[False, True]))

selection_table = validation_results[
    validation_results["threshold_rule"] == "validation_tuned"
].sort_values(PRIMARY_METRIC, ascending=False)
best_model_name = selection_table.iloc[0]["model"]
best_model = trained_models[best_model_name]
best_threshold = thresholds[best_model_name]

print("Selected model:", best_model_name)
print("Validation-selected F2 threshold:", best_threshold)

plot_validation_comparison(
    validation_results,
    beta=FBETA,
    title="ULB validation model comparison (natural prevalence)",
)
plt.show()

plot_threshold_tradeoff(
    y_validation,
    validation_scores[best_model_name],
    selected_threshold=best_threshold,
    beta=FBETA,
    title=f"ULB validation threshold trade-off — {best_model_name}",
)
plt.show()

selected_model_step = best_model.named_steps["model"]
if hasattr(selected_model_step, "feature_importances_"):
    importance_values = selected_model_step.feature_importances_
    importance_label = "Tree importance"
elif hasattr(selected_model_step, "coef_"):
    importance_values = np.abs(selected_model_step.coef_[0])
    importance_label = "Absolute standardized coefficient"
else:
    importance_values = None

if importance_values is not None:
    selected_importance = pd.Series(importance_values, index=X_train.columns).nlargest(18)
    fig, ax = plt.subplots(figsize=(10, 6))
    selected_importance.sort_values().plot.barh(
        color=sns.color_palette("flare", len(selected_importance)),
        ax=ax,
    )
    ax.set(
        title=f"Selected ULB model signals — {best_model_name}",
        xlabel=importance_label,
    )
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. One-time untouched test evaluation

# %%
test_score = positive_class_scores(best_model, X_test)
test_results = evaluation_rows(
    dataset="ULB",
    model_name=best_model_name,
    split="test",
    y_true=y_test,
    y_score=test_score,
    tuned_threshold=best_threshold,
    beta=FBETA,
)
display(test_results)

plot_diagnostics(
    y_test,
    test_score,
    title=f"ULB untouched test diagnostics — {best_model_name}",
)
plt.show()

plot_confusion_summary(
    y_test,
    test_score,
    threshold=best_threshold,
    title=f"ULB untouched test confusion summary — threshold {best_threshold:.4f}",
)
plt.show()

# %% [markdown]
# ## 9. Save reproducible artifacts

# %%
joblib.dump(best_model, ARTIFACT_DIR / "selected_pipeline.joblib")
validation_results.to_csv(ARTIFACT_DIR / "validation_metrics.csv", index=False)
test_results.to_csv(ARTIFACT_DIR / "test_metrics.csv", index=False)
pd.DataFrame({
    "row_index": X_test.index,
    "actual": y_test.to_numpy(),
    "fraud_probability": test_score,
}).to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)

llm_alert_records = build_alert_records(
    dataset_id="ULB",
    features=X_test,
    y_true=y_test,
    scores=test_score,
    threshold=best_threshold,
    model_name=best_model_name,
    record_ids=[f"ulb-test-{index}" for index in X_test.index],
)
write_jsonl(llm_alert_records, ARTIFACT_DIR / "llm_alert_evidence.jsonl")

save_json({
    "dataset": "ULB",
    "data_sha256": data_hash,
    "seed": SEED,
    "duplicate_policy": "retain first row from each exact duplicate group before split",
    "selected_model": best_model_name,
    "primary_metric": PRIMARY_METRIC,
    "threshold_selection": f"maximum validation F{FBETA:g}",
    "selected_threshold": best_threshold,
    "split_manifest": split_manifest.to_dict(orient="records"),
    "versions": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imblearn.__version__,
    },
}, ARTIFACT_DIR / "run_manifest.json")

print("Artifacts saved to", ARTIFACT_DIR)
print("LLM-ready alert records:", len(llm_alert_records))

# %% [markdown]
# ## 10. Interpretation boundaries
#
# - ULB contains only two days of transactions and 492 labelled frauds before duplicate handling.
# - `V1`-`V28` are anonymized components, so domain interpretation is limited.
# - Random stratification estimates interpolation within this snapshot, not future deployment drift.
# - A time-ordered sensitivity analysis and bootstrap confidence intervals should accompany final thesis tables.
# - The test set must not be revisited after results are incorporated into the thesis.
