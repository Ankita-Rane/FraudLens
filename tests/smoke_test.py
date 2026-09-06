"""Fast API and preprocessing checks for both thesis workflows.

These fits use deliberately small, class-enriched subsets. Their scores are not
scientific results; the purpose is to catch broken paths, schemas, and estimators.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/thesis_mpl")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    binary_metrics,
    positive_class_scores,
    select_fbeta_threshold,
)

SEED = 42


def check_scores(model: Pipeline, X_train, y_train, X_test, y_test) -> None:
    model.fit(X_train, y_train)
    scores = positive_class_scores(model, X_test)
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()
    threshold = select_fbeta_threshold(y_test, scores)["threshold"]
    metrics = binary_metrics(y_test, scores, threshold=threshold)
    assert 0 <= metrics["average_precision"] <= 1


def smoke_ulb() -> None:
    data = pd.read_csv(PROJECT_ROOT / "datasets" / "ULB_creditCard.csv")
    subset = pd.concat(
        [
            data[data["Class"] == 0].sample(3_000, random_state=SEED),
            data[data["Class"] == 1].sample(180, random_state=SEED),
        ],
        ignore_index=True,
    )
    X = subset.drop(columns="Class")
    y = subset["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    models = [
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=200)),
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=10,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                class_weight="balanced", max_iter=10, random_state=SEED
            )),
        ]),
    ]
    for model in models:
        check_scores(model, X_train, y_train, X_test, y_test)
    print("ULB smoke test: passed")


def engineer_sparkov(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    timestamp = frame["trans_date_trans_time"]
    lat1 = np.radians(frame["lat"])
    lon1 = np.radians(frame["long"])
    lat2 = np.radians(frame["merch_lat"])
    lon2 = np.radians(frame["merch_long"])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )

    X = pd.DataFrame(index=frame.index)
    X["amt"] = frame["amt"]
    X["log_amt"] = np.log1p(frame["amt"])
    X["city_pop"] = frame["city_pop"]
    X["lat"] = frame["lat"]
    X["long"] = frame["long"]
    X["merch_lat"] = frame["merch_lat"]
    X["merch_long"] = frame["merch_long"]
    X["age"] = ((timestamp - frame["dob"]).dt.days / 365.2425).clip(0, 110)
    X["distance_km"] = 2 * 6_371.0088 * np.arcsin(np.sqrt(a))
    X["hour"] = timestamp.dt.hour
    X["day_of_week"] = timestamp.dt.dayofweek
    X["month"] = timestamp.dt.month
    X["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype("int8")
    X["category"] = frame["category"].astype("category")
    X["gender"] = frame["gender"].astype("category")
    X["state"] = frame["state"].astype("category")
    return X, frame["is_fraud"].astype("int8")


def smoke_sparkov() -> None:
    columns = [
        "trans_date_trans_time", "category", "amt", "gender", "state",
        "city_pop", "lat", "long", "dob", "merch_lat", "merch_long",
        "is_fraud",
    ]
    data = pd.read_csv(
        PROJECT_ROOT / "datasets" / "sparkov" / "fraudTrain.csv",
        usecols=columns,
        parse_dates=["trans_date_trans_time", "dob"],
    )
    subset = pd.concat(
        [
            data[data["is_fraud"] == 0].sample(4_000, random_state=SEED),
            data[data["is_fraud"] == 1].sample(500, random_state=SEED),
        ],
        ignore_index=True,
    )
    X, y = engineer_sparkov(subset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    numeric = [
        "amt", "log_amt", "city_pop", "lat", "long", "merch_lat",
        "merch_long", "age", "distance_km", "hour", "day_of_week",
        "month", "is_weekend",
    ]
    categorical = ["category", "gender", "state"]
    one_hot = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=10,
            sparse_output=True,
        ), categorical),
    ])
    ordinal = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        ), categorical),
    ])
    categorical_mask = [False] * len(numeric) + [True] * len(categorical)

    models = [
        Pipeline([
            ("preprocess", ordinal),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        Pipeline([
            ("preprocess", one_hot),
            ("model", SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                max_iter=50,
                random_state=SEED,
            )),
        ]),
        Pipeline([
            ("preprocess", ordinal),
            ("model", RandomForestClassifier(
                n_estimators=10,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
        Pipeline([
            ("preprocess", ordinal),
            ("model", HistGradientBoostingClassifier(
                categorical_features=categorical_mask,
                class_weight="balanced",
                max_iter=10,
                min_samples_leaf=20,
                random_state=SEED,
            )),
        ]),
    ]
    for model in models:
        check_scores(model, X_train, y_train, X_test, y_test)
    print("Sparkov smoke test: passed")


if __name__ == "__main__":
    smoke_ulb()
    smoke_sparkov()
    print("All smoke tests passed; scores were intentionally not reported.")
