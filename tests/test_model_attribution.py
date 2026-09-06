"""Tests for explicitly bounded local model perturbation evidence."""

from pathlib import Path
import sys
import unittest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_attribution import (  # noqa: E402
    grouped_reference_perturbation,
    tree_shap_attribution,
)


class AttributionTests(unittest.TestCase):
    def test_method_is_not_mislabelled_as_shap(self):
        frame = pd.DataFrame({"amount": [1.0, 2.0, 100.0, 120.0], "hour": [1, 2, 23, 22]})
        model = RandomForestClassifier(n_estimators=20, random_state=42).fit(
            frame, [0, 0, 1, 1]
        )
        result = grouped_reference_perturbation(
            pipeline=model,
            sample={"amount": 100.0, "hour": 23},
            reference_profile={"amount": 1.5, "hour": 1},
            feature_groups={"amount": ("amount",), "time": ("hour",)},
        )
        self.assertFalse(result["method_is_shap"])
        self.assertEqual(result["method"], "grouped_training_reference_perturbation")
        self.assertEqual(len(result["top_features"]), 2)

    def test_tree_shap_reconstructs_binary_fraud_probability(self):
        frame = pd.DataFrame({"amount": [1.0, 2.0, 100.0, 120.0], "hour": [1, 2, 23, 22]})
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=20, random_state=42)),
        ]).fit(frame, [0, 0, 1, 1])
        result = tree_shap_attribution(
            pipeline=pipeline,
            sample={"amount": 100.0, "hour": 23},
        )
        self.assertTrue(result["method_is_shap"])
        self.assertLessEqual(result["additivity_absolute_error"], 1e-6)
        self.assertEqual(result["all_feature_count"], 2)
