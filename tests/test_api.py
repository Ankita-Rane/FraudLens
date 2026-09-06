"""HTTP adapter smoke tests without loading local Qwen."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import create_app  # noqa: E402


RUNTIME_FIXTURES = (
    PROJECT_ROOT / "datasets/ulb/runtime_transaction_samples.csv",
    PROJECT_ROOT / "datasets/sparkov/runtime_transaction_samples.csv",
)


@unittest.skipUnless(
    all(path.exists() for path in RUNTIME_FIXTURES),
    "requires locally reconstructed runtime transaction fixtures",
)
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    def test_health_and_samples_do_not_expose_labels(self):
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)
        response = self.client.get("/api/v1/samples")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(len(payload["samples"]), 0)
        self.assertNotIn("evaluation_only_actual_class", payload["samples"][0])

    def test_demo_comparison_is_paired(self):
        response = self.client.post("/api/v1/comparisons", json={
            "question": "What evidence supports review under the supplied policy?",
            "sample_id": "S1_HIGH_RISK",
            "engine": "demo",
            "repeats": 1,
        })
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["paired_design"]["paired"])
        self.assertEqual(
            [row["configuration"] for row in payload["results"]], ["A", "B", "C"]
        )
        experiment_id = payload["experimental_control"]["experiment_id"]
        self.assertTrue(experiment_id)
        self.assertTrue(all(
            row["trace"]["experiment_id"] == experiment_id
            for row in payload["results"]
        ))
        self.assertTrue(all(row["model_evidence"] for row in payload["results"]))

    def test_operational_alert_forces_configuration_c(self):
        response = self.client.post("/api/v1/operational-alerts", json={
            "sample_id": "S1_HIGH_RISK",
            "engine": "demo",
        })
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["operational_mode"])
        self.assertEqual(payload["configuration_forced"], "C")
        self.assertEqual(payload["result"]["configuration"], "C")
        self.assertIsNone(payload["result"]["trace"]["experiment_id"])

    def test_reinvented_ui_read_contracts_preserve_boundaries(self):
        readiness = self.client.get("/api/v1/ui/readiness")
        self.assertEqual(readiness.status_code, 200)
        self.assertFalse(readiness.json()["jira"]["credentials_exposed"])
        self.assertNotIn("email", readiness.text.lower())
        workflows = self.client.get("/api/v1/workflows").json()
        self.assertFalse(workflows["arbitrary_command_execution"])
        self.assertTrue(all("argv" not in item for item in workflows["job_types"]))
        eda = self.client.get("/api/v1/eda-models").json()
        self.assertEqual([item["dataset_id"] for item in eda["datasets"]], ["ULB", "Sparkov"])

    def test_operational_batch_requires_explicit_write_confirmation(self):
        preview = self.client.post("/api/v1/operational-alerts/preview", json={
            "sample_ids": ["S1_HIGH_RISK", "S1_HIGH_RISK"], "engine": "demo"
        })
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.json()["maximum_possible_jira_writes"], 1)
        blocked = self.client.post("/api/v1/operational-alerts/batch", json={
            "sample_ids": ["S1_HIGH_RISK"], "engine": "demo", "confirm_write": False
        })
        self.assertEqual(blocked.status_code, 422)


if __name__ == "__main__":
    unittest.main()
