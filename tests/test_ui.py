"""Startup check for the Streamlit three-tab application."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StreamlitTests(unittest.TestCase):
    def test_app_starts_without_model_artifacts(self) -> None:
        app = AppTest.from_file(
            str(PROJECT_ROOT / "ui" / "streamlit_app.py"),
            default_timeout=20,
        ).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Fraud Investigation A/B/C Lab")
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["A · Baseline", "B · Retrieval augmented", "C · Guardrail-aware agent"],
        )
        self.assertFalse(app.chat_input[0].disabled)
        app.chat_input[0].set_value(
            "Compare historical fraud rates in ULB and Sparkov"
        ).run()
        self.assertEqual(len(app.exception), 0)
        rendered_messages = [
            markdown.value
            for message in app.chat_message
            for markdown in message.markdown
        ]
        self.assertTrue(any("0.1663%" in value for value in rendered_messages))
        self.assertTrue(any("0.5753%" in value for value in rendered_messages))


if __name__ == "__main__":
    unittest.main()
