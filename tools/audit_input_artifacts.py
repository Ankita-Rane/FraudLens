"""CLI for the frozen thesis input-artifact integrity audit."""

from __future__ import annotations

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_audit import audit_input_artifacts  # noqa: E402


def main() -> None:
    result = audit_input_artifacts(PROJECT_ROOT)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
