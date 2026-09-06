"""Run a contiguous range of locked follow-up batches with per-batch Excel logging."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= 55:
        parser.error("Batch range must satisfy 1 <= start <= end <= 55.")
    return args


def runner_command(batch_number: int) -> str:
    batch_id = f"FOLLOWUP-BATCH-{batch_number:03d}"
    parts = [
        str(PYTHON), "tools/run_followup_retrieval_experiment.py",
        "--mode", "final", "--engine", "qwen",
        "--questions", "datasets/evaluation/abc_questions_primary_candidate.json",
        "--repeats", "1",
        "--model-name", "Qwen/Qwen3-0.6B",
        "--model-revision", "c1899de289a04d12100db370d81485cdf75e47ca",
        "--local-files-only", "--max-new-tokens", "360",
        "--model-score-gate", "0.5", "--policy-top-k", "4",
        "--dense-backend", "sentence-transformer",
        "--dense-model-name", "sentence-transformers/all-MiniLM-L6-v2",
        "--dense-model-revision", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "--dense-device", "mps", "--rrf-k", "60",
        "--rrf-candidate-multiplier", "4",
        "--study-id", "FOLLOWUP-RETRIEVAL-20260902-02",
        "--protocol-lock", "datasets/evaluation/followup_retrieval/protocol_locked.json",
        "--batch-id", batch_id, "--confirm-generation-authorized",
        "--ulb-attribution-directory", "artifacts/followup_detectors/ulb",
        "--sparkov-attribution-directory", "artifacts/followup_detectors/sparkov",
    ]
    return " ".join(parts)


def main() -> None:
    args = arguments()
    for number in range(args.start, args.end + 1):
        batch_id = f"FOLLOWUP-BATCH-{number:03d}"
        command = [
            str(PYTHON), "tools/experiment_execution_ledger.py", "run",
            "--stage", "S11",
            "--purpose", f"Execute locked final batch {number:03d} of 055 on pinned Qwen and SBERT using MPS",
            "--command", runner_command(number),
            "--notes", f"Reportable {batch_id}: exactly 10 registered cases x 6 conditions = 60 calls.",
        ]
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            raise SystemExit(
                f"Stopped after failed {batch_id}; inspect its ledger log before resuming."
            )


if __name__ == "__main__":
    main()
