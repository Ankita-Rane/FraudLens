"""Consolidate executed notebook outputs into thesis-ready result artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json
import logging
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "results"
LOG_PATH = PROJECT_ROOT / "log" / "application.log"


def configure_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_results_report")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == LOG_PATH
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    return logger


LOGGER = configure_logging()


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional dependencies."""
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number").columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.6f}")
    headers = [str(column) for column in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.itertuples(index=False)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def notebook_status(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    return {
        "notebook": str(path.relative_to(PROJECT_ROOT)),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        "error_outputs": len(error_outputs),
        "status": "passed" if not error_outputs and all(
            cell.get("execution_count") is not None for cell in code_cells
        ) else "incomplete_or_failed",
    }


def main() -> None:
    LOGGER.info("results_report_build_started")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([
        pd.read_csv(PROJECT_ROOT / "artifacts" / "ulb" / "test_metrics.csv"),
        pd.read_csv(PROJECT_ROOT / "artifacts" / "sparkov" / "test_metrics.csv"),
    ], ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "model_test_comparison.csv", index=False)
    (OUTPUT_DIR / "model_test_comparison.json").write_text(
        metrics.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    LOGGER.info("model_test_metrics_consolidated rows=%d", len(metrics))

    tuned = metrics[metrics["threshold_rule"] == "validation_tuned"].copy()
    columns = [
        "dataset", "model", "split", "threshold", "average_precision", "roc_auc",
        "precision", "recall", "f2", "mcc", "fp", "fn", "tp",
    ]
    tuned = tuned[columns]

    executed_notebooks = [
        PROJECT_ROOT / "notebooks" / "ULB_EDA_MODELING.ipynb",
        PROJECT_ROOT / "notebooks" / "SPARKOV_EDA_MODELING.ipynb",
        PROJECT_ROOT / "notebooks" / "CONFIGURATION_A_UNIFIED_BASELINE.ipynb",
        PROJECT_ROOT / "notebooks" / "CONFIGURATION_A_B_C_EXPERIMENT.ipynb",
        PROJECT_ROOT / "notebooks" / "ABC_UI_BACKEND_WALKTHROUGH.ipynb",
    ]
    statuses = [notebook_status(path) for path in executed_notebooks]
    (OUTPUT_DIR / "notebook_execution_status.json").write_text(
        json.dumps(statuses, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("notebook_status_catalogued notebooks=%d", len(statuses))

    config_a_path = (
        PROJECT_ROOT / "artifacts" / "configuration_a" / "configuration_a_result.json"
    )
    config_a = (
        json.loads(config_a_path.read_text(encoding="utf-8"))
        if config_a_path.is_file()
        else None
    )
    if config_a is None:
        LOGGER.info("configuration_a_archive_absent path=%s", config_a_path)
        config_a_section = """## Configuration A direct-LLM result

No historical standalone Configuration A result was supplied in this reproduction.
This optional archive is not required for detector generation, local paired A/B/C
pilots, or replay of a complete locked-study evidence package.
"""
    else:
        LOGGER.info("configuration_a_archive_loaded path=%s", config_a_path)
        quoted_answer = config_a["llm_answer"].replace(chr(10), chr(10) + "> ")
        config_a_section = f"""## Configuration A direct-LLM result

- Model: `{config_a['llm_model']}`
- Retrieval used: `{config_a['retrieval_used']}`
- Guardrail enforcement used: `{config_a['guardrail_enforcement_used']}`
- Supplied evidence records: {len(config_a['evidence_ids'])}

Generated answer:

> {quoted_answer}

### Baseline-quality observation

The answer introduces “NodeList” terminology that is not present in the supplied
evidence and makes overbroad statements about real-bank validity. It must therefore not
be treated as a verified investigation conclusion. This failure is relevant experimental
evidence for comparing ungrounded Configuration A with retrieval-grounded B and guarded C.
"""
    alert_counts = {
        "ULB": sum(1 for _ in (
            PROJECT_ROOT / "artifacts" / "ulb" / "llm_alert_evidence.jsonl"
        ).open(encoding="utf-8")),
        "Sparkov": sum(1 for _ in (
            PROJECT_ROOT / "artifacts" / "sparkov" / "llm_alert_evidence.jsonl"
        ).open(encoding="utf-8")),
    }

    table = markdown_table(tuned)
    status_table = markdown_table(pd.DataFrame(statuses))
    report = f"""# Executed Notebook Results

Generated: {datetime.now(timezone.utc).isoformat()}

## Locked test results

The following rows use model and threshold choices made on validation data. They are
point estimates; confidence intervals have not yet been added.

{table}

ULB produced {alert_counts['ULB']:,} test alerts at its validation-selected threshold.
Sparkov produced {alert_counts['Sparkov']:,} official-test alerts at its
validation-selected threshold.

{config_a_section}

## A/B/C experiment status

The executed A/B/C notebook uses the deterministic integration engine. It verifies the
paired service, transaction-aware retrieval, C contract enforcement, telemetry, and
label-leakage controls; it is **not an LLM effectiveness result**. No configuration may
be declared superior. The DEC-020 current study may report only registered automated
structural, provenance, governance and Mac-specific operational outcomes from the locked
repeated Qwen experiment. Human-rated factual grounding, hallucination, semantic citation
correctness and retrieval relevance remain future work; draft judgments stay unscored.

## Notebook execution status

{status_table}

## Deliberately excluded archival notebooks

- `ML_Code.ipynb`: reviewed legacy workflow with leakage and an altered test set; its
  historical scores are not regenerated as thesis evidence.
- `notebooks/LLM_Baseline_Configuration_A_ORIGINAL.ipynb`: immutable Colab/Google Drive
  archive tied to personal paths and the obsolete balanced-test artifacts.

Their corrected replacements were executed above.
"""
    (OUTPUT_DIR / "RESULTS_SUMMARY.md").write_text(report, encoding="utf-8")
    LOGGER.info("results_report_build_completed output=%s", OUTPUT_DIR)
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("results_report_build_failed")
        raise
