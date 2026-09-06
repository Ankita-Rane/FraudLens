# %% [markdown]
# # Configuration A/B/C controlled investigation experiment
#
# This notebook runs the same frozen Sparkov transaction and investigator question
# through all three LLM-system conditions. Detection output is held fixed; only the
# explanation system changes. The evaluation-only fraud label is never included in a
# prompt.
#
# - **A:** direct explanation from model evidence.
# - **B:** model evidence plus retrieval from frozen, provenance-registered sources.
# - **C:** the same evidence as B plus fraud-model score gating, input controls, per-claim
#   citations, public-source scope disclosure, response blocking, and observability.
#
# The deterministic engine below is an inspectable integration demonstration, not an
# LLM experimental result. Set `USE_LOCAL_QWEN = True` for the local Qwen condition.

# %%
from pathlib import Path
from hashlib import sha256
import json
import sys
import uuid

import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc_evaluation import METRIC_CATALOG, aggregate_runs, validate_paired_design
from src.abc_service import (
    CsvSampleRepository,
    DeterministicDemoGenerator,
    InvestigationCommand,
    build_default_service,
)
from src.artifact_audit import audit_input_artifacts
from src.llm_provider import LocalQwenProvider

# %% [markdown]
# ## 1. Reproducibility and input integrity
#
# The run must stop if a required input is absent, outside `datasets/`, or has a hash
# mismatch. Generated models and run records remain outputs under `artifacts/`.

# %%
artifact_audit = audit_input_artifacts(PROJECT_ROOT)
print(json.dumps(artifact_audit, indent=2))
assert artifact_audit["status"] == "passed", artifact_audit["errors"]

# %% [markdown]
# ## 2. Frozen transaction cases
#
# `evaluation_only_actual_class` is available only to the later offline evaluator. The
# repository's default listing and the service prompt deliberately remove it.

# %%
sample_repository = CsvSampleRepository(
    PROJECT_ROOT / "datasets" / "sparkov" / "runtime_transaction_samples.csv"
)
samples = sample_repository.list(include_evaluation_labels=False)
pd.DataFrame(samples)

# %% [markdown]
# ## 3. Controlled run settings
#
# Freeze these settings before collecting final thesis results. The pilot uses four
# deliberately varied cases and one common question. Final evaluation should add a
# larger stratified sample, repeated runs, blinded claim annotation, and a retrieval
# relevance gold set.

# %%
USE_LOCAL_QWEN = False
REPEATS = 1
MODEL_SCORE_GATE = 0.70
QUESTION = (
    "Assess this transaction. What evidence supports the risk finding, what policy "
    "context applies, and what should a human investigator do next?"
)

generator = LocalQwenProvider() if USE_LOCAL_QWEN else DeterministicDemoGenerator()
service = build_default_service(PROJECT_ROOT, generator)
EXPERIMENT_ID = str(uuid.uuid4())
QUESTION_ID = sha256(" ".join(QUESTION.lower().split()).encode()).hexdigest()[:16]

# %% [markdown]
# ## 4. Paired A/B/C execution
#
# Each `(sample_id, question, repeat)` cell is run once under A, B, and C. C may
# legitimately suppress a below-gate model score; suppression is recorded as an outcome,
# not silently dropped.

# %%
runs = []
base_order = ("A", "B", "C")
for repeat in range(1, REPEATS + 1):
    for sample in samples:
        offset = (repeat - 1) % len(base_order)
        sequence = base_order[offset:] + base_order[:offset]
        for run_order, configuration in enumerate(sequence, start=1):
            result = service.run(InvestigationCommand(
                configuration=configuration,
                question=QUESTION,
                sample_id=sample["sample_id"],
                model_score_gate=MODEL_SCORE_GATE,
                experiment_id=EXPERIMENT_ID,
                question_id=QUESTION_ID,
                repeat=repeat,
                run_order=run_order,
            )).to_dict()
            runs.append(result)

paired_design = validate_paired_design(runs)
paired_design

# %% [markdown]
# ## 5. Side-by-side operational results

# %%
run_table = pd.DataFrame([
    {
        "sample_id": run["sample_id"],
        "configuration": run["configuration"],
        "status": run["status"],
        "end_to_end_latency_ms": run["trace"].get("end_to_end_latency_ms"),
        "generation_latency_ms": run["trace"].get("latency_ms"),
        "retrieval_latency_ms": run["trace"].get("retrieval_latency_ms"),
        "input_tokens": run["trace"].get("input_tokens"),
        "output_tokens": run["trace"].get("output_tokens"),
        "cost_usd": run["trace"].get("estimated_cost_usd"),
        "guardrail_violations": len(run["guardrail_violations"]),
        "retrieved_passages": len(run["retrieved_policy"]),
        "answer": run["answer"],
    }
    for run in runs
])
run_table

# %% [markdown]
# ## 6. Automatic system metrics
#
# Latency, trace completeness, cost, resources, blocking, and consistency can be
# calculated from telemetry. Hallucination rate, citation correctness, and policy
# grounding completeness stay null until an independent blinded claim annotation is
# supplied. This prevents circular LLM self-evaluation.

# %%
automatic_summary = pd.DataFrame(aggregate_runs(runs))
automatic_summary

# %%
pd.DataFrame(METRIC_CATALOG)

# %% [markdown]
# ## 7. Interpretation limits and next evaluation step
#
# - ULB is real but anonymized; never assign business meanings to V1-V28.
# - Sparkov is synthetic and does not establish production-bank validity.
# - The public CFPB/Federal Reserve/NIST/BIS corpus is versioned research grounding,
#   not an issuing bank's operating policy or legal advice.
# - Retrieval similarity is document-query relevance, not fraud probability.
# - Allowed citation IDs prove provenance, not claim-to-source entailment.
# - Grouped reference perturbation is model sensitivity, not SHAP or causality.
# - The deterministic integration engine must not be reported as an LLM result.
# - Final comparison requires a frozen question set, at least two blinded reviewers,
#   inter-rater agreement, repeated runs, confidence intervals, and paired tests.
