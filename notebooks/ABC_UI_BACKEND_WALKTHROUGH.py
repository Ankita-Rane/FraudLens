# %% [markdown]
# # A/B/C UI and Backend Integration Walkthrough
#
# This notebook is the readable companion to the reusable Python modules:
#
# - `src/evidence.py` combines ULB and Sparkov outputs without merging predictors;
# - `src/retrieval.py` retrieves approved policy chunks for B and C;
# - `src/llm_backend.py` prepares controlled A/B/C requests and validates C;
# - `src/llm_provider.py` supplies the optional local Qwen implementation;
# - `src/abc_service.py` provides the SOLID A/B/C orchestration;
# - `web/` renders the primary Node.js comparison interface.
#
# Keep the `.py` modules as the application's importable implementation. Use this
# notebook to understand and demonstrate the flow in the thesis.

# %%
from __future__ import annotations

from pathlib import Path
import json
import sys

from IPython.display import display


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "llm_backend.py").exists():
            return candidate
    raise FileNotFoundError("Run from thesis_code or one of its subdirectories.")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import build_unified_evidence, load_unified_evidence  # noqa: E402
from src.historical import load_historical_profiles  # noqa: E402
from src.llm_backend import (  # noqa: E402
    CONFIGURATION_REGISTRY,
    Configuration,
    prepare_request,
    validate_configuration_c_response,
)
from src.retrieval import load_policy_directory, retrieve_policy_evidence  # noqa: E402
from src.model_evidence_retrieval import scope_evidence_for_question  # noqa: E402
from src.abc_service import build_transaction_aware_retrieval_query  # noqa: E402

display(CONFIGURATION_REGISTRY)

# %% [markdown]
# ## Historical questions supported by Configuration A
#
# Training-only aggregate profiles let A answer questions about prevalence, transaction
# amount, category, time, age, and monthly patterns without receiving raw personal data.
# Generate them once with:
#
# ```bash
# python3 tools/build_historical_profiles.py
# ```

# %%
try:
    historical_profiles = load_historical_profiles(PROJECT_ROOT)
    for dataset in historical_profiles["datasets"]:
        prevalence = next(
            record for record in dataset["records"]
            if record["topic"] == "historical_prevalence"
        )
        display({
            "dataset": dataset["dataset_id"],
            "scope": dataset["scope"],
            **prevalence["facts"],
        })
except FileNotFoundError as error:
    historical_profiles = None
    print(error)

# %% [markdown]
# ## 1. Generate the shared evidence artifact
#
# Run the ULB and Sparkov modelling notebooks first. Each writes a source-labelled
# `llm_alert_evidence.jsonl`. The following function then builds one compact context and
# one complete alert stream.

# %%
try:
    unified_evidence = build_unified_evidence(
        PROJECT_ROOT,
        max_alerts_per_dataset=25,
    )
    print("Unified evidence generated.")
except FileNotFoundError as error:
    unified_evidence = None
    print(error)

# %% [markdown]
# ## 2. Inspect source-preserving fusion
#
# The two dataset blocks appear in one JSON object. Their alert features remain nested,
# and the compact LLM context does not contain the offline `actual_class` label.

# %%
if unified_evidence is not None:
    overview = [
        {
            "dataset": dataset["dataset_id"],
            "model": dataset["model_name"],
            "threshold": dataset["selected_threshold"],
            "total_alerts": dataset["alert_count"],
            "alerts_in_context": len(dataset["top_alerts"]),
        }
        for dataset in unified_evidence["datasets"]
    ]
    display(overview)
    display(unified_evidence["cross_dataset_test_comparison"])

# %% [markdown]
# ## 3. Prepare Configuration A
#
# A receives only model evidence. No retriever is called and no policy passage is added.

# %%
QUESTION = "Which high-risk alerts should an analyst prioritize across both datasets?"

if unified_evidence is not None:
    question_evidence, routing = scope_evidence_for_question(
        unified_evidence,
        QUESTION,
        historical_records_per_dataset=3,
        alerts_per_dataset=3,
    )
    display(routing)
    request_a = prepare_request(
        configuration=Configuration.A,
        question=QUESTION,
        unified_evidence=question_evidence,
    )
    print(request_a.request_id)
    print(request_a.prompt[:3_000])

# %% [markdown]
# ## 4. Retrieve approved policy evidence for B and C
#
# Frozen HTML/text documents registered under `datasets/policy_sources/` are loaded.
# Retrieval uses an inspectable TF-IDF baseline and returns stable evidence IDs, source
# metadata, chunk hashes, ranks, and relevance scores.

# %%
policy_chunks = load_policy_directory(
    PROJECT_ROOT / "datasets" / "policy_sources" / "source"
)
retrieval_query = (
    build_transaction_aware_retrieval_query(QUESTION, question_evidence)
    if unified_evidence is not None else QUESTION
)
retrieved_documents = retrieve_policy_evidence(
    retrieval_query,
    policy_chunks,
    top_k=4,
)
print("Policy chunks:", len(policy_chunks))
print("Retrieved:", [document.evidence_id for document in retrieved_documents])

# %% [markdown]
# ## 5. Prepare Configuration B
#
# B receives the same model evidence plus retrieved policy passages. It does not enforce
# the output structure or citations after generation.

# %%
if unified_evidence is not None and retrieved_documents:
    request_b = prepare_request(
        configuration=Configuration.B,
        question=QUESTION,
        unified_evidence=question_evidence,
        retrieved_documents=retrieved_documents,
    )
    print(request_b.request_id)
    print(request_b.prompt[:3_000])
else:
    print("Add approved policy documents before preparing Configuration B.")

# %% [markdown]
# ## 6. Prepare and validate Configuration C
#
# C applies a fraud-model score gate before generation; this is not LLM confidence.
# After generation, the response must be valid JSON, attach evidence IDs to each atomic
# claim, disclose public-source scope, and cite both model and policy evidence.

# %%
if unified_evidence is not None and retrieved_documents:
    request_c = prepare_request(
        configuration=Configuration.C,
        question=QUESTION,
        unified_evidence=question_evidence,
        retrieved_documents=retrieved_documents,
        model_score_gate=0.70,
    )
    print({
        "request_id": request_c.request_id,
        "should_call_llm": request_c.should_call_llm,
        "suppression_reason": request_c.suppression_reason,
        "allowed_evidence_ids": request_c.evidence_ids,
    })
else:
    request_c = None
    print("Add model artifacts and approved policy documents before preparing C.")

# %%
# Replace this illustrative shape with the actual LLM response before validation.
if request_c is not None and request_c.should_call_llm:
    model_citation = next(
        value for value in request_c.evidence_ids
        if value.startswith("model-output:")
    )
    policy_citation = next(
        value for value in request_c.evidence_ids
        if value.startswith("policy:")
    )
    citations = [model_citation, policy_citation]
    response_shape = {
        "answer": "Evidence-grounded answer goes here.",
        "risk_level": "high",
        "self_reported_confidence": None,
        "citations": citations,
        "claims": [{
            "statement": "This alert requires human investigation under the supplied evidence.",
            "citations": citations,
        }],
        "recommended_actions": ["Route to human analyst review."],
        "limitations": [
            "Public sources are not institution-specific operating policy or legal advice."
        ],
    }
    print(json.dumps(response_shape, indent=2))
    validated = validate_configuration_c_response(
        json.dumps(response_shape),
        allowed_evidence_ids=request_c.evidence_ids,
    )
    print("Valid C contract:", bool(validated))

# %% [markdown]
# ## 7. Launch the Node.js comparison website
#
# From a terminal in `thesis_code`:
#
# ```bash
# ./run_node_ui.sh
# ```
#
# Start with the deterministic integration engine, then use local Qwen only for a locked
# experiment. The legacy Streamlit UI remains for compatibility but is not the primary
# research interface.
