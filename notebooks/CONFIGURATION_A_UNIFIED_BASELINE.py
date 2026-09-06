# %% [markdown]
# # Configuration A — Unified Direct-LLM Baseline
#
# This local replacement preserves the supplied baseline's experimental condition:
#
# `ULB + Sparkov model evidence → direct LLM → investigation answer`
#
# It uses no policy retrieval and no programmatic guardrail enforcement. Unlike the
# original Colab prototype, it consumes the leakage-safe outputs of both current model
# notebooks and keeps their incompatible feature spaces source-labelled and nested.

# %%
from __future__ import annotations

from pathlib import Path
import json
import os
import platform
import sys


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "evidence.py").exists():
            return candidate
    raise FileNotFoundError("Run from thesis_code or one of its subdirectories.")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "configuration_a"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import load_unified_evidence  # noqa: E402
from src.llm_backend import Configuration, prepare_request  # noqa: E402
from src.model_evidence_retrieval import scope_evidence_for_question  # noqa: E402

print({"python": platform.python_version(), "project_root": str(PROJECT_ROOT)})

# %% [markdown]
# ## 1. Load the combined model evidence
#
# First run both modelling notebooks, then run:
#
# ```bash
# python3 tools/build_unified_evidence.py
# ```
#
# Offline `actual_class` values are deliberately absent from this LLM context.

# %%
unified_evidence = load_unified_evidence(PROJECT_ROOT)
print("Evidence schema:", unified_evidence["schema_version"])
for dataset in unified_evidence["datasets"]:
    print({
        "dataset": dataset["dataset_id"],
        "selected_model": dataset["model_name"],
        "total_alerts": dataset["alert_count"],
        "alerts_in_context": len(dataset["top_alerts"]),
    })

# %% [markdown]
# ## 2. Bound the direct prompt size
#
# Configuration A still receives evidence from both datasets, but only the highest-score
# alerts are included. This is deterministic score ordering, not retrieval.

# %%
QUESTION = (
    "Considering the ULB and Sparkov model outputs together, summarize the highest-risk "
    "alerts, explain what can and cannot be inferred, and recommend investigation priorities."
)

prompt_evidence, evidence_routing = scope_evidence_for_question(
    unified_evidence,
    QUESTION,
    historical_records_per_dataset=1,
    alerts_per_dataset=2,
)
print("Question-aware evidence routing:", evidence_routing)

prepared_request = prepare_request(
    configuration=Configuration.A,
    question=QUESTION,
    unified_evidence=prompt_evidence,
)
print("Request ID:", prepared_request.request_id)
print("Evidence records:", len(prepared_request.evidence_ids))
print(prepared_request.prompt[:4_000])

# %% [markdown]
# ## 3. Optional local Qwen baseline
#
# Install the optional packages in `requirements-llm.txt`. Model weights are downloaded
# from Hugging Face on the first run. Set `RUN_LOCAL_LLM = True` only when ready.

# %%
LLM_MODEL = "Qwen/Qwen3-0.6B"
RUN_LOCAL_LLM = os.getenv("RUN_LOCAL_LLM", "false").lower() == "true"


def load_local_model(model_name: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install optional packages with: "
            "python3 -m pip install -r requirements-llm.txt"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, torch


def generate_direct_response(prompt: str, tokenizer, model, torch) -> str:
    """Direct generation matching the supplied Configuration A design."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a credit-card fraud investigation assistant. Analyze only "
                "the model evidence in the user prompt and do not invent unavailable "
                "facts or meanings for anonymized features."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        torch.manual_seed(42)
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
        )
    generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# %% [markdown]
# ## 4. Execute and save Configuration A

# %%
if RUN_LOCAL_LLM:
    tokenizer, model, torch = load_local_model(LLM_MODEL)
    llm_answer = generate_direct_response(
        prepared_request.prompt,
        tokenizer,
        model,
        torch,
    )
else:
    llm_answer = "NOT_RUN: set RUN_LOCAL_LLM=True after installing optional dependencies."

result = {
    "request_id": prepared_request.request_id,
    "configuration": prepared_request.configuration,
    "question": prepared_request.question,
    "evidence_ids": prepared_request.evidence_ids,
    "llm_model": LLM_MODEL,
    "llm_answer": llm_answer,
    "retrieval_used": False,
    "guardrail_enforcement_used": False,
}

output_path = ARTIFACT_DIR / "configuration_a_result.json"
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(llm_answer)
print("Saved:", output_path)

# %% [markdown]
# ## Interpretation boundary
#
# Configuration A intentionally lacks retrieval grounding and enforcement. Its output
# is an experimental baseline, not an operational fraud decision. Compare it with B and
# C using the same question, alert evidence, LLM version, decoding parameters, and run
# conditions.
