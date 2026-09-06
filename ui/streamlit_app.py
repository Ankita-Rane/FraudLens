"""Three-tab thesis prototype for fraud investigation configurations A, B, and C."""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import time

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import load_unified_evidence  # noqa: E402
from src.historical import load_historical_profiles  # noqa: E402
from src.evidence_answer import build_evidence_answer  # noqa: E402
from src.llm_backend import (  # noqa: E402
    CONFIGURATION_REGISTRY,
    Configuration,
    build_observation,
    prepare_request,
    validate_configuration_c_response,
)
from src.llm_provider import LocalQwenProvider, provider_is_available  # noqa: E402
from src.model_evidence_retrieval import scope_evidence_for_question  # noqa: E402
from src.retrieval import (  # noqa: E402
    chunk_policy_text,
    load_policy_directory,
    retrieve_policy_evidence,
)


st.set_page_config(
    page_title="Fraud Investigation A/B/C Lab",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1450px;}
      [data-testid="stMetric"] {border: 1px solid #d9e2ec; border-radius: 12px; padding: 12px;}
      .small-note {color: #52606d; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_evidence_or_none():
    try:
        return load_unified_evidence(PROJECT_ROOT), None
    except FileNotFoundError as error:
        return None, str(error)


def load_historical_or_none():
    try:
        return load_historical_profiles(PROJECT_ROOT)
    except FileNotFoundError:
        return None


def historical_only_bundle(profiles: dict) -> dict:
    """Allow an honest Config A historical demo before model fitting is complete."""
    return {
        "combination_policy": (
            "Historical training aggregates only; final model alerts are not generated yet."
        ),
        "interpretation_notes": {
            dataset["dataset_id"]: dataset["interpretation_limit"]
            for dataset in profiles["datasets"]
        },
        "cross_dataset_test_comparison": [],
        "historical_profiles": profiles,
        "datasets": [
            {
                "dataset_id": dataset["dataset_id"],
                "model_name": "Not generated",
                "selected_threshold": None,
                "primary_metric": "average_precision",
                "test_metrics": [],
                "alert_count": 0,
                "top_alerts": [],
            }
            for dataset in profiles["datasets"]
        ],
    }


def load_uploaded_policies(uploaded_files) -> list:
    chunks = []
    for uploaded_file in uploaded_files or []:
        try:
            text = uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            st.sidebar.error(f"{uploaded_file.name} is not UTF-8 text.")
            continue
        chunks.extend(chunk_policy_text(
            title=Path(uploaded_file.name).stem,
            text=text,
            source=f"uploaded://{uploaded_file.name}",
        ))
    return chunks


def local_provider() -> LocalQwenProvider:
    if "local_qwen_provider" not in st.session_state:
        st.session_state["local_qwen_provider"] = LocalQwenProvider()
    return st.session_state["local_qwen_provider"]


def save_ui_run(payload: dict, request_id: str) -> Path:
    output_dir = PROJECT_ROOT / "artifacts" / "ui_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{request_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def render_configuration(
    mode: Configuration,
    bundle: dict | None,
    policy_chunks: list,
    engine: str,
    maximum_alerts: int,
) -> None:
    metadata = CONFIGURATION_REGISTRY[mode.value]
    st.subheader(f"Configuration {mode.value} — {metadata['title']}")

    capabilities = []
    capabilities.append("ULB + Sparkov fraud-classifier evidence")
    capabilities.append("policy retrieval" if metadata["retrieval"] else "no retrieval")
    capabilities.append(
        "enforced guardrails" if metadata["guardrail_enforcement"] else "no programmatic guardrails"
    )
    st.caption(" · ".join(capabilities))

    model_score_gate = 0.70
    if mode is Configuration.C:
        model_score_gate = st.slider(
            "Fraud-model score gate for LLM generation",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.01,
            key="c_model_score_gate",
        )

    history_key = f"chat_history_{mode.value}"
    messages = st.session_state.setdefault(history_key, [])

    header_columns = st.columns([5, 1])
    with header_columns[1]:
        if st.button("Clear", key=f"clear_{mode.value}", use_container_width=True):
            st.session_state[history_key] = []
            st.rerun()

    if not messages:
        examples = {
            Configuration.A: (
                "Ask naturally, for example: “Compare historical fraud rates in ULB "
                "and Sparkov” or “Which Sparkov categories had the highest fraud rates?”"
            ),
            Configuration.B: (
                "Ask a model-and-policy question after adding an approved policy document."
            ),
            Configuration.C: (
                "Ask for a guarded escalation decision after model alerts and policy "
                "evidence are available."
            ),
        }[mode]
        with st.chat_message("assistant"):
            st.write(examples)

    chat_container = st.container(height=480)
    with chat_container:
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                metadata = message.get("metadata", {})
                evidence_ids = metadata.get("evidence_ids", [])
                if evidence_ids:
                    st.caption("Evidence: " + ", ".join(evidence_ids))
                if metadata.get("prompt"):
                    with st.expander("Backend request and routed evidence"):
                        st.code(metadata["prompt"], language="json")
                if metadata.get("trace"):
                    with st.expander("Trace"):
                        st.json(metadata["trace"])

    disabled = bundle is None
    prompt = st.chat_input(
        "Ask about historical fraud, model performance, or high-risk alerts...",
        key=f"chat_input_{mode.value}",
        disabled=disabled,
    )
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    scoped_bundle, routing = scope_evidence_for_question(
        bundle,
        prompt,
        historical_records_per_dataset=3,
        alerts_per_dataset=maximum_alerts,
    )
    retrieved = (
        retrieve_policy_evidence(prompt, policy_chunks, top_k=4)
        if mode in {Configuration.B, Configuration.C}
        else []
    )

    try:
        request = prepare_request(
            configuration=mode,
            question=prompt,
            unified_evidence=scoped_bundle,
            retrieved_documents=retrieved,
            model_score_gate=model_score_gate,
        )
    except ValueError as error:
        messages.append({
            "role": "assistant",
            "content": f"I cannot answer this configuration yet: {error}",
        })
        st.rerun()

    if not request.should_call_llm:
        messages.append({
            "role": "assistant",
            "content": request.suppression_reason,
            "metadata": {"evidence_ids": request.evidence_ids, "prompt": request.prompt},
        })
        st.rerun()

    started = time.monotonic()
    if engine == "Evidence-only demo":
        answer, cited_ids = build_evidence_answer(prompt, scoped_bundle)
        if retrieved:
            policy_lines = "\n".join(
                f"- `{document.evidence_id}`: {document.text}"
                for document in retrieved
            )
            answer += "\n\n**Retrieved policy passages:**\n" + policy_lines
            cited_ids.extend(document.evidence_id for document in retrieved)
        payload = {
            "request_id": request.request_id,
            "configuration": request.configuration,
            "question": prompt,
            "engine": "deterministic-evidence-demo",
            "answer": answer,
            "evidence_ids": list(dict.fromkeys(cited_ids)),
            "routing": routing,
        }
        save_ui_run(payload, request.request_id)
        messages.append({
            "role": "assistant",
            "content": answer,
            "metadata": {
                "evidence_ids": payload["evidence_ids"],
                "prompt": request.prompt,
            },
        })
        st.rerun()

    try:
        with st.spinner("Running local Qwen model..."):
            generation = local_provider().generate(request.prompt)
    except Exception as error:
        messages.append({
            "role": "assistant",
            "content": f"The local LLM could not run: {error}",
            "metadata": {"evidence_ids": request.evidence_ids},
        })
        st.rerun()

    payload = {
        "request_id": request.request_id,
        "configuration": request.configuration,
        "question": prompt,
        "evidence_ids": request.evidence_ids,
        "model_name": generation.model_name,
        "answer": generation.text,
        "routing": routing,
    }
    answer_text = generation.text
    trace = None
    if mode is Configuration.C:
        try:
            payload["validated_answer"] = validate_configuration_c_response(
                generation.text,
                allowed_evidence_ids=request.evidence_ids,
            )
            answer_text = json.dumps(payload["validated_answer"], indent=2)
        except ValueError as error:
            payload["guardrail_error"] = str(error)
            answer_text = f"Response blocked by Configuration C: {error}"
        trace = build_observation(
            request,
            model_name=generation.model_name,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            started_monotonic=started,
        )
        payload["trace"] = trace
    save_ui_run(payload, request.request_id)
    messages.append({
        "role": "assistant",
        "content": answer_text,
        "metadata": {
            "evidence_ids": request.evidence_ids,
            "prompt": request.prompt,
            "trace": trace,
        },
    })
    st.rerun()


st.title("Fraud Investigation A/B/C Lab")
st.write(
    "Compare direct, retrieval-grounded, and guardrail-aware explanations using the "
    "same source-labelled ULB and Sparkov model evidence."
)

bundle, evidence_error = load_evidence_or_none()
historical_profiles = load_historical_or_none()
configuration_a_bundle = bundle
if configuration_a_bundle is None and historical_profiles is not None:
    configuration_a_bundle = historical_only_bundle(historical_profiles)

with st.sidebar:
    st.header("Run controls")
    engine = st.selectbox(
        "Response engine",
        ["Evidence-only demo", "Local Qwen"],
        help=(
            "Evidence-only demo returns transparent deterministic summaries. Local Qwen "
            "performs the thesis Configuration A/B/C generation."
        ),
    )
    maximum_alerts = st.slider("Alerts per dataset", 1, 10, 3)
    uploaded_files = st.file_uploader(
        "Approved policy documents for B/C",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    st.caption(
        "The frozen public-source corpus is loaded from datasets/policy_sources/source/."
    )
    if engine == "Local Qwen" and not provider_is_available():
        st.warning("Install requirements-llm.txt before using Local Qwen.")
    elif engine == "Local Qwen":
        qwen_cache = (
            PROJECT_ROOT / ".cache" / "huggingface" / "hub" /
            "models--Qwen--Qwen3-0.6B"
        )
        if qwen_cache.exists():
            st.success("Local Qwen dependencies and cached weights are available.")
        else:
            st.info("Qwen weights will download on first use.")
        if st.button("Load Qwen now", use_container_width=True):
            try:
                with st.spinner("Loading Qwen/Qwen3-0.6B into this UI session..."):
                    local_provider()
                st.success("Qwen is loaded and ready for chat.")
            except Exception as error:
                st.error(f"Qwen could not load: {error}")

local_policy_chunks = load_policy_directory(
    PROJECT_ROOT / "datasets" / "policy_sources" / "source"
)
policy_chunks = local_policy_chunks + load_uploaded_policies(uploaded_files)

if bundle is None:
    st.warning(
        "Unified model evidence is not generated yet. Run the ULB notebook, then the "
        "Sparkov notebook, then `python3 tools/build_unified_evidence.py`."
    )
    with st.expander("Technical detail"):
        st.code(evidence_error)
    if historical_profiles is not None:
        st.info(
            "Configuration A historical prompt demos are available from real training "
            "aggregates. B/C and alert-level questions remain disabled until model "
            "artifacts are generated."
        )
else:
    columns = st.columns(len(bundle["datasets"]))
    for column, dataset in zip(columns, bundle["datasets"]):
        with column:
            st.metric(f"{dataset['dataset_id']} alerts", dataset["alert_count"])
            st.caption(f"Selected model: {dataset['model_name']}")

tab_a, tab_b, tab_c = st.tabs([
    "A · Baseline",
    "B · Retrieval augmented",
    "C · Guardrail-aware agent",
])

with tab_a:
    render_configuration(
        Configuration.A, configuration_a_bundle, policy_chunks, engine, maximum_alerts
    )
with tab_b:
    render_configuration(
        Configuration.B, bundle, policy_chunks, engine, maximum_alerts
    )
with tab_c:
    render_configuration(
        Configuration.C, bundle, policy_chunks, engine, maximum_alerts
    )

st.divider()
st.caption(
    "Research prototype: model alerts and LLM explanations support human investigation; "
    "they are not autonomous payment or enforcement decisions."
)
