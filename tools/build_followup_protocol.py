"""Build, validate and explicitly lock the six-condition follow-up protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.followup_conditions import (  # noqa: E402
    FOLLOWUP_CONDITION_IDS,
    condition_registry_payload,
    condition_registry_sha256,
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--qwen-revision", required=True)
    parser.add_argument(
        "--dense-model-id", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--dense-revision", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--model-score-gate", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--approve-panel", action="store_true")
    parser.add_argument("--confirm-retrieval-diagnostic-locked", action="store_true")
    parser.add_argument(
        "--retrieval-diagnostic", type=Path,
        help="Completed locked researcher-authored known-item diagnostic JSON.",
    )
    parser.add_argument("--retrieval-diagnostic-audit", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    panel_path = args.panel_manifest.resolve()
    question_path = args.questions.resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    question_count = len(questions.get("questions", []))
    if question_count != 1:
        raise RuntimeError(
            "The registered follow-up design requires exactly one primary question."
        )
    if panel.get("scientific_status") not in {
        "candidate_panel_not_protocol_locked", "researcher_approved_followup_panel"
    }:
        raise RuntimeError("Panel manifest is not an eligible final follow-up panel.")
    sample_ids: list[str] = []
    for row in panel.get("panels", []):
        generation_path = PROJECT_ROOT / row["generation_panel_path"]
        import pandas as pd
        sample_ids.extend(pd.read_csv(generation_path)["sample_id"].astype(str).tolist())
    if len(sample_ids) != 550 or len(set(sample_ids)) != 550:
        raise RuntimeError("The final protocol requires exactly 550 unique sample IDs.")
    if args.batch_size < 1 or args.repeats != 1:
        raise ValueError("Final design requires repeats=1 so the total remains 3,300 calls.")
    batches = []
    for index, start in enumerate(range(0, len(sample_ids), args.batch_size), start=1):
        members = sample_ids[start:start + args.batch_size]
        batches.append({
            "batch_id": f"FOLLOWUP-BATCH-{index:03d}",
            "sample_ids": members,
            "planned_calls": (
                len(members) * question_count * len(FOLLOWUP_CONDITION_IDS) * args.repeats
            ),
            "cell_offset": start * question_count,
        })
    locked = bool(args.lock)
    if locked and not (args.approve_panel and args.confirm_retrieval_diagnostic_locked):
        raise RuntimeError(
            "Locking requires --approve-panel and --confirm-retrieval-diagnostic-locked."
        )
    retrieval_diagnostic = None
    retrieval_diagnostic_path = None
    if args.retrieval_diagnostic is not None:
        retrieval_diagnostic_path = args.retrieval_diagnostic.resolve()
        retrieval_diagnostic = json.loads(
            retrieval_diagnostic_path.read_text(encoding="utf-8")
        )
    if locked:
        if retrieval_diagnostic is None:
            raise RuntimeError("Locking requires --retrieval-diagnostic evidence.")
        if retrieval_diagnostic.get("scientific_status") != "completed_scope_safe_known_item_diagnostic":
            raise RuntimeError(
                "Retrieval diagnostic is not a completed scope-safe known-item diagnostic."
            )
        approved = set(retrieval_diagnostic.get("evaluated_strategy_ids", []))
        if not {"lexical_tfidf", "dense", "hybrid_rrf"}.issubset(approved):
            raise RuntimeError("Retrieval diagnostic did not evaluate all three strategies.")
        if retrieval_diagnostic.get("all_strategies_retained_regardless_of_result") is not True:
            raise RuntimeError("Diagnostic must retain all strategies regardless of result.")
        dense_manifest = retrieval_diagnostic["strategies"]["dense"]["manifest"]
        hybrid_manifest = retrieval_diagnostic["strategies"]["hybrid_rrf"]["manifest"]
        if retrieval_diagnostic.get("top_k") != args.top_k:
            raise RuntimeError("Protocol top_k does not match diagnostic evidence.")
        if (dense_manifest.get("encoder_model_id"), dense_manifest.get("encoder_model_revision")) != (args.dense_model_id, args.dense_revision):
            raise RuntimeError("Protocol dense encoder ID/revision does not match diagnostic evidence.")
        if (hybrid_manifest.get("rrf_k"), hybrid_manifest.get("candidate_multiplier")) != (args.rrf_k, args.candidate_multiplier):
            raise RuntimeError("Protocol RRF parameters do not match diagnostic evidence.")
        if args.retrieval_diagnostic_audit is None:
            raise RuntimeError("Locking requires --retrieval-diagnostic-audit proof.")
        audit_path = args.retrieval_diagnostic_audit.resolve()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("scientific_status") != "verified_scope_safe_retrieval_diagnostic_audit" or audit.get("all_checks_passed") is not True:
            raise RuntimeError("Retrieval diagnostic audit has not passed all checks.")
        if audit.get("results_sha256") != _sha256(retrieval_diagnostic_path):
            raise RuntimeError("Retrieval diagnostic audit does not bind the supplied results hash.")
    else:
        audit_path = args.retrieval_diagnostic_audit.resolve() if args.retrieval_diagnostic_audit else None
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "locked_followup_protocol" if locked else "candidate_followup_protocol"
        ),
        "study_id": args.study_id,
        "analysis_unit": "one evaluation transaction within one dataset",
        "sampling_frame": "label_stratified_TP_FP_FN_TN_transactions",
        "sample_count": len(sample_ids),
        "condition_ids": list(FOLLOWUP_CONDITION_IDS),
        "condition_registry": condition_registry_payload(),
        "condition_registry_sha256": condition_registry_sha256(),
        "panel_manifest_path": str(panel_path.relative_to(PROJECT_ROOT)),
        "panel_manifest_sha256": _sha256(panel_path),
        "question_manifest_path": str(question_path.relative_to(PROJECT_ROOT)),
        "question_manifest_sha256": _sha256(question_path),
        "question_count": question_count,
        "retrieval_diagnostic": {
            "path": (
                str(retrieval_diagnostic_path.relative_to(PROJECT_ROOT))
                if retrieval_diagnostic_path else None
            ),
            "sha256": _sha256(retrieval_diagnostic_path) if retrieval_diagnostic_path else None,
            "scientific_status": (
                retrieval_diagnostic.get("scientific_status")
                if retrieval_diagnostic else None
            ),
            "audit_path": str(audit_path.relative_to(PROJECT_ROOT)) if audit_path else None,
            "audit_sha256": _sha256(audit_path) if audit_path else None,
        },
        "qwen": {
            "model_id": args.qwen_model_id,
            "revision": args.qwen_revision,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "deterministic provider settings frozen by implementation",
        },
        "dense_encoder": {
            "model_id": args.dense_model_id,
            "revision": args.dense_revision,
            "normalization": "l2",
            "similarity": "cosine_dot_product_of_normalized_embeddings",
        },
        "retrieval": {
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "candidate_multiplier": args.candidate_multiplier,
        },
        "model_score_gate": args.model_score_gate,
        "repeats": args.repeats,
        "ground_truth_withheld_from_generation": True,
        "comparisons": [
            "B_lexical-A_direct", "B_dense-A_direct", "B_dense-B_lexical",
            "C_lexical-B_lexical", "C_dense-B_dense", "C_dense-C_lexical",
            "retrieval_x_guardrail_interaction", "D_hybrid-C_dense",
            "D_hybrid-C_lexical", "D_hybrid-A_direct",
        ],
        "primary_continuous_test": "Wilcoxon signed-rank",
        "multiplicity": "Holm within preregistered endpoint families",
        "dataset_pooling": "prohibited",
        "batches": batches,
        "planned_call_count": (
            len(sample_ids) * question_count * len(FOLLOWUP_CONDITION_IDS) * args.repeats
        ),
        "approvals": {
            "panel": bool(args.approve_panel),
            "retrieval_diagnostic_locked": bool(args.confirm_retrieval_diagnostic_locked),
        },
    }
    if payload["planned_call_count"] != 3300 or len(batches) != 55:
        raise RuntimeError("Final design must contain exactly 55 batches and 3,300 calls.")
    output_directory = (
        PROJECT_ROOT / "datasets" / "evaluation" / "followup_retrieval"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / (
        "protocol_locked.json" if locked else "protocol_candidate.json"
    )
    if locked and output.exists():
        raise FileExistsError(f"Refusing to overwrite locked protocol: {output}")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(PROJECT_ROOT)),
        "scientific_status": payload["scientific_status"],
        "sample_count": payload["sample_count"],
        "batch_count": len(batches),
        "planned_call_count": payload["planned_call_count"],
        "sha256": _sha256(output),
    }, indent=2))


if __name__ == "__main__":
    main()
