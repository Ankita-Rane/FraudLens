"""Run the six-condition follow-up retrieval experiment or a non-reportable pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import json
import platform
import random
import subprocess
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.abc_evaluation import aggregate_runs, validate_paired_design  # noqa: E402
from src.abc_service import (  # noqa: E402
    CompositeSampleRepository,
    CsvSampleRepository,
    DeterministicDemoGenerator,
    InvestigationCommand,
    LocalPolicyRepository,
    build_default_service,
)
from src.followup_conditions import (  # noqa: E402
    FOLLOWUP_CONDITION_IDS,
    condition_registry_payload,
    condition_registry_sha256,
    counterbalanced_condition_sequence,
    resolve_condition,
)
from src.llm_provider import LocalQwenProvider, provider_is_available  # noqa: E402
from src.retrieval_strategies import (  # noqa: E402
    DensePolicyRepository,
    HashingSmokeTextEncoder,
    HybridRRFPolicyRepository,
    LexicalPolicyRepository,
    SentenceTransformerTextEncoder,
    retrieval_manifest,
)
from src.statistical_analysis import paired_followup_analysis  # noqa: E402


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return {"commit": commit, "worktree_clean": not bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_clean": False}


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "numpy", "pandas", "scikit-learn", "sentence-transformers", "torch",
        "transformers", "shap", "scipy",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "final"), required=True)
    parser.add_argument("--engine", choices=("demo", "qwen"), default="demo")
    parser.add_argument(
        "--questions", type=Path,
        default=PROJECT_ROOT / "datasets" / "evaluation" / "abc_questions_primary_candidate.json",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--order-seed", type=int, default=20260901)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--model-score-gate", type=float, default=0.50)
    parser.add_argument("--policy-top-k", type=int, default=4)
    parser.add_argument(
        "--dense-backend", choices=("hashing-smoke", "sentence-transformer"),
        default="hashing-smoke",
    )
    parser.add_argument(
        "--dense-model-name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--dense-model-revision", default=None)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rrf-candidate-multiplier", type=int, default=4)
    parser.add_argument("--study-id", default=None)
    parser.add_argument("--protocol-lock", type=Path, default=None)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--confirm-generation-authorized", action="store_true")
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument("--analyst-hourly-cost-usd", type=float, default=0.0)
    parser.add_argument("--estimated-review-minutes", type=float, default=0.0)
    parser.add_argument("--compute-hourly-cost-usd", type=float, default=0.0)
    parser.add_argument(
        "--ulb-attribution-directory", type=Path,
        default=PROJECT_ROOT / "artifacts" / "ulb",
    )
    parser.add_argument(
        "--sparkov-attribution-directory", type=Path,
        default=PROJECT_ROOT / "artifacts" / "sparkov",
    )
    return parser.parse_args()


def _validate_protocol(args: argparse.Namespace, protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if protocol.get("scientific_status") != "locked_followup_protocol":
        errors.append("protocol is not locked_followup_protocol")
    if protocol.get("condition_ids") != list(FOLLOWUP_CONDITION_IDS):
        errors.append("protocol condition IDs differ from the implementation registry")
    if protocol.get("condition_registry_sha256") != condition_registry_sha256():
        errors.append("condition registry hash mismatch")
    diagnostic = protocol.get("retrieval_diagnostic", {})
    if diagnostic.get("scientific_status") != "completed_scope_safe_known_item_diagnostic":
        errors.append("scope-safe retrieval diagnostic is absent or incomplete")
    if not diagnostic.get("audit_sha256"):
        errors.append("verified retrieval diagnostic audit is absent")
    if protocol.get("planned_call_count") != 3300 or len(protocol.get("batches", [])) != 55:
        errors.append("protocol is not the exact 55-batch/3,300-call design")
    if protocol.get("repeats") != 1:
        errors.append("protocol repeat count must be exactly one")
    if args.study_id != protocol.get("study_id"):
        errors.append("--study-id differs from the locked protocol")
    if args.engine != "qwen":
        errors.append("final protocol permits only the Qwen engine")
    if args.model_name != protocol.get("qwen", {}).get("model_id"):
        errors.append("Qwen model ID differs from the protocol")
    if args.dense_backend != "sentence-transformer":
        errors.append("final protocol requires sentence-transformer dense retrieval")
    if not args.model_revision or args.model_revision != protocol.get("qwen", {}).get("revision"):
        errors.append("Qwen revision is absent or differs from the protocol")
    dense = protocol.get("dense_encoder", {})
    if args.dense_model_name != dense.get("model_id"):
        errors.append("dense model ID differs from the protocol")
    if not args.dense_model_revision or args.dense_model_revision != dense.get("revision"):
        errors.append("dense model revision is absent or differs from the protocol")
    if args.policy_top_k != protocol.get("retrieval", {}).get("top_k"):
        errors.append("policy top-k differs from the protocol")
    if args.rrf_k != protocol.get("retrieval", {}).get("rrf_k"):
        errors.append("RRF k differs from the protocol")
    if args.rrf_candidate_multiplier != protocol.get("retrieval", {}).get("candidate_multiplier"):
        errors.append("RRF candidate multiplier differs from the protocol")
    if args.repeats != protocol.get("repeats"):
        errors.append("repeat count differs from the protocol")
    if args.max_new_tokens != protocol.get("qwen", {}).get("max_new_tokens"):
        errors.append("max-new-tokens differs from the protocol")
    if args.model_score_gate != protocol.get("model_score_gate"):
        errors.append("model-score-gate differs from the protocol")
    question_path = args.questions.resolve()
    if _sha256(question_path) != protocol.get("question_manifest_sha256"):
        errors.append("question manifest hash differs from the protocol")
    batches = {row["batch_id"]: row for row in protocol.get("batches", [])}
    if args.batch_id not in batches:
        errors.append("--batch-id is absent or unregistered")
    if not args.confirm_generation_authorized:
        errors.append("explicit generation authorization is absent")
    return errors


def _protocol_batch(protocol: dict[str, Any], batch_id: str | None) -> dict[str, Any]:
    for row in protocol.get("batches", []):
        if row.get("batch_id") == batch_id:
            return row
    raise ValueError(f"Unregistered protocol batch: {batch_id}")


def _load_samples(
    mode: str,
    *,
    ulb_attribution_directory: Path,
    sparkov_attribution_directory: Path,
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    panel = "followup_pilot" if mode == "pilot" else "followup"
    if not ulb_attribution_directory.is_absolute():
        ulb_attribution_directory = PROJECT_ROOT / ulb_attribution_directory
    if not sparkov_attribution_directory.is_absolute():
        sparkov_attribution_directory = PROJECT_ROOT / sparkov_attribution_directory
    ulb_attribution_directory = ulb_attribution_directory.resolve()
    sparkov_attribution_directory = sparkov_attribution_directory.resolve()
    sample_paths = [
        PROJECT_ROOT / "datasets" / key / f"{panel}_transaction_samples.csv"
        for key in ("ulb", "sparkov")
    ]
    attribution_paths = [
        ulb_attribution_directory / f"{panel}_sample_attributions.json",
        sparkov_attribution_directory / f"{panel}_sample_attributions.json",
    ]
    samples = CompositeSampleRepository([
        CsvSampleRepository(path) for path in sample_paths
    ]).list(include_evaluation_labels=False)
    return sample_paths, attribution_paths, samples


def _select_samples(
    samples: list[dict[str, Any]], args: argparse.Namespace, protocol: dict[str, Any] | None
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in samples}
    requested = list(args.sample_id)
    if protocol is not None:
        batch = _protocol_batch(protocol, args.batch_id)
        requested = list(batch["sample_ids"])
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown sample IDs: {unknown}")
        return [by_id[sample_id] for sample_id in requested]
    return samples[:args.max_cases]


def main() -> None:
    args = _arguments()
    if args.repeats < 1 or args.max_cases < 1:
        raise ValueError("repeats and max-cases must be positive.")
    protocol = None
    protocol_path = None
    if args.protocol_lock is not None:
        protocol_path = args.protocol_lock.resolve()
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if args.mode == "final" and protocol is None:
        raise RuntimeError("Final generation requires --protocol-lock.")
    if args.mode == "pilot" and protocol is not None:
        raise RuntimeError("Pilot mode cannot use a final protocol lock.")
    if args.mode == "final":
        errors = _validate_protocol(args, protocol or {})
        if errors:
            raise RuntimeError("Locked follow-up validation failed: " + "; ".join(errors))
        if not _git_state()["worktree_clean"]:
            raise RuntimeError("Final generation requires a clean Git worktree.")
    if args.mode == "pilot" and args.dense_backend != "hashing-smoke" and not args.dense_model_revision:
        raise RuntimeError("A semantic dense pilot must record --dense-model-revision.")

    if args.engine == "qwen":
        if not provider_is_available():
            raise RuntimeError("Local Qwen dependencies are unavailable.")
        generator = LocalQwenProvider(
            model_name=args.model_name,
            revision=args.model_revision,
            local_files_only=args.local_files_only,
        )
    else:
        if args.mode == "final":
            raise RuntimeError("The deterministic demo engine cannot produce final evidence.")
        generator = DeterministicDemoGenerator()

    source_directory = PROJECT_ROOT / "datasets" / "policy_sources" / "source"
    lexical = LexicalPolicyRepository(source_directory)
    encoder = (
        HashingSmokeTextEncoder()
        if args.dense_backend == "hashing-smoke"
        else SentenceTransformerTextEncoder(
            model_id=args.dense_model_name,
            model_revision=args.dense_model_revision,
            local_files_only=args.local_files_only,
            device=args.dense_device,
        )
    )
    dense = DensePolicyRepository(
        source_directory,
        encoder=encoder,
        cache_directory=PROJECT_ROOT / "artifacts" / "retrieval_cache",
    )
    hybrid = HybridRRFPolicyRepository(
        lexical=lexical,
        dense=dense,
        rrf_k=args.rrf_k,
        candidate_multiplier=args.rrf_candidate_multiplier,
    )
    repositories = {
        "lexical_tfidf": lexical,
        "dense": dense,
        "hybrid_rrf": hybrid,
    }
    sample_paths, attribution_paths, all_samples = _load_samples(
        args.mode,
        ulb_attribution_directory=args.ulb_attribution_directory,
        sparkov_attribution_directory=args.sparkov_attribution_directory,
    )
    selected_samples = _select_samples(all_samples, args, protocol)
    questions_path = args.questions.resolve()
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    cell_offset = (
        int(_protocol_batch(protocol, args.batch_id).get("cell_offset", 0))
        if protocol is not None else 0
    )
    experiment_id = str(uuid.uuid4())
    output_directory = (
        PROJECT_ROOT / "artifacts" / "experiments" / "followup" / experiment_id
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    raw_directory = output_directory / "raw_runs"
    service = build_default_service(
        PROJECT_ROOT,
        generator,
        sample_paths=sample_paths,
        attribution_paths=attribution_paths,
        run_directory=raw_directory,
        policy_repositories=repositories,
    )
    started = datetime.now(timezone.utc).isoformat()
    running_manifest = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "study_id": args.study_id,
        "batch_id": args.batch_id,
        "mode": args.mode,
        "scientific_status": "running_not_complete",
        "started_at_utc": started,
        "condition_ids": list(FOLLOWUP_CONDITION_IDS),
        "condition_registry_sha256": condition_registry_sha256(),
        "global_cell_offset": cell_offset,
        "completed_run_count": 0,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(running_manifest, indent=2), encoding="utf-8")
    results: list[dict[str, Any]] = []
    cells = [(sample, question) for sample in selected_samples for question in questions]
    random.Random(args.order_seed).shuffle(cells)
    try:
        for local_cell_index, (sample, question) in enumerate(cells):
            cell_index = cell_offset + local_cell_index
            for repeat in range(1, args.repeats + 1):
                sequence = counterbalanced_condition_sequence(cell_index, repeat)
                for order, condition_id in enumerate(sequence, start=1):
                    condition = resolve_condition(condition_id)
                    result = service.run(InvestigationCommand(
                        configuration=condition.base_configuration,
                        condition_id=condition.condition_id,
                        retrieval_strategy_id=condition.retrieval_strategy,
                        guardrail_profile=condition.guardrail_profile,
                        question=question["text"],
                        sample_id=sample["sample_id"],
                        model_score_gate=args.model_score_gate,
                        policy_top_k=args.policy_top_k,
                        max_new_tokens=args.max_new_tokens,
                        input_cost_per_million=args.input_cost_per_million,
                        output_cost_per_million=args.output_cost_per_million,
                        analyst_hourly_cost_usd=args.analyst_hourly_cost_usd,
                        estimated_review_minutes=args.estimated_review_minutes,
                        compute_hourly_cost_usd=args.compute_hourly_cost_usd,
                        experiment_id=experiment_id,
                        study_id=args.study_id,
                        question_id=question["question_id"],
                        repeat=repeat,
                        run_order=order,
                    ))
                    payload = result.to_dict()
                    payload["question_family"] = question.get("family")
                    results.append(payload)
                    (output_directory / "run_index.partial.json").write_text(json.dumps([
                        {
                            "request_id": row["request_id"],
                            "sample_id": row["sample_id"],
                            "condition_id": row["condition_id"],
                            "question_id": row["trace"]["question_id"],
                            "repeat": row["trace"]["repeat"],
                            "status": row["status"],
                        } for row in results
                    ], indent=2), encoding="utf-8")
    except Exception as error:
        manifest_path.write_text(json.dumps({
            **running_manifest,
            "scientific_status": "failed_incomplete_not_reportable",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_run_count": len(results),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        }, indent=2), encoding="utf-8")
        raise

    manifests = {key: retrieval_manifest(value) for key, value in repositories.items()}
    complete = {
        **running_manifest,
        "scientific_status": (
            "locked_followup_qwen_raw_outputs_pending_analysis"
            if args.mode == "final" else "non_reportable_followup_orchestration_pilot"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_run_count": len(results),
        "case_count": len(selected_samples),
        "question_count": len(questions),
        "repeats": args.repeats,
        "expected_run_count": len(selected_samples) * len(questions) * args.repeats * len(FOLLOWUP_CONDITION_IDS),
        "sample_ids": [row["sample_id"] for row in selected_samples],
        "condition_registry": condition_registry_payload(),
        "counterbalancing": "cyclic six-condition starting-position rotation",
        "global_cell_offset": cell_offset,
        "order_seed": args.order_seed,
        "paired_design": validate_paired_design(
            results, expected_conditions=FOLLOWUP_CONDITION_IDS
        ),
        "automatic_summary": aggregate_runs(results),
        "paired_statistical_analysis": paired_followup_analysis(results),
        "retrieval_manifests": manifests,
        "dense_backend": args.dense_backend,
        "ground_truth_withheld_from_generation": True,
        "git": _git_state(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": _versions(),
        },
        "qwen": {
            "engine": args.engine,
            "model_id": args.model_name if args.engine == "qwen" else None,
            "requested_revision": args.model_revision if args.engine == "qwen" else None,
            "resolved_revision": getattr(generator, "model_revision", None),
            "generation_parameters": getattr(generator, "generation_parameters", {}),
        },
        "inputs": {
            "questions_path": str(questions_path.relative_to(PROJECT_ROOT)),
            "questions_sha256": _sha256(questions_path),
            "sample_panels": {
                str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in sample_paths
            },
            "attributions": {
                str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in attribution_paths
            },
            "protocol_path": (
                str(protocol_path.relative_to(PROJECT_ROOT)) if protocol_path else None
            ),
            "protocol_sha256": _sha256(protocol_path) if protocol_path else None,
        },
    }
    manifest_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    (output_directory / "run_index.json").write_text(
        (output_directory / "run_index.partial.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_directory / "run_index.partial.json").unlink()
    print(json.dumps({
        "experiment_id": experiment_id,
        "output_directory": str(output_directory),
        "scientific_status": complete["scientific_status"],
        "case_count": complete["case_count"],
        "completed_run_count": complete["completed_run_count"],
        "paired": complete["paired_design"]["paired"],
    }, indent=2))


if __name__ == "__main__":
    main()
