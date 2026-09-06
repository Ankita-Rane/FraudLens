"""Run the locked, researcher-authored known-item retrieval diagnostic."""

from __future__ import annotations

import argparse
import platform
import resource
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_evaluation import (  # noqa: E402
    RetrievalJudgment,
    LOCKED_KNOWN_ITEM_BASIS,
    evaluate_retrieval,
    macro_average,
)
from src.retrieval_strategies import (  # noqa: E402
    DensePolicyRepository,
    HybridRRFPolicyRepository,
    LexicalPolicyRepository,
    SentenceTransformerTextEncoder,
    retrieval_manifest,
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument(
        "--dense-model-id", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--dense-revision", required=True)
    parser.add_argument("--dense-device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--document-candidate-multiplier", type=int, default=32)
    parser.add_argument(
        "--output", type=Path,
        default=(
            PROJECT_ROOT / "datasets" / "evaluation" / "followup_retrieval"
            / "retrieval_diagnostic_results.json"
        ),
    )
    return parser.parse_args()


def _locked_expectations(packet: dict[str, Any], *, project_root: Path = PROJECT_ROOT) -> list[RetrievalJudgment]:
    if packet.get("scientific_status") != "locked_researcher_authored_known_item_diagnostic":
        raise RuntimeError("Diagnostic manifest is not locked.")
    if packet.get("human_relevance_judgments_used") is not False:
        raise RuntimeError("Manifest must explicitly state that no human relevance judgments are used.")
    registry_info = packet.get("source_registry") or {}
    registry_path = project_root / str(registry_info.get("path", ""))
    if not registry_path.is_file() or _sha256(registry_path) != registry_info.get("sha256"):
        raise RuntimeError("Frozen policy-source registry is missing or its hash does not match.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    by_id = {row["local_filename"]: row for row in registry.get("sources", [])}
    source_root = project_root / "datasets" / "policy_sources" / "source"
    for document_id, metadata in by_id.items():
        source_path = source_root / document_id
        if not source_path.is_file() or _sha256(source_path) != metadata.get("sha256"):
            raise RuntimeError(f"Frozen source integrity failure: {document_id}")
    judgments: list[RetrievalJudgment] = []
    seen: set[str] = set()
    for task in packet.get("queries", []):
        query_id = str(task.get("query_id", "")).strip()
        if not query_id or query_id in seen:
            raise RuntimeError("Every diagnostic query_id must be non-empty and unique.")
        seen.add(query_id)
        relevant = {str(value) for value in task.get("expected_document_ids", [])}
        if not relevant:
            raise RuntimeError(f"Expected document set is empty for query {query_id}.")
        assignments = task.get("expected_document_assignments") or []
        assigned_ids = {str(row.get("document_id")) for row in assignments}
        if assigned_ids != relevant:
            raise RuntimeError(f"Assignment evidence does not match expected documents for {query_id}.")
        for assignment in assignments:
            document_id = str(assignment["document_id"])
            if document_id not in by_id:
                raise RuntimeError(f"Unknown expected document {document_id} for {query_id}.")
            fields = assignment.get("registry_evidence") or {}
            for field in ("title", "authority", "use_for", "not_for"):
                if fields.get(field) != by_id[document_id].get(field):
                    raise RuntimeError(f"Registry evidence mismatch for {query_id}/{document_id}/{field}.")
        judgments.append(RetrievalJudgment(
            query_id=query_id,
            query=str(task["query"]),
            relevant_document_ids=frozenset(relevant),
            expectation_basis=LOCKED_KNOWN_ITEM_BASIS,
        ))
    if not judgments:
        raise RuntimeError("Diagnostic manifest contains no queries.")
    return judgments


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _unique_document_results(repository: Any, query: str, top_k: int, multiplier: int = 8) -> list[Any]:
    rows = repository.search(query, top_k=top_k * multiplier)
    result = []
    seen: set[str] = set()
    for row in rows:
        document_id = row.document_id or row.evidence_id
        if document_id not in seen:
            seen.add(document_id)
            result.append(row)
        if len(result) == top_k:
            break
    return result


def main() -> None:
    args = _arguments()
    if args.top_k < 1:
        raise ValueError("top-k must be positive.")
    packet_path = args.diagnostic_manifest.resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    judgments = _locked_expectations(packet)
    source = PROJECT_ROOT / "datasets" / "policy_sources" / "source"
    lexical = LexicalPolicyRepository(source)
    dense = DensePolicyRepository(
        source,
        encoder=SentenceTransformerTextEncoder(
            model_id=args.dense_model_id,
            model_revision=args.dense_revision,
            local_files_only=args.local_files_only,
            device=args.dense_device,
        ),
        cache_directory=PROJECT_ROOT / "artifacts" / "retrieval_cache",
    )
    hybrid = HybridRRFPolicyRepository(
        lexical=lexical,
        dense=dense,
        rrf_k=args.rrf_k,
        candidate_multiplier=args.candidate_multiplier,
    )
    repositories = {
        "lexical_tfidf": lexical,
        "dense": dense,
        "hybrid_rrf": hybrid,
    }
    strategy_results: dict[str, Any] = {}
    for strategy_id, repository in repositories.items():
        results = []
        for judgment in judgments:
            rss_before = _max_rss_bytes()
            started = time.perf_counter_ns()
            retrieved = _unique_document_results(
                repository, judgment.query, args.top_k,
                multiplier=args.document_candidate_multiplier,
            )
            latency_ns = time.perf_counter_ns() - started
            result = evaluate_retrieval(retrieved, judgment, cutoff_k=args.top_k)
            result.update({
                "latency_ns": latency_ns,
                "max_rss_before_bytes": rss_before,
                "max_rss_after_bytes": _max_rss_bytes(),
                "ranked_document_ids": [row.document_id or row.evidence_id for row in retrieved],
                "ranked_evidence_ids": [row.evidence_id for row in retrieved],
            })
            results.append(result)
        macro = macro_average(results)
        macro["mean_latency_ms"] = round(sum(row["latency_ns"] for row in results) / len(results) / 1_000_000, 6)
        strategy_results[strategy_id] = {
            "manifest": retrieval_manifest(repository),
            "query_results": results,
            "macro_average": macro,
        }
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "completed_scope_safe_known_item_diagnostic",
        "interpretation_boundary": "Researcher-authored functional diagnostic; not an independently adjudicated relevance gold standard.",
        "diagnostic_manifest_path": str(packet_path.relative_to(PROJECT_ROOT)),
        "diagnostic_manifest_sha256": _sha256(packet_path),
        "source_registry_sha256": packet["source_registry"]["sha256"],
        "query_count": len(judgments),
        "top_k": args.top_k,
        "document_candidate_multiplier": args.document_candidate_multiplier,
        "evaluated_strategy_ids": list(repositories),
        "all_strategies_retained_regardless_of_result": True,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "strategies": strategy_results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite retrieval diagnostic: {output}")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(PROJECT_ROOT)),
        "scientific_status": payload["scientific_status"],
        "query_count": payload["query_count"],
        "metrics": {
            key: value["macro_average"] for key, value in strategy_results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
