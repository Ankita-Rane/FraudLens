"""Recompute and hash-audit the scope-safe retrieval diagnostic evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_strategies import LexicalPolicyRepository  # noqa: E402


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _metrics(ranked: list[str], expected: set[str], cutoff_k: int) -> dict[str, float | bool]:
    hits = [int(value in expected) for value in ranked]
    first = next((index for index, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_count = min(len(expected), cutoff_k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "precision_at_k": round(sum(hits) / cutoff_k, 6),
        "recall_at_k": round(sum(hits) / len(expected), 6),
        "reciprocal_rank": round(1 / first, 6) if first else 0.0,
        "ndcg_at_k": round(dcg / ideal, 6) if ideal else 0.0,
        "top_1_success": bool(hits and hits[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    results_path = args.results.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    registry_path = PROJECT_ROOT / manifest["source_registry"]["path"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    source_root = PROJECT_ROOT / "datasets" / "policy_sources" / "source"
    current_corpus_hash = LexicalPolicyRepository(source_root).corpus_sha256
    checks: dict[str, bool] = {
        "manifest_hash_matches_results": results.get("diagnostic_manifest_sha256") == _sha(manifest_path),
        "registry_hash_matches_results": results.get("source_registry_sha256") == manifest.get("source_registry", {}).get("sha256"),
        "status_is_scope_safe_completed": results.get("scientific_status") == "completed_scope_safe_known_item_diagnostic",
        "no_human_relevance_judgments": manifest.get("human_relevance_judgments_used") is False,
        "all_strategies_retained": results.get("all_strategies_retained_regardless_of_result") is True,
        "exact_strategy_set": set(results.get("strategies", {})) == {"lexical_tfidf", "dense", "hybrid_rrf"},
        "current_registry_hash_matches_manifest": _sha(registry_path) == manifest["source_registry"]["sha256"],
        "every_current_source_hash_matches_registry": all(
            (source_root / row["local_filename"]).is_file()
            and _sha(source_root / row["local_filename"]) == row["sha256"]
            for row in registry.get("sources", [])
        ),
        "current_corpus_hash_matches_results": all(
            strategy.get("manifest", {}).get("corpus_sha256") == current_corpus_hash
            for strategy in results.get("strategies", {}).values()
        ),
    }
    queries = {row["query_id"]: row for row in manifest.get("queries", [])}
    recomputed = 0
    bounded = True
    complete = True
    corpus_hashes = set()
    for strategy in results.get("strategies", {}).values():
        corpus_hashes.add(strategy.get("manifest", {}).get("corpus_sha256"))
        rows = strategy.get("query_results", [])
        complete &= {row.get("query_id") for row in rows} == set(queries)
        for row in rows:
            query = queries[row["query_id"]]
            ranked = row.get("ranked_document_ids", [])
            expected = set(query["expected_document_ids"])
            calculated = _metrics(ranked, expected, int(results["top_k"]))
            for key, value in calculated.items():
                if row.get(key) != value:
                    raise RuntimeError(f"Metric mismatch: {row['query_id']} {key}: {row.get(key)} != {value}")
            bounded &= all(0.0 <= float(calculated[key]) <= 1.0 for key in ("precision_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k"))
            recomputed += 1
    checks.update({
        "all_query_strategy_cells_present": complete,
        "all_metrics_recomputed_exactly": recomputed == len(queries) * 3,
        "all_rank_metrics_bounded_0_1": bounded,
        "identical_corpus_across_strategies": len(corpus_hashes) == 1,
        "pinned_dense_revision_recorded": bool(results.get("strategies", {}).get("dense", {}).get("manifest", {}).get("encoder_model_revision")),
        "rankings_and_evidence_ids_recorded": all(
            row.get("ranked_document_ids") is not None and row.get("ranked_evidence_ids") is not None
            for strategy in results.get("strategies", {}).values()
            for row in strategy.get("query_results", [])
        ),
        "latency_and_memory_observations_recorded": all(
            isinstance(row.get("latency_ns"), int)
            and isinstance(row.get("max_rss_before_bytes"), int)
            and isinstance(row.get("max_rss_after_bytes"), int)
            for strategy in results.get("strategies", {}).values()
            for row in strategy.get("query_results", [])
        ),
    })
    if not all(checks.values()):
        raise RuntimeError(f"Retrieval diagnostic audit failed: {[key for key, value in checks.items() if not value]}")
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "verified_scope_safe_retrieval_diagnostic_audit",
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": _sha(manifest_path),
        "results_path": str(results_path.relative_to(PROJECT_ROOT)),
        "results_sha256": _sha(results_path),
        "recomputed_query_strategy_cells": recomputed,
        "checks": checks,
        "all_checks_passed": True,
        "claim_boundary": manifest["interpretation_boundary"],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic audit: {output}")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(PROJECT_ROOT)), "sha256": _sha(output), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
