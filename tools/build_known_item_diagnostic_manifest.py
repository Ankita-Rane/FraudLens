"""Build a non-overwritable known-item diagnostic from the frozen source registry.

The expected documents below are design targets, not human relevance labels.  Each
target is copied verbatim from the registry fields that justify the assignment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "datasets" / "policy_sources" / "policy_source_registry.json"
OUTPUT = PROJECT_ROOT / "datasets" / "evaluation" / "followup_retrieval" / "known_item_diagnostic_locked.json"


# Authored from title/authority/use_for/not_for metadata.  Do not modify after lock;
# create a new versioned manifest if the diagnostic design changes.
QUERY_SPECS = (
    ("DIRECT_RZ12", "direct", "Which source is titled 12 CFR 1026.12 - Special credit card provisions?", ("regulation_z_1026_12.html",)),
    ("PARA_RZ12_A", "paraphrased", "Which registered source provides context on consumer liability for unauthorized credit-card use?", ("regulation_z_1026_12.html",)),
    ("PARA_RZ12_B", "paraphrased", "Find the official regulatory reference intended for unauthorized card-use context, not fraud classification labels.", ("regulation_z_1026_12.html",)),
    ("DIRECT_RZ13", "direct", "Which source is titled 12 CFR 1026.13 - Billing error resolution?", ("regulation_z_1026_13.html",)),
    ("PARA_RZ13_A", "paraphrased", "Which registered source covers billing-error investigation procedure, timing, and evidence context?", ("regulation_z_1026_13.html",)),
    ("PARA_RZ13_B", "paraphrased", "Find the official regulatory reference for resolving a billing dispute rather than setting an institution's alert threshold.", ("regulation_z_1026_13.html",)),
    ("DIRECT_SR", "direct", "Which source is titled SR 26-2 - Revised Guidance on Model Risk Management?", ("federal_reserve_sr_26_2.html",)),
    ("PARA_SR_A", "paraphrased", "Which supervisory source presents model risk management principles and emphasizes a risk-based approach?", ("federal_reserve_sr_26_2.html",)),
    ("PARA_SR_B", "paraphrased", "Find the source for risk-based model risk management rather than transaction-level fraud decisions.", ("federal_reserve_sr_26_2.html",)),
    ("DIRECT_NIST", "direct", "Which source is titled NIST AI 600-1 - Generative AI Profile?", ("nist_ai_600_1.html",)),
    ("PARA_NIST_A", "paraphrased", "Which voluntary framework source covers generative-AI risk management, trustworthiness considerations, and AI measurement and evaluation?", ("nist_ai_600_1.html",)),
    ("DIRECT_BIS", "direct", "Which source is titled FSI Insights 63 - Regulating AI in the financial sector?", ("bis_fsi_insights_63.html",)),
    ("PARA_BIS_A", "paraphrased", "Which international research and supervisory source covers financial-sector AI governance and hallucination or model-risk context?", ("bis_fsi_insights_63.html",)),
    ("MULTI_CONSUMER", "multi_concept", "Find the registered consumer-credit sources for both unauthorized card-use liability and billing-error investigation procedure.", ("regulation_z_1026_12.html", "regulation_z_1026_13.html")),
    ("MULTI_AI_GOV", "multi_concept", "Find the registered sources covering risk-based model risk management, generative-AI trustworthiness and evaluation, and financial-sector AI hallucination risk.", ("federal_reserve_sr_26_2.html", "nist_ai_600_1.html", "bis_fsi_insights_63.html")),
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {row["local_filename"]: row for row in registry["sources"]}
    queries = []
    for query_id, query_type, query, expected_ids in QUERY_SPECS:
        assignments = []
        for document_id in expected_ids:
            source = by_id[document_id]
            assignments.append({
                "document_id": document_id,
                "assignment_rule": "Known item selected only from frozen registry metadata.",
                "registry_evidence": {
                    field: source[field]
                    for field in ("title", "authority", "use_for", "not_for")
                },
            })
        queries.append({
            "query_id": query_id,
            "query_type": query_type,
            "query": query,
            "expected_document_ids": list(expected_ids),
            "expected_document_assignments": assignments,
        })
    query_hash = sha256(json.dumps(queries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload = {
        "schema_version": "1.0",
        "scientific_status": "locked_researcher_authored_known_item_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "human_relevance_judgments_used": False,
        "purpose": "Functional known-item comparison of document ranking under a frozen local corpus.",
        "interpretation_boundary": "Expected documents are researcher-authored design targets derived from registry metadata; they are not an independently adjudicated relevance gold standard and do not prove semantic correctness, entailment, legal correctness, or operational usefulness.",
        "circularity_control": "Expected document assignments are stored in this non-overwritable manifest; the benchmark consumes this hash-bound file and does not infer targets from retrieval results.",
        "residual_limitation": "A file hash proves immutability after creation, not that the researcher had never observed results from earlier exploratory retrieval runs.",
        "source_registry": {
            "path": str(REGISTRY.relative_to(PROJECT_ROOT)),
            "sha256": _hash(REGISTRY),
        },
        "query_set_sha256": query_hash,
        "query_type_counts": {"direct": 5, "paraphrased": 8, "multi_concept": 2},
        "method_references": [
            {"method": "dense sentence embeddings compared with cosine similarity", "citation": "Reimers and Gurevych (2019), Sentence-BERT", "doi": "10.18653/v1/D19-1410", "url": "https://aclanthology.org/D19-1410/"},
            {"method": "reciprocal rank fusion", "citation": "Cormack, Clarke, and Buettcher (2009)", "doi": "10.1145/1571941.1572114", "url": "https://dl.acm.org/doi/10.1145/1571941.1572114"},
            {"method": "discounted cumulative gain evaluation", "citation": "Jarvelin and Kekalainen (2002)", "doi": "10.1145/582415.582418", "url": "https://dl.acm.org/doi/10.1145/582415.582418"},
            {"method": "inverse document frequency", "citation": "Sparck Jones (1972)", "doi": "10.1108/eb026526", "url": "https://doi.org/10.1108/eb026526"},
        ],
        "queries": queries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite locked diagnostic: {OUTPUT}")
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(PROJECT_ROOT)), "sha256": _hash(OUTPUT), "query_set_sha256": query_hash}, indent=2))


if __name__ == "__main__":
    main()
