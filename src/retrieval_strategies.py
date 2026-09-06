"""Versioned lexical, dense and hybrid policy retrieval strategies.

Dense model loading is lazy.  Locked runs must use ``SentenceTransformerTextEncoder``
with an immutable model revision.  ``HashingSmokeTextEncoder`` exists solely so unit
tests and non-reportable orchestration pilots can run without downloading a model.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, Sequence

import json
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from src.llm_backend import PolicyEvidence
from src.retrieval import load_policy_directory, retrieve_policy_evidence


class TextEncoder(Protocol):
    model_id: str
    model_revision: str | None
    scientific_status: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Encoder output must be a two-dimensional matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms == 0.0, 1.0, norms)


def _documents_sha256(documents: Sequence[PolicyEvidence]) -> str:
    payload = [
        {
            "evidence_id": document.evidence_id,
            "chunk_hash": document.chunk_hash,
            "text_sha256": sha256(document.text.encode("utf-8")).hexdigest(),
        }
        for document in documents
    ]
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class SentenceTransformerTextEncoder:
    scientific_status = "pinned_dense_semantic_encoder"

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str | None,
        local_files_only: bool = False,
        device: str | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("Dense model_id must be non-empty.")
        self.model_id = model_id
        self.model_revision = model_revision
        self.local_files_only = bool(local_files_only)
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Dense retrieval requires sentence-transformers; install the "
                    "pinned follow-up requirements before running."
                ) from error
            self._model = SentenceTransformer(
                self.model_id,
                revision=self.model_revision,
                local_files_only=self.local_files_only,
                device=self.device,
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        values = self._load().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _normalized(np.asarray(values, dtype=np.float32))


class HashingSmokeTextEncoder:
    """Deterministic non-semantic encoder for tests and non-reportable pilots only."""

    model_id = "hashing-char-ngram-smoke-only"
    model_revision = "1"
    scientific_status = "non_semantic_smoke_only_not_reportable"

    def __init__(self, n_features: int = 512) -> None:
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=n_features,
            alternate_sign=False,
            norm=None,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return _normalized(self._vectorizer.transform(list(texts)).toarray())


class LexicalPolicyRepository:
    strategy_id = "lexical_tfidf"
    scientific_status = "frozen_lexical_baseline"

    def __init__(self, source_directory: Path) -> None:
        self._documents = tuple(load_policy_directory(source_directory))
        if not self._documents:
            raise RuntimeError("The approved policy corpus is empty.")

    @property
    def corpus_sha256(self) -> str:
        return _documents_sha256(self._documents)

    def search(self, question: str, *, top_k: int) -> list[PolicyEvidence]:
        results = retrieve_policy_evidence(question, self._documents, top_k=top_k)
        return [
            replace(
                result,
                retrieval_strategy=self.strategy_id,
                lexical_rank=result.retrieval_rank,
                lexical_similarity=result.retrieval_similarity,
            )
            for result in results
        ]


class DensePolicyRepository:
    strategy_id = "dense"

    def __init__(
        self,
        source_directory: Path,
        *,
        encoder: TextEncoder,
        cache_directory: Path | None = None,
    ) -> None:
        self._documents = tuple(load_policy_directory(source_directory))
        if not self._documents:
            raise RuntimeError("The approved policy corpus is empty.")
        self.encoder = encoder
        self.scientific_status = encoder.scientific_status
        self._corpus_sha256 = _documents_sha256(self._documents)
        self._cache_directory = cache_directory
        self._embeddings = self._load_or_encode_documents()

    @property
    def corpus_sha256(self) -> str:
        return self._corpus_sha256

    def _cache_key(self) -> str:
        payload = {
            "encoder": self.encoder.model_id,
            "revision": self.encoder.model_revision,
            "corpus_sha256": self._corpus_sha256,
            "normalization": "l2",
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _load_or_encode_documents(self) -> np.ndarray:
        cache_path = (
            self._cache_directory / f"{self._cache_key()}.npz"
            if self._cache_directory is not None else None
        )
        if cache_path is not None and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as payload:
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if embeddings.shape[0] != len(self._documents):
                raise ValueError("Dense embedding cache row count does not match corpus.")
            return _normalized(embeddings)
        embeddings = self.encoder.encode([document.text for document in self._documents])
        if embeddings.shape[0] != len(self._documents):
            raise ValueError("Dense encoder returned the wrong number of document rows.")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, embeddings=embeddings)
            metadata_path = cache_path.with_suffix(".json")
            metadata_path.write_text(json.dumps({
                "schema_version": "1.0",
                "cache_key": self._cache_key(),
                "encoder_model_id": self.encoder.model_id,
                "encoder_model_revision": self.encoder.model_revision,
                "encoder_scientific_status": self.encoder.scientific_status,
                "corpus_sha256": self._corpus_sha256,
                "document_count": len(self._documents),
            }, indent=2), encoding="utf-8")
        return _normalized(embeddings)

    def search(self, question: str, *, top_k: int) -> list[PolicyEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query = self.encoder.encode([question])
        if query.shape[0] != 1:
            raise ValueError("Dense encoder did not return exactly one query embedding.")
        scores = (query @ self._embeddings.T).ravel()
        ranked = sorted(
            ((float(score), index) for index, score in enumerate(scores)),
            key=lambda item: (-item[0], self._documents[item[1]].evidence_id),
        )
        return [
            replace(
                self._documents[index],
                retrieval_rank=rank,
                retrieval_similarity=round(score, 8),
                retrieval_strategy=self.strategy_id,
                dense_rank=rank,
                dense_similarity=round(score, 8),
            )
            for rank, (score, index) in enumerate(ranked[:top_k], start=1)
        ]


class HybridRRFPolicyRepository:
    strategy_id = "hybrid_rrf"

    def __init__(
        self,
        *,
        lexical: LexicalPolicyRepository,
        dense: DensePolicyRepository,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if rrf_k < 1 or candidate_multiplier < 1:
            raise ValueError("rrf_k and candidate_multiplier must be positive.")
        if lexical.corpus_sha256 != dense.corpus_sha256:
            raise ValueError("Hybrid components must use the identical policy corpus.")
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.scientific_status = dense.scientific_status
        self.corpus_sha256 = dense.corpus_sha256

    def search(self, question: str, *, top_k: int) -> list[PolicyEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        candidate_k = top_k * self.candidate_multiplier
        lexical = self.lexical.search(question, top_k=candidate_k)
        dense = self.dense.search(question, top_k=candidate_k)
        by_id: dict[str, PolicyEvidence] = {}
        component: dict[str, dict[str, float | int]] = {}
        for document in lexical:
            by_id[document.evidence_id] = document
            component.setdefault(document.evidence_id, {}).update({
                "lexical_rank": int(document.retrieval_rank or 0),
                "lexical_similarity": float(document.retrieval_similarity or 0.0),
            })
        for document in dense:
            by_id.setdefault(document.evidence_id, document)
            component.setdefault(document.evidence_id, {}).update({
                "dense_rank": int(document.retrieval_rank or 0),
                "dense_similarity": float(document.retrieval_similarity or 0.0),
            })
        fused: list[tuple[float, str]] = []
        for evidence_id, values in component.items():
            score = sum(
                1.0 / (self.rrf_k + int(values[key]))
                for key in ("lexical_rank", "dense_rank") if key in values
            )
            fused.append((score, evidence_id))
        fused.sort(key=lambda item: (-item[0], item[1]))
        results: list[PolicyEvidence] = []
        for rank, (score, evidence_id) in enumerate(fused[:top_k], start=1):
            values = component[evidence_id]
            results.append(replace(
                by_id[evidence_id],
                retrieval_rank=rank,
                retrieval_similarity=round(score, 10),
                retrieval_strategy=self.strategy_id,
                lexical_rank=int(values["lexical_rank"]) if "lexical_rank" in values else None,
                lexical_similarity=(
                    round(float(values["lexical_similarity"]), 8)
                    if "lexical_similarity" in values else None
                ),
                dense_rank=int(values["dense_rank"]) if "dense_rank" in values else None,
                dense_similarity=(
                    round(float(values["dense_similarity"]), 8)
                    if "dense_similarity" in values else None
                ),
                rrf_score=round(score, 10),
                rrf_k=self.rrf_k,
            ))
        return results


def retrieval_manifest(repository: Any) -> dict[str, Any]:
    payload = {
        "strategy_id": repository.strategy_id,
        "scientific_status": repository.scientific_status,
        "corpus_sha256": repository.corpus_sha256,
    }
    if isinstance(repository, DensePolicyRepository):
        payload.update({
            "encoder_model_id": repository.encoder.model_id,
            "encoder_model_revision": repository.encoder.model_revision,
        })
    if isinstance(repository, HybridRRFPolicyRepository):
        payload.update({
            "encoder_model_id": repository.dense.encoder.model_id,
            "encoder_model_revision": repository.dense.encoder.model_revision,
            "rrf_k": repository.rrf_k,
            "candidate_multiplier": repository.candidate_multiplier,
        })
    return payload
