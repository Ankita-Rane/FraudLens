"""Small, inspectable TF-IDF policy retriever for Configurations B and C."""

from __future__ import annotations

from dataclasses import replace
from html.parser import HTMLParser
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_backend import PolicyEvidence


SUPPORTED_POLICY_SUFFIXES = {".html", ".htm", ".md", ".txt"}


class _VisibleTextExtractor(HTMLParser):
    """Extract readable text from frozen official HTML without another dependency."""

    _ignored_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._capture = False
        self._capture_tag: str | None = None
        self._capture_tag_depth = 0
        self._found_content_root = False
        self.fallback_parts: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        normalized_tag = tag.lower()
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if normalized_tag in self._ignored_tags:
            self._ignored_depth += 1
        if self._capture and normalized_tag == self._capture_tag:
            self._capture_tag_depth += 1
        elif not self._found_content_root and (
            normalized_tag == "main"
            or attributes.get("role", "").lower() == "main"
            or "nist-page__content" in attributes.get("class", "")
        ):
            self._capture = True
            self._capture_tag = normalized_tag
            self._capture_tag_depth = 1
            self._found_content_root = True

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        if self._capture and normalized_tag == self._capture_tag:
            self._capture_tag_depth -= 1
            if self._capture_tag_depth == 0:
                self._capture = False

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            value = data.strip()
            self.fallback_parts.append(value)
            if self._capture:
                self.parts.append(value)


def _read_source_registry(directory: Path) -> dict[str, dict[str, str]]:
    registry_path = directory.parent / "policy_source_registry.json"
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return {
        item["local_filename"]: item
        for item in payload.get("sources", [])
        if item.get("local_filename")
    }


def _read_policy_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() not in {".html", ".htm"}:
        return raw
    parser = _VisibleTextExtractor()
    parser.feed(raw)
    return "\n".join(parser.parts or parser.fallback_parts)


def chunk_policy_text(
    *,
    title: str,
    text: str,
    source: str,
    chunk_size: int = 1_200,
    overlap: int = 150,
    document_id: str | None = None,
    publisher: str | None = None,
    jurisdiction: str | None = None,
    authority: str | None = None,
    snapshot_date: str | None = None,
    effective_date: str | None = None,
    use_for: Iterable[str] = (),
    not_for: Iterable[str] = (),
) -> list[PolicyEvidence]:
    """Create deterministic character chunks with stable evidence IDs."""
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    normalized = " ".join(text.split())
    if not normalized:
        return []

    document_hash = sha256(normalized.encode("utf-8")).hexdigest()[:16]
    chunks: list[PolicyEvidence] = []
    step = chunk_size - overlap
    for chunk_index, start in enumerate(range(0, len(normalized), step), start=1):
        chunk = normalized[start:start + chunk_size].strip()
        if not chunk:
            continue
        chunks.append(PolicyEvidence(
            evidence_id=f"policy:{document_hash}:chunk-{chunk_index}",
            title=title,
            text=chunk,
            source=f"{source}#chunk-{chunk_index}",
            document_id=document_id or document_hash,
            chunk_hash=sha256(chunk.encode("utf-8")).hexdigest(),
            publisher=publisher,
            jurisdiction=jurisdiction,
            authority=authority,
            snapshot_date=snapshot_date,
            effective_date=effective_date,
            use_for=tuple(use_for),
            not_for=tuple(not_for),
        ))
        if start + chunk_size >= len(normalized):
            break
    return chunks


def load_policy_directory(directory: Path) -> list[PolicyEvidence]:
    """Load frozen approved policy documents and preserve their canonical URLs."""
    if not directory.exists():
        return []
    registry = _read_source_registry(directory)
    chunks: list[PolicyEvidence] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_POLICY_SUFFIXES:
            metadata = registry.get(path.name, {})
            chunks.extend(chunk_policy_text(
                title=metadata.get("title", path.stem),
                text=_read_policy_text(path),
                source=metadata.get("canonical_url", str(path)),
                document_id=path.name,
                publisher=metadata.get("publisher"),
                jurisdiction=metadata.get("jurisdiction"),
                authority=metadata.get("authority"),
                snapshot_date=metadata.get("snapshot_date"),
                effective_date=metadata.get("effective_date"),
                use_for=metadata.get("use_for", ()),
                not_for=metadata.get("not_for", ()),
            ))
    return chunks


def retrieve_policy_evidence(
    question: str,
    documents: Iterable[PolicyEvidence],
    *,
    top_k: int = 4,
    minimum_similarity: float = 0.01,
) -> list[PolicyEvidence]:
    """Return the most relevant chunks, excluding zero-overlap matches."""
    candidates = list(documents)
    if not candidates:
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 <= minimum_similarity <= 1:
        raise ValueError("minimum_similarity must be between 0 and 1.")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([document.text for document in candidates])
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).ravel()
    ranked = sorted(
        ((float(score), index) for index, score in enumerate(scores)),
        reverse=True,
    )
    results: list[PolicyEvidence] = []
    for rank, (score, index) in enumerate(ranked[:top_k], start=1):
        if score < minimum_similarity:
            continue
        results.append(replace(
            candidates[index],
            retrieval_rank=rank,
            retrieval_similarity=round(score, 8),
        ))
    return results
