"""In-memory RAG: ingest documents, retrieve ranked chunks, cite sources.

Pure Python token-overlap scoring. No network, embeddings, or external store.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


@dataclass(frozen=True, slots=True)
class Document:
    """An ingested source document."""

    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable piece of a document."""

    chunk_id: str
    source_id: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with its retrieval score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    """Citation object returned to callers: source id + snippet."""

    source_id: str
    snippet: str
    chunk_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Ingester:
    """Add documents and expose them (and their chunks) for retrieval."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0) -> None:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if (
            not isinstance(chunk_overlap, int)
            or isinstance(chunk_overlap, bool)
            or chunk_overlap < 0
            or chunk_overlap >= chunk_size
        ):
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []

    def add(
        self,
        source_id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Document:
        """Ingest one document. Replaces any previous document with the same id."""
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id is required")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cleaned_id = source_id.strip()
        meta = dict(metadata or {})
        document = Document(source_id=cleaned_id, text=text, metadata=meta)
        # Drop any previous chunks for this source.
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != cleaned_id]
        self._documents[cleaned_id] = document
        for index, piece in enumerate(self._split(text)):
            self._chunks.append(
                Chunk(
                    chunk_id=f"{cleaned_id}:{index}",
                    source_id=cleaned_id,
                    text=piece,
                    index=index,
                    metadata=meta,
                )
            )
        return document

    def remove(self, source_id: str) -> Document:
        """Withdraw one source and its chunks. Returns the removed document."""
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id is required")
        cleaned_id = source_id.strip()
        document = self._documents.pop(cleaned_id, None)
        if document is None:
            raise ValueError(f"unknown source_id: {cleaned_id}")
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != cleaned_id]
        return document

    def find(self, metadata: Mapping[str, Any] | None = None) -> list[Document]:
        """Return documents whose metadata contains every key/value in the filter."""
        documents = list(self._documents.values())
        if metadata is None:
            return documents
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        return [
            document
            for document in documents
            if all(document.metadata.get(key) == value for key, value in metadata.items())
        ]

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            pieces.append(text[start:end])
            if end == len(text):
                break
            start = end - self.chunk_overlap
        return pieces

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(self._documents.values())

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)


class Retriever:
    """Rank ingested chunks by token overlap with a query."""

    def __init__(self, ingester: Ingester) -> None:
        if not isinstance(ingester, Ingester):
            raise TypeError("ingester must be an Ingester")
        self._ingester = ingester

    def query(
        self,
        query: str,
        top_k: int = 3,
        *,
        metadata: Mapping[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        """Return the top-k chunks matching the query, highest score first."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be >= 1")
        if isinstance(min_score, bool) or not isinstance(min_score, int | float):
            raise TypeError("min_score must be a number")
        if not 0.0 <= float(min_score) <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        query_tokens = _tokenize(query)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        scored: list[ScoredChunk] = []
        for chunk in self._ingester.chunks:
            if metadata is not None and not all(
                chunk.metadata.get(key) == value for key, value in metadata.items()
            ):
                continue
            score = _overlap_score(query_tokens, chunk.text)
            # Zero-overlap chunks are never relevant, even when min_score is 0.
            if score > 0.0 and score >= float(min_score):
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def query_with_citations(
        self,
        query: str,
        top_k: int = 3,
        *,
        metadata: Mapping[str, Any] | None = None,
        min_score: float = 0.0,
        max_snippet: int | None = None,
    ) -> list[Citation]:
        """Retrieve and return citation objects (source id + snippet)."""
        hits = self.query(query, top_k, metadata=metadata, min_score=min_score)
        return [cite(hit, max_snippet=max_snippet) for hit in hits]


def cite(scored: ScoredChunk, *, max_snippet: int | None = None) -> Citation:
    """Build a citation from a scored chunk.

    When max_snippet is set, the snippet is truncated to that many characters
    and ends with an ellipsis if the original text was longer.
    """
    if not isinstance(scored, ScoredChunk):
        raise TypeError("scored must be a ScoredChunk")
    text = scored.chunk.text
    if max_snippet is not None:
        if not isinstance(max_snippet, int) or isinstance(max_snippet, bool) or max_snippet < 1:
            raise ValueError("max_snippet must be >= 1")
        if len(text) > max_snippet:
            # Reserve one character for the ellipsis when truncating.
            cut = max(1, max_snippet - 1)
            text = text[:cut] + "\u2026"
    return Citation(
        source_id=scored.chunk.source_id,
        snippet=text,
        chunk_id=scored.chunk.chunk_id,
        score=scored.score,
        metadata=dict(scored.chunk.metadata),
    )


def _overlap_score(query_tokens: frozenset[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / len(query_tokens)
