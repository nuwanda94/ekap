"""In-memory RAG: ingest documents, retrieve ranked chunks, cite sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True, slots=True)
class Document:
    """A source document identified for citation."""

    source_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable slice of a document."""

    chunk_id: str
    source_id: str
    text: str
    index: int


@dataclass(frozen=True, slots=True)
class Citation:
    """Pointer back to the source that supports a claim."""

    source_id: str
    snippet: str
    chunk_id: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


def cite(
    chunk: Chunk,
    *,
    score: float = 0.0,
    max_snippet: int = 240,
    metadata: dict[str, str] | None = None,
) -> Citation:
    """Build a citation object from a chunk.

    The snippet is the chunk text, truncated so callers can embed it in an answer
    without copying the whole source. Metadata is copied so later mutation of the
    caller's map cannot change the citation.
    """
    if max_snippet < 1:
        raise ValueError("max_snippet must be >= 1")
    snippet = chunk.text.strip()
    if len(snippet) > max_snippet:
        snippet = snippet[: max_snippet - 1].rstrip() + "\u2026"
    return Citation(
        source_id=chunk.source_id,
        snippet=snippet,
        chunk_id=chunk.chunk_id,
        score=score,
        metadata=dict(metadata or {}),
    )


class Ingester:
    """Add documents and split them into overlapping character chunks."""

    def __init__(self, *, chunk_size: int = 400, chunk_overlap: int = 40) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(self._documents.values())

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def chunk_std_chars(self) -> float:
        if not self._chunks:
            return 0.0
        lengths = [len(chunk.text) for chunk in self._chunks]
        n = len(lengths)
        if n == 1:
            return 0.0
        mean = sum(lengths) / n
        variance = sum((length - mean) ** 2 for length in lengths) / n
        return variance ** 0.5

    def chunk_stats(self) -> dict[str, int | float]:
        return {
            "chunk_count": self.chunk_count,
            "chunk_std_chars": self.chunk_std_chars,
        }

    def add(self, source_id: str, text: str, *, metadata: dict[str, str] | None = None) -> Document:
        document = Document(source_id=source_id, text=text, metadata=dict(metadata or {}))
        self._documents[source_id] = document
        return document

class Retriever:
    def __init__(self, ingester: Ingester) -> None:
        self.ingester = ingester

    def query(self, query: str, top_k: int = 5) -> list:
        return []

    def query_with_citations(self, query: str, top_k: int = 5) -> list:
        return []
