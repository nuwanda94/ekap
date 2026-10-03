"""In-memory RAG: ingest documents, retrieve ranked chunks, cite sources."""

from __future__ import annotations

import re
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


def cite(chunk: Chunk, *, score: float = 0.0, max_snippet: int = 240) -> Citation:
    """Build a citation object from a chunk.

    The snippet is the chunk text, truncated so callers can embed it in an answer
    without copying the whole source.
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

    def add(
        self,
        source_id: str,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Document:
        """Ingest one document, replacing any previous document with the same id."""
        if not source_id or not source_id.strip():
            raise ValueError("source_id is required")
        if not text or not text.strip():
            raise ValueError("text is required")
        document = Document(source_id=source_id, text=text, metadata=dict(metadata or {}))
        self._documents[source_id] = document
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != source_id]
        self._chunks.extend(self._split(document))
        return document

    def add_documents(self, documents: list[Document]) -> list[Document]:
        return [self.add(doc.source_id, doc.text, metadata=doc.metadata) for doc in documents]

    def _split(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        size = self.chunk_size
        if len(text) <= size:
            pieces = [text]
        else:
            step = size - self.chunk_overlap
            pieces = []
            start = 0
            while start < len(text):
                pieces.append(text[start : start + size])
                if start + size >= len(text):
                    break
                start += step
        return [
            Chunk(
                chunk_id=f"{document.source_id}:{index}",
                source_id=document.source_id,
                text=piece,
                index=index,
            )
            for index, piece in enumerate(pieces)
        ]


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever:
    """Rank ingested chunks by token overlap with the query. No network calls."""

    def __init__(self, ingester: Ingester) -> None:
        self._ingester = ingester

    def query(self, text: str, *, top_k: int = 3) -> list[ScoredChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_tokens = _tokens(text)
        if not query_tokens:
            return []
        scored: list[ScoredChunk] = []
        for chunk in self._ingester.chunks:
            chunk_tokens = set(_tokens(chunk.text))
            if not chunk_tokens:
                continue
            overlap = sum(1 for token in query_tokens if token in chunk_tokens)
            if overlap == 0:
                continue
            scored.append(ScoredChunk(chunk=chunk, score=overlap / len(query_tokens)))
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.source_id, hit.chunk.index))
        return scored[:top_k]

    def query_with_citations(self, text: str, *, top_k: int = 3) -> list[Citation]:
        return [cite(hit.chunk, score=hit.score) for hit in self.query(text, top_k=top_k)]
