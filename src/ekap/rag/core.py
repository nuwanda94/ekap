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

    def add(
        self,
        source_id: str,
        text: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Document:
        """Ingest one document, replacing any previous document with the same id."""
        cleaned_id = _require_source_id(source_id)
        if not text or not text.strip():
            raise ValueError("text is required")
        document = Document(source_id=cleaned_id, text=text, metadata=dict(metadata or {}))
        self._documents[cleaned_id] = document
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != cleaned_id]
        self._chunks.extend(self._split(document))
        return document

    def add_documents(self, documents: list[Document]) -> list[Document]:
        return [self.add(doc.source_id, doc.text, metadata=doc.metadata) for doc in documents]

    def remove(self, source_id: str) -> Document:
        """Withdraw a source so its chunks are no longer retrieved."""
        cleaned_id = _require_source_id(source_id)
        if cleaned_id not in self._documents:
            raise KeyError(f"unknown source: {cleaned_id}")
        document = self._documents.pop(cleaned_id)
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != cleaned_id]
        return document

    def find(self, *, metadata: dict[str, str] | None = None) -> tuple[Document, ...]:
        """Return ingested documents whose metadata matches every filter pair.

        An empty filter returns every source, in insertion order. Each result is a
        copy, so mutating the returned metadata cannot change the stored corpus.
        """
        required = _normalize_metadata_filter(metadata)
        return tuple(
            _copy_document(document)
            for document in self._documents.values()
            if not required or _metadata_matches(document, required)
        )

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


def _copy_document(document: Document) -> Document:
    return Document(
        source_id=document.source_id,
        text=document.text,
        metadata=dict(document.metadata),
    )


def _require_source_id(source_id: str) -> str:
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id is required")
    return source_id.strip()


def _normalize_metadata_filter(metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata filter must be a dict of strings")
    cleaned: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata keys must be non-empty strings")
        if not isinstance(value, str):
            raise TypeError("metadata values must be strings")
        cleaned[key.strip()] = value
    return cleaned


def _metadata_matches(document: Document | None, required: dict[str, str]) -> bool:
    if document is None:
        return False
    return all(document.metadata.get(key) == value for key, value in required.items())


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever:
    """Rank ingested chunks by token overlap with the query. No network calls."""

    def __init__(self, ingester: Ingester) -> None:
        self._ingester = ingester

    def query(
        self,
        text: str,
        *,
        top_k: int = 3,
        metadata: dict[str, str] | None = None,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        """Rank chunks. When metadata is set, every pair must match the source.

        Chunks whose token-overlap ratio is below min_score are omitted. The
        corpus is not changed. A non-numeric score or a value outside 0 to 1
        fails closed.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        threshold = _require_min_score(min_score)
        required = _normalize_metadata_filter(metadata)
        query_tokens = _tokens(text)
        if not query_tokens:
            return []
        documents = {document.source_id: document for document in self._ingester.documents}
        scored: list[ScoredChunk] = []
        for chunk in self._ingester.chunks:
            if required and not _metadata_matches(documents.get(chunk.source_id), required):
                continue
            chunk_tokens = set(_tokens(chunk.text))
            if not chunk_tokens:
                continue
            overlap = sum(1 for token in query_tokens if token in chunk_tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens)
            if score < threshold:
                continue
            scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.source_id, hit.chunk.index))
        return scored[:top_k]

    def query_with_citations(
        self,
        text: str,
        *,
        top_k: int = 3,
        metadata: dict[str, str] | None = None,
        min_score: float = 0.0,
    ) -> list[Citation]:
        documents = {document.source_id: document for document in self._ingester.documents}
        return [
            cite(
                hit.chunk,
                score=hit.score,
                metadata=_source_metadata(documents.get(hit.chunk.source_id)),
            )
            for hit in self.query(
                text,
                top_k=top_k,
                metadata=metadata,
                min_score=min_score,
            )
        ]


def _require_min_score(min_score: float) -> float:
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise TypeError("min_score must be a number")
    if min_score < 0 or min_score > 1:
        raise ValueError("min_score must be between 0 and 1")
    return float(min_score)


def _source_metadata(document: Document | None) -> dict[str, str]:
    if document is None:
        return {}
    return document.metadata
