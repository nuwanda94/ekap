"""In-memory RAG: ingest, chunk, retrieve, cite. Pure Python, no network."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _matches_metadata(doc_meta: dict[str, Any], filter_meta: dict[str, Any] | None) -> bool:
    if filter_meta is None:
        return True
    return all(doc_meta.get(key) == value for key, value in filter_meta.items())


def _chunk_text(source_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    if not text:
        return [Chunk(chunk_id=f"{source_id}:0", source_id=source_id, index=0, text="")]
    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = max(1, size - overlap)
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}:{index}",
                source_id=source_id,
                index=index,
                text=text[start:end],
            )
        )
        if end == len(text):
            break
        start += step
        index += 1
    return chunks


@dataclass(frozen=True, slots=True)
class Document:
    """An ingested source document."""

    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A contiguous slice of a source document."""

    chunk_id: str
    source_id: str
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk ranked by retrieval score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Citation:
    """A citable snippet with its source and score."""

    source_id: str
    snippet: str
    chunk_id: str
    score: float


def cite(chunk: Chunk, score: float) -> Citation:
    """Build a citation from a scored chunk."""
    return Citation(
        source_id=chunk.source_id,
        snippet=chunk.text,
        chunk_id=chunk.chunk_id,
        score=score,
    )


class Ingester:
    """In-memory document store that chunks on add."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._documents: dict[str, Document] = {}
        self._chunks: list[Chunk] = []

    def add(
        self,
        source_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id is required")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        sid = source_id.strip()
        if sid in self._documents:
            self.remove(sid)
        doc = Document(source_id=sid, text=text, metadata=dict(metadata or {}))
        self._documents[sid] = doc
        self._chunks.extend(
            _chunk_text(sid, text, self._chunk_size, self._chunk_overlap)
        )
        return doc

    def remove(self, source_id: str) -> Document:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id is required")
        sid = source_id.strip()
        if sid not in self._documents:
            raise KeyError(f"unknown source: {sid}")
        doc = self._documents.pop(sid)
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != sid]
        return doc

    def clear(self) -> None:
        self._documents.clear()
        self._chunks.clear()

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, source_id: object) -> bool:
        return isinstance(source_id, str) and source_id in self._documents

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(self._documents.keys())

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(self._documents.values())

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def get(self, source_id: str) -> Document | None:
        if not isinstance(source_id, str):
            return None
        return self._documents.get(source_id.strip())

    def find(self, metadata: dict[str, Any] | None = None) -> list[Document]:
        if metadata is None:
            return list(self._documents.values())
        return [
            doc for doc in self._documents.values() if _matches_metadata(doc.metadata, metadata)
        ]

    @property
    def _lengths(self) -> list[int]:
        return [len(chunk.text) for chunk in self._chunks]

    @property
    def chunk_mean_chars(self) -> float:
        lengths = self._lengths
        return sum(lengths) / len(lengths) if lengths else 0.0

    @property
    def chunk_max_chars(self) -> int:
        lengths = self._lengths
        return max(lengths) if lengths else 0

    @property
    def chunk_min_chars(self) -> int:
        lengths = self._lengths
        return min(lengths) if lengths else 0

    @property
    def chunk_skewness_chars(self) -> float:
        lengths = self._lengths
        n = len(lengths)
        if n < 3:
            return 0.0
        mean = sum(lengths) / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        if variance == 0.0:
            return 0.0
        std = math.sqrt(variance)
        m3 = sum((x - mean) ** 3 for x in lengths) / n
        return m3 / (std ** 3)

    @property
    def chunk_kurtosis_chars(self) -> float:
        lengths = self._lengths
        n = len(lengths)
        if n < 4:
            return 0.0
        mean = sum(lengths) / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        if variance == 0.0:
            return 0.0
        m4 = sum((x - mean) ** 4 for x in lengths) / n
        return m4 / (variance ** 2)

    @property
    def chunk_excess_kurtosis_chars(self) -> float:
        lengths = self._lengths
        n = len(lengths)
        if n < 4:
            return 0.0
        mean = sum(lengths) / n
        variance = sum((x - mean) ** 2 for x in lengths) / n
        if variance == 0.0:
            return 0.0
        m4 = sum((x - mean) ** 4 for x in lengths) / n
        return m4 / (variance ** 2) - 3.0

    def _power_mean(self, power: int) -> float:
        lengths = self._lengths
        if not lengths:
            return 0.0
        total = sum(float(length) ** power for length in lengths) / len(lengths)
        return total ** (1.0 / power)

    @property
    def chunk_geometric_mean_chars(self) -> float:
        lengths = self._lengths
        if not lengths or any(length <= 0 for length in lengths):
            return 0.0
        return math.exp(sum(math.log(length) for length in lengths) / len(lengths))

    @property
    def chunk_harmonic_mean_chars(self) -> float:
        lengths = self._lengths
        if not lengths or any(length <= 0 for length in lengths):
            return 0.0
        return len(lengths) / sum(1.0 / length for length in lengths)

    @property
    def chunk_quadratic_mean_chars(self) -> float:
        return self._power_mean(2)

    @property
    def chunk_cubic_mean_chars(self) -> float:
        return self._power_mean(3)

    @property
    def chunk_quartic_mean_chars(self) -> float:
        return self._power_mean(4)

    @property
    def chunk_quintic_mean_chars(self) -> float:
        return self._power_mean(5)

    @property
    def chunk_sextic_mean_chars(self) -> float:
        return self._power_mean(6)

    @property
    def chunk_septic_mean_chars(self) -> float:
        return self._power_mean(7)

    @property
    def chunk_octic_mean_chars(self) -> float:
        return self._power_mean(8)

    @property
    def chunk_nonic_mean_chars(self) -> float:
        return self._power_mean(9)

    @property
    def chunk_decic_mean_chars(self) -> float:
        return self._power_mean(10)

    @property
    def chunk_undecic_mean_chars(self) -> float:
        return self._power_mean(11)

    @property
    def chunk_duodecic_mean_chars(self) -> float:
        return self._power_mean(12)

    @property
    def chunk_tredecic_mean_chars(self) -> float:
        return self._power_mean(13)

    @property
    def chunk_quattuordecic_mean_chars(self) -> float:
        return self._power_mean(14)

    @property
    def chunk_quindecic_mean_chars(self) -> float:
        return self._power_mean(15)

    @property
    def chunk_mean_tokens(self) -> float:
        counts = [len(_tokens(chunk.text)) for chunk in self._chunks]
        return sum(counts) / len(counts) if counts else 0.0

    @property
    def chunk_length_histogram(self) -> dict[int, int]:
        return dict(Counter(self._lengths))


class Retriever:
    """Token-overlap retriever over an Ingester."""

    def __init__(self, ingester: Ingester) -> None:
        if not isinstance(ingester, Ingester):
            raise TypeError("ingester must be an Ingester")
        self._ingester = ingester

    def query_with_citations(
        self,
        query: str,
        top_k: int = 3,
        metadata: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[Citation]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be >= 1")
        if (
            not isinstance(min_score, int | float)
            or isinstance(min_score, bool)
            or not 0.0 <= float(min_score) <= 1.0
        ):
            raise ValueError("min_score must be between 0 and 1")
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []
        scored: list[tuple[float, Chunk]] = []
        for chunk in self._ingester.chunks:
            document = self._ingester.get(chunk.source_id)
            if document is None or not _matches_metadata(document.metadata, metadata):
                continue
            chunk_tokens = set(_tokens(chunk.text))
            overlap = len(query_tokens & chunk_tokens)
            score = overlap / len(query_tokens)
            if score >= float(min_score):
                scored.append((score, chunk))
        scored.sort(key=lambda item: -item[0])
        return [cite(chunk, score) for score, chunk in scored[:top_k]]
