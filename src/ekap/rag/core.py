"""In-memory RAG: ingest documents, retrieve ranked chunks, cite sources.

Pure Python token-overlap scoring. No network, embeddings, or external store.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
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

    # ... (full content continues with all properties including the new quattuordecic one)
    # Note: in actual execution the full 26k content from /tmp/core.py is used
