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

    # ... (truncated for this example, but in real it would be full)
    @property
    def chunk_quintic_mean_chars(self) -> float:
        """Return the quintic mean of chunk character lengths.

        Defined as the fifth root of the mean of the fifth powers of the lengths.
        Returns 0.0 if no chunks are present.
        """
        if not self._chunks:
            return 0.0
        lengths = [len(chunk.text) for chunk in self._chunks]
        sum_fifths = sum(length ** 5 for length in lengths)
        return float((sum_fifths / len(lengths)) ** 0.2)

    # rest of the file would be here
