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
        """Return the number of chunks currently in the corpus.

        This is the total across all sources. The value changes when documents
        are added, updated, removed, or cleared. It is independent of document
        count when chunking produces multiple slices.
        """
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        """Return the number of documents currently in the corpus.

        This matches ``len(ingester)``. The value changes when documents are
        added, removed, or cleared. It is independent of chunk count.
        """
        return len(self._documents)

    @property
    def total_chars(self) -> int:
        """Return the total number of characters across all document texts.

        This is the sum of ``len(document.text)`` for every ingested document.
        The value changes when documents are added, updated (text replacement),
        removed, or cleared. It is independent of chunk count and of any
        whitespace that chunking may strip.
        """
        return sum(len(document.text) for document in self._documents.values())

    @property
    def max_chars(self) -> int:
        """Return the length of the longest document text, or 0 if empty.

        This is the maximum of ``len(document.text)`` across ingested documents.
        The value changes when documents are added, updated (text replacement),
        removed, or cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0
        return max(len(document.text) for document in self._documents.values())

    @property
    def min_chars(self) -> int:
        """Return the length of the shortest document text, or 0 if empty.

        This is the minimum of ``len(document.text)`` across ingested documents.
        The value changes when documents are added, updated (text replacement),
        removed, or cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0
        return min(len(document.text) for document in self._documents.values())

    @property
    def avg_chars(self) -> float:
        """Return the average length of document texts, or 0.0 if empty.

        This is the mean of ``len(document.text)`` across ingested documents.
        The value changes when documents are added, updated (text replacement),
        removed, or cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0.0
        total = sum(len(document.text) for document in self._documents.values())
        return total / len(self._documents)

    @property
    def median_chars(self) -> float:
        """Return the median length of document texts, or 0.0 if empty.

        This is the median of ``len(document.text)`` across ingested documents.
        For an even number of documents it is the average of the two central
        values. The value changes when documents are added, updated (text
        replacement), removed, or cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0.0
        lengths = sorted(len(document.text) for document in self._documents.values())
        n = len(lengths)
        mid = n // 2
        if n % 2 == 1:
            return float(lengths[mid])
        return (lengths[mid - 1] + lengths[mid]) / 2.0

    @property
    def std_chars(self) -> float:
        """Return the population standard deviation of document text lengths, or 0.0 if empty.

        This is the square root of the mean squared deviation of ``len(document.text)``
        from the mean length. For a single document it is 0.0. The value changes when
        documents are added, updated (text replacement), removed, or cleared. It is
        independent of chunk count.
        """
        if not self._documents:
            return 0.0
        lengths = [len(document.text) for document in self._documents.values()]
        n = len(lengths)
        if n == 1:
            return 0.0
        mean = sum(lengths) / n
        variance = sum((length - mean) ** 2 for length in lengths) / n
        return variance ** 0.5

    @property
    def var_chars(self) -> float:
        """Return the population variance of document text lengths, or 0.0 if empty.

        This is the mean squared deviation of ``len(document.text)`` from the mean
        length. For a single document it is 0.0. The value changes when documents
        are added, updated (text replacement), removed, or cleared. It is
        independent of chunk count. ``std_chars`` is the square root of this value.
        """
        if not self._documents:
            return 0.0
        lengths = [len(document.text) for document in self._documents.values()]
        n = len(lengths)
        if n == 1:
            return 0.0
        mean = sum(lengths) / n
        return sum((length - mean) ** 2 for length in lengths) / n

    @property
    def range_chars(self) -> int:
        """Return the difference between the longest and shortest document lengths, or 0 if empty.

        This is ``max_chars - min_chars``. For a single document it is 0. The value
        changes when documents are added, updated (text replacement), removed, or
        cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0
        return self.max_chars - self.min_chars

    @property
    def mode_chars(self) -> int:
        """Return the most frequent document text length, or 0 if empty.

        When multiple lengths share the highest frequency, the smallest such length
        is returned. For a single document it is that document's length. The value
        changes when documents are added, updated (text replacement), removed, or
        cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0
        lengths = [len(document.text) for document in self._documents.values()]
        counts = Counter(lengths)
        max_count = max(counts.values())
        modes = [length for length, count in counts.items() if count == max_count]
        return min(modes)

    @property
    def nunique_chars(self) -> int:
        """Return the number of distinct document text lengths, or 0 if empty.

        This is the cardinality of the set of ``len(document.text)`` values.
        For a single document it is 1. The value changes when documents are added,
        updated (text replacement), removed, or cleared. It is independent of
        chunk count.
        """
        if not self._documents:
            return 0
        lengths = {len(document.text) for document in self._documents.values()}
        return len(lengths)

    @property
    def iqr_chars(self) -> float:
        """Return the interquartile range of document text lengths, or 0.0 if empty.

        This is the difference between the 75th and 25th percentiles of
        ``len(document.text)``, computed with linear interpolation. For fewer
        than two documents it is 0.0. The value changes when documents are added,
        updated (text replacement), removed, or cleared. It is independent of
        chunk count.
        """
        if not self._documents:
            return 0.0
        lengths = sorted(len(document.text) for document in self._documents.values())
        n = len(lengths)
        if n < 2:
            return 0.0

        def _percentile(p: float) -> float:
            idx = p * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return lengths[lo] * (1 - frac) + lengths[hi] * frac

        return _percentile(0.75) - _percentile(0.25)

    @property
    def p90_chars(self) -> float:
        """Return the 90th percentile of document text lengths, or 0.0 if empty.

        This is the 90th percentile of ``len(document.text)``, computed with linear
        interpolation. For a single document it is the length itself. The value
        changes when documents are added, updated (text replacement), removed, or
        cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0.0
        lengths = sorted(len(document.text) for document in self._documents.values())
        n = len(lengths)
        if n == 1:
            return float(lengths[0])
        idx = 0.9 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return lengths[lo] * (1 - frac) + lengths[hi] * frac

    @property
    def p10_chars(self) -> float:
        """Return the 10th percentile of document text lengths, or 0.0 if empty.

        This is the 10th percentile of ``len(document.text)``, computed with linear
        interpolation. For a single document it is the length itself. The value
        changes when documents are added, updated (text replacement), removed, or
        cleared. It is independent of chunk count.
        """
        if not self._documents:
            return 0.0
        lengths = sorted(len(document.text) for document in self._documents.values())
        n = len(lengths)
        if n == 1:
            return float(lengths[0])
        idx = 0.1 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return lengths[lo] * (1 - frac) + lengths[hi] * frac

    @property
    def p25_chars(self) -> float:
        """Return the 25th percentile of document text lengths, or 0.0 if empty.

        This is the 25th percentile of ``len(document.text)``, computed with linear
        interpolation. For a single document it is the length itself. The value
        changes when documents are added, updated (text replacement), removed, or
        cleared. It is independent of chunk count. Together with ``p75_chars``
        (when present) this forms the interquartile range reported by ``iqr_chars``.
        """
        if not self._documents:
            return 0.0
        lengths = sorted(len(document.text) for document in self._documents.values())
        n = len(lengths)
        if n == 1:
            return float(lengths[0])
        idx = 0.25 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return lengths[lo] * (1 - frac) + lengths[hi] * frac

    @property
    def p75_chars(self) -> float:
        """Return the 75th percentile of document text lengths, or 0.0 if empty.

        This is the 75th percentile of ``len(document.text)``, computed with linear
        interpolation. For a single document it is the length itself. The value
        changes when documents are added, updated (text replacement), removed, or
        cleared. It is independent of chunk count. Together with ``p25_chars``
        this forms the interquartile range reported by ``iqr_chars``.
        """
        if not self._documents:
            return 0.0
        lengths = sorted(len(document.text) for document in self._documents.values())
        n = len(lengths)
        if n == 1:
            return float(lengths[0])
        idx = 0.75 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return lengths[lo] * (1 - frac) + lengths[hi] * frac

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Return source ids of ingested documents in insertion order."""
        return tuple(self._documents.keys())

    def __len__(self) -> int:
        """Return the number of ingested documents."""
        return len(self._documents)

    def __contains__(self, source_id: object) -> bool:
        """Return True if a document with the given source id is present.

        Non-string values and blank ids return False. Leading and trailing
        whitespace is stripped to match add and remove.
        """
        if not isinstance(source_id, str):
            return False
        cleaned = source_id.strip()
        if not cleaned:
            return False
        return cleaned in self._documents

    def __iter__(self) -> Iterator[Document]:
        """Yield defensive copies of ingested documents in insertion order.

        Mutating a yielded document's metadata cannot change the stored corpus.
        """
        return (_copy_document(document) for document in self._documents.values())

    def get(self, source_id: str) -> Document:
        """Return a copy of the document for the given source id.

        Leading and trailing whitespace is stripped to match add and remove.
        Raises KeyError if the source is not present. The returned document's
        metadata is a copy so later mutation cannot change the stored corpus.
        """
        cleaned_id = _require_source_id(source_id)
        if cleaned_id not in self._documents:
            raise KeyError(f"unknown source: {cleaned_id}")
        return _copy_document(self._documents[cleaned_id])

    def chunks_for(self, source_id: str) -> tuple[Chunk, ...]:
        """Return the chunks belonging to the given source id.

        Leading and trailing whitespace is stripped from the source id.
        Raises KeyError if the source is not present. Chunks are returned in
        index order. The corpus is not changed.
        """
        cleaned_id = _require_source_id(source_id)
        if cleaned_id not in self._documents:
            raise KeyError(f"unknown source: {cleaned_id}")
        return tuple(chunk for chunk in self._chunks if chunk.source_id == cleaned_id)

    def update_metadata(self, source_id: str, metadata: dict[str, str]) -> Document:
        """Replace metadata on an existing document without re-chunking.

        Leading and trailing whitespace is stripped from the source id.
        Raises KeyError if the source is not present. The returned document is a
        copy so later mutation cannot change the stored corpus. Chunks and text
        are left unchanged. Empty metadata is allowed.
        """
        cleaned_id = _require_source_id(source_id)
        if cleaned_id not in self._documents:
            raise KeyError(f"unknown source: {cleaned_id}")
        cleaned_meta = _normalize_metadata_filter(metadata)
        existing = self._documents[cleaned_id]
        updated = Document(
            source_id=existing.source_id,
            text=existing.text,
            metadata=cleaned_meta,
        )
        self._documents[cleaned_id] = updated
        return _copy_document(updated)

    def update_text(self, source_id: str, text: str) -> Document:
        """Replace the text of an existing document and re-chunk it.

        Leading and trailing whitespace is stripped from the source id.
        Raises KeyError if the source is not present. Raises ValueError if text
        is blank. Metadata is preserved. The returned document is a copy so later
        mutation cannot change the stored corpus. Previous chunks for this source
        are discarded and replaced.
        """
        cleaned_id = _require_source_id(source_id)
        if cleaned_id not in self._documents:
            raise KeyError(f"unknown source: {cleaned_id}")
        if not text or not text.strip():
            raise ValueError("text is required")
        existing = self._documents[cleaned_id]
        updated = Document(
            source_id=existing.source_id,
            text=text,
            metadata=dict(existing.metadata),
        )
        self._documents[cleaned_id] = updated
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != cleaned_id]
        self._chunks.extend(self._split(updated))
        return _copy_document(updated)

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
        return _copy_document(document)

    def remove(self, source_id: str) -> Document:
        """Remove one document and its chunks."""
        cleaned_id = _require_source_id(source_id)
        if cleaned_id not in self._documents:
            raise KeyError(f"unknown source: {cleaned_id}")
        document = self._documents.pop(cleaned_id)
        self._chunks = [chunk for chunk in self._chunks if chunk.source_id != cleaned_id]
        return _copy_document(document)

    def clear(self) -> None:
        """Remove every document and chunk."""
        self._documents.clear()
        self._chunks.clear()

    def _split(self, document: Document) -> list[Chunk]:
        text = document.text
        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=f"{document.source_id}:0",
                    source_id=document.source_id,
                    text=text,
                    index=0,
                )
            ]
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_id}:{index}",
                    source_id=document.source_id,
                    text=text[start:end],
                    index=index,
                )
            )
            if end == len(text):
                break
            start += step
            index += 1
        return chunks

    def corpus_stats(self) -> dict[str, int | float]:
        """Return a snapshot of corpus statistics.

        Keys match the corresponding properties (document_count, total_chars,
        max_chars, min_chars, avg_chars, median_chars, std_chars, var_chars,
        range_chars, mode_chars, nunique_chars, iqr_chars, p10_chars, p25_chars,
        p75_chars, p90_chars). The returned dict is independent of later
        mutations to the corpus.
        """
        return {
            "document_count": self.document_count,
            "total_chars": self.total_chars,
            "max_chars": self.max_chars,
            "min_chars": self.min_chars,
            "avg_chars": self.avg_chars,
            "median_chars": self.median_chars,
            "std_chars": self.std_chars,
            "var_chars": self.var_chars,
            "range_chars": self.range_chars,
            "mode_chars": self.mode_chars,
            "nunique_chars": self.nunique_chars,
            "iqr_chars": self.iqr_chars,
            "p10_chars": self.p10_chars,
            "p25_chars": self.p25_chars,
            "p75_chars": self.p75_chars,
            "p90_chars": self.p90_chars,
        }

    def documents_matching(self, metadata: dict[str, str] | None = None) -> tuple[Document, ...]:
        """Return documents whose metadata satisfies every key/value in the filter."""
        required = _normalize_metadata_filter(metadata)
        return tuple(
            _copy_document(document)
            for document in self._documents.values()
            if _metadata_matches(document, required)
        )

    def find(self, metadata: dict[str, str] | None = None) -> tuple[Document, ...]:
        """Return documents whose metadata satisfies every key/value in the filter.

        Alias for ``documents_matching``. Empty or omitted metadata returns every
        document in insertion order. Returned documents are copies so later
        mutation of metadata cannot change the stored corpus.
        """
        return self.documents_matching(metadata)


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
        raise TypeError("metadata must be a dict")
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
        return not required
    return all(document.metadata.get(key) == value for key, value in required.items())


class Retriever:
    """Rank chunks by token-overlap with the query."""

    def __init__(self, ingester: Ingester) -> None:
        self.ingester = ingester

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        """Return the top matching chunks for a question.

        Scores are the fraction of query tokens that appear in the chunk.
        Empty or whitespace-only questions return no hits.
        Chunks with zero overlap are never returned.
        Optional metadata requires every key/value pair to match the source document.
        """
        _require_min_score(min_score)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        required = _normalize_metadata_filter(metadata)
        query_tokens = _tokens(question)
        if not query_tokens:
            return []
        scored: list[ScoredChunk] = []
        for chunk in self.ingester.chunks:
            document = self.ingester._documents.get(chunk.source_id)
            if not _metadata_matches(document, required):
                continue
            chunk_tokens = set(_tokens(chunk.text))
            if not chunk_tokens:
                continue
            overlap = sum(1 for token in query_tokens if token in chunk_tokens)
            score = overlap / len(query_tokens)
            if score > 0 and score >= min_score:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda item: (-item.score, item.chunk.source_id, item.chunk.index))
        return scored[:top_k]

    def query_with_citations(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata: dict[str, str] | None = None,
    ) -> list[Citation]:
        """Return citations for the top matching chunks."""
        hits = self.query(
            question,
            top_k=top_k,
            min_score=min_score,
            metadata=metadata,
        )
        citations: list[Citation] = []
        for hit in hits:
            document = self.ingester._documents.get(hit.chunk.source_id)
            citations.append(
                cite(
                    hit.chunk,
                    score=hit.score,
                    metadata=_source_metadata(document),
                )
            )
        return citations


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with its relevance score."""

    chunk: Chunk
    score: float


def _require_min_score(min_score: float) -> float:
    if not isinstance(min_score, (int, float)) or isinstance(min_score, bool):
        raise TypeError("min_score must be a number")
    if min_score < 0 or min_score > 1:
        raise ValueError("min_score must be between 0 and 1")
    return float(min_score)


def _source_metadata(document: Document | None) -> dict[str, str]:
    if document is None:
        return {}
    return dict(document.metadata)


