"""Unit tests for in-memory ingest, retrieve, and citation."""

import pytest

from ekap.rag import Citation, Ingester, Retriever, cite

POLICY = (
    "Expense reports over 500 dollars require manager approval. "
    "Travel must be booked through the corporate portal."
)
HANDBOOK = (
    "Vacation accrues at 1.5 days per month. "
    "Unused vacation may roll over up to 10 days."
)


def test_ingest_at_least_two_documents() -> None:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    ingester.add("policy-1", POLICY, metadata={"kind": "policy"})
    ingester.add("handbook-1", HANDBOOK)
    assert len(ingester.documents) == 2
    assert {doc.source_id for doc in ingester.documents} == {"policy-1", "handbook-1"}
    assert len(ingester.chunks) >= 2
    assert {chunk.source_id for chunk in ingester.chunks} == {"policy-1", "handbook-1"}


def test_retrieve_returns_relevant_chunk() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    ingester.add("handbook-1", HANDBOOK)
    hits = Retriever(ingester).query("Who must approve expense reports over 500?", top_k=1)
    assert hits
    assert hits[0].chunk.source_id == "policy-1"
    assert "manager approval" in hits[0].chunk.text
    assert hits[0].score > 0


def test_citation_shape_has_source_id_and_snippet() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    citations = Retriever(ingester).query_with_citations("manager approval", top_k=1)
    assert len(citations) == 1
    citation = citations[0]
    assert isinstance(citation, Citation)
    assert citation.source_id == "policy-1"
    assert citation.chunk_id == "policy-1:0"
    assert isinstance(citation.snippet, str) and citation.snippet
    assert "manager approval" in citation.snippet
    assert citation.score > 0


def test_cite_helper_truncates_snippet() -> None:
    ingester = Ingester(chunk_size=400)
    ingester.add("long-1", "alpha " * 80)
    chunk = ingester.chunks[0]
    citation = cite(chunk, score=0.5, max_snippet=24)
    assert citation.source_id == "long-1"
    assert citation.snippet.endswith("\u2026")
    assert len(citation.snippet) <= 24


def test_empty_query_returns_no_hits() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    assert Retriever(ingester).query("   ") == []


def test_reingest_replaces_chunks_for_same_source() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    ingester.add("policy-1", "Remote work requires a written agreement.")
    assert len(ingester.documents) == 1
    assert all("written agreement" in chunk.text for chunk in ingester.chunks)
    assert all("manager approval" not in chunk.text for chunk in ingester.chunks)


def test_rejects_blank_document() -> None:
    ingester = Ingester()
    with pytest.raises(ValueError):
        ingester.add("policy-1", "   ")
