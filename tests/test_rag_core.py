"""Unit tests for the in-memory RAG core."""

import pytest

from ekap.rag import Citation, Ingester, Retriever, cite


def test_ingest_two_documents() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    ingester.add(
        "handbook-1",
        "Vacation accrues at 1.5 days per month.",
        metadata={"kind": "handbook"},
    )
    assert len(ingester.documents) == 2
    assert {document.source_id for document in ingester.documents} == {"policy-1", "handbook-1"}
    assert len(ingester.chunks) == 2


def test_retrieve_relevant_chunk() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    ingester.add("handbook-1", "Vacation accrues at 1.5 days per month.")
    hits = Retriever(ingester).query("Who must approve expense reports over 500?", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk.source_id == "policy-1"
    assert "manager approval" in hits[0].chunk.text
    assert hits[0].score > 0


def test_citation_shape_has_source_id_and_snippet() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    ingester.add("handbook-1", "Vacation accrues at 1.5 days per month.")
    citations = Retriever(ingester).query_with_citations(
        "Who must approve expense reports over 500?",
        top_k=1,
    )
    assert len(citations) == 1
    citation = citations[0]
    assert isinstance(citation, Citation)
    assert citation.source_id == "policy-1"
    assert "manager approval" in citation.snippet
    assert citation.chunk_id
    assert isinstance(citation.score, float)


def test_cite_helper_copies_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Approval required.", metadata={"kind": "policy"})
    hit = Retriever(ingester).query("approval", top_k=1)[0]
    citation = cite(hit)
    assert citation.metadata == {"kind": "policy"}
    citation.metadata["kind"] = "mutated"
    assert hit.chunk.metadata["kind"] == "policy"


def test_find_filters_by_metadata() -> None:
    ingester = Ingester()
    ingester.add("a", "one", metadata={"kind": "policy"})
    ingester.add("b", "two", metadata={"kind": "handbook"})
    found = ingester.find(metadata={"kind": "policy"})
    assert [document.source_id for document in found] == ["a"]


def test_blank_source_id_rejected() -> None:
    with pytest.raises(ValueError, match="source_id is required"):
        Ingester().add("   ", "text")
