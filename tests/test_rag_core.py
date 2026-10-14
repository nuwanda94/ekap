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


def test_cite_truncates_long_snippet() -> None:
    ingester = Ingester(chunk_size=400)
    long_text = "alpha " * 40
    ingester.add("long-1", long_text)
    hit = Retriever(ingester).query("alpha", top_k=1)[0]
    citation = cite(hit, max_snippet=20)
    assert citation.source_id == "long-1"
    assert citation.snippet.endswith("\u2026")
    assert len(citation.snippet) <= 20
    # Without max_snippet the full text is returned.
    full = cite(hit)
    assert full.snippet == long_text
    # query_with_citations also accepts the limit.
    limited = Retriever(ingester).query_with_citations("alpha", top_k=1, max_snippet=20)
    assert limited[0].snippet.endswith("\u2026")
    assert len(limited[0].snippet) <= 20


def test_clear_resets_corpus_preserves_settings() -> None:
    ingester = Ingester(chunk_size=100, chunk_overlap=10)
    ingester.add("a", "one")
    ingester.add("b", "two")
    assert len(ingester) == 2
    assert len(ingester.documents) == 2
    assert len(ingester.chunks) == 2
    ingester.clear()
    assert len(ingester) == 0
    assert len(ingester.documents) == 0
    assert len(ingester.chunks) == 0
    # Settings are preserved.
    assert ingester.chunk_size == 100
    assert ingester.chunk_overlap == 10
    # Can ingest again after clear.
    ingester.add("c", "three")
    assert len(ingester) == 1
    assert ingester.documents[0].source_id == "c"


def test_source_ids_returns_insertion_order() -> None:
    ingester = Ingester()
    assert ingester.source_ids == ()
    ingester.add("policy-1", "Expense reports require approval.")
    ingester.add("handbook-1", "Vacation accrues monthly.")
    ingester.add("policy-1", "Updated expense policy.")  # replace keeps position
    assert ingester.source_ids == ("policy-1", "handbook-1")
    ingester.remove("policy-1")
    assert ingester.source_ids == ("handbook-1",)
    ingester.clear()
    assert ingester.source_ids == ()


def test_contains_source_id() -> None:
    ingester = Ingester()
    assert "policy-1" not in ingester
    assert 123 not in ingester
    assert None not in ingester
    assert "" not in ingester
    ingester.add("policy-1", "Expense reports require approval.")
    assert "policy-1" in ingester
    assert "  policy-1  " in ingester
    assert "handbook-1" not in ingester
    ingester.remove("policy-1")
    assert "policy-1" not in ingester
    assert "  policy-1  " not in ingester


def test_get_returns_document_or_none() -> None:
    ingester = Ingester()
    assert ingester.get("policy-1") is None
    with pytest.raises(TypeError, match="source_id must be a string"):
        ingester.get(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_id is required"):
        ingester.get("   ")
    doc = ingester.add("policy-1", "Expense reports require approval.", metadata={"kind": "policy"})
    fetched = ingester.get("policy-1")
    assert fetched is not None
    assert fetched.source_id == "policy-1"
    assert fetched.text == doc.text
    assert fetched.metadata == {"kind": "policy"}
    # Strip whitespace.
    assert ingester.get("  policy-1  ") is fetched
    assert ingester.get("handbook-1") is None
    ingester.remove("policy-1")
    assert ingester.get("policy-1") is None


def test_chunks_for_returns_source_chunks() -> None:
    ingester = Ingester(chunk_size=20, chunk_overlap=0)
    assert ingester.chunks_for("policy-1") == ()
    with pytest.raises(TypeError, match="source_id must be a string"):
        ingester.chunks_for(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_id is required"):
        ingester.chunks_for("   ")
    long_text = "Expense reports require manager approval and review."
    ingester.add("policy-1", long_text, metadata={"kind": "policy"})
    ingester.add("handbook-1", "Vacation accrues monthly.")
    chunks = ingester.chunks_for("policy-1")
    assert len(chunks) >= 1
    assert all(chunk.source_id == "policy-1" for chunk in chunks)
    assert chunks[0].metadata == {"kind": "policy"}
    # Index order is preserved.
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    # Whitespace is stripped.
    assert ingester.chunks_for("  policy-1  ") == chunks
    assert ingester.chunks_for("missing") == ()
    ingester.remove("policy-1")
    assert ingester.chunks_for("policy-1") == ()
