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
BETA_POLICY = (
    "Expense reports over 500 dollars require director approval at Beta. "
    "Travel must be booked through the corporate portal."
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


def test_remove_withdraws_source_from_retrieval() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    ingester.add("handbook-1", HANDBOOK)
    removed = ingester.remove(" policy-1 ")
    assert removed.source_id == "policy-1"
    assert {doc.source_id for doc in ingester.documents} == {"handbook-1"}
    assert all(chunk.source_id != "policy-1" for chunk in ingester.chunks)
    hits = Retriever(ingester).query("Who must approve expense reports over 500?")
    assert hits == [] or all(hit.chunk.source_id != "policy-1" for hit in hits)
    with pytest.raises(KeyError):
        ingester.remove("policy-1")


def test_clear_empties_corpus_and_leaves_settings() -> None:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    ingester.add("policy-1", POLICY)
    ingester.add("handbook-1", HANDBOOK)
    assert len(ingester.documents) == 2
    assert len(ingester.chunks) >= 2
    ingester.clear()
    assert ingester.documents == ()
    assert ingester.chunks == ()
    assert ingester.chunk_size == 80
    assert ingester.chunk_overlap == 10
    # Can still ingest after clear
    ingester.add("policy-1", POLICY)
    assert len(ingester.documents) == 1
    hits = Retriever(ingester).query("manager approval", top_k=1)
    assert hits and hits[0].chunk.source_id == "policy-1"


def test_len_reports_document_count() -> None:
    ingester = Ingester()
    assert len(ingester) == 0
    ingester.add("policy-1", POLICY)
    assert len(ingester) == 1
    ingester.add("handbook-1", HANDBOOK)
    assert len(ingester) == 2
    ingester.remove("policy-1")
    assert len(ingester) == 1
    ingester.clear()
    assert len(ingester) == 0


def test_contains_reports_source_presence() -> None:
    ingester = Ingester()
    assert "policy-1" not in ingester
    assert 1 not in ingester
    assert "" not in ingester
    assert "  " not in ingester
    ingester.add("policy-1", POLICY)
    assert "policy-1" in ingester
    assert "  policy-1  " in ingester
    assert "handbook-1" not in ingester
    ingester.add("handbook-1", HANDBOOK)
    assert "handbook-1" in ingester
    ingester.remove("policy-1")
    assert "policy-1" not in ingester
    assert "handbook-1" in ingester
    ingester.clear()
    assert "handbook-1" not in ingester


def test_iter_yields_document_copies_in_insertion_order() -> None:
    ingester = Ingester()
    assert list(ingester) == []
    ingester.add("policy-1", POLICY, metadata={"kind": "policy"})
    ingester.add("handbook-1", HANDBOOK)
    docs = list(ingester)
    assert [doc.source_id for doc in docs] == ["policy-1", "handbook-1"]
    assert docs[0].text == POLICY
    assert docs[0].metadata == {"kind": "policy"}
    docs[0].metadata["kind"] = "mutated"
    again = list(ingester)
    assert again[0].metadata == {"kind": "policy"}
    ingester.clear()
    assert list(ingester) == []


def test_source_ids_reports_insertion_order() -> None:
    ingester = Ingester()
    assert ingester.source_ids == ()
    ingester.add("policy-1", POLICY)
    assert ingester.source_ids == ("policy-1",)
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.source_ids == ("policy-1", "handbook-1")
    ingester.add("policy-1", "updated")
    assert ingester.source_ids == ("policy-1", "handbook-1")
    ingester.remove("policy-1")
    assert ingester.source_ids == ("handbook-1",)
    ingester.clear()
    assert ingester.source_ids == ()


def test_get_returns_document_copy() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY, metadata={"kind": "policy"})
    document = ingester.get("  policy-1  ")
    assert document.source_id == "policy-1"
    assert document.text == POLICY
    assert document.metadata == {"kind": "policy"}
    document.metadata["kind"] = "mutated"
    again = ingester.get("policy-1")
    assert again.metadata == {"kind": "policy"}
    with pytest.raises(KeyError):
        ingester.get("missing")
    with pytest.raises(ValueError):
        ingester.get("   ")


def test_chunks_for_returns_source_chunks() -> None:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    ingester.add("policy-1", POLICY, metadata={"kind": "policy"})
    ingester.add("handbook-1", HANDBOOK)
    policy_chunks = ingester.chunks_for("  policy-1  ")
    assert all(chunk.source_id == "policy-1" for chunk in policy_chunks)
    assert any("manager approval" in chunk.text for chunk in policy_chunks)
    assert any("corporate portal" in chunk.text for chunk in policy_chunks)
    assert len(policy_chunks) >= 1
    assert policy_chunks == tuple(c for c in ingester.chunks if c.source_id == "policy-1")
    handbook_chunks = ingester.chunks_for("handbook-1")
    assert all(chunk.source_id == "handbook-1" for chunk in handbook_chunks)
    assert len(handbook_chunks) >= 1
    # Order by index
    assert [c.index for c in policy_chunks] == sorted(c.index for c in policy_chunks)
    with pytest.raises(KeyError):
        ingester.chunks_for("missing")
    with pytest.raises(ValueError):
        ingester.chunks_for("   ")
    # After update, chunks_for sees the new chunks
    ingester.update_text("policy-1", "Short new policy text.")
    new_chunks = ingester.chunks_for("policy-1")
    assert len(new_chunks) == 1
    assert "Short new policy" in new_chunks[0].text
    assert "manager approval" not in new_chunks[0].text


def test_chunk_count_reports_total_chunks() -> None:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    assert ingester.chunk_count == 0
    ingester.add("policy-1", POLICY)
    first = ingester.chunk_count
    assert first >= 1
    assert first == len(ingester.chunks)
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.chunk_count > first
    assert ingester.chunk_count == len(ingester.chunks)
    ingester.remove("policy-1")
    assert ingester.chunk_count == len(ingester.chunks)
    ingester.clear()
    assert ingester.chunk_count == 0
    # After clear, adding again works
    ingester.add("policy-1", POLICY)
    assert ingester.chunk_count == len(ingester.chunks)


def test_document_count_reports_total_documents() -> None:
    ingester = Ingester()
    assert ingester.document_count == 0
    assert ingester.document_count == len(ingester)
    ingester.add("policy-1", POLICY)
    assert ingester.document_count == 1
    assert ingester.document_count == len(ingester)
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.document_count == 2
    assert ingester.document_count == len(ingester)
    # Re-add same source does not increase count
    ingester.add("policy-1", "updated text")
    assert ingester.document_count == 2
    ingester.remove("policy-1")
    assert ingester.document_count == 1
    assert ingester.document_count == len(ingester)
    ingester.clear()
    assert ingester.document_count == 0
    assert ingester.document_count == len(ingester)
    # After clear, adding again works
    ingester.add("policy-1", POLICY)
    assert ingester.document_count == 1


def test_total_chars_reports_sum_of_document_lengths() -> None:
    ingester = Ingester()
    assert ingester.total_chars == 0
    ingester.add("policy-1", POLICY)
    assert ingester.total_chars == len(POLICY)
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.total_chars == len(POLICY) + len(HANDBOOK)
    # Replacing text updates the total
    new_text = "Short."
    ingester.update_text("policy-1", new_text)
    assert ingester.total_chars == len(new_text) + len(HANDBOOK)
    # Removing reduces the total
    ingester.remove("handbook-1")
    assert ingester.total_chars == len(new_text)
    ingester.clear()
    assert ingester.total_chars == 0
    # After clear, adding again works
    ingester.add("policy-1", POLICY)
    assert ingester.total_chars == len(POLICY)
    # Re-add same source replaces the length
    ingester.add("policy-1", "even shorter")
    assert ingester.total_chars == len("even shorter")


def test_max_chars_reports_longest_document_length() -> None:
    ingester = Ingester()
    assert ingester.max_chars == 0
    ingester.add("policy-1", POLICY)
    assert ingester.max_chars == len(POLICY)
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.max_chars == max(len(POLICY), len(HANDBOOK))
    # Updating a shorter one does not change max
    short = "x"
    ingester.update_text("handbook-1", short)
    assert ingester.max_chars == len(POLICY)
    # Updating the long one to something longer increases it
    longer = POLICY + " extra"
    ingester.update_text("policy-1", longer)
    assert ingester.max_chars == len(longer)
    # Removing the longest reduces to the remaining
    ingester.remove("policy-1")
    assert ingester.max_chars == len(short)
    ingester.clear()
    assert ingester.max_chars == 0
    # After clear, adding again works
    ingester.add("policy-1", POLICY)
    assert ingester.max_chars == len(POLICY)
    # Re-add same source with shorter text reduces it
    ingester.add("policy-1", "short")
    assert ingester.max_chars == len("short")


def test_min_chars_reports_shortest_document_length() -> None:
    ingester = Ingester()
    assert ingester.min_chars == 0
    ingester.add("policy-1", POLICY)
    assert ingester.min_chars == len(POLICY)
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.min_chars == min(len(POLICY), len(HANDBOOK))
    # Updating a longer one does not change min
    longer = POLICY + " extra text here"
    ingester.update_text("policy-1", longer)
    assert ingester.min_chars == len(HANDBOOK)
    # Updating the short one to something shorter decreases it
    shorter = "x"
    ingester.update_text("handbook-1", shorter)
    assert ingester.min_chars == len(shorter)
    # Removing the shortest increases to the remaining
    ingester.remove("handbook-1")
    assert ingester.min_chars == len(longer)
    ingester.clear()
    assert ingester.min_chars == 0
    # After clear, adding again works
    ingester.add("policy-1", POLICY)
    assert ingester.min_chars == len(POLICY)
    # Re-add same source with longer text increases it
    ingester.add("policy-1", longer)
    assert ingester.min_chars == len(longer)


def test_update_metadata_replaces_without_rechunking() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY, metadata={"kind": "policy", "tenant": "acme"})
    original_chunks = ingester.chunks
    assert len(original_chunks) >= 1
    updated = ingester.update_metadata("  policy-1  ", {"kind": "handbook", "status": "active"})
    assert updated.source_id == "policy-1"
    assert updated.text == POLICY
    assert updated.metadata == {"kind": "handbook", "status": "active"}
    # Returned copy is defensive
    updated.metadata["status"] = "mutated"
    stored = ingester.get("policy-1")
    assert stored.metadata == {"kind": "handbook", "status": "active"}
    # Chunks unchanged
    assert ingester.chunks == original_chunks
    # Filtering now sees the new metadata
    hits = Retriever(ingester).query("manager approval", metadata={"kind": "handbook"})
    assert hits and hits[0].chunk.source_id == "policy-1"
    # Empty metadata is allowed
    cleared = ingester.update_metadata("policy-1", {})
    assert cleared.metadata == {}
    # Missing source
    with pytest.raises(KeyError):
        ingester.update_metadata("missing", {"x": "y"})
    # Invalid metadata
    with pytest.raises(TypeError):
        ingester.update_metadata("policy-1", {"x": 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        ingester.update_metadata("policy-1", {"": "y"})
    with pytest.raises(ValueError):
        ingester.update_metadata("   ", {"x": "y"})


def test_update_text_replaces_and_rechunks() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY, metadata={"kind": "policy"})
    original_chunk_texts = [chunk.text for chunk in ingester.chunks]
    updated = ingester.update_text("  policy-1  ", "New policy: all expenses need CFO approval.")
    assert updated.source_id == "policy-1"
    assert updated.text == "New policy: all expenses need CFO approval."
    assert updated.metadata == {"kind": "policy"}
    # Returned copy is defensive
    updated.metadata["kind"] = "mutated"
    stored = ingester.get("policy-1")
    assert stored.metadata == {"kind": "policy"}
    # Chunks updated to new text
    assert all("CFO approval" in chunk.text for chunk in ingester.chunks)
    assert all("manager approval" not in chunk.text for chunk in ingester.chunks)
    assert [chunk.text for chunk in ingester.chunks] != original_chunk_texts
    # Retrieval uses the new text
    hits = Retriever(ingester).query("CFO approval", top_k=1)
    assert hits and hits[0].chunk.source_id == "policy-1"
    # Missing source
    with pytest.raises(KeyError):
        ingester.update_text("missing", "text")
    # Blank text
    with pytest.raises(ValueError):
        ingester.update_text("policy-1", "   ")
    # Blank source id
    with pytest.raises(ValueError):
        ingester.update_text("   ", "text")


def test_rejects_blank_document() -> None:
    ingester = Ingester()
    with pytest.raises(ValueError):
        ingester.add("policy-1", "   ")


def test_query_filters_by_document_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    retriever = Retriever(ingester)
    unfiltered = retriever.query("expense reports over 500 dollars", top_k=2)
    assert {hit.chunk.source_id for hit in unfiltered} == {"policy-a", "policy-b"}

    filtered = retriever.query(
        "expense reports over 500 dollars",
        top_k=2,
        metadata={"tenant": "beta"},
    )
    assert [hit.chunk.source_id for hit in filtered] == ["policy-b"]
    assert "director approval" in filtered[0].chunk.text

    citations = retriever.query_with_citations(
        "expense reports over 500 dollars",
        top_k=2,
        metadata={" tenant ": "acme", "kind": "policy"},
    )
    assert [citation.source_id for citation in citations] == ["policy-a"]
    assert "manager approval" in citations[0].snippet
    assert retriever.query("expense reports", metadata={"tenant": "missing"}) == []

    with pytest.raises(TypeError):
        retriever.query("expense reports", metadata={"tenant": 1})  # type: ignore[dict-item]


def test_citation_carries_source_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK)
    citations = Retriever(ingester).query_with_citations("manager approval", top_k=1)
    assert len(citations) == 1
    assert citations[0].source_id == "policy-a"
    assert citations[0].metadata == {"tenant": "acme", "kind": "policy"}
    citations[0].metadata["tenant"] = "mutated"
    again = Retriever(ingester).query_with_citations("manager approval", top_k=1)
    assert again[0].metadata == {"tenant": "acme", "kind": "policy"}
    assert cite(ingester.chunks[0]).metadata == {}


def test_find_lists_sources_by_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})

    everything = ingester.find()
    assert [document.source_id for document in everything] == [
        "policy-a",
        "handbook-1",
        "policy-b",
    ]

    beta = ingester.find(metadata={" tenant ": "beta"})
    assert [document.source_id for document in beta] == ["policy-b"]
    assert beta[0].text == BETA_POLICY
    assert beta[0].metadata == {"tenant": "beta", "kind": "policy"}

    acme_policies = ingester.find(metadata={"tenant": "acme", "kind": "policy"})
    assert [document.source_id for document in acme_policies] == ["policy-a"]
    assert ingester.find(metadata={"tenant": "missing"}) == ()

    beta[0].metadata["tenant"] = "mutated"
    stored = ingester.find(metadata={"tenant": "beta"})
    assert stored[0].metadata == {"tenant": "beta", "kind": "policy"}

    with pytest.raises(TypeError):
        ingester.find(metadata={"tenant": 1})  # type: ignore[dict-item]


def test_query_drops_hits_below_min_score() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    ingester.add("handbook-1", HANDBOOK)
    retriever = Retriever(ingester)
    question = "manager approval unused vacation roll"

    unfiltered = retriever.query(question, top_k=2)
    assert [hit.chunk.source_id for hit in unfiltered] == ["handbook-1", "policy-1"]
    assert unfiltered[0].score == pytest.approx(0.6)
    assert unfiltered[1].score == pytest.approx(0.4)

    kept = retriever.query(question, top_k=2, min_score=0.5)
    assert [hit.chunk.source_id for hit in kept] == ["handbook-1"]
    assert "unused vacation" in kept[0].chunk.text.lower()

    citations = retriever.query_with_citations(question, top_k=2, min_score=0.5)
    assert [citation.source_id for citation in citations] == ["handbook-1"]
    assert citations[0].score == pytest.approx(0.6)
    assert "roll over" in citations[0].snippet

    assert retriever.query(question, min_score=1) == []
    assert len(ingester.documents) == 2
    assert {doc.source_id for doc in ingester.documents} == {"policy-1", "handbook-1"}

    with pytest.raises(ValueError):
        retriever.query(question, min_score=-0.1)
    with pytest.raises(ValueError):
        retriever.query_with_citations(question, min_score=1.1)
    with pytest.raises(TypeError):
        retriever.query(question, min_score="high")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        retriever.query(question, min_score=True)  # type: ignore[arg-type]
