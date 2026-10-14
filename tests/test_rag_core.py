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
    with pytest.raises(KeyError):
        ingester.chunks_for("missing")
    with pytest.raises(ValueError):
        ingester.chunks_for("   ")


def test_chunk_count_reports_total_chunks() -> None:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    assert ingester.chunk_count == 0
    ingester.add("policy-1", POLICY)
    assert ingester.chunk_count >= 1
    before = ingester.chunk_count
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.chunk_count > before
    ingester.remove("policy-1")
    assert ingester.chunk_count < before + 1
    ingester.clear()
    assert ingester.chunk_count == 0


def test_document_count_reports_total_documents() -> None:
    ingester = Ingester()
    assert ingester.document_count == 0
    ingester.add("policy-1", POLICY)
    assert ingester.document_count == 1
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.document_count == 2
    ingester.remove("policy-1")
    assert ingester.document_count == 1
    ingester.clear()
    assert ingester.document_count == 0


def test_total_chars_reports_sum_of_document_lengths() -> None:
    ingester = Ingester()
    assert ingester.total_chars == 0
    ingester.add("a", "ab")
    assert ingester.total_chars == 2
    ingester.add("b", "cdef")
    assert ingester.total_chars == 6
    ingester.update_text("a", "xyz")
    assert ingester.total_chars == 7
    ingester.remove("b")
    assert ingester.total_chars == 3
    ingester.clear()
    assert ingester.total_chars == 0


def test_max_chars_reports_longest_document_length() -> None:
    ingester = Ingester()
    assert ingester.max_chars == 0
    ingester.add("a", "ab")
    assert ingester.max_chars == 2
    ingester.add("b", "cdef")
    assert ingester.max_chars == 4
    ingester.update_text("a", "longertext")
    assert ingester.max_chars == 10
    ingester.remove("a")
    assert ingester.max_chars == 4
    ingester.clear()
    assert ingester.max_chars == 0


def test_min_chars_reports_shortest_document_length() -> None:
    ingester = Ingester()
    assert ingester.min_chars == 0
    ingester.add("a", "ab")
    assert ingester.min_chars == 2
    ingester.add("b", "cdef")
    assert ingester.min_chars == 2
    ingester.update_text("a", "x")
    assert ingester.min_chars == 1
    ingester.remove("a")
    assert ingester.min_chars == 4
    ingester.clear()
    assert ingester.min_chars == 0


def test_avg_chars_reports_mean_document_length() -> None:
    ingester = Ingester()
    assert ingester.avg_chars == 0.0
    ingester.add("a", "ab")
    assert abs(ingester.avg_chars - 2.0) < 1e-9
    ingester.add("b", "cdef")
    assert abs(ingester.avg_chars - 3.0) < 1e-9
    ingester.update_text("a", "x")
    assert abs(ingester.avg_chars - 2.5) < 1e-9
    ingester.remove("b")
    assert abs(ingester.avg_chars - 1.0) < 1e-9
    ingester.clear()
    assert ingester.avg_chars == 0.0


def test_median_chars_reports_median_document_length() -> None:
    ingester = Ingester()
    assert ingester.median_chars == 0.0
    ingester.add("a", "a")
    assert abs(ingester.median_chars - 1.0) < 1e-9
    ingester.add("b", "bb")
    assert abs(ingester.median_chars - 1.5) < 1e-9
    ingester.add("c", "ccc")
    assert abs(ingester.median_chars - 2.0) < 1e-9
    ingester.add("d", "dddd")
    assert abs(ingester.median_chars - 2.5) < 1e-9
    ingester.remove("d")
    assert abs(ingester.median_chars - 2.0) < 1e-9
    ingester.clear()
    assert ingester.median_chars == 0.0


def test_std_chars_reports_population_standard_deviation() -> None:
    ingester = Ingester()
    assert ingester.std_chars == 0.0
    ingester.add("a", "a")
    assert abs(ingester.std_chars - 0.0) < 1e-9
    ingester.add("b", "bb")
    # mean 1.5, deviations 0.5, variance 0.25, std 0.5
    assert abs(ingester.std_chars - 0.5) < 1e-9
    ingester.add("c", "ccc")
    # lengths 1,2,3 mean 2, variance 2/3, std ~0.8165
    assert abs(ingester.std_chars - (2/3)**0.5) < 1e-9
    ingester.clear()
    assert ingester.std_chars == 0.0


def test_var_chars_reports_population_variance() -> None:
    ingester = Ingester()
    assert ingester.var_chars == 0.0
    ingester.add("a", "a")
    assert abs(ingester.var_chars - 0.0) < 1e-9
    ingester.add("b", "bb")
    assert abs(ingester.var_chars - 0.25) < 1e-9
    ingester.add("c", "ccc")
    assert abs(ingester.var_chars - (2/3)) < 1e-9
    ingester.clear()
    assert ingester.var_chars == 0.0


def test_range_chars_reports_max_minus_min() -> None:
    ingester = Ingester()
    assert ingester.range_chars == 0
    ingester.add("a", "a")
    assert ingester.range_chars == 0
    ingester.add("b", "bbb")
    assert ingester.range_chars == 2
    ingester.update_text("a", "cccc")
    assert ingester.range_chars == 1
    ingester.clear()
    assert ingester.range_chars == 0


def test_mode_chars_reports_most_frequent_length() -> None:
    ingester = Ingester()
    assert ingester.mode_chars == 0
    ingester.add("a", "aa")
    assert ingester.mode_chars == 2
    ingester.add("b", "bb")
    assert ingester.mode_chars == 2
    ingester.add("c", "c")
    assert ingester.mode_chars == 2
    ingester.add("d", "c")
    # modes 1 and 2, smallest is 1
    assert ingester.mode_chars == 1
    ingester.clear()
    assert ingester.mode_chars == 0


def test_nunique_chars_reports_distinct_lengths() -> None:
    ingester = Ingester()
    assert ingester.nunique_chars == 0
    ingester.add("a", "a")
    assert ingester.nunique_chars == 1
    ingester.add("b", "bb")
    assert ingester.nunique_chars == 2
    ingester.add("c", "a")
    assert ingester.nunique_chars == 2
    ingester.clear()
    assert ingester.nunique_chars == 0


def test_iqr_chars_reports_interquartile_range() -> None:
    ingester = Ingester()
    assert ingester.iqr_chars == 0.0
    ingester.add("a", "a")
    assert ingester.iqr_chars == 0.0
    ingester.add("b", "bb")
    ingester.add("c", "ccc")
    ingester.add("d", "dddd")
    # sorted 1,2,3,4 p75-p25
    assert abs(ingester.iqr_chars - (ingester.p75_chars - ingester.p25_chars)) < 1e-9
    ingester.clear()
    assert ingester.iqr_chars == 0.0


def test_p90_chars_reports_90th_percentile() -> None:
    ingester = Ingester()
    assert ingester.p90_chars == 0.0
    ingester.add("a", "a")
    assert abs(ingester.p90_chars - 1.0) < 1e-9
    ingester.add("b", "bb")
    ingester.add("c", "ccc")
    # n=3 idx=0.9*2=1.8 → 2*0.2 + 3*0.8 = 2.8
    assert abs(ingester.p90_chars - 2.8) < 1e-9
    ingester.clear()
    assert ingester.p90_chars == 0.0


def test_p10_chars_reports_10th_percentile() -> None:
    ingester = Ingester()
    assert ingester.p10_chars == 0.0
    ingester.add("a", "a")
    assert abs(ingester.p10_chars - 1.0) < 1e-9
    ingester.add("b", "bb")
    ingester.add("c", "ccc")
    # n=3 idx=0.1*2=0.2 → 1*0.8 + 2*0.2 = 1.2
    assert abs(ingester.p10_chars - 1.2) < 1e-9
    ingester.clear()
    assert ingester.p10_chars == 0.0


def test_p25_chars_reports_25th_percentile() -> None:
    ingester = Ingester()
    assert ingester.p25_chars == 0.0
    ingester.add("a", "a")
    assert abs(ingester.p25_chars - 1.0) < 1e-9
    ingester.add("b", "bb")
    ingester.add("c", "ccc")
    # n=3 idx=0.25*2=0.5 → 1*0.5 + 2*0.5 = 1.5
    assert abs(ingester.p25_chars - 1.5) < 1e-9
    ingester.clear()
    assert ingester.p25_chars == 0.0


def test_p75_chars_reports_75th_percentile() -> None:
    ingester = Ingester()
    assert ingester.p75_chars == 0.0
    ingester.add("a", "a")
    assert abs(ingester.p75_chars - 1.0) < 1e-9
    ingester.add("b", "bb")
    ingester.add("c", "ccc")
    # n=3 idx=0.75*2=1.5 → 2*0.5 + 3*0.5 = 2.5
    assert abs(ingester.p75_chars - 2.5) < 1e-9
    ingester.add("d", "dddd")
    # n=4 idx=0.75*3=2.25 → 3*0.75 + 4*0.25 = 3.25
    assert abs(ingester.p75_chars - 3.25) < 1e-9
    # Remove
    ingester.remove("d")  # [1,2,3]
    # n=3, idx=0.75*2=1.5 → 2*0.5 + 3*0.5 = 2.5
    assert abs(ingester.p75_chars - 2.5) < 1e-9
    ingester.clear()
    assert ingester.p75_chars == 0.0
    # Consistent with iqr
    ingester.add("a", "a")
    ingester.add("b", "bb")
    ingester.add("c", "ccc")
    ingester.add("d", "dddd")
    assert abs(ingester.iqr_chars - (ingester.p75_chars - ingester.p25_chars)) < 1e-9


def test_update_metadata_replaces_without_rechunking() -> None:
    ingester = Ingester()
    ingester.add("policy-1", POLICY, metadata={"kind": "policy", "tenant": "acme"})
    original_chunks = ingester.chunks
    updated = ingester.update_metadata("  policy-1  ", {"kind": "handbook", "tenant": "beta"})
    assert updated.source_id == "policy-1"
    assert updated.metadata == {"kind": "handbook", "tenant": "beta"}
    assert updated.text == POLICY
    # Chunks unchanged
    assert ingester.chunks == original_chunks
    # Metadata mutation on return does not affect store
    updated.metadata["kind"] = "mutated"
    again = ingester.get("policy-1")
    assert again.metadata == {"kind": "handbook", "tenant": "beta"}
    with pytest.raises(KeyError):
        ingester.update_metadata("missing", {"kind": "x"})
    with pytest.raises(ValueError):
        ingester.update_metadata("   ", {"kind": "x"})
    with pytest.raises(ValueError):
        empty_key = ""
        ingester.update_metadata("policy-1", {empty_key: "x"})
    with pytest.raises(TypeError):
        ingester.update_metadata("policy-1", {"kind": 1})  # type: ignore[arg-type]


def test_update_text_replaces_and_rechunks() -> None:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    ingester.add("policy-1", POLICY, metadata={"kind": "policy"})
    new_text = "Short new policy text that is different."
    updated = ingester.update_text("  policy-1  ", new_text)
    assert updated.source_id == "policy-1"
    assert updated.text == new_text
    assert updated.metadata == {"kind": "policy"}
    # Chunks changed
    assert any("Short new policy" in c.text for c in ingester.chunks)
    assert all("manager approval" not in c.text for c in ingester.chunks)
    # Mutation on return does not affect store
    updated.metadata["kind"] = "mutated"
    again = ingester.get("policy-1")
    assert again.metadata == {"kind": "policy"}
    with pytest.raises(KeyError):
        ingester.update_text("missing", "text")
    with pytest.raises(ValueError):
        ingester.update_text("policy-1", "   ")
    with pytest.raises(ValueError):
        ingester.update_text("   ", "text")


def test_corpus_stats_snapshot_matches_properties() -> None:
    ingester = Ingester()
    empty = ingester.corpus_stats()
    assert empty["document_count"] == 0
    assert empty["total_chars"] == 0
    assert empty["max_chars"] == 0
    assert empty["min_chars"] == 0
    assert empty["avg_chars"] == 0.0
    assert empty["median_chars"] == 0.0
    assert empty["std_chars"] == 0.0
    assert empty["var_chars"] == 0.0
    assert empty["range_chars"] == 0
    assert empty["mode_chars"] == 0
    assert empty["nunique_chars"] == 0
    assert empty["iqr_chars"] == 0.0
    assert empty["p10_chars"] == 0.0
    assert empty["p25_chars"] == 0.0
    assert empty["p75_chars"] == 0.0
    assert empty["p90_chars"] == 0.0
    ingester.add("a", "ab")
    ingester.add("b", "cdef")
    stats = ingester.corpus_stats()
    assert stats["document_count"] == 2
    assert stats["total_chars"] == 6
    assert stats["max_chars"] == 4
    assert stats["min_chars"] == 2
    assert abs(stats["avg_chars"] - 3.0) < 1e-9
    # Snapshot is independent
    ingester.clear()
    assert stats["document_count"] == 2


def test_chunk_stats_snapshot_matches_properties() -> None:
    ingester = Ingester(chunk_size=10, chunk_overlap=0)
    empty = ingester.chunk_stats()
    assert empty["chunk_count"] == 0
    assert empty["chunk_total_chars"] == 0
    assert empty["chunk_max_chars"] == 0
    assert empty["chunk_min_chars"] == 0
    assert empty["chunk_avg_chars"] == 0.0
    ingester.add("a", "abcdefghij")  # 10 chars, one chunk
    ingester.add("b", "abcdefghijabcdefghij")  # 20 chars, two chunks
    stats = ingester.chunk_stats()
    assert stats["chunk_count"] == 3
    assert stats["chunk_total_chars"] == 30
    assert stats["chunk_max_chars"] == 10
    assert stats["chunk_min_chars"] == 10
    assert abs(stats["chunk_avg_chars"] - 10.0) < 1e-9
    # Snapshot is independent
    ingester.clear()
    assert stats["chunk_count"] == 3


def test_chunk_median_chars_reports_median_chunk_length() -> None:
    ingester = Ingester(chunk_size=20, chunk_overlap=0)
    assert ingester.chunk_median_chars == 0.0
    ingester.add("a", "abcdefghij")  # one chunk of 10
    assert abs(ingester.chunk_median_chars - 10.0) < 1e-9
    ingester.add("b", "abcdefghijabcdefghij")  # one chunk of 20 (chunk_size=20)
    # lengths 10,20 → median 15
    assert abs(ingester.chunk_median_chars - 15.0) < 1e-9
    ingester.add("c", "short")  # one chunk of 5
    # lengths 10,20,5 → sorted 5,10,20 median 10
    assert abs(ingester.chunk_median_chars - 10.0) < 1e-9
    stats = ingester.chunk_stats()
    assert "chunk_median_chars" in stats
    assert abs(stats["chunk_median_chars"] - 10.0) < 1e-9
    ingester.clear()
    assert ingester.chunk_median_chars == 0.0



def test_chunk_std_chars_reports_population_standard_deviation() -> None:
    ingester = Ingester(chunk_size=20, chunk_overlap=0)
    assert ingester.chunk_std_chars == 0.0
    ingester.add("a", "abcdefghij")  # one chunk of 10
    assert abs(ingester.chunk_std_chars - 0.0) < 1e-9
    ingester.add("b", "abcdefghijabcdefghij")  # one chunk of 20
    # lengths 10,20 mean 15, deviations 5, variance 25, std 5
    assert abs(ingester.chunk_std_chars - 5.0) < 1e-9
    ingester.add("c", "short")  # one chunk of 5
    # lengths [10, 20, 5], mean = 35/3
    expected_var = ((10 - 35/3)**2 + (20 - 35/3)**2 + (5 - 35/3)**2) / 3
    assert abs(ingester.chunk_std_chars - expected_var**0.5) < 1e-9
    stats = ingester.chunk_stats()
    assert "chunk_std_chars" in stats
    assert abs(stats["chunk_std_chars"] - expected_var**0.5) < 1e-9
    ingester.clear()
    assert ingester.chunk_std_chars == 0.0

