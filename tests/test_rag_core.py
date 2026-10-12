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


def test_avg_chars_reports_mean_document_length() -> None:
    ingester = Ingester()
    assert ingester.avg_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.avg_chars == float(len(POLICY))
    ingester.add("handbook-1", HANDBOOK)
    expected = (len(POLICY) + len(HANDBOOK)) / 2
    assert ingester.avg_chars == expected
    # Updating one changes the average
    short = "xy"
    ingester.update_text("handbook-1", short)
    expected = (len(POLICY) + len(short)) / 2
    assert ingester.avg_chars == expected
    # Removing reduces to the remaining single document
    ingester.remove("handbook-1")
    assert ingester.avg_chars == float(len(POLICY))
    ingester.clear()
    assert ingester.avg_chars == 0.0
    # After clear, adding again works
    ingester.add("policy-1", POLICY)
    assert ingester.avg_chars == float(len(POLICY))
    # Re-add same source replaces
    ingester.add("policy-1", short)
    assert ingester.avg_chars == float(len(short))
    # Three documents
    ingester.add("extra-1", "abc")
    expected = (len(short) + 3) / 2
    assert abs(ingester.avg_chars - expected) < 1e-9


def test_median_chars_reports_median_document_length() -> None:
    ingester = Ingester()
    assert ingester.median_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.median_chars == float(len(POLICY))
    # Two documents: average of the two lengths
    ingester.add("handbook-1", HANDBOOK)
    expected = (len(POLICY) + len(HANDBOOK)) / 2.0
    assert ingester.median_chars == expected
    # Three documents: middle value after sorting
    short = "xy"
    ingester.add("extra-1", short)
    lengths = sorted([len(POLICY), len(HANDBOOK), len(short)])
    assert ingester.median_chars == float(lengths[1])
    # Updating changes the median
    medium = "abcde"
    ingester.update_text("extra-1", medium)
    lengths = sorted([len(POLICY), len(HANDBOOK), len(medium)])
    assert ingester.median_chars == float(lengths[1])
    # Removing one leaves even count again
    ingester.remove("handbook-1")
    expected = (len(POLICY) + len(medium)) / 2.0
    assert ingester.median_chars == expected
    ingester.clear()
    assert ingester.median_chars == 0.0
    # Re-add works
    ingester.add("a", "hello")
    ingester.add("b", "hi")
    ingester.add("c", "hey")
    assert ingester.median_chars == 3.0  # 2, 3, 5 → 3


def test_std_chars_reports_population_standard_deviation() -> None:
    ingester = Ingester()
    assert ingester.std_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.std_chars == 0.0  # single document
    ingester.add("handbook-1", HANDBOOK)
    lengths = [len(POLICY), len(HANDBOOK)]
    mean = sum(lengths) / 2
    variance = sum((x - mean) ** 2 for x in lengths) / 2
    assert abs(ingester.std_chars - variance ** 0.5) < 1e-9
    # Three documents
    short = "xy"
    ingester.add("extra-1", short)
    lengths = [len(POLICY), len(HANDBOOK), len(short)]
    mean = sum(lengths) / 3
    variance = sum((x - mean) ** 2 for x in lengths) / 3
    assert abs(ingester.std_chars - variance ** 0.5) < 1e-9
    # Update changes it
    medium = "abcde"
    ingester.update_text("extra-1", medium)
    lengths = [len(POLICY), len(HANDBOOK), len(medium)]
    mean = sum(lengths) / 3
    variance = sum((x - mean) ** 2 for x in lengths) / 3
    assert abs(ingester.std_chars - variance ** 0.5) < 1e-9
    # Remove back to two
    ingester.remove("handbook-1")
    lengths = [len(POLICY), len(medium)]
    mean = sum(lengths) / 2
    variance = sum((x - mean) ** 2 for x in lengths) / 2
    assert abs(ingester.std_chars - variance ** 0.5) < 1e-9
    ingester.clear()
    assert ingester.std_chars == 0.0
    # Equal lengths → 0
    ingester.add("a", "aaaa")
    ingester.add("b", "bbbb")
    assert ingester.std_chars == 0.0


def test_var_chars_reports_population_variance() -> None:
    ingester = Ingester()
    assert ingester.var_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.var_chars == 0.0  # single document
    ingester.add("handbook-1", HANDBOOK)
    lengths = [len(POLICY), len(HANDBOOK)]
    mean = sum(lengths) / 2
    variance = sum((x - mean) ** 2 for x in lengths) / 2
    assert abs(ingester.var_chars - variance) < 1e-9
    # Consistent with std
    assert abs(ingester.var_chars - ingester.std_chars ** 2) < 1e-9
    # Three documents
    short = "xy"
    ingester.add("extra-1", short)
    lengths = [len(POLICY), len(HANDBOOK), len(short)]
    mean = sum(lengths) / 3
    variance = sum((x - mean) ** 2 for x in lengths) / 3
    assert abs(ingester.var_chars - variance) < 1e-9
    # Update changes it
    medium = "abcde"
    ingester.update_text("extra-1", medium)
    lengths = [len(POLICY), len(HANDBOOK), len(medium)]
    mean = sum(lengths) / 3
    variance = sum((x - mean) ** 2 for x in lengths) / 3
    assert abs(ingester.var_chars - variance) < 1e-9
    # Remove back to two
    ingester.remove("handbook-1")
    lengths = [len(POLICY), len(medium)]
    mean = sum(lengths) / 2
    variance = sum((x - mean) ** 2 for x in lengths) / 2
    assert abs(ingester.var_chars - variance) < 1e-9
    ingester.clear()
    assert ingester.var_chars == 0.0
    # Equal lengths → 0
    ingester.add("a", "aaaa")
    ingester.add("b", "bbbb")
    assert ingester.var_chars == 0.0


def test_range_chars_reports_length_difference() -> None:
    ingester = Ingester()
    assert ingester.range_chars == 0
    ingester.add("policy-1", POLICY)
    assert ingester.range_chars == 0  # single
    ingester.add("handbook-1", HANDBOOK)
    assert ingester.range_chars == abs(len(POLICY) - len(HANDBOOK))
    # Update
    short = "x"
    ingester.update_text("handbook-1", short)
    assert ingester.range_chars == len(POLICY) - len(short)
    longer = POLICY + " extra"
    ingester.update_text("policy-1", longer)
    assert ingester.range_chars == len(longer) - len(short)
    ingester.remove("handbook-1")
    assert ingester.range_chars == 0
    ingester.clear()
    assert ingester.range_chars == 0


def test_mode_chars_reports_most_frequent_length() -> None:
    ingester = Ingester()
    assert ingester.mode_chars == 0
    ingester.add("policy-1", POLICY)
    assert ingester.mode_chars == len(POLICY)
    ingester.add("handbook-1", HANDBOOK)
    # If different lengths, smallest of modes? Wait, depends
    # For simplicity, add duplicates
    ingester.clear()
    ingester.add("a", "aa")
    ingester.add("b", "bb")
    ingester.add("c", "c")
    assert ingester.mode_chars == 2  # two of length 2, one of 1 → mode 2
    ingester.add("d", "d")
    assert ingester.mode_chars == 1  # two of 1, two of 2 → smallest 1
    ingester.clear()
    assert ingester.mode_chars == 0


def test_nunique_chars_reports_distinct_lengths() -> None:
    ingester = Ingester()
    assert ingester.nunique_chars == 0
    ingester.add("policy-1", POLICY)
    assert ingester.nunique_chars == 1
    ingester.add("handbook-1", HANDBOOK)
    expected = 1 if len(POLICY) == len(HANDBOOK) else 2
    assert ingester.nunique_chars == expected
    ingester.clear()
    ingester.add("a", "aa")
    ingester.add("b", "bb")
    assert ingester.nunique_chars == 1
    ingester.add("c", "c")
    assert ingester.nunique_chars == 2
    ingester.clear()
    assert ingester.nunique_chars == 0


def test_iqr_chars_reports_interquartile_range() -> None:
    ingester = Ingester()
    assert ingester.iqr_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.iqr_chars == 0.0  # single
    # Controlled
    ingester.clear()
    ingester.add("a", "a")
    ingester.add("b", "bb")
    assert abs(ingester.iqr_chars - 0.5) < 1e-9  # n=2: Q1=1.25, Q3=1.75, IQR=0.5
    # More precise controlled
    ingester.clear()
    for i, length in enumerate([1, 2, 3, 4, 5]):
        ingester.add(str(i), "x" * length)
    # lengths [1,2,3,4,5], n=5, Q1 idx=1, Q3 idx=3 → 2 and 4, IQR=2
    assert abs(ingester.iqr_chars - 2.0) < 1e-9
    ingester.clear()
    assert ingester.iqr_chars == 0.0


def test_p90_chars_reports_90th_percentile() -> None:
    ingester = Ingester()
    assert ingester.p90_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.p90_chars == float(len(POLICY))  # single document
    # Controlled lengths
    ingester.clear()
    ingester.add("a", "a")  # len 1
    ingester.add("b", "bb")  # len 2
    # n=2: idx = 0.9 * 1 = 0.9 → 1*0.1 + 2*0.9 = 1.9
    assert abs(ingester.p90_chars - 1.9) < 1e-9
    ingester.add("c", "ccc")  # [1,2,3]
    # idx = 0.9 * 2 = 1.8 → 2*0.2 + 3*0.8 = 2.8
    assert abs(ingester.p90_chars - 2.8) < 1e-9
    ingester.add("d", "dddd")  # [1,2,3,4]
    # idx = 0.9 * 3 = 2.7 → 3*0.3 + 4*0.7 = 3.7
    assert abs(ingester.p90_chars - 3.7) < 1e-9
    # Update changes it
    ingester.update_text("d", "ddddd")  # [1,2,3,5]
    # idx = 2.7 → 3*0.3 + 5*0.7 = 4.4
    assert abs(ingester.p90_chars - 4.4) < 1e-9
    # Remove
    ingester.remove("a")  # [2,3,5]
    # n=3, idx=0.9*2=1.8 → 3*0.2 + 5*0.8 = 4.6
    assert abs(ingester.p90_chars - 4.6) < 1e-9
    ingester.clear()
    assert ingester.p90_chars == 0.0


def test_p10_chars_reports_10th_percentile() -> None:
    ingester = Ingester()
    assert ingester.p10_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.p10_chars == float(len(POLICY))  # single document
    # Controlled lengths
    ingester.clear()
    ingester.add("a", "a")  # len 1
    ingester.add("b", "bb")  # len 2
    # n=2: idx = 0.1 * 1 = 0.1 → 1*0.9 + 2*0.1 = 1.1
    assert abs(ingester.p10_chars - 1.1) < 1e-9
    ingester.add("c", "ccc")  # [1,2,3]
    # idx = 0.1 * 2 = 0.2 → 1*0.8 + 2*0.2 = 1.2
    assert abs(ingester.p10_chars - 1.2) < 1e-9
    ingester.add("d", "dddd")  # [1,2,3,4]
    # idx = 0.1 * 3 = 0.3 → 1*0.7 + 2*0.3 = 1.3
    assert abs(ingester.p10_chars - 1.3) < 1e-9
    # Update
    ingester.update_text("d", "ddddd")  # [1,2,3,5]
    # idx = 0.3 → 1*0.7 + 2*0.3 = 1.3
    assert abs(ingester.p10_chars - 1.3) < 1e-9
    # Remove
    ingester.remove("d")  # [1,2,3]
    # n=3, idx=0.1*2=0.2 → 1*0.8 + 2*0.2 = 1.2
    assert abs(ingester.p10_chars - 1.2) < 1e-9
    ingester.clear()
    assert ingester.p10_chars == 0.0


def test_p25_chars_reports_25th_percentile() -> None:
    ingester = Ingester()
    assert ingester.p25_chars == 0.0
    ingester.add("policy-1", POLICY)
    assert ingester.p25_chars == float(len(POLICY))  # single document
    # Controlled lengths
    ingester.clear()
    ingester.add("a", "a")  # len 1
    ingester.add("b", "bb")  # len 2
    # n=2: idx = 0.25 * 1 = 0.25 → 1*0.75 + 2*0.25 = 1.25
    assert abs(ingester.p25_chars - 1.25) < 1e-9
    ingester.add("c", "ccc")  # [1,2,3]
    # idx = 0.25 * 2 = 0.5 → 1*0.5 + 2*0.5 = 1.5
    assert abs(ingester.p25_chars - 1.5) < 1e-9
    ingester.add("d", "dddd")  # [1,2,3,4]
    # idx = 0.25 * 3 = 0.75 → 1*0.25 + 2*0.75 = 1.75
    assert abs(ingester.p25_chars - 1.75) < 1e-9
    # Update
    ingester.update_text("d", "ddddd")  # [1,2,3,5]
    # idx = 0.75 → 1*0.25 + 2*0.75 = 1.75
    assert abs(ingester.p25_chars - 1.75) < 1e-9
    # Remove
    ingester.remove("d")  # [1,2,3]
    # n=3, idx=0.25*2=0.5 → 1*0.5 + 2*0.5 = 1.5
    assert abs(ingester.p25_chars - 1.5) < 1e-9
    ingester.clear()
    assert ingester.p25_chars == 0.0


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
        ingester.update_metadata("policy-1", {"": : "x"})
    with pytest.raises(TypeError):
        ingester.update_metadata("policy-1", {"kind": 1})  # type: ignore[arg-type]


def test_update_text_replaces_and_rechunks() -> None:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    ingester.add("policy-1", POLICY, metadata={"kind": "policy"})
    original_count = ingester.chunk_count
    new_text = "Short new policy text that is different."
    updated = ingester.update_text("  policy-1  ", new_text)
    assert updated.source_id == "policy-1"
    assert updated.text == new_text
    assert updated.metadata == {"kind": "policy"}
    # Chunks changed
    assert ingester.chunk_count != original_count or any("Short new policy" in c.text for c in ingester.chunks)
    assert all("manager approval" not in c.text for c in ingester.chunks)
    assert any("Short new policy" in c.text for c in ingester.chunks)
    # Metadata mutation on return does not affect store
    updated.metadata["kind"] = "mutated"
    again = ingester.get("policy-1")
    assert again.metadata == {"kind": "policy"}
    with pytest.raises(KeyError):
        ingester.update_text("missing", "text")
    with pytest.raises(ValueError):
        ingester.update_text("policy-1", "   ")
    with pytest.raises(ValueError):
        ingester.update_text("   ", "text")


def test_find_filters_by_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"kind": "policy", "tenant": "acme"})
    ingester.add("policy-b", BETA_POLICY, metadata={"kind": "policy", "tenant": "beta"})
    ingester.add("handbook-1", HANDBOOK, metadata={"kind": "handbook", "tenant": "acme"})
    acme = ingester.find(metadata={"tenant": "acme"})
    assert {d.source_id for d in acme} == {"policy-a", "handbook-1"}
    beta = ingester.find(metadata={"tenant": "beta"})
    assert [document.source_id for document in beta] == ["policy-b"]
    assert beta[0].text == BETA_POLICY
    beta[0].metadata["tenant"] = "mutated"
    again = ingester.find(metadata={"tenant": "beta"})
    assert again[0].metadata == {"tenant": "beta", "kind": "policy"}

    both = ingester.find(metadata={"tenant": "acme", "kind": "policy"})
    assert [document.source_id for document in both] == ["policy-a"]

    assert ingester.find(metadata={}) == tuple(ingester.documents)
    assert ingester.find() == tuple(ingester.documents)
    with pytest.raises(ValueError):
        ingester.find(metadata={"": "acme"})
