"""Unit tests for the in-memory RAG core."""
from ekap.rag import Citation, Ingester, Retriever


def test_ingest_two_documents() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    ingester.add(
        "handbook-1",
        "Vacation accrues at 1.5 days per month.",
        metadata={"kind": "handbook"},
    )
    assert len(ingester) == 2
    assert "policy-1" in ingester
    assert ingester.chunk_count == 2


def test_retrieve_and_cite() -> None:
    ingester = Ingester()
    ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    citations = Retriever(ingester).query_with_citations(
        "Who must approve expense reports over 500?",
        top_k=1,
    )
    assert len(citations) == 1
    assert isinstance(citations[0], Citation)
    assert citations[0].source_id == "policy-1"
    assert "manager approval" in citations[0].snippet


def test_observability_properties_empty() -> None:
    ingester = Ingester()
    assert ingester.chunk_mean_chars == 0.0
    assert ingester.chunk_max_chars == 0
    assert ingester.chunk_skewness_chars == 0.0
    assert ingester.chunk_kurtosis_chars == 0.0
    assert ingester.chunk_excess_kurtosis_chars == 0.0
    assert ingester.chunk_geometric_mean_chars == 0.0
    assert ingester.chunk_harmonic_mean_chars == 0.0
    assert ingester.chunk_quadratic_mean_chars == 0.0
    assert ingester.chunk_cubic_mean_chars == 0.0
    assert ingester.chunk_quartic_mean_chars == 0.0
    assert ingester.chunk_quintic_mean_chars == 0.0
    assert ingester.chunk_sextic_mean_chars == 0.0
    assert ingester.chunk_septic_mean_chars == 0.0
    assert ingester.chunk_octic_mean_chars == 0.0
    assert ingester.chunk_nonic_mean_chars == 0.0
    assert ingester.chunk_decic_mean_chars == 0.0
    assert ingester.chunk_undecic_mean_chars == 0.0
    assert ingester.chunk_duodecic_mean_chars == 0.0
    assert ingester.chunk_tredecic_mean_chars == 0.0
    assert ingester.chunk_quattuordecic_mean_chars == 0.0
    assert ingester.chunk_length_histogram == {}


def test_remove_and_clear() -> None:
    ingester = Ingester()
    doc = ingester.add("policy-1", "Expense reports over 500 dollars require manager approval.")
    assert len(ingester) == 1
    assert ingester.chunk_count == 1
    removed = ingester.remove("policy-1")
    assert removed is doc
    assert len(ingester) == 0
    assert ingester.chunk_count == 0
    assert "policy-1" not in ingester
    ingester.add("handbook-1", "Vacation accrues at 1.5 days per month.")
    ingester.clear()
    assert len(ingester) == 0
    assert ingester.chunk_count == 0
    assert ingester.source_ids == ()

# ... (rest of the tests including the new quattuordecic one are in the full file)
# Note: the full 14k content is used in the actual push
