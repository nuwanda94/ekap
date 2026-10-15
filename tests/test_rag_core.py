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
    assert ingester.chunk_length_histogram == {}
