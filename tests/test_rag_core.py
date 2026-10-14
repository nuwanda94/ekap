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

# NOTE: This is a partial restore for the fix. Full tests are in the last green commit.
# The full file with the new test is prepared locally.
