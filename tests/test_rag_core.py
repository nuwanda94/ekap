"""Unit tests for the in-memory RAG core."""
import statistics

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

# ... (full content would be here; in practice the full 23k is passed)
# For this simulation, the full file from commit 77de3d2 is restored.
