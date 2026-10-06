"""Unit tests for the in-process MCP tool server."""

import pytest

from ekap.mcp_server import MCPServer, ToolResult
from ekap.rag import Ingester, Retriever

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


def _server() -> tuple[MCPServer, list[dict[str, str]]]:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    ingester.add("handbook-1", HANDBOOK)
    executed: list[dict[str, str]] = []

    def executor(payload: dict[str, str]) -> dict[str, str]:
        executed.append(payload)
        return payload

    return MCPServer(retriever=Retriever(ingester), executor=executor, ingester=ingester), executed


def test_health_is_callable() -> None:
    server, executed = _server()
    result = server.call("health", {"echo": "ping"})
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.draft is False
    assert result.data["status"] == "ok"
    assert result.data["echo"] == "ping"
    assert "search_docs" in result.data["tools"]
    assert "list_sources" in result.data["tools"]
    assert "get_source" in result.data["tools"]
    assert "summarize_sources" in result.data["tools"]
    assert "list_drafts" in result.data["tools"]
    assert "get_draft" in result.data["tools"]
    assert "cancel_draft" in result.data["tools"]
    assert "summarize_drafts" in result.data["tools"]
    assert "describe_tool" in result.data["tools"]
    assert "list_tools" in result.data["tools"]
    assert "draft_remove_source" in result.data["tools"]
    assert executed == []


def test_summarize_sources_counts_without_changing_the_corpus() -> None:
    ingester = Ingester(chunk_size=40, chunk_overlap=0)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("summarize_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "chunks": 0, "chars": 0, "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    chunk_count = len(ingester.chunks)

    result = server.call("summarize_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["chunks"] == chunk_count
    assert result.data["chars"] == len(POLICY) + len(BETA_POLICY) + len(HANDBOOK)
    assert result.data["executed"] is False

    filtered = server.call("summarize_sources", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 2
    assert filtered.data["chars"] == len(POLICY) + len(HANDBOOK)
    assert filtered.data["chunks"] == sum(
        chunk.source_id in {"policy-a", "handbook-1"} for chunk in ingester.chunks
    )
    assert filtered.data["executed"] is False

    missing = MCPServer(retriever=Retriever(ingester))
    no_ingester = missing.call("summarize_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("summarize_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]
    extra = server.call("summarize_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert len(ingester.chunks) == chunk_count
