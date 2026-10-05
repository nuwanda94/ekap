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

    return MCPServer(retriever=Retriever(ingester), executor=executor), executed


def test_health_is_callable() -> None:
    server, executed = _server()
    result = server.call("health", {"echo": "ping"})
    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.draft is False
    assert result.data["status"] == "ok"
    assert result.data["echo"] == "ping"
    assert "search_docs" in result.data["tools"]
    assert executed == []


def test_search_docs_returns_ranked_citation() -> None:
    server, _executed = _server()
    result = server.call("search_docs", {"query": "manager approval for expenses", "top_k": 1})
    assert result.ok is True
    assert result.draft is False
    citations = result.data["citations"]
    assert len(citations) == 1
    assert citations[0]["source_id"] == "policy-1"
    assert "manager approval" in citations[0]["snippet"]
    assert citations[0]["score"] > 0


def test_search_docs_filters_by_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    server = MCPServer(retriever=Retriever(ingester))
    result = server.call(
        "search_docs",
        {
            "query": "expense reports over 500 dollars",
            "top_k": 2,
            "metadata": {"tenant": "beta"},
        },
    )
    assert result.ok is True
    assert result.draft is False
    assert [item["source_id"] for item in result.data["citations"]] == ["policy-b"]
    assert "director approval" in result.data["citations"][0]["snippet"]

    unfiltered = server.call(
        "search_docs",
        {"query": "expense reports over 500 dollars", "top_k": 2},
    )
    assert {item["source_id"] for item in unfiltered.data["citations"]} == {"policy-a", "policy-b"}

    bad = server.call(
        "search_docs",
        {"query": "expense reports", "metadata": {"tenant": 1}},
    )
    assert bad.ok is False
    assert "metadata" in bad.data["error"]
    assert server.executed == []


def test_draft_action_returns_draft_and_does_not_execute() -> None:
    server, executed = _server()
    result = server.call(
        "draft_action",
        {"action": "file_expense", "target": "report-42"},
    )
    assert result.ok is True
    assert result.draft is True
    assert result.data["status"] == "pending"
    assert result.data["executed"] is False
    assert result.data["draft_id"] == "draft-1"
    assert executed == []
    assert server.executed == []
    assert server.drafts() == (
        {
            "draft_id": "draft-1",
            "action": "file_expense",
            "target": "report-42",
            "status": "pending",
        },
    )


def test_unknown_tool_and_missing_args_fail_closed() -> None:
    server, executed = _server()
    unknown = server.call("delete_all")
    assert unknown.ok is False
    assert "unknown tool" in unknown.data["error"]
    missing = server.call("draft_action", {"action": "file_expense"})
    assert missing.ok is False
    assert executed == []
    with pytest.raises(TypeError):
        server.call()  # type: ignore[call-arg]
