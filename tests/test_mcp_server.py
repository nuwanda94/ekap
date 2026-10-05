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
    assert "list_drafts" in result.data["tools"]
    assert "draft_remove_source" in result.data["tools"]
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
    assert citations[0]["metadata"] == {}


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


def test_search_docs_citations_include_source_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK)
    server = MCPServer(retriever=Retriever(ingester))
    result = server.call("search_docs", {"query": "manager approval", "top_k": 1})
    assert result.ok is True
    item = result.data["citations"][0]
    assert item["source_id"] == "policy-a"
    assert item["metadata"] == {"tenant": "acme", "kind": "policy"}
    item["metadata"]["tenant"] = "mutated"
    again = server.call("search_docs", {"query": "manager approval", "top_k": 1})
    assert again.data["citations"][0]["metadata"] == {"tenant": "acme", "kind": "policy"}
    handbook = server.call("search_docs", {"query": "unused vacation", "top_k": 1})
    assert handbook.data["citations"][0]["source_id"] == "handbook-1"
    assert handbook.data["citations"][0]["metadata"] == {}


def test_list_sources_filters_by_metadata_without_executing() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    result = server.call("list_sources", {"metadata": {"tenant": "acme"}})
    assert result.ok is True
    assert result.draft is False
    assert [item["source_id"] for item in result.data["sources"]] == ["policy-a", "handbook-1"]
    assert result.data["sources"][0]["metadata"] == {"tenant": "acme", "kind": "policy"}
    assert result.data["sources"][0]["chars"] == len(POLICY)
    result.data["sources"][0]["metadata"]["tenant"] = "mutated"

    again = server.call("list_sources", {"metadata": {"tenant": "acme", "kind": "policy"}})
    assert [item["source_id"] for item in again.data["sources"]] == ["policy-a"]
    assert again.data["sources"][0]["metadata"] == {"tenant": "acme", "kind": "policy"}

    missing = MCPServer(retriever=Retriever(ingester))
    no_ingester = missing.call("list_sources")
    assert no_ingester.ok is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("list_sources", {"metadata": ["tenant"]})
    assert bad.ok is False
    assert "metadata" in bad.data["error"]
    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()


def test_get_source_returns_text_and_copied_metadata() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    result = server.call("get_source", {"source_id": " policy-a "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["source_id"] == "policy-a"
    assert result.data["text"] == POLICY
    assert result.data["chars"] == len(POLICY)
    assert result.data["metadata"] == {"tenant": "acme", "kind": "policy"}
    result.data["metadata"]["tenant"] = "mutated"

    again = server.call("get_source", {"source_id": "policy-a"})
    assert again.data["metadata"] == {"tenant": "acme", "kind": "policy"}
    handbook = server.call("get_source", {"source_id": "handbook-1"})
    assert handbook.data["text"] == HANDBOOK
    assert handbook.data["metadata"] == {}

    missing = MCPServer(retriever=Retriever(ingester))
    no_ingester = missing.call("get_source", {"source_id": "policy-a"})
    assert no_ingester.ok is False
    assert "ingester" in no_ingester.data["error"]

    unknown = server.call("get_source", {"source_id": "missing"})
    assert unknown.ok is False
    assert "unknown source" in unknown.data["error"]
    blank = server.call("get_source", {"source_id": "  "})
    assert blank.ok is False
    assert "source_id" in blank.data["error"]
    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()


def test_list_drafts_returns_recorded_drafts_without_executing() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme"})
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("list_drafts")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"drafts": []}

    filed = server.call("draft_action", {"action": "file_expense", "target": "report-42"})
    removed = server.call("draft_remove_source", {"source_id": "policy-a"})
    assert filed.ok and removed.ok

    listed = server.call("list_drafts")
    assert listed.ok is True
    assert listed.draft is False
    assert listed.data["drafts"] == [
        {
            "draft_id": "draft-1",
            "action": "file_expense",
            "target": "report-42",
            "status": "pending",
            "executed": False,
        },
        {
            "draft_id": "draft-2",
            "action": "remove_source",
            "target": "policy-a",
            "status": "pending",
            "executed": False,
        },
    ]
    listed.data["drafts"][0]["target"] = "mutated"
    again = server.call("list_drafts")
    assert again.data["drafts"][0]["target"] == "report-42"

    bad = server.call("list_drafts", {"status": "pending"})
    assert bad.ok is False
    assert bad.draft is False
    assert "no arguments" in bad.data["error"]
    assert executed == []
    assert server.executed == []
    assert [document.source_id for document in ingester.documents] == ["policy-a"]
    assert len(server.drafts()) == 2


def test_draft_remove_source_stays_pending_and_keeps_the_corpus() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme"})
    ingester.add("handbook-1", HANDBOOK)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    result = server.call("draft_remove_source", {"source_id": " policy-a "})
    assert result.ok is True
    assert result.draft is True
    assert result.data["status"] == "pending"
    assert result.data["executed"] is False
    assert result.data["action"] == "remove_source"
    assert result.data["target"] == "policy-a"
    assert result.data["draft_id"] == "draft-1"
    assert [document.source_id for document in ingester.documents] == ["policy-a", "handbook-1"]
    assert any(chunk.source_id == "policy-a" for chunk in ingester.chunks)
    assert server.drafts() == (
        {
            "draft_id": "draft-1",
            "action": "remove_source",
            "target": "policy-a",
            "status": "pending",
        },
    )

    missing = MCPServer(retriever=Retriever(ingester))
    no_ingester = missing.call("draft_remove_source", {"source_id": "policy-a"})
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    unknown = server.call("draft_remove_source", {"source_id": "missing"})
    assert unknown.ok is False
    assert "unknown source" in unknown.data["error"]
    blank = server.call("draft_remove_source", {"source_id": "  "})
    assert blank.ok is False
    assert "source_id" in blank.data["error"]
    assert executed == []
    assert server.executed == []
    assert len(server.drafts()) == 1
    assert [document.source_id for document in ingester.documents] == ["policy-a", "handbook-1"]


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
