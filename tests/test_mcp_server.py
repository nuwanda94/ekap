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
    assert "get_draft" in result.data["tools"]
    assert "cancel_draft" in result.data["tools"]
    assert "summarize_drafts" in result.data["tools"]
    assert "describe_tool" in result.data["tools"]
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

    bad = server.call("list_drafts", {"execute": True})
    assert bad.ok is False
    assert bad.draft is False
    assert "status" in bad.data["error"]
    assert executed == []
    assert server.executed == []
    assert [document.source_id for document in ingester.documents] == ["policy-a"]
    assert len(server.drafts()) == 2


def test_list_drafts_filters_by_status_without_executing() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme"})
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )
    filed = server.call("draft_action", {"action": "file_expense", "target": "report-42"})
    removed = server.call("draft_remove_source", {"source_id": "policy-a"})
    assert filed.ok and removed.ok
    cancelled = server.call("cancel_draft", {"draft_id": "draft-2"})
    assert cancelled.ok

    pending = server.call("list_drafts", {"status": " pending "})
    assert pending.ok is True
    assert pending.draft is False
    assert [item["draft_id"] for item in pending.data["drafts"]] == ["draft-1"]
    assert pending.data["drafts"][0]["status"] == "pending"
    assert pending.data["drafts"][0]["executed"] is False
    pending.data["drafts"][0]["target"] = "mutated"

    cancelled_only = server.call("list_drafts", {"status": "cancelled"})
    assert cancelled_only.ok is True
    assert [item["draft_id"] for item in cancelled_only.data["drafts"]] == ["draft-2"]
    assert cancelled_only.data["drafts"][0]["status"] == "cancelled"
    assert cancelled_only.data["drafts"][0]["target"] == "policy-a"

    blank = server.call("list_drafts", {"status": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "status" in blank.data["error"]
    unknown = server.call("list_drafts", {"status": "executed"})
    assert unknown.ok is False
    assert "status" in unknown.data["error"]
    typed = server.call("list_drafts", {"status": 1})
    assert typed.ok is False
    assert "status" in typed.data["error"]
    extra = server.call("list_drafts", {"status": "pending", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "status" in extra.data["error"]

    assert executed == []
    assert server.executed == []
    assert [document.source_id for document in ingester.documents] == ["policy-a"]
    assert [item["status"] for item in server.drafts()] == ["pending", "cancelled"]


def test_get_draft_returns_one_recorded_draft_without_executing() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme"})
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )
    filed = server.call("draft_action", {"action": "file_expense", "target": "report-42"})
    removed = server.call("draft_remove_source", {"source_id": "policy-a"})
    assert filed.ok and removed.ok

    result = server.call("get_draft", {"draft_id": " draft-2 "})
    assert result.ok is True
    assert result.draft is False
    assert result.data == {
        "draft_id": "draft-2",
        "action": "remove_source",
        "target": "policy-a",
        "status": "pending",
        "executed": False,
    }
    result.data["target"] = "mutated"
    again = server.call("get_draft", {"draft_id": "draft-2"})
    assert again.data["target"] == "policy-a"
    first = server.call("get_draft", {"draft_id": "draft-1"})
    assert first.data["action"] == "file_expense"
    assert first.data["target"] == "report-42"

    unknown = server.call("get_draft", {"draft_id": "draft-9"})
    assert unknown.ok is False
    assert unknown.draft is False
    assert "unknown draft" in unknown.data["error"]
    blank = server.call("get_draft", {"draft_id": "  "})
    assert blank.ok is False
    assert "draft_id" in blank.data["error"]
    missing = server.call("get_draft")
    assert missing.ok is False
    assert "draft_id" in missing.data["error"]
    extra = server.call("get_draft", {"draft_id": "draft-1", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "draft_id" in extra.data["error"]

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


def test_cancel_draft_withdraws_pending_draft_without_executing() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme"})
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )
    filed = server.call("draft_action", {"action": "file_expense", "target": "report-42"})
    removed = server.call("draft_remove_source", {"source_id": "policy-a"})
    assert filed.ok and removed.ok

    result = server.call("cancel_draft", {"draft_id": " draft-2 "})
    assert result.ok is True
    assert result.draft is False
    assert result.data == {
        "draft_id": "draft-2",
        "action": "remove_source",
        "target": "policy-a",
        "status": "cancelled",
        "executed": False,
    }
    result.data["target"] = "mutated"
    again = server.call("get_draft", {"draft_id": "draft-2"})
    assert again.data["status"] == "cancelled"
    assert again.data["target"] == "policy-a"
    listed = server.call("list_drafts")
    assert [item["status"] for item in listed.data["drafts"]] == ["pending", "cancelled"]
    assert server.drafts()[1]["status"] == "cancelled"

    repeat = server.call("cancel_draft", {"draft_id": "draft-2"})
    assert repeat.ok is False
    assert repeat.draft is False
    assert "not pending" in repeat.data["error"]
    unknown = server.call("cancel_draft", {"draft_id": "draft-9"})
    assert unknown.ok is False
    assert "unknown draft" in unknown.data["error"]
    blank = server.call("cancel_draft", {"draft_id": "  "})
    assert blank.ok is False
    assert "draft_id" in blank.data["error"]
    missing = server.call("cancel_draft")
    assert missing.ok is False
    assert "draft_id" in missing.data["error"]
    extra = server.call("cancel_draft", {"draft_id": "draft-1", "execute": True})
    assert extra.ok is False
    assert "draft_id" in extra.data["error"]

    still_pending = server.call("get_draft", {"draft_id": "draft-1"})
    assert still_pending.data["status"] == "pending"
    assert executed == []
    assert server.executed == []
    assert [document.source_id for document in ingester.documents] == ["policy-a"]
    assert len(server.drafts()) == 2


def test_summarize_drafts_counts_statuses_without_executing() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme"})
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("summarize_drafts")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"pending": 0, "cancelled": 0, "total": 0}

    filed = server.call("draft_action", {"action": "file_expense", "target": "report-42"})
    removed = server.call("draft_remove_source", {"source_id": "policy-a"})
    assert filed.ok and removed.ok
    cancelled = server.call("cancel_draft", {"draft_id": "draft-2"})
    assert cancelled.ok

    counted = server.call("summarize_drafts")
    assert counted.ok is True
    assert counted.draft is False
    assert counted.data == {"pending": 1, "cancelled": 1, "total": 2}
    counted.data["pending"] = 0
    again = server.call("summarize_drafts")
    assert again.data == {"pending": 1, "cancelled": 1, "total": 2}

    bad = server.call("summarize_drafts", {"status": "pending"})
    assert bad.ok is False
    assert bad.draft is False
    assert "no arguments" in bad.data["error"]
    assert executed == []
    assert server.executed == []
    assert [document.source_id for document in ingester.documents] == ["policy-a"]
    assert [item["status"] for item in server.drafts()] == ["pending", "cancelled"]


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


def test_describe_tool_returns_spec_without_executing() -> None:
    ingester = Ingester()
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme"})
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )
    filed = server.call("draft_action", {"action": "file_expense", "target": "report-42"})
    assert filed.ok

    health = server.call("describe_tool", {"name": " health "})
    assert health.ok is True
    assert health.draft is False
    assert health.data == {
        "name": "health",
        "description": "Liveness check for the tool server.",
        "mutates": False,
        "executed": False,
    }
    health.data["description"] = "mutated"
    again = server.call("describe_tool", {"name": "health"})
    assert again.data["description"] == "Liveness check for the tool server."

    mutating = server.call("describe_tool", {"name": "draft_action"})
    assert mutating.ok is True
    assert mutating.draft is False
    assert mutating.data["name"] == "draft_action"
    assert mutating.data["mutates"] is True
    assert mutating.data["executed"] is False
    assert "Does not execute" in mutating.data["description"]

    unknown = server.call("describe_tool", {"name": "delete_all"})
    assert unknown.ok is False
    assert unknown.draft is False
    assert "unknown tool" in unknown.data["error"]
    blank = server.call("describe_tool", {"name": "  "})
    assert blank.ok is False
    assert "name" in blank.data["error"]
    missing = server.call("describe_tool")
    assert missing.ok is False
    assert "name" in missing.data["error"]
    typed = server.call("describe_tool", {"name": 1})
    assert typed.ok is False
    assert "name" in typed.data["error"]
    extra = server.call("describe_tool", {"name": "health", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "name" in extra.data["error"]

    assert executed == []
    assert server.executed == []
    assert [document.source_id for document in ingester.documents] == ["policy-a"]
    assert len(server.drafts()) == 1
    assert server.drafts()[0]["status"] == "pending"
