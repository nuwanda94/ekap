"""Unit tests for the in-process MCP tool server."""

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
    assert "list_chunks" in result.data["tools"]
    assert "get_chunk" in result.data["tools"]
    assert "get_chunk_context" in result.data["tools"]
    assert "find_chunks" in result.data["tools"]
    assert "count_phrase" in result.data["tools"]
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


def test_search_docs_drops_hits_below_min_score() -> None:
    server, executed = _server()
    unfiltered = server.call("search_docs", {"query": "unused vacation", "top_k": 3})
    assert unfiltered.ok is True
    assert unfiltered.draft is False
    assert unfiltered.data["min_score"] == 0.0
    assert any(item["source_id"] == "handbook-1" for item in unfiltered.data["citations"])

    filtered = server.call(
        "search_docs",
        {"query": "unused vacation", "top_k": 3, "min_score": 1.0},
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["min_score"] == 1.0
    assert filtered.data["citations"]
    assert all(item["score"] >= 1.0 for item in filtered.data["citations"])
    assert all(item["source_id"] == "handbook-1" for item in filtered.data["citations"])

    weak = server.call(
        "search_docs",
        {"query": "unused vacation manager portal", "top_k": 3, "min_score": 0.75},
    )
    assert weak.ok is True
    assert weak.data["citations"] == []

    bad_type = server.call("search_docs", {"query": "vacation", "min_score": True})
    assert bad_type.ok is False
    assert bad_type.draft is False
    assert "min_score" in bad_type.data["error"]

    bad_range = server.call("search_docs", {"query": "vacation", "min_score": 1.5})
    assert bad_range.ok is False
    assert bad_range.draft is False
    assert "between 0 and 1" in bad_range.data["error"]

    sources = server.call("list_sources")
    assert sources.ok is True
    assert [item["source_id"] for item in sources.data["sources"]] == ["policy-1", "handbook-1"]
    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()


def test_list_chunks_lists_slices_without_changing_the_corpus() -> None:
    ingester = Ingester(chunk_size=40, chunk_overlap=0)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("list_chunks")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"chunks": [], "total": 0, "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("list_chunks")
    assert result.ok is True
    assert result.draft is False
    assert result.data["total"] == len(stored)
    assert result.data["executed"] is False
    assert result.data["chunks"] == [
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "index": chunk.index,
            "chars": len(chunk.text),
        }
        for chunk in stored
    ]
    assert "text" not in result.data["chunks"][0]

    by_source = server.call("list_chunks", {"source_id": " handbook-1 "})
    assert by_source.ok is True
    assert by_source.draft is False
    assert by_source.data["total"] == sum(chunk.source_id == "handbook-1" for chunk in stored)
    assert all(item["source_id"] == "handbook-1" for item in by_source.data["chunks"])
    assert by_source.data["executed"] is False

    filtered = server.call("list_chunks", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.data["executed"] is False
    assert {item["source_id"] for item in filtered.data["chunks"]} == {"policy-a", "handbook-1"}
    assert filtered.data["total"] == sum(
        chunk.source_id in {"policy-a", "handbook-1"} for chunk in stored
    )

    both = server.call(
        "list_chunks",
        {"source_id": "policy-a", "metadata": {"tenant": "beta"}},
    )
    assert both.ok is True
    assert both.data["chunks"] == []
    assert both.data["total"] == 0

    missing = MCPServer(retriever=Retriever(ingester))
    no_ingester = missing.call("list_chunks")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("list_chunks", {"source_id": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "source_id" in blank.data["error"]

    unknown = server.call("list_chunks", {"source_id": "missing"})
    assert unknown.ok is False
    assert unknown.draft is False
    assert "unknown source" in unknown.data["error"]

    bad = server.call("list_chunks", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("list_chunks", {"source_id": "policy-a", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "source_id" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "list_chunks"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_get_chunk_returns_text_without_changing_the_corpus() -> None:
    ingester = Ingester(chunk_size=40, chunk_overlap=0)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    missing = server.call("get_chunk", {"chunk_id": "policy-a:0"})
    assert missing.ok is False
    assert missing.draft is False
    assert "unknown chunk" in missing.data["error"]

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)
    target = next(chunk for chunk in stored if chunk.source_id == "handbook-1")

    result = server.call("get_chunk", {"chunk_id": f" {target.chunk_id} "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["chunk_id"] == target.chunk_id
    assert result.data["source_id"] == "handbook-1"
    assert result.data["index"] == target.index
    assert result.data["text"] == target.text
    assert result.data["chars"] == len(target.text)
    assert result.data["metadata"] == {"tenant": "acme", "kind": "handbook"}
    assert result.data["executed"] is False
    result.data["metadata"]["tenant"] = "mutated"
    again = server.call("get_chunk", {"chunk_id": target.chunk_id})
    assert again.data["metadata"]["tenant"] == "acme"

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "get_chunk",
        {"chunk_id": target.chunk_id},
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("get_chunk", {"chunk_id": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "chunk_id" in blank.data["error"]

    unknown = server.call("get_chunk", {"chunk_id": "missing:0"})
    assert unknown.ok is False
    assert unknown.draft is False
    assert "unknown chunk" in unknown.data["error"]

    extra = server.call("get_chunk", {"chunk_id": target.chunk_id, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "chunk_id" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "get_chunk"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_get_chunk_context_returns_neighbors_without_changing_the_corpus() -> None:
    ingester = Ingester(chunk_size=40, chunk_overlap=0)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    missing = server.call("get_chunk_context", {"chunk_id": "policy-a:1"})
    assert missing.ok is False
    assert missing.draft is False
    assert "unknown chunk" in missing.data["error"]

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)
    policy = [chunk for chunk in stored if chunk.source_id == "policy-a"]
    assert len(policy) >= 3
    center = policy[1]

    result = server.call("get_chunk_context", {"chunk_id": f" {center.chunk_id} "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["chunk_id"] == center.chunk_id
    assert result.data["source_id"] == "policy-a"
    assert result.data["index"] == center.index
    assert result.data["radius"] == 1
    assert result.data["metadata"] == {"tenant": "acme", "kind": "policy"}
    assert result.data["executed"] is False
    assert result.data["total"] == 3
    assert [item["offset"] for item in result.data["chunks"]] == [-1, 0, 1]
    assert [item["chunk_id"] for item in result.data["chunks"]] == [
        policy[0].chunk_id,
        policy[1].chunk_id,
        policy[2].chunk_id,
    ]
    assert all(item["source_id"] == "policy-a" for item in result.data["chunks"])
    assert result.data["chunks"][1]["text"] == center.text
    result.data["metadata"]["tenant"] = "mutated"
    again = server.call("get_chunk_context", {"chunk_id": center.chunk_id, "radius": 0})
    assert again.ok is True
    assert again.data["metadata"]["tenant"] == "acme"
    assert again.data["total"] == 1
    assert again.data["chunks"][0]["offset"] == 0
    assert again.data["chunks"][0]["text"] == center.text

    edge = server.call("get_chunk_context", {"chunk_id": policy[0].chunk_id, "radius": 1})
    assert edge.ok is True
    assert [item["offset"] for item in edge.data["chunks"]] == [0, 1]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "get_chunk_context",
        {"chunk_id": center.chunk_id},
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("get_chunk_context", {"chunk_id": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "chunk_id" in blank.data["error"]

    unknown = server.call("get_chunk_context", {"chunk_id": "missing:0"})
    assert unknown.ok is False
    assert unknown.draft is False
    assert "unknown chunk" in unknown.data["error"]

    bad_radius = server.call("get_chunk_context", {"chunk_id": center.chunk_id, "radius": -1})
    assert bad_radius.ok is False
    assert bad_radius.draft is False
    assert "radius" in bad_radius.data["error"]
    bad_bool = server.call("get_chunk_context", {"chunk_id": center.chunk_id, "radius": True})
    assert bad_bool.ok is False
    assert "radius" in bad_bool.data["error"]

    extra = server.call("get_chunk_context", {"chunk_id": center.chunk_id, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "chunk_id" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "get_chunk_context"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_find_chunks_matches_a_phrase_without_changing_the_corpus() -> None:
    ingester = Ingester(chunk_size=40, chunk_overlap=0)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("find_chunks", {"phrase": "vacation"})
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data["chunks"] == []
    assert empty.data["total"] == 0
    assert empty.data["executed"] is False

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("find_chunks", {"phrase": " Unused Vacation "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["phrase"] == "Unused Vacation"
    assert result.data["executed"] is False
    assert result.data["total"] >= 1
    assert all(item["source_id"] == "handbook-1" for item in result.data["chunks"])
    assert any("vacation" in item["snippet"].lower() for item in result.data["chunks"])

    limited = server.call("find_chunks", {"phrase": "a", "limit": 1})
    assert limited.ok is True
    assert limited.data["total"] == 1

    scoped = server.call(
        "find_chunks",
        {"phrase": "vacation", "source_id": "policy-a"},
    )
    assert scoped.ok is True
    assert scoped.data["chunks"] == []
    assert scoped.data["total"] == 0

    filtered = server.call(
        "find_chunks",
        {"phrase": "approval", "metadata": {"kind": "handbook"}},
    )
    assert filtered.ok is True
    assert filtered.data["chunks"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "find_chunks",
        {"phrase": "vacation"},
    )
    assert no_ingester.ok is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("find_chunks", {"phrase": "  "})
    assert blank.ok is False
    assert "phrase" in blank.data["error"]

    unknown = server.call("find_chunks", {"phrase": "vacation", "source_id": "missing"})
    assert unknown.ok is False
    assert "unknown source" in unknown.data["error"]

    bad_limit = server.call("find_chunks", {"phrase": "vacation", "limit": 0})
    assert bad_limit.ok is False
    assert "limit" in bad_limit.data["error"]
    bad_bool = server.call("find_chunks", {"phrase": "vacation", "limit": True})
    assert bad_bool.ok is False
    assert "limit" in bad_bool.data["error"]

    extra = server.call("find_chunks", {"phrase": "vacation", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "phrase" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "find_chunks"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored



def test_count_phrase_groups_hits_without_returning_text() -> None:
    ingester = Ingester(chunk_size=40, chunk_overlap=0)
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("count_phrase", {"phrase": "vacation"})
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data["sources"] == []
    assert empty.data["total"] == 0
    assert empty.data["executed"] is False

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("count_phrase", {"phrase": " Unused Vacation "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["phrase"] == "Unused Vacation"
    assert result.data["executed"] is False
    assert result.data["sources"] == [
        {
            "source_id": "handbook-1",
            "matches": sum(
                "unused vacation" in chunk.text.lower() and chunk.source_id == "handbook-1"
                for chunk in stored
            ),
        }
    ]
    assert result.data["total"] == result.data["sources"][0]["matches"]
    assert "text" not in result.data
    assert "snippet" not in result.data["sources"][0]

    scoped = server.call("count_phrase", {"phrase": "approval", "source_id": " policy-a "})
    assert scoped.ok is True
    assert scoped.data["sources"] == [
        {
            "source_id": "policy-a",
            "matches": sum(
                "approval" in chunk.text.lower() and chunk.source_id == "policy-a"
                for chunk in stored
            ),
        }
    ]
    assert scoped.data["total"] == scoped.data["sources"][0]["matches"]

    filtered = server.call("count_phrase", {"phrase": "approval", "metadata": {"tenant": "beta"}})
    assert filtered.ok is True
    assert filtered.data["sources"] == [
        {
            "source_id": "policy-b",
            "matches": sum(
                "approval" in chunk.text.lower() and chunk.source_id == "policy-b"
                for chunk in stored
            ),
        }
    ]
    missed = server.call(
        "count_phrase",
        {"phrase": "vacation", "source_id": "policy-a", "metadata": {"tenant": "acme"}},
    )
    assert missed.ok is True
    assert missed.data["sources"] == []
    assert missed.data["total"] == 0

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("count_phrase", {"phrase": "vacation"})
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("count_phrase", {"phrase": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "phrase" in blank.data["error"]

    bad_source = server.call("count_phrase", {"phrase": "vacation", "source_id": "  "})
    assert bad_source.ok is False
    assert "source_id" in bad_source.data["error"]

    unknown = server.call("count_phrase", {"phrase": "vacation", "source_id": "missing"})
    assert unknown.ok is False
    assert "unknown source" in unknown.data["error"]

    bad_meta = server.call("count_phrase", {"phrase": "vacation", "metadata": {"tenant": 1}})
    assert bad_meta.ok is False
    assert "metadata" in bad_meta.data["error"]

    extra = server.call("count_phrase", {"phrase": "vacation", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "phrase" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "count_phrase"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored