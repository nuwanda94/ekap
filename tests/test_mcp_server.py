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
    assert "overlap_sources" in result.data["tools"]
    assert "token_stats" in result.data["tools"]
    assert "query_coverage" in result.data["tools"]
    assert "query_gaps" in result.data["tools"]
    assert "exclusive_tokens" in result.data["tools"]
    assert "shared_tokens" in result.data["tools"]
    assert "metadata_facets" in result.data["tools"]
    assert "duplicate_sources" in result.data["tools"]
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

def test_overlap_sources_counts_shared_tokens_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    missing = server.call("overlap_sources", {"left": "policy-1", "right": "handbook-1"})
    assert missing.ok is False
    assert missing.draft is False
    assert "unknown source" in missing.data["error"]

    ingester.add("policy-1", POLICY, metadata={"tenant": "acme"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("overlap_sources", {"left": " policy-1 ", "right": "policy-b"})
    assert result.ok is True
    assert result.draft is False
    assert result.data["left"] == "policy-1"
    assert result.data["right"] == "policy-b"
    assert result.data["executed"] is False
    assert "travel" in result.data["shared"]
    assert "portal" in result.data["shared"]
    assert "manager" not in result.data["shared"]
    assert "director" not in result.data["shared"]
    assert result.data["shared"] == sorted(result.data["shared"])
    assert result.data["shared_count"] == len(result.data["shared"])
    assert result.data["left_only"] >= 1
    assert result.data["right_only"] >= 1
    assert "text" not in result.data
    assert all(token.isalnum() and token == token.lower() for token in result.data["shared"])

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "overlap_sources",
        {"left": "policy-1", "right": "handbook-1"},
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("overlap_sources", {"left": "  ", "right": "handbook-1"})
    assert blank.ok is False
    assert blank.draft is False
    assert "left" in blank.data["error"]

    same = server.call("overlap_sources", {"left": "policy-1", "right": " policy-1 "})
    assert same.ok is False
    assert same.draft is False
    assert "different" in same.data["error"]

    unknown = server.call("overlap_sources", {"left": "policy-1", "right": "missing"})
    assert unknown.ok is False
    assert unknown.draft is False
    assert "unknown source" in unknown.data["error"]

    extra = server.call(
        "overlap_sources",
        {"left": "policy-1", "right": "handbook-1", "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "left" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "overlap_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_token_stats_counts_tokens_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )
    ingester.add("policy-1", POLICY)
    ingester.add("handbook-1", HANDBOOK)
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("token_stats", {"source_id": " handbook-1 "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["source_id"] == "handbook-1"
    assert result.data["executed"] is False
    assert result.data["limit"] == 5
    assert result.data["unique"] >= 1
    assert result.data["tokens"] >= result.data["unique"]
    assert "text" not in result.data
    top = {item["token"]: item["count"] for item in result.data["top"]}
    assert top["vacation"] == 2
    assert top["days"] == 2
    assert all(
        item["token"].isalnum() and item["token"] == item["token"].lower()
        for item in result.data["top"]
    )
    assert result.data["top"] == sorted(
        result.data["top"],
        key=lambda item: (-item["count"], item["token"]),
    )

    limited = server.call("token_stats", {"source_id": "handbook-1", "limit": 1})
    assert limited.ok is True
    assert limited.draft is False
    assert limited.data["limit"] == 1
    assert len(limited.data["top"]) == 1
    assert limited.data["top"][0]["token"] == "days"
    assert limited.data["tokens"] == result.data["tokens"]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "token_stats",
        {"source_id": "handbook-1"},
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("token_stats", {"source_id": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "source_id" in blank.data["error"]

    unknown = server.call("token_stats", {"source_id": "missing"})
    assert unknown.ok is False
    assert unknown.draft is False
    assert "unknown source" in unknown.data["error"]

    bad_limit = server.call("token_stats", {"source_id": "handbook-1", "limit": 0})
    assert bad_limit.ok is False
    assert bad_limit.draft is False
    assert "limit" in bad_limit.data["error"]
    bad_bool = server.call("token_stats", {"source_id": "handbook-1", "limit": True})
    assert bad_bool.ok is False
    assert "limit" in bad_bool.data["error"]

    extra = server.call("token_stats", {"source_id": "handbook-1", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "source_id" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "token_stats"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored



def test_query_coverage_lists_sources_per_token_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )
    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("query_coverage", {"query": " Vacation portal vacation "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["query"] == "Vacation portal vacation"
    assert result.data["tokens"] == ["vacation", "portal"]
    assert result.data["executed"] is False
    assert "text" not in result.data
    assert result.data["coverage"] == [
        {"token": "vacation", "source_ids": ["handbook-1"]},
        {"token": "portal", "source_ids": ["policy-a", "policy-b"]},
    ]
    assert result.data["all"] == []

    both = server.call("query_coverage", {"query": "expense travel"})
    assert both.ok is True
    assert both.draft is False
    assert both.data["all"] == ["policy-a", "policy-b"]
    assert both.data["coverage"][0]["source_ids"] == ["policy-a", "policy-b"]

    filtered = server.call(
        "query_coverage",
        {"query": "expense travel", "metadata": {"tenant": "acme"}},
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["all"] == ["policy-a"]
    assert filtered.data["coverage"] == [
        {"token": "expense", "source_ids": ["policy-a"]},
        {"token": "travel", "source_ids": ["policy-a"]},
    ]
    assert "text" not in filtered.data

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "query_coverage",
        {"query": "vacation"},
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("query_coverage", {"query": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "query" in blank.data["error"]

    punctuation = server.call("query_coverage", {"query": "..."})
    assert punctuation.ok is False
    assert punctuation.draft is False
    assert "token" in punctuation.data["error"]

    bad = server.call("query_coverage", {"query": "vacation", "metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("query_coverage", {"query": "vacation", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "query" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "query_coverage"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_query_gaps_lists_uncovered_tokens_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("query_gaps", {"query": "vacation portal"})
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data["tokens"] == ["vacation", "portal"]
    assert empty.data["missing"] == ["vacation", "portal"]
    assert empty.data["partial"] == []
    assert empty.data["executed"] is False
    assert "text" not in empty.data

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("query_gaps", {"query": "Vacation portal! unicorn"})
    assert result.ok is True
    assert result.draft is False
    assert result.data["query"] == "Vacation portal! unicorn"
    assert result.data["tokens"] == ["vacation", "portal", "unicorn"]
    assert result.data["missing"] == ["unicorn"]
    assert result.data["executed"] is False
    assert result.data["partial"] == [
        {"source_id": "policy-a", "missing": ["vacation", "unicorn"]},
        {"source_id": "policy-b", "missing": ["vacation", "unicorn"]},
        {"source_id": "handbook-1", "missing": ["portal", "unicorn"]},
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["partial"])

    filtered = server.call(
        "query_gaps",
        {"query": "vacation portal", "metadata": {"tenant": "acme"}},
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["missing"] == []
    assert [item["source_id"] for item in filtered.data["partial"]] == [
        "policy-a",
        "handbook-1",
    ]
    assert filtered.data["partial"][0]["missing"] == ["vacation"]
    assert filtered.data["partial"][1]["missing"] == ["portal"]
    assert "text" not in filtered.data

    covered = server.call(
        "query_gaps",
        {"query": "portal", "metadata": {"tenant": "beta"}},
    )
    assert covered.ok is True
    assert covered.data["missing"] == []
    assert covered.data["partial"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "query_gaps",
        {"query": "vacation"},
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("query_gaps", {"query": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "query" in blank.data["error"]

    punctuation = server.call("query_gaps", {"query": "..."})
    assert punctuation.ok is False
    assert punctuation.draft is False
    assert "token" in punctuation.data["error"]

    bad = server.call("query_gaps", {"query": "vacation", "metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("query_gaps", {"query": "vacation", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "query" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "query_gaps"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_exclusive_tokens_lists_unique_tokens_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("exclusive_tokens")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "exclusive": [], "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("exclusive_tokens")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    by_source = {item["source_id"]: item["tokens"] for item in result.data["exclusive"]}
    assert "manager" in by_source["policy-a"]
    assert "director" in by_source["policy-b"]
    assert "beta" in by_source["policy-b"]
    assert "vacation" in by_source["handbook-1"]
    assert "portal" not in by_source["policy-a"]
    assert "approval" not in by_source["policy-a"]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["exclusive"])

    filtered = server.call("exclusive_tokens", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 2
    filtered_tokens = {item["source_id"]: item["tokens"] for item in filtered.data["exclusive"]}
    assert "portal" in filtered_tokens["policy-a"]
    assert "vacation" in filtered_tokens["handbook-1"]
    assert "director" not in {token for tokens in filtered_tokens.values() for token in tokens}

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("exclusive_tokens")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("exclusive_tokens", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("exclusive_tokens", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "exclusive_tokens"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored

def test_shared_tokens_lists_tokens_in_two_or_more_sources() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("shared_tokens")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "shared": [], "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("shared_tokens")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    by_token = {item["token"]: item for item in result.data["shared"]}
    assert by_token["portal"]["source_ids"] == ["policy-a", "policy-b"]
    assert by_token["portal"]["sources"] == 2
    assert "approval" in by_token
    assert "manager" not in by_token
    assert "vacation" not in by_token
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["shared"])
    portal_index = next(
        index for index, item in enumerate(result.data["shared"]) if item["token"] == "portal"
    )
    expense_index = next(
        index for index, item in enumerate(result.data["shared"]) if item["token"] == "expense"
    )
    assert expense_index < portal_index

    alone = server.call("shared_tokens", {"metadata": {"kind": "handbook"}})
    assert alone.ok is True
    assert alone.data["sources"] == 1
    assert alone.data["shared"] == []
    assert alone.data["executed"] is False

    filtered = server.call("shared_tokens", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 2
    filtered_by_token = {item["token"]: item for item in filtered.data["shared"]}
    # Policy and handbook both contain "over"; other acme tokens stay exclusive.
    assert set(filtered_by_token) == {"over"}
    assert filtered_by_token["over"]["source_ids"] == ["policy-a", "handbook-1"]
    assert filtered_by_token["over"]["sources"] == 2
    assert "portal" not in filtered_by_token
    assert "text" not in filtered.data
    assert "text" not in filtered_by_token["over"]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("shared_tokens")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("shared_tokens", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("shared_tokens", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "shared_tokens"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored



def test_metadata_facets_lists_keys_and_values_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("metadata_facets")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "facets": [], "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    ingester.add("note-1", "Internal note.", metadata={})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("metadata_facets")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 4
    assert result.data["executed"] is False
    assert [item["key"] for item in result.data["facets"]] == ["tenant", "kind"]
    by_key = {item["key"]: item for item in result.data["facets"]}
    assert by_key["tenant"]["values"] == [
        {"value": "acme", "source_ids": ["policy-a", "handbook-1"], "sources": 2},
        {"value": "beta", "source_ids": ["policy-b"], "sources": 1},
    ]
    assert by_key["kind"]["values"] == [
        {"value": "policy", "source_ids": ["policy-a", "policy-b"], "sources": 2},
        {"value": "handbook", "source_ids": ["handbook-1"], "sources": 1},
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["facets"])
    assert all("text" not in value for item in result.data["facets"] for value in item["values"])

    filtered = server.call("metadata_facets", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 2
    assert filtered.data["executed"] is False
    filtered_by_key = {item["key"]: item for item in filtered.data["facets"]}
    assert filtered_by_key["tenant"]["values"] == [
        {"value": "acme", "source_ids": ["policy-a", "handbook-1"], "sources": 2},
    ]
    assert filtered_by_key["kind"]["values"] == [
        {"value": "policy", "source_ids": ["policy-a"], "sources": 1},
        {"value": "handbook", "source_ids": ["handbook-1"], "sources": 1},
    ]
    assert "text" not in filtered.data

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("metadata_facets")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("metadata_facets", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("metadata_facets", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "metadata_facets"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_duplicate_sources_groups_identical_text_without_returning_it() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("duplicate_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "duplicates": [], "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-copy", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    ingester.add("handbook-copy", HANDBOOK, metadata={"tenant": "beta", "kind": "handbook"})
    ingester.add("policy-spaced", POLICY + " ", metadata={"tenant": "acme", "kind": "policy"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("duplicate_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 6
    assert result.data["executed"] is False
    assert result.data["duplicates"] == [
        {"source_ids": ["policy-a", "policy-copy"], "sources": 2, "chars": len(POLICY)},
        {"source_ids": ["handbook-1", "handbook-copy"], "sources": 2, "chars": len(HANDBOOK)},
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["duplicates"])

    filtered = server.call("duplicate_sources", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 4
    assert filtered.data["executed"] is False
    assert filtered.data["duplicates"] == [
        {"source_ids": ["policy-a", "policy-copy"], "sources": 2, "chars": len(POLICY)},
    ]
    assert "text" not in filtered.data

    distinct = server.call("duplicate_sources", {"metadata": {"kind": "policy"}})
    assert distinct.ok is True
    assert distinct.draft is False
    assert distinct.data["duplicates"] == [
        {"source_ids": ["policy-a", "policy-copy"], "sources": 2, "chars": len(POLICY)},
    ]
    assert "text" not in distinct.data

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("duplicate_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("duplicate_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("duplicate_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "duplicate_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_missing_metadata_lists_sources_without_key_and_does_not_return_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("missing_metadata", {"key": "owner"})
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"key": "owner", "sources": 0, "missing": [], "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-b", BETA_POLICY, metadata={"tenant": "beta", "owner": "sam"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("missing_metadata", {"key": " owner "})
    assert result.ok is True
    assert result.draft is False
    assert result.data["key"] == "owner"
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["missing"] == [
        {
            "source_id": "policy-a",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(POLICY),
        },
        {
            "source_id": "handbook-1",
            "metadata": {"tenant": "acme", "kind": "handbook"},
            "chars": len(HANDBOOK),
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["missing"])
    result.data["missing"][0]["metadata"]["tenant"] = "mutated"
    assert ingester.documents[0].metadata["tenant"] == "acme"

    filtered = server.call(
        "missing_metadata",
        {"key": "owner", "metadata": {"tenant": "acme"}},
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 2
    assert [item["source_id"] for item in filtered.data["missing"]] == [
        "policy-a",
        "handbook-1",
    ]
    assert "text" not in filtered.data

    present = server.call("missing_metadata", {"key": "tenant"})
    assert present.ok is True
    assert present.data["missing"] == []
    assert present.data["sources"] == 3

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "missing_metadata",
        {"key": "owner"},
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    blank = server.call("missing_metadata", {"key": "  "})
    assert blank.ok is False
    assert blank.draft is False
    assert "key" in blank.data["error"]

    bad = server.call("missing_metadata", {"key": "owner", "metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("missing_metadata", {"key": "owner", "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "missing_metadata"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored



def test_padded_sources_lists_leading_or_trailing_whitespace_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("padded_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "padded": [], "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-pad", "  " + POLICY + "\n", metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("handbook-pad", "\t" + HANDBOOK, metadata={"tenant": "beta", "kind": "handbook"})
    ingester.add("interior", "unused  vacation", metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("padded_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 4
    assert result.data["executed"] is False
    assert result.data["padded"] == [
        {
            "source_id": "policy-pad",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len("  " + POLICY + "\n"),
            "leading": 2,
            "trailing": 1,
        },
        {
            "source_id": "handbook-pad",
            "metadata": {"tenant": "beta", "kind": "handbook"},
            "chars": len("\t" + HANDBOOK),
            "leading": 1,
            "trailing": 0,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["padded"])
    result.data["padded"][0]["metadata"]["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "acme"

    filtered = server.call("padded_sources", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 3
    assert [item["source_id"] for item in filtered.data["padded"]] == ["policy-pad"]
    assert "text" not in filtered.data

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("padded_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("padded_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("padded_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "padded_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored



def test_casefold_sources_groups_case_variants_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("casefold_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "groups": [], "executed": False}

    ingester.add("policy-a", POLICY, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-upper", POLICY.upper(), metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("policy-copy", POLICY, metadata={"tenant": "beta", "kind": "policy"})
    ingester.add("handbook-1", HANDBOOK, metadata={"tenant": "acme", "kind": "handbook"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("casefold_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 4
    assert result.data["executed"] is False
    assert result.data["groups"] == [
        {
            "source_ids": ["policy-a", "policy-upper", "policy-copy"],
            "size": 3,
            "chars": len(POLICY),
            "identical": False,
        }
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["groups"])

    filtered = server.call("casefold_sources", {"metadata": {"tenant": "beta"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert filtered.data["sources"] == 1
    assert filtered.data["groups"] == []

    acme = server.call("casefold_sources", {"metadata": {"tenant": "acme"}})
    assert acme.ok is True
    assert acme.data["groups"] == [
        {
            "source_ids": ["policy-a", "policy-upper"],
            "size": 2,
            "chars": len(POLICY),
            "identical": False,
        }
    ]

    copies = Ingester()
    copies.add("one", POLICY, metadata={"tenant": "acme"})
    copies.add("two", POLICY, metadata={"tenant": "acme"})
    exact = MCPServer(ingester=copies).call("casefold_sources")
    assert exact.ok is True
    assert exact.data["groups"] == [
        {
            "source_ids": ["one", "two"],
            "size": 2,
            "chars": len(POLICY),
            "identical": True,
        }
    ]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("casefold_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert "ingester" in no_ingester.data["error"]

    bad = server.call("casefold_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("casefold_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "casefold_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored
