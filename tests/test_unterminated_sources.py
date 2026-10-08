"""Unit tests for unterminated source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.unterminated import unterminated_counts
from ekap.rag import Ingester, Retriever


def test_unterminated_counts_ignores_empty_segment_after_terminator() -> None:
    assert unterminated_counts("Keep this line") == (1, 14)
    assert unterminated_counts("Page one\nPage two") == (2, 8)
    assert unterminated_counts("Closed\n") == (1, 6)
    assert unterminated_counts("Alpha\r\nBeta") == (2, 4)
    assert unterminated_counts("Bare\r") == (1, 4)


def test_unterminated_sources_lists_missing_terminator_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("unterminated_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    open_one = "Keep this line"
    open_two = "Page one\nPage two"
    closed = "Travel must be booked\nthrough the portal.\n"
    crlf = "Alpha\r\nBeta\r\n"
    bare = "Bare close\r"
    open_crlf = "Alpha\r\nBeta"
    ingester.add("open-a", open_one, metadata={"tenant": "acme", "kind": "note"})
    ingester.add("closed-a", closed, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("open-b", open_two, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("crlf-a", crlf, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("bare-a", bare, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("open-c", open_crlf, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("unterminated_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 6
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "open-a",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(open_one),
            "lines": 1,
            "last_line_chars": len(open_one),
        },
        {
            "source_id": "open-b",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(open_two),
            "lines": 2,
            "last_line_chars": len("Page two"),
        },
        {
            "source_id": "open-c",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(open_crlf),
            "lines": 2,
            "last_line_chars": len("Beta"),
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[0].metadata["tenant"] == "acme"

    filtered = server.call("unterminated_sources", {"metadata": {"kind": "note"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["open-a", "open-c"]

    policies = server.call(
        "unterminated_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert [item["source_id"] for item in policies.data["flagged"]] == ["open-b"]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("unterminated_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("unterminated_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "unterminated_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "unterminated_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_unterminated_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "unterminated_sources" in result.data["tools"]
    assert "trailing_blank_sources" in result.data["tools"]
