"""Unit tests for long-line source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.long_line import long_line_counts
from ekap.rag import Ingester, Retriever


def test_long_line_counts_ignores_final_empty_segment() -> None:
    assert long_line_counts("short", 10) == (1, 0, 5)
    assert long_line_counts("short\n" + "x" * 11 + "\n", 10) == (2, 1, 11)
    assert long_line_counts("exact-ten!\n", 10) == (1, 0, 10)
    assert long_line_counts("", 10) == (0, 0, 0)
    assert long_line_counts("ab\r\ncd", 1) == (2, 2, 2)


def test_long_line_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("long_line_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "max_chars": 120, "flagged": [], "executed": False}

    short = "Policy text.\nshort"
    long_text = "ok\n" + ("n" * 12) + "\nend"
    exact = "x" * 8
    ingester.add("short-a", short, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("long-a", long_text, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("exact-a", exact, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("long_line_sources", {"max_chars": 8})
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["max_chars"] == 8
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "short-a",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(short),
            "line_count": 2,
            "long_line_count": 1,
            "longest": len("Policy text."),
            "max_chars": 8,
        },
        {
            "source_id": "long-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(long_text),
            "line_count": 3,
            "long_line_count": 1,
            "longest": 12,
            "max_chars": 8,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[0].metadata["tenant"] == "acme"

    filtered = server.call(
        "long_line_sources",
        {"metadata": {"tenant": "beta"}, "max_chars": 8},
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["long-a"]

    policies = server.call(
        "long_line_sources",
        {"metadata": {"tenant": "acme", "kind": "note"}, "max_chars": 8},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("long_line_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("long_line_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    bad_limit = server.call("long_line_sources", {"max_chars": 0})
    assert bad_limit.ok is False
    assert bad_limit.data["error"] == "max_chars must be a positive integer"
    bad_bool = server.call("long_line_sources", {"max_chars": True})
    assert bad_bool.ok is False
    assert bad_bool.data["error"] == "max_chars must be a positive integer"

    extra = server.call("long_line_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "long_line_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_long_line_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "long_line_sources" in result.data["tools"]
