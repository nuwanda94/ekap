"""Unit tests for trailing-whitespace source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.trailing_space import trailing_space_counts
from ekap.rag import Ingester, Retriever


def test_trailing_space_counts_edges_and_interior() -> None:
    assert trailing_space_counts("   ") == (1, 0)
    assert trailing_space_counts("keep\n  \n") == (1, 0)
    assert trailing_space_counts("Alpha \t\r\nBeta\t \n") == (2, 2)
    assert trailing_space_counts("interior  spaces\n") == (0, 0)
    assert trailing_space_counts("unterminated ") == (1, 0)
    assert trailing_space_counts("ends clean\n") == (0, 0)


def test_trailing_space_sources_lists_line_end_whitespace_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("trailing_space_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    clean = "Travel must be booked\nthrough the portal.\n"
    spaces = "Vacation rolls over \r\nup to 10 days.  \r\n"
    tabs = "Page one\t\rPage two\t\t"
    mixed = "Alpha \r\nBeta\t\nGamma"
    interior = "Keep the  spaces inside\nbut not at the edge"
    ingester.add("clean-a", clean, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("space-a", spaces, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("plain-b", "No breaks here.", metadata={"tenant": "acme", "kind": "note"})
    ingester.add("tab-a", tabs, metadata={"tenant": "acme", "kind": "note"})
    ingester.add("mix-a", mixed, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("inner-a", interior, metadata={"tenant": "acme", "kind": "policy"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("trailing_space_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 6
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "space-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(spaces),
            "space_lines": 2,
            "tab_lines": 0,
            "mixed": False,
        },
        {
            "source_id": "tab-a",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(tabs),
            "space_lines": 0,
            "tab_lines": 2,
            "mixed": False,
        },
        {
            "source_id": "mix-a",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(mixed),
            "space_lines": 1,
            "tab_lines": 1,
            "mixed": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call("trailing_space_sources", {"metadata": {"tenant": "beta"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["space-a"]

    notes = server.call(
        "trailing_space_sources",
        {"metadata": {"tenant": "acme", "kind": "note"}},
    )
    assert notes.ok is True
    assert [item["source_id"] for item in notes.data["flagged"]] == ["tab-a"]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("trailing_space_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("trailing_space_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "trailing_space_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "trailing_space_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_trailing_space_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "trailing_space_sources" in result.data["tools"]
