"""Unit tests for mixed space/tab indentation listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.mixed_indent import mixed_indent_counts
from ekap.rag import Ingester, Retriever


def test_mixed_indent_counts_first_character_only() -> None:
    assert mixed_indent_counts("    alpha\n\tbeta\n") == (1, 1)
    assert mixed_indent_counts(" \talpha\r\n\t beta\r\n") == (1, 1)
    assert mixed_indent_counts("    only spaces\n  more\n") == (2, 0)
    assert mixed_indent_counts("\tonly tabs\n\t\tmore") == (0, 2)
    assert mixed_indent_counts("no indent\n  trailing  \n") == (1, 0)
    assert mixed_indent_counts("\n\nplain\n") == (0, 0)
    assert mixed_indent_counts("interior\t tab") == (0, 0)


def test_mixed_indent_sources_lists_both_styles_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("mixed_indent_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    mixed = "def run():\r\n    return 1\r\n\tif False:\r\n\t    pass\n"
    spaces = "    only spaces\n    still spaces\n"
    tabs = "\tonly tabs\r\tonly tabs again"
    plain = "No indent here.\nSecond line."
    trailing = "value  \nnext\t"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("mix-a", mixed, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("space-a", spaces, metadata={"tenant": "acme", "kind": "note"})
    ingester.add("tab-a", tabs, metadata={"tenant": "acme", "kind": "note"})
    ingester.add("trail-a", trailing, metadata={"tenant": "acme", "kind": "policy"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("mixed_indent_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 5
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "mix-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(mixed),
            "space_lines": 1,
            "tab_lines": 2,
            "mixed": True,
        }
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call("mixed_indent_sources", {"metadata": {"tenant": "beta"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["mix-a"]

    notes = server.call(
        "mixed_indent_sources",
        {"metadata": {"tenant": "acme", "kind": "note"}},
    )
    assert notes.ok is True
    assert notes.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("mixed_indent_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("mixed_indent_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "mixed_indent_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "mixed_indent_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_mixed_indent_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "mixed_indent_sources" in result.data["tools"]
