"""Unit tests for tab source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.tab import tab_counts
from ekap.rag import Ingester, Retriever


def test_tab_counts_ignores_spaces() -> None:
    assert tab_counts("plain text") == (0, 0, 0)
    assert tab_counts("  indented") == (0, 0, 0)
    assert tab_counts("\talpha\n\t\tbeta") == (3, 2, 0)
    assert tab_counts("a\tb") == (1, 0, 1)
    assert tab_counts("\tkeep\tmiddle") == (2, 1, 1)
    assert tab_counts("") == (0, 0, 0)
    assert tab_counts(" \tnot indent") == (1, 0, 1)


def test_tab_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("tab_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    indented = "\tkeep\n\t\tthis"
    inline = "alpha\tbeta"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("tab-a", indented, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("tab-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("tab_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "tab-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(indented),
            "tab_count": 3,
            "indent_tab_lines": 2,
            "inline_tab_count": 0,
        },
        {
            "source_id": "tab-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "tab_count": 1,
            "indent_tab_lines": 0,
            "inline_tab_count": 1,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call("tab_sources", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["tab-b"]

    policies = server.call(
        "tab_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("tab_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("tab_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("tab_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "tab_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_tab_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "tab_sources" in result.data["tools"]
