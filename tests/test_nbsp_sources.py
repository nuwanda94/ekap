"""Unit tests for non-breaking space source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.nbsp import nbsp_counts
from ekap.rag import Ingester, Retriever


def test_nbsp_counts_ignores_ordinary_spaces() -> None:
    assert nbsp_counts("plain text") == (0, False, False)
    assert nbsp_counts("\u00a0inner \u00a0") == (2, True, True)
    assert nbsp_counts("end\u00a0") == (1, False, True)
    assert nbsp_counts("") == (0, False, False)
    assert nbsp_counts(" \t\n\u00a0mid") == (1, False, False)


def test_nbsp_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("nbsp_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    padded = "\u00a0keep\u00a0this\u00a0"
    interior = "alpha\u00a0beta"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("nbsp-a", padded, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("nbsp-b", interior, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("nbsp_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "nbsp-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(padded),
            "nbsp_count": 3,
            "leading": True,
            "trailing": True,
        },
        {
            "source_id": "nbsp-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(interior),
            "nbsp_count": 1,
            "leading": False,
            "trailing": False,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call("nbsp_sources", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["nbsp-b"]

    policies = server.call(
        "nbsp_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("nbsp_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("nbsp_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("nbsp_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "nbsp_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_nbsp_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "nbsp_sources" in result.data["tools"]
