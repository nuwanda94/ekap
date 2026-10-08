"""Unit tests for BOM (U+FEFF) source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.bom import bom_counts
from ekap.rag import Ingester, Retriever


def test_bom_counts_leading_and_interior() -> None:
    assert bom_counts("plain") == (0, False)
    assert bom_counts("\ufeffalpha") == (1, True)
    assert bom_counts("alpha\ufeffbeta\ufeff") == (2, False)
    assert bom_counts("\ufeff\ufeff") == (2, True)
    assert bom_counts("") == (0, False)


def test_bom_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("bom_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    leading = "\ufeffPolicy text."
    interior = "Note\ufeffwith\ufeffmarks"
    plain = "No mark here."
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("lead-a", leading, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("inner-a", interior, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("bom_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "lead-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(leading),
            "bom_count": 1,
            "leading": True,
        },
        {
            "source_id": "inner-a",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(interior),
            "bom_count": 2,
            "leading": False,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call("bom_sources", {"metadata": {"tenant": "beta"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["lead-a"]

    policies = server.call(
        "bom_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("bom_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("bom_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call("bom_sources", {"metadata": {"tenant": "acme"}, "execute": True})
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "bom_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_bom_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "bom_sources" in result.data["tools"]
