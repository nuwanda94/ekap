"""Unit tests for zero-width-non-joiner source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.zero_width_non_joiner import zero_width_non_joiner_counts
from ekap.rag import Ingester, Retriever


def test_zero_width_non_joiner_counts_ignores_other_invisible_characters() -> None:
    assert zero_width_non_joiner_counts("plain text") == (0, 0, False, False)
    assert zero_width_non_joiner_counts(" line\n\t\f\v\u0085\u2028\u2029") == (0, 0, False, False)
    assert zero_width_non_joiner_counts("\ufeff\u200b\u200d\u2060") == (0, 0, False, False)
    assert zero_width_non_joiner_counts("\u200c\u200ckeep") == (2, 1, True, False)
    assert zero_width_non_joiner_counts("a\u200cb\u200c\u200c") == (3, 2, False, True)
    assert zero_width_non_joiner_counts("\u200c") == (1, 1, True, True)
    assert zero_width_non_joiner_counts("") == (0, 0, False, False)
    assert zero_width_non_joiner_counts("end\u200c") == (1, 1, False, True)


def test_zero_width_non_joiner_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("zero_width_non_joiner_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    hidden = "\u200c\u200ckeep this"
    inline = "alpha\u200cbeta\u200c\u200c"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("zwnj-a", hidden, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("zwnj-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("zero_width_non_joiner_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "zwnj-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(hidden),
            "zero_width_non_joiner_count": 2,
            "run_count": 1,
            "starts_with_zero_width_non_joiner": True,
            "ends_with_zero_width_non_joiner": False,
        },
        {
            "source_id": "zwnj-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "zero_width_non_joiner_count": 3,
            "run_count": 2,
            "starts_with_zero_width_non_joiner": False,
            "ends_with_zero_width_non_joiner": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call(
        "zero_width_non_joiner_sources", {"metadata": {"tenant": "acme"}}
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["zwnj-b"]

    policies = server.call(
        "zero_width_non_joiner_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "zero_width_non_joiner_sources"
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("zero_width_non_joiner_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "zero_width_non_joiner_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "zero_width_non_joiner_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_zero_width_non_joiner_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "zero_width_non_joiner_sources" in result.data["tools"]
