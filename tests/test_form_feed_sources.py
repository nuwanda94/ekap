"""Unit tests for form-feed source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.form_feed import form_feed_counts
from ekap.rag import Ingester, Retriever


def test_form_feed_counts_ignores_other_whitespace() -> None:
    assert form_feed_counts("plain text") == (0, 0, False, False)
    assert form_feed_counts(" line\n\t") == (0, 0, False, False)
    assert form_feed_counts("\f\fpage") == (2, 1, True, False)
    assert form_feed_counts("a\fb\f\f") == (3, 2, False, True)
    assert form_feed_counts("\f") == (1, 1, True, True)
    assert form_feed_counts("") == (0, 0, False, False)
    assert form_feed_counts("end\f") == (1, 1, False, True)


def test_form_feed_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("form_feed_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    paged = "\f\fkeep this"
    inline = "alpha\fbeta\f\f"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("ff-a", paged, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("ff-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("form_feed_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "ff-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(paged),
            "form_feed_count": 2,
            "run_count": 1,
            "starts_with_form_feed": True,
            "ends_with_form_feed": False,
        },
        {
            "source_id": "ff-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "form_feed_count": 3,
            "run_count": 2,
            "starts_with_form_feed": False,
            "ends_with_form_feed": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call("form_feed_sources", {"metadata": {"tenant": "acme"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["ff-b"]

    policies = server.call(
        "form_feed_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("form_feed_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("form_feed_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "form_feed_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "form_feed_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_form_feed_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "form_feed_sources" in result.data["tools"]
