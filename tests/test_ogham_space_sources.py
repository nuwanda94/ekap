"""Unit tests for Ogham space mark source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.ogham_space import ogham_space_counts
from ekap.rag import Ingester, Retriever


def test_ogham_space_counts_ignores_other_invisible_characters() -> None:
    assert ogham_space_counts("plain text") == (0, 0, False, False)
    assert ogham_space_counts(" line\n\t\f\v\u0085\u2028\u2029") == (
        0,
        0,
        False,
        False,
    )
    assert ogham_space_counts(
        "\ufeff\u200b\u200c\u200d\u200e\u200f\u2060\u061c"
        "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f"
        "\u00ad\u202f\u2009\u200a\u2008\u2007\u2003\u2002\u2004\u2005\u2006\u3000\u205f"
    ) == (
        0,
        0,
        False,
        False,
    )
    assert ogham_space_counts("\u1680\u1680keep") == (2, 1, True, False)
    assert ogham_space_counts("a\u1680b\u1680\u1680") == (3, 2, False, True)
    assert ogham_space_counts("\u1680") == (1, 1, True, True)
    assert ogham_space_counts("") == (0, 0, False, False)
    assert ogham_space_counts("end\u1680") == (1, 1, False, True)


def test_ogham_space_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("ogham_space_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    hidden = "\u1680\u1680keep this"
    inline = "alpha\u1680beta\u1680\u1680"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("ogham-a", hidden, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("ogham-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("ogham_space_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "ogham-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(hidden),
            "ogham_space_count": 2,
            "run_count": 1,
            "starts_with_ogham_space": True,
            "ends_with_ogham_space": False,
        },
        {
            "source_id": "ogham-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "ogham_space_count": 3,
            "run_count": 2,
            "starts_with_ogham_space": False,
            "ends_with_ogham_space": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call(
        "ogham_space_sources", {"metadata": {"tenant": "acme"}}
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["ogham-b"]

    policies = server.call(
        "ogham_space_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "ogham_space_sources"
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("ogham_space_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "ogham_space_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "ogham_space_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_ogham_space_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "ogham_space_sources" in result.data["tools"]
