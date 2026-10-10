"""Unit tests for thin space source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.thin_space import thin_space_counts
from ekap.rag import Ingester, Retriever


def test_thin_space_counts_ignores_other_invisible_characters() -> None:
    assert thin_space_counts("plain text") == (0, 0, False, False)
    assert thin_space_counts(" line\n\t\f\v\u0085\u2028\u2029") == (
        0,
        0,
        False,
        False,
    )
    assert thin_space_counts(
        "\ufeff\u200b\u200c\u200d\u200e\u200f\u2060\u061c"
        "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f"
        "\u00ad\u202f"
    ) == (
        0,
        0,
        False,
        False,
    )
    assert thin_space_counts("\u2009\u2009keep") == (2, 1, True, False)
    assert thin_space_counts("a\u2009b\u2009\u2009") == (3, 2, False, True)
    assert thin_space_counts("\u2009") == (1, 1, True, True)
    assert thin_space_counts("") == (0, 0, False, False)
    assert thin_space_counts("end\u2009") == (1, 1, False, True)


def test_thin_space_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("thin_space_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    hidden = "\u2009\u2009keep this"
    inline = "alpha\u2009beta\u2009\u2009"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("thin-a", hidden, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("thin-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("thin_space_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "thin-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(hidden),
            "thin_space_count": 2,
            "run_count": 1,
            "starts_with_thin_space": True,
            "ends_with_thin_space": False,
        },
        {
            "source_id": "thin-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "thin_space_count": 3,
            "run_count": 2,
            "starts_with_thin_space": False,
            "ends_with_thin_space": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call(
        "thin_space_sources", {"metadata": {"tenant": "acme"}}
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["thin-b"]

    policies = server.call(
        "thin_space_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "thin_space_sources"
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("thin_space_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "thin_space_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "thin_space_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_thin_space_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "thin_space_sources" in result.data["tools"]
