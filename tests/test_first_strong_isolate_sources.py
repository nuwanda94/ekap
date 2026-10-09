"""Unit tests for first-strong-isolate source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.first_strong_isolate import first_strong_isolate_counts
from ekap.rag import Ingester, Retriever


def test_first_strong_isolate_counts_ignores_other_invisible_characters() -> None:
    assert first_strong_isolate_counts("plain text") == (0, 0, False, False)
    assert first_strong_isolate_counts(" line\n\t\f\v\u0085\u2028\u2029") == (
        0,
        0,
        False,
        False,
    )
    assert first_strong_isolate_counts(
        "\ufeff\u200b\u200c\u200d\u200e\u200f\u2060\u061c\u202a\u202b\u202c\u202d\u202e\u2066\u2067"
    ) == (
        0,
        0,
        False,
        False,
    )
    assert first_strong_isolate_counts("\u2068\u2068keep") == (2, 1, True, False)
    assert first_strong_isolate_counts("a\u2068b\u2068\u2068") == (3, 2, False, True)
    assert first_strong_isolate_counts("\u2068") == (1, 1, True, True)
    assert first_strong_isolate_counts("") == (0, 0, False, False)
    assert first_strong_isolate_counts("end\u2068") == (1, 1, False, True)


def test_first_strong_isolate_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("first_strong_isolate_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    hidden = "\u2068\u2068keep this"
    inline = "alpha\u2068beta\u2068\u2068"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("fsi-a", hidden, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("fsi-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("first_strong_isolate_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "fsi-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(hidden),
            "first_strong_isolate_count": 2,
            "run_count": 1,
            "starts_with_first_strong_isolate": True,
            "ends_with_first_strong_isolate": False,
        },
        {
            "source_id": "fsi-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "first_strong_isolate_count": 3,
            "run_count": 2,
            "starts_with_first_strong_isolate": False,
            "ends_with_first_strong_isolate": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call(
        "first_strong_isolate_sources", {"metadata": {"tenant": "acme"}}
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["fsi-b"]

    policies = server.call(
        "first_strong_isolate_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "first_strong_isolate_sources"
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("first_strong_isolate_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "first_strong_isolate_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call(
        "describe_tool", {"name": "first_strong_isolate_sources"}
    )
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_first_strong_isolate_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "first_strong_isolate_sources" in result.data["tools"]
