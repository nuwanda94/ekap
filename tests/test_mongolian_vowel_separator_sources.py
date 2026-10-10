"""Unit tests for Mongolian vowel separator source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.mongolian_vowel_separator import mongolian_vowel_separator_counts
from ekap.rag import Ingester, Retriever


def test_mongolian_vowel_separator_counts_ignores_other_invisible_characters() -> None:
    assert mongolian_vowel_separator_counts("plain text") == (0, 0, False, False)
    assert mongolian_vowel_separator_counts(" line\n\t\f\v\u0085\u2028\u2029") == (
        0,
        0,
        False,
        False,
    )
    assert mongolian_vowel_separator_counts(
        "\ufeff\u200b\u200c\u200d\u200e\u200f\u2060\u061c"
        "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f"
        "\u00ad\u202f\u2009\u200a\u2008\u2007\u2003\u2002\u2004\u2005\u2006\u3000\u205f\u1680"
    ) == (
        0,
        0,
        False,
        False,
    )
    assert mongolian_vowel_separator_counts("\u180e\u180ekeep") == (2, 1, True, False)
    assert mongolian_vowel_separator_counts("a\u180eb\u180e\u180e") == (3, 2, False, True)
    assert mongolian_vowel_separator_counts("\u180e") == (1, 1, True, True)
    assert mongolian_vowel_separator_counts("") == (0, 0, False, False)
    assert mongolian_vowel_separator_counts("end\u180e") == (1, 1, False, True)


def test_mongolian_vowel_separator_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("mongolian_vowel_separator_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    hidden = "\u180e\u180ekeep this"
    inline = "alpha\u180ebeta\u180e\u180e"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("mvs-a", hidden, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("mvs-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("mongolian_vowel_separator_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "mvs-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(hidden),
            "mongolian_vowel_separator_count": 2,
            "run_count": 1,
            "starts_with_mongolian_vowel_separator": True,
            "ends_with_mongolian_vowel_separator": False,
        },
        {
            "source_id": "mvs-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "mongolian_vowel_separator_count": 3,
            "run_count": 2,
            "starts_with_mongolian_vowel_separator": False,
            "ends_with_mongolian_vowel_separator": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call(
        "mongolian_vowel_separator_sources", {"metadata": {"tenant": "acme"}}
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["mvs-b"]

    policies = server.call(
        "mongolian_vowel_separator_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "mongolian_vowel_separator_sources"
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("mongolian_vowel_separator_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "mongolian_vowel_separator_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "mongolian_vowel_separator_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_mongolian_vowel_separator_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "mongolian_vowel_separator_sources" in result.data["tools"]
