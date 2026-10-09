"""Unit tests for national-digit-shapes source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.national_digit_shapes import national_digit_shapes_counts
from ekap.rag import Ingester, Retriever


def test_national_digit_shapes_counts_ignores_other_invisible_characters() -> None:
    assert national_digit_shapes_counts("plain text") == (0, 0, False, False)
    assert national_digit_shapes_counts(" line\n\t\f\v\u0085\u2028\u2029") == (
        0,
        0,
        False,
        False,
    )
    assert national_digit_shapes_counts(
        "\ufeff\u200b\u200c\u200d\u200e\u200f\u2060\u061c"
        "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d"
    ) == (
        0,
        0,
        False,
        False,
    )
    assert national_digit_shapes_counts("\u206e\u206ekeep") == (2, 1, True, False)
    assert national_digit_shapes_counts("a\u206eb\u206e\u206e") == (3, 2, False, True)
    assert national_digit_shapes_counts("\u206e") == (1, 1, True, True)
    assert national_digit_shapes_counts("") == (0, 0, False, False)
    assert national_digit_shapes_counts("end\u206e") == (1, 1, False, True)


def test_national_digit_shapes_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("national_digit_shapes_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    hidden = "\u206e\u206ekeep this"
    inline = "alpha\u206ebeta\u206e\u206e"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("nds-a", hidden, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("nds-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("national_digit_shapes_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "nds-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(hidden),
            "national_digit_shapes_count": 2,
            "run_count": 1,
            "starts_with_national_digit_shapes": True,
            "ends_with_national_digit_shapes": False,
        },
        {
            "source_id": "nds-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "national_digit_shapes_count": 3,
            "run_count": 2,
            "starts_with_national_digit_shapes": False,
            "ends_with_national_digit_shapes": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call(
        "national_digit_shapes_sources", {"metadata": {"tenant": "acme"}}
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["nds-b"]

    policies = server.call(
        "national_digit_shapes_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "national_digit_shapes_sources"
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("national_digit_shapes_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "national_digit_shapes_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "national_digit_shapes_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_national_digit_shapes_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "national_digit_shapes_sources" in result.data["tools"]
