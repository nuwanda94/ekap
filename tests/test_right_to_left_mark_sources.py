"""Unit tests for right-to-left-mark source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.right_to_left_mark import right_to_left_mark_counts
from ekap.rag import Ingester, Retriever


def test_right_to_left_mark_counts_ignores_other_invisible_characters() -> None:
    assert right_to_left_mark_counts("plain text") == (0, 0, False, False)
    assert right_to_left_mark_counts(" line\n\t\f\v\u0085\u2028\u2029") == (0, 0, False, False)
    assert right_to_left_mark_counts("\ufeff\u200b\u200c\u200d\u200e\u2060") == (
        0,
        0,
        False,
        False,
    )
    assert right_to_left_mark_counts("\u200f\u200fkeep") == (2, 1, True, False)
    assert right_to_left_mark_counts("a\u200fb\u200f\u200f") == (3, 2, False, True)
    assert right_to_left_mark_counts("\u200f") == (1, 1, True, True)
    assert right_to_left_mark_counts("") == (0, 0, False, False)
    assert right_to_left_mark_counts("end\u200f") == (1, 1, False, True)


def test_right_to_left_mark_sources_lists_matches_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("right_to_left_mark_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    plain = "Policy text.\nordinary spaces"
    hidden = "\u200f\u200fkeep this"
    inline = "alpha\u200fbeta\u200f\u200f"
    ingester.add("plain-a", plain, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("rlm-a", hidden, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("rlm-b", inline, metadata={"tenant": "acme", "kind": "note"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("right_to_left_mark_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 3
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "rlm-a",
            "metadata": {"tenant": "beta", "kind": "note"},
            "chars": len(hidden),
            "right_to_left_mark_count": 2,
            "run_count": 1,
            "starts_with_right_to_left_mark": True,
            "ends_with_right_to_left_mark": False,
        },
        {
            "source_id": "rlm-b",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(inline),
            "right_to_left_mark_count": 3,
            "run_count": 2,
            "starts_with_right_to_left_mark": False,
            "ends_with_right_to_left_mark": True,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[1].metadata["tenant"] == "beta"

    filtered = server.call(
        "right_to_left_mark_sources", {"metadata": {"tenant": "acme"}}
    )
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["rlm-b"]

    policies = server.call(
        "right_to_left_mark_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert policies.data["flagged"] == []

    no_ingester = MCPServer(retriever=Retriever(ingester)).call(
        "right_to_left_mark_sources"
    )
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("right_to_left_mark_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "right_to_left_mark_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "right_to_left_mark_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_right_to_left_mark_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "right_to_left_mark_sources" in result.data["tools"]
