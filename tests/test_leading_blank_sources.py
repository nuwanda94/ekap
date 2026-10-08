"""Unit tests for leading blank-line source listing."""

from ekap.mcp_server import MCPServer
from ekap.mcp_server.leading_blank import leading_blank_counts
from ekap.rag import Ingester, Retriever


def test_leading_blank_counts_blank_only_and_interior() -> None:
    assert leading_blank_counts("\n\t\n") == (2, 0)
    assert leading_blank_counts("Vacation rolls over\n\nup to 10 days.\n") == (0, 2)
    assert leading_blank_counts("Keep this line") == (0, 1)
    assert leading_blank_counts("\nPage one\n") == (1, 1)
    assert leading_blank_counts(" \r\nAlpha\r\n") == (1, 1)


def test_leading_blank_sources_lists_opening_blanks_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("leading_blank_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    clean = "Travel must be booked\nthrough the portal.\n"
    interior = "Vacation rolls over\n\nup to 10 days.\n"
    one_blank = "\nPage one\n"
    spaced = " \r\nAlpha\r\n"
    two_blank = "\n\nPage two\n"
    unterminated = "Keep this line"
    ingester.add("clean-a", clean, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("inner-a", interior, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("plain-b", "No breaks here.", metadata={"tenant": "acme", "kind": "note"})
    ingester.add("lead-a", one_blank, metadata={"tenant": "acme", "kind": "note"})
    ingester.add("space-a", spaced, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("lead-b", two_blank, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("open-a", unterminated, metadata={"tenant": "acme", "kind": "policy"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("leading_blank_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 7
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "lead-a",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(one_blank),
            "leading_blank_lines": 1,
            "content_lines": 1,
            "blank_only": False,
        },
        {
            "source_id": "space-a",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(spaced),
            "leading_blank_lines": 1,
            "content_lines": 1,
            "blank_only": False,
        },
        {
            "source_id": "lead-b",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(two_blank),
            "leading_blank_lines": 2,
            "content_lines": 1,
            "blank_only": False,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[3].metadata["tenant"] == "acme"

    filtered = server.call("leading_blank_sources", {"metadata": {"kind": "note"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["lead-a"]

    policies = server.call(
        "leading_blank_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert [item["source_id"] for item in policies.data["flagged"]] == ["space-a", "lead-b"]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("leading_blank_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("leading_blank_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "leading_blank_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "leading_blank_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_leading_blank_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "leading_blank_sources" in result.data["tools"]
    assert "unterminated_sources" in result.data["tools"]
