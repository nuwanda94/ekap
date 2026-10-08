"""Unit tests for repeated blank-line source listing."""

from ekap.mcp_server import MCPServer
from ekap.rag import Ingester, Retriever


def test_repeated_blank_sources_lists_consecutive_blanks_without_returning_text() -> None:
    ingester = Ingester()
    executed: list[dict[str, str]] = []
    server = MCPServer(
        retriever=Retriever(ingester),
        executor=executed.append,
        ingester=ingester,
    )

    empty = server.call("repeated_blank_sources")
    assert empty.ok is True
    assert empty.draft is False
    assert empty.data == {"sources": 0, "flagged": [], "executed": False}

    clean = "Travel must be booked\nthrough the portal.\n"
    single = "Vacation rolls over\n\nup to 10 days.\n"
    doubled = "Page one\n\n\nPage two\n"
    spaced = "Alpha\r\n \r\n \r\nBeta"
    split_runs = "One\n\n\nTwo\n\n\n\nThree"
    interior = "Keep the  spaces inside\nbut not a blank line"
    ingester.add("clean-a", clean, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("single-a", single, metadata={"tenant": "beta", "kind": "note"})
    ingester.add("plain-b", "No breaks here.", metadata={"tenant": "acme", "kind": "note"})
    ingester.add("double-a", doubled, metadata={"tenant": "acme", "kind": "note"})
    ingester.add("space-a", spaced, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("runs-a", split_runs, metadata={"tenant": "acme", "kind": "policy"})
    ingester.add("inner-a", interior, metadata={"tenant": "acme", "kind": "policy"})
    before = [document.source_id for document in ingester.documents]
    stored = list(ingester.chunks)

    result = server.call("repeated_blank_sources")
    assert result.ok is True
    assert result.draft is False
    assert result.data["sources"] == 7
    assert result.data["executed"] is False
    assert result.data["flagged"] == [
        {
            "source_id": "double-a",
            "metadata": {"tenant": "acme", "kind": "note"},
            "chars": len(doubled),
            "blank_lines": 2,
            "repeated_runs": 1,
            "max_run": 2,
        },
        {
            "source_id": "space-a",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(spaced),
            "blank_lines": 2,
            "repeated_runs": 1,
            "max_run": 2,
        },
        {
            "source_id": "runs-a",
            "metadata": {"tenant": "acme", "kind": "policy"},
            "chars": len(split_runs),
            "blank_lines": 5,
            "repeated_runs": 2,
            "max_run": 3,
        },
    ]
    assert "text" not in result.data
    assert all("text" not in item for item in result.data["flagged"])
    returned = result.data["flagged"][0]["metadata"]
    returned["tenant"] = "mutated"
    assert ingester.documents[3].metadata["tenant"] == "acme"

    filtered = server.call("repeated_blank_sources", {"metadata": {"kind": "note"}})
    assert filtered.ok is True
    assert filtered.draft is False
    assert [item["source_id"] for item in filtered.data["flagged"]] == ["double-a"]

    policies = server.call(
        "repeated_blank_sources",
        {"metadata": {"tenant": "acme", "kind": "policy"}},
    )
    assert policies.ok is True
    assert [item["source_id"] for item in policies.data["flagged"]] == ["space-a", "runs-a"]

    no_ingester = MCPServer(retriever=Retriever(ingester)).call("repeated_blank_sources")
    assert no_ingester.ok is False
    assert no_ingester.draft is False
    assert no_ingester.data["error"] == "no ingester configured"

    bad = server.call("repeated_blank_sources", {"metadata": {"tenant": 1}})
    assert bad.ok is False
    assert bad.draft is False
    assert "metadata" in bad.data["error"]

    extra = server.call(
        "repeated_blank_sources",
        {"metadata": {"tenant": "acme"}, "execute": True},
    )
    assert extra.ok is False
    assert extra.draft is False
    assert "only accepts metadata" in extra.data["error"]

    advertised = server.call("describe_tool", {"name": "repeated_blank_sources"})
    assert advertised.ok is True
    assert advertised.data["mutates"] is False
    assert advertised.data["executed"] is False

    assert executed == []
    assert server.executed == []
    assert server.drafts() == ()
    assert [document.source_id for document in ingester.documents] == before
    assert list(ingester.chunks) == stored


def test_health_lists_repeated_blank_sources() -> None:
    ingester = Ingester()
    server = MCPServer(retriever=Retriever(ingester), ingester=ingester)
    result = server.call("health")
    assert result.ok is True
    assert "repeated_blank_sources" in result.data["tools"]
    assert "leading_space_sources" in result.data["tools"]
