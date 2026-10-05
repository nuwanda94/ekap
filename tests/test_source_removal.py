"""Approval of a source-removal draft withdraws the corpus; reject does not."""

import pytest

from ekap.agents import ApprovalGate, source_removal_executor
from ekap.mcp_server import MCPServer
from ekap.rag import Ingester, Retriever


def _corpus() -> Ingester:
    ingester = Ingester(chunk_size=80, chunk_overlap=10)
    ingester.add("policy-leave", "Annual leave requires manager approval before booking.")
    ingester.add("policy-travel", "Travel bookings need a cost center and a receipt.")
    return ingester


def test_approve_removes_source_and_drops_chunks() -> None:
    ingester = _corpus()
    retriever = Retriever(ingester)
    server = MCPServer(ingester=ingester)
    drafted = server.call("draft_remove_source", {"source_id": "policy-leave"})
    assert drafted.ok and drafted.draft

    gate = ApprovalGate(executor=source_removal_executor(ingester))
    pending = gate.submit_draft(drafted.data)
    assert pending.status == "pending"
    assert [document.source_id for document in ingester.documents] == [
        "policy-leave",
        "policy-travel",
    ]
    assert server.executed == []

    executed = gate.approve(pending.action_id)
    assert executed.status == "executed"
    assert executed.action == "remove_source"
    assert executed.target == "policy-leave"
    assert [document.source_id for document in ingester.documents] == ["policy-travel"]
    assert all(chunk.source_id != "policy-leave" for chunk in ingester.chunks)
    assert retriever.query("annual leave manager approval") == []
    assert retriever.query("travel cost center receipt")[0].chunk.source_id == "policy-travel"
    assert server.executed == []


def test_reject_leaves_source_and_chunks_in_place() -> None:
    ingester = _corpus()
    retriever = Retriever(ingester)
    gate = ApprovalGate(executor=source_removal_executor(ingester))
    pending = gate.draft("remove_source", "policy-travel")
    rejected = gate.reject(pending.action_id, reason="still in force")
    assert rejected.status == "rejected"
    assert [document.source_id for document in ingester.documents] == [
        "policy-leave",
        "policy-travel",
    ]
    assert retriever.query("travel cost center")[0].chunk.source_id == "policy-travel"
    assert gate.executed == []


def test_executor_refuses_other_actions_without_mutating() -> None:
    ingester = _corpus()
    gate = ApprovalGate(executor=source_removal_executor(ingester))
    pending = gate.draft("delete_record", "policy-leave")
    with pytest.raises(ValueError, match="only handles remove_source"):
        gate.approve(pending.action_id)
    assert gate.get(pending.action_id).status == "pending"
    assert [document.source_id for document in ingester.documents] == [
        "policy-leave",
        "policy-travel",
    ]
    assert gate.executed == []

    with pytest.raises(TypeError, match="Ingester"):
        source_removal_executor(object())  # type: ignore[arg-type]
