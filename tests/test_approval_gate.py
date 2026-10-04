"""Unit tests for the approval gate."""

import pytest

from ekap.agents import ApprovalGate, PendingAction
from ekap.mcp_server import MCPServer


def test_draft_stays_pending_and_does_not_execute() -> None:
    ledger: list[str] = []
    gate = ApprovalGate(executor=lambda body: ledger.append(body["action"]))
    pending = gate.draft("file_expense", "travel-report", {"amount": 640})
    assert isinstance(pending, PendingAction)
    assert pending.status == "pending"
    assert pending.action_id == "action-1"
    assert gate.get(pending.action_id).status == "pending"
    assert gate.executed == []
    assert ledger == []


def test_approve_executes_once_and_records_payload() -> None:
    ledger: list[dict[str, object]] = []
    gate = ApprovalGate(executor=ledger.append)
    pending = gate.draft("book_travel", "SFO", {"traveler": "ada"})
    executed = gate.approve(pending.action_id)
    assert executed.status == "executed"
    assert gate.get(pending.action_id).status == "executed"
    assert len(gate.executed) == 1
    assert gate.executed[0]["action"] == "book_travel"
    assert gate.executed[0]["target"] == "SFO"
    assert gate.executed[0]["payload"] == {"traveler": "ada"}
    assert ledger == gate.executed
    with pytest.raises(ValueError, match="not pending"):
        gate.approve(pending.action_id)
    assert len(ledger) == 1


def test_reject_leaves_external_state_unchanged() -> None:
    ledger: list[str] = ["initial"]
    gate = ApprovalGate(executor=lambda body: ledger.append(body["target"]))
    pending = gate.submit_draft(
        {
            "draft_id": "draft-1",
            "action": "delete_record",
            "target": "policy-1",
            "status": "pending",
            "executed": False,
        }
    )
    rejected = gate.reject(pending.action_id, reason="missing manager sign-off")
    assert rejected.status == "rejected"
    assert rejected.reason == "missing manager sign-off"
    assert gate.get(pending.action_id).status == "rejected"
    assert gate.executed == []
    assert ledger == ["initial"]
    with pytest.raises(ValueError, match="not pending"):
        gate.approve(pending.action_id)
    assert ledger == ["initial"]


def test_mcp_draft_can_be_approved_without_server_executing() -> None:
    calls: list[dict[str, object]] = []
    server = MCPServer(executor=lambda payload: calls.append(payload))
    tool_draft = server.call(
        "draft_action",
        {"action": "file_expense", "target": "travel report"},
    )
    gate = ApprovalGate(executor=lambda body: calls.append(body))
    pending = gate.submit_draft(tool_draft.data)
    assert pending.status == "pending"
    assert server.executed == []
    assert calls == []
    gate.approve(pending.action_id)
    assert server.executed == []
    assert calls[0]["action"] == "file_expense"
    assert calls[0]["payload"]["draft_id"] == "draft-1"


def test_unknown_and_blank_inputs_are_rejected() -> None:
    gate = ApprovalGate(executor=lambda body: body)
    with pytest.raises(ValueError, match="action is required"):
        gate.draft("  ", "target")
    with pytest.raises(KeyError, match="unknown action"):
        gate.approve("action-99")
    ungated = ApprovalGate()
    pending = ungated.draft("file", "x")
    with pytest.raises(RuntimeError, match="no executor configured"):
        ungated.approve(pending.action_id)
