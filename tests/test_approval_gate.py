"""Unit tests for the approval gate."""

import pytest

from ekap.agents import ApprovalGate, DecisionRecord, PendingAction
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


def test_decision_log_records_draft_approve_and_reject() -> None:
    ledger: list[str] = []
    gate = ApprovalGate(executor=lambda body: ledger.append(body["action_id"]))
    filed = gate.draft("file_expense", "travel-report")
    kept = gate.draft("delete_record", "policy-1")
    gate.approve(filed.action_id)
    gate.reject(kept.action_id, reason="  missing sign-off  ")

    decisions = gate.decisions()
    assert [type(item) for item in decisions] == [DecisionRecord] * 4
    assert [(item.action_id, item.status, item.reason) for item in decisions] == [
        ("action-1", "pending", ""),
        ("action-2", "pending", ""),
        ("action-1", "executed", ""),
        ("action-2", "rejected", "missing sign-off"),
    ]
    assert decisions[2].action == "file_expense"
    assert decisions[2].target == "travel-report"
    assert decisions[3].target == "policy-1"
    assert ledger == ["action-1"]
    assert gate.executed[0]["action_id"] == "action-1"

    ungated = ApprovalGate()
    pending = ungated.draft("file", "x")
    with pytest.raises(RuntimeError, match="no executor configured"):
        ungated.approve(pending.action_id)
    assert [(item.status) for item in ungated.decisions()] == ["pending"]


def test_submit_draft_rejects_non_pending_without_recording() -> None:
    ledger: list[str] = []
    gate = ApprovalGate(executor=lambda body: ledger.append(body["action"]))
    server = MCPServer(executor=lambda payload: ledger.append("executed"))
    tool_draft = server.call(
        "draft_action",
        {"action": "file_expense", "target": "travel report"},
    )
    cancelled = server.call("cancel_draft", {"draft_id": tool_draft.data["draft_id"]})
    assert cancelled.ok
    assert cancelled.data["status"] == "cancelled"

    with pytest.raises(ValueError, match="not pending"):
        gate.submit_draft(cancelled.data)
    with pytest.raises(ValueError, match="not pending"):
        gate.submit_draft(
            {
                "action": "file_expense",
                "target": "travel report",
                "status": "executed",
                "executed": True,
            }
        )
    with pytest.raises(TypeError, match="status must be a string"):
        gate.submit_draft({"action": "file_expense", "target": "travel report", "status": 1})

    assert gate.decisions() == ()
    assert gate.executed == []
    assert ledger == []
    assert server.executed == []
    assert server.drafts()[0]["status"] == "cancelled"
