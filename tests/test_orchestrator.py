"""Unit tests for the planning orchestrator."""

import pytest

from ekap.agents import ApprovalGate, DraftAnswer, Orchestrator, Plan, RunTrace
from ekap.mcp_server import MCPServer
from ekap.rag import Ingester, Retriever

POLICY = (
    "Expense reports over 500 dollars require manager approval. "
    "Travel must be booked through the corporate portal."
)
HANDBOOK = (
    "Vacation accrues at 1.5 days per month. "
    "Unused vacation may roll over up to 10 days."
)


def _retriever() -> Retriever:
    ingester = Ingester()
    ingester.add("policy-1", POLICY)
    ingester.add("handbook-1", HANDBOOK)
    return Retriever(ingester)


def test_plan_shape_includes_retrieve_and_synthesize() -> None:
    orchestrator = Orchestrator(_retriever())
    plan = orchestrator.plan("What is the vacation accrual rate?")
    assert isinstance(plan, Plan)
    assert plan.question == "What is the vacation accrual rate?"
    kinds = [step.kind for step in plan.steps]
    assert kinds[0] == "retrieve"
    assert kinds[-1] == "synthesize"
    assert all(step.step_id and step.detail for step in plan.steps)


def test_run_returns_draft_that_cites_relevant_source() -> None:
    orchestrator = Orchestrator(_retriever())
    draft = orchestrator.run("How fast does vacation accrue?")
    assert isinstance(draft, DraftAnswer)
    assert draft.status == "draft"
    assert draft.plan.steps[-1].kind == "synthesize"
    assert len(draft.citations) >= 1
    assert draft.citations[0].source_id == "handbook-1"
    assert "1.5 days" in draft.citations[0].snippet
    assert "handbook-1" in draft.text
    assert draft.citations[0].snippet in draft.text
    assert draft.pending_action is None


def test_run_uses_search_tool_and_does_not_execute_mutation() -> None:
    retriever = _retriever()
    executed: list[dict[str, str]] = []

    def executor(payload: dict[str, str]) -> dict[str, str]:
        executed.append(payload)
        return payload

    server = MCPServer(retriever=retriever, executor=executor)
    orchestrator = Orchestrator(retriever, tools=server)
    draft = orchestrator.run("Please file the expense report for travel")
    kinds = [step.kind for step in draft.plan.steps]
    assert kinds.count("tool") == 2
    assert any(step.detail.startswith("Draft the requested action") for step in draft.plan.steps)
    assert draft.citations[0].source_id == "policy-1"
    assert "policy-1" in draft.text
    names = [result.name for result in draft.tool_results]
    assert names == ["search_docs", "draft_action"]
    assert draft.tool_results[-1].draft is True
    assert draft.tool_results[-1].data["executed"] is False
    assert draft.pending_action is None
    assert executed == []
    assert server.executed == []


def test_unsupported_question_refuses_without_citations() -> None:
    draft = Orchestrator(_retriever()).run("xyzzy plugh cafeteria menu")
    assert draft.citations == ()
    assert draft.text.startswith("No approved source supports an answer")
    assert "xyzzy plugh cafeteria menu" in draft.text


def test_blank_question_is_rejected() -> None:
    orchestrator = Orchestrator(_retriever())
    with pytest.raises(ValueError, match="question is required"):
        orchestrator.plan("   ")


def test_run_trace_records_plan_retrieve_and_synthesize() -> None:
    draft = Orchestrator(_retriever()).run("  How fast does vacation accrue?  ")
    assert isinstance(draft.trace, RunTrace)
    assert draft.trace.question == "How fast does vacation accrue?"
    stages = [event.stage for event in draft.trace.events]
    assert stages == ["plan", "retrieve", "synthesize"]
    assert all(event.ok for event in draft.trace.events)
    assert draft.trace.events[1].detail == "3 citations; top=handbook-1" or (
        "top=handbook-1" in draft.trace.events[1].detail
    )
    assert "handbook-1" in draft.trace.events[1].detail
    assert draft.trace.events[-1].detail.endswith("citations")


def test_run_trace_records_tool_events_without_execution() -> None:
    retriever = _retriever()
    executed: list[dict[str, str]] = []
    server = MCPServer(retriever=retriever, executor=executed.append)
    draft = Orchestrator(retriever, tools=server).run("Please file the expense report")
    assert draft.trace is not None
    tool_events = [event for event in draft.trace.events if event.stage == "tool"]
    assert [event.detail for event in tool_events] == ["search_docs", "draft_action"]
    assert all(event.ok for event in tool_events)
    assert executed == []


def test_mutation_is_queued_on_gate_and_not_executed() -> None:
    retriever = _retriever()
    ledger: list[dict[str, object]] = []
    gate = ApprovalGate(executor=ledger.append)
    server = MCPServer(retriever=retriever, executor=ledger.append)
    draft = Orchestrator(retriever, tools=server, gate=gate).run(
        "Please file the expense report for travel"
    )
    pending = draft.pending_action
    assert pending is not None
    assert pending.status == "pending"
    assert pending.action == "requested_action"
    assert pending.target == "Please file the expense report for travel"
    assert gate.get(pending.action_id).status == "pending"
    assert gate.executed == []
    assert ledger == []
    assert server.executed == []
    assert draft.trace is not None
    assert draft.trace.events[-2].stage == "gate"
    assert draft.trace.events[-2].detail == f"{pending.action_id} pending"

    gate.approve(pending.action_id)
    assert ledger[0]["action"] == "requested_action"
    assert gate.get(pending.action_id).status == "executed"


def test_non_mutation_does_not_queue_on_gate() -> None:
    retriever = _retriever()
    ledger: list[dict[str, object]] = []
    gate = ApprovalGate(executor=ledger.append)
    server = MCPServer(retriever=retriever, executor=ledger.append)
    draft = Orchestrator(retriever, tools=server, gate=gate).run(
        "How fast does vacation accrue?"
    )
    assert draft.pending_action is None
    assert gate.decisions() == ()
    assert ledger == []
    assert draft.trace is not None
    assert "gate" not in [event.stage for event in draft.trace.events]
