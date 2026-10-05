"""Unit tests for the planning orchestrator."""

import pytest

from ekap.agents import (
    ApprovalGate,
    DraftAnswer,
    Orchestrator,
    Plan,
    RunTrace,
    queue_mutation_drafts,
)
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
    assert next(step.kind for step in plan.steps) == "retrieve"
    assert plan.steps[-1].kind == "synthesize"
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
    assert "top=handbook-1" in draft.trace.events[1].detail
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


def test_queue_mutation_drafts_stays_pending_until_approve() -> None:
    retriever = _retriever()
    ledger: list[str] = []
    server = MCPServer(retriever=retriever, executor=lambda body: ledger.append("server"))
    draft = Orchestrator(retriever, tools=server).run("Please file the expense report")
    gate = ApprovalGate(executor=lambda body: ledger.append(body["action"]))

    queued = queue_mutation_drafts(draft, gate)

    assert len(queued) == 1
    assert queued[0].status == "pending"
    assert queued[0].action == "requested_action"
    assert queued[0].payload["draft_id"] == "draft-1"
    assert gate.executed == []
    assert ledger == []
    assert server.executed == []
    assert [item.status for item in gate.decisions()] == ["pending"]

    gate.approve(queued[0].action_id)
    assert ledger == ["requested_action"]
    assert gate.get(queued[0].action_id).status == "executed"


def test_queue_mutation_drafts_ignores_search_only_runs() -> None:
    retriever = _retriever()
    server = MCPServer(retriever=retriever)
    draft = Orchestrator(retriever, tools=server).run("How fast does vacation accrue?")
    gate = ApprovalGate(executor=lambda body: body)

    assert queue_mutation_drafts(draft, gate) == ()
    assert gate.decisions() == ()
    assert gate.executed == []
