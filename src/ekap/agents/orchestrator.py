"""Plan, retrieve, and synthesize a cited draft. No network or model calls."""

from __future__ import annotations

from dataclasses import dataclass

from ekap.agents.gate import ApprovalGate, PendingAction
from ekap.mcp_server import MCPServer, ToolResult
from ekap.rag import Citation, Retriever

_ACTION_VERBS = ("file", "submit", "book", "approve", "create", "update", "delete")


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One explicit step in an agent plan."""

    step_id: str
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class Plan:
    """Structured plan for a question. Execution has not started."""

    question: str
    steps: tuple[PlanStep, ...]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One audited stage of an orchestrator run."""

    stage: str
    detail: str
    ok: bool = True


@dataclass(frozen=True, slots=True)
class RunTrace:
    """Ordered audit trail for a single question. Side effects are not included."""

    question: str
    events: tuple[TraceEvent, ...]


@dataclass(frozen=True, slots=True)
class DraftAnswer:
    """Synthesized draft that must carry its citation trail."""

    question: str
    plan: Plan
    text: str
    citations: tuple[Citation, ...]
    tool_results: tuple[ToolResult, ...] = ()
    status: str = "draft"
    trace: RunTrace | None = None
    pending_action: PendingAction | None = None


class Orchestrator:
    """plan → retrieve/tool → synthesize, without executing side effects."""

    def __init__(
        self,
        retriever: Retriever,
        tools: MCPServer | None = None,
        gate: ApprovalGate | None = None,
    ) -> None:
        self._retriever = retriever
        self._tools = tools
        self._gate = gate

    def plan(self, question: str) -> Plan:
        """Build a reviewable plan. Does not retrieve or call tools."""
        cleaned = _require_question(question)
        steps = [
            PlanStep("1", "retrieve", f"Search approved documents for: {cleaned}"),
        ]
        if self._tools is not None:
            steps.append(PlanStep("2", "tool", "Call search_docs and keep ranked citations"))
            if _looks_like_mutation(cleaned):
                steps.append(
                    PlanStep("3", "tool", "Draft the requested action; do not execute it")
                )
        steps.append(
            PlanStep(
                str(len(steps) + 1),
                "synthesize",
                "Draft an answer that cites retrieved sources",
            )
        )
        return Plan(question=cleaned, steps=tuple(steps))

    def run(self, question: str) -> DraftAnswer:
        """Execute the plan and return a draft answer with citations and a trace."""
        planned = self.plan(question)
        events = [TraceEvent("plan", f"{len(planned.steps)} steps")]
        citations = self._retriever.query_with_citations(planned.question, top_k=3)
        top = citations[0].source_id if citations else "none"
        events.append(TraceEvent("retrieve", f"{len(citations)} citations; top={top}"))
        tool_results: list[ToolResult] = []
        pending_action: PendingAction | None = None
        if self._tools is not None:
            search = self._tools.call(
                "search_docs",
                {"query": planned.question, "top_k": 3},
            )
            tool_results.append(search)
            events.append(TraceEvent("tool", search.name, ok=search.ok))
            if search.ok and not citations:
                citations = _citations_from_tool(search)
            if _looks_like_mutation(planned.question):
                drafted = self._tools.call(
                    "draft_action",
                    {
                        "action": "requested_action",
                        "target": planned.question,
                    },
                )
                tool_results.append(drafted)
                events.append(TraceEvent("tool", drafted.name, ok=drafted.ok))
                pending_action = self._queue_draft(drafted, events)
        events.append(TraceEvent("synthesize", f"{len(citations)} citations"))
        trace = RunTrace(question=planned.question, events=tuple(events))
        return DraftAnswer(
            question=planned.question,
            plan=planned,
            text=_synthesize(planned.question, citations),
            citations=tuple(citations),
            tool_results=tuple(tool_results),
            trace=trace,
            pending_action=pending_action,
        )

    def _queue_draft(
        self,
        drafted: ToolResult,
        events: list[TraceEvent],
    ) -> PendingAction | None:
        """Submit a tool draft to the gate. Does not approve or execute it."""
        if self._gate is None or not drafted.ok or not drafted.draft:
            return None
        pending = self._gate.submit_draft(drafted.data)
        events.append(TraceEvent("gate", f"{pending.action_id} pending"))
        return pending


def _require_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required")
    return question.strip()


def _looks_like_mutation(question: str) -> bool:
    tokens = set(question.lower().split())
    return any(verb in tokens for verb in _ACTION_VERBS)


def _citations_from_tool(result: ToolResult) -> list[Citation]:
    raw = result.data.get("citations", [])
    if not isinstance(raw, list):
        return []
    citations: list[Citation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        source_id = item.get("source_id")
        snippet = item.get("snippet")
        chunk_id = item.get("chunk_id")
        score = item.get("score", 0.0)
        if not isinstance(source_id, str) or not isinstance(snippet, str):
            continue
        if not isinstance(chunk_id, str):
            chunk_id = source_id
        if not isinstance(score, int | float):
            score = 0.0
        citations.append(
            Citation(
                source_id=source_id,
                snippet=snippet,
                chunk_id=chunk_id,
                score=float(score),
            )
        )
    return citations


def _synthesize(question: str, citations: list[Citation] | tuple[Citation, ...]) -> str:
    if not citations:
        return f"No approved source supports an answer to: {question}"
    lines = [
        f"Draft answer for: {question}",
        f"Supported by {citations[0].source_id}: {citations[0].snippet}",
        "Citations:",
    ]
    lines.extend(f"- [{citation.source_id}] {citation.snippet}" for citation in citations)
    return "\n".join(lines)
