"""In-process MCP-style tool server. No network or real protocol transport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ekap.rag import Retriever

Executor = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Callable tool advertised by the server."""

    name: str
    description: str
    mutates: bool


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of an in-process tool call."""

    name: str
    ok: bool
    data: dict[str, Any]
    draft: bool = False


@dataclass(frozen=True, slots=True)
class _ActionDraft:
    draft_id: str
    action: str
    target: str
    status: str = "pending"


class MCPServer:
    """Registry of tools an agent can call without leaving the process.

    Side-effect tools only return drafts. An injected executor is stored so a
    later approval gate can run it; this server never calls it.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        executor: Executor | None = None,
    ) -> None:
        self._retriever = retriever
        self._executor = executor
        self._drafts: list[_ActionDraft] = []
        self.executed: list[dict[str, Any]] = []

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec("health", "Liveness check for the tool server.", mutates=False),
            ToolSpec(
                "search_docs",
                "Search ingested documents and return ranked citations.",
                mutates=False,
            ),
            ToolSpec(
                "draft_action",
                "Draft a mutating action. Does not execute it.",
                mutates=True,
            ),
        )

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Invoke a tool by name. Unknown tools return ok=False."""
        args = dict(arguments or {})
        if name == "health":
            return self._health(args)
        if name == "search_docs":
            return self._search_docs(args)
        if name == "draft_action":
            return self._draft_action(args)
        return ToolResult(name=name, ok=False, data={"error": f"unknown tool: {name}"})

    def drafts(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "draft_id": draft.draft_id,
                "action": draft.action,
                "target": draft.target,
                "status": draft.status,
            }
            for draft in self._drafts
        )

    def _health(self, args: dict[str, Any]) -> ToolResult:
        echo = args.get("echo", "ok")
        if not isinstance(echo, str):
            return ToolResult(name="health", ok=False, data={"error": "echo must be a string"})
        return ToolResult(
            name="health",
            ok=True,
            data={"status": "ok", "echo": echo, "tools": [spec.name for spec in self.list_tools()]},
        )

    def _search_docs(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(name="search_docs", ok=False, data={"error": "query is required"})
        top_k = args.get("top_k", 3)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            return ToolResult(name="search_docs", ok=False, data={"error": "top_k must be >= 1"})
        if self._retriever is None:
            return ToolResult(
                name="search_docs",
                ok=False,
                data={"error": "no retriever configured"},
            )
        citations = self._retriever.query_with_citations(query, top_k=top_k)
        return ToolResult(
            name="search_docs",
            ok=True,
            data={
                "query": query,
                "citations": [
                    {
                        "source_id": citation.source_id,
                        "snippet": citation.snippet,
                        "chunk_id": citation.chunk_id,
                        "score": citation.score,
                    }
                    for citation in citations
                ],
            },
        )

    def _draft_action(self, args: dict[str, Any]) -> ToolResult:
        action = args.get("action", "")
        target = args.get("target", "")
        if not isinstance(action, str) or not action.strip():
            return ToolResult(name="draft_action", ok=False, data={"error": "action is required"})
        if not isinstance(target, str) or not target.strip():
            return ToolResult(name="draft_action", ok=False, data={"error": "target is required"})
        draft = _ActionDraft(
            draft_id=f"draft-{len(self._drafts) + 1}",
            action=action.strip(),
            target=target.strip(),
        )
        self._drafts.append(draft)
        return ToolResult(
            name="draft_action",
            ok=True,
            draft=True,
            data={
                "draft_id": draft.draft_id,
                "action": draft.action,
                "target": draft.target,
                "status": draft.status,
                "executed": False,
            },
        )
