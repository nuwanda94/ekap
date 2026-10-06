"""In-process MCP-style tool server. No network or real protocol transport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ekap.rag import Document, Ingester, Retriever

Executor = Callable[[dict[str, Any]], Any]
_DRAFT_STATUSES = frozenset({"pending", "cancelled"})


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
        ingester: Ingester | None = None,
    ) -> None:
        self._retriever = retriever
        self._executor = executor
        self._ingester = ingester
        self._drafts: list[_ActionDraft] = []
        self.executed: list[dict[str, Any]] = []

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec("health", "Liveness check for the tool server.", mutates=False),
            ToolSpec(
                "search_docs",
                "Search ingested documents and return ranked citations above an optional min_score.",
                mutates=False,
            ),
            ToolSpec(
                "list_sources",
                "List ingested sources, optionally filtered by metadata.",
                mutates=False,
            ),
            ToolSpec(
                "get_source",
                "Fetch one ingested source by id, including text and metadata.",
                mutates=False,
            ),
            ToolSpec(
                "summarize_sources",
                "Count ingested sources and chunks, optionally by metadata. Does not change them.",
                mutates=False,
            ),
            ToolSpec(
                "list_chunks",
                "List ingested chunks, optionally by source or metadata. Does not change them.",
                mutates=False,
            ),
            ToolSpec(
                "list_drafts",
                "List recorded action drafts, optionally by status. Does not execute them.",
                mutates=False,
            ),
            ToolSpec(
                "get_draft",
                "Fetch one recorded action draft by id. Does not execute it.",
                mutates=False,
            ),
            ToolSpec(
                "cancel_draft",
                "Cancel one pending action draft. Does not execute it.",
                mutates=False,
            ),
            ToolSpec(
                "summarize_drafts",
                "Count recorded action drafts by status. Does not execute them.",
                mutates=False,
            ),
            ToolSpec(
                "describe_tool",
                "Describe one advertised tool. Does not call it.",
                mutates=False,
            ),
            ToolSpec(
                "list_tools",
                "List advertised tools, optionally by mutates. Does not call them.",
                mutates=False,
            ),
            ToolSpec(
                "draft_action",
                "Draft a mutating action. Does not execute it.",
                mutates=True,
            ),
            ToolSpec(
                "draft_remove_source",
                "Draft withdrawal of one ingested source. Does not remove it.",
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
        if name == "list_sources":
            return self._list_sources(args)
        if name == "get_source":
            return self._get_source(args)
        if name == "summarize_sources":
            return self._summarize_sources(args)
        if name == "list_chunks":
            return self._list_chunks(args)
        if name == "list_drafts":
            return self._list_drafts(args)
        if name == "get_draft":
            return self._get_draft(args)
        if name == "cancel_draft":
            return self._cancel_draft(args)
        if name == "summarize_drafts":
            return self._summarize_drafts(args)
        if name == "describe_tool":
            return self._describe_tool(args)
        if name == "list_tools":
            return self._list_tools_catalog(args)
        if name == "draft_action":
            return self._draft_action(args)
        if name == "draft_remove_source":
            return self._draft_remove_source(args)
        return ToolResult(name=name, ok=False, data={"error": f"unknown tool: {name}"})
