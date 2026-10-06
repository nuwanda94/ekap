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
                "Search ingested documents and return ranked citations.",
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
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="search_docs", ok=False, data={"error": metadata_error})
        if self._retriever is None:
            return ToolResult(
                name="search_docs",
                ok=False,
                data={"error": "no retriever configured"},
            )
        try:
            citations = self._retriever.query_with_citations(
                query,
                top_k=top_k,
                metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            return ToolResult(name="search_docs", ok=False, data={"error": str(exc)})
        return ToolResult(
            name="search_docs",
            ok=True,
            data={
                "query": query,
                "citations": [_citation_payload(citation) for citation in citations],
            },
        )

    def _list_sources(self, args: dict[str, Any]) -> ToolResult:
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="list_sources", ok=False, data={"error": metadata_error})
        if self._ingester is None:
            return ToolResult(
                name="list_sources",
                ok=False,
                data={"error": "no ingester configured"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="list_sources", ok=False, data={"error": str(exc)})
        return ToolResult(
            name="list_sources",
            ok=True,
            data={
                "sources": [
                    {
                        "source_id": document.source_id,
                        "metadata": dict(document.metadata),
                        "chars": len(document.text),
                    }
                    for document in documents
                ]
            },
        )

    def _get_source(self, args: dict[str, Any]) -> ToolResult:
        source_id = args.get("source_id", "")
        if not isinstance(source_id, str) or not source_id.strip():
            return ToolResult(name="get_source", ok=False, data={"error": "source_id is required"})
        if self._ingester is None:
            return ToolResult(
                name="get_source",
                ok=False,
                data={"error": "no ingester configured"},
            )
        document = _find_source(self._ingester, source_id.strip())
        if document is None:
            return ToolResult(
                name="get_source",
                ok=False,
                data={"error": f"unknown source: {source_id.strip()}"},
            )
        return ToolResult(
            name="get_source",
            ok=True,
            data={
                "source_id": document.source_id,
                "text": document.text,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
            },
        )

    def _summarize_sources(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"metadata"}
        if extra:
            return ToolResult(
                name="summarize_sources",
                ok=False,
                data={"error": "summarize_sources only accepts metadata"},
            )
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="summarize_sources", ok=False, data={"error": metadata_error})
        if self._ingester is None:
            return ToolResult(
                name="summarize_sources",
                ok=False,
                data={"error": "no ingester configured"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="summarize_sources", ok=False, data={"error": str(exc)})
        source_ids = {document.source_id for document in documents}
        chunks = sum(chunk.source_id in source_ids for chunk in self._ingester.chunks)
        chars = sum(len(document.text) for document in documents)
        return ToolResult(
            name="summarize_sources",
            ok=True,
            data={
                "sources": len(documents),
                "chunks": chunks,
                "chars": chars,
                "executed": False,
            },
        )

    def _list_drafts(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"status"}
        if extra:
            return ToolResult(
                name="list_drafts",
                ok=False,
                data={"error": "list_drafts only accepts status"},
            )
        status = args.get("status")
        cleaned: str | None
        if status is None:
            cleaned = None
        elif not isinstance(status, str):
            return ToolResult(
                name="list_drafts",
                ok=False,
                data={"error": "status must be a string"},
            )
        else:
            cleaned = status.strip()
            if cleaned not in _DRAFT_STATUSES:
                return ToolResult(
                    name="list_drafts",
                    ok=False,
                    data={"error": "status must be pending or cancelled"},
                )
        return ToolResult(
            name="list_drafts",
            ok=True,
            data={
                "drafts": [
                    {
                        "draft_id": draft.draft_id,
                        "action": draft.action,
                        "target": draft.target,
                        "status": draft.status,
                        "executed": False,
                    }
                    for draft in self._drafts
                    if cleaned is None or draft.status == cleaned
                ]
            },
        )

    def _get_draft(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"draft_id"}
        if extra:
            return ToolResult(
                name="get_draft",
                ok=False,
                data={"error": "get_draft only accepts draft_id"},
            )
        draft_id = args.get("draft_id", "")
        if not isinstance(draft_id, str) or not draft_id.strip():
            return ToolResult(name="get_draft", ok=False, data={"error": "draft_id is required"})
        cleaned_id = draft_id.strip()
        for draft in self._drafts:
            if draft.draft_id == cleaned_id:
                return ToolResult(name="get_draft", ok=True, data=_draft_payload(draft))
        return ToolResult(
            name="get_draft",
            ok=False,
            data={"error": f"unknown draft: {cleaned_id}"},
        )

    def _cancel_draft(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"draft_id"}
        if extra:
            return ToolResult(
                name="cancel_draft",
                ok=False,
                data={"error": "cancel_draft only accepts draft_id"},
            )
        draft_id = args.get("draft_id", "")
        if not isinstance(draft_id, str) or not draft_id.strip():
            return ToolResult(
                name="cancel_draft",
                ok=False,
                data={"error": "draft_id is required"},
            )
        cleaned_id = draft_id.strip()
        for index, draft in enumerate(self._drafts):
            if draft.draft_id != cleaned_id:
                continue
            if draft.status != "pending":
                return ToolResult(
                    name="cancel_draft",
                    ok=False,
                    data={"error": f"draft {cleaned_id} is {draft.status}, not pending"},
                )
            cancelled = _ActionDraft(
                draft_id=draft.draft_id,
                action=draft.action,
                target=draft.target,
                status="cancelled",
            )
            self._drafts[index] = cancelled
            return ToolResult(name="cancel_draft", ok=True, data=_draft_payload(cancelled))
        return ToolResult(
            name="cancel_draft",
            ok=False,
            data={"error": f"unknown draft: {cleaned_id}"},
        )

    def _summarize_drafts(self, args: dict[str, Any]) -> ToolResult:
        if args:
            return ToolResult(
                name="summarize_drafts",
                ok=False,
                data={"error": "summarize_drafts takes no arguments"},
            )
        pending = sum(draft.status == "pending" for draft in self._drafts)
        cancelled = sum(draft.status == "cancelled" for draft in self._drafts)
        return ToolResult(
            name="summarize_drafts",
            ok=True,
            data={"pending": pending, "cancelled": cancelled, "total": pending + cancelled},
        )

    def _describe_tool(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"name"}
        if extra:
            return ToolResult(
                name="describe_tool",
                ok=False,
                data={"error": "describe_tool only accepts name"},
            )
        tool_name = args.get("name", "")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return ToolResult(name="describe_tool", ok=False, data={"error": "name is required"})
        cleaned = tool_name.strip()
        for spec in self.list_tools():
            if spec.name == cleaned:
                return ToolResult(
                    name="describe_tool",
                    ok=True,
                    data={
                        "name": spec.name,
                        "description": spec.description,
                        "mutates": spec.mutates,
                        "executed": False,
                    },
                )
        return ToolResult(
            name="describe_tool",
            ok=False,
            data={"error": f"unknown tool: {cleaned}"},
        )

    def _list_tools_catalog(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"mutates"}
        if extra:
            return ToolResult(
                name="list_tools",
                ok=False,
                data={"error": "list_tools only accepts mutates"},
            )
        mutates = args.get("mutates")
        if mutates is not None and not isinstance(mutates, bool):
            return ToolResult(
                name="list_tools",
                ok=False,
                data={"error": "mutates must be a boolean"},
            )
        tools = [
            {
                "name": spec.name,
                "description": spec.description,
                "mutates": spec.mutates,
                "executed": False,
            }
            for spec in self.list_tools()
            if mutates is None or spec.mutates is mutates
        ]
        return ToolResult(
            name="list_tools",
            ok=True,
            data={"tools": tools, "total": len(tools)},
        )

    def _draft_action(self, args: dict[str, Any]) -> ToolResult:
        action = args.get("action", "")
        target = args.get("target", "")
        if not isinstance(action, str) or not action.strip():
            return ToolResult(name="draft_action", ok=False, data={"error": "action is required"})
        if not isinstance(target, str) or not target.strip():
            return ToolResult(name="draft_action", ok=False, data={"error": "target is required"})
        draft = self._record_draft(action.strip(), target.strip())
        return ToolResult(
            name="draft_action",
            ok=True,
            draft=True,
            data=_draft_payload(draft),
        )

    def _draft_remove_source(self, args: dict[str, Any]) -> ToolResult:
        source_id = args.get("source_id", "")
        if not isinstance(source_id, str) or not source_id.strip():
            return ToolResult(
                name="draft_remove_source",
                ok=False,
                data={"error": "source_id is required"},
            )
        cleaned_id = source_id.strip()
        if self._ingester is None:
            return ToolResult(
                name="draft_remove_source",
                ok=False,
                data={"error": "no ingester configured"},
            )
        if _find_source(self._ingester, cleaned_id) is None:
            return ToolResult(
                name="draft_remove_source",
                ok=False,
                data={"error": f"unknown source: {cleaned_id}"},
            )
        draft = self._record_draft("remove_source", cleaned_id)
        return ToolResult(
            name="draft_remove_source",
            ok=True,
            draft=True,
            data=_draft_payload(draft),
        )

    def _record_draft(self, action: str, target: str) -> _ActionDraft:
        draft = _ActionDraft(
            draft_id=f"draft-{len(self._drafts) + 1}",
            action=action,
            target=target,
        )
        self._drafts.append(draft)
        return draft


def _find_source(ingester: Ingester, source_id: str) -> Document | None:
    for document in ingester.documents:
        if document.source_id == source_id:
            return document
    return None


def _draft_payload(draft: _ActionDraft) -> dict[str, Any]:
    return {
        "draft_id": draft.draft_id,
        "action": draft.action,
        "target": draft.target,
        "status": draft.status,
        "executed": False,
    }


def _citation_payload(citation: Any) -> dict[str, Any]:
    """Serialize a citation, copying metadata so callers cannot mutate the source."""
    return {
        "source_id": citation.source_id,
        "snippet": citation.snippet,
        "chunk_id": citation.chunk_id,
        "score": citation.score,
        "metadata": dict(citation.metadata),
    }


def _metadata_error(metadata: object) -> str | None:
    """Return an error when metadata is present but not a string map."""
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return "metadata must be a dict of strings"
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return "metadata must be a dict of strings"
    return None
