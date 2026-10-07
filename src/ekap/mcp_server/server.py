"""In-process MCP-style tool server. No network or real protocol transport."""

from __future__ import annotations

import re
from collections import Counter
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
                "get_chunk",
                "Fetch one ingested chunk by id, including text. Does not change it.",
                mutates=False,
            ),
            ToolSpec(
                "get_chunk_context",
                "Fetch one chunk and same-source neighbors. Does not change them.",
                mutates=False,
            ),
            ToolSpec(
                "find_chunks",
                "Find chunks containing a literal phrase. Does not change them.",
                mutates=False,
            ),
            ToolSpec(
                "count_phrase",
                "Count chunks containing a literal phrase, grouped by source. Does not return text.",
                mutates=False,
            ),
            ToolSpec(
                "overlap_sources",
                "Compare alphanumeric tokens shared by two sources. Does not return source text.",
                mutates=False,
            ),
            ToolSpec(
                "token_stats",
                "Count alphanumeric tokens in one source. Does not return source text.",
                mutates=False,
            ),
            ToolSpec(
                "query_coverage",
                "List sources covering each query token. Does not return source text.",
                mutates=False,
            ),
            ToolSpec(
                "query_gaps",
                "List query tokens no matching source covers. Does not return source text.",
                mutates=False,
            ),
            ToolSpec(
                "exclusive_tokens",
                "List tokens unique to one matching source. Does not return source text.",
                mutates=False,
            ),
            ToolSpec(
                "shared_tokens",
                "List tokens shared by two or more matching sources. Does not return source text.",
                mutates=False,
            ),
            ToolSpec(
                "metadata_facets",
                "List distinct metadata keys and values across matching sources. Does not return source text.",
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
        if name == "get_chunk":
            return self._get_chunk(args)
        if name == "get_chunk_context":
            return self._get_chunk_context(args)
        if name == "find_chunks":
            return self._find_chunks(args)
        if name == "count_phrase":
            return self._count_phrase(args)
        if name == "overlap_sources":
            return self._overlap_sources(args)
        if name == "token_stats":
            return self._token_stats(args)
        if name == "query_coverage":
            return self._query_coverage(args)
        if name == "query_gaps":
            return self._query_gaps(args)
        if name == "exclusive_tokens":
            return self._exclusive_tokens(args)
        if name == "shared_tokens":
            return self._shared_tokens(args)
        if name == "metadata_facets":
            return self._metadata_facets(args)
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
        min_score = args.get("min_score", 0.0)
        score_error = _min_score_error(min_score)
        if score_error is not None:
            return ToolResult(name="search_docs", ok=False, data={"error": score_error})
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
                min_score=min_score,
            )
        except (TypeError, ValueError) as exc:
            return ToolResult(name="search_docs", ok=False, data={"error": str(exc)})
        return ToolResult(
            name="search_docs",
            ok=True,
            data={
                "query": query,
                "min_score": float(min_score),
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

    def _list_chunks(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"source_id", "metadata"}
        if extra:
            return ToolResult(
                name="list_chunks",
                ok=False,
                data={"error": "list_chunks only accepts source_id and metadata"},
            )
        source_id = args.get("source_id")
        cleaned_id: str | None
        if source_id is None:
            cleaned_id = None
        elif not isinstance(source_id, str) or not source_id.strip():
            return ToolResult(
                name="list_chunks",
                ok=False,
                data={"error": "source_id is required"},
            )
        else:
            cleaned_id = source_id.strip()
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="list_chunks", ok=False, data={"error": metadata_error})
        if self._ingester is None:
            return ToolResult(
                name="list_chunks",
                ok=False,
                data={"error": "no ingester configured"},
            )
        if cleaned_id is not None and _find_source(self._ingester, cleaned_id) is None:
            return ToolResult(
                name="list_chunks",
                ok=False,
                data={"error": f"unknown source: {cleaned_id}"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="list_chunks", ok=False, data={"error": str(exc)})
        source_ids = {document.source_id for document in documents}
        if cleaned_id is not None:
            source_ids &= {cleaned_id}
        chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "index": chunk.index,
                "chars": len(chunk.text),
            }
            for chunk in self._ingester.chunks
            if chunk.source_id in source_ids
        ]
        return ToolResult(
            name="list_chunks",
            ok=True,
            data={"chunks": chunks, "total": len(chunks), "executed": False},
        )


    def _get_chunk(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"chunk_id"}
        if extra:
            return ToolResult(
                name="get_chunk",
                ok=False,
                data={"error": "get_chunk only accepts chunk_id"},
            )
        chunk_id = args.get("chunk_id", "")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            return ToolResult(name="get_chunk", ok=False, data={"error": "chunk_id is required"})
        cleaned_id = chunk_id.strip()
        if self._ingester is None:
            return ToolResult(
                name="get_chunk",
                ok=False,
                data={"error": "no ingester configured"},
            )
        for chunk in self._ingester.chunks:
            if chunk.chunk_id != cleaned_id:
                continue
            document = _find_source(self._ingester, chunk.source_id)
            metadata = dict(document.metadata) if document is not None else {}
            return ToolResult(
                name="get_chunk",
                ok=True,
                data={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "index": chunk.index,
                    "text": chunk.text,
                    "chars": len(chunk.text),
                    "metadata": metadata,
                    "executed": False,
                },
            )
        return ToolResult(
            name="get_chunk",
            ok=False,
            data={"error": f"unknown chunk: {cleaned_id}"},
        )


    def _get_chunk_context(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"chunk_id", "radius"}
        if extra:
            return ToolResult(
                name="get_chunk_context",
                ok=False,
                data={"error": "get_chunk_context only accepts chunk_id and radius"},
            )
        chunk_id = args.get("chunk_id", "")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            return ToolResult(
                name="get_chunk_context",
                ok=False,
                data={"error": "chunk_id is required"},
            )
        cleaned_id = chunk_id.strip()
        radius = args.get("radius", 1)
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            return ToolResult(
                name="get_chunk_context",
                ok=False,
                data={"error": "radius must be an integer >= 0"},
            )
        if self._ingester is None:
            return ToolResult(
                name="get_chunk_context",
                ok=False,
                data={"error": "no ingester configured"},
            )
        center = next(
            (chunk for chunk in self._ingester.chunks if chunk.chunk_id == cleaned_id),
            None,
        )
        if center is None:
            return ToolResult(
                name="get_chunk_context",
                ok=False,
                data={"error": f"unknown chunk: {cleaned_id}"},
            )
        document = _find_source(self._ingester, center.source_id)
        metadata = dict(document.metadata) if document is not None else {}
        neighbors = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "index": chunk.index,
                "offset": chunk.index - center.index,
                "text": chunk.text,
                "chars": len(chunk.text),
            }
            for chunk in self._ingester.chunks
            if chunk.source_id == center.source_id and abs(chunk.index - center.index) <= radius
        ]
        return ToolResult(
            name="get_chunk_context",
            ok=True,
            data={
                "chunk_id": center.chunk_id,
                "source_id": center.source_id,
                "index": center.index,
                "radius": radius,
                "metadata": metadata,
                "chunks": neighbors,
                "total": len(neighbors),
                "executed": False,
            },
        )

    def _find_chunks(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"phrase", "source_id", "metadata", "limit"}
        if extra:
            return ToolResult(
                name="find_chunks",
                ok=False,
                data={"error": "find_chunks only accepts phrase, source_id, metadata, and limit"},
            )
        phrase = args.get("phrase", "")
        if not isinstance(phrase, str) or not phrase.strip():
            return ToolResult(name="find_chunks", ok=False, data={"error": "phrase is required"})
        needle = phrase.strip()
        source_id = args.get("source_id")
        cleaned_id: str | None
        if source_id is None:
            cleaned_id = None
        elif not isinstance(source_id, str) or not source_id.strip():
            return ToolResult(
                name="find_chunks",
                ok=False,
                data={"error": "source_id is required"},
            )
        else:
            cleaned_id = source_id.strip()
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="find_chunks", ok=False, data={"error": metadata_error})
        limit = args.get("limit")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            return ToolResult(
                name="find_chunks",
                ok=False,
                data={"error": "limit must be an integer >= 1"},
            )
        if self._ingester is None:
            return ToolResult(
                name="find_chunks",
                ok=False,
                data={"error": "no ingester configured"},
            )
        if cleaned_id is not None and _find_source(self._ingester, cleaned_id) is None:
            return ToolResult(
                name="find_chunks",
                ok=False,
                data={"error": f"unknown source: {cleaned_id}"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="find_chunks", ok=False, data={"error": str(exc)})
        source_ids = {document.source_id for document in documents}
        if cleaned_id is not None:
            source_ids &= {cleaned_id}
        lowered = needle.lower()
        matches = []
        for chunk in self._ingester.chunks:
            if chunk.source_id not in source_ids:
                continue
            if lowered not in chunk.text.lower():
                continue
            matches.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "index": chunk.index,
                    "chars": len(chunk.text),
                    "snippet": _phrase_snippet(chunk.text, needle),
                }
            )
            if limit is not None and len(matches) >= limit:
                break
        return ToolResult(
            name="find_chunks",
            ok=True,
            data={
                "phrase": needle,
                "chunks": matches,
                "total": len(matches),
                "executed": False,
            },
        )


    def _count_phrase(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"phrase", "source_id", "metadata"}
        if extra:
            return ToolResult(
                name="count_phrase",
                ok=False,
                data={"error": "count_phrase only accepts phrase, source_id, and metadata"},
            )
        phrase = args.get("phrase", "")
        if not isinstance(phrase, str) or not phrase.strip():
            return ToolResult(name="count_phrase", ok=False, data={"error": "phrase is required"})
        needle = phrase.strip()
        source_id = args.get("source_id")
        cleaned_id: str | None
        if source_id is None:
            cleaned_id = None
        elif not isinstance(source_id, str) or not source_id.strip():
            return ToolResult(
                name="count_phrase",
                ok=False,
                data={"error": "source_id is required"},
            )
        else:
            cleaned_id = source_id.strip()
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="count_phrase", ok=False, data={"error": metadata_error})
        if self._ingester is None:
            return ToolResult(
                name="count_phrase",
                ok=False,
                data={"error": "no ingester configured"},
            )
        if cleaned_id is not None and _find_source(self._ingester, cleaned_id) is None:
            return ToolResult(
                name="count_phrase",
                ok=False,
                data={"error": f"unknown source: {cleaned_id}"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="count_phrase", ok=False, data={"error": str(exc)})
        source_ids = {document.source_id for document in documents}
        if cleaned_id is not None:
            source_ids &= {cleaned_id}
        lowered = needle.lower()
        counts: dict[str, int] = {}
        for chunk in self._ingester.chunks:
            if chunk.source_id not in source_ids:
                continue
            if lowered not in chunk.text.lower():
                continue
            counts[chunk.source_id] = counts.get(chunk.source_id, 0) + 1
        sources = [
            {"source_id": document.source_id, "matches": counts[document.source_id]}
            for document in documents
            if document.source_id in counts
        ]
        return ToolResult(
            name="count_phrase",
            ok=True,
            data={
                "phrase": needle,
                "sources": sources,
                "total": sum(counts.values()),
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

    def _overlap_sources(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"left", "right"}
        if extra:
            return ToolResult(
                name="overlap_sources",
                ok=False,
                data={"error": "overlap_sources only accepts left and right"},
            )
        left = args.get("left", "")
        right = args.get("right", "")
        if not isinstance(left, str) or not left.strip():
            return ToolResult(name="overlap_sources", ok=False, data={"error": "left is required"})
        if not isinstance(right, str) or not right.strip():
            return ToolResult(name="overlap_sources", ok=False, data={"error": "right is required"})
        left_id = left.strip()
        right_id = right.strip()
        if left_id == right_id:
            return ToolResult(
                name="overlap_sources",
                ok=False,
                data={"error": "left and right must be different sources"},
            )
        if self._ingester is None:
            return ToolResult(
                name="overlap_sources",
                ok=False,
                data={"error": "no ingester configured"},
            )
        left_doc = _find_source(self._ingester, left_id)
        if left_doc is None:
            return ToolResult(
                name="overlap_sources",
                ok=False,
                data={"error": f"unknown source: {left_id}"},
            )
        right_doc = _find_source(self._ingester, right_id)
        if right_doc is None:
            return ToolResult(
                name="overlap_sources",
                ok=False,
                data={"error": f"unknown source: {right_id}"},
            )
        left_tokens = _alnum_tokens(left_doc.text)
        right_tokens = _alnum_tokens(right_doc.text)
        shared = sorted(left_tokens & right_tokens)
        return ToolResult(
            name="overlap_sources",
            ok=True,
            data={
                "left": left_doc.source_id,
                "right": right_doc.source_id,
                "shared": shared,
                "shared_count": len(shared),
                "left_only": len(left_tokens - right_tokens),
                "right_only": len(right_tokens - left_tokens),
                "executed": False,
            },
        )

    def _token_stats(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"source_id", "limit"}
        if extra:
            return ToolResult(
                name="token_stats",
                ok=False,
                data={"error": "token_stats only accepts source_id and limit"},
            )
        source_id = args.get("source_id", "")
        if not isinstance(source_id, str) or not source_id.strip():
            return ToolResult(
                name="token_stats",
                ok=False,
                data={"error": "source_id is required"},
            )
        limit = args.get("limit", 5)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return ToolResult(
                name="token_stats",
                ok=False,
                data={"error": "limit must be >= 1"},
            )
        if self._ingester is None:
            return ToolResult(
                name="token_stats",
                ok=False,
                data={"error": "no ingester configured"},
            )
        cleaned_id = source_id.strip()
        document = _find_source(self._ingester, cleaned_id)
        if document is None:
            return ToolResult(
                name="token_stats",
                ok=False,
                data={"error": f"unknown source: {cleaned_id}"},
            )
        counts = _alnum_token_counts(document.text)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return ToolResult(
            name="token_stats",
            ok=True,
            data={
                "source_id": document.source_id,
                "tokens": sum(counts.values()),
                "unique": len(counts),
                "limit": limit,
                "top": [{"token": token, "count": count} for token, count in ranked],
                "executed": False,
            },
        )

    def _query_coverage(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"query", "metadata"}
        if extra:
            return ToolResult(
                name="query_coverage",
                ok=False,
                data={"error": "query_coverage only accepts query and metadata"},
            )
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                name="query_coverage",
                ok=False,
                data={"error": "query is required"},
            )
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="query_coverage", ok=False, data={"error": metadata_error})
        tokens = _query_tokens(query)
        if not tokens:
            return ToolResult(
                name="query_coverage",
                ok=False,
                data={"error": "query must contain a token"},
            )
        if self._ingester is None:
            return ToolResult(
                name="query_coverage",
                ok=False,
                data={"error": "no ingester configured"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="query_coverage", ok=False, data={"error": str(exc)})
        token_sets = [(document.source_id, _alnum_tokens(document.text)) for document in documents]
        coverage = [
            {
                "token": token,
                "source_ids": [
                    source_id for source_id, source_tokens in token_sets if token in source_tokens
                ],
            }
            for token in tokens
        ]
        all_ids = [
            source_id
            for source_id, source_tokens in token_sets
            if all(token in source_tokens for token in tokens)
        ]
        return ToolResult(
            name="query_coverage",
            ok=True,
            data={
                "query": query.strip(),
                "tokens": tokens,
                "coverage": coverage,
                "all": all_ids,
                "executed": False,
            },
        )


    def _query_gaps(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"query", "metadata"}
        if extra:
            return ToolResult(
                name="query_gaps",
                ok=False,
                data={"error": "query_gaps only accepts query and metadata"},
            )
        query = args.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                name="query_gaps",
                ok=False,
                data={"error": "query is required"},
            )
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="query_gaps", ok=False, data={"error": metadata_error})
        tokens = _query_tokens(query)
        if not tokens:
            return ToolResult(
                name="query_gaps",
                ok=False,
                data={"error": "query must contain a token"},
            )
        if self._ingester is None:
            return ToolResult(
                name="query_gaps",
                ok=False,
                data={"error": "no ingester configured"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="query_gaps", ok=False, data={"error": str(exc)})
        token_sets = [(document.source_id, _alnum_tokens(document.text)) for document in documents]
        missing = [
            token
            for token in tokens
            if not any(token in source_tokens for _, source_tokens in token_sets)
        ]
        partial = [
            {
                "source_id": source_id,
                "missing": [token for token in tokens if token not in source_tokens],
            }
            for source_id, source_tokens in token_sets
            if any(token not in source_tokens for token in tokens)
        ]
        return ToolResult(
            name="query_gaps",
            ok=True,
            data={
                "query": query.strip(),
                "tokens": tokens,
                "missing": missing,
                "partial": partial,
                "executed": False,
            },
        )


    def _exclusive_tokens(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"metadata"}
        if extra:
            return ToolResult(
                name="exclusive_tokens",
                ok=False,
                data={"error": "exclusive_tokens only accepts metadata"},
            )
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="exclusive_tokens", ok=False, data={"error": metadata_error})
        if self._ingester is None:
            return ToolResult(
                name="exclusive_tokens",
                ok=False,
                data={"error": "no ingester configured"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="exclusive_tokens", ok=False, data={"error": str(exc)})
        ordered = [(document.source_id, _query_tokens(document.text)) for document in documents]
        counts: dict[str, int] = {}
        for _, tokens in ordered:
            for token in set(tokens):
                counts[token] = counts.get(token, 0) + 1
        exclusive = [
            {
                "source_id": source_id,
                "tokens": [token for token in tokens if counts[token] == 1],
            }
            for source_id, tokens in ordered
            if any(counts[token] == 1 for token in tokens)
        ]
        return ToolResult(
            name="exclusive_tokens",
            ok=True,
            data={
                "sources": len(documents),
                "exclusive": exclusive,
                "executed": False,
            },
        )

    def _shared_tokens(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"metadata"}
        if extra:
            return ToolResult(
                name="shared_tokens",
                ok=False,
                data={"error": "shared_tokens only accepts metadata"},
            )
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="shared_tokens", ok=False, data={"error": metadata_error})
        if self._ingester is None:
            return ToolResult(
                name="shared_tokens",
                ok=False,
                data={"error": "no ingester configured"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="shared_tokens", ok=False, data={"error": str(exc)})
        owners: dict[str, list[str]] = {}
        order: list[str] = []
        for document in documents:
            for token in _query_tokens(document.text):
                if token not in owners:
                    owners[token] = []
                    order.append(token)
                if document.source_id not in owners[token]:
                    owners[token].append(document.source_id)
        shared = [
            {
                "token": token,
                "source_ids": owners[token],
                "sources": len(owners[token]),
            }
            for token in order
            if len(owners[token]) >= 2
        ]
        return ToolResult(
            name="shared_tokens",
            ok=True,
            data={
                "sources": len(documents),
                "shared": shared,
                "executed": False,
            },
        )

    def _metadata_facets(self, args: dict[str, Any]) -> ToolResult:
        extra = set(args) - {"metadata"}
        if extra:
            return ToolResult(
                name="metadata_facets",
                ok=False,
                data={"error": "metadata_facets only accepts metadata"},
            )
        metadata = args.get("metadata")
        metadata_error = _metadata_error(metadata)
        if metadata_error is not None:
            return ToolResult(name="metadata_facets", ok=False, data={"error": metadata_error})
        if self._ingester is None:
            return ToolResult(
                name="metadata_facets",
                ok=False,
                data={"error": "no ingester configured"},
            )
        try:
            documents = self._ingester.find(metadata=metadata)
        except (TypeError, ValueError) as exc:
            return ToolResult(name="metadata_facets", ok=False, data={"error": str(exc)})
        key_order: list[str] = []
        value_order: dict[str, list[str]] = {}
        owners: dict[str, dict[str, list[str]]] = {}
        for document in documents:
            for key, value in document.metadata.items():
                if key not in owners:
                    owners[key] = {}
                    value_order[key] = []
                    key_order.append(key)
                if value not in owners[key]:
                    owners[key][value] = []
                    value_order[key].append(value)
                if document.source_id not in owners[key][value]:
                    owners[key][value].append(document.source_id)
        facets = [
            {
                "key": key,
                "values": [
                    {
                        "value": value,
                        "source_ids": owners[key][value],
                        "sources": len(owners[key][value]),
                    }
                    for value in value_order[key]
                ],
            }
            for key in key_order
        ]
        return ToolResult(
            name="metadata_facets",
            ok=True,
            data={
                "sources": len(documents),
                "facets": facets,
                "executed": False,
            },
        )

    def _record_draft(self, action: str, target: str) -> _ActionDraft:
        draft = _ActionDraft(
            draft_id=f"draft-{len(self._drafts) + 1}",
            action=action,
            target=target,
        )
        self._drafts.append(draft)
        return draft




_ALNUM = re.compile(r"[A-Za-z0-9]+")



def _query_tokens(text: str) -> list[str]:
    """Unique lowercase alphanumeric tokens in first-seen order."""
    seen: set[str] = set()
    tokens: list[str] = []
    for match in _ALNUM.finditer(text):
        token = match.group(0).lower()
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _alnum_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens. Punctuation is not part of a token."""
    return {match.group(0).lower() for match in _ALNUM.finditer(text)}


def _alnum_token_counts(text: str) -> Counter[str]:
    """Occurrence counts of lowercase alphanumeric tokens."""
    return Counter(match.group(0).lower() for match in _ALNUM.finditer(text))


def _phrase_snippet(text: str, phrase: str, window: int = 24) -> str:
    """Return a short copy of text around the first case-insensitive phrase hit."""
    start = text.lower().find(phrase.lower())
    if start < 0:
        return text[:window]
    left = max(0, start - window)
    right = min(len(text), start + len(phrase) + window)
    snippet = text[left:right].strip()
    if left > 0:
        snippet = "…" + snippet
    if right < len(text):
        snippet = snippet + "…"
    return snippet


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


def _min_score_error(min_score: object) -> str | None:
    """Return an error when min_score is present but not in 0 to 1."""
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        return "min_score must be a number"
    if min_score < 0 or min_score > 1:
        return "min_score must be between 0 and 1"
    return None


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