"""Repeated blank-line source listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "repeated_blank_sources"


def repeated_blank_spec() -> ToolSpec:
    """Advertised spec for sources with consecutive blank lines."""
    return ToolSpec(
        _TOOL,
        "List sources with two or more consecutive blank lines. Does not return text.",
        mutates=False,
    )


def repeated_blank_counts(text: str) -> tuple[int, int, int]:
    """Return blank-line count, repeated-run count, and longest blank run.

    A blank line is empty or only ASCII spaces and tabs. CRLF and bare CR are
    line boundaries. The empty segment after a final terminator is not a line.
    A repeated run is two or more consecutive blank lines.
    """
    blank_lines = 0
    repeated_runs = 0
    max_run = 0
    current = 0

    def consider(line: str) -> None:
        nonlocal blank_lines, repeated_runs, max_run, current
        if line.strip(" \t") == "":
            blank_lines += 1
            current += 1
            if current == 2:
                repeated_runs += 1
            if current > max_run:
                max_run = current
            return
        current = 0

    start = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "\r":
            consider(text[start:index])
            index += 2 if index + 1 < length and text[index + 1] == "\n" else 1
            start = index
            continue
        if text[index] == "\n":
            consider(text[start:index])
            index += 1
            start = index
            continue
        index += 1
    if start < length:
        consider(text[start:])
    return blank_lines, repeated_runs, max_run


def repeated_blank_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List matching sources with consecutive blank lines. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "repeated_blank_sources only accepts metadata"},
        )
    metadata = args.get("metadata")
    metadata_error = _metadata_error(metadata)
    if metadata_error is not None:
        return ToolResult(name=_TOOL, ok=False, data={"error": metadata_error})
    if server._ingester is None:
        return ToolResult(name=_TOOL, ok=False, data={"error": "no ingester configured"})
    try:
        documents = server._ingester.find(metadata=metadata)
    except (TypeError, ValueError) as exc:
        return ToolResult(name=_TOOL, ok=False, data={"error": str(exc)})
    flagged = []
    for document in documents:
        blank_lines, repeated_runs, max_run = repeated_blank_counts(document.text)
        if repeated_runs == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "blank_lines": blank_lines,
                "repeated_runs": repeated_runs,
                "max_run": max_run,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={"sources": len(documents), "flagged": flagged, "executed": False},
    )
