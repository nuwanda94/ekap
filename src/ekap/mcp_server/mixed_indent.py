"""Mixed space/tab indentation listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "mixed_indent_sources"


def mixed_indent_spec() -> ToolSpec:
    """Advertised spec for sources that mix space and tab indentation."""
    return ToolSpec(
        _TOOL,
        "List sources that indent some lines with spaces and others with tabs. Does not return text.",
        mutates=False,
    )


def mixed_indent_counts(text: str) -> tuple[int, int]:
    """Count lines whose first character is a space and lines whose first is a tab.

    CRLF and bare CR are line boundaries. The empty segment after a final terminator
    is not a line. Only the first character decides the indent style. Interior spaces,
    trailing whitespace, and blank lines do not count.
    """
    space_lines = 0
    tab_lines = 0

    def consider(line: str) -> None:
        nonlocal space_lines, tab_lines
        if not line:
            return
        if line[0] == " ":
            space_lines += 1
        elif line[0] == "\t":
            tab_lines += 1

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
    return space_lines, tab_lines


def mixed_indent_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that mix space and tab indentation. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "mixed_indent_sources only accepts metadata"},
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
        space_lines, tab_lines = mixed_indent_counts(document.text)
        if space_lines == 0 or tab_lines == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "space_lines": space_lines,
                "tab_lines": tab_lines,
                "mixed": True,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={"sources": len(documents), "flagged": flagged, "executed": False},
    )
