"""Function application listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "function_application_sources"
_FUNCTION_APPLICATION = "\u2062"


def function_application_spec() -> ToolSpec:
    """Advertised spec for sources that contain a function application."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+2062. Does not return text.",
        mutates=False,
    )


def function_application_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return function-application count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+2062. Ordinary whitespace,
    BOM, zero-width spaces, word joiners, zero-width joiners or non-joiners,
    left-to-right or right-to-left marks, Arabic letter marks, left-to-right
    or right-to-left embeddings, pop directional formatting, left-to-right or
    right-to-left overrides, left-to-right, right-to-left, or first strong
    isolates, pop directional isolates, inhibit symmetric swapping, activate
    symmetric swapping, inhibit arabic form shaping, activate arabic form
    shaping, national digit shapes, nominal digit shapes, soft hyphens,
    narrow no-break spaces, thin spaces, hair spaces, punctuation spaces,
    figure spaces, em spaces, en spaces, three-per-em spaces, four-per-em
    spaces, six-per-em spaces, ideographic spaces, medium mathematical
    spaces, Ogham space marks, Mongolian vowel separators, invisible
    separators, invisible times, and invisible plus are not counted.
    """
    total = text.count(_FUNCTION_APPLICATION)
    runs = 0
    in_run = False
    for char in text:
        if char == _FUNCTION_APPLICATION:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_FUNCTION_APPLICATION),
        text.endswith(_FUNCTION_APPLICATION),
    )


def function_application_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+2062. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "function_application_sources only accepts metadata"},
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
        total, runs, starts, ends = function_application_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "function_application_count": total,
                "run_count": runs,
                "starts_with_function_application": starts,
                "ends_with_function_application": ends,
            }
        )
    return ToolResult(
        name=_TOOL,
        ok=True,
        data={
            "sources": len(documents),
            "flagged": flagged,
            "executed": False,
        },
    )
