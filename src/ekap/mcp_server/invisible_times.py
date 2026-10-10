"""Invisible times listing for the in-process MCP server."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.server import ToolResult, ToolSpec, _metadata_error

_TOOL = "invisible_times_sources"
_INVISIBLE_TIMES = "\u2061"


def invisible_times_spec() -> ToolSpec:
    """Advertised spec for sources that contain an invisible times."""
    return ToolSpec(
        _TOOL,
        "List sources that contain U+2061. Does not return text.",
        mutates=False,
    )


def invisible_times_counts(text: str) -> tuple[int, int, bool, bool]:
    """Return invisible-times count, run count, and start/end flags.

    A run is a maximal consecutive sequence of U+2061. Ordinary whitespace,
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
    spaces, Ogham space marks, Mongolian vowel separators, and invisible
    separators are not counted.
    """
    total = text.count(_INVISIBLE_TIMES)
    runs = 0
    in_run = False
    for char in text:
        if char == _INVISIBLE_TIMES:
            if not in_run:
                runs += 1
            in_run = True
        else:
            in_run = False
    return (
        total,
        runs,
        text.startswith(_INVISIBLE_TIMES),
        text.endswith(_INVISIBLE_TIMES),
    )


def invisible_times_sources(server: Any, args: dict[str, Any]) -> ToolResult:
    """List sources that contain U+2061. Never mutates."""
    extra = set(args) - {"metadata"}
    if extra:
        return ToolResult(
            name=_TOOL,
            ok=False,
            data={"error": "invisible_times_sources only accepts metadata"},
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
        total, runs, starts, ends = invisible_times_counts(document.text)
        if total == 0:
            continue
        flagged.append(
            {
                "source_id": document.source_id,
                "metadata": dict(document.metadata),
                "chars": len(document.text),
                "invisible_times_count": total,
                "run_count": runs,
                "starts_with_invisible_times": starts,
                "ends_with_invisible_times": ends,
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
