"""MCP server and tools."""

from __future__ import annotations

from typing import Any

from ekap.mcp_server.activate_arabic_form_shaping import (
    activate_arabic_form_shaping_sources,
    activate_arabic_form_shaping_spec,
)
from ekap.mcp_server.activate_symmetric_swapping import (
    activate_symmetric_swapping_sources,
    activate_symmetric_swapping_spec,
)
from ekap.mcp_server.arabic_letter_mark import (
    arabic_letter_mark_sources,
    arabic_letter_mark_spec,
)
from ekap.mcp_server.bom import bom_sources, bom_spec
from ekap.mcp_server.crlf import crlf_sources, crlf_spec
from ekap.mcp_server.em_space import em_space_sources, em_space_spec
from ekap.mcp_server.figure_space import (
    figure_space_sources,
    figure_space_spec,
)
from ekap.mcp_server.first_strong_isolate import (
    first_strong_isolate_sources,
    first_strong_isolate_spec,
)
from ekap.mcp_server.form_feed import form_feed_sources, form_feed_spec
from ekap.mcp_server.hair_space import hair_space_sources, hair_space_spec
from ekap.mcp_server.inhibit_arabic_form_shaping import (
    inhibit_arabic_form_shaping_sources,
    inhibit_arabic_form_shaping_spec,
)
from ekap.mcp_server.inhibit_symmetric_swapping import (
    inhibit_symmetric_swapping_sources,
    inhibit_symmetric_swapping_spec,
)
from ekap.mcp_server.leading_blank import leading_blank_sources, leading_blank_spec
from ekap.mcp_server.leading_space import leading_space_sources, leading_space_spec
from ekap.mcp_server.left_to_right_embedding import (
    left_to_right_embedding_sources,
    left_to_right_embedding_spec,
)
from ekap.mcp_server.left_to_right_isolate import (
    left_to_right_isolate_sources,
    left_to_right_isolate_spec,
)
from ekap.mcp_server.left_to_right_mark import (
    left_to_right_mark_sources,
    left_to_right_mark_spec,
)
from ekap.mcp_server.left_to_right_override import (
    left_to_right_override_sources,
    left_to_right_override_spec,
)
from ekap.mcp_server.line_separator import line_separator_sources, line_separator_spec
from ekap.mcp_server.long_line import long_line_sources, long_line_spec
from ekap.mcp_server.mixed_indent import mixed_indent_sources, mixed_indent_spec
from ekap.mcp_server.narrow_no_break_space import (
    narrow_no_break_space_sources,
    narrow_no_break_space_spec,
)
from ekap.mcp_server.national_digit_shapes import (
    national_digit_shapes_sources,
    national_digit_shapes_spec,
)
from ekap.mcp_server.nbsp import nbsp_sources, nbsp_spec
from ekap.mcp_server.nel import nel_sources, nel_spec
from ekap.mcp_server.nominal_digit_shapes import (
    nominal_digit_shapes_sources,
    nominal_digit_shapes_spec,
)
from ekap.mcp_server.paragraph_separator import (
    paragraph_separator_sources,
    paragraph_separator_spec,
)
from ekap.mcp_server.pop_directional_formatting import (
    pop_directional_formatting_sources,
    pop_directional_formatting_spec,
)
from ekap.mcp_server.pop_directional_isolate import (
    pop_directional_isolate_sources,
    pop_directional_isolate_spec,
)
from ekap.mcp_server.punctuation_space import (
    punctuation_space_sources,
    punctuation_space_spec,
)
from ekap.mcp_server.repeated_blank import repeated_blank_sources, repeated_blank_spec
from ekap.mcp_server.right_to_left_embedding import (
    right_to_left_embedding_sources,
    right_to_left_embedding_spec,
)
from ekap.mcp_server.right_to_left_isolate import (
    right_to_left_isolate_sources,
    right_to_left_isolate_spec,
)
from ekap.mcp_server.right_to_left_mark import (
    right_to_left_mark_sources,
    right_to_left_mark_spec,
)
from ekap.mcp_server.right_to_left_override import (
    right_to_left_override_sources,
    right_to_left_override_spec,
)
from ekap.mcp_server.server import MCPServer as _MCPServer
from ekap.mcp_server.server import ToolResult, ToolSpec
from ekap.mcp_server.soft_hyphen import soft_hyphen_sources, soft_hyphen_spec
from ekap.mcp_server.tab import tab_sources, tab_spec
from ekap.mcp_server.thin_space import thin_space_sources, thin_space_spec
from ekap.mcp_server.trailing_blank import trailing_blank_sources, trailing_blank_spec
from ekap.mcp_server.trailing_space import trailing_space_sources, trailing_space_spec
from ekap.mcp_server.unterminated import unterminated_sources, unterminated_spec
from ekap.mcp_server.vertical_tab import vertical_tab_sources, vertical_tab_spec
from ekap.mcp_server.word_joiner import word_joiner_sources, word_joiner_spec
from ekap.mcp_server.zero_width_joiner import (
    zero_width_joiner_sources,
    zero_width_joiner_spec,
)
from ekap.mcp_server.zero_width_non_joiner import (
    zero_width_non_joiner_sources,
    zero_width_non_joiner_spec,
)
from ekap.mcp_server.zero_width_space import (
    zero_width_space_sources,
    zero_width_space_spec,
)


class MCPServer(_MCPServer):
    """Server that also lists whitespace hygiene and blank-line sources."""

    def list_tools(self) -> tuple[ToolSpec, ...]:
        tools = super().list_tools()
        extra = (
            activate_arabic_form_shaping_spec(),
            activate_symmetric_swapping_spec(),
            arabic_letter_mark_spec(),
            bom_spec(),
            crlf_spec(),
            em_space_spec(),
            figure_space_spec(),
            first_strong_isolate_spec(),
            form_feed_spec(),
            hair_space_spec(),
            inhibit_arabic_form_shaping_spec(),
            inhibit_symmetric_swapping_spec(),
            leading_blank_spec(),
            leading_space_spec(),
            left_to_right_embedding_spec(),
            left_to_right_isolate_spec(),
            left_to_right_mark_spec(),
            left_to_right_override_spec(),
            line_separator_spec(),
            long_line_spec(),
            mixed_indent_spec(),
            narrow_no_break_space_spec(),
            national_digit_shapes_spec(),
            nbsp_spec(),
            nel_spec(),
            nominal_digit_shapes_spec(),
            paragraph_separator_spec(),
            pop_directional_formatting_spec(),
            pop_directional_isolate_spec(),
            punctuation_space_spec(),
            repeated_blank_spec(),
            right_to_left_embedding_spec(),
            right_to_left_isolate_spec(),
            right_to_left_mark_spec(),
            right_to_left_override_spec(),
            soft_hyphen_spec(),
            tab_spec(),
            thin_space_spec(),
            trailing_blank_spec(),
            trailing_space_spec(),
            unterminated_spec(),
            vertical_tab_spec(),
            word_joiner_spec(),
            zero_width_joiner_spec(),
            zero_width_non_joiner_spec(),
            zero_width_space_spec(),
        )
        names = {tool.name for tool in tools}
        return (*tools, *(spec for spec in extra if spec.name not in names))

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        args = dict(arguments or {})
        if name == "activate_arabic_form_shaping_sources":
            return activate_arabic_form_shaping_sources(self, args)
        if name == "activate_symmetric_swapping_sources":
            return activate_symmetric_swapping_sources(self, args)
        if name == "arabic_letter_mark_sources":
            return arabic_letter_mark_sources(self, args)
        if name == "bom_sources":
            return bom_sources(self, args)
        if name == "crlf_sources":
            return crlf_sources(self, args)
        if name == "em_space_sources":
            return em_space_sources(self, args)
        if name == "figure_space_sources":
            return figure_space_sources(self, args)
        if name == "first_strong_isolate_sources":
            return first_strong_isolate_sources(self, args)
        if name == "form_feed_sources":
            return form_feed_sources(self, args)
        if name == "hair_space_sources":
            return hair_space_sources(self, args)
        if name == "inhibit_arabic_form_shaping_sources":
            return inhibit_arabic_form_shaping_sources(self, args)
        if name == "inhibit_symmetric_swapping_sources":
            return inhibit_symmetric_swapping_sources(self, args)
        if name == "leading_blank_sources":
            return leading_blank_sources(self, args)
        if name == "long_line_sources":
            return long_line_sources(self, args)
        if name == "leading_space_sources":
            return leading_space_sources(self, args)
        if name == "left_to_right_embedding_sources":
            return left_to_right_embedding_sources(self, args)
        if name == "left_to_right_isolate_sources":
            return left_to_right_isolate_sources(self, args)
        if name == "left_to_right_mark_sources":
            return left_to_right_mark_sources(self, args)
        if name == "left_to_right_override_sources":
            return left_to_right_override_sources(self, args)
        if name == "line_separator_sources":
            return line_separator_sources(self, args)
        if name == "mixed_indent_sources":
            return mixed_indent_sources(self, args)
        if name == "narrow_no_break_space_sources":
            return narrow_no_break_space_sources(self, args)
        if name == "national_digit_shapes_sources":
            return national_digit_shapes_sources(self, args)
        if name == "nbsp_sources":
            return nbsp_sources(self, args)
        if name == "nel_sources":
            return nel_sources(self, args)
        if name == "nominal_digit_shapes_sources":
            return nominal_digit_shapes_sources(self, args)
        if name == "paragraph_separator_sources":
            return paragraph_separator_sources(self, args)
        if name == "pop_directional_formatting_sources":
            return pop_directional_formatting_sources(self, args)
        if name == "pop_directional_isolate_sources":
            return pop_directional_isolate_sources(self, args)
        if name == "punctuation_space_sources":
            return punctuation_space_sources(self, args)
        if name == "repeated_blank_sources":
            return repeated_blank_sources(self, args)
        if name == "right_to_left_embedding_sources":
            return right_to_left_embedding_sources(self, args)
        if name == "right_to_left_isolate_sources":
            return right_to_left_isolate_sources(self, args)
        if name == "right_to_left_mark_sources":
            return right_to_left_mark_sources(self, args)
        if name == "right_to_left_override_sources":
            return right_to_left_override_sources(self, args)
        if name == "soft_hyphen_sources":
            return soft_hyphen_sources(self, args)
        if name == "tab_sources":
            return tab_sources(self, args)
        if name == "thin_space_sources":
            return thin_space_sources(self, args)
        if name == "trailing_blank_sources":
            return trailing_blank_sources(self, args)
        if name == "trailing_space_sources":
            return trailing_space_sources(self, args)
        if name == "unterminated_sources":
            return unterminated_sources(self, args)
        if name == "vertical_tab_sources":
            return vertical_tab_sources(self, args)
        if name == "word_joiner_sources":
            return word_joiner_sources(self, args)
        if name == "zero_width_joiner_sources":
            return zero_width_joiner_sources(self, args)
        if name == "zero_width_non_joiner_sources":
            return zero_width_non_joiner_sources(self, args)
        if name == "zero_width_space_sources":
            return zero_width_space_sources(self, args)
        return super().call(name, arguments)


__all__ = [
    "MCPServer",
    "ToolResult",
    "ToolSpec",
]
