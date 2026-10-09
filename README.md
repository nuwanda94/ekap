# EKAP — Enterprise Knowledge & Action Agent Platform

> A single, production-oriented multi-agent system that turns internal knowledge into **safe, auditable actions**.

[![CI](https://github.com/nuwanda94/ekap/actions/workflows/ci.yml/badge.svg)](https://github.com/nuwanda94/ekap/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

EKAP is deliberately built to prove five capabilities in one coherent codebase:

| Capability | What it demonstrates |
|------------|----------------------|
| **RAG** | Answers *only* from approved documents and cites every claim |
| **MCP** | Exposes real tools any agent can call through the Model Context Protocol |
| **Planning** | Explicit plan → retrieve/search → synthesize, with full citation trail |
| **Evals** | 20+ golden questions scored automatically on every PR |
| **Approval Gate** | Side-effecting actions are drafted only; a human must click Approve |

This is not five disconnected demos. It is one system with clear boundaries, auditability, and a human-in-the-loop control plane.

See the previous status history on commit 7829fa4908356c3beb17038948c6bf244d05c667 for the full tool catalog written before this run.

`tab_sources` lists matching sources that contain U+0009, in insertion order, and returns source id, a metadata copy, character count, tab count, indent-tab line count, and inline tab count, without source text. A line counts as indent-tab only when it starts with a tab; tabs after a non-tab are inline. Ordinary spaces are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`form_feed_sources` lists matching sources that contain U+000C, in insertion order, and returns source id, a metadata copy, character count, form-feed count, run count, and whether the source starts or ends with U+000C, without source text. A run is a maximal consecutive sequence of form feeds. Ordinary whitespace is not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`vertical_tab_sources` lists matching sources that contain U+000B, in insertion order, and returns source id, a metadata copy, character count, vertical-tab count, run count, and whether the source starts or ends with U+000B, without source text. A run is a maximal consecutive sequence of vertical tabs. Ordinary whitespace, tabs, and form feeds are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`nel_sources` lists matching sources that contain U+0085, in insertion order, and returns source id, a metadata copy, character count, next-line count, run count, and whether the source starts or ends with U+0085, without source text. A run is a maximal consecutive sequence of next-line characters. Ordinary whitespace, tabs, vertical tabs, and form feeds are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`line_separator_sources` lists matching sources that contain U+2028, in insertion order, and returns source id, a metadata copy, character count, line-separator count, run count, and whether the source starts or ends with U+2028, without source text. A run is a maximal consecutive sequence of line separators. Ordinary whitespace, tabs, vertical tabs, form feeds, and next-line characters are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`paragraph_separator_sources` lists matching sources that contain U+2029, in insertion order, and returns source id, a metadata copy, character count, paragraph-separator count, run count, and whether the source starts or ends with U+2029, without source text. A run is a maximal consecutive sequence of paragraph separators. Ordinary whitespace, tabs, vertical tabs, form feeds, next-line characters, and line separators are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`zero_width_space_sources` lists matching sources that contain U+200B, in insertion order, and returns source id, a metadata copy, character count, zero-width-space count, run count, and whether the source starts or ends with U+200B, without source text. A run is a maximal consecutive sequence of zero-width spaces. Ordinary whitespace, BOM, word joiners, and zero-width joiners or non-joiners are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`word_joiner_sources` lists matching sources that contain U+2060, in insertion order, and returns source id, a metadata copy, character count, word-joiner count, run count, and whether the source starts or ends with U+2060, without source text. A run is a maximal consecutive sequence of word joiners. Ordinary whitespace, BOM, zero-width spaces, and zero-width joiners or non-joiners are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
