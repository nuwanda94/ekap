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

This is not five disconnected demos.
It is one system with clear boundaries, auditability, and a human-in-the-loop control plane.

## MCP Tools

The in-process MCP server exposes many read-only inspection tools for Unicode and whitespace hygiene, plus search and draft tools.

`invisible_plus_sources` lists matching sources that contain U+2064 (INVISIBLE PLUS), in insertion order, and returns source id, a metadata copy, character count, invisible-plus count, run count, and whether the source starts or ends with U+2064, without source text.
Ordinary spaces, em spaces, en spaces, medium mathematical spaces, Ogham space marks, Mongolian vowel separators, invisible separators, invisible times, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`invisible_times_sources` lists matching sources that contain U+2061 (INVISIBLE TIMES), in insertion order, and returns source id, a metadata copy, character count, invisible-times count, run count, and whether the source starts or ends with U+2061, without source text.
Ordinary spaces, em spaces, en spaces, medium mathematical spaces, Ogham space marks, Mongolian vowel separators, invisible separators, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`invisible_separator_sources` lists matching sources that contain U+2063 (INVISIBLE SEPARATOR), in insertion order, and returns source id, a metadata copy, character count, invisible-separator count, run count, and whether the source starts or ends with U+2063, without source text.
Ordinary spaces, em spaces, en spaces, medium mathematical spaces, Ogham space marks, Mongolian vowel separators, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`mongolian_vowel_separator_sources` lists matching sources that contain U+180E (MONGOLIAN VOWEL SEPARATOR), in insertion order, and returns source id, a metadata copy, character count, mongolian-vowel-separator count, run count, and whether the source starts or ends with U+180E, without source text.
Ordinary spaces, em spaces, en spaces, medium mathematical spaces, Ogham space marks, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`ogham_space_sources` lists matching sources that contain U+1680 (OGHAM SPACE MARK), in insertion order, and returns source id, a metadata copy, character count, ogham-space count, run count, and whether the source starts or ends with U+1680, without source text.
Ordinary spaces, em spaces, en spaces, medium mathematical spaces, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`en_quad_sources` lists matching sources that contain U+2000 (EN QUAD), in insertion order, and returns source id, a metadata copy, character count, en-quad count, run count, and whether the source starts or ends with U+2000, without source text.
Ordinary spaces, em spaces, en spaces, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`em_quad_sources` lists matching sources that contain U+2001 (EM QUAD), in insertion order, and returns source id, a metadata copy, character count, em-quad count, run count, and whether the source starts or ends with U+2001, without source text.
Ordinary spaces, em spaces, en spaces, EN QUADs, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

See source under `src/ekap/mcp_server/` for the full catalog of similar listing tools (spaces, controls, bidi marks, etc.).

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
