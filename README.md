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

`en_quad_sources` lists matching sources that contain U+2000 (EN QUAD), in insertion order, and returns source id, a metadata copy, character count, en-quad count, run count, and whether the source starts or ends with U+2000, without source text.
Ordinary spaces, em spaces, en spaces, and other whitespace are not a match.
A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

See source under `src/ekap/mcp_server/` for the full catalog of similar listing tools (spaces, controls, bidi marks, etc.).

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
