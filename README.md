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

---

## Architecture (high level)

```
User / API Request
        |
        v
+-----------------------+
|  Orchestrator Agent   |  <- Planning
|  (plan -> retrieve -> |
|   tool-use -> draft)  |
+-----------+-----------+
            |
    +-------+-------+
    v               v
+---------+   +--------------+
| RAG     |   | MCP Server   |  <- real tools
| Layer   |   | (2-4 tools)  |
+----+----+   +------+-------+
     |               |
     +-------+-------+
             v
+-----------------------+
| Draft Answer / Action |
| + full citation trail |
+-----------+-----------+
            |
            v
+-----------------------+
| Approval Gate         |  <- human-in-the-loop
| (pending until Approve)|
+-----------+-----------+
            | (only after Approve)
            v
      Execute / Respond
            |
            v
+-----------------------+
| Eval Harness          |  <- CI-gated
| (golden set + scoring)|
+-----------------------+
```

---

## Repository Layout

```
ekap/
├── src/ekap/
│   ├── agents/           # orchestrator + approval-gate
│   ├── mcp_server/       # MCP tools and server
│   ├── rag/              # ingest, retrieve, cite
│   └── evals/            # golden set + runner
├── tests/
├── .github/workflows/
│   └── ci.yml
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Development Model

- PR titles are conventional: `chore:`, `feat:`, `fix:`, or `test:`
- Every change lands via pull request
- CI must be green before merge

**Do not push directly to `main`.** All work goes through PRs.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and workflow details.

---

## Roadmap (in order)

1. ~~Project skeleton + GitHub Actions CI~~
2. ~~RAG core (ingest, retrieve, citations)~~
3. ~~MCP server with 2–3 real tools~~
4. ~~Orchestrator / planning agent~~
5. ~~Approval-gate agent~~
6. Eval harness (20 golden questions + automatic scoring)

Later: multi-tenancy, richer observability, Slack/Teams approval UI, production packaging.

---

## Local Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev

uv run pytest
uv run ruff check .
```

---

## Design Principles

- **Citation fidelity over fluency** — if it cannot be cited, it is not answered as fact.
- **Tools are the only side-effect path** — no hidden writes.
- **Human approval for mutation** — drafts are cheap; execution is gated.
- **Evaluation is a first-class citizen** — regressions fail the build.
- **Small, reviewable PRs** — one concern per change.

---

## Status

RAG core, an in-process MCP server (`health`, `search_docs`, `draft_action`), a planning orchestrator, and an approval gate are in place. `ApprovalGate.draft` records a pending action and does not call the executor. `approve` runs the injected executor once; `reject` leaves external state unchanged. MCP `draft_action` results can be submitted to the gate; the server itself still never executes. Next up: eval harness.

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
