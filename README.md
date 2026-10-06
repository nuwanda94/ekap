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
+---------+    +--------------+
| RAG     |    | MCP Server   |  <- real tools
| Layer   |    | (2-4 tools)  |
+----+----+    +------+-------+
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
6. ~~Eval harness (20 golden questions + automatic scoring)~~

Later: multi-tenancy, richer observability, Slack/Teams approval UI, production packaging.

---

## Local Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev

uv run pytest
uv run ruff check .
uv run python -m ekap.evals
```

`python -m ekap.evals` scores the bundled golden set and exits non-zero on regression. CI runs that command after lint and tests.

---

## Design Principles

- **Citation fidelity over fluency** — if it cannot be cited, it is not answered as fact.
- **Tools are the only side-effect path** — no hidden writes.
- **Human approval for mutation** — drafts are cheap; execution is gated.
- **Evaluation is a first-class citizen** — regressions fail the build.
- **Small, reviewable PRs** — one concern per change.

---

## Status

RAG core, an in-process MCP server (`health`, `search_docs`, `list_sources`, `get_source`, `list_drafts`, `get_draft`, `cancel_draft`, `draft_action`, `draft_remove_source`), a planning orchestrator, an approval gate, and an eval harness are in place. The harness scores 22 golden retrieval questions against a bundled corpus and returns exit code 1 when the top chunk source or required phrase is wrong. CI invokes `python -m ekap.evals` after tests, so a retrieval regression fails the build. `ApprovalGate.draft` records a pending action and does not call the executor. `approve` runs the injected executor once; `reject` leaves external state unchanged. `ApprovalGate.decisions()` returns a `DecisionRecord` for each draft, approve, and reject so gate outcomes are auditable without replaying the executor. MCP `draft_action` results can be submitted to the gate; the server itself still never executes. `ApprovalGate.submit_draft` accepts only a pending tool draft: a cancelled or executed draft fails closed and is not recorded, so a cancelled MCP draft cannot be approved. `queue_mutation_drafts` submits successful orchestrator `draft_action` results and ignores search results; actions stay pending until `approve`. `Orchestrator.run` attaches a `RunTrace` of plan, retrieve, tool, and synthesize events so a draft is auditable without executing side effects. `Ingester.remove` withdraws a source and drops its chunks so later retrieval cannot cite it. `Retriever.query` accepts a metadata filter and drops chunks whose source does not match every pair. `search_docs` forwards an optional metadata map to that filter and returns an error result when the map is not strings, without executing anything. `query_with_citations` copies source metadata onto each `Citation` so a snippet still names its tenant or collection; the map is a copy, so callers cannot mutate the stored document. `search_docs` citation payloads include that same metadata copy, so a tool caller can see tenant or collection without a second lookup. `Ingester.find` lists ingested sources that match every metadata pair, in insertion order, and returns copies so an audit cannot mutate the corpus. `list_sources` calls that finder when an ingester is configured and returns source id, character count, and a metadata copy; a missing ingester or invalid map fails closed and does not execute. `get_source` returns one source's text and a metadata copy by id; a missing ingester, blank id, or unknown source fails closed and does not execute. `draft_remove_source` records a pending `remove_source` draft for a known source id and leaves the corpus in place; a missing ingester, blank id, or unknown source fails closed and does not call `Ingester.remove`. `source_removal_executor` is the gate executor for that draft: `approve` calls `Ingester.remove` once, while `reject` and any other action leave the corpus and its chunks unchanged. The MCP server still never executes the draft. `list_drafts` returns recorded drafts in order, each with draft id, action, target, pending status, and `executed: false`; extra arguments fail closed, and the tool never calls the executor. `get_draft` returns one recorded draft by id, including action, target, pending status, and `executed: false`; a blank id, unknown id, or extra argument fails closed, and the tool never calls the executor. `cancel_draft` marks one pending draft `cancelled` and leaves the corpus and executor untouched; a blank id, unknown id, extra argument, or draft that is not pending fails closed, and the tool never calls the executor. `ApprovalGate.cancel` marks one pending gate action `cancelled` and records the transition; an unknown id, a non-pending action, or a non-string reason fails closed, and the executor is never called. `ApprovalGate.actions` returns recorded actions in insertion order, optionally filtered by status, and copies payloads so an audit cannot mutate the gate; a blank or unknown status fails closed, and the method never calls the executor.

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*
