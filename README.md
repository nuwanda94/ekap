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

RAG core, an in-process MCP server (`health`, `search_docs`, `list_sources`, `get_source`, `summarize_sources`, `list_chunks`, `get_chunk`, `list_drafts`, `get_draft`, `cancel_draft`, `summarize_drafts`, `describe_tool`, `list_tools`, `metadata_facets`, `duplicate_sources`, `missing_metadata`, `padded_sources`, `casefold_sources`, `strip_sources`, `collapse_sources`, `normalize_sources`, `punctfold_sources`, `accentfold_sources`, `blank_sources`, `control_sources`, `replacement_sources`, `line_ending_sources`, `trailing_blank_sources`, `draft_action`, `draft_remove_source`), a planning orchestrator, an approval gate, and an eval harness are in place. The harness scores 22 golden retrieval questions against a bundled corpus and returns exit code 1 when the top chunk source or required phrase is wrong. CI invokes `python -m ekap.evals` after tests, so a retrieval regression fails the build. `ApprovalGate.draft` records a pending action and does not call the executor. `approve` runs the injected executor once; `reject` leaves external state unchanged. `ApprovalGate.decisions()` returns a `DecisionRecord` for each draft, approve, and reject so gate outcomes are auditable without replaying the executor. MCP `draft_action` results can be submitted to the gate; the server itself still never executes. `ApprovalGate.submit_draft` accepts only a pending tool draft: a cancelled or executed draft fails closed and is not recorded, so a cancelled MCP draft cannot be approved. `queue_mutation_drafts` submits successful orchestrator `draft_action` results and ignores search results; actions stay pending until `approve`. `Orchestrator.run` attaches a `RunTrace` of plan, retrieve, tool, and synthesize events so a draft is auditable without executing side effects. `Ingester.remove` withdraws a source and drops its chunks so later retrieval cannot cite it. `Retriever.query` accepts a metadata filter and drops chunks whose source does not match every pair. `search_docs` forwards an optional metadata map to that filter and returns an error result when the map is not strings, without executing anything. `query_with_citations` copies source metadata onto each `Citation` so a snippet still names its tenant or collection; the map is a copy, so callers cannot mutate the stored document. `search_docs` citation payloads include that same metadata copy, so a tool caller can see tenant or collection without a second lookup. `Ingester.find` lists ingested sources that match every metadata pair, in insertion order, and returns copies so an audit cannot mutate the corpus. `list_sources` calls that finder when an ingester is configured and returns source id, character count, and a metadata copy; a missing ingester or invalid map fails closed and does not execute. `get_source` returns one source's text and a metadata copy by id; a missing ingester, blank id, or unknown source fails closed and does not execute. `draft_remove_source` records a pending `remove_source` draft for a known source id and leaves the corpus in place; a missing ingester, blank id, or unknown source fails closed and does not call `Ingester.remove`. `source_removal_executor` is the gate executor for that draft: `approve` calls `Ingester.remove` once, while `reject` and any other action leave the corpus and its chunks unchanged. The MCP server still never executes the draft. `list_drafts` returns recorded drafts in order, each with draft id, action, target, status, and `executed: false`. An optional status keeps only pending or cancelled drafts; a blank, unknown, or non-string status fails closed, extra arguments fail closed, and the tool never calls the executor. `get_draft` returns one recorded draft by id, including action, target, pending status, and `executed: false`; a blank id, unknown id, or extra argument fails closed, and the tool never calls the executor. `cancel_draft` marks one pending draft `cancelled` and leaves the corpus and executor untouched; a blank id, unknown id, extra argument, or draft that is not pending fails closed, and the tool never calls the executor. `ApprovalGate.cancel` marks one pending gate action `cancelled` and records the transition; an unknown id, a non-pending action, or a non-string reason fails closed, and the executor is never called. `ApprovalGate.actions` returns recorded actions in insertion order, optionally filtered by status, and copies payloads so an audit cannot mutate the gate; a blank or unknown status fails closed, and the method never calls the executor. `ApprovalGate.get`, `draft`, `approve`, `reject`, and `cancel` also return payload copies, so mutating a returned action does not change the stored record; only `approve` calls the executor. `summarize_drafts` counts recorded MCP drafts by pending and cancelled status and returns a total; extra arguments fail closed, and the tool never calls the executor. `describe_tool` returns one advertised tool's name, description, and mutates flag; a blank, unknown, or non-string name fails closed, extra arguments fail closed, and the tool never calls the named tool or the executor. `list_tools` returns advertised tools in order, each with name, description, mutates, and `executed: false`, and an optional boolean `mutates` filter; a non-boolean filter or extra argument fails closed, and the tool never calls the named tools or the executor. `summarize_sources` counts ingested sources, chunks, and characters, optionally filtered by metadata; a missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor. `search_docs` forwards an optional `min_score` to retrieval and omits citations below that token-overlap ratio; a non-numeric score or a value outside 0 to 1 fails closed, and the tool never changes the corpus or calls the executor. `list_chunks` returns ingested chunk id, source id, index, and character count, optionally filtered by source id or metadata; a missing ingester, blank or unknown source id, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor. `get_chunk` returns one ingested chunk by id, including text, index, character count, and a metadata copy; a missing ingester, blank id, unknown chunk, or extra argument fails closed, and the tool never changes the corpus or calls the executor. `get_chunk_context` returns that chunk plus same-source neighbors within an optional radius, each with offset, text, and character count; a missing ingester, blank id, unknown chunk, negative radius, or extra argument fails closed, and the tool never changes the corpus or calls the executor. `find_chunks` returns chunks whose text contains a literal phrase, optionally limited to a source, metadata map, or match count, each with a short snippet; a missing ingester, blank phrase, unknown source, invalid limit, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`count_phrase` returns source-level match counts for a literal phrase, optionally limited to a source or metadata map, and does not return chunk text; a missing ingester, blank phrase, unknown source, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`overlap_sources` returns shared and unique alphanumeric token counts for two known sources, plus the shared token list, and does not return source text; a missing ingester, blank id, identical ids, unknown source, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`token_stats` returns occurrence and unique alphanumeric token counts for one known source, plus the most frequent tokens, and does not return source text; a missing ingester, blank id, unknown source, invalid limit, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`query_coverage` returns source ids for each alphanumeric query token, plus sources that contain every token, optionally limited by metadata, and does not return source text; a missing ingester, blank query, query with no token, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`query_gaps` returns query tokens that no matching source covers, plus the tokens each source is missing, optionally limited by metadata, and does not return source text; a missing ingester, blank query, query with no token, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`exclusive_tokens` returns alphanumeric tokens that appear in exactly one matching source, in first-seen order, optionally limited by metadata, and does not return source text; a missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`shared_tokens` returns alphanumeric tokens that appear in two or more matching sources, in first-seen order with the source ids that contain each token, optionally limited by metadata, and does not return source text; a missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`duplicate_sources` groups matching sources that share identical text, in first-seen order, and returns source ids, group size, and character count without the text; a whitespace difference is not a duplicate. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`missing_metadata` lists matching sources that lack a metadata key, in insertion order, with a metadata copy and character count and without source text; an empty value still counts as present. A missing ingester, blank key, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`padded_sources` lists matching sources whose text has leading or trailing whitespace, in insertion order, with a metadata copy, character count, and leading and trailing counts, and without source text. Interior whitespace alone is not padding. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`casefold_sources` groups matching sources whose text is identical after case folding, in first-seen order, and returns source ids, group size, character count of the first text, and whether the raw texts are identical, without source text. A single source is not a group. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`strip_sources` groups matching sources whose text is identical after stripping leading and trailing whitespace, in first-seen order, and returns source ids, group size, stripped character count, whether the raw texts are identical, and whether any member is padded, without source text. A single source is not a group. Interior whitespace alone is not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`collapse_sources` groups matching sources whose text is identical after stripping edges and collapsing internal whitespace runs to a single space, in first-seen order, and returns source ids, group size, collapsed character count, whether the raw texts are identical, whether any member is padded, and whether any member needed collapsing, without source text. A single source is not a group. Case differences are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`normalize_sources` groups matching sources whose text is identical after stripping edges, collapsing internal whitespace, and case folding, in first-seen order, and returns source ids, group size, normalized character count, whether the raw texts are identical, whether any member is padded, whether any member needed collapsing, and whether any member differs by case, without source text. A single source is not a group. Punctuation differences are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`punctfold_sources` groups matching sources whose text is identical after stripping edges, collapsing internal whitespace, case folding, and dropping punctuation, in first-seen order, and returns source ids, group size, folded character count, whether the raw texts are identical, whether any member is padded, whether any member needed collapsing, whether any member differs by case, and whether any member differs by punctuation, without source text. A single source is not a group. Letter or digit differences are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`accentfold_sources` groups matching sources whose text is identical after stripping edges, collapsing internal whitespace, case folding, dropping punctuation, and stripping combining marks, in first-seen order, and returns source ids, group size, folded character count, whether the raw texts are identical, whether any member is padded, whether any member needed collapsing, whether any member differs by case, whether any member differs by punctuation, and whether any member differs by accent, without source text. A single source is not a group. Letter or digit differences are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`blank_sources` lists matching sources whose text has no letter or digit after that same accent fold, in insertion order, and returns source id, a metadata copy, raw and folded character counts, and whether the source is empty, padded, needed collapsing, or is punctuation-only, without source text. A letter or digit keeps the source off the list. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`control_sources` lists matching sources that contain Unicode format characters or controls other than tab, newline, and carriage return, in insertion order, and returns source id, a metadata copy, character count, control count, format count, and zero-width count, without source text. Ordinary whitespace is not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.
`replacement_sources` lists matching sources that contain the Unicode replacement character U+FFFD, in insertion order, and returns source id, a metadata copy, character count, and replacement count, without source text. Ordinary text, including punctuation, is not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`line_ending_sources` lists matching sources that contain a carriage return, either bare or inside CRLF, in insertion order, and returns source id, a metadata copy, character count, LF count, CRLF count, bare CR count, and whether more than one ending style is present, without source text. LF-only text is not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`trailing_space_sources` lists matching sources whose lines end in ASCII spaces or tabs, in insertion order, and returns source id, a metadata copy, character count, trailing-space line count, trailing-tab line count, and whether both styles are present, without source text. Interior spaces and lines with no trailing space or tab are not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`trailing_blank_sources` lists matching sources that end with one or more blank lines, in insertion order, and returns source id, a metadata copy, character count, trailing blank-line count, content-line count, and whether the source is blank-only, without source text. A blank line is empty or only ASCII spaces and tabs. The empty segment after a final line terminator is not a line, and an interior blank line is not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

`leading_blank_sources` lists matching sources that start with one or more blank lines, in insertion order, and returns source id, a metadata copy, character count, leading blank-line count, content-line count, and whether the source is blank-only, without source text. A blank line is empty or only ASCII spaces and tabs. The empty segment after a final terminator is not a line, and an interior blank line is not a match. A missing ingester, invalid metadata map, or extra argument fails closed, and the tool never changes the corpus or calls the executor.

---

*Built as a principal-staff reference implementation of a safe, evaluable, human-supervised agent platform.*