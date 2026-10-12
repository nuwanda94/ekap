# Changelog

All notable changes to EKAP are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-11

### Added
- In-memory RAG core: `Ingester`, `Retriever`, and `cite()` helper with citation objects (source id + snippet).
- In-process MCP server with tool catalog including `search_docs`, draft actions, and read-only Unicode/whitespace hygiene inspectors.
- Orchestrator for plan → retrieve/tool → synthesize with full citation trail and run trace.
- Approval gate: draft → approve → execute path; reject leaves state unchanged.
- Evals: 22 golden questions and runner that exits non-zero on regression; invoked from CI.
- Source-removal helpers under the approval-gate pattern.
- Zero-runtime-deps package layout, typed public APIs, ruff + pytest + golden-evals CI.
- `Ingester.clear()` to reset the in-memory corpus while preserving chunk settings.
- `Ingester.__len__` reports the number of ingested documents.
- `Ingester.__contains__` supports membership testing for source ids (whitespace-tolerant, fails closed on non-strings).
- `Ingester.__iter__` yields defensive copies of ingested documents in insertion order.
- `Ingester.get` returns a defensive copy of a document by source id (whitespace-tolerant; raises KeyError if absent).
- `Ingester.source_ids` returns source ids in insertion order.
- `Ingester.update_metadata` replaces document metadata without re-chunking (whitespace-tolerant source id; fails closed on missing source or invalid metadata).
- `Ingester.update_text` replaces document text and re-chunks it while preserving metadata (whitespace-tolerant source id; fails closed on missing source or blank text).
- `Ingester.chunks_for` returns the chunks belonging to a source id in index order (whitespace-tolerant; raises KeyError if absent).
- `Ingester.chunk_count` reports the total number of chunks in the corpus.
- `Ingester.document_count` reports the total number of documents in the corpus (matches `len()`).
- `Ingester.total_chars` reports the sum of character lengths of all document texts (independent of chunking).
- `Ingester.max_chars` reports the length of the longest document text (0 if the corpus is empty).
- `Ingester.min_chars` reports the length of the shortest document text (0 if the corpus is empty).
- `Ingester.avg_chars` reports the average length of document texts (0.0 if the corpus is empty).
- `Ingester.median_chars` reports the median length of document texts (0.0 if the corpus is empty; averages the two central values when the count is even).
- `Ingester.std_chars` reports the population standard deviation of document text lengths (0.0 if the corpus is empty or has one document).
- `Ingester.var_chars` reports the population variance of document text lengths (0.0 if the corpus is empty or has one document; square of `std_chars`).
- `Ingester.range_chars` reports the difference between the longest and shortest document text lengths (0 if the corpus is empty or has one document).
- `Ingester.mode_chars` reports the most frequent document text length (0 if empty; smallest length when there is a tie).
- `Ingester.nunique_chars` reports the number of distinct document text lengths (0 if empty).

### Notes
- Alpha reference implementation. No network calls or real LLM keys required for unit tests.
- Side effects stay behind the approval gate.
