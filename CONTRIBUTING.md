# Contributing to EKAP

Thank you for contributing.

## Ground rules

- Prefer **focused, single-concern** changes.
- Commit subjects use conventional commits: `chore:`, `feat:`, `fix:`, `test:`, `docs:`.
- **CI must be green** on `main` (ruff + pytest + golden evals).
- External contributions should open a pull request against `main`.
- Maintainers may land small, reviewable fixes directly on `main` when the change is a single concern and CI stays green.

## Local setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest --cov=ekap -q
uv run python -m ekap.evals
```

## External contributor workflow

1. Branch from latest `main`.
2. Implement the change + tests (every `feat:` needs at least one unit test).
3. Open a PR against `main`.
4. Wait for CI to pass.
5. Merge only when green.

## Code style

- Python 3.12+
- `ruff` for linting
- Type hints on public APIs
- Tests under `tests/`, mirroring `src/ekap/`
- No network calls or real LLM keys in unit tests; inject fakes
- Package code under `src/ekap/` only — do not invent a parallel tree
