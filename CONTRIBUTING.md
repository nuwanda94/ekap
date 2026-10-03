# Contributing to EKAP

Thank you for contributing.

## Ground rules

- All changes land via pull request.
- PR titles use conventional commits: `chore:`, `feat:`, `fix:`, `test:`, `docs:`.
- CI must be green before merge.
- Keep PRs focused on a single concern.

## Local setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Branch & PR workflow

1. Branch from latest `main`.
2. Implement the change + tests.
3. Open a PR against `main`.
4. Wait for CI to pass.
5. Merge only when green.

## Code style

- Python 3.12+
- `ruff` for linting
- Type hints preferred for public APIs
- Tests live under `tests/` and mirror the package structure
