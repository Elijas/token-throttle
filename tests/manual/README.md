# Manual Tests

Tests in this directory require real API keys and are **not** run in CI/CD.

## Setup

Copy `.env.example` to `.env` and fill in a **short-lived or scope-restricted** API key.
Never commit `.env` — it is gitignored at all directory depths.

## Running

```bash
uv run pytest tests/manual/ -v
```
