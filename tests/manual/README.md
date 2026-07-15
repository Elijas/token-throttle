# Manual Tests

CI collects this directory as part of the all-tests coverage gate. The
synthetic/offline tests run without network access. Live OpenAI tests are
skipped unless `OPENAI_API_KEY` is set; the Anthropic smoke test additionally
requires `TOKEN_THROTTLE_RUN_LIVE_ANTHROPIC=1` plus the real RPM, ITPM, and OTPM
limit environment variables documented by the worked example.

## Setup

Copy `.env.example` to `.env` and fill in a **short-lived or scope-restricted** API key.
Never commit `.env` — it is gitignored at all directory depths.

## Running

```bash
uv run pytest tests/manual/ -v
```
