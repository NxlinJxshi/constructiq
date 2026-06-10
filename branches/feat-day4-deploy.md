# Branch: feat/day4-deploy

## What this branch does

Deploy-readiness pass. Makes the system safe to demo under real load, hardens two fragile code paths, adds Cloud Run packaging, and eliminates repeated MongoDB round-trips per audit.

## Changed files

| File | What changed |
|---|---|
| `agent/audit_agent.py` | Gemini payload cap — sends only top `NARRATE_LIMIT` flags; returns ALL flags every time |
| `ingestion/pdf_parser.py` | Async-response check replaced with duck-typing; deep SDK import removed |
| `detection/categorical_anomalies.py` | Process-level baseline cache — MongoDB queried once per process |
| `detection/feature_engineering.py` | Process-level encoder cache — MongoDB queried once per process |
| `requirements.txt` | Added `gunicorn` |
| `.env.example` | Added `GEMINI_MODEL=gemini-2.5-flash` |
| `Dockerfile` | New — Cloud Run deployment target |
| `.dockerignore` | New — excludes secrets, data, artifacts, caches |

## Gemini payload cap — data flow

Previously `narrate_and_rank(flags)` sent every flag to Gemini in one prompt. On the 200-day synthetic dataset that is 400+ flags, causing multi-minute hangs.

New behavior:
```
all_flags (N total)
  └─► sort by severity (high → medium → low)
        └─► subset = top NARRATE_LIMIT (default 25)
              └─► Gemini narrates subset only
                    └─► merge: update explanation on subset; leave rest unchanged
                          └─► return ALL N flags, severity-sorted
```

**ABSOLUTE INVARIANT**: `len(flags_in) == len(flags_out)` always. The cap only limits what is sent to Gemini, never what is returned. Override the cap with `GEMINI_NARRATE_LIMIT` env var.

On any Gemini failure, the full flag list is returned severity-sorted with original rule-generated explanations.

## Deploy-readiness — Cloud Run

`Dockerfile` targets `api.routes:app` (the Flask app object). Cloud Run injects `$PORT`; gunicorn binds to it with 1 worker / 8 threads / 120-second timeout.

**No secrets baked in.** The container starts with:
- Cloud Run env vars: `MONGODB_URI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION`, `GEMINI_MODEL`, `REDUCTO_API_KEY` (optional — pipeline falls back to synthetic data if absent), `VERTEX_ENDPOINT_NAME` (optional — numerical layer skipped non-fatally if absent)
- Google auth via Application Default Credentials from the Cloud Run runtime service account — no `GOOGLE_APPLICATION_CREDENTIALS` file or env var needed in the container

All runtime code already uses `os.environ.get(...)` (no hard bracket access) — verified by grep.

## New architectural rules

**Rule**: `NARRATE_LIMIT` caps only the Gemini payload, never the returned flag list. Every flag that enters `narrate_and_rank` must exit it.
**Why**: A PM-facing report that silently drops flags is worse than a slow one. The cap is a latency optimization, not a filter.
**Consequence if violated**: Flags disappear from the report, breaking §6 rule 9 and destroying demo credibility.

**Rule**: The async-response check in `pdf_parser.py` uses duck-typing (`hasattr(response, "job_id")`), not a deep SDK type import.
**Why**: A deep import outside a try/except converts any SDK version change into a bare `ImportError` that escapes the `ReductoUnavailableError` handler, causing a 500 instead of a graceful fallback.
**Consequence if violated**: Any Reducto SDK update that restructures its type hierarchy breaks the PDF path silently for all callers.

## What's pending

- **Cloud Run deploy**: `gcloud run deploy constructiq-api --source . --region <REGION> --set-env-vars ...`
- **Agent Builder wiring**: register the `/audit` endpoint as a tool; wire MongoDB MCP for direct collection access
- **LICENSE + README** for the repo
- **Demo video** recording for hackathon submission (deadline June 11, 2026)
