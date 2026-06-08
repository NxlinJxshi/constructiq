# Branch: feat/day3-agent-pipeline-reducto

## What this branch does

Turns the standalone detection modules into a working end-to-end agent: an HCSS HeavyJob timesheet PDF (or a built-in synthetic dataset) goes in, and a ranked, plain-English audit report comes out. Adds Reducto Extract integration for PDF ingestion with a graceful synthetic-data fallback, wires all four detectors through a single pipeline function, adds Gemini narration via Vertex AI, and exposes the pipeline through a Flask HTTP endpoint and a Streamlit demo UI.

## New and changed files

| File | Status | What it does | Calls | Called by |
|---|---|---|---|---|
| `ingestion/pdf_parser.py` | Changed (stub → full) | Uploads a PDF to Reducto Extract; returns raw HCSS row dicts for the normalizer. Raises `ReductoUnavailableError` if key is missing or call fails. | `reducto.Reducto` | `api.routes`, `frontend.app` |
| `agent/prompts.py` | Changed (stub → full) | Constants-only: `AUDIT_SYSTEM_INSTRUCTION` and `AUDIT_USER_TEMPLATE` for Gemini | — | `agent.audit_agent` |
| `agent/audit_agent.py` | Changed (stub → full) | `narrate_and_rank(flags)` calls Gemini 2.0 Flash via Vertex AI; enriches explanations; sorts by severity; falls back to original flags on any failure | `vertexai.GenerativeModel`, `agent.prompts` | `main` |
| `main.py` | Changed (stub → full) | `run_audit(records)` — chains 4 detectors, validates §5 schema, calls Gemini, writes to `audit_reports`, returns report dict | `detection.*`, `agent.audit_agent`, `database.mongo_client` | `api.routes`, `frontend.app` |
| `api/routes.py` | Changed (stub → full) | Flask `POST /audit` (PDF or JSON records) + `GET /health` | `main.run_audit`, `ingestion.pdf_parser`, `ingestion.normalizer` | Agent Builder, HTTP clients |
| `frontend/app.py` | Changed (stub → full) | Streamlit demo UI: sample button (primary), PDF upload, spinner, severity-coloured flag cards | `main.run_audit`, `ingestion.pdf_parser`, `ingestion.normalizer` | `streamlit run frontend/app.py` |
| `requirements.txt` | Changed | Added `reductoai` | — | — |
| `.env.example` | Changed | Added `GOOGLE_CLOUD_PROJECT=`, `GOOGLE_CLOUD_REGION=`, `VERTEX_ENDPOINT_NAME=` | — | — |

## New data flow

```
PDF upload
  └─► ingestion.pdf_parser.parse_timesheet_pdf  (Reducto Extract, array_extract=True)
        └─► ingestion.normalizer.normalize_timesheet
              └─► main.run_audit ───────────────────────────────────────────────┐
                    ├─ detection.missing_values.detect                          │
                    ├─ detection.categorical_anomalies.detect                   │
                    ├─ detection.equipment_standby.detect                       │
                    └─ detection.numerical_outliers.detect  (non-fatal)         │
                          └─ agent.audit_agent.narrate_and_rank (Gemini)        │
                                └─► database.mongo_client.audit_reports ────────┘
                                      └─► Flask JSON response / Streamlit UI

Synthetic fallback (no Reducto key):
  data/synthetic/timesheets.json → main.run_audit  (same path from normalize step)
  Triggered by: Streamlit "Use sample timesheet" button,
                python3 main.py CLI,
                or caught ReductoUnavailableError in pdf path
```

## New architectural rules

**Rule**: Reducto Extract output feeds `ingestion.normalizer`; it does NOT bypass it.
**Why**: The normalizer owns all HCSS cell parsing ("8/2.5", "/1", etc.) and is covered by 24 passing tests. Bypassing it would skip the only tested ingestion path.
**Consequence if violated**: Downstream detection receives unparsed strings instead of typed floats; every numeric check silently fails.

**Rule**: The numerical-outlier layer is non-fatal. `main.run_audit` wraps `detection.numerical_outliers.detect` in try/except; a Vertex outage skips numerical flags and the audit continues.
**Why**: The hackathon demo foregrounds categorical detection (§7 of Rundown.md); Vertex availability must not block a live demo.
**Consequence if violated**: A Vertex timeout or missing `VERTEX_ENDPOINT_NAME` takes down the entire audit.

**Rule**: Gemini narrates and ranks only. `len(flags_in) == len(flags_out)` is asserted inside `audit_agent.narrate_and_rank` before returning; any violation triggers a fallback to the original flags.
**Why**: Detection is deterministic and owned by the detection layer. An LLM inventing anomalies would undermine the system's credibility with judges who know the ground truth.
**Consequence if violated**: Fabricated flags reach the PM-facing UI and destroy demo credibility.

**Rule**: `array_extract: True` is mandatory in the Reducto Extract call (`ingestion/pdf_parser.py`).
**Why**: Without it, rows toward the bottom of a 15–30 row HCSS timesheet are truncated.
**Consequence if violated**: Partial timesheet extraction; lower portions of the workday silently disappear.

**Rule**: One Google credential path — `audit_agent.py` uses the same `GOOGLE_APPLICATION_CREDENTIALS` / `service_account.json` as the Vertex IsolationForest endpoint. No separate Gemini API key.
**Why**: Minimises secret surface and keeps the GCP project consistent across ML and LLM calls.
**Consequence if violated**: Two credential paths to rotate, two projects to bill, two quota limits to manage.

## New environment variables

- `REDUCTO_API_KEY` — Reducto Extract API key. Pipeline falls back to synthetic data if absent; no code change needed when the key is added.
- `GOOGLE_CLOUD_PROJECT` — GCP project ID (already used by Vertex; now also required by Gemini init).
- `GOOGLE_CLOUD_REGION` — GCP region (e.g. `us-central1`; already used by Vertex).
- `VERTEX_ENDPOINT_NAME` — Vertex AI endpoint resource name for the IsolationForest model (pre-existing; written by `scripts/deploy_model.py`).

Gemini reuses the `GOOGLE_APPLICATION_CREDENTIALS` / `service_account.json` already in use.

## If you're picking this up

**What's done:** All five stubs are fully implemented. The pipeline runs end-to-end on synthetic data today (`python3 main.py`). The Streamlit demo UI works with no PDF and no Reducto key — click "Use sample timesheet".

**What's pending:**
- Real Reducto key from Marco's org — add to `.env` as `REDUCTO_API_KEY=...`; zero code change required.
- Live Vertex endpoint — deploy with `python3 scripts/deploy_model.py` and set `VERTEX_ENDPOINT_NAME` in `.env`; numerical flags will then appear in the report.
- A real HCSS PDF to validate the Reducto extraction schema against an actual timesheet card.

**To see it run immediately:**
```bash
# Full pipeline on synthetic data (no keys needed):
python3 main.py

# Streamlit demo:
streamlit run frontend/app.py
# Then click "▶ Use sample timesheet"
```
