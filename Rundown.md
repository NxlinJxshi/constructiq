# ConstructIQ — Codebase Rundown

> Onboarding reference for teammates and a cold-start context file for a Claude Code session. File paths are exact and quoted; the flag schema in §5 and rules in §6 are reflected from the actual code, not the design brief. Where this document and the seed/design comments disagree, this document follows the code.

## 1. What This System Does

ConstructIQ is an AI-powered construction timesheet audit agent that ingests HCSS HeavyJob timesheet PDFs, detects billing anomalies (missing data, statistical outliers, idle-equipment standby, mis-classified labor), and produces plain-English audit findings for a project manager. It serves construction PMs and replaces manual line-by-line review of daily timesheet cards. It is built for the **Google Agent Builder Hackathon 2026 (deadline June 11, 2026)**. A cold-start session should note: the ingestion, agent, API, and frontend layers are **docstring-only stubs** today (see §8); the detection layer and ML pipeline are fully built.

## 2. Pipeline Architecture

Intended flow: **Reducto OCR → structured records → MongoDB baseline lookup → three-layer detection → Gemini reasoning → audit report.** A timesheet PDF is sent to Reducto (`ingestion/pdf_parser.py`), normalized into canonical worker/equipment record dicts (`ingestion/normalizer.py`), then each record is checked against per-cost-code baseline documents in MongoDB. Three detection layers run, all emitting the unified flag schema (§5). The flag list is passed to a Vertex AI / Gemini agent (`agent/audit_agent.py`) for natural-language explanation, and findings are written to the `audit_reports` collection.

- **Categorical** (`detection/categorical_anomalies.py`): rule-based; catches labor-class mismatches, wrong equipment on a cost code, and wrong operator class for a given equipment type.
- **Equipment standby** (`detection/equipment_standby.py`): rule-based; catches equipment hours logged without matching operator hours (`eq_hours > operator_hours + 1.0`).
- **Numerical outliers** (`detection/numerical_outliers.py`): IsolationForest deployed on Vertex AI; catches hours outside the historical baseline per cost code and labor class.
- **Missing values** (`detection/missing_values.py`): rule-based presence check, the first gate.

**Why hybrid rule-based + ML:** structural anomalies (missing fields, categorical mismatches, standby pairs) have deterministic rules that need no training data and give ~100% recall; multivariate numerical outliers have no simple threshold — the anomaly boundary lives in a 10-feature space — so they require a model.

## 3. Repository Structure

```
constructiq/
├── README.md
├── requirements.txt
├── .env.example              # MONGODB_URI, REDUCTO_API_KEY, GOOGLE_APPLICATION_CREDENTIALS
├── .gitignore
├── main.py                   # STUB (docstring only) — top-level orchestrator
├── agent/
│   ├── __init__.py
│   ├── audit_agent.py        # STUB — Vertex/Gemini reasoning over flags
│   └── prompts.py            # STUB — prompt templates
├── api/
│   ├── __init__.py
│   └── routes.py             # STUB — Flask upload/report endpoints
├── database/
│   ├── __init__.py
│   ├── mongo_client.py       # BUILT — client + timesheets/baselines/audit_reports
│   ├── baselines.py          # STUB
│   └── timesheets.py         # STUB
├── detection/
│   ├── __init__.py
│   ├── missing_values.py         # BUILT
│   ├── categorical_anomalies.py  # BUILT
│   ├── equipment_standby.py      # BUILT
│   ├── feature_engineering.py    # BUILT — 10-feature matrix + AnomalyScorer
│   └── numerical_outliers.py     # BUILT — Vertex endpoint inference
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py         # STUB — Reducto OCR call
│   └── normalizer.py         # BUILT — HCSS cell parsing
├── frontend/
│   └── app.py                # STUB — Streamlit dashboard
├── scripts/
│   ├── seed_mongodb.py            # BUILT — seeds 5 baseline docs
│   ├── generate_synthetic_data.py # BUILT — synthetic + planted anomalies
│   ├── train_model.py             # BUILT — trains IsolationForest
│   └── deploy_model.py            # BUILT — Vertex deploy
├── tests/
│   ├── __init__.py
│   ├── test_normalizer.py
│   └── test_feature_engineering.py
├── data/synthetic/timesheets.json # GENERATED (gitignored) — 3829 records
└── artifacts/model.joblib         # GENERATED (gitignored) — trained artifact
```
(`__pycache__/`, `.pytest_cache/`, `.env`, `service_account.json` exist on disk but are gitignored.)

## 4. File Reference

| File path | What it does | What it contributes | Calls | Called by |
|---|---|---|---|---|
| `main.py` | Stub orchestrator (docstring only) | Planned top-level pipeline entry | — | standalone |
| `database/mongo_client.py` | Builds shared `MongoClient` with `certifi` TLS; exposes `timesheets`, `baselines`, `audit_reports`; `test_connection()` | Single DB access point | `os`, `certifi`, `dotenv`, `pymongo` | `detection.categorical_anomalies`, `detection.feature_engineering`, `scripts.seed_mongodb`, `scripts.generate_synthetic_data` |
| `database/baselines.py` | Stub | Planned baseline read helpers | — | standalone |
| `database/timesheets.py` | Stub | Planned timesheet CRUD | — | standalone |
| `ingestion/pdf_parser.py` | Stub | Planned Reducto OCR call | — | standalone |
| `ingestion/normalizer.py` | Parses HCSS cells (`"8/2.5"`, `"/1"`), builds canonical worker/equipment records | Turns OCR rows into typed dicts | `uuid` | `tests.test_normalizer` |
| `detection/missing_values.py` | Flags null/empty required fields per entity; field-based severity | First detection gate | `json`, `uuid` | (intended) `agent.audit_agent` |
| `detection/categorical_anomalies.py` | Labor-class / equipment / operator-pairing rule checks vs baselines | Categorical layer | `database.mongo_client.baselines`, `uuid` | (intended) pipeline |
| `detection/equipment_standby.py` | Per-operator + group-avg standby detection | Standby layer | `collections`, `uuid` | (intended) pipeline |
| `detection/feature_engineering.py` | Builds `(N,10)` matrix; `build_encoders()`; `AnomalyScorer` wrapper | Feature prep for IsolationForest | `numpy`, `database.mongo_client.baselines` | `scripts.train_model`, `scripts.deploy_model`, `detection.numerical_outliers`, `tests.test_feature_engineering` |
| `detection/numerical_outliers.py` | Scores workers via Vertex endpoint in batches of 100; flags below threshold | Numerical layer (online inference) | `numpy`, `joblib`, `dotenv`, `google.cloud.aiplatform`, `detection.feature_engineering` | standalone (CLI) |
| `agent/audit_agent.py` | Stub | Planned Gemini reasoning | — | standalone |
| `agent/prompts.py` | Stub | Planned prompt templates | — | standalone |
| `api/routes.py` | Stub | Planned Flask routes | — | standalone |
| `frontend/app.py` | Stub | Planned Streamlit UI | — | standalone |
| `scripts/seed_mongodb.py` | Inserts 5 hard-coded baseline docs | Seeds reference data | `database.mongo_client.baselines` | standalone |
| `scripts/generate_synthetic_data.py` | Simulates 200 days / 12 workers; plants 4 labeled anomaly types | Ground-truth dataset | `numpy`, `database.mongo_client.baselines` | standalone |
| `scripts/train_model.py` | Trains IsolationForest, evals recall, saves `model.joblib` | Produces ML artifact | `joblib`, `sklearn`, `detection.feature_engineering` | standalone |
| `scripts/deploy_model.py` | Uploads scorer to GCS, registers model, deploys Vertex endpoint, writes `VERTEX_ENDPOINT_NAME` | Vertex deployment | `joblib`, `dotenv`, `google.cloud.aiplatform`, `google.cloud.storage`, `detection.feature_engineering` | standalone |
| `tests/test_normalizer.py` | Tests all cell formats + `normalize_timesheet` | Ingestion test | `ingestion.normalizer` | pytest |
| `tests/test_feature_engineering.py` | Tests encoder determinism, shapes, skip logic, overnight shifts | Feature test | `detection.feature_engineering` | pytest |

## 5. Unified Flag Schema

This is the contract between all three detectors and the downstream Gemini call. **Treat as fixed.** This is the schema the code actually emits (it differs from the design brief — there is no top-level `worker_id`/`cost_code`/`expected`; record identity is carried by `record_id`, and expected/actual values live in `details`).

```python
{
  "flag_id":     str,   # uuid4 per flag
  "record_id":   str,   # FK to record_id in the timesheet record (worker or equipment)
  "flag_type":   str,   # "missing_value" | "numerical_outlier" | "equipment_standby"
                        #   | "labor_class_anomaly" | "equipment_on_wrong_code"
                        #   | "wrong_operator_for_equipment"
  "severity":    str,   # "high" | "medium" | "low"
  "entity_type": str,   # "worker" | "equipment"
  "field":       str,   # which field triggered the flag (e.g. "total_hours", "labor_class", "hours")
  "observed":    Any,   # value found in the record (None for missing_value)
  "explanation": str,   # human-readable; passed to Gemini for NL generation
  "details":     dict,  # OPTIONAL; present on categorical/standby/numerical flags.
                        #   carries {"actual":..., "expected":...} or
                        #   {"anomaly_score","threshold","score_delta"} or
                        #   {"equipment_hours","operator_hours","delta_hours","check_method","grace_window_hours"}
}
```
`missing_values.detect()` omits `details`; the other detectors include it. Timesheet records carry `anomaly_type`/`anomaly_detail` ground-truth labels only in synthetic data.

## 6. Architectural Rules

**Rule**: Label encoders are derived from **sorted** MongoDB baseline values (`detection/feature_engineering.py::build_encoders`), never fitted on training data.
**Why**: Sorting the baseline cost-code/labor-class sets gives a deterministic integer mapping reproducible at train and inference time with no serialized encoder.
**Consequence if violated**: Train/inference encodings drift, `cost_code_id`/`labor_class_id` features shift, and the IsolationForest scores garbage — silent recall collapse.

**Rule**: The IsolationForest artifact must be named `model.joblib`, and `artifact_uri` must point to its **directory**, not the file (`scripts/deploy_model.py`: blob `constructiq-v1/model.joblib`, `artifact_uri="gs://<bucket>/constructiq-v1/"`).
**Why**: The Vertex sklearn pre-built container loads `model.joblib` from the directory given by `artifact_uri`.
**Consequence if violated**: Model registration/deploy fails or the endpoint serves nothing.

**Rule**: The anomaly threshold is `model.offset_`, set in `scripts/train_model.py` (`threshold = float(model.offset_)`, saved as `artifact["threshold"]`) and consumed in `detection/numerical_outliers.py` (`threshold = float(artifact["threshold"])`).
**Why**: `offset_` is the `score_samples` value at `contamination=0.05`; both sides must use the same number.
**Consequence if violated**: Flagging boundary moves; recall/FP rate become meaningless.

**Rule**: `avg_overtime_hours` is read directly from the baseline document, never derived as `total_hours - 8.0`.
**Why**: Foreman admin (`73020.7000`) runs 0.5 OT while field codes run 2.5; the `8.0` assumption is false for some codes.
**Consequence if violated**: Synthetic OT sampling and outlier baselines skew, mislabeling normal foreman rows as outliers.

**Rule**: All three detectors must emit flags conforming to §5; the Gemini call is written against that schema.
**Why**: A single schema lets the agent reason uniformly over every flag type.
**Consequence if violated**: Downstream reasoning/serialization breaks on missing keys.

**Rule**: Records are scored via the Vertex endpoint in **batches of 100** (`detection/numerical_outliers.py::_chunks(..., 100)`).
**Why**: Bounds per-request payload size for online prediction.
**Consequence if violated**: Oversized requests are rejected by the endpoint.

## 7. Known Limitations

- **Numerical-outlier recall (~25%)**: Equipment-standby records are more statistically extreme than numerical outliers in the 10-feature space, so the IsolationForest assigns anomaly mass to standby workers, crowding out the numerical-outlier signal. *Mitigation*: the demo foregrounds categorical detection.
- **Equipment-standby recall (~21%, FP 0.23%)**: The per-operator primary check plus group-average fallback (`equipment_standby.py`) does not cover all edge cases. Acceptable for hackathon demo scope.

## 8. What Is Not Yet Built

These exist as **docstring-only stubs** (or are entirely absent) and must be implemented. Paths below are where the functionality lives/will live:

- `main.py` — single callable chaining all detectors into a unified flag list (the design brief calls this `agent/pipeline.py`; in this repo it belongs in `main.py`).
- `api/routes.py` — Flask endpoints wrapping the pipeline; required for Agent Builder to invoke the system over HTTP (brief refers to `api/main.py`).
- `ingestion/pdf_parser.py` — Reducto OCR call converting HCSS PDF export to raw rows (brief refers to `ingestion/reducto_client.py`).
- `agent/audit_agent.py` + `agent/prompts.py` — take the flag list, call Gemini, return a natural-language explanation per flag (brief refers to `agent/gemini_reasoning.py`).
- `frontend/app.py` — Streamlit audit report UI; ranked flag display legible to non-technical judges in under 30 seconds.
- `database/baselines.py`, `database/timesheets.py` — read/write helpers (detectors currently query collections directly via `database.mongo_client`).

## 9. Commit History Summary

| Commit | Message | What it added/changed | Why it matters |
|---|---|---|---|
| `849212e` | Initial scaffold for ConstructIQ | Package skeleton + stub modules | Establishes the 4-stage layout (ingestion→detection→agent→api) the rest fills in |
| `422b30e` | Add inline comments, header blocks, and real HCSS baseline data | Module headers + initial HCSS baselines | Grounds the system in real timesheet structure, not toy data |
| `6a63419` | Refine baseline schema with real HCSS timesheet structure | Finalized baseline doc fields (equipment pairs, units, OT per code) | Defines the baseline contract that detectors and the encoder depend on (§6) |
| `7538ba3` | Fix MongoDB Atlas SSL connection on macOS Python 3.9 | `certifi` `tlsCAFile` in `mongo_client.py` | Without it the python.org installer can't reach Atlas — blocks every DB call |
| `edf7829` | Implement normalizer, missing-value detector, and synthetic data generator | `normalizer.py`, `missing_values.py`, `generate_synthetic_data.py` | Produces labeled ground truth and the first working detector |
| `2746c03` | Add Day 2 detection pipeline: feature engineering, model training, Vertex AI deploy | `feature_engineering.py`, `categorical_anomalies.py`, `equipment_standby.py`, `numerical_outliers.py`, `train_model.py`, `deploy_model.py` | Completes all three detection layers and the ML deploy path — the core of the system |

## 10. Environment and Dependencies

**Environment variables** (`.env`, see `.env.example`):
- `MONGODB_URI` — Atlas connection string (`database/mongo_client.py`)
- `REDUCTO_API_KEY` — Reducto OCR (planned ingestion)
- `GOOGLE_APPLICATION_CREDENTIALS` — path to `service_account.json`
- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION` — required by `deploy_model.py` / `numerical_outliers.py`
- `VERTEX_ENDPOINT_NAME` — appended to `.env` by `deploy_model.py`; consumed by `numerical_outliers.py`

**External services**: MongoDB Atlas (`constructiq` db: `timesheets`, `baselines`, `audit_reports`); Reducto OCR API; Google Cloud — Vertex AI (Model Registry + online endpoint, sklearn-cpu.1-3 container), Cloud Storage (`<project>-constructiq-models` bucket), Gemini (planned).

**Python dependencies** (`requirements.txt`): `pymongo[srv]`, `python-dotenv`, `certifi`, `python-dateutil`, `requests`, `flask`, `streamlit`, `google-cloud-aiplatform`, `vertexai`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `google-cloud-storage`.
