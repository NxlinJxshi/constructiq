# ============================================================
# MODULE: frontend/app.py
# PURPOSE: Streamlit demo UI — lets a project manager (or a hackathon judge)
#          upload an HCSS timesheet PDF or load the built-in sample data,
#          run the full audit pipeline, and see ranked, plain-English findings
#          in under 30 seconds.
# PIPELINE STAGE: Frontend
# ============================================================

"""
Streamlit audit dashboard.

Two entry paths:
  1. "Use sample timesheet" button (default / no key needed):
     Loads data/synthetic/timesheets.json → run_audit → renders report.
  2. PDF upload:
     parse_timesheet_pdf → normalize_timesheet → run_audit → renders report.
     If REDUCTO_API_KEY is not configured, shows an info banner and falls back
     to the sample path.

Design principle: a non-technical judge must understand the output in under 30
seconds.  Severity is communicated with colour (red/amber/grey), high-severity
flags appear first, and every card shows who is affected, what was wrong, and
why it matters.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


def _bootstrap_secrets() -> None:
    """Populate os.environ from Streamlit secrets when running on Streamlit
    Community Cloud (no .env file there — config comes from st.secrets).

    Locally, .env (loaded via python-dotenv in main.py / database modules)
    already populates os.environ, and st.secrets is empty, so this is a no-op.
    """
    try:
        secrets = st.secrets
    except Exception:
        return

    for key in (
        "MONGODB_URI", "REDUCTO_API_KEY", "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION", "VERTEX_ENDPOINT_NAME", "GEMINI_MODEL",
    ):
        if key in secrets and not os.environ.get(key):
            os.environ[key] = str(secrets[key])

    # Service account JSON is stored as a secret block and written to a temp
    # file at startup so GOOGLE_APPLICATION_CREDENTIALS can point at it (the
    # Vertex/Gemini SDKs require a file path, not inline JSON).
    if "GOOGLE_SERVICE_ACCOUNT_JSON" in secrets and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        creds_path = os.path.join(tempfile.gettempdir(), "constructiq_service_account.json")
        with open(creds_path, "w") as f:
            json.dump(dict(secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]), f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


_bootstrap_secrets()

# ── Severity colours ──────────────────────────────────────────────────────────
_SEVERITY_COLOR = {
    "high":   "#FF4B4B",
    "medium": "#FFA500",
    "low":    "#808080",
}

# ── Human-readable flag type labels ──────────────────────────────────────────
_FLAG_LABEL = {
    "missing_value":               "Missing required field",
    "numerical_outlier":           "Hours outside historical norm",
    "equipment_standby":           "Equipment idle billing",
    "labor_class_anomaly":         "Labor classification mismatch",
    "equipment_on_wrong_code":     "Wrong equipment for cost code",
    "wrong_operator_for_equipment": "Wrong operator for equipment",
}

_SYNTHETIC_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "timesheets.json"

# Single-day slice of the synthetic dataset used for the demo "sample timesheet"
# button. The full dataset is 3829 records / 200 days (built for ML training);
# sending all of it through Gemini narration produces 400+ flags in one prompt,
# which is far too slow for a live demo. This date has a realistic ~20-record
# day with all four anomaly types represented.
_SAMPLE_DATE = "2025-01-11"


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _run_on_records(records: list[dict]) -> dict:
    """Call main.run_audit and return the report dict."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import run_audit
    return run_audit(records)


def _load_sample_and_run() -> dict:
    """Load synthetic data and run the full pipeline."""
    if not _SYNTHETIC_PATH.exists():
        st.error(
            f"Sample data not found at `{_SYNTHETIC_PATH}`.  "
            "Run `python3 scripts/generate_synthetic_data.py` first."
        )
        st.stop()
    with open(_SYNTHETIC_PATH) as f:
        records = json.load(f)
    records = [r for r in records if r.get("date") == _SAMPLE_DATE]
    return _run_on_records(records)


def _run_on_pdf(tmp_path: str, project_id: str, date_str: str) -> dict:
    """Parse a PDF with Reducto, normalise, and run the pipeline.

    Falls back to sample data on ReductoUnavailableError.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.pdf_parser import parse_timesheet_pdf, ReductoUnavailableError
    from ingestion.normalizer import normalize_timesheet

    try:
        raw_rows = parse_timesheet_pdf(tmp_path)
        records  = normalize_timesheet(raw_rows, project_id, date_str)
        return _run_on_records(records)
    except ReductoUnavailableError as exc:
        st.info(
            f"OCR key not configured — using sample data.  ({exc})"
        )
        return _load_sample_and_run()


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_flag_card(flag: dict) -> None:
    """Render a single flag as a colour-coded card."""
    sev     = flag.get("severity", "low")
    color   = _SEVERITY_COLOR.get(sev, "#808080")
    label   = _FLAG_LABEL.get(flag.get("flag_type", ""), flag.get("flag_type", "Unknown"))
    details = flag.get("details") or {}

    with st.container(border=True):
        # Severity stripe + type label on one line
        st.markdown(
            f'<span style="color:{color}; font-weight:700; font-size:1.05em;">'
            f'● {sev.upper()}</span>&nbsp;&nbsp;'
            f'<span style="font-weight:600;">{label}</span>',
            unsafe_allow_html=True,
        )
        st.write(flag.get("explanation", ""))

        # Observed / expected details when available
        if "actual" in details and "expected" in details:
            col1, col2 = st.columns(2)
            col1.caption(f"**Observed:** {details['actual']}")
            col2.caption(f"**Expected:** {details['expected']}")
        elif "equipment_hours" in details and "operator_hours" in details:
            col1, col2 = st.columns(2)
            col1.caption(f"**Equipment hours:** {details['equipment_hours']}")
            col2.caption(f"**Operator hours:** {details['operator_hours']}")
        elif "anomaly_score" in details:
            st.caption(
                f"Anomaly score: {details['anomaly_score']:.4f}  "
                f"(threshold: {details['threshold']:.4f})"
            )


def _render_report(report: dict) -> None:
    """Render the full audit report: metrics, summary, ranked flag list."""
    counts  = report.get("counts", {})
    flags   = report.get("flags", [])
    summary = report.get("summary", "")
    total   = report.get("total_records", 0)

    # ── Top metrics row ───────────────────────────────────────────────────────
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Records Scanned",  total)
    c1.metric("🔴 High",   counts.get("high",   0))
    c2.metric("🟠 Medium", counts.get("medium", 0))
    c3.metric("⚪ Low",    counts.get("low",    0))

    st.divider()

    # ── Audit summary ─────────────────────────────────────────────────────────
    if summary:
        st.subheader("Audit Summary")
        st.write(summary)
        st.divider()

    # ── Flags grouped by severity ─────────────────────────────────────────────
    if not flags:
        st.success("No anomalies found in this timesheet.")
        return

    st.subheader(f"Findings ({len(flags)} total)")

    for sev_label, sev_key in [("High Severity", "high"), ("Medium Severity", "medium"), ("Low Severity", "low")]:
        group = [f for f in flags if f.get("severity") == sev_key]
        if not group:
            continue
        color = _SEVERITY_COLOR[sev_key]
        st.markdown(
            f'<h4 style="color:{color};">{sev_label} ({len(group)})</h4>',
            unsafe_allow_html=True,
        )
        for flag in group:
            _render_flag_card(flag)


# ── Page layout ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="ConstructIQ — Timesheet Audit",
        page_icon="🏗️",
        layout="wide",
    )

    st.title("🏗️ ConstructIQ — Timesheet Audit")
    st.caption(
        "AI-powered HCSS HeavyJob timesheet audit agent · "
        "Google Agent Builder Hackathon 2026"
    )

    st.divider()

    # ── Input controls ────────────────────────────────────────────────────────
    col_btn, col_upload = st.columns([1, 2])

    with col_btn:
        use_sample = st.button(
            "▶ Use sample timesheet",
            type="primary",
            help="Runs the full audit pipeline on built-in synthetic data — no PDF or API key needed.",
            use_container_width=True,
        )

    with col_upload:
        uploaded_pdf = st.file_uploader(
            "Or upload an HCSS HeavyJob PDF or screenshot",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Requires REDUCTO_API_KEY in .env.  Falls back to sample data if not configured.",
            label_visibility="collapsed",
        )

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if use_sample:
        with st.spinner("Running audit on sample timesheet data…"):
            report = _load_sample_and_run()
        _render_report(report)

    elif uploaded_pdf is not None:
        project_id = st.text_input("Project ID (optional)", value="unknown")
        date_str   = st.text_input("Shift date YYYY-MM-DD (optional)", value="unknown")

        if st.button("Run audit on uploaded PDF", type="primary"):
            suffix = os.path.splitext(uploaded_pdf.name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Extracting timesheet and running audit…"):
                    report = _run_on_pdf(tmp_path, project_id, date_str)
                _render_report(report)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    else:
        st.info(
            "Click **▶ Use sample timesheet** to see the full pipeline run immediately, "
            "or upload an HCSS PDF to audit your own data."
        )


if __name__ == "__main__":
    main()
