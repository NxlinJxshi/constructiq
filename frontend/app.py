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

import altair as alt
import pandas as pd
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
    "high":   "#EF4444",
    "medium": "#F59E0B",
    "low":    "#94A3B8",
}
_SEVERITY_ORDER = ["high", "medium", "low"]
_SEVERITY_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}

# ── Human-readable flag type labels + icons ───────────────────────────────────
_FLAG_LABEL = {
    "missing_value":               "Missing required field",
    "numerical_outlier":           "Hours outside historical norm",
    "equipment_standby":           "Equipment idle billing",
    "labor_class_anomaly":         "Labor classification mismatch",
    "equipment_on_wrong_code":     "Wrong equipment for cost code",
    "wrong_operator_for_equipment": "Wrong operator for equipment",
}
_FLAG_ICON = {
    "missing_value":               "❓",
    "numerical_outlier":           "📈",
    "equipment_standby":           "🚜",
    "labor_class_anomaly":         "👷",
    "equipment_on_wrong_code":     "🔧",
    "wrong_operator_for_equipment": "🧑‍🔧",
}

_SYNTHETIC_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "timesheets.json"

# Single-day slice of the synthetic dataset used for the demo "sample timesheet"
# button. The full dataset is 3829 records / 200 days (built for ML training);
# sending all of it through Gemini narration produces 400+ flags in one prompt,
# which is far too slow for a live demo. This date has a realistic ~20-record
# day with all four anomaly types represented.
_SAMPLE_DATE = "2025-01-11"


# ── Look & feel ────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .ciq-hero {
            background: linear-gradient(135deg, #0F2A43 0%, #1B3A5C 55%, #FF6A00 160%);
            border-radius: 16px;
            padding: 2rem 2.25rem;
            color: #FFFFFF;
            margin-bottom: 1.25rem;
        }
        .ciq-hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.2;
        }
        .ciq-hero p {
            margin: 0.4rem 0 0;
            opacity: 0.88;
            font-size: 1.02rem;
        }
        .ciq-badge-row { margin-top: 0.85rem; }
        .ciq-badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            font-size: 0.78rem;
            margin-right: 0.4rem;
            backdrop-filter: blur(2px);
        }
        .ciq-metric-card {
            border: 1px solid rgba(127,127,127,0.18);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            text-align: center;
            background: rgba(127,127,127,0.04);
        }
        .ciq-metric-value {
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .ciq-metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.65;
            margin-top: 0.2rem;
        }
        .ciq-flag-card {
            border-left: 5px solid var(--ciq-sev-color, #94A3B8);
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.7rem;
            background: rgba(127,127,127,0.045);
        }
        .ciq-flag-head {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 1.0rem;
            margin-bottom: 0.35rem;
        }
        .ciq-sev-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.05rem 0.6rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: white;
        }
        .ciq-empty-state {
            text-align: center;
            padding: 2.5rem 1rem;
            opacity: 0.75;
        }
        .ciq-empty-state .ciq-emoji {
            font-size: 2.6rem;
            display: block;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="ciq-hero">
            <h1>🏗️ ConstructIQ — Timesheet Audit</h1>
            <p>AI agent that reads HCSS HeavyJob timesheets, checks every line
            against project baselines, and explains what's wrong in plain
            English — in under 30 seconds.</p>
            <div class="ciq-badge-row">
                <span class="ciq-badge">⚡ Vertex AI · Gemini</span>
                <span class="ciq-badge">🍃 MongoDB Atlas</span>
                <span class="ciq-badge">📄 Reducto Extract</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🏗️ ConstructIQ")
        st.caption("Google Agent Builder Hackathon 2026 · MongoDB track")
        st.markdown("---")
        st.markdown("#### How it works")
        st.markdown(
            "1. **Ingest** — Reducto Extract parses the HCSS timesheet PDF "
            "or screenshot into structured rows.\n"
            "2. **Detect** — four checks run against project baselines: "
            "missing fields, labor/equipment mismatches, equipment standby, "
            "and ML-based hour outliers (Vertex AI).\n"
            "3. **Narrate** — Gemini 2.5 Flash rewrites each flag in plain "
            "English, ranks them by severity, and writes an executive "
            "summary.\n"
            "4. **Review** — every flag, baseline, and report is stored in "
            "MongoDB Atlas."
        )
        st.markdown("---")
        st.markdown(
            "📂 [Source on GitHub](https://github.com/NxlinJxshi/constructiq)"
        )


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


def _run_with_status(fn, *args) -> dict:
    """Run a pipeline function inside a st.status block with friendly steps."""
    with st.status("Running ConstructIQ audit…", expanded=True) as status:
        st.write("📄 Parsing timesheet into structured records…")
        st.write("🔎 Running detection checks (missing data, classifications, standby, outliers)…")
        st.write("🤖 Asking Gemini to narrate findings and write a summary…")
        report = fn(*args)
        status.update(label="Audit complete", state="complete", expanded=False)
    return report


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _metric_card(label: str, value, color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return (
        f'<div class="ciq-metric-card">'
        f'<div class="ciq-metric-value"{style}>{value}</div>'
        f'<div class="ciq-metric-label">{label}</div>'
        f'</div>'
    )


def _severity_chart(counts: dict) -> alt.Chart | None:
    rows = [
        {"Severity": _SEVERITY_LABEL[k], "Count": counts.get(k, 0), "order": i}
        for i, k in enumerate(_SEVERITY_ORDER)
        if counts.get(k, 0) > 0
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_arc(innerRadius=55, outerRadius=100)
        .encode(
            theta=alt.Theta("Count:Q", stack=True),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(
                    domain=["High", "Medium", "Low"],
                    range=[_SEVERITY_COLOR["high"], _SEVERITY_COLOR["medium"], _SEVERITY_COLOR["low"]],
                ),
                legend=alt.Legend(title="Severity", orient="bottom"),
            ),
            order=alt.Order("order:Q"),
            tooltip=["Severity", "Count"],
        )
        .properties(height=240)
    )


def _render_flag_card(flag: dict) -> None:
    """Render a single flag as a colour-coded card with expandable detail."""
    sev     = flag.get("severity", "low")
    color   = _SEVERITY_COLOR.get(sev, "#94A3B8")
    flag_type = flag.get("flag_type", "")
    icon    = _FLAG_ICON.get(flag_type, "⚠️")
    label   = _FLAG_LABEL.get(flag_type, flag_type or "Unknown")
    details = flag.get("details") or {}

    st.markdown(
        f"""
        <div class="ciq-flag-card" style="--ciq-sev-color: {color};">
            <div class="ciq-flag-head">
                <span class="ciq-sev-pill" style="background:{color};">{sev.upper()}</span>
                <span>{icon} {label}</span>
            </div>
            <div>{flag.get("explanation", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


def _render_findings(flags: list[dict]) -> None:
    """Filterable, searchable list of flags."""
    sev_options = [s for s in _SEVERITY_ORDER if any(f.get("severity") == s for f in flags)]
    type_options = sorted({f.get("flag_type", "") for f in flags})

    fcol1, fcol2 = st.columns([2, 3])
    with fcol1:
        selected_sevs = st.pills(
            "Severity",
            options=sev_options,
            format_func=lambda s: f"{_SEVERITY_LABEL[s]}",
            selection_mode="multi",
            default=sev_options,
        )
    with fcol2:
        selected_types = st.multiselect(
            "Issue type",
            options=type_options,
            default=type_options,
            format_func=lambda t: f"{_FLAG_ICON.get(t, '⚠️')} {_FLAG_LABEL.get(t, t)}",
        )

    search = st.text_input("Search findings (worker name, ID, cost code…)", "")

    filtered = [
        f for f in flags
        if f.get("severity") in (selected_sevs or [])
        and f.get("flag_type") in (selected_types or [])
        and (search.lower() in f.get("explanation", "").lower() if search else True)
    ]

    if not filtered:
        st.markdown(
            '<div class="ciq-empty-state"><span class="ciq-emoji">🔍</span>'
            'No findings match the current filters.</div>',
            unsafe_allow_html=True,
        )
        return

    st.caption(f"Showing {len(filtered)} of {len(flags)} findings")

    for sev_label, sev_key in [("High Severity", "high"), ("Medium Severity", "medium"), ("Low Severity", "low")]:
        group = [f for f in filtered if f.get("severity") == sev_key]
        if not group:
            continue
        color = _SEVERITY_COLOR[sev_key]
        st.markdown(
            f'<h4 style="color:{color};">{sev_label} ({len(group)})</h4>',
            unsafe_allow_html=True,
        )
        for flag in group:
            _render_flag_card(flag)


def _render_report(report: dict) -> None:
    """Render the full audit report: metrics, summary, ranked flag list."""
    counts  = report.get("counts", {})
    flags   = report.get("flags", [])
    summary = report.get("summary", "")
    total   = report.get("total_records", 0)

    st.divider()

    overview_tab, findings_tab = st.tabs([
        "📊 Overview",
        f"🔍 Findings ({len(flags)})",
    ])

    with overview_tab:
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(_metric_card("Records Scanned", total), unsafe_allow_html=True)
        m2.markdown(_metric_card("High", counts.get("high", 0), _SEVERITY_COLOR["high"]), unsafe_allow_html=True)
        m3.markdown(_metric_card("Medium", counts.get("medium", 0), _SEVERITY_COLOR["medium"]), unsafe_allow_html=True)
        m4.markdown(_metric_card("Low", counts.get("low", 0), _SEVERITY_COLOR["low"]), unsafe_allow_html=True)

        st.write("")

        chart_col, summary_col = st.columns([1, 2])
        with chart_col:
            chart = _severity_chart(counts)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.markdown(
                    '<div class="ciq-empty-state"><span class="ciq-emoji">✅</span>'
                    'No issues found.</div>',
                    unsafe_allow_html=True,
                )
        with summary_col:
            if summary:
                st.markdown("##### Audit Summary")
                st.info(summary)

        st.download_button(
            "⬇️ Download full report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"constructiq_audit_{report.get('report_id', 'report')}.json",
            mime="application/json",
        )

    with findings_tab:
        if not flags:
            st.markdown(
                '<div class="ciq-empty-state"><span class="ciq-emoji">✅</span>'
                'No anomalies found in this timesheet.</div>',
                unsafe_allow_html=True,
            )
        else:
            _render_findings(flags)


# ── Page layout ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="ConstructIQ — Timesheet Audit",
        page_icon="🏗️",
        layout="wide",
    )

    _inject_css()
    _render_sidebar()
    _render_hero()

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
        report = _run_with_status(_load_sample_and_run)
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
                report = _run_with_status(_run_on_pdf, tmp_path, project_id, date_str)
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
