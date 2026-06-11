# ============================================================
# MODULE: frontend/app.py
# PURPOSE: Streamlit demo UI — a dark "operations command" dashboard for
#          the ConstructIQ audit agent. Four views: Dashboard (KPIs +
#          upload), Audit Report detail, Project Baselines, Audit History.
#          Wired to live MongoDB Atlas data where available.
# PIPELINE STAGE: Frontend
# ============================================================

"""
Streamlit audit dashboard — operations-command edition.

Navigation (sidebar): Dashboard · Audit Report · Baselines · History.

Two audit entry paths (Dashboard page):
  1. "Run sample audit" button (default / no key needed):
     Loads data/synthetic/timesheets.json → run_audit → opens report view.
  2. PDF upload:
     parse_timesheet_pdf → normalize_timesheet → run_audit → report view.
     If REDUCTO_API_KEY is not configured, falls back to the sample path.

Live data: KPI cards, the anomaly trend chart, History, and Baselines read
from MongoDB Atlas (audit_reports / baselines collections) and degrade
gracefully when Atlas is unreachable.

The current report and audited records live in st.session_state so widget
interactions survive Streamlit reruns.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
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

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Palette ───────────────────────────────────────────────────────────────────
_ORANGE = "#FF7A1A"
_GREEN  = "#2BD576"
_RED    = "#FF4D4D"
_AMBER  = "#FFB020"
_MUTED  = "#8696AB"

_SEVERITY_COLOR = {"high": _RED, "medium": _AMBER, "low": _MUTED}
_SEVERITY_ORDER = ["high", "medium", "low"]
_SEVERITY_LABEL = {"high": "HIGH SEVERITY", "medium": "MEDIUM SEVERITY", "low": "LOW SEVERITY"}

_FLAG_LABEL = {
    "missing_value":               "Missing Required Field",
    "numerical_outlier":           "Numerical Outlier",
    "equipment_standby":           "Equipment Standby Anomaly",
    "labor_class_anomaly":         "Categorical Anomaly",
    "equipment_on_wrong_code":     "Equipment / Cost-Code Mismatch",
    "wrong_operator_for_equipment": "Operator Certification Mismatch",
}

# ── Toolbox Talks: the learning layer behind each flag ────────────────────────
_TOOLBOX_TALK = {
    "missing_value": {
        "what": "A required field (hours, labor class, cost code…) was left blank on the timesheet.",
        "cost": "Incomplete records stall payroll, break certified-payroll compliance, and force someone to chase the foreman days later.",
        "prevent": "Make required fields mandatory at entry time and review timesheets the same day they're submitted.",
    },
    "numerical_outlier": {
        "what": "Logged hours fall far outside the historical norm for this cost code and labor class (ML model trained on this project's baseline).",
        "cost": "Outlier hours signal fat-finger entries, duplicated lines, or padded time — each one distorts the cost-to-complete forecast.",
        "prevent": "Compare daily hours against the crew baseline before approval; investigate anything beyond the normal range the same week.",
    },
    "equipment_standby": {
        "what": "A machine logged significantly more hours than the operator assigned to it — billed while sitting idle.",
        "cost": "Idle rentals are pure burn: a machine on the clock with no operator can cost hundreds of dollars a day.",
        "prevent": "Reconcile equipment hours against operator hours daily; put standby grace windows in writing with the rental provider.",
    },
    "labor_class_anomaly": {
        "what": "A worker was charged to a cost code under a labor class that isn't approved for that work.",
        "cost": "Wrong classifications mean wrong billing rates — mischarged labor on cost-plus work and certified-payroll violation risk on public jobs.",
        "prevent": "Keep an approved labor-class list per cost code (the baseline this audit checks) and validate at entry.",
    },
    "equipment_on_wrong_code": {
        "what": "A piece of equipment was charged to a cost code it isn't approved for on this project.",
        "cost": "Cost-code contamination: unit costs look wrong in both directions and estimators inherit bad history for the next bid.",
        "prevent": "Maintain an equipment-to-cost-code map per project and flag mismatches before the daily is approved.",
    },
    "wrong_operator_for_equipment": {
        "what": "The operator's labor class doesn't match what this equipment type requires.",
        "cost": "An unqualified operator on heavy iron is a safety incident waiting to happen — plus insurance and OSHA exposure.",
        "prevent": "Tie operator certifications to equipment types in the baseline; block assignments that don't match.",
    },
}
_DEFAULT_TALK = {
    "what": "An entry on this timesheet doesn't match the project baseline.",
    "cost": "Unexplained deviations compound into overruns and disputes if nobody catches them the day they happen.",
    "prevent": "Review flagged entries the same day; update the baseline when the work legitimately changed.",
}

# Standby equipment hourly rate used for the estimated-recoverable-cost KPI.
_STANDBY_RATE = 85

_SYNTHETIC_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "timesheets.json"

# Single-day slice of the synthetic dataset used for the demo sample button.
# The full dataset is 3829 records / 200 days (built for ML training); sending
# all of it through Gemini narration produces 400+ flags in one prompt, far
# too slow for a live demo. This date has a realistic ~20-record day with all
# four anomaly types represented.
_SAMPLE_DATE = "2025-01-11"

_PAGES = ["Dashboard", "Audit Report", "Baselines", "History"]


# ── Look & feel ────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 3.6rem; max-width: 1250px; }

        /* ── Cards ─────────────────────────────────────── */
        .ciq-card {
            background: #121B27;
            border: 1px solid #1E2A3A;
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.9rem;
        }
        .ciq-card h5 {
            margin: 0 0 0.6rem;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8696AB;
        }
        .ciq-kpi-value { font-size: 1.9rem; font-weight: 750; line-height: 1.1; color: #E8EEF5; }
        .ciq-kpi-sub   { font-size: 0.78rem; color: #8696AB; margin-top: 0.2rem; }

        /* ── Page header ───────────────────────────────── */
        .ciq-page-title { font-size: 1.5rem; font-weight: 700; margin: 0; color: #E8EEF5; }
        .ciq-page-sub   { color: #8696AB; font-size: 0.9rem; margin: 0.15rem 0 1.1rem; }

        /* ── Badges / chips ────────────────────────────── */
        .ciq-chip {
            display: inline-block;
            border-radius: 6px;
            padding: 0.1rem 0.55rem;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .ciq-chip-red    { background: rgba(255,77,77,0.14);  color: #FF6B6B; border: 1px solid rgba(255,77,77,0.4); }
        .ciq-chip-amber  { background: rgba(255,176,32,0.13); color: #FFB020; border: 1px solid rgba(255,176,32,0.4); }
        .ciq-chip-green  { background: rgba(43,213,118,0.12); color: #2BD576; border: 1px solid rgba(43,213,118,0.4); }
        .ciq-chip-grey   { background: rgba(134,150,171,0.12); color: #8696AB; border: 1px solid rgba(134,150,171,0.35); }
        .ciq-chip-orange { background: rgba(255,122,26,0.14); color: #FF7A1A; border: 1px solid rgba(255,122,26,0.45); }

        /* ── Tables ────────────────────────────────────── */
        table.ciq-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
        table.ciq-table th {
            text-align: left;
            color: #8696AB;
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 0.45rem 0.6rem;
            border-bottom: 1px solid #1E2A3A;
        }
        table.ciq-table td {
            padding: 0.5rem 0.6rem;
            border-bottom: 1px solid #18222F;
            color: #C9D5E3;
            vertical-align: middle;
        }
        table.ciq-table tr:last-child td { border-bottom: none; }
        .ciq-mono { font-family: ui-monospace, monospace; font-size: 0.8rem; color: #9FB0C3; }

        /* ── Flag cards (report right rail) ────────────── */
        .ciq-flag {
            background: #121B27;
            border: 1px solid #1E2A3A;
            border-left: 4px solid var(--sev, #8696AB);
            border-radius: 12px;
            padding: 0.8rem 1rem 0.7rem;
            margin-bottom: 0.7rem;
        }
        .ciq-flag-sev { font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em; color: var(--sev, #8696AB); }
        .ciq-flag-title { font-weight: 650; margin: 0.15rem 0 0.3rem; color: #E8EEF5; }
        .ciq-flag-body { font-size: 0.84rem; color: #A9B8C9; }

        /* ── Sidebar ───────────────────────────────────── */
        section[data-testid="stSidebar"] .stButton button { justify-content: flex-start; }
        .ciq-logo { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.2rem; }
        .ciq-logo-box {
            width: 34px; height: 34px;
            border-radius: 9px;
            background: #FF7A1A;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.1rem;
        }
        .ciq-logo-name { font-weight: 750; font-size: 1.05rem; color: #E8EEF5; line-height: 1.1; }
        .ciq-logo-sub  { font-size: 0.68rem; color: #8696AB; letter-spacing: 0.08em; text-transform: uppercase; }

        /* ── History cards ─────────────────────────────── */
        .ciq-hist-band { height: 6px; border-radius: 6px 6px 0 0; }
        .ciq-hist-card {
            background: #121B27;
            border: 1px solid #1E2A3A;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 0.4rem;
        }
        .ciq-hist-body { padding: 0.8rem 1rem 0.7rem; }
        .ciq-hist-date { font-size: 0.7rem; color: #8696AB; letter-spacing: 0.08em; text-transform: uppercase; }
        .ciq-hist-title { font-weight: 650; color: #E8EEF5; margin: 0.1rem 0 0.5rem; }
        .ciq-hist-stat-val { font-size: 1.25rem; font-weight: 750; }
        .ciq-hist-stat-lbl { font-size: 0.66rem; color: #8696AB; text-transform: uppercase; letter-spacing: 0.08em; }

        .ciq-upload-zone {
            border: 2px dashed #2A3950;
            border-radius: 12px;
            text-align: center;
            padding: 1.0rem;
            color: #8696AB;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Data access (MongoDB, graceful degradation) ───────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _load_history(limit: int = 40) -> list[dict]:
    try:
        from database.mongo_client import audit_reports
        return list(
            audit_reports.find({}, {"_id": 0, "flags": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _load_full_report(report_id: str) -> dict | None:
    try:
        from database.mongo_client import audit_reports
        return audit_reports.find_one({"report_id": report_id}, {"_id": 0})
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _load_baselines() -> list[dict]:
    try:
        from database.mongo_client import baselines
        return list(baselines.find({}, {"_id": 0}))
    except Exception:
        return []


def _fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %Y")
    except Exception:
        return iso[:10] if iso else "—"


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _run_on_records(records: list[dict]) -> dict:
    from main import run_audit
    return run_audit(records)


def _load_sample_and_run() -> dict:
    if not _SYNTHETIC_PATH.exists():
        st.error(
            f"Sample data not found at `{_SYNTHETIC_PATH}`.  "
            "Run `python3 scripts/generate_synthetic_data.py` first."
        )
        st.stop()
    with open(_SYNTHETIC_PATH) as f:
        records = json.load(f)
    records = [r for r in records if r.get("date") == _SAMPLE_DATE]
    st.session_state["records"] = records
    return _run_on_records(records)


def _run_on_pdf(tmp_path: str, project_id: str, date_str: str) -> dict:
    from ingestion.pdf_parser import parse_timesheet_pdf, ReductoUnavailableError
    from ingestion.normalizer import normalize_timesheet

    try:
        raw_rows = parse_timesheet_pdf(tmp_path)
        records  = normalize_timesheet(raw_rows, project_id, date_str)
        st.session_state["records"] = records
        return _run_on_records(records)
    except ReductoUnavailableError as exc:
        st.info(f"OCR key not configured — using sample data.  ({exc})")
        return _load_sample_and_run()


def _run_with_status(fn, *args) -> dict:
    with st.status("Audit in progress…", expanded=True) as status:
        st.write("📄 Reading the timesheet into structured records…")
        st.write("🔎 Running 4 detection checks against project baselines…")
        st.write("🤖 Gemini is writing up the findings…")
        report = fn(*args)
        status.update(label="✅ Audit complete", state="complete", expanded=False)
    return report


def _store_report(report: dict) -> None:
    st.session_state["report"] = report
    st.session_state["report_status"] = None
    st.session_state["page"] = "Audit Report"
    _load_history.clear()       # new report changes history + KPIs


# ── Small HTML builders ───────────────────────────────────────────────────────

def _chip(text: str, kind: str) -> str:
    return f'<span class="ciq-chip ciq-chip-{kind}">{text}</span>'


def _sev_chip(sev: str) -> str:
    kind = {"high": "red", "medium": "amber", "low": "grey"}.get(sev, "grey")
    return _chip(sev.upper(), kind)


def _kpi_card(title: str, value: str, sub: str, accent: str | None = None) -> str:
    style = f' style="color:{accent}"' if accent else ""
    return (
        f'<div class="ciq-card"><h5>{title}</h5>'
        f'<div class="ciq-kpi-value"{style}>{value}</div>'
        f'<div class="ciq-kpi-sub">{sub}</div></div>'
    )


def _page_header(title: str, sub: str) -> None:
    st.markdown(
        f'<p class="ciq-page-title">{title}</p><p class="ciq-page-sub">{sub}</p>',
        unsafe_allow_html=True,
    )


# ── Charts ────────────────────────────────────────────────────────────────────

def _trend_chart(history: list[dict]) -> alt.Chart | None:
    rows = []
    for r in history:
        day = (r.get("created_at") or "")[:10]
        if not day:
            continue
        counts = r.get("counts", {})
        for sev in _SEVERITY_ORDER:
            if counts.get(sev, 0):
                rows.append({"Date": day, "Severity": sev.capitalize(), "Flags": counts[sev]})
    if not rows:
        return None
    df = pd.DataFrame(rows).groupby(["Date", "Severity"], as_index=False).sum()
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=22)
        .encode(
            x=alt.X("Date:O", axis=alt.Axis(labelColor=_MUTED, title=None, labelAngle=0)),
            y=alt.Y("Flags:Q", axis=alt.Axis(labelColor=_MUTED, title=None, gridColor="#1E2A3A")),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(domain=["High", "Medium", "Low"], range=[_RED, _AMBER, _MUTED]),
                legend=alt.Legend(orient="top", labelColor=_MUTED, title=None),
            ),
            tooltip=["Date", "Severity", "Flags"],
        )
        .properties(height=230, background="transparent")
        .configure_view(strokeWidth=0)
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="ciq-logo">
                <div class="ciq-logo-box">🏗️</div>
                <div>
                    <div class="ciq-logo-name">ConstructIQ</div>
                    <div class="ciq-logo-sub">Audit Agent</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")

        icons = {"Dashboard": "📊", "Audit Report": "📋", "Baselines": "📐", "History": "🕘"}
        current = st.session_state.get("page", "Dashboard")
        for page in _PAGES:
            if st.button(
                f"{icons[page]}  {page}",
                key=f"nav_{page}",
                type="primary" if page == current else "tertiary",
                use_container_width=True,
            ):
                st.session_state["page"] = page
                st.rerun()

        st.markdown("---")
        if st.button("⬆️  Upload Timesheet", use_container_width=True):
            st.session_state["page"] = "Dashboard"
            st.rerun()

        st.caption("Google Agent Builder Hackathon 2026 · MongoDB track")
        st.markdown("📂 [Source on GitHub](https://github.com/NxlinJxshi/constructiq)")


# ── Page: Dashboard ───────────────────────────────────────────────────────────

def _page_dashboard() -> None:
    _page_header("Operations Command", "AI timesheet auditing · Westmorland Elementary · 31062-00")

    history = _load_history()
    latest  = history[0] if history else None

    # ── KPI row ──────────────────────────────────────────────────────────────
    total_audits = len(history)
    anomaly_rate = "—"
    high_flags   = "—"
    savings      = "—"
    if latest:
        counts  = latest.get("counts", {})
        n_flags = sum(counts.values())
        n_rec   = latest.get("total_records", 0) or 0
        anomaly_rate = f"{(n_flags / n_rec * 100):.0f}%" if n_rec else "—"
        high_flags   = str(counts.get("high", 0))
        full = _load_full_report(latest.get("report_id", "")) or {}
        standby_hours = sum(
            (f.get("details") or {}).get("delta_hours", 0)
            for f in full.get("flags", [])
            if f.get("flag_type") == "equipment_standby"
        )
        savings = f"${standby_hours * _STANDBY_RATE:,.0f}"

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi_card("Total Audits", str(total_audits) if history else "—",
                          "stored in MongoDB Atlas"), unsafe_allow_html=True)
    k2.markdown(_kpi_card("Anomaly Rate", anomaly_rate, "latest audit", _AMBER), unsafe_allow_html=True)
    k3.markdown(_kpi_card("High Severity", high_flags, "latest audit", _RED), unsafe_allow_html=True)
    k4.markdown(_kpi_card("Eq. Standby Cost", savings, f"recoverable @ ${_STANDBY_RATE}/hr", _GREEN), unsafe_allow_html=True)

    # ── Upload + trends ──────────────────────────────────────────────────────
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown('<div class="ciq-card"><h5>📥 New Audit</h5>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ciq-upload-zone">Drop an HCSS HeavyJob PDF or screenshot here —<br>'
            "ConstructIQ extracts every line and verifies it against site baselines.</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload timesheet",
            type=["pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        run_upload = uploaded is not None and st.button(
            "Run audit on uploaded file", type="primary", use_container_width=True
        )
        run_sample = st.button(
            "▶  Run sample audit (no file needed)", use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if run_sample:
            report = _run_with_status(_load_sample_and_run)
            _store_report(report)
            st.rerun()
        if run_upload:
            suffix = os.path.splitext(uploaded.name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                report = _run_with_status(_run_on_pdf, tmp_path, "unknown", "unknown")
                _store_report(report)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            st.rerun()

    with right:
        st.markdown('<div class="ciq-card"><h5>📈 Anomaly Trends</h5></div>', unsafe_allow_html=True)
        chart = _trend_chart(history)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No audit history yet — run an audit to populate trends.")

    # ── Recent audits table ──────────────────────────────────────────────────
    st.markdown('<div class="ciq-card"><h5>🕘 Recent Audits</h5></div>', unsafe_allow_html=True)
    if not history:
        st.caption("MongoDB Atlas unreachable or empty — recent audits unavailable.")
        return

    rows = []
    for r in history[:6]:
        counts = r.get("counts", {})
        n_high = counts.get("high", 0)
        status = (
            _chip("CRITICAL", "red") if n_high >= 5
            else _chip("REVIEW", "amber") if sum(counts.values()) > 0
            else _chip("CLEAN", "green")
        )
        rows.append(
            f"<tr><td class='ciq-mono'>{r.get('report_id', '')[:8]}</td>"
            f"<td>{_fmt_date(r.get('created_at', ''))}</td>"
            f"<td>{r.get('total_records', 0)}</td>"
            f"<td>{sum(counts.values())} ({n_high} high)</td>"
            f"<td>{status}</td></tr>"
        )
    st.markdown(
        "<table class='ciq-table'><tr><th>Report</th><th>Audit Date</th>"
        "<th>Records</th><th>Flags</th><th>Status</th></tr>"
        + "".join(rows) + "</table>",
        unsafe_allow_html=True,
    )
    st.caption("Open any past report from the **History** page.")


# ── Page: Audit Report ────────────────────────────────────────────────────────

def _render_records_table(records: list[dict], flags: list[dict]) -> None:
    by_record: dict[str, dict] = {}
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    for f in flags:
        rid = f.get("record_id")
        if rid and (rid not in by_record or sev_rank.get(f.get("severity"), 3) < sev_rank.get(by_record[rid].get("severity"), 3)):
            by_record[rid] = f

    rows = []
    for r in records:
        flag = by_record.get(r.get("record_id"))
        if flag:
            kind  = {"high": "red", "medium": "amber"}.get(flag.get("severity"), "grey")
            badge = _chip(_FLAG_LABEL.get(flag.get("flag_type"), "FLAGGED").upper(), kind)
        else:
            badge = _chip("✓ VERIFIED", "green")
        is_equip = r.get("entity_type") == "equipment"
        name  = r.get("equipment_type") if is_equip else r.get("worker_name", "—")
        cls   = "EQUIPMENT" if is_equip else (r.get("labor_class") or "—")
        hours = r.get("total_hours") or r.get("equipment_hours") or r.get("straight_time_hours") or 0
        rows.append(
            f"<tr><td class='ciq-mono'>{(r.get('worker_id') or r.get('record_id') or '')[:8]}</td>"
            f"<td>{name}</td>"
            f"<td class='ciq-mono'>{r.get('cost_code', '—')}</td>"
            f"<td>{cls}</td>"
            f"<td>{hours}</td>"
            f"<td>{badge}</td></tr>"
        )

    st.markdown(
        "<table class='ciq-table'><tr><th>Resource ID</th><th>Name</th><th>Cost Code</th>"
        "<th>Class</th><th>Hours</th><th>Status</th></tr>"
        + "".join(rows) + "</table>",
        unsafe_allow_html=True,
    )


def _page_report() -> None:
    report = st.session_state.get("report")
    if report is None:
        _page_header("Audit Report", "No audit loaded")
        st.info("Run an audit from the **Dashboard**, or open one from **History**.")
        return

    date_str = _fmt_date(report.get("created_at", ""))
    _page_header(f"Audit Report — {date_str}",
                 f"Report {report.get('report_id', '')[:8]} · {report.get('total_records', 0)} records scanned")

    flags   = report.get("flags", [])
    counts  = report.get("counts", {})
    summary = report.get("summary", "")

    left, right = st.columns([1.7, 1])

    with left:
        # Executive summary + approve / flag actions
        st.markdown(
            f'<div class="ciq-card"><h5>📝 Executive Summary</h5>'
            f'<div style="color:#C9D5E3; font-size:0.92rem;">{summary or "No summary available."}</div></div>',
            unsafe_allow_html=True,
        )

        status = st.session_state.get("report_status")
        if status:
            kind, label = ("green", "✓ AUDIT APPROVED") if status == "approved" else ("orange", "⚑ FLAGGED FOR REVIEW")
            st.markdown(_chip(label, kind), unsafe_allow_html=True)
            st.write("")
        else:
            b1, b2, _sp = st.columns([1, 1, 1.4])
            if b1.button("Approve Audit", type="primary", use_container_width=True):
                st.session_state["report_status"] = "approved"
                st.rerun()
            if b2.button("Flag for Review", use_container_width=True):
                st.session_state["report_status"] = "review"
                st.rerun()

        # HCSS export detail (only available for audits run this session)
        records = st.session_state.get("records")
        st.markdown('<div class="ciq-card"><h5>📄 HCSS Export Detail</h5></div>', unsafe_allow_html=True)
        if records:
            _render_records_table(records, flags)
        else:
            st.caption(
                "Line-level records aren't stored for historical reports — "
                "run a new audit to see the full HCSS export detail."
            )

        st.write("")
        st.download_button(
            "⬇️ Download full report (JSON)",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"constructiq_audit_{report.get('report_id', 'report')}.json",
            mime="application/json",
        )

    with right:
        st.markdown(
            f'<div class="ciq-card"><h5>Severity Breakdown</h5>'
            f'<span style="color:{_RED}; font-weight:750;">{counts.get("high", 0)} high</span> · '
            f'<span style="color:{_AMBER}; font-weight:750;">{counts.get("medium", 0)} medium</span> · '
            f'<span style="color:{_MUTED}; font-weight:750;">{counts.get("low", 0)} low</span></div>',
            unsafe_allow_html=True,
        )
        for sev in _SEVERITY_ORDER:
            for f in [f for f in flags if f.get("severity") == sev]:
                color = _SEVERITY_COLOR[sev]
                title = _FLAG_LABEL.get(f.get("flag_type"), f.get("flag_type", "Anomaly"))
                st.markdown(
                    f'<div class="ciq-flag" style="--sev:{color};">'
                    f'<div class="ciq-flag-sev">{_SEVERITY_LABEL[sev]}</div>'
                    f'<div class="ciq-flag-title">{title}</div>'
                    f'<div class="ciq-flag-body">{f.get("explanation", "")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                talk = _TOOLBOX_TALK.get(f.get("flag_type"), _DEFAULT_TALK)
                with st.expander("📚 Toolbox Talk — learn from this flag"):
                    st.markdown(
                        f"**🔍 What happened:** {talk['what']}\n\n"
                        f"**💸 Why it costs money:** {talk['cost']}\n\n"
                        f"**🛠️ How to prevent it:** {talk['prevent']}"
                    )


# ── Page: Baselines ───────────────────────────────────────────────────────────

def _page_baselines() -> None:
    _page_header("Project Baselines", "Historical norms the audit checks every timesheet line against")

    baselines = _load_baselines()

    # ML model status + Atlas status cards
    c1, c2 = st.columns(2)
    endpoint = os.environ.get("VERTEX_ENDPOINT_NAME", "")
    ml_status = _chip("● ACTIVE", "green") if endpoint else _chip("○ OFFLINE", "grey")
    c1.markdown(
        f'<div class="ciq-card"><h5>🤖 ML Model Status</h5>'
        f'<div style="font-weight:700; color:#E8EEF5;">IsolationForest &nbsp;{ml_status}</div>'
        f'<div class="ciq-kpi-sub">Host: Vertex AI (GCP) · numerical-outlier detection'
        f'{"" if endpoint else " · rules engine carries the audit when offline"}</div></div>',
        unsafe_allow_html=True,
    )
    atlas_status = _chip("● CONNECTED", "green") if baselines else _chip("○ UNREACHABLE", "red")
    c2.markdown(
        f'<div class="ciq-card"><h5>🍃 MongoDB Atlas</h5>'
        f'<div style="font-weight:700; color:#E8EEF5;">constructiq.baselines &nbsp;{atlas_status}</div>'
        f'<div class="ciq-kpi-sub">{len(baselines)} baseline document(s) · timesheets · audit_reports</div></div>',
        unsafe_allow_html=True,
    )

    if not baselines:
        st.info("Baselines unavailable — check the MongoDB Atlas connection.")
        return

    options = {f"{b.get('cost_code')} — {b.get('description', '')}": b for b in baselines}
    choice = st.selectbox("Cost code", list(options.keys()))
    b = options[choice]

    k1, k2, k3 = st.columns(3)
    k1.markdown(_kpi_card("Avg Hours / Worker", str(b.get("avg_hours_per_worker", "—")),
                          f"± {b.get('std_dev', '—')} std dev"), unsafe_allow_html=True)
    k2.markdown(_kpi_card("Straight / Overtime", f"{b.get('avg_straight_time_hours', '—')} / {b.get('avg_overtime_hours', '—')}",
                          "typical shift split"), unsafe_allow_html=True)
    k3.markdown(_kpi_card("Avg Equipment Hours", str(b.get("avg_equipment_hours", "—")),
                          "per shift on this code"), unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="ciq-card"><h5>👷 Approved Labor Classes</h5></div>', unsafe_allow_html=True)
        chips = " ".join(_chip(c, "orange") for c in b.get("typical_labor_classes", []))
        st.markdown(chips or "—", unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="ciq-card"><h5>🚜 Equipment → Operator Pairs</h5></div>', unsafe_allow_html=True)
        pairs = b.get("equipment_operator_pairs", [])
        rows = "".join(
            f"<tr><td>{p.get('equipment_type', '—')}</td>"
            f"<td class='ciq-mono'>{p.get('typical_operator_class', '—')}</td></tr>"
            for p in pairs
        )
        st.markdown(
            "<table class='ciq-table'><tr><th>Equipment</th><th>Operator Class</th></tr>"
            + (rows or "<tr><td colspan=2>—</td></tr>") + "</table>",
            unsafe_allow_html=True,
        )


# ── Page: History ─────────────────────────────────────────────────────────────

def _page_history() -> None:
    _page_header("Audit History", "Every audit report stored in MongoDB Atlas")

    history = _load_history()
    if not history:
        st.info("No audit history available — MongoDB Atlas unreachable or empty.")
        return

    per_row = 4
    for start in range(0, min(len(history), 12), per_row):
        cols = st.columns(per_row)
        for col, r in zip(cols, history[start:start + per_row]):
            counts  = r.get("counts", {})
            n_flags = sum(counts.values())
            n_rec   = r.get("total_records", 0) or 0
            clean   = max(0, 100 - (n_flags / n_rec * 100)) if n_rec else 0
            n_high  = counts.get("high", 0)
            band    = _RED if n_high >= 5 else _AMBER if n_flags else _GREEN
            with col:
                st.markdown(
                    f"""
                    <div class="ciq-hist-card">
                        <div class="ciq-hist-band" style="background:{band};"></div>
                        <div class="ciq-hist-body">
                            <div class="ciq-hist-date">{_fmt_date(r.get("created_at", ""))}</div>
                            <div class="ciq-hist-title">Audit {r.get("report_id", "")[:8]}</div>
                            <div style="display:flex; gap:1.1rem;">
                                <div><div class="ciq-hist-stat-val" style="color:{_GREEN};">{clean:.0f}%</div>
                                     <div class="ciq-hist-stat-lbl">Clean</div></div>
                                <div><div class="ciq-hist-stat-val" style="color:{band};">{n_flags}</div>
                                     <div class="ciq-hist-stat-lbl">Issues</div></div>
                                <div><div class="ciq-hist-stat-val">{n_rec}</div>
                                     <div class="ciq-hist-stat-lbl">Records</div></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("View Full Report", key=f"hist_{r.get('report_id')}", use_container_width=True):
                    full = _load_full_report(r.get("report_id", ""))
                    if full:
                        st.session_state["report"] = full
                        st.session_state["records"] = None
                        st.session_state["report_status"] = None
                        st.session_state["page"] = "Audit Report"
                        st.rerun()

    st.caption(f"Showing {min(len(history), 12)} of {len(history)} audits.")


# ── Router ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="ConstructIQ — Audit Agent",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()
    _render_sidebar()

    page = st.session_state.get("page", "Dashboard")
    if page == "Audit Report":
        _page_report()
    elif page == "Baselines":
        _page_baselines()
    elif page == "History":
        _page_history()
    else:
        _page_dashboard()


if __name__ == "__main__":
    main()
