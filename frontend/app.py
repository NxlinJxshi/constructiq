# ============================================================
# MODULE: frontend/app.py
# PURPOSE: Streamlit demo UI — lets a project manager (or a hackathon judge)
#          upload an HCSS timesheet PDF or load the built-in sample data,
#          run the full audit pipeline, and review the findings as a
#          gamified, construction-themed "site inspection".
# PIPELINE STAGE: Frontend
# ============================================================

"""
Streamlit audit dashboard — construction-site edition.

Two entry paths:
  1. "Run sample inspection" button (default / no key needed):
     Loads data/synthetic/timesheets.json → run_audit → renders report.
  2. PDF upload:
     parse_timesheet_pdf → normalize_timesheet → run_audit → renders report.
     If REDUCTO_API_KEY is not configured, shows an info banner and falls back
     to the sample path.

Design principles:
  - A non-technical judge must understand the output in under 30 seconds.
  - Every flag is a "roadblock" 🚧 the inspector clears one by one, each with
    a "Toolbox Talk" explaining what happened, why it costs money, and how to
    prevent it — the user learns the domain while reviewing.
  - A Site Safety Score (A–F) and inspector rank make the audit feel like a
    job-site walkthrough, not a spreadsheet.

The audit report lives in st.session_state so interactive widgets (filters,
clear buttons) survive Streamlit reruns.
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

# ── Construction palette ──────────────────────────────────────────────────────
_SEVERITY_COLOR = {
    "high":   "#EF4444",   # stop-sign red
    "medium": "#FF8C00",   # safety orange
    "low":    "#A1A1AA",   # steel grey
}
_SEVERITY_ORDER = ["high", "medium", "low"]
_SEVERITY_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}
_SEVERITY_SIGN  = {"high": "🛑", "medium": "⚠️", "low": "🔹"}

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

# ── Toolbox Talks: the learning layer behind each roadblock ───────────────────
# Each flag type gets a short field-guide entry: what it is, why it burns
# money, and how a crew prevents it. Shown inside every roadblock card.
_TOOLBOX_TALK = {
    "missing_value": {
        "what": "A required field (hours, labor class, cost code…) was left "
                "blank on the timesheet.",
        "cost": "Incomplete records stall payroll, break certified-payroll "
                "compliance, and make job-cost reports unreliable — someone "
                "has to chase the foreman days later when memories are fuzzy.",
        "prevent": "Make required fields mandatory at entry time and review "
                   "timesheets the same day they're submitted.",
    },
    "numerical_outlier": {
        "what": "The logged hours fall far outside the historical norm for "
                "this cost code and labor class (caught by an ML model "
                "trained on this project's baseline).",
        "cost": "Outlier hours are the #1 signal of fat-finger entries, "
                "duplicated lines, or padded time — each one silently "
                "distorts the project's cost-to-complete forecast.",
        "prevent": "Compare daily hours against the crew's baseline before "
                   "approval; investigate anything beyond the normal range "
                   "the same week.",
    },
    "equipment_standby": {
        "what": "A machine logged significantly more hours than the operator "
                "assigned to it — it was billed while sitting idle.",
        "cost": "Idle rentals are pure burn: a machine on the clock with no "
                "operator can cost hundreds of dollars a day and is a "
                "classic source of disputed owner billings.",
        "prevent": "Reconcile equipment hours against operator hours daily, "
                   "and put standby rules (grace windows) in writing with "
                   "the rental provider.",
    },
    "labor_class_anomaly": {
        "what": "A worker was charged to a cost code under a labor class "
                "that isn't approved for that kind of work.",
        "cost": "Wrong classifications mean wrong billing rates — that's "
                "mischarged labor on cost-plus work and a prevailing-wage / "
                "certified-payroll violation risk on public jobs.",
        "prevent": "Keep an approved labor-class list per cost code (that's "
                   "the baseline this audit checks) and validate at entry.",
    },
    "equipment_on_wrong_code": {
        "what": "A piece of equipment was charged to a cost code it isn't "
                "approved for on this project.",
        "cost": "Cost-code contamination: the wrong activity absorbs the "
                "expense, so unit costs look wrong in both directions and "
                "estimators inherit bad history for the next bid.",
        "prevent": "Maintain an equipment-to-cost-code map per project and "
                   "flag mismatches before the daily is approved.",
    },
    "wrong_operator_for_equipment": {
        "what": "The operator's labor class doesn't match what this type of "
                "equipment requires.",
        "cost": "An unqualified operator on heavy iron is a safety incident "
                "waiting to happen — and an insurance and OSHA exposure even "
                "when nothing goes wrong.",
        "prevent": "Tie operator certifications to equipment types in the "
                   "baseline, and block assignments that don't match.",
    },
}

_DEFAULT_TALK = {
    "what": "An entry on this timesheet doesn't match the project baseline.",
    "cost": "Unexplained deviations compound into cost overruns and disputes "
            "if nobody catches them the day they happen.",
    "prevent": "Review flagged entries the same day and update the baseline "
               "when the work legitimately changed.",
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
        /* Hazard stripe — the visual signature of the whole app */
        .ciq-hazard {
            height: 10px;
            border-radius: 4px;
            background: repeating-linear-gradient(
                45deg, #FFC400 0 14px, #1A1A1A 14px 28px);
            margin: 0.2rem 0 1rem;
        }
        .ciq-hero {
            background: linear-gradient(135deg, #0F2A43 0%, #1B3A5C 55%, #FF6A00 165%);
            border-radius: 16px;
            padding: 2rem 2.25rem 1.8rem;
            color: #FFFFFF;
        }
        .ciq-hero h1 { margin: 0; font-size: 2.1rem; line-height: 1.2; }
        .ciq-hero p  { margin: 0.45rem 0 0; opacity: 0.9; font-size: 1.02rem; }
        .ciq-badge-row { margin-top: 0.9rem; }
        .ciq-badge {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }

        /* Site safety scoreboard — "days without incident" style */
        .ciq-scoreboard {
            background: #14181D;
            border: 3px solid #FFC400;
            border-radius: 14px;
            padding: 1.1rem 1rem 1.2rem;
            text-align: center;
            color: #F5F5F4;
        }
        .ciq-score-title {
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            opacity: 0.8;
            text-transform: uppercase;
        }
        .ciq-score-value {
            font-size: 3.4rem;
            font-weight: 800;
            line-height: 1.05;
            font-variant-numeric: tabular-nums;
        }
        .ciq-score-grade {
            display: inline-block;
            margin-top: 0.3rem;
            border-radius: 999px;
            padding: 0.1rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            background: rgba(255,255,255,0.12);
        }

        .ciq-metric-card {
            border: 1px solid rgba(127,127,127,0.2);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            text-align: center;
            background: rgba(127,127,127,0.04);
        }
        .ciq-metric-value { font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
        .ciq-metric-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.65;
            margin-top: 0.2rem;
        }

        /* Roadblock cards */
        .ciq-roadblock {
            border: 1px solid rgba(127,127,127,0.25);
            border-left: 6px solid var(--ciq-sev-color, #A1A1AA);
            border-radius: 12px;
            padding: 0 0 0.2rem;
            margin-bottom: 0.35rem;
            background: rgba(127,127,127,0.045);
            overflow: hidden;
        }
        .ciq-roadblock.cleared {
            border-left-color: #22C55E;
            opacity: 0.62;
        }
        .ciq-roadblock-stripe {
            height: 7px;
            background: repeating-linear-gradient(
                45deg, var(--ciq-sev-color, #A1A1AA) 0 12px, #1A1A1A 12px 24px);
        }
        .ciq-roadblock.cleared .ciq-roadblock-stripe {
            background: repeating-linear-gradient(
                45deg, #22C55E 0 12px, #14532D 12px 24px);
        }
        .ciq-roadblock-body { padding: 0.75rem 1.1rem 0.55rem; }
        .ciq-roadblock-head {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 650;
            font-size: 1.0rem;
            margin-bottom: 0.35rem;
        }
        .ciq-sev-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.05rem 0.6rem;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: white;
        }

        /* Road progress bar */
        .ciq-road {
            position: relative;
            height: 34px;
            border-radius: 10px;
            background: #2B2F36;
            overflow: hidden;
            border: 1px solid rgba(127,127,127,0.3);
        }
        .ciq-road-fill {
            position: absolute;
            inset: 0 auto 0 0;
            background: linear-gradient(90deg, #15803D, #22C55E);
            transition: width 0.4s ease;
        }
        .ciq-road-line {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 3px;
            transform: translateY(-50%);
            background: repeating-linear-gradient(
                90deg, #FFC400 0 18px, transparent 18px 34px);
        }
        .ciq-road-truck {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -54%);
            font-size: 1.35rem;
            z-index: 2;
        }
        .ciq-road-flag {
            position: absolute;
            top: 50%;
            right: 8px;
            transform: translateY(-52%);
            font-size: 1.1rem;
        }

        .ciq-empty-state { text-align: center; padding: 2.4rem 1rem; opacity: 0.78; }
        .ciq-empty-state .ciq-emoji { font-size: 2.6rem; display: block; margin-bottom: 0.5rem; }

        .ciq-roadopen {
            border: 2px solid #22C55E;
            border-radius: 12px;
            background: rgba(34,197,94,0.1);
            text-align: center;
            padding: 1rem;
            font-weight: 650;
            margin-bottom: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="ciq-hero">
            <h1>🏗️ ConstructIQ — Site Inspection</h1>
            <p>An AI inspector that walks your HCSS HeavyJob timesheet like a
            job site: every bad entry becomes a 🚧 roadblock you clear — and
            learn from — one Toolbox Talk at a time.</p>
            <div class="ciq-badge-row">
                <span class="ciq-badge">⚡ Vertex AI · Gemini</span>
                <span class="ciq-badge">🍃 MongoDB Atlas</span>
                <span class="ciq-badge">📄 Reducto Extract</span>
            </div>
        </div>
        <div class="ciq-hazard"></div>
        """,
        unsafe_allow_html=True,
    )


def _inspector_rank(cleared: int, total: int) -> str:
    if total == 0 or cleared == 0:
        return "👷 Rookie Inspector"
    if cleared >= total:
        return "🏆 Site Superintendent"
    if cleared / total >= 0.5:
        return "📋 Foreman"
    return "🦺 Site Inspector"


def _render_sidebar(report: dict | None, cleared: set) -> None:
    with st.sidebar:
        st.markdown("### 🏗️ ConstructIQ")
        st.caption("Google Agent Builder Hackathon 2026 · MongoDB track")

        if report is not None:
            flags = report.get("flags", [])
            st.markdown("---")
            st.markdown("#### 🦺 Inspector status")
            st.markdown(f"**{_inspector_rank(len(cleared), len(flags))}**")
            st.caption(f"{len(cleared)} of {len(flags)} roadblocks cleared")

        st.markdown("---")
        st.markdown("#### How the inspection works")
        st.markdown(
            "1. **Ingest** — Reducto Extract parses the HCSS timesheet PDF "
            "or screenshot into structured rows.\n"
            "2. **Detect** — four checks run against project baselines: "
            "missing fields, labor/equipment mismatches, equipment standby, "
            "and ML-based hour outliers (Vertex AI).\n"
            "3. **Narrate** — Gemini 2.5 Flash rewrites each flag in plain "
            "English, ranks them by severity, and writes the foreman's "
            "briefing.\n"
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
    """Run a pipeline function inside a st.status block with site-walk steps."""
    with st.status("🚧 Inspection in progress — site walk underway…", expanded=True) as status:
        st.write("📄 Reading the timesheet into structured records…")
        st.write("🔎 Walking the site — 4 detection checks against project baselines…")
        st.write("🤖 Foreman Gemini is writing up the findings…")
        report = fn(*args)
        status.update(label="✅ Inspection complete", state="complete", expanded=False)
    return report


def _store_report(report: dict) -> None:
    """Put a fresh report into session state and reset inspection progress."""
    st.session_state["report"] = report
    st.session_state["cleared"] = set()
    st.session_state["celebrated"] = False


# ── Scoring ───────────────────────────────────────────────────────────────────

def _safety_score(counts: dict) -> tuple[int, str, str, str]:
    """Return (score, grade, grade label, colour) from severity counts."""
    score = max(
        0,
        100
        - 15 * counts.get("high", 0)
        - 7 * counts.get("medium", 0)
        - 3 * counts.get("low", 0),
    )
    if score >= 90:
        return score, "A", "EXCELLENT SITE", "#22C55E"
    if score >= 80:
        return score, "B", "SOLID WORK", "#84CC16"
    if score >= 70:
        return score, "C", "NEEDS ATTENTION", "#FFC400"
    if score >= 60:
        return score, "D", "AT RISK", "#FF8C00"
    return score, "F", "STOP-WORK ORDER", "#EF4444"


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


def _render_road_progress(cleared: int, total: int) -> None:
    pct = 100.0 if total == 0 else (cleared / total) * 100.0
    st.markdown(
        f"""
        <div class="ciq-road">
            <div class="ciq-road-fill" style="width: {pct:.1f}%;"></div>
            <div class="ciq-road-line"></div>
            <div class="ciq-road-truck" style="left: {max(pct, 2.5):.1f}%;">🚜</div>
            <div class="ciq-road-flag">🏁</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"🚦 {cleared} of {total} roadblocks cleared — keep the road moving!")


def _flag_details_caption(details: dict) -> None:
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


def _render_roadblock(flag: dict, idx: int, cleared: set) -> None:
    """Render one flag as a clearable roadblock with its Toolbox Talk."""
    sev       = flag.get("severity", "low")
    color     = _SEVERITY_COLOR.get(sev, "#A1A1AA")
    flag_type = flag.get("flag_type", "")
    icon      = _FLAG_ICON.get(flag_type, "⚠️")
    label     = _FLAG_LABEL.get(flag_type, flag_type or "Unknown")
    details   = flag.get("details") or {}
    flag_id   = flag.get("flag_id") or f"flag_{idx}"
    is_cleared = flag_id in cleared

    cleared_cls  = " cleared" if is_cleared else ""
    status_pill  = (
        '<span class="ciq-sev-pill" style="background:#22C55E;">✓ CLEARED</span>'
        if is_cleared
        else f'<span class="ciq-sev-pill" style="background:{color};">{sev.upper()}</span>'
    )

    st.markdown(
        f"""
        <div class="ciq-roadblock{cleared_cls}" style="--ciq-sev-color: {color};">
            <div class="ciq-roadblock-stripe"></div>
            <div class="ciq-roadblock-body">
                <div class="ciq-roadblock-head">
                    🚧 {status_pill}
                    <span>{icon} {label}</span>
                </div>
                <div>{flag.get("explanation", "")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _flag_details_caption(details)

    talk = _TOOLBOX_TALK.get(flag_type, _DEFAULT_TALK)
    talk_col, btn_col = st.columns([4, 1])
    with talk_col:
        with st.expander(f"📚 Toolbox Talk — learn from this roadblock"):
            st.markdown(
                f"**🔍 What happened:** {talk['what']}\n\n"
                f"**💸 Why it costs money:** {talk['cost']}\n\n"
                f"**🛠️ How to prevent it:** {talk['prevent']}"
            )
    with btn_col:
        if not is_cleared:
            if st.button("✅ Clear", key=f"clear_{flag_id}", use_container_width=True):
                cleared.add(flag_id)
                st.rerun()

    st.write("")


def _render_roadblocks_tab(flags: list[dict], cleared: set) -> None:
    """The gamified findings view: progress road + clearable roadblocks."""
    if not flags:
        st.markdown(
            '<div class="ciq-empty-state"><span class="ciq-emoji">🛣️</span>'
            '<b>Road open!</b> No roadblocks on this site.</div>',
            unsafe_allow_html=True,
        )
        return

    _render_road_progress(len(cleared), len(flags))

    if len(cleared) >= len(flags):
        st.markdown(
            '<div class="ciq-roadopen">🏁 Road open — every roadblock reviewed. '
            'Great inspection, Superintendent!</div>',
            unsafe_allow_html=True,
        )
        if not st.session_state.get("celebrated"):
            st.session_state["celebrated"] = True
            st.balloons()

    st.write("")

    # Filters
    sev_options = [s for s in _SEVERITY_ORDER if any(f.get("severity") == s for f in flags)]
    fcol1, fcol2 = st.columns([2, 3])
    with fcol1:
        selected_sevs = st.pills(
            "Severity",
            options=sev_options,
            format_func=lambda s: f"{_SEVERITY_SIGN[s]} {_SEVERITY_LABEL[s]}",
            selection_mode="multi",
            default=sev_options,
        )
    with fcol2:
        search = st.text_input("Search roadblocks (worker name, ID, cost code…)", "")

    filtered = [
        (i, f) for i, f in enumerate(flags)
        if f.get("severity") in (selected_sevs or [])
        and (search.lower() in f.get("explanation", "").lower() if search else True)
    ]

    if not filtered:
        st.markdown(
            '<div class="ciq-empty-state"><span class="ciq-emoji">🔍</span>'
            'No roadblocks match the current filters.</div>',
            unsafe_allow_html=True,
        )
        return

    st.caption(f"Showing {len(filtered)} of {len(flags)} roadblocks")

    for sev_label, sev_key in [("High Severity", "high"), ("Medium Severity", "medium"), ("Low Severity", "low")]:
        group = [(i, f) for i, f in filtered if f.get("severity") == sev_key]
        if not group:
            continue
        color = _SEVERITY_COLOR[sev_key]
        st.markdown(
            f'<h4 style="color:{color};">{_SEVERITY_SIGN[sev_key]} {sev_label} ({len(group)})</h4>',
            unsafe_allow_html=True,
        )
        for idx, flag in group:
            _render_roadblock(flag, idx, cleared)


def _render_report(report: dict, cleared: set) -> None:
    """Render the full audit report: site report tab + roadblocks tab."""
    counts  = report.get("counts", {})
    flags   = report.get("flags", [])
    summary = report.get("summary", "")
    total   = report.get("total_records", 0)

    overview_tab, roadblocks_tab = st.tabs([
        "📋 Site Report",
        f"🚧 Roadblocks ({len(flags)})",
    ])

    with overview_tab:
        score, grade, grade_label, score_color = _safety_score(counts)

        score_col, m1, m2, m3, m4 = st.columns([1.5, 1, 1, 1, 1])
        with score_col:
            st.markdown(
                f"""
                <div class="ciq-scoreboard">
                    <div class="ciq-score-title">⛑️ Site Safety Score</div>
                    <div class="ciq-score-value" style="color:{score_color};">{score}</div>
                    <div class="ciq-score-grade" style="color:{score_color};">GRADE {grade} · {grade_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        m1.markdown(_metric_card("Records Scanned", total), unsafe_allow_html=True)
        m2.markdown(_metric_card("🛑 High", counts.get("high", 0), _SEVERITY_COLOR["high"]), unsafe_allow_html=True)
        m3.markdown(_metric_card("⚠️ Medium", counts.get("medium", 0), _SEVERITY_COLOR["medium"]), unsafe_allow_html=True)
        m4.markdown(_metric_card("🔹 Low", counts.get("low", 0), _SEVERITY_COLOR["low"]), unsafe_allow_html=True)

        st.write("")

        chart_col, summary_col = st.columns([1, 2])
        with chart_col:
            chart = _severity_chart(counts)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.markdown(
                    '<div class="ciq-empty-state"><span class="ciq-emoji">✅</span>'
                    'Clean site — no issues found.</div>',
                    unsafe_allow_html=True,
                )
        with summary_col:
            if summary:
                st.markdown("##### 📣 Foreman's Briefing")
                st.info(summary)

        dl_col, reset_col = st.columns([1, 1])
        dl_col.download_button(
            "⬇️ Download full report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"constructiq_audit_{report.get('report_id', 'report')}.json",
            mime="application/json",
        )
        if reset_col.button("🔄 Start a new inspection"):
            for key in ("report", "cleared", "celebrated"):
                st.session_state.pop(key, None)
            st.rerun()

    with roadblocks_tab:
        _render_roadblocks_tab(flags, cleared)


# ── Page layout ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="ConstructIQ — Site Inspection",
        page_icon="🏗️",
        layout="wide",
    )

    _inject_css()
    _render_hero()

    # ── Input controls ────────────────────────────────────────────────────────
    col_btn, col_upload = st.columns([1, 2])

    with col_btn:
        use_sample = st.button(
            "🦺 Run sample inspection",
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

    # ── Run pipeline (results persist in session state across reruns) ─────────
    if use_sample:
        report = _run_with_status(_load_sample_and_run)
        _store_report(report)

    elif uploaded_pdf is not None and "report" not in st.session_state:
        project_id = st.text_input("Project ID (optional)", value="unknown")
        date_str   = st.text_input("Shift date YYYY-MM-DD (optional)", value="unknown")

        if st.button("🦺 Run inspection on uploaded file", type="primary"):
            suffix = os.path.splitext(uploaded_pdf.name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            try:
                report = _run_with_status(_run_on_pdf, tmp_path, project_id, date_str)
                _store_report(report)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    report  = st.session_state.get("report")
    cleared = st.session_state.get("cleared", set())

    _render_sidebar(report, cleared)

    if report is not None:
        _render_report(report, cleared)
    else:
        st.markdown(
            '<div class="ciq-empty-state"><span class="ciq-emoji">🚧</span>'
            '<b>The site is waiting for its inspector.</b><br>'
            'Click <b>🦺 Run sample inspection</b> to walk a sample timesheet, '
            'or upload an HCSS PDF to inspect your own.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
