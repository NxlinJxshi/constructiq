# ============================================================
# MODULE: main.py
# PURPOSE: Pipeline spine — chains all four detectors into a unified flag
#          list, passes flags to Gemini for narration, stores the report
#          in MongoDB, and returns a structured audit report dict.
# PIPELINE STAGE: Orchestration (top-level)
# INPUTS: Normalized timesheet records from ingestion.normalizer
# OUTPUTS: Audit report dict + one document written to audit_reports collection
# ============================================================

"""
Entry point and pipeline orchestrator for the ConstructIQ audit pipeline.

Runs the full audit workflow:
  1. Four detection layers in order (missing_values → categorical → standby → numerical)
  2. Flag validation against the §5 schema
  3. Gemini narration and ranking via agent.audit_agent
  4. Persist report to MongoDB audit_reports collection
  5. Return structured report dict to caller

The numerical outlier layer is non-fatal: if the Vertex endpoint is
unreachable or VERTEX_ENDPOINT_NAME is unset, it is skipped with a warning
and the other three detectors continue.  The demo foregrounds categorical
detection, so Vertex availability is never a blocker for a live demo.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Required flag schema keys (§5) ───────────────────────────────────────────
# Every detector must emit flags with these keys.  If one is missing it is
# a bug in that detector — fail loudly rather than passing garbage to Gemini.
_REQUIRED_FLAG_KEYS = frozenset({
    "flag_id", "record_id", "flag_type", "severity",
    "entity_type", "field", "observed", "explanation",
})


def _validate_flags(flags: list[dict]) -> None:
    """Raise ValueError if any flag is missing a required §5 key."""
    for flag in flags:
        missing = _REQUIRED_FLAG_KEYS - set(flag.keys())
        if missing:
            raise ValueError(
                f"Flag from detector is missing required §5 keys {missing}. "
                f"Offending flag: {json.dumps(flag, default=str)}"
            )


def run_audit(records: list[dict]) -> dict:
    """Run the full ConstructIQ audit pipeline on a set of normalized records.

    records: canonical worker/equipment dicts produced by ingestion.normalizer.
    Returns an audit report dict:
      {
        "flags":         list of flag dicts (§5 schema, Gemini-narrated),
        "summary":       str — one-paragraph plain-English audit summary,
        "counts":        {"high": int, "medium": int, "low": int},
        "total_records": int
      }
    """
    from detection import missing_values, categorical_anomalies, equipment_standby, numerical_outliers
    from agent import audit_agent
    from database.mongo_client import audit_reports

    all_flags: list[dict] = []

    # ── Layer 1: Missing values ───────────────────────────────────────────────
    mv_flags = missing_values.detect(records)
    _validate_flags(mv_flags)
    all_flags.extend(mv_flags)
    logger.info("Missing-value detector: %d flag(s)", len(mv_flags))

    # ── Layer 2: Categorical anomalies ────────────────────────────────────────
    cat_flags = categorical_anomalies.detect(records)
    _validate_flags(cat_flags)
    all_flags.extend(cat_flags)
    logger.info("Categorical detector: %d flag(s)", len(cat_flags))

    # ── Layer 3: Equipment standby ────────────────────────────────────────────
    sb_flags = equipment_standby.detect(records)
    _validate_flags(sb_flags)
    all_flags.extend(sb_flags)
    logger.info("Equipment-standby detector: %d flag(s)", len(sb_flags))

    # ── Layer 4: Numerical outliers (non-fatal) ───────────────────────────────
    # Requires a live Vertex AI endpoint.  Any failure is caught and logged;
    # the audit continues with the three rule-based detectors' flags intact.
    try:
        num_flags = numerical_outliers.detect(records)
        _validate_flags(num_flags)
        all_flags.extend(num_flags)
        logger.info("Numerical-outlier detector: %d flag(s)", len(num_flags))
    except Exception as exc:
        logger.warning(
            "Numerical-outlier detector skipped (Vertex endpoint unavailable or "
            "errored): %s.  Audit continues with the other three detectors.", exc
        )

    # ── Gemini narration and ranking ──────────────────────────────────────────
    narration = audit_agent.narrate_and_rank(all_flags)
    narrated_flags = narration["flags"]
    summary        = narration["summary"]

    # ── Tally severity counts ─────────────────────────────────────────────────
    counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for flag in narrated_flags:
        sev = flag.get("severity", "low")
        counts[sev] = counts.get(sev, 0) + 1

    report = {
        "report_id":     str(uuid.uuid4()),
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "flags":         narrated_flags,
        "summary":       summary,
        "counts":        counts,
        "total_records": len(records),
    }

    # ── Persist to MongoDB ────────────────────────────────────────────────────
    try:
        audit_reports.insert_one({**report})
        logger.info("Audit report saved to MongoDB (report_id=%s)", report["report_id"])
    except Exception as exc:
        logger.warning("Failed to write audit report to MongoDB: %s", exc)

    return report


# ── CLI entry point — runs the full pipeline on synthetic data ────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )

    data_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "data", "synthetic", "timesheets.json")
    )
    if not os.path.exists(data_path):
        print(f"Synthetic data not found at {data_path}.")
        print("Run:  python3 scripts/generate_synthetic_data.py")
        raise SystemExit(1)

    with open(data_path) as f:
        records = json.load(f)

    # The full synthetic dataset is 3829 records / 200 days, built for ML
    # training. Sending all of it through Gemini narration produces 400+
    # flags in one prompt, which is too slow for an interactive run. Use the
    # same single-day demo slice as the Streamlit "sample timesheet" button
    # (frontend/app.py: _SAMPLE_DATE) — ~20 records covering all four
    # anomaly types.
    sample_date = "2025-01-11"
    records = [r for r in records if r.get("date") == sample_date]

    print(f"Loaded {len(records)} synthetic records.  Running audit pipeline...\n")

    report = run_audit(records)

    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total records scanned : {report['total_records']}")
    print(f"Flags found           : {len(report['flags'])} "
          f"(high={report['counts']['high']}, "
          f"medium={report['counts']['medium']}, "
          f"low={report['counts']['low']})")
    print()
    print("SUMMARY:")
    print(report["summary"])
    print()
    print("FIRST 10 FLAGS:")
    print(json.dumps(report["flags"][:10], indent=2, default=str))
