# ============================================================
# MODULE: normalizer.py
# PURPOSE: Convert raw cell strings parsed from HCSS PDFs into canonical
#          worker and equipment records conforming to the project schema.
# PIPELINE STAGE: Ingestion (step 1b — normalization)
# INPUTS: List of raw row dicts produced by pdf_parser.py post-OCR extraction
# OUTPUTS: List of canonical worker/equipment record dicts; anomaly fields
#          are always None here (only synthetic generator plants anomalies)
# ============================================================

"""
Converts raw Reducto OCR output into typed, schema-conforming records.

HCSS timesheet cells use a slash-delimited format to encode straight time
and overtime (e.g. "8/2.5" = 8 ST hrs, 2.5 OT hrs; "/1" = pure overtime).
This module is the single place in the codebase that understands that format.
Everything downstream receives clean dicts with typed float fields.

Concrete example:
  raw cell "8/2.5" on cost code "51214.4" for worker "903784"
  → {"entity_type": "worker", "straight_time_hours": 8.0, "overtime_hours": 2.5,
     "total_hours": 10.5, "cost_code": "51214.4", "worker_id": "903784", ...}
"""

from __future__ import annotations  # allows str | None syntax on Python 3.9

import uuid
from typing import Optional


def parse_cell(cell: Optional[str]) -> tuple[float, float]:
    """Parse an HCSS cell string into (straight_time, overtime) hours.

    Handles all HCSS formats: "8/2.5", "8 / 2.5", "/1" (pure OT), "8" (pure ST),
    "8/" (trailing slash), "", "   ", and None. Raises ValueError on genuinely
    malformed input like "abc" or "8/2/3" (too many slashes).
    """
    if cell is None:
        return (0.0, 0.0)
    cell = cell.strip()
    if not cell:
        return (0.0, 0.0)

    if "/" in cell:
        parts = [p.strip() for p in cell.split("/")]
        if len(parts) != 2:
            # e.g. "8/2/3" — more than one slash is always malformed
            raise ValueError(f"Malformed cell (too many slashes): {cell!r}")
        st_str, ot_str = parts
        try:
            # Empty left side → 0 ST (leading-slash format); empty right → 0 OT (trailing slash)
            st = float(st_str) if st_str else 0.0
            ot = float(ot_str) if ot_str else 0.0
        except ValueError:
            raise ValueError(f"Malformed cell (non-numeric): {cell!r}")
        return (st, ot)

    # No slash — the entire value is pure straight time
    try:
        return (float(cell), 0.0)
    except ValueError:
        raise ValueError(f"Malformed cell (non-numeric): {cell!r}")


def parse_equipment_cell(cell: Optional[str]) -> float:
    """Parse an HCSS equipment cell into a single hours float.

    Equipment rows have no ST/OT split — just total hours run.
    Returns 0.0 for empty or None input (equipment not deployed on that code).
    """
    if cell is None:
        return 0.0
    cell = cell.strip()
    if not cell:
        return 0.0
    try:
        return float(cell)
    except ValueError:
        raise ValueError(f"Malformed equipment cell: {cell!r}")


def parse_total_column(cell: Optional[str]) -> float:
    """Parse the HCSS Total column, returning only the daily total.

    Reducto may return the daily total and the project-to-date cumulative stacked
    in one string (e.g. "10.5\n24/7"). Only the first token — the daily value — is used.
    """
    if cell is None:
        return 0.0
    cell = cell.strip()
    if not cell:
        return 0.0
    first_token = cell.split()[0]  # split() handles both \n and space separators
    try:
        return float(first_token)
    except ValueError:
        raise ValueError(f"Cannot parse total column value: {cell!r}")


# ── Private record builders ───────────────────────────────────────────────────

def _make_worker_record(
    row: dict, cost_code: str, st: float, ot: float,
    project_id: str, date: str,
) -> dict:
    """Assemble a canonical worker record from a parsed row and cell values."""
    return {
        "record_id": str(uuid.uuid4()),
        "date": date,
        "project_id": project_id,
        "worker_id": row["id"],
        "worker_name": row["name"],
        "labor_class": row.get("labor_class"),
        "cost_code": cost_code,
        "straight_time_hours": st,
        "overtime_hours": ot,
        "total_hours": round(st + ot, 4),
        "shift_start": row.get("shift_start"),
        "shift_end": row.get("shift_end"),
        "entity_type": "worker",
        "anomaly_type": None,
        "anomaly_detail": None,
    }


def _make_equipment_record(
    row: dict, cost_code: str, hours: float,
    project_id: str, date: str,
) -> dict:
    """Assemble a canonical equipment record from a parsed row and cell value."""
    return {
        "record_id": str(uuid.uuid4()),
        "date": date,
        "project_id": project_id,
        "equipment_id": row["id"],
        "equipment_name": row["name"],
        "cost_code": cost_code,
        "hours": hours,
        "operator_worker_id": row.get("parent_worker_id"),
        "operator_labor_class": row.get("labor_class"),
        "entity_type": "equipment",
        "anomaly_type": None,
        "anomaly_detail": None,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def normalize_timesheet(
    raw_rows: list[dict],
    project_id: str,
    date: str,
) -> list[dict]:
    """Convert raw Reducto row dicts into canonical worker and equipment records.

    Emits one record per (entity, cost_code) pair where parsed hours > 0.
    HCSS prints a cell for every cost code column in every row even when a
    worker had no activity there — those zero-hour cells are skipped.
    anomaly_type and anomaly_detail are always None (anomalies are only
    planted by the synthetic generator, never during real ingestion).
    """
    records: list[dict] = []

    for row in raw_rows:
        for cost_code, cell_str in row["cells"].items():
            if row["row_type"] == "worker":
                st, ot = parse_cell(cell_str)
                if st + ot == 0.0:
                    continue  # no activity on this cost code — skip the blank cell
                records.append(
                    _make_worker_record(row, cost_code, st, ot, project_id, date)
                )
            elif row["row_type"] == "equipment":
                hours = parse_equipment_cell(cell_str)
                if hours == 0.0:
                    continue
                records.append(
                    _make_equipment_record(row, cost_code, hours, project_id, date)
                )

    return records
