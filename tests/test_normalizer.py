# ============================================================
# MODULE: test_normalizer.py
# PURPOSE: Unit tests for ingestion/normalizer.py covering all parse
#          functions and the full normalize_timesheet pipeline.
# PIPELINE STAGE: Test suite (Ingestion)
# INPUTS: Hard-coded cell strings and raw row dicts
# OUTPUTS: pytest pass/fail results
# ============================================================

"""
Tests every cell-parsing rule and the full normalize_timesheet pipeline.

The parse_cell tests cover all eight HCSS cell formats by name (standard slash,
leading slash, trailing slash, pure ST, empty, whitespace, None, and malformed).
The normalize_timesheet tests use a minimal 2-worker + 1-equipment fixture that
exercises zero-skip logic, the "/1" leading-slash format, and schema propagation.
"""

import sys
import os

# Add project root to path so `from ingestion.normalizer import ...` works
# when pytest is run from any directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ingestion.normalizer import (
    parse_cell,
    parse_equipment_cell,
    parse_total_column,
    normalize_timesheet,
)


# ── parse_cell ────────────────────────────────────────────────────────────────

class TestParseCell:
    # Standard "ST/OT" format — the most common cell on a real HCSS card.
    def test_slash_both_values(self):
        assert parse_cell("8/2.5") == (8.0, 2.5)

    # Some PDF exports add spaces around the slash; must still parse correctly.
    def test_slash_with_surrounding_whitespace(self):
        assert parse_cell("8 / 2.5") == (8.0, 2.5)

    # Leading slash ("/1") means pure overtime — worker logged no straight time
    # on this cost code. Without this case, the parser would raise ValueError
    # and the entire timesheet row would fail ingestion.
    def test_leading_slash_st_is_zero(self):
        assert parse_cell("/1") == (0.0, 1.0)

    # No slash at all means pure straight time — the simplest possible entry.
    def test_no_slash_pure_straight_time(self):
        assert parse_cell("8") == (8.0, 0.0)

    # Trailing slash ("8/") is an OCR artifact; treat the missing OT side as 0.
    def test_trailing_slash_ot_is_zero(self):
        assert parse_cell("8/") == (8.0, 0.0)

    # Blank cell — worker had no activity on this cost code today.
    def test_empty_string(self):
        assert parse_cell("") == (0.0, 0.0)

    # OCR sometimes returns whitespace for visually blank cells.
    def test_whitespace_only(self):
        assert parse_cell("   ") == (0.0, 0.0)

    # Reducto returns None for completely empty cells (no text at all).
    def test_none_input(self):
        assert parse_cell(None) == (0.0, 0.0)

    # Non-numeric string means OCR misread the cell; must raise so the caller
    # can route the row to manual review instead of silently storing garbage.
    def test_malformed_alpha_raises(self):
        with pytest.raises(ValueError):
            parse_cell("abc")

    # Two slashes has no valid HCSS interpretation — always malformed.
    def test_malformed_too_many_slashes_raises(self):
        with pytest.raises(ValueError):
            parse_cell("8/2/3")


# ── parse_equipment_cell ──────────────────────────────────────────────────────

class TestParseEquipmentCell:
    def test_integer_eight(self):
        assert parse_equipment_cell("8") == 8.0

    def test_integer_four(self):
        assert parse_equipment_cell("4") == 4.0

    # Equipment not deployed on this cost code — blank cell, not an error.
    def test_empty_string(self):
        assert parse_equipment_cell("") == 0.0

    def test_none_input(self):
        assert parse_equipment_cell(None) == 0.0


# ── parse_total_column ────────────────────────────────────────────────────────

class TestParseTotalColumn:
    # Reducto stacks the daily total and project-to-date cumulative in one string.
    # We only want the first (daily) value; the cumulative is discarded.
    def test_stacked_with_newline(self):
        assert parse_total_column("10.5\n24/7") == 10.5

    def test_stacked_with_space(self):
        assert parse_total_column("10.5 24/7") == 10.5

    def test_single_value(self):
        assert parse_total_column("10.5") == 10.5


# ── normalize_timesheet ───────────────────────────────────────────────────────

class TestNormalizeTimesheet:
    """
    Minimal 2-worker + 1-equipment fixture covering:
    - Standard "8/2.5" cell (worker 1, code 51214.4)
    - Leading-slash "/1" cell (worker 1, code 51234.5)
    - Pure straight-time "8" cell (worker 2, code 51214.4)
    - Empty cell that must be skipped (worker 2, code 51234.5)
    - Equipment active on one code, blank on another
    """

    def _raw_rows(self):
        return [
            {
                "row_type": "worker",
                "id": "903784",
                "name": "Steven Powell",
                "labor_class": "SLLS-L1",
                "shift_start": "06:30",
                "shift_end": "17:30",
                "parent_worker_id": None,
                "cells": {
                    "51214.4": "8/2.5",
                    "51234.5": "/1",       # leading slash: 0 ST, 1 OT
                },
            },
            {
                "row_type": "worker",
                "id": "903785",
                "name": "Maria Garcia",
                "labor_class": "SLSDL-FM",
                "shift_start": "06:30",
                "shift_end": "17:30",
                "parent_worker_id": None,
                "cells": {
                    "51214.4": "8",
                    "51234.5": "",         # empty → must be skipped
                },
            },
            {
                "row_type": "equipment",
                "id": "UAT7698",
                "name": "SKYTRAK 10054",
                "labor_class": "SLLS-A1",
                "shift_start": None,
                "shift_end": None,
                "parent_worker_id": "903784",
                "cells": {
                    "51234.5": "8",
                    "51214.4": "",         # empty → must be skipped
                },
            },
        ]

    def test_total_record_count(self):
        # Worker 1: 2 records | Worker 2: 1 record | Equipment: 1 record = 4 total
        records = normalize_timesheet(self._raw_rows(), "31082-00", "2025-01-06")
        assert len(records) == 4

    def test_worker_record_hours(self):
        records = normalize_timesheet(self._raw_rows(), "31082-00", "2025-01-06")
        r = next(r for r in records
                 if r.get("worker_id") == "903784" and r["cost_code"] == "51214.4")
        assert r["straight_time_hours"] == 8.0
        assert r["overtime_hours"] == 2.5
        assert r["total_hours"] == 10.5

    def test_leading_slash_cell(self):
        # "/1" must produce 0 ST and 1 OT — not a ValueError
        records = normalize_timesheet(self._raw_rows(), "31082-00", "2025-01-06")
        r = next(r for r in records
                 if r.get("worker_id") == "903784" and r["cost_code"] == "51234.5")
        assert r["straight_time_hours"] == 0.0
        assert r["overtime_hours"] == 1.0
        assert r["total_hours"] == 1.0

    def test_anomaly_fields_are_none(self):
        # The normalizer must never plant anomalies — that's the generator's job
        records = normalize_timesheet(self._raw_rows(), "31082-00", "2025-01-06")
        for r in records:
            assert r["anomaly_type"] is None
            assert r["anomaly_detail"] is None

    def test_equipment_record_schema(self):
        records = normalize_timesheet(self._raw_rows(), "31082-00", "2025-01-06")
        eq = [r for r in records if r["entity_type"] == "equipment"]
        assert len(eq) == 1
        assert eq[0]["hours"] == 8.0
        assert eq[0]["operator_worker_id"] == "903784"
        assert eq[0]["cost_code"] == "51234.5"
        assert eq[0]["equipment_name"] == "SKYTRAK 10054"

    def test_empty_cells_skipped(self):
        # HCSS prints all cost code columns in every row; blank ones must not become records
        records = normalize_timesheet(self._raw_rows(), "31082-00", "2025-01-06")
        worker2 = [r for r in records if r.get("worker_id") == "903785"]
        assert len(worker2) == 1
        assert worker2[0]["cost_code"] == "51214.4"

    def test_project_id_and_date_propagated(self):
        records = normalize_timesheet(self._raw_rows(), "31082-00", "2025-01-06")
        for r in records:
            assert r["project_id"] == "31082-00"
            assert r["date"] == "2025-01-06"
