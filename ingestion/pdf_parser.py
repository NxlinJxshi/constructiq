# ============================================================
# MODULE: ingestion/pdf_parser.py
# PURPOSE: Upload an HCSS HeavyJob timesheet PDF to Reducto Extract and
#          return raw row dicts that ingestion.normalizer can consume directly.
# PIPELINE STAGE: Ingestion (step 1a — OCR extraction)
# INPUTS: Path to an HCSS HeavyJob daily timesheet PDF
# OUTPUTS: List of raw HCSS row dicts (row_type, id, name, cells, …) ready
#          for ingestion.normalizer.normalize_timesheet — does NOT normalize.
#
# ALTERNATIVE INGESTION PATH (not implemented here):
#   Reducto also ships an MCP server.  Options:
#   - Hosted:  https://mcp.reducto.ai/mcp  (Bearer auth with REDUCTO_API_KEY)
#   - Local:   uvx mcp-server-reducto
#   If Claude Code is later wired to Reducto via .mcp.json, the extraction
#   could be done via MCP tool calls instead of this SDK integration.
#   That path is useful for interactive exploration; this module is the
#   production path used by the pipeline.
# ============================================================

"""
Calls Reducto Extract to pull structured rows from an HCSS HeavyJob PDF.

Uses the `reductoai` SDK (from reducto import Reducto).  The client reads
REDUCTO_API_KEY from the environment automatically — never hardcoded here.

If the key is missing or the API call fails, raises ReductoUnavailableError so
the pipeline can fall back to synthetic data with zero code change.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ReductoUnavailableError(Exception):
    """Raised when Reducto is unreachable or unconfigured.

    Callers should catch this and fall back to synthetic data rather than
    crashing the audit pipeline.
    """


# ── HCSS HeavyJob extraction schema ──────────────────────────────────────────
#
# This schema describes one HCSS daily timesheet.  Reducto uses the
# descriptions to drive extraction accuracy — write them exactly as the
# fields appear on a real HCSS HeavyJob card.
#
# The entries array matches the raw_rows contract that ingestion.normalizer
# expects: row_type / id / name / labor_class / shift_start / shift_end /
# parent_worker_id / cells (cost_code → raw hour string).
HCSS_SCHEMA = {
    "type": "object",
    "properties": {
        "project_name": {
            "type": "string",
            "description": (
                "The project name printed in the header of the HCSS HeavyJob "
                "daily timesheet, e.g. 'Westmorland Elementary School'."
            ),
        },
        "project_id": {
            "type": "string",
            "description": (
                "The HCSS project number in the timesheet header, "
                "e.g. '31062-00'."
            ),
        },
        "date": {
            "type": "string",
            "description": (
                "The shift date from the timesheet header in YYYY-MM-DD format, "
                "e.g. '2024-03-15'.  Convert any printed date format to ISO 8601."
            ),
        },
        "foreman": {
            "type": "string",
            "description": (
                "The foreman name printed in the timesheet header, "
                "as it appears on the HCSS card."
            ),
        },
        "shift_start": {
            "type": "string",
            "description": (
                "Project-wide default shift start time in HH:MM 24-hour format, "
                "e.g. '06:30'.  This appears in the timesheet header and applies "
                "to all worker rows that do not have their own start time."
            ),
        },
        "shift_end": {
            "type": "string",
            "description": (
                "Project-wide default shift end time in HH:MM 24-hour format, "
                "e.g. '17:30'.  From the timesheet header."
            ),
        },
        "entries": {
            "type": "array",
            "description": (
                "Every worker and equipment row in the timesheet body, in the "
                "order they appear.  Worker rows are followed immediately by any "
                "equipment rows nested under that worker in the HCSS layout."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "row_type": {
                        "type": "string",
                        "description": (
                            "'worker' for an employee row (person billing hours), "
                            "'equipment' for a piece of equipment or vehicle row.  "
                            "In HCSS, equipment rows appear indented beneath the "
                            "operator who used them."
                        ),
                    },
                    "id": {
                        "type": "string",
                        "description": (
                            "The employee ID or asset/equipment ID as printed in "
                            "HCSS.  For workers this is a numeric employee number "
                            "(e.g. '903784'); for equipment it is the asset tag "
                            "(e.g. 'UAT7698')."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            "For worker rows: the employee full name exactly as "
                            "printed (e.g. 'Steven Powell').  For equipment rows: "
                            "the HCSS asset description (e.g. 'SKYTRAK 10054' or "
                            "'JLG 400S BOOM LIFT')."
                        ),
                    },
                    "labor_class": {
                        "type": "string",
                        "description": (
                            "The HCSS labor classification code for this row, "
                            "e.g. 'SLLS-L1', 'SLSDL-FM', 'SLLS-A1'.  For "
                            "equipment rows this is the operator's labor class."
                        ),
                    },
                    "shift_start": {
                        "type": "string",
                        "description": (
                            "This row's shift start time in HH:MM 24-hour format.  "
                            "Use the project-wide default from the header if no "
                            "per-row start time is printed.  Null for equipment rows."
                        ),
                    },
                    "shift_end": {
                        "type": "string",
                        "description": (
                            "This row's shift end time in HH:MM 24-hour format.  "
                            "Use the project-wide default from the header if no "
                            "per-row end time is printed.  Null for equipment rows."
                        ),
                    },
                    "parent_worker_id": {
                        "type": "string",
                        "description": (
                            "For equipment rows only: the employee ID of the worker "
                            "who operated this equipment on this shift.  This is the "
                            "'id' value of the worker row immediately above this "
                            "equipment row in the HCSS layout.  Null for worker rows."
                        ),
                    },
                    "cells": {
                        "type": "object",
                        "description": (
                            "Mapping of HCSS cost code (e.g. '51214.4') to the raw "
                            "hours string exactly as printed in that cell of the "
                            "timesheet grid.  Worker cells use an ST/OT slash format: "
                            "'8/2.5' means 8 straight-time and 2.5 overtime hours; "
                            "'/1' means 1 overtime hour only; '8' means 8 straight-time "
                            "only.  Equipment cells are a single number (e.g. '8').  "
                            "Omit cost codes where the cell is blank — do not include "
                            "empty strings."
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["row_type", "id", "name", "cells"],
            },
        },
    },
    "required": ["date", "entries"],
}

_SYSTEM_PROMPT = (
    "This document is an HCSS HeavyJob daily construction timesheet.  "
    "The timesheet header contains the project name, project ID, date, and foreman.  "
    "The timesheet body is a grid: rows are workers or pieces of equipment; "
    "columns are HCSS cost codes.  Each cell records hours worked by that worker "
    "or equipment on that cost code for the shift.  "
    "Worker cells use an ST/OT slash format ('8/2.5' = 8 straight-time + 2.5 overtime).  "
    "Equipment rows appear indented below the operator who used them.  "
    "Extract every row on every page — do not stop early."
)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_timesheet_pdf(pdf_path: str) -> list[dict]:
    """Send a PDF to Reducto Extract and return raw HCSS row dicts.

    Returns a list of raw row dicts that ingestion.normalizer.normalize_timesheet
    expects as its raw_rows argument.  Does NOT normalize — that is the
    normalizer's job.

    Raises ReductoUnavailableError if REDUCTO_API_KEY is missing or the
    Reducto call fails, so callers can fall back to synthetic data.
    """
    api_key = os.environ.get("REDUCTO_API_KEY", "").strip()
    if not api_key:
        raise ReductoUnavailableError(
            "REDUCTO_API_KEY is not set.  Add it to .env to enable PDF ingestion.  "
            "Until then, use the synthetic data fallback path."
        )

    try:
        from reducto import Reducto
    except ImportError as exc:
        raise ReductoUnavailableError(
            f"reductoai package not installed: {exc}.  Run: pip install reductoai"
        ) from exc

    try:
        client = Reducto(api_key=api_key)

        with open(pdf_path, "rb") as f:
            upload = client.upload(file=f)

        response = client.extract.run(
            input=upload.file_id,
            instructions={
                "schema": HCSS_SCHEMA,
                "system_prompt": _SYSTEM_PROMPT,
            },
            settings={"array_extract": True},
        )
    except ReductoUnavailableError:
        raise
    except Exception as exc:
        raise ReductoUnavailableError(
            f"Reducto Extract call failed: {exc}"
        ) from exc

    # Reducto returns a sync result or an async job object (URL-backed extract).
    # The async case carries a job_id but no result data.  Use duck-typing rather
    # than a deep SDK import so any SDK version change can't escape as a bare ImportError.
    if hasattr(response, "job_id") and not hasattr(response, "result"):
        raise ReductoUnavailableError(
            f"Reducto returned an async job (job_id={response.job_id}) instead of "
            "an inline result.  The SDK does not expose a get_job endpoint for "
            "Extract in this version.  Retry the extract call or contact Reducto support."
        )

    result = response.result
    if not result:
        raise ReductoUnavailableError(
            "Reducto returned an empty result.  The PDF may be unreadable or the "
            "extraction schema may not have matched any content."
        )

    # result is List[object] with one element per document chunk.
    # With array_extract=True and disable_chunking=True (default), it is a list
    # of length 1.  The single element is the extracted dict matching HCSS_SCHEMA.
    doc = result[0] if isinstance(result, list) else result
    if not isinstance(doc, dict):
        doc = dict(doc)

    entries = doc.get("entries", [])
    if not entries:
        raise ReductoUnavailableError(
            "Reducto returned a result with no entries.  The timesheet may be "
            "empty or the extraction schema did not match the document structure."
        )

    return entries
