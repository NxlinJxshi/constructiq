# ============================================================
# MODULE: timesheets.py
# PURPOSE: Encapsulates all read/write operations for the timesheets
#          collection so the rest of the pipeline never writes raw
#          PyMongo queries against that collection directly.
# PIPELINE STAGE: Database (write layer after Ingestion; read layer for Detection)
# INPUTS: Normalized timesheet dicts from the ingestion package
# OUTPUTS: Timesheet documents persisted in MongoDB; query results
#          returned as lists of dicts to callers
# ============================================================

"""Handles CRUD operations for timesheet documents stored in MongoDB."""
