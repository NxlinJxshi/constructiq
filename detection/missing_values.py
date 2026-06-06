# ============================================================
# MODULE: missing_values.py
# PURPOSE: Scans normalized timesheet rows for required fields that are
#          blank, null, or zero when a real value is expected — the
#          simplest class of billing error to catch automatically.
# PIPELINE STAGE: Detection (step 2a)
# INPUTS: List of normalized timesheet dicts from the ingestion package
# OUTPUTS: List of anomaly dicts, each identifying the row, the missing
#          field name, and a human-readable description of the problem
# ============================================================

"""Detects missing or incomplete values in normalized timesheet records."""
