# ============================================================
# MODULE: pdf_parser.py
# PURPOSE: Sends raw timesheet PDF files to the Reducto API and returns
#          structured JSON data representing each row of the timesheet.
# PIPELINE STAGE: Ingestion (step 1a — extraction)
# INPUTS: PDF file path or binary blob from the user upload
# OUTPUTS: List of raw row dicts with keys like worker_name, cost_code,
#          hours, date, equipment — as returned by Reducto
# ============================================================

"""Handles PDF parsing via the Reducto API to extract structured timesheet data."""
