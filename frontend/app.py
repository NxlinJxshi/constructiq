# ============================================================
# MODULE: app.py
# PURPOSE: Streamlit web app that lets project managers upload timesheet
#          PDFs, trigger an audit run, and view the generated findings
#          in a human-readable dashboard without touching the API directly.
# PIPELINE STAGE: Frontend
# INPUTS: PDF file uploaded via the browser; audit report_id for result lookup
# OUTPUTS: Rendered Streamlit UI showing anomaly table, severity badges,
#          and recommended actions from the audit agent
# ============================================================

"""Streamlit frontend for uploading timesheets and visualizing audit findings."""
