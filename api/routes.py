# ============================================================
# MODULE: routes.py
# PURPOSE: Defines the HTTP endpoints that external clients (the
#          Streamlit frontend, CI scripts, or direct API calls) use
#          to submit timesheets and retrieve audit results.
# PIPELINE STAGE: API (step 4 of 4)
# INPUTS: Multipart PDF upload (POST /upload) or a report ID (GET /report/<id>)
# OUTPUTS: JSON responses — either a job confirmation with a report_id,
#          or the full audit report document from MongoDB
# ============================================================

"""Flask route definitions for uploading timesheets and retrieving audit results."""
