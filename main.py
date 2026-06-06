# ============================================================
# MODULE: main.py
# PURPOSE: Entry point that orchestrates the full ConstructIQ audit pipeline
#          from PDF ingestion through anomaly detection to agent reporting.
# PIPELINE STAGE: Orchestration (top-level)
# INPUTS: Raw timesheet PDF files (via the ingestion package)
# OUTPUTS: Audit report documents written to MongoDB audit_reports collection
# ============================================================

"""
Entry point for the ConstructIQ pipeline.

Orchestrates the full audit workflow: ingesting timesheet PDFs via Reducto,
normalizing the extracted data, running anomaly detection (missing values,
numerical outliers, equipment standby), and invoking the Vertex AI audit
agent to generate findings stored in MongoDB.
"""
