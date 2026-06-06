"""
Entry point for the ConstructIQ pipeline.

Orchestrates the full audit workflow: ingesting timesheet PDFs via Reducto,
normalizing the extracted data, running anomaly detection (missing values,
numerical outliers, equipment standby), and invoking the Vertex AI audit
agent to generate findings stored in MongoDB.
"""
