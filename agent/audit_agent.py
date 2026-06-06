# ============================================================
# MODULE: audit_agent.py
# PURPOSE: Sends detected anomalies to a Vertex AI generative model
#          which reasons over them and produces a structured audit
#          finding (severity, explanation, recommended action).
# PIPELINE STAGE: Agent (step 3 of 4)
# INPUTS: List of anomaly dicts from the detection package + prompt
#         templates from prompts.py
# OUTPUTS: Structured audit finding dicts saved to MongoDB audit_reports;
#          each finding includes severity, a plain-English explanation,
#          and a recommended corrective action for the project manager
# ============================================================

"""Vertex AI agent that reasons over detected anomalies and produces audit findings."""
