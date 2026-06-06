# ============================================================
# MODULE: prompts.py
# PURPOSE: Stores the system and user prompt templates that instruct
#          the Vertex AI model on how to analyze anomalies and format
#          its findings consistently.
# PIPELINE STAGE: Agent (step 3 of 4 — prompt layer)
# INPUTS: Anomaly dict values interpolated into the template strings
# OUTPUTS: Fully rendered prompt strings ready to send to the model API
# ============================================================

"""Prompt templates used by the audit agent for anomaly explanation and report generation."""
