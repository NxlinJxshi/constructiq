# ============================================================
# MODULE: normalizer.py
# PURPOSE: Converts Reducto's raw extraction output into a consistent
#          internal schema so all downstream modules can make the same
#          field-name assumptions.
# PIPELINE STAGE: Ingestion (step 1b — normalization)
# INPUTS: Raw row dicts from pdf_parser (varied field names, mixed types)
# OUTPUTS: Normalized dicts with guaranteed fields: worker_name, cost_code,
#          date (ISO string), regular_hours (float), overtime_hours (float),
#          equipment_hours (float), labor_class (string)
# ============================================================

"""Normalizes raw extracted timesheet data into a consistent schema for downstream processing."""
