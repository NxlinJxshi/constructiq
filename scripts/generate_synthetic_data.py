# ============================================================
# MODULE: generate_synthetic_data.py
# PURPOSE: Generates realistic fake timesheet CSV/JSON data so the
#          pipeline can be tested end-to-end without real client files.
# PIPELINE STAGE: Development utility (not part of production pipeline)
# INPUTS: Configuration constants defined at the top of this file
#         (number of workers, date range, cost codes to include)
# OUTPUTS: Synthetic timesheet records written to data/synthetic/ as
#          JSON or CSV files that pdf_parser and normalizer can process
# ============================================================

"""Generates synthetic construction timesheet data for development and testing."""
