# ============================================================
# MODULE: baselines.py
# PURPOSE: Provides helper functions to read baseline documents from
#          MongoDB so detection modules can look up expected values
#          for any cost code without writing raw queries everywhere.
# PIPELINE STAGE: Database (read layer for Detection stage)
# INPUTS: cost_code string (query key)
# OUTPUTS: Baseline dict containing avg_hours_per_worker, std_dev,
#          avg_equipment_hours, typical_labor_classes, max_overtime_ratio
# ============================================================

"""Reads and writes baseline cost and labor metrics used for anomaly comparison."""
