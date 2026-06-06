# ============================================================
# MODULE: equipment_standby.py
# PURPOSE: Identifies equipment that was billed as operating but whose
#          hours suggest it was idle or on standby, which inflates costs
#          without corresponding productive output.
# PIPELINE STAGE: Detection (step 2c)
# INPUTS: Normalized timesheet dicts (equipment_hours field) + baselines
#         (avg_equipment_hours per cost code)
# OUTPUTS: List of standby anomaly dicts describing which equipment item,
#          which shift, and the gap between billed and expected hours
# ============================================================

"""Flags equipment recorded as idle or on standby when it should be active."""
