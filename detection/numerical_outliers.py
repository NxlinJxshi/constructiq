# ============================================================
# MODULE: numerical_outliers.py
# PURPOSE: Compares numeric fields (hours, rates) against the stored
#          baselines and flags rows that fall more than 2 standard
#          deviations from the expected mean for their cost code.
# PIPELINE STAGE: Detection (step 2b)
# INPUTS: Normalized timesheet dicts + baselines collection (from MongoDB)
# OUTPUTS: List of outlier anomaly dicts with the field, observed value,
#          expected range, and z-score so the agent can explain the finding
# ============================================================

"""Identifies statistical outliers in numerical timesheet fields such as hours and rates."""
