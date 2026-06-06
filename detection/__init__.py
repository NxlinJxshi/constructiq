# ============================================================
# MODULE: detection/__init__.py
# PURPOSE: Marks the detection directory as a Python package and groups
#          the three anomaly-detection checks under one namespace.
# PIPELINE STAGE: Detection (step 2 of 4)
# INPUTS: None — package initializer only
# OUTPUTS: Makes missing_values, numerical_outliers, and equipment_standby
#          importable as detection.<module>
# ============================================================

"""Detection package containing anomaly detection modules for timesheet auditing."""
