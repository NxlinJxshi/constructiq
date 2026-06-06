# ============================================================
# MODULE: database/__init__.py
# PURPOSE: Marks the database directory as a Python package so the
#          MongoDB client and collection helpers are importable by name.
# PIPELINE STAGE: Database (used by all other stages)
# INPUTS: None — package initializer only
# OUTPUTS: Makes mongo_client, baselines, and timesheets modules
#          importable as database.<module>
# ============================================================

"""Database package for MongoDB connectivity and collection access."""
