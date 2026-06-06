# ============================================================
# MODULE: ingestion/__init__.py
# PURPOSE: Marks the ingestion directory as a Python package and
#          exposes its public interface to the rest of the pipeline.
# PIPELINE STAGE: Ingestion (step 1 of 4)
# INPUTS: None — package initializer only
# OUTPUTS: Makes pdf_parser and normalizer importable as ingestion.pdf_parser, etc.
# ============================================================

"""Ingestion package for parsing and normalizing construction timesheet PDFs."""
