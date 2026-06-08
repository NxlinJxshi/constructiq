# ============================================================
# MODULE: api/routes.py
# PURPOSE: Flask HTTP endpoints that Google Agent Builder and the Streamlit
#          frontend call to submit timesheets and retrieve audit results.
# PIPELINE STAGE: API (external interface)
# INPUTS: POST /audit — multipart PDF upload or JSON records body
# OUTPUTS: JSON audit report dict from main.run_audit
# ============================================================

"""
Two endpoints:

  POST /audit
    Accepts either an HCSS timesheet PDF (multipart, field name "file") or a
    JSON body with a "records" key containing already-normalised timesheet dicts.
    PDF path: pdf_parser → normalizer → run_audit.
    Records path: run_audit directly.
    Returns the full audit report dict as JSON.

  GET /health
    Returns {"status": "ok"} — used by Agent Builder and uptime checks.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

from flask import Flask, jsonify, request

app = Flask(__name__)
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    """Lightweight liveness check used by Agent Builder and monitoring."""
    return jsonify({"status": "ok"}), 200


@app.post("/audit")
def audit():
    """Run the full audit pipeline and return a ranked anomaly report.

    Accepts two input shapes:
      1. Multipart form with a 'file' field containing an HCSS PDF.
      2. JSON body with a 'records' key containing normalised timesheet dicts.

    Returns the audit report dict:
      {"flags": [...], "summary": str, "counts": {...}, "total_records": int}
    """
    from main import run_audit

    # ── Path A: PDF upload ────────────────────────────────────────────────────
    if "file" in request.files:
        from ingestion.pdf_parser import parse_timesheet_pdf, ReductoUnavailableError
        from ingestion.normalizer import normalize_timesheet

        uploaded = request.files["file"]
        project_id = request.form.get("project_id", "unknown")
        date_str   = request.form.get("date", "unknown")

        # Write the upload to a temp file; Reducto SDK needs a file path.
        suffix = os.path.splitext(uploaded.filename or ".pdf")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            uploaded.save(tmp_path)

        try:
            raw_rows = parse_timesheet_pdf(tmp_path)
            records  = normalize_timesheet(raw_rows, project_id, date_str)
        except ReductoUnavailableError as exc:
            return jsonify({
                "error": "ocr_unavailable",
                "message": (
                    f"Reducto OCR is not configured: {exc}  "
                    "POST a JSON body with a 'records' key to bypass OCR."
                ),
            }), 503
        except Exception as exc:
            logger.exception("PDF ingestion failed")
            return jsonify({"error": "ingestion_error", "message": str(exc)}), 500
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ── Path B: pre-normalised records JSON ───────────────────────────────────
    elif request.is_json:
        body = request.get_json(force=True, silent=True) or {}
        records = body.get("records")
        if not isinstance(records, list):
            return jsonify({
                "error": "bad_request",
                "message": "JSON body must have a 'records' key with a list of timesheet record dicts.",
            }), 400

    else:
        return jsonify({
            "error": "bad_request",
            "message": "Send a multipart PDF upload (field 'file') or a JSON body with 'records'.",
        }), 400

    # ── Run the audit pipeline ────────────────────────────────────────────────
    try:
        report = run_audit(records)
    except Exception as exc:
        logger.exception("Audit pipeline failed")
        return jsonify({"error": "pipeline_error", "message": str(exc)}), 500

    return jsonify(report), 200


if __name__ == "__main__":
    # Development server only — not for production.
    app.run(debug=True, port=5000)
