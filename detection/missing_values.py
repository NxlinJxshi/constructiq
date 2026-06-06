# ============================================================
# MODULE: missing_values.py
# PURPOSE: Flag records with null or empty required fields. Pure rule-based;
#          no ML or statistical reasoning — just presence checks.
# PIPELINE STAGE: Detection (step 2a)
# INPUTS: List of canonical worker/equipment record dicts
# OUTPUTS: List of flag dicts, one per missing field per record, using the
#          shared flag schema (flag_id, record_id, flag_type, severity, ...)
# ============================================================

"""
First gate in the detection pipeline: checks that every record has its
required fields populated before statistical detectors run on them.

A field is "missing" if it is None, an empty string, or whitespace-only.
A numeric 0.0 is NOT missing — a worker can legitimately log zero overtime.
Severity (high/medium/low) is a property of the field, not the record:
missing straight_time_hours is always HIGH; missing shift_start is MEDIUM.

Concrete example:
  record with labor_class=None
  → flag {"flag_type": "missing_value", "severity": "medium",
           "field": "labor_class", "explanation": "Worker ... is missing ..."}
"""

from __future__ import annotations

import json
import os
import sys
import uuid

# ── Field definitions ─────────────────────────────────────────────────────────

# Which fields must be non-empty for each entity type
WORKER_REQUIRED = [
    "date", "worker_id", "labor_class", "cost_code",
    "straight_time_hours", "overtime_hours", "total_hours",
    "shift_start", "shift_end",
]
EQUIPMENT_REQUIRED = [
    "date", "equipment_id", "cost_code", "hours",
]

# Severity by field — reflects downstream impact if the field is absent
_HIGH   = {"straight_time_hours", "overtime_hours", "total_hours",
           "cost_code", "hours", "worker_id", "equipment_id"}
_MEDIUM = {"labor_class", "shift_start", "shift_end", "date"}
# Anything not in HIGH or MEDIUM is LOW (e.g. worker_name, equipment_name)

# Maps each field to a plain-English purpose phrase used in the explanation string
_FIELD_PURPOSE = {
    "straight_time_hours": "payroll calculation",
    "overtime_hours":      "payroll calculation",
    "total_hours":         "payroll calculation",
    "hours":               "payroll calculation",
    "cost_code":           "payroll calculation",
    "worker_id":           "payroll calculation",
    "equipment_id":        "payroll calculation",
    "labor_class":         "compliance documentation",
    "shift_start":         "compliance documentation",
    "shift_end":           "compliance documentation",
    "date":                "compliance documentation",
    "worker_name":         "downstream analysis",
    "equipment_name":      "downstream analysis",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _severity(field: str) -> str:
    if field in _HIGH:
        return "high"
    if field in _MEDIUM:
        return "medium"
    return "low"


def _is_missing(value) -> bool:
    """True if value is None, empty string, or whitespace-only. 0.0 is NOT missing."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _explanation(record: dict, field: str, severity: str) -> str:
    """Build a plain-English explanation sentence for the Gemini audit agent."""
    purpose = _FIELD_PURPOSE.get(field, "downstream analysis")
    if record["entity_type"] == "worker":
        return (
            f"Worker {record.get('worker_name', 'Unknown')} "
            f"(ID: {record.get('worker_id', 'Unknown')}) on {record.get('date', 'Unknown')} "
            f"for cost code {record.get('cost_code', 'Unknown')} is missing the {field} field. "
            f"This is a {severity.upper()} severity issue because {field} is required for "
            f"{purpose}."
        )
    return (
        f"Equipment {record.get('equipment_name', 'Unknown')} "
        f"(ID: {record.get('equipment_id', 'Unknown')}) on {record.get('date', 'Unknown')} "
        f"for cost code {record.get('cost_code', 'Unknown')} is missing the {field} field. "
        f"This is a {severity.upper()} severity issue because {field} is required for "
        f"{purpose}."
    )


def _check_record(record: dict) -> list[dict]:
    """Return one flag dict per missing required field in a single record."""
    required = WORKER_REQUIRED if record["entity_type"] == "worker" else EQUIPMENT_REQUIRED
    flags = []
    for field in required:
        if _is_missing(record.get(field)):
            sev = _severity(field)
            flags.append({
                "flag_id":     str(uuid.uuid4()),
                "record_id":   record["record_id"],
                "flag_type":   "missing_value",
                "severity":    sev,
                "entity_type": record["entity_type"],
                "field":       field,
                "observed":    None,
                "explanation": _explanation(record, field, sev),
            })
    return flags


# ── Public API ────────────────────────────────────────────────────────────────

def detect(records: list[dict]) -> list[dict]:
    """Scan all records and return one flag per missing required field.

    The flag schema (flag_id, record_id, flag_type, severity, entity_type,
    field, observed, explanation) is shared with the other detection modules.
    """
    flags: list[dict] = []
    for record in records:
        flags.extend(_check_record(record))
    return flags


# ── CLI entry point ───────────────────────────────────────────────────────────

def _precision_recall(records: list[dict], flags: list[dict]) -> None:
    """Compare flags against ground-truth labels in the synthetic dataset.

    Prints recall (how many planted missing_value anomalies were caught)
    and false positives (how many clean records were incorrectly flagged).
    """
    planted = [r for r in records if r.get("anomaly_type") == "missing_value"]
    clean   = [r for r in records if r.get("anomaly_type") is None]
    flagged_ids     = {f["record_id"] for f in flags}
    true_positives  = sum(1 for r in planted if r["record_id"] in flagged_ids)
    false_positives = sum(1 for r in clean   if r["record_id"] in flagged_ids)
    total = len(planted)
    pct = (true_positives / total * 100) if total else 0.0
    print(f"Recall on planted missing_value anomalies: {true_positives}/{total} ({pct:.1f}%)")
    print(f"False positives on clean records: {false_positives}")


if __name__ == "__main__":
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "timesheets.json")
    )
    with open(data_path) as f:
        records = json.load(f)

    flags = detect(records)

    counts = {"high": 0, "medium": 0, "low": 0}
    for flag in flags:
        counts[flag["severity"]] += 1

    print(
        f"Scanned {len(records)} records, found {len(flags)} flags "
        f"({counts['high']} high, {counts['medium']} medium, {counts['low']} low)"
    )
    print()
    print("First 5 flags:")
    for flag in flags[:5]:
        print(json.dumps(flag, indent=2))
    print()
    _precision_recall(records, flags)
