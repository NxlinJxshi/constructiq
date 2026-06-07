# ============================================================
# MODULE: equipment_standby.py
# PURPOSE: Detects equipment billed as operating while the operator hours on the
#          same cost code and day are suspiciously low — a common billing fraud
#          pattern in HCSS where idle equipment is left running on the clock.
# PIPELINE STAGE: Detection (step 2c)
# INPUTS: List of canonical record dicts (both worker and equipment entity types)
# OUTPUTS: List of flag dicts with flag_type="equipment_standby", one per equipment
#          record where eq_hours > operator_hours + 1.0 (per-operator or group-level)
# EXAMPLE: Equipment logs 9 hrs; its assigned operator logged only 4 hrs that day
#          → equipment record flagged HIGH with delta and both hourly values in details.
# ============================================================

"""Flags equipment that ran significantly longer than the operators working that shift."""

from __future__ import annotations
import json
import os
import sys
import uuid
from collections import defaultdict


def detect(records: list[dict]) -> list[dict]:
    """Flag equipment records where equipment hours > operator hours + 1.0.

    Uses a two-tier check:
    1. Per-operator (primary): if the equipment record has operator_worker_id,
       find that specific worker and compare hours directly. This is the most
       meaningful check in HCSS, where each piece of equipment is assigned to
       one operator per shift.
    2. Group-level fallback: for equipment without operator_worker_id, compare
       the equipment's hours to the average worker hours on the same (cost_code, date).
    The 1.0-hour grace window absorbs schedule slippage without generating noise.
    """
    GRACE = 1.0

    # ── Pre-index all records ─────────────────────────────────────────────────
    # worker_index: (cost_code, date, worker_id) → total_hours
    # group_workers: (cost_code, date) → list of total_hours values for avg fallback
    worker_index:  dict[tuple, float] = {}
    group_workers: dict[tuple, list]  = defaultdict(list)

    for r in records:
        if r.get("entity_type") != "worker":
            continue
        h = r.get("total_hours") or 0.0
        worker_index[(r.get("cost_code"), r.get("date"), r.get("worker_id"))] = float(h)
        group_workers[(r.get("cost_code"), r.get("date"))].append(float(h))

    flags: list[dict] = []
    already_flagged: set[str] = set()

    for r in records:
        if r.get("entity_type") != "equipment":
            continue

        code     = r.get("cost_code")
        date     = r.get("date")
        eq_hours = float(r.get("hours") or 0.0)
        eq_rid   = r["record_id"]

        if eq_rid in already_flagged:
            continue

        op_id     = r.get("operator_worker_id")
        op_hours  = None
        check_key = None

        if op_id:
            # Primary: compare this equipment to its assigned operator.
            op_hours  = worker_index.get((code, date, op_id))
            check_key = "per_operator"

        if op_hours is None:
            # Fallback: compare to average worker hours for the (cost_code, date) group.
            group_h = group_workers.get((code, date), [])
            if not group_h:
                continue
            op_hours  = sum(group_h) / len(group_h)
            check_key = "group_average"

        if eq_hours <= op_hours + GRACE:
            continue

        delta = round(eq_hours - op_hours, 2)

        if check_key == "per_operator":
            explanation = (
                f"Equipment {r.get('equipment_name')} (ID: {r.get('equipment_id')}) on "
                f"{date} for cost code {code} logged {eq_hours} hours, but its assigned "
                f"operator (ID: {op_id}) only logged {round(op_hours, 2)} hours on the same "
                f"cost code and shift. Equipment hours exceed operator hours by {delta} hours. "
                f"This is a HIGH severity issue because equipment running without its assigned "
                f"operator likely indicates idle billing or unauthorized standby time."
            )
        else:
            explanation = (
                f"Equipment {r.get('equipment_name')} (ID: {r.get('equipment_id')}) on "
                f"{date} for cost code {code} logged {eq_hours} hours, but the average worker "
                f"hours on this cost code that day was only {round(op_hours, 2)} hours. "
                f"Equipment hours exceed the group average by {delta} hours. This is a HIGH "
                f"severity issue because equipment substantially out-running the crew suggests "
                f"it was left running without productive output."
            )

        flags.append({
            "flag_id":     str(uuid.uuid4()),
            "record_id":   eq_rid,
            "flag_type":   "equipment_standby",
            "severity":    "high",
            "entity_type": "equipment",
            "field":       "hours",
            "observed":    eq_hours,
            "explanation": explanation,
            "details": {
                "equipment_hours":      eq_hours,
                "operator_hours":       round(op_hours, 2),
                "delta_hours":          delta,
                "check_method":         check_key,
                "grace_window_hours":   GRACE,
            },
        })
        already_flagged.add(eq_rid)

    return flags


def _precision_recall(records: list[dict], flags: list[dict]) -> None:
    """Compare emitted flags against planted equipment_standby labels in synthetic data."""
    planted   = {r["record_id"] for r in records if r.get("anomaly_type") == "equipment_standby"}
    clean_ids = {r["record_id"] for r in records if r.get("anomaly_type") is None}
    flagged   = {f["record_id"] for f in flags}
    tp = len(planted & flagged)
    fp = len(clean_ids & flagged)
    total = len(planted)
    pct = (tp / total * 100) if total else 0.0
    print(f"Recall on planted equipment_standby records: {tp}/{total} ({pct:.1f}%)")
    print(f"False positives on clean records: {fp}")


if __name__ == "__main__":
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "timesheets.json")
    )
    with open(data_path) as f:
        records = json.load(f)

    flags = detect(records)

    h = sum(1 for f in flags if f["severity"] == "high")
    m = sum(1 for f in flags if f["severity"] == "medium")
    print(f"Scanned {len(records)} records, found {len(flags)} equipment_standby flags ({h} high, {m} medium)")
    print()
    _precision_recall(records, flags)
