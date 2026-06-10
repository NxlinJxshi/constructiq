# ============================================================
# MODULE: categorical_anomalies.py
# PURPOSE: Rule-based detection of mismatched labor classes and equipment types.
#          Compares each record's categorical fields against the expected values
#          stored in MongoDB baselines for that cost code.
# PIPELINE STAGE: Detection (step 2b — runs after missing_values)
# INPUTS: List of canonical record dicts + baselines collection (MongoDB)
# OUTPUTS: List of flag dicts with flag_type ∈ {labor_class_anomaly,
#          equipment_on_wrong_code, wrong_operator_for_equipment}
# EXAMPLE: Worker with SLLS-04 labor class on cost code 51300.1 (which expects
#          only SLSDL-FM) → flag with severity "medium" and explanation.
# ============================================================

"""Rule-based detection of labor class and equipment type mismatches per cost code."""

from __future__ import annotations
import json
import os
import sys
import uuid

from database.mongo_client import baselines as baselines_col

# Process-level cache: baselines are fetched from MongoDB once and reused for
# every subsequent audit in the same process, eliminating repeated network calls.
_baseline_cache: list[dict] | None = None


def _load_baselines() -> dict:
    """Return a cost_code → baseline dict, fetching from MongoDB only on first call."""
    global _baseline_cache
    if _baseline_cache is None:
        _baseline_cache = list(baselines_col.find({}, {"_id": 0}))
    return {doc["cost_code"]: doc for doc in _baseline_cache}


def detect(records: list[dict]) -> list[dict]:
    """Flag records whose labor class or equipment type is unexpected for their cost code.

    Checks three rules against each cost code's baseline document:
      1. Worker's labor_class not in baseline["typical_labor_classes"] → labor_class_anomaly
      2. Equipment's equipment_name not in baseline["common_equipment"] → equipment_on_wrong_code
      3. Equipment's operator_labor_class mismatches baseline["equipment_operator_pairs"] → wrong_operator_for_equipment
    Returns an empty list if no violations are found.
    """
    baseline_map = _load_baselines()

    flags: list[dict] = []

    for record in records:
        code = record.get("cost_code")
        if code not in baseline_map:
            continue

        bl = baseline_map[code]
        entity = record.get("entity_type")

        if entity == "worker":
            lc = record.get("labor_class")
            if lc is None:
                continue  # missing_values detector handles this separately

            allowed = bl.get("typical_labor_classes", [])
            if lc not in allowed:
                flags.append({
                    "flag_id":     str(uuid.uuid4()),
                    "record_id":   record["record_id"],
                    "flag_type":   "labor_class_anomaly",
                    "severity":    "medium",
                    "entity_type": "worker",
                    "field":       "labor_class",
                    "observed":    lc,
                    "explanation": (
                        f"Worker {record.get('worker_name')} (ID: {record.get('worker_id')}) "
                        f"on {record.get('date')} is billed to cost code {code} "
                        f"under labor class {lc}, but the expected labor classes for this cost code "
                        f"are {allowed}. This is a MEDIUM severity issue because billing under an "
                        f"unexpected classification can cause cost code mis-allocation."
                    ),
                    "details": {"actual": lc, "expected": allowed},
                })

        elif entity == "equipment":
            eq_name = record.get("equipment_name")
            if eq_name is None:
                continue

            common = bl.get("common_equipment", [])

            # Rule 2: equipment type not expected for this cost code.
            if eq_name not in common:
                flags.append({
                    "flag_id":     str(uuid.uuid4()),
                    "record_id":   record["record_id"],
                    "flag_type":   "equipment_on_wrong_code",
                    "severity":    "high",
                    "entity_type": "equipment",
                    "field":       "equipment_name",
                    "observed":    eq_name,
                    "explanation": (
                        f"Equipment {eq_name} (ID: {record.get('equipment_id')}) on "
                        f"{record.get('date')} is logged against cost code {code}, but the expected "
                        f"equipment for this code is {common}. This is a HIGH severity issue because "
                        f"equipment not normally used on this activity may indicate billing error "
                        f"or asset misuse."
                    ),
                    "details": {"actual": eq_name, "expected": common},
                })

            # Rule 3: operator labor class doesn't match the equipment-operator pairing.
            op_class = record.get("operator_labor_class")
            if op_class is None:
                continue

            for pair in bl.get("equipment_operator_pairs", []):
                if pair["equipment_type"] == eq_name:
                    expected_op = pair["typical_operator_class"]
                    if op_class != expected_op:
                        flags.append({
                            "flag_id":     str(uuid.uuid4()),
                            "record_id":   record["record_id"],
                            "flag_type":   "wrong_operator_for_equipment",
                            "severity":    "medium",
                            "entity_type": "equipment",
                            "field":       "operator_labor_class",
                            "observed":    op_class,
                            "explanation": (
                                f"Equipment {eq_name} (ID: {record.get('equipment_id')}) on "
                                f"{record.get('date')} for cost code {code} is operated by labor "
                                f"class {op_class}, but the expected operator class is {expected_op}. "
                                f"This is a MEDIUM severity issue because incorrect operator "
                                f"classification can indicate a compliance or safety risk."
                            ),
                            "details": {"actual": op_class, "expected": expected_op},
                        })
                    break

    return flags


def _precision_recall(records: list[dict], flags: list[dict]) -> None:
    """Compare emitted flags against planted categorical_anomaly labels in synthetic data."""
    planted    = {r["record_id"] for r in records if r.get("anomaly_type") == "categorical_anomaly"}
    clean_ids  = {r["record_id"] for r in records if r.get("anomaly_type") is None}
    flagged    = {f["record_id"] for f in flags}
    tp = len(planted & flagged)
    fp = len(clean_ids & flagged)
    total = len(planted)
    pct = (tp / total * 100) if total else 0.0
    print(f"Recall on planted categorical_anomaly records: {tp}/{total} ({pct:.1f}%)")
    print(f"False positives on clean records: {fp}")


if __name__ == "__main__":
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "timesheets.json")
    )
    with open(data_path) as f:
        records = json.load(f)

    flags = detect(records)

    by_type: dict[str, int] = {}
    for f in flags:
        by_type[f["flag_type"]] = by_type.get(f["flag_type"], 0) + 1

    print(f"Scanned {len(records)} records, found {len(flags)} categorical flags")
    for ft, count in sorted(by_type.items()):
        print(f"  {ft}: {count}")
    print()
    _precision_recall(records, flags)
