# ============================================================
# MODULE: generate_synthetic_data.py
# PURPOSE: Generate synthetic timesheet records calibrated to the seeded
#          MongoDB baselines, with labeled anomalies planted for evaluation
#          of the detection pipeline.
# PIPELINE STAGE: Development utility (not part of production pipeline)
# INPUTS: Baseline documents from MongoDB baselines collection; random seed 42
# OUTPUTS: data/synthetic/timesheets.json — JSON array of canonical records
# ============================================================

"""
Simulates 200 working days of construction timesheet activity for 12 workers
and writes the result to data/synthetic/timesheets.json.

The simulation reads all hour distributions from MongoDB baselines (never
hardcodes values), then post-processes the clean dataset to plant four types
of labeled anomalies (~5% each): missing_value, numerical_outlier,
equipment_standby, and categorical_anomaly. The labels make this file the
ground truth for evaluating the detection pipeline.

Requires seed_mongodb.py to have been run first.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import uuid
from datetime import date, datetime, timedelta

import numpy as np

from database.mongo_client import baselines as baselines_col

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ID  = "31082-00"          # Westmorland Elementary School
SIM_START   = date(2025, 1, 6)
NUM_DAYS    = 200
SHIFT_START = "06:30"             # project-wide default per HCSS card

# Only these cost codes have equipment in their baselines — standby anomalies
# require equipment, so this gates which records are eligible for that mutation
STANDBY_ELIGIBLE_CODES = {"51214.4", "51234.5", "51055.6"}

ALL_LABOR_CLASSES = ["SLSDL-FM", "SLLS-L1", "SLLS-L2", "SLLS-A1", "SLLS-04"]

# Exactly 12 workers with a fixed class distribution matching the real project
LABOR_CLASS_DIST = [
    ("SLSDL-FM", 1),
    ("SLLS-L1",  4),
    ("SLLS-L2",  3),
    ("SLLS-A1",  2),
    ("SLLS-04",  2),
]

_FIRST = ["James", "Maria", "David", "Sarah", "Michael", "Jennifer",
          "Robert", "Linda", "William", "Barbara", "Richard", "Susan"]
_LAST  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
          "Miller", "Davis", "Martinez", "Hernandez", "Lopez", "Gonzalez"]

# Which worker fields a missing_value anomaly can null out
_MISSING_TARGETS = ["shift_start", "shift_end", "labor_class", "straight_time_hours"]


# ── Setup helpers ─────────────────────────────────────────────────────────────

def load_baselines() -> dict:
    """Load baseline docs from MongoDB keyed by cost_code. Exits if not seeded."""
    if baselines_col.count_documents({}) < 5:
        print("Baselines not seeded. Run scripts/seed_mongodb.py first.")
        sys.exit(1)
    docs = list(baselines_col.find({}, {"_id": 0}))
    return {doc["cost_code"]: doc for doc in docs}


def build_worker_pool() -> list[dict]:
    """Build the fixed 12-worker pool with stable IDs, names, and labor classes.

    The same workers appear on every simulated day so downstream per-worker
    feature engineering is possible.
    """
    workers = []
    idx = 0
    for labor_class, count in LABOR_CLASS_DIST:
        for _ in range(count):
            workers.append({
                "worker_id":   str(900000 + idx),
                "worker_name": f"{_FIRST[idx]} {_LAST[idx]}",
                "labor_class": labor_class,
            })
            idx += 1
    return workers


def build_equipment_pool(baselines: dict) -> dict:
    """Return a stable equipment_name → equipment_id mapping (e.g. "SKYTRAK 10054" → "UAT1002").

    Built once at startup so the same machine has the same ID on every day.
    """
    pool: dict[str, str] = {}
    eq_idx = 1000
    for baseline in baselines.values():
        for eq_name in baseline.get("common_equipment", []):
            if eq_name not in pool:
                pool[eq_name] = f"UAT{eq_idx}"
                eq_idx += 1
    return pool


def working_days(start: date, count: int) -> list[date]:
    """Return `count` calendar days starting from `start`, skipping Sundays."""
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() != 6:   # 6 = Sunday
            days.append(current)
        current += timedelta(days=1)
    return days


# ── Per-day generation ────────────────────────────────────────────────────────

def _assign_cost_codes(worker: dict, baselines: dict) -> list[str]:
    """Assign 1 or 2 cost codes to a worker, weighted by their labor class eligibility.

    A worker is only eligible for codes where their labor_class appears in
    typical_labor_classes. From that eligible set: 70% chance of 1 code, 30% of 2.
    """
    eligible = [
        code for code, bl in baselines.items()
        if worker["labor_class"] in bl.get("typical_labor_classes", [])
    ]
    if not eligible:
        return []
    if len(eligible) < 2 or random.random() < 0.70:
        return [random.choice(eligible)]
    return random.sample(eligible, 2)


def _sample_hours(baseline: dict) -> tuple[float, float]:
    """Sample ST and OT independently from per-baseline Normal distributions.

    Parameters come directly from the baseline document so each cost code
    has its own realistic OT level (e.g. foreman ~0.5 OT, field labor ~2.5 OT).
    ST ~ Normal(avg_straight_time_hours, 0.25), clipped [0, 8].
    OT ~ Normal(avg_overtime_hours, std_dev * 0.4), clipped [0, 6].
    """
    st = float(np.clip(
        np.random.normal(baseline["avg_straight_time_hours"], 0.25),
        0.0, 8.0,
    ))
    ot = float(np.clip(
        np.random.normal(baseline["avg_overtime_hours"], baseline["std_dev"] * 0.4),
        0.0, 6.0,
    ))
    return round(st, 2), round(ot, 2)


def _compute_shift_end(total_hours: float, start: str = SHIFT_START) -> str:
    """Compute shift_end = shift_start + total_hours + 0.5 hr unpaid lunch."""
    end = datetime.strptime(start, "%H:%M") + timedelta(hours=total_hours + 0.5)
    return end.strftime("%H:%M")


def _build_worker_records(
    worker: dict, cost_codes: list[str], baselines: dict, day: date,
) -> list[dict]:
    """Generate 1–2 worker records for one shift, with shared shift_start/shift_end.

    For 2-code days, resamples up to 5 times to keep the combined daily total
    in [8.0, 13.0]. Both records always get the same shift times, computed from
    the sum of hours across both cost codes.
    """
    date_str = day.isoformat()
    pairs: list[tuple[str, float, float]] = []  # (cost_code, st, ot)

    for code in cost_codes:
        st, ot = _sample_hours(baselines[code])
        pairs.append((code, st, ot))

    if len(pairs) == 2:
        for _ in range(4):  # already have 1 attempt; try 4 more = 5 total
            if 8.0 <= sum(st + ot for _, st, ot in pairs) <= 13.0:
                break
            pairs = [(code, *_sample_hours(baselines[code])) for code in cost_codes]

    day_total = sum(st + ot for _, st, ot in pairs)
    shift_end = _compute_shift_end(day_total)

    return [
        {
            "record_id": str(uuid.uuid4()),
            "date": date_str,
            "project_id": PROJECT_ID,
            "worker_id": worker["worker_id"],
            "worker_name": worker["worker_name"],
            "labor_class": worker["labor_class"],
            "cost_code": code,
            "straight_time_hours": st,
            "overtime_hours": ot,
            "total_hours": round(st + ot, 2),
            "shift_start": SHIFT_START,
            "shift_end": shift_end,
            "entity_type": "worker",
            "anomaly_type": None,
            "anomaly_detail": None,
        }
        for code, st, ot in pairs
    ]


def _find_operator(
    eq_name: str,
    baseline: dict,
    worker_pool: list[dict],
    day_ids_for_code: set[str],
) -> tuple[str | None, str | None]:
    """Find which worker (if any) should be linked as the operator for this equipment.

    Looks up the expected operator class from equipment_operator_pairs, then
    checks whether a worker with that class is on today's roster for this code.
    Returns (None, None) if no match — equipment ran standalone.
    """
    op_class: str | None = None
    for pair in baseline.get("equipment_operator_pairs", []):
        if pair["equipment_type"] == eq_name:
            op_class = pair["typical_operator_class"]
            break
    if op_class is None:
        return None, None

    candidates = [
        w for w in worker_pool
        if w["labor_class"] == op_class and w["worker_id"] in day_ids_for_code
    ]
    if not candidates:
        return None, None
    return random.choice(candidates)["worker_id"], op_class


def _build_equipment_records(
    cost_code: str,
    baseline: dict,
    day: date,
    equipment_pool: dict,
    worker_pool: list[dict],
    day_ids_for_code: set[str],
) -> list[dict]:
    """Generate equipment records for one cost code on one day.

    Each machine in common_equipment appears with 70% probability.
    Codes with an empty common_equipment list (e.g. admin, electrical) produce
    no equipment records at all.
    Hours ~ Normal(avg_equipment_hours, 1.0), clipped to [1.0, 10.0].
    """
    date_str = day.isoformat()
    records: list[dict] = []

    for eq_name in baseline.get("common_equipment", []):
        if random.random() > 0.7:
            continue
        hours = round(float(np.clip(
            np.random.normal(baseline["avg_equipment_hours"], 1.0),
            1.0, 10.0,
        )), 2)
        op_id, op_class = _find_operator(eq_name, baseline, worker_pool, day_ids_for_code)
        records.append({
            "record_id": str(uuid.uuid4()),
            "date": date_str,
            "project_id": PROJECT_ID,
            "equipment_id": equipment_pool[eq_name],
            "equipment_name": eq_name,
            "cost_code": cost_code,
            "hours": hours,
            "operator_worker_id": op_id,
            "operator_labor_class": op_class,
            "entity_type": "equipment",
            "anomaly_type": None,
            "anomaly_detail": None,
        })
    return records


def generate_clean_records(
    baselines: dict,
    worker_pool: list[dict],
    equipment_pool: dict,
    days: list[date],
) -> list[dict]:
    """Run the full simulation and return all clean (anomaly-free) records.

    For each day: assigns cost codes to each worker, generates their hours,
    then generates equipment records for every cost code that had worker activity.
    """
    all_records: list[dict] = []

    for day in days:
        day_ids: dict[str, set[str]] = {}  # cost_code → set of worker_ids active today
        worker_records: list[dict] = []

        for worker in worker_pool:
            codes = _assign_cost_codes(worker, baselines)
            if not codes:
                continue
            recs = _build_worker_records(worker, codes, baselines, day)
            worker_records.extend(recs)
            for r in recs:
                day_ids.setdefault(r["cost_code"], set()).add(worker["worker_id"])

        all_records.extend(worker_records)

        for code, worker_ids in day_ids.items():
            eq_recs = _build_equipment_records(
                code, baselines[code], day, equipment_pool, worker_pool, worker_ids
            )
            all_records.extend(eq_recs)

    return all_records


# ── Anomaly mutators ──────────────────────────────────────────────────────────
# Each function takes a clean record and returns a mutated copy tagged with
# anomaly_type and anomaly_detail. No function modifies records in place.

def _plant_missing_value(record: dict) -> dict:
    """Null out one randomly chosen required field and tag the record."""
    field = random.choice(_MISSING_TARGETS)
    r = dict(record)
    r[field] = None
    r["anomaly_type"]   = "missing_value"
    r["anomaly_detail"] = {"field": field}
    return r


def _plant_numerical_outlier(record: dict, baselines: dict) -> dict:
    """Shift total_hours by ±3.5 std_devs, scaling ST/OT proportionally.

    Direction is chosen 50/50. ST is capped at 8.0 after scaling; any
    remaining hours go to OT. shift_end is recomputed from the new total.
    """
    bl = baselines.get(record["cost_code"], {})
    std_dev = bl.get("std_dev", 1.0)
    mean    = bl.get("avg_hours_per_worker", 10.5)

    direction = 1 if random.random() < 0.5 else -1
    original  = record["total_hours"]
    new_total = round(max(0.0, original + direction * 3.5 * std_dev), 2)

    ratio  = (new_total / original) if original > 0 else 0.0
    new_st = round(min(record["straight_time_hours"] * ratio, 8.0), 2)
    new_ot = round(max(0.0, new_total - new_st), 2)

    r = dict(record)
    r["straight_time_hours"] = new_st
    r["overtime_hours"]      = new_ot
    r["total_hours"]         = new_total
    r["shift_end"]           = _compute_shift_end(new_total, r.get("shift_start") or SHIFT_START)
    r["anomaly_type"]        = "numerical_outlier"
    r["anomaly_detail"]      = {"z_score": direction * 3.5, "baseline_mean": mean, "baseline_std": std_dev}
    return r


def _plant_equipment_standby(
    record: dict,
    available_eq: list[dict],
    equipment_pool: dict,
    baselines: dict,
) -> tuple[dict, dict]:
    """Create a standby anomaly pair: worker with ~4 hrs, equipment with 8+ hrs.

    Reuses an existing equipment record for the same (cost_code, date) if one
    exists; otherwise creates a new one. Both records get the same pair_id so
    the agent can find and reason about them together.
    Only called for STANDBY_ELIGIBLE_CODES (codes with equipment in their baseline).
    """
    pair_id  = str(uuid.uuid4())
    code     = record["cost_code"]
    date_str = record["date"]
    bl       = baselines[code]

    st = round(float(np.clip(np.random.normal(4.0, 0.5), 2.0, 8.0)), 2)
    worker = dict(record)
    worker["straight_time_hours"] = st
    worker["overtime_hours"]      = 0.0
    worker["total_hours"]         = st
    worker["shift_end"]           = _compute_shift_end(st, worker.get("shift_start") or SHIFT_START)

    eq_hours = round(float(np.clip(np.random.normal(9.0, 0.5), 8.0, 10.0)), 2)

    existing = next(
        (r for r in available_eq if r["cost_code"] == code and r["date"] == date_str),
        None,
    )
    if existing:
        eq_rec = dict(existing)
        eq_rec["hours"] = eq_hours
    else:
        eq_names = bl.get("common_equipment", [])
        eq_name  = eq_names[0] if eq_names else "UNKNOWN"
        eq_rec = {
            "record_id":            str(uuid.uuid4()),
            "date":                 date_str,
            "project_id":           PROJECT_ID,
            "equipment_id":         equipment_pool.get(eq_name, "UAT0000"),
            "equipment_name":       eq_name,
            "cost_code":            code,
            "hours":                eq_hours,
            "operator_worker_id":   worker["worker_id"],
            "operator_labor_class": worker["labor_class"],
            "entity_type":          "equipment",
            "anomaly_type":         None,
            "anomaly_detail":       None,
        }

    detail = {"pair_id": pair_id, "operator_hours": st, "equipment_hours": eq_hours}
    worker["anomaly_type"]   = "equipment_standby"
    worker["anomaly_detail"] = detail
    eq_rec["anomaly_type"]   = "equipment_standby"
    eq_rec["anomaly_detail"] = detail
    return worker, eq_rec


def _plant_categorical_anomaly(record: dict, baselines: dict) -> dict:
    """Swap the worker's labor_class to one not allowed on this cost code."""
    bl      = baselines.get(record["cost_code"], {})
    allowed = set(bl.get("typical_labor_classes", []))
    wrong   = [c for c in ALL_LABOR_CLASSES if c not in allowed] or ["SLLS-L1"]

    original  = record["labor_class"]
    new_class = random.choice(wrong)
    r = dict(record)
    r["labor_class"]    = new_class
    r["anomaly_type"]   = "categorical_anomaly"
    r["anomaly_detail"] = {
        "actual_class":    new_class,
        "original_class":  original,
        "allowed_classes": list(allowed),
    }
    return r


# ── Anomaly orchestration ─────────────────────────────────────────────────────

def _build_pools(
    worker_records: list[dict], n_each: int
) -> tuple[list, list, list, list]:
    """Divide shuffled worker records into four non-overlapping anomaly pools.

    Each pool gets n_each records (~5% of the total). Standby pool is drawn
    only from STANDBY_ELIGIBLE_CODES after the first two pools are reserved.
    No record appears in more than one pool.
    """
    shuffled = list(worker_records)
    random.shuffle(shuffled)

    pool_mv = shuffled[:n_each]
    pool_no = shuffled[n_each:2 * n_each]
    used    = {r["record_id"] for r in pool_mv + pool_no}

    eligible_sb = [r for r in shuffled if r["cost_code"] in STANDBY_ELIGIBLE_CODES
                   and r["record_id"] not in used]
    pool_sb = eligible_sb[:n_each]
    used   |= {r["record_id"] for r in pool_sb}

    pool_cat = [r for r in shuffled if r["record_id"] not in used][:n_each]
    return pool_mv, pool_no, pool_sb, pool_cat


def plant_anomalies(
    clean_records: list[dict],
    baselines: dict,
    equipment_pool: dict,
) -> list[dict]:
    """Post-process the clean dataset, mutating ~20% into four labeled anomaly types.

    Generates four non-overlapping pools (~5% each) from worker records, applies
    one mutation per pool, then rebuilds the full record list with mutations
    substituted in place. Newly created standby equipment records are appended.
    """
    worker_records = [r for r in clean_records if r["entity_type"] == "worker"]
    eq_records     = [r for r in clean_records if r["entity_type"] == "equipment"]

    n_each = max(1, int(len(worker_records) * 0.05))
    pool_mv, pool_no, pool_sb, pool_cat = _build_pools(worker_records, n_each)

    mutated: dict[str, dict] = {}
    extra_eq: list[dict]     = []
    used_eq_ids: set[str]    = set()

    for r in pool_mv:
        mutated[r["record_id"]] = _plant_missing_value(r)

    for r in pool_no:
        mutated[r["record_id"]] = _plant_numerical_outlier(r, baselines)

    for r in pool_sb:
        avail = [e for e in eq_records if e["record_id"] not in used_eq_ids]
        mut_worker, eq_rec = _plant_equipment_standby(r, avail, equipment_pool, baselines)
        mutated[r["record_id"]] = mut_worker
        if eq_rec["record_id"] in {e["record_id"] for e in eq_records}:
            mutated[eq_rec["record_id"]] = eq_rec
            used_eq_ids.add(eq_rec["record_id"])
        else:
            extra_eq.append(eq_rec)

    for r in pool_cat:
        mutated[r["record_id"]] = _plant_categorical_anomaly(r, baselines)

    final = [mutated.get(r["record_id"], r) for r in clean_records]
    final.extend(extra_eq)
    return final


# ── Summary and entry point ───────────────────────────────────────────────────

def print_summary(records: list[dict]) -> None:
    """Print a breakdown of total records, entity types, and anomaly counts."""
    n_worker = sum(1 for r in records if r["entity_type"] == "worker")
    n_eq     = sum(1 for r in records if r["entity_type"] == "equipment")
    counts: dict[str, int] = {}
    for r in records:
        key = r.get("anomaly_type") or "clean"
        counts[key] = counts.get(key, 0) + 1

    print(f"Total records:        {len(records)}")
    print(f"Worker records:       {n_worker}")
    print(f"Equipment records:    {n_eq}")
    print(f"Clean:                {counts.get('clean', 0)}")
    print(f"missing_value:        {counts.get('missing_value', 0)}")
    print(f"numerical_outlier:    {counts.get('numerical_outlier', 0)}")
    print(f"equipment_standby:    {counts.get('equipment_standby', 0)}")
    print(f"categorical_anomaly:  {counts.get('categorical_anomaly', 0)}")


def main() -> None:
    baselines      = load_baselines()
    worker_pool    = build_worker_pool()
    equipment_pool = build_equipment_pool(baselines)
    days           = working_days(SIM_START, NUM_DAYS)

    print(f"Generating records for {len(days)} working days, {len(worker_pool)} workers...")
    clean   = generate_clean_records(baselines, worker_pool, equipment_pool, days)
    print(f"Clean records generated: {len(clean)}")

    records = plant_anomalies(clean, baselines, equipment_pool)

    out_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "timesheets.json")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} records to {out_path}\n")
    print_summary(records)


if __name__ == "__main__":
    main()
