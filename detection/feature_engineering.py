# ============================================================
# MODULE: feature_engineering.py
# PURPOSE: Transforms canonical timesheet records into a 10-feature numeric
#          vector for the Isolation Forest anomaly detector.
# PIPELINE STAGE: Detection (feature prep, runs before train_model and numerical_outliers)
# INPUTS: List of canonical record dicts from ingestion/normalizer.py
# OUTPUTS: np.ndarray of shape (N, 10) and a parallel list of N record_ids
# EXAMPLE: build_worker_features(records) → X shape (3088, 10), record_ids list[str]
# NOTE: Encoders are built deterministically from sorted MongoDB baselines at
#       runtime so no encoder artifact needs to be serialized alongside model.joblib.
# ============================================================

"""Builds the 10-feature numeric matrix used to train and score the Isolation Forest."""

from __future__ import annotations
import numpy as np
from database.mongo_client import baselines as baselines_col

# Process-level cache for baseline documents.  build_encoders() is called at
# every audit; fetching from MongoDB each time adds unnecessary latency.
_encoder_cache: dict | None = None

# Canonical ordered feature names — must stay in sync with the training script.
FEATURE_NAMES = [
    "labor_class_id",
    "cost_code_id",
    "straight_time_hours",
    "overtime_hours",
    "total_hours",
    "overtime_ratio",
    "shift_duration_hours",
    "hours_to_shift_ratio",
    "equipment_hours_same_code",
    "equipment_to_operator_ratio",
]

# Fields that must be non-None for a worker record to be scored.
_REQUIRED = [
    "straight_time_hours", "overtime_hours", "total_hours",
    "shift_start", "shift_end", "labor_class", "cost_code",
]


class AnomalyScorer:
    """Wraps IsolationForest so that predict() returns score_samples() for the Vertex AI container.

    The sklearn pre-built container calls predict() on whatever object is in model.joblib.
    Returning score_samples() gives continuous anomaly scores instead of binary -1/1,
    letting numerical_outliers.py apply a meaningful threshold.
    """

    def __init__(self, forest):
        self.forest = forest

    def predict(self, X):
        return self.forest.score_samples(np.array(X)).tolist()

    def fit(self, X, y=None):
        return self


def _parse_hhmm(hhmm: str) -> float:
    """Convert "HH:MM" to float hour-of-day. "06:30" → 6.5, "17:30" → 17.5.

    Raises ValueError on malformed input (not exactly two colon-separated parts).
    """
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse time string: {hhmm!r}")
    return int(parts[0]) + int(parts[1]) / 60.0


def build_encoders() -> dict:
    """Build deterministic integer encodings for cost codes and labor classes.

    Reads all baseline documents from MongoDB on the first call, then returns
    the cached result on every subsequent call in the same process.  The mapping
    is always derived from sorted sets so it is identical across train and inference.

    Returns {"cost_code_to_id": {...}, "labor_class_to_id": {...}}.
    """
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    docs = list(baselines_col.find({}, {"_id": 0}))

    # Collect all cost codes and labor classes across all baseline documents.
    cost_codes: set[str] = set()
    all_labor_classes: set[str] = set()
    for doc in docs:
        cost_codes.add(doc["cost_code"])
        for lc in doc.get("typical_labor_classes", []):
            all_labor_classes.add(lc)

    _encoder_cache = {
        "cost_code_to_id":   {code: i for i, code in enumerate(sorted(cost_codes))},
        "labor_class_to_id": {lc: i for i, lc in enumerate(sorted(all_labor_classes))},
    }
    return _encoder_cache


def build_worker_features(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build the (N, 10) feature matrix for all scoreable worker records.

    Skips equipment records and any worker records with missing required fields.
    Feature[8] (equipment_hours_same_code) aggregates hours from equipment records
    in the same (cost_code, date) group whose operator_worker_id matches this worker.
    Returns a zero-row array (shape (0, 10)) and an empty list if no records qualify.
    """
    encoders = build_encoders()
    cc_map = encoders["cost_code_to_id"]
    lc_map = encoders["labor_class_to_id"]

    # Pre-index equipment hours: (cost_code, date, operator_worker_id) → total hours.
    # This lets feature[8] be computed in O(1) per worker instead of O(N).
    eq_index: dict[tuple, float] = {}
    for r in records:
        if r.get("entity_type") != "equipment":
            continue
        op = r.get("operator_worker_id")
        if op is None:
            continue
        key = (r["cost_code"], r["date"], op)
        eq_index[key] = eq_index.get(key, 0.0) + float(r.get("hours", 0.0))

    rows: list[list[float]] = []
    record_ids: list[str] = []

    for r in records:
        if r.get("entity_type") != "worker":
            continue
        if any(r.get(f) is None for f in _REQUIRED):
            continue

        try:
            lc_id = lc_map.get(r["labor_class"], -1)
            cc_id = cc_map.get(r["cost_code"], -1)
            st    = float(r["straight_time_hours"])
            ot    = float(r["overtime_hours"])
            total = float(r["total_hours"])

            ot_ratio = ot / total if total > 0 else 0.0

            start_h = _parse_hhmm(r["shift_start"])
            end_h   = _parse_hhmm(r["shift_end"])
            if end_h < start_h:
                end_h += 24.0  # overnight shift — shift_end crosses midnight
            shift_dur = end_h - start_h

            hours_ratio = total / shift_dur if shift_dur > 0 else 1.0

            eq_key   = (r["cost_code"], r["date"], r["worker_id"])
            eq_hours = eq_index.get(eq_key, 0.0)
            eq_ratio = eq_hours / total if total > 0 else 0.0

            rows.append([lc_id, cc_id, st, ot, total, ot_ratio,
                         shift_dur, hours_ratio, eq_hours, eq_ratio])
            record_ids.append(r["record_id"])

        except (ValueError, TypeError, KeyError):
            continue

    if not rows:
        return np.empty((0, 10)), []

    return np.array(rows, dtype=float), record_ids
