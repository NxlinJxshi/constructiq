# ============================================================
# MODULE: test_feature_engineering.py
# PURPOSE: Unit tests for detection/feature_engineering.py covering encoder
#          determinism, feature vector shape, skip logic, cross-record equipment
#          aggregation, unknown encoding, and overnight shift handling.
# PIPELINE STAGE: Test suite (Detection — feature engineering)
# INPUTS: Hard-coded record dicts; build_encoders() reads live MongoDB baselines
# OUTPUTS: pytest pass/fail results
# ============================================================

"""
Tests for build_encoders(), build_worker_features(), and _parse_hhmm().

build_encoders() must produce identical mappings on repeated calls (no random state).
build_worker_features() must: skip equipment records, skip records with missing required
fields, compute equipment_hours from matching (cost_code, date, worker_id) tuples,
encode unknown values as -1, and handle overnight shifts correctly.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import numpy as np
import pytest

from detection.feature_engineering import build_encoders, build_worker_features, _parse_hhmm


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _worker(**overrides):
    """Return a minimal fully-valid worker record; override any field via kwargs."""
    r = {
        "record_id":           str(uuid.uuid4()),
        "entity_type":         "worker",
        "date":                "2025-01-06",
        "project_id":          "31082-00",
        "worker_id":           "900000",
        "worker_name":         "James Smith",
        "labor_class":         "SLLS-L1",
        "cost_code":           "51214.4",
        "straight_time_hours": 8.0,
        "overtime_hours":      2.5,
        "total_hours":         10.5,
        "shift_start":         "06:30",
        "shift_end":           "17:30",
        "anomaly_type":        None,
        "anomaly_detail":      None,
    }
    r.update(overrides)
    return r


def _equipment(**overrides):
    """Return a minimal valid equipment record linked to worker_id 900000."""
    r = {
        "record_id":            str(uuid.uuid4()),
        "entity_type":          "equipment",
        "date":                 "2025-01-06",
        "project_id":           "31082-00",
        "equipment_id":         "UAT1000",
        "equipment_name":       "BOBCAT S510",
        "cost_code":            "51214.4",
        "hours":                5.0,
        "operator_worker_id":   "900000",
        "operator_labor_class": "SLLS-A1",
        "anomaly_type":         None,
        "anomaly_detail":       None,
    }
    r.update(overrides)
    return r


# ── build_encoders ────────────────────────────────────────────────────────────

class TestBuildEncoders:
    def test_determinism(self):
        # Two independent calls against the same MongoDB baselines must produce
        # identical mappings — this is the core invariant that makes train/inference
        # encodings consistent without serializing any encoder artifact.
        enc1 = build_encoders()
        enc2 = build_encoders()
        assert enc1["cost_code_to_id"]   == enc2["cost_code_to_id"]
        assert enc1["labor_class_to_id"] == enc2["labor_class_to_id"]

    def test_ids_start_at_zero(self):
        enc = build_encoders()
        assert 0 in enc["cost_code_to_id"].values()
        assert 0 in enc["labor_class_to_id"].values()

    def test_known_codes_present(self):
        enc = build_encoders()
        assert "51214.4" in enc["cost_code_to_id"]
        assert "SLLS-L1" in enc["labor_class_to_id"]


# ── build_worker_features ─────────────────────────────────────────────────────

class TestBuildWorkerFeatures:
    def test_vector_shape_five_workers(self):
        records = [_worker(record_id=str(uuid.uuid4()), worker_id=str(i)) for i in range(5)]
        X, ids = build_worker_features(records)
        assert X.shape == (5, 10)
        assert len(ids) == 5

    def test_skips_equipment_records(self):
        # Equipment records are not scoreable; only worker records enter the matrix.
        records = [_worker(), _equipment()]
        X, ids = build_worker_features(records)
        assert X.shape == (1, 10)

    def test_skips_record_with_missing_labor_class(self):
        records = [_worker(), _worker(record_id=str(uuid.uuid4()), labor_class=None)]
        X, ids = build_worker_features(records)
        assert X.shape == (1, 10)

    def test_skips_record_with_missing_shift_start(self):
        records = [_worker(), _worker(record_id=str(uuid.uuid4()), shift_start=None)]
        X, ids = build_worker_features(records)
        assert X.shape == (1, 10)

    def test_equipment_hours_with_matching_record(self):
        # Equipment record linked to worker_id "900001" on the same (cost_code, date)
        # must appear in feature[8] for that worker.
        w  = _worker(worker_id="900001")
        eq = _equipment(operator_worker_id="900001", hours=6.0)
        X, ids = build_worker_features([w, eq])
        assert X[0, 8] == pytest.approx(6.0)

    def test_equipment_hours_zero_when_no_match(self):
        # No equipment records in the input → feature[8] must be 0.0.
        w = _worker()
        X, ids = build_worker_features([w])
        assert X[0, 8] == pytest.approx(0.0)

    def test_unknown_cost_code_encodes_as_minus_one(self):
        # Cost codes not in any baseline must be encoded as -1, not raise an error.
        w = _worker(cost_code="99999.9")
        X, ids = build_worker_features([w])
        assert X[0, 1] == pytest.approx(-1.0)

    def test_overnight_shift_duration(self):
        # shift_start=22:00, shift_end=06:00 → 8 hours across midnight.
        # Without the +24 correction the duration would compute as -16 hours.
        w = _worker(
            shift_start="22:00", shift_end="06:00",
            total_hours=8.0, straight_time_hours=8.0, overtime_hours=0.0,
        )
        X, ids = build_worker_features([w])
        assert X[0, 6] == pytest.approx(8.0)

    def test_empty_input_returns_empty_array(self):
        X, ids = build_worker_features([])
        assert X.shape == (0, 10)
        assert ids == []


# ── _parse_hhmm ───────────────────────────────────────────────────────────────

class TestParseHhmm:
    def test_morning_shift_start(self):
        assert _parse_hhmm("06:30") == pytest.approx(6.5)

    def test_afternoon_shift_end(self):
        assert _parse_hhmm("17:30") == pytest.approx(17.5)

    def test_midnight(self):
        assert _parse_hhmm("00:00") == pytest.approx(0.0)

    def test_malformed_three_colons_raises(self):
        # "6:30:00" has three colon-separated parts — not a valid HH:MM string.
        with pytest.raises(ValueError):
            _parse_hhmm("6:30:00")
