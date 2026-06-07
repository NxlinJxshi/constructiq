# ============================================================
# MODULE: train_model.py
# PURPOSE: Train an Isolation Forest anomaly detector on synthetic timesheet
#          records, evaluate recall on planted numerical_outlier anomalies, and
#          save the model artifact to artifacts/model.joblib.
# PIPELINE STAGE: Offline training — run once to produce the artifact that
#          deploy_model.py uploads to Vertex AI.
# INPUTS: data/synthetic/timesheets.json (from scripts/generate_synthetic_data.py)
# OUTPUTS: artifacts/model.joblib containing AnomalyScorer + threshold metadata
# ============================================================

"""Trains and evaluates the Isolation Forest model for numerical outlier detection."""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import joblib
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import IsolationForest

from detection.feature_engineering import build_worker_features, FEATURE_NAMES, AnomalyScorer

DATA_PATH     = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "timesheets.json"))
ARTIFACT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.joblib"))
ARTIFACTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))


def main() -> None:
    # ── Load synthetic data ───────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("Synthetic data not found. Run scripts/generate_synthetic_data.py first.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {DATA_PATH}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    # build_worker_features skips equipment records and any worker records that
    # have missing required fields (caught separately by missing_values detector).
    X, record_ids = build_worker_features(records)
    n_records, n_features = X.shape
    print(f"Built feature matrix: {n_records} worker records × {n_features} features")

    if n_records == 0:
        print("No scoreable worker records — aborting.")
        sys.exit(1)

    # ── Train Isolation Forest ────────────────────────────────────────────────
    # contamination=0.05 tells sklearn that ~5% of training records are anomalous,
    # which sets model.offset_ (the score_samples threshold) accordingly.
    print("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    # model.offset_ is the score_samples() value below which contamination fraction
    # of training data falls. Records scoring below this value are predicted anomalies.
    threshold = float(model.offset_)

    # ── Evaluate on synthetic ground truth ────────────────────────────────────
    # Planted numerical_outlier records have hours shifted ±3.5 std devs from the
    # mean, making them the most distinguishable anomaly type in feature space.
    scores     = model.score_samples(X)
    id_to_type = {r["record_id"]: r.get("anomaly_type") for r in records}

    y_true = [id_to_type.get(rid) == "numerical_outlier" for rid in record_ids]
    y_pred = [float(s) < threshold for s in scores]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    n_clean   = sum(1 for t in y_true if not t)
    fp_rate   = fp / n_clean if n_clean > 0 else 0.0

    # ── Save artifact ─────────────────────────────────────────────────────────
    # AnomalyScorer wraps the trained forest so that predict() returns score_samples()
    # output, which the Vertex AI sklearn container will call at inference time.
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump({
        "model":                AnomalyScorer(model),
        "threshold":            threshold,
        "feature_names":        FEATURE_NAMES,
        "trained_on_n_records": n_records,
        "trained_at":           datetime.now(timezone.utc).isoformat(),
    }, ARTIFACT_PATH)

    print(f"\nTrained Isolation Forest on {n_records} worker records ({n_features} features)")
    print(f"Threshold (score_samples offset at contamination=0.05): {threshold:.4f}")
    print(f"Evaluation on planted numerical_outlier anomalies:")
    print(f"  Precision:  {precision:.2f}  ({tp} TP, {fp} FP)")
    print(f"  Recall:     {recall:.2f}  ({tp} TP, {fn} FN)")
    print(f"  FP rate on clean records: {fp_rate*100:.2f}%")
    print(f"Saved model artifact to {ARTIFACT_PATH}")

    if recall < 0.85:
        print(f"\nWARNING: Recall {recall:.2f} is below target 0.85.")
    if precision < 0.50:
        print(f"\nWARNING: Precision {precision:.2f} is below target 0.50.")


if __name__ == "__main__":
    main()
