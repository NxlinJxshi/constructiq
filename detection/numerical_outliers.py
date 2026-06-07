# ============================================================
# MODULE: numerical_outliers.py
# PURPOSE: Calls a deployed Vertex AI endpoint to score worker records with the
#          Isolation Forest model and flags those below the anomaly threshold.
# PIPELINE STAGE: Detection (step 2d — online inference)
# INPUTS: List of canonical record dicts; VERTEX_ENDPOINT_NAME from .env;
#         artifacts/model.joblib (threshold only — scoring runs at the endpoint)
# OUTPUTS: List of flag dicts with flag_type="numerical_outlier" and severity
#          high/medium depending on how far below threshold the score falls.
# EXAMPLE: Worker with 14 hours on a cost code averaging 10.5 → score well below
#          threshold → flag with severity "high" and score_delta in details.
# ============================================================

"""Online inference: scores worker records via Vertex AI and flags numerical outliers."""

from __future__ import annotations
import json
import os
import sys
import uuid

import numpy as np
import joblib
from dotenv import load_dotenv

load_dotenv()


def _chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def detect(records: list[dict]) -> list[dict]:
    """Score all scoreable worker records via the Vertex AI endpoint and flag anomalies.

    Builds feature vectors using build_worker_features(), sends them to the deployed
    endpoint in batches of 100, and compares response scores against the threshold
    saved in artifacts/model.joblib. Records scoring below the threshold are flagged;
    those more than 0.05 below the threshold receive HIGH severity, others MEDIUM.
    Raises RuntimeError if VERTEX_ENDPOINT_NAME is not set in .env.
    """
    endpoint_name = os.environ.get("VERTEX_ENDPOINT_NAME", "").strip()
    if not endpoint_name:
        raise RuntimeError(
            "VERTEX_ENDPOINT_NAME is not set. Run scripts/deploy_model.py first."
        )

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    region  = os.environ.get("GOOGLE_CLOUD_REGION", "")

    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)

    # AnomalyScorer must be importable for joblib to reconstruct it from the artifact.
    from detection.feature_engineering import build_worker_features, AnomalyScorer  # noqa: F401

    X, record_ids = build_worker_features(records)

    if len(X) == 0:
        return []

    # Load threshold from the local artifact; the endpoint handles actual scoring.
    artifact_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.joblib")
    )
    artifact  = joblib.load(artifact_path)
    threshold = float(artifact["threshold"])

    # ── Batch inference via Vertex AI endpoint ────────────────────────────────
    # The endpoint calls AnomalyScorer.predict() which returns score_samples() values.
    # Each value is a float: more negative → further from normal → more anomalous.
    endpoint  = aiplatform.Endpoint(endpoint_name)
    instances = X.tolist()
    all_scores: list[float] = []

    for batch in _chunks(instances, 100):
        response = endpoint.predict(instances=batch)
        all_scores.extend(response.predictions)

    scores        = np.array(all_scores, dtype=float)
    records_by_id = {r["record_id"]: r for r in records}

    # ── Flag records below threshold ──────────────────────────────────────────
    flags: list[dict] = []
    for rid, score in zip(record_ids, scores):
        if score >= threshold:
            continue

        rec = records_by_id.get(rid, {})
        # Records substantially below the threshold are HIGH; others are MEDIUM.
        severity = "high" if score < threshold - 0.05 else "medium"

        flags.append({
            "flag_id":     str(uuid.uuid4()),
            "record_id":   rid,
            "flag_type":   "numerical_outlier",
            "severity":    severity,
            "entity_type": "worker",
            "field":       "total_hours",
            "observed":    rec.get("total_hours"),
            "explanation": (
                f"Worker {rec.get('worker_name')} (ID: {rec.get('worker_id')}) on "
                f"{rec.get('date')} for cost code {rec.get('cost_code')} logged total hours of "
                f"{rec.get('total_hours')} (ST={rec.get('straight_time_hours')}, "
                f"OT={rec.get('overtime_hours')}). The Isolation Forest model scored this record "
                f"at {score:.4f}, below the threshold of {threshold:.4f}. This score indicates "
                f"the worker's hours are statistically inconsistent with historical patterns "
                f"for this cost code and labor class."
            ),
            "details": {
                "anomaly_score": float(score),
                "threshold":     float(threshold),
                "score_delta":   float(threshold - score),
            },
        })

    return flags


if __name__ == "__main__":
    data_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "timesheets.json")
    )

    with open(data_path) as f:
        records = json.load(f)

    try:
        flags = detect(records)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    h = sum(1 for f in flags if f["severity"] == "high")
    m = sum(1 for f in flags if f["severity"] == "medium")

    # Count how many worker records were scoreable (passed through build_worker_features).
    from detection.feature_engineering import build_worker_features
    _, scored_ids = build_worker_features(records)
    n_scored = len(scored_ids)

    print(f"Scored {n_scored} worker records via Vertex AI endpoint")
    print(f"Emitted {len(flags)} numerical_outlier flags ({h} high, {m} medium)")
    print()

    # Recall and false positive computation against synthetic ground truth labels.
    flagged_ids = {f["record_id"] for f in flags}
    planted     = [r for r in records if r.get("anomaly_type") == "numerical_outlier"]
    clean       = [r for r in records if r.get("anomaly_type") is None]

    tp    = sum(1 for r in planted if r["record_id"] in flagged_ids)
    fp    = sum(1 for r in clean   if r["record_id"] in flagged_ids)
    total = len(planted)

    recall  = (tp / total * 100) if total else 0.0
    fp_rate = (fp / len(clean) * 100) if clean else 0.0

    print(f"Recall on planted numerical_outlier anomalies: {tp}/{total} ({recall:.1f}%)")
    print(f"False positives on clean records: {fp} ({fp_rate:.2f}%)")

    if recall < 85.0:
        print(f"WARNING: Recall {recall:.1f}% is below target 85%. Check for train/inference feature drift.")
    if fp_rate > 10.0:
        print(f"WARNING: FP rate {fp_rate:.2f}% exceeds 10%. Consider tuning contamination parameter.")
