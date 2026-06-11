# ============================================================
# MODULE: deploy_model.py
# PURPOSE: Upload the trained model artifact to GCS, register it in Vertex AI
#          Model Registry, create an endpoint, and deploy the model to it.
#          Appends VERTEX_ENDPOINT_NAME to .env on success.
# PIPELINE STAGE: Deployment — run after scripts/train_model.py
# INPUTS: artifacts/model.joblib, .env (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION,
#         GOOGLE_APPLICATION_CREDENTIALS)
# OUTPUTS: Deployed Vertex AI endpoint; VERTEX_ENDPOINT_NAME written to .env
# ============================================================

"""Deploys the Isolation Forest artifact to a Vertex AI online prediction endpoint."""

from __future__ import annotations
import sys
import os
import random
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
from dotenv import load_dotenv

# AnomalyScorer must be importable so joblib can reconstruct it when loading the artifact.
from detection.feature_engineering import AnomalyScorer  # noqa: F401

load_dotenv()

ARTIFACT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.joblib")
)
ENV_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)
# sklearn pre-built container — must match the sklearn minor version that
# pickled the artifact (local training env runs sklearn 1.5.x).
SERVING_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"


def _require_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"ERROR: {key} is not set in .env")
        sys.exit(1)
    return val


def _get_or_create_bucket(gcs_client, project_id: str, region: str) -> tuple:
    """Return (bucket_object, bucket_name), creating the bucket if it does not exist."""
    import google.api_core.exceptions

    bucket_name = f"{project_id}-constructiq-models"

    for attempt in range(2):
        try:
            bucket = gcs_client.get_bucket(bucket_name)
            print(f"Using existing GCS bucket: gs://{bucket_name}/")
            return bucket, bucket_name
        except google.api_core.exceptions.NotFound:
            try:
                bucket = gcs_client.create_bucket(bucket_name, location=region)
                print(f"Created GCS bucket: gs://{bucket_name}/")
                return bucket, bucket_name
            except google.api_core.exceptions.Conflict:
                # Name claimed by another project; append a random suffix and retry.
                suffix = random.randint(1000, 9999)
                bucket_name = f"{project_id}-constructiq-models-{suffix}"
                print(f"Bucket name conflict; retrying as: {bucket_name}")

    print("ERROR: Could not create or retrieve a GCS bucket after 2 attempts.")
    sys.exit(1)


def _append_to_env(key: str, value: str) -> None:
    """Append KEY=value to .env without overwriting existing lines."""
    with open(ENV_PATH, "a") as f:
        f.write(f"\n{key}={value}\n")


def main() -> None:
    if not os.path.exists(ARTIFACT_PATH):
        print("Model artifact not found. Run scripts/train_model.py first.")
        sys.exit(1)

    project_id = _require_env("GOOGLE_CLOUD_PROJECT")
    region     = _require_env("GOOGLE_CLOUD_REGION")
    _require_env("GOOGLE_APPLICATION_CREDENTIALS")

    print(f"Project: {project_id}")
    print(f"Region:  {region}")
    print()

    # ── Initialize Vertex AI SDK ──────────────────────────────────────────────
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)

    # ── Load artifact and extract scorer ──────────────────────────────────────
    # model.joblib contains a dict. The Vertex AI sklearn container requires a single
    # object with a predict() method, so we extract and re-save just the AnomalyScorer.
    artifact = joblib.load(ARTIFACT_PATH)
    # Upload the raw IsolationForest, NOT the AnomalyScorer wrapper: the
    # pre-built sklearn container cannot import detection.feature_engineering,
    # so unpickling a custom class crashes the model server at startup.
    # The container's predict() then returns -1/1 labels (-1 = anomaly), which
    # encode the same decision as score_samples() < offset_ — the threshold
    # stored in the artifact. numerical_outliers.py handles the label format.
    scorer = artifact["model"].forest  # raw sklearn IsolationForest

    # ── Upload scorer to GCS ──────────────────────────────────────────────────
    from google.cloud import storage as gcs

    gcs_client  = gcs.Client(project=project_id)
    bucket, bucket_name = _get_or_create_bucket(gcs_client, project_id, region)

    gcs_blob_path = "constructiq-v1/model.joblib"
    artifact_uri  = f"gs://{bucket_name}/constructiq-v1/"

    print(f"Uploading scorer to {artifact_uri}...")
    # Write scorer to a temp file then stream it to GCS — avoids leaving local files.
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump(scorer, tmp_path)
        bucket.blob(gcs_blob_path).upload_from_filename(tmp_path)
        print(f"Uploaded: gs://{bucket_name}/{gcs_blob_path}")
    finally:
        os.unlink(tmp_path)

    # ── Register model in Vertex AI Model Registry ────────────────────────────
    print("\nRegistering model in Vertex AI Model Registry...")
    try:
        model = aiplatform.Model.upload(
            display_name="constructiq-anomaly-detector-v1",
            artifact_uri=artifact_uri,
            serving_container_image_uri=SERVING_IMAGE,
            sync=True,
        )
        print(f"Model registered: {model.resource_name}")
    except Exception as e:
        print(f"ERROR registering model: {e}")
        print("Check: Vertex AI API enabled, IAM role 'Vertex AI User', quota not exceeded.")
        sys.exit(1)

    # ── Create (or reuse) endpoint ────────────────────────────────────────────
    try:
        existing = aiplatform.Endpoint.list(
            filter='display_name="constructiq-anomaly-detector-endpoint"'
        )
        if existing:
            endpoint = existing[0]
            print(f"\nReusing existing endpoint: {endpoint.resource_name}")
        else:
            print("\nCreating endpoint (2-5 minutes)...")
            endpoint = aiplatform.Endpoint.create(
                display_name="constructiq-anomaly-detector-endpoint",
                sync=True,
            )
            print(f"Endpoint created: {endpoint.resource_name}")
    except Exception as e:
        print(f"ERROR creating endpoint: {e}")
        sys.exit(1)

    # ── Deploy model to endpoint ──────────────────────────────────────────────
    print("\nDeploying model to endpoint (8-12 minutes — do not interrupt)...")
    try:
        endpoint.deploy(
            model=model,
            deployed_model_display_name="constructiq-v1-deployed",
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=1,
            sync=True,
        )
        print("Deployment complete.")
    except Exception as e:
        print(f"ERROR deploying model: {e}")
        print("Common cause: quota exceeded. Check: https://console.cloud.google.com/iam-admin/quotas")
        sys.exit(1)

    # ── Persist endpoint name ─────────────────────────────────────────────────
    endpoint_name = endpoint.resource_name
    _append_to_env("VERTEX_ENDPOINT_NAME", endpoint_name)

    print(f"\nDeployment successful.")
    print(f"Endpoint: {endpoint_name}")
    print(f"VERTEX_ENDPOINT_NAME written to .env")
    print(f"\nNext step: python3 -m detection.numerical_outliers")


if __name__ == "__main__":
    main()
