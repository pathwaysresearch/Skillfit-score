from google.cloud import storage
import os, requests

BUCKET = "skillfit-artifacts"
FILES = [
    "cleaned_data.parquet",
    "llm_ready_jobs.parquet",
    "skill_projection_best.pt",
    "train_embeddings_128d.npy",
    "train_job_ids.npy",
    "train_labels.npy",
]

# Fetch project ID from GCP metadata server (available in Cloud Build and Cloud Run)
try:
    project_id = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/project/project-id",
        headers={"Metadata-Flavor": "Google"},
        timeout=3
    ).text
except Exception:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

os.makedirs("outputs", exist_ok=True)
client = storage.Client(project=project_id)
bucket = client.bucket(BUCKET)

for f in FILES:
    print(f"Downloading {f}...")
    bucket.blob(f).download_to_filename(f"outputs/{f}")

print("All outputs downloaded.")
