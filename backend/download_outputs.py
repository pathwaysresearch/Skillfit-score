from google.cloud import storage
import os

BUCKET = "skillfit-artifacts"
FILES = [
    "cleaned_data.parquet",
    "llm_ready_jobs.parquet",
    "skill_projection_best.pt",
    "train_embeddings_128d.npy",
    "train_job_ids.npy",
    "train_labels.npy",
]

os.makedirs("outputs", exist_ok=True)
client = storage.Client()
bucket = client.bucket(BUCKET)

for f in FILES:
    print(f"Downloading {f}...")
    bucket.blob(f).download_to_filename(f"outputs/{f}")

print("All outputs downloaded.")
