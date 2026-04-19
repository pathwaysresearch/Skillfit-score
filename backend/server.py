import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import pyarrow.parquet as pq
from transformers import AutoTokenizer, PretrainedConfig
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

OUTPUTS_DIR   = "outputs"
MODEL_DIR     = "jina-embeddings-v3"
ONNX_FILE     = "jina-embeddings-v3/onnx/model.onnx"
PT_MODEL_FILE = f"{OUTPUTS_DIR}/skill_projection_best.pt"
TRAIN_VECS    = f"{OUTPUTS_DIR}/train_embeddings_128d.npy"
TRAIN_LABELS  = f"{OUTPUTS_DIR}/train_labels.npy"

K_NEIGHBORS = 50
artifacts = {}

# ==========================================
# 1. PyTorch Model Architectures
# ==========================================

class AttnNet(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.scorer = nn.Linear(in_dim, 1)

    def forward(self, x, mask=None):
        scores = self.scorer(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask <= 0, -1e9)
        weights = F.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)


class CompressNet(nn.Module):
    def __init__(self, in_dim=1024, h1=512, h2=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),    nn.LayerNorm(h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, out_dim), nn.LayerNorm(out_dim),
        )
        self.skip  = nn.Linear(in_dim, out_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = F.normalize(self.main(x) + self.skip(x), p=2, dim=-1)
        return self.scale * z


class SkillProjectionModel(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128, n_groups=180):
        super().__init__()
        self.attn     = AttnNet(in_dim)
        self.compress = CompressNet(in_dim, out_dim=out_dim)
        self.centers  = nn.Parameter(
            F.normalize(torch.randn(n_groups, out_dim), p=2, dim=-1)
        )

    def encode_set(self, x, mask=None): return self.attn(x, mask=mask)
    def project_vector(self, x):        return self.compress(x)
    def forward(self, x, mask=None):    return self.compress(self.attn(x, mask=mask))


# ==========================================
# 2. Helper Functions
# ==========================================
import re

_TITLE_PREFIXES = re.compile(
    r'^(job\s*title|title|position|role|job|designation|post|vacancy)\s*[:\-]\s*',
    re.IGNORECASE
)
_SENIORITY = {
    'senior', 'junior', 'lead', 'associate', 'principal', 'staff', 'head',
    'chief', 'sr', 'jr', 'entry', 'mid', 'level', 'executive', 'assistant',
}

def extract_title_and_text(raw_text: str):
    lines = str(raw_text).strip().split('\n')
    title = lines[0].strip() if lines else "Unknown Job Title"
    body  = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return title, body

def parse_input_title(description: str) -> str:
    first_line = description.strip().split('\n')[0].strip()
    first_line = _TITLE_PREFIXES.sub('', first_line).strip()
    return first_line

def _title_words(title: str) -> set:
    words = re.sub(r'[^a-z\s]', '', title.lower()).split()
    return {w for w in words if w not in _SENIORITY and len(w) > 2}

def titles_too_similar(input_title: str, candidate_title: str) -> bool:
    w1 = _title_words(input_title)
    w2 = _title_words(candidate_title)
    if not w1 or not w2:
        return False
    overlap = w1 & w2
    return len(overlap) / min(len(w1), len(w2)) >= 0.6

def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    token_embeddings     = model_output
    input_mask_expanded  = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded  = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings       = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask             = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def embed_texts(texts: list[str]) -> np.ndarray:
    tokenizer, session, task_id = artifacts["tokenizer"], artifacts["session"], artifacts["task_id"]
    input_text = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs = {
        "input_ids":      input_text["input_ids"],
        "attention_mask": input_text["attention_mask"],
        "task_id":        np.array([task_id] * len(texts), dtype=np.int64),
    }
    outputs    = session.run(None, inputs)[0]
    embeddings = mean_pooling(outputs, input_text["attention_mask"])
    return (embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)).astype(np.float32)

# ==========================================
# 3. Startup Logic
# ==========================================
@app.on_event("startup")
def load_artifacts():
    global artifacts

    # CPU-only — no GPU on Cloud Run
    session   = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    task_id   = PretrainedConfig.from_pretrained(MODEL_DIR).lora_adaptations.index("text-matching")

    # --- cleaned_data: only load id + group columns, then convert to dicts and drop ---
    schema       = pq.read_schema(f"{OUTPUTS_DIR}/cleaned_data.parquet")
    id_col       = next((c for c in schema.names if "id" in c.lower()), "Job ID")
    cleaned_df   = pd.read_parquet(f"{OUTPUTS_DIR}/cleaned_data.parquet",
                                   columns=[id_col, "Assigned_Occupation_Group"])

    # Precompute: sorted occupation counts list
    occ_counts = (cleaned_df["Assigned_Occupation_Group"]
                  .value_counts()
                  .reset_index()
                  .rename(columns={"index": "name", "Assigned_Occupation_Group": "count"})
                  .to_dict(orient="records"))

    # Precompute: group → list of job IDs (for sampling in occupation_jobs)
    occ_id_map = cleaned_df.groupby("Assigned_Occupation_Group")[id_col].apply(list).to_dict()
    del cleaned_df  # no longer needed

    # --- llm_jobs: only load id + job_text, convert to dict for O(1) lookups ---
    llm_schema  = pq.read_schema(f"{OUTPUTS_DIR}/llm_ready_jobs.parquet")
    llm_id_col  = next((c for c in llm_schema.names if "id" in c.lower()), "Job ID")
    llm_df      = pd.read_parquet(f"{OUTPUTS_DIR}/llm_ready_jobs.parquet",
                                  columns=[llm_id_col, "job_text"])
    job_text_lookup = dict(zip(llm_df[llm_id_col], llm_df["job_text"]))
    del llm_df  # no longer needed

    # --- Numpy arrays ---
    train_vecs   = torch.from_numpy(np.load(TRAIN_VECS)).float()
    train_labels = np.load(TRAIN_LABELS, allow_pickle=True)
    train_job_ids = np.load(f"{OUTPUTS_DIR}/train_job_ids.npy", allow_pickle=True)

    # --- PyTorch model ---
    device   = torch.device("cpu")
    n_groups = len(np.unique(train_labels))
    model    = SkillProjectionModel(in_dim=1024, out_dim=128, n_groups=n_groups).to(device)
    if os.path.exists(PT_MODEL_FILE):
        model.load_state_dict(torch.load(PT_MODEL_FILE, map_location=device))
    model.eval()

    artifacts = {
        "session": session, "tokenizer": tokenizer, "task_id": task_id,
        "occ_counts": occ_counts, "occ_id_map": occ_id_map,
        "job_text_lookup": job_text_lookup,
        "model": model, "device": device,
        "train_vecs": train_vecs, "train_labels": train_labels,
        "train_job_ids": train_job_ids,
    }
    print("Backend Server Ready!")

# ==========================================
# 4. API Endpoints
# ==========================================
class CompareRequest(BaseModel):
    job1: str
    job2: str

class PredictRequest(BaseModel):
    description: str
    title: str = ""

@app.post("/api/compare_jobs")
def compare_jobs(req: CompareRequest):
    vecs_1024 = np.vstack([embed_texts([req.job1]), embed_texts([req.job2])])
    model, device = artifacts["model"], artifacts["device"]
    with torch.no_grad():
        vecs_128 = model.project_vector(torch.from_numpy(vecs_1024).to(device))
        sim = F.cosine_similarity(vecs_128[0].unsqueeze(0), vecs_128[1].unsqueeze(0)).item()
    return JSONResponse({"skill_fit_score": round(max(0.0, float(sim)) * 100, 2)})

@app.post("/api/predict_occupation")
def predict_occupation(req: PredictRequest):
    vec_1024 = embed_texts([req.description])
    model, device = artifacts["model"], artifacts["device"]

    with torch.no_grad():
        query_128   = model.project_vector(torch.from_numpy(vec_1024).to(device))
        similarities = F.cosine_similarity(query_128, artifacts["train_vecs"])
        topk_vals, topk_indices = torch.topk(similarities, K_NEIGHBORS)
        topk_vals   = topk_vals.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

    group_scores = {}
    for idx, sim_score in zip(topk_indices, topk_vals):
        label = artifacts["train_labels"][idx]
        group_scores[label] = group_scores.get(label, 0) + float(sim_score)

    predicted_group = max(group_scores, key=group_scores.get)
    confidence      = group_scores[predicted_group] / sum(group_scores.values())

    input_title     = parse_input_title(req.description)
    job_text_lookup = artifacts["job_text_lookup"]
    similar_jobs    = []
    seen_titles     = set()

    for idx in topk_indices:
        job_id = artifacts["train_job_ids"][idx]
        text   = job_text_lookup.get(job_id)
        if text is None: continue

        title, body = extract_title_and_text(text)
        if titles_too_similar(input_title, title): continue
        if title.lower() in seen_titles: continue

        similar_jobs.append({"job_id": str(job_id), "title": title, "job_text": body})
        seen_titles.add(title.lower())
        if len(similar_jobs) >= 10: break

    return JSONResponse({
        "predicted_group": str(predicted_group),
        "confidence": round(confidence, 3),
        "similar_jobs": similar_jobs,
    })

@app.get("/api/occupations")
def get_occupations():
    return JSONResponse(artifacts["occ_counts"])

@app.get("/api/occupation_jobs/{group_name}")
def get_occupation_jobs(group_name: str):
    id_pool = artifacts["occ_id_map"].get(group_name, [])
    if not id_pool:
        return JSONResponse([])

    sampled_ids     = random.sample(id_pool, min(10, len(id_pool)))
    job_text_lookup = artifacts["job_text_lookup"]
    output_jobs     = []

    for job_id in sampled_ids:
        text = job_text_lookup.get(job_id)
        if text is None: continue
        title, body = extract_title_and_text(text)
        output_jobs.append({"title": title.upper(), "job_text": body})

    return JSONResponse(output_jobs)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
