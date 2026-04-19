import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
from transformers import AutoTokenizer, PretrainedConfig
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure outputs directory exists for static mounting
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# --- Configuration & Paths ---
OUTPUTS_DIR      = "outputs"
MODEL_DIR        = "jina-embeddings-v3"
ONNX_FILE        = "jina-embeddings-v3/onnx/model.onnx"
PT_MODEL_FILE    = f"{OUTPUTS_DIR}/skill_projection_best.pt"
TRAIN_VECS_128D  = f"{OUTPUTS_DIR}/train_embeddings_128d.npy"
TRAIN_LABELS_NPY = f"{OUTPUTS_DIR}/train_labels.npy" 

K_NEIGHBORS      = 50
artifacts = {}

# ==========================================
# 1. PyTorch Model Architectures (UPDATED)
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

        # FIX: learnable scale — lets model control how "spread out" vectors are
        # on the hypersphere, preventing trivial collapse to a small patch
        self.scale = nn.Parameter(torch.tensor(10.0))   # init=10 per NormFace recommendation

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = F.normalize(self.main(x) + self.skip(x), p=2, dim=-1)
        return self.scale * z    # scale after normalize — magnitude is now learnable


class SkillProjectionModel(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128, n_groups=180):
        super().__init__()
        self.attn     = AttnNet(in_dim)
        self.compress = CompressNet(in_dim, out_dim=out_dim)

        # FIX: Center loss centers — one 128-d center per group
        # Updated via their own gradient at a slower LR
        self.centers  = nn.Parameter(
            F.normalize(torch.randn(n_groups, out_dim), p=2, dim=-1)
        )

    def encode_set(self, x, mask=None): return self.attn(x, mask=mask)
    def project_vector(self, x):        return self.compress(x)
    def forward(self, x, mask=None):    return self.compress(self.attn(x, mask=mask))


# ==========================================
# 2. Helper Functions
# ==========================================
def extract_title_and_text(raw_text: str):
    """Parses first line as title, rest as body."""
    lines = str(raw_text).strip().split('\n')
    title = lines[0].strip() if lines else "Unknown Job Title"
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return title, body

def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def embed_texts(texts: list[str]) -> np.ndarray:
    tokenizer, session, task_id = artifacts["tokenizer"], artifacts["session"], artifacts["task_id"]
    input_text = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs = {
        "input_ids": input_text["input_ids"],
        "attention_mask": input_text["attention_mask"],
        "task_id": np.array([task_id]*len(texts), dtype=np.int64),
    }
    outputs = session.run(None, inputs)[0]
    embeddings = mean_pooling(outputs, input_text["attention_mask"])
    return (embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)).astype(np.float32)

# ==========================================
# 3. Startup Logic
# ==========================================
@app.on_event("startup")
def load_artifacts():
    global artifacts
    
    providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    try: session = ort.InferenceSession(ONNX_FILE, providers=providers)
    except Exception: session = ort.InferenceSession(ONNX_FILE, providers=['CPUExecutionProvider'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    task_id = PretrainedConfig.from_pretrained(MODEL_DIR).lora_adaptations.index('text-matching')

    cleaned_df = pd.read_parquet(f"{OUTPUTS_DIR}/cleaned_data.parquet")
    llm_jobs_df = pd.read_parquet(f"{OUTPUTS_DIR}/llm_ready_jobs.parquet")
    
    train_vecs = torch.from_numpy(np.load(TRAIN_VECS_128D)).float()
    train_labels = np.load(TRAIN_LABELS_NPY, allow_pickle=True)

    train_job_ids = np.load(f"{OUTPUTS_DIR}/train_job_ids.npy", allow_pickle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure n_groups matches the unique count in your labels
    n_groups = len(np.unique(train_labels))
    model = SkillProjectionModel(in_dim=1024, out_dim=128, n_groups=n_groups).to(device)
    
    if os.path.exists(PT_MODEL_FILE):
        print(f"Loading trained model from {PT_MODEL_FILE}...")
        model.load_state_dict(torch.load(PT_MODEL_FILE, map_location=device))
    model.eval()

    artifacts = {
        "session": session, "tokenizer": tokenizer, "task_id": task_id,
        "cleaned_df": cleaned_df, "llm_jobs_df": llm_jobs_df,
        "model": model, "device": device,
        "train_vecs": train_vecs.to(device), "train_labels": train_labels,
        "train_job_ids": train_job_ids
    }
    print("Backend Server Ready (k-NN Mode)!")

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
    vec1 = embed_texts([req.job1])
    vec2 = embed_texts([req.job2])
    vecs_1024 = np.vstack([vec1, vec2])
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
        query_128 = model.project_vector(torch.from_numpy(vec_1024).to(device))
        similarities = F.cosine_similarity(query_128, artifacts["train_vecs"])
        
        # Use K_NEIGHBORS (200)
        topk_vals, topk_indices = torch.topk(similarities, K_NEIGHBORS)
        topk_vals = topk_vals.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

    # 1. WEIGHTED VOTING: Score = Sum of Cosine Similarities
    # This prevents high-frequency jobs from winning by "luck"
    group_scores = {}
    for idx, sim_score in zip(topk_indices, topk_vals):
        label = artifacts["train_labels"][idx]
        group_scores[label] = group_scores.get(label, 0) + float(sim_score)

    predicted_group = max(group_scores, key=group_scores.get)
    confidence = group_scores[predicted_group] / sum(group_scores.values())

    # 2. RETRIEVE ACTUAL NEIGHBORS (No random sampling)
    llm_jobs_df = artifacts["llm_jobs_df"]
    llm_id_col = next((c for c in llm_jobs_df.columns if "id" in c.lower()), "Job ID")
    
    similar_jobs = []
    seen_titles = set()
    
    for idx in topk_indices:
        # Get the EXACT Job ID that matched this vector
        neighbor_job_id = artifacts["train_job_ids"][idx]
        
        job_row = llm_jobs_df[llm_jobs_df[llm_id_col] == neighbor_job_id]
        if job_row.empty: continue
            
        row = job_row.iloc[0]
        title, body = extract_title_and_text(row["job_text"])
        
        if req.title and title.lower() == req.title.lower(): continue
        if title.lower() in seen_titles: continue
        
        similar_jobs.append({
            "job_id": str(neighbor_job_id),
            "title": title,
            "job_text": body
        })
        seen_titles.add(title.lower())
        if len(similar_jobs) >= 10: break

    return JSONResponse({
        "predicted_group": str(predicted_group),
        "confidence": round(confidence, 3),
        "similar_jobs": similar_jobs
    })

@app.get("/api/occupations")
def get_occupations():
    counts = artifacts["cleaned_df"]["Assigned_Occupation_Group"].value_counts().reset_index()
    counts.columns = ["name", "count"]
    return JSONResponse(counts.sort_values(by="count", ascending=False).to_dict(orient="records"))

@app.get("/api/occupation_jobs/{group_name}")
def get_occupation_jobs(group_name: str):
    cleaned_df, llm_jobs_df = artifacts["cleaned_df"], artifacts["llm_jobs_df"]
    id_col = next((c for c in cleaned_df.columns if "id" in c.lower()), "Job ID")
    llm_id_col = next((c for c in llm_jobs_df.columns if "id" in c.lower()), "Job ID")

    group_pool = cleaned_df[cleaned_df["Assigned_Occupation_Group"] == group_name]
    if group_pool.empty: return JSONResponse([])
        
    sampled_ids = group_pool.sample(n=min(10, len(group_pool)))[id_col].values
    jobs_data = llm_jobs_df[llm_jobs_df[llm_id_col].isin(sampled_ids)]
    
    output_jobs = []
    for _, row in jobs_data.iterrows():
        title, body = extract_title_and_text(row.get("job_text", "Unknown\nNo Content"))
        output_jobs.append({
            "title": title.upper(),
            "job_text": body
        })
        
    return JSONResponse(output_jobs)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)