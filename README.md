# SkillFit Analytics

A system for measuring skill transferability between jobs using deep metric learning on 600,000+ Indian job postings.

Inspired by:
- *Technology and the Shifting Architecture of Occupational Skills*
- *From Posting to Prediction: Building Validated Workforce Analytics*

---

## What it does

Given any job description, SkillFit embeds it into a 128-dimensional skill space and finds the most similar roles in the dataset — quantifying how transferable skills are between occupations.

**Two tools:**
- **Skill Match Comparator** — paste two job descriptions, get a 0–100 compatibility score
- **Occupation Predictor** — paste a job description, get its predicted occupation group and nearest matches

---

## Architecture (Merge of 2 papers Architecture)

```
Raw job text
    │
    ▼
Jina Embeddings v3 (ONNX)  →  1024-dim semantic vector
    │
    ▼
AttnNet  →  aggregates K postings into one prototype vector
    │
    ▼
CompressNet  →  projects 1024-dim → 128-dim skill vector (L2 normalized)
    │
    ▼
Cosine similarity against 600k trained vectors  →  occupation prediction
```

**Training:** Stage 1 AttnNet warmup (MSE vs centroid, 6 epochs) → Stage 2 end-to-end (35 epochs, loss = 0.3×triplet + 0.7×SupCon + 0.01×center, cosine margin=0.8)

**Dataset:** 600,561 job postings across 180 occupation groups

---

## Results

| Metric | Baseline (Jina) | SkillFit | Δ |
|---|---|---|---|
| Recall@1 | 0.6279 | 0.6631 | +0.0352 |
| MRR | 0.7169 | 0.7453 | +0.0284 |
| Silhouette | −0.0330 | 0.0405 | +0.0735 |
| Separation | −0.1359 | 0.5860 | +0.7219 |

125 of 180 occupation groups improved in cluster silhouette score.

---

## Stack

| Layer | Technology |
|---|---|
| Embeddings | Jina Embeddings v3 (ONNX Runtime) |
| Model | PyTorch (AttnNet + CompressNet) |
| Nearest neighbor | torch.topk cosine similarity |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla JS / HTML / CSS |
| Deployment | Google Cloud Run (backend) · Vercel (frontend) |
| Artifact storage | Google Cloud Storage |

---

## Running locally

**Backend**
```bash
cd backend
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python server.py
```
Requires `outputs/` and `jina-embeddings-v3/` to be present locally.

**Frontend**
```bash
# Any static server, e.g.:
python -m http.server 3000
```
Open `http://localhost:3000`. The frontend reads `config.js` for the backend URL — leave it as-is for local dev (falls back to `http://localhost:8000`).

---

## Deployment

- Backend → Google Cloud Run (Dockerfile in `backend/`, artifacts pulled from GCS at build time)
- Frontend → Vercel (reads `vercel.json`, `backend/` is ignored)


After deploying, set your Cloud Run URL in `config.js`:
```js
window.BACKEND_URL = "https://your-service.run.app";
```
