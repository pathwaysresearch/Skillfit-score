FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first to avoid pulling 2GB+ CUDA wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download Jina embeddings model (ONNX + tokenizer, skip PyTorch weights)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='jinaai/jina-embeddings-v3', \
    local_dir='jina-embeddings-v3', \
    ignore_patterns=['*.bin', '*.pth', '*.safetensors', '*fp16*'] \
)"

COPY . .

ENV PORT=8080

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port $PORT"]
