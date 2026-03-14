# PDF Parser RAG – Serving Layer
# Base: PyTorch + CUDA 12.1 (for BGE-M3 and reranker on GPU)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (serving only, no OCR/eval)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy source code
COPY src/ ./src/
COPY scripts/gradio_demo.py ./scripts/gradio_demo.py
COPY config/ ./config/

# Models and ChromaDB data are mounted at runtime (too large to bake in)
# - ./models        -> /app/models
# - ./vectors       -> /app/vectors
# - ./.env          -> /app/.env

EXPOSE 8000 7860

# Default: start FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
