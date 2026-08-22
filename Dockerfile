FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY pyproject.toml .
# The GGUF extra (llama-cpp-python) is a slow C++ build only the GPU worker's
# local fallback needs — skip it for every other service. Set
# INSTALL_GGUF=true in the gpu-worker build args to enable local GGUF models.
ARG INSTALL_GGUF=false
RUN if [ "$INSTALL_GGUF" = "true" ]; then \
    pip install --no-cache-dir -e ".[gguf]"; \
    else \
    pip install --no-cache-dir -e .; \
    fi

# Optional: pre-download embedding models into the image layer.
# Set PREDOWNLOAD_MODELS=true in the gpu-worker build args to bake models in
# so the worker starts with zero network downloads.
# Placed here (after pip install, before COPY src/) so Docker caches this layer
# until Python dependencies change — not on every source file edit.
ARG PREDOWNLOAD_MODELS=false
RUN if [ "$PREDOWNLOAD_MODELS" = "true" ]; then \
    python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
SentenceTransformer('Qwen/Qwen3-0.6B', trust_remote_code=True); \
print('Embedding models cached.')"; \
fi

# Copy source
COPY src/ src/

# Ensure the package is importable as event_driven_rag_service
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "event_driven_rag_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
