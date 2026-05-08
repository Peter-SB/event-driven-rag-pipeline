FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ src/

# Ensure the package is importable as event_driven_rag_service
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "event_driven_rag_service.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
