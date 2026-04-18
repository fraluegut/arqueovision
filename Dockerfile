# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System libs required by Pillow / OpenCV-compatible builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source  (models/ is mounted at runtime via docker-compose volume)
COPY src/ ./src/

# Ensure src is importable from /app
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
