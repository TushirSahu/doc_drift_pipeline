# DocDrift API image.
# Why: reproducible deploys — the same pinned dependencies run on a laptop or a
# server, so "works on my machine" stops being a deployment risk.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first so Docker can cache this layer across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Liveness probe hits the dependency-aware /health endpoint (honors $PORT).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.getenv('PORT','8000')}/health\").status==200 else 1)" || exit 1

# Shell form so $PORT (injected by Render / Koyeb / HF Spaces) is honored.
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
