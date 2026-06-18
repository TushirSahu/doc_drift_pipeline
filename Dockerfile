# DocDrift API image — runs on Render/Koyeb and Hugging Face Spaces.
# HF Spaces run the container as a non-root user (uid 1000), so we create that
# user and chown the app; otherwise runtime writes to metrics/ fail.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd -m -u 1000 user
WORKDIR /app

# Install deps first (as root, system-wide) so this layer caches across changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source owned by the runtime user; ensure metrics/ is writable.
COPY --chown=user:user . .
RUN mkdir -p metrics && chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 8000

# Liveness probe hits /health (honors $PORT). HF ignores HEALTHCHECK; Render uses it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.getenv('PORT','8000')}/health\").status==200 else 1)" || exit 1

# Shell form so $PORT (Render/Koyeb) is honored; HF Spaces uses app_port in README.
CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
