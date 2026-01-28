FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal). Add build tools only if you later need them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Default for most platforms
ENV PORT=8000
EXPOSE 8000

# Use gunicorn for production. Keep workers modest; this app does outbound IO.
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker visitassist_rag.app:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 180"]
