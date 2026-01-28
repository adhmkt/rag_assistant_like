# Deployment

This repo contains a FastAPI RAG service (query + ingest).

## Build/run locally with Docker

```powershell
docker build -t visitassist-rag .
docker run --rm -p 8000:8000 `
  -e OPENAI_API_KEY=... `
  -e PINECONE_API_KEY=... `
  -e PINECONE_INDEX=... `
  -e SUPABASE_URL=... `
  -e SUPABASE_SERVICE_ROLE_KEY=... `
  visitassist-rag
```

Service will be at `http://localhost:8000`.

## Required environment variables

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`

If using Supabase persistence:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

## Deploy from GitHub (recommended flow)

Use any platform that supports deploying a Dockerfile from a GitHub repo:

- **Render**: New Web Service → Connect GitHub repo → Environment = Docker → set env vars → Deploy.
- **Fly.io**: `fly launch` (Dockerfile detected) → set secrets via `fly secrets set` → deploy.
- **Railway**: New Project → Deploy from GitHub → Dockerfile → set env vars.

## Notes

- Ingestion can be slow (embedding + upsert). For production, consider async ingestion via a queue/worker.
- Keep `debug_no_filter` off in production to avoid cross-tenant leakage.
