# Ingestion UI (Design Notes & Requirements)

This document captures options, requirements, and implementation suggestions for adding a web UI for ingesting content into the VisitAssist RAG service.

## Context

Today the service supports ingestion via API (text-based) and via a local script for PDFs (batching/resume):

- API ingestion: `POST /v1/kb/{kb_id}/ingest/text`
- Query endpoints:
  - `POST /v1/kb/{kb_id}/query`
  - `POST /v1/kb/{kb_id}/query/answer`
- Query testing UI already exists:
  - `{BASE_URL}/docs` (Swagger)
  - `{BASE_URL}/ui/test`

The ingestion UI should make it easy for a non-technical user to ingest:
- Paste text
- Upload PDF / DOCX / TXT
- Potentially ingest large documents safely (long-running)

## Goals

- Provide a browser-based workflow to ingest documents into a chosen `kb_id`.
- Support common source types (start with PDF and text paste).
- Reduce human error (wrong KB, wrong language, missing title, etc.).
- Provide clear status/error feedback.
- Keep security sane (avoid turning ingestion into an anonymous public upload endpoint).

## Non-goals (for the first version)

- Full document management UI (listing, deleting, re-ingesting, versioning).
- Role-based access control / multi-user accounts (unless required by deployment).
- Perfect extraction for every file type.

## Key design decision: where does text extraction happen?

There are three viable approaches.

### Option A — Text-only UI (fastest, safest)

UI collects metadata + pasted text and calls the existing endpoint:

- `POST /v1/kb/{kb_id}/ingest/text`

Pros:
- Minimal backend changes.
- No file-upload security risks.
- Works with any hosting.

Cons:
- Users must paste text (or do their own extraction).
- Large PDFs become inconvenient.

When to choose:
- You want value quickly.
- Admin users can paste or provide extracted text.

### Option B — Upload UI + server-side extraction (best UX)

UI uploads a file. Server extracts text and then runs the same ingestion pipeline.

Recommended new endpoint(s):

- `POST /v1/kb/{kb_id}/ingest/file` (multipart/form-data)
  - `file`: the uploaded file
  - optional form fields: `title`, `language`, `source_type`, `doc_date`, `doc_year`

Extraction suggestions:
- PDF: PyMuPDF (`fitz`) (already used in `pdf_ingestor.py`)
- DOCX: `python-docx`
- TXT/MD: decode as UTF-8 (with fallback)

Pros:
- Best user experience.
- One-click ingestion.

Cons:
- Security: must protect endpoint.
- Reliability: large files can be slow/time out.
- Operational: may require async job + progress.

When to choose:
- You want non-technical ingestion.
- You’re comfortable protecting ingestion behind auth.

### Option C — Upload UI + async job/worker (most scalable)

UI uploads a file, server enqueues ingestion, returns a job id, and UI polls for progress.

Suggested endpoints:

- `POST /v1/kb/{kb_id}/ingest/jobs` (multipart upload or reference URL)
  - returns `{job_id}`
- `GET /v1/ingest/jobs/{job_id}`
  - returns status/progress/result

Pros:
- Best for very large PDFs and flaky networks.
- Allows resumability and rate-limited processing.

Cons:
- Needs a queue/worker layer (Celery/RQ/BackgroundTasks + persistent store).

When to choose:
- PDFs can be hundreds of pages.
- You need robust progress and retries.

## Recommended phased plan

Phase 1 (1–2 hours): **Text-only UI**
- Add `{BASE_URL}/ui/ingest` HTML page.
- Paste text + metadata.
- Calls `POST /v1/kb/{kb_id}/ingest/text`.
- Shows response and errors.

Phase 2: **PDF upload UI (admin-only)**
- Add `POST /v1/kb/{kb_id}/ingest/file` for PDF/TXT.
- Start with a conservative max size and page limit.
- Return extracted text length and/or a preview for transparency.

Phase 3: **Async ingestion**
- Add jobs + progress.
- Move large PDFs to background processing.

## UI requirements

### Required fields
- `kb_id` (string): which KB/tenant to ingest into
- `title` (string): displayed in snippets and debug
- `language` (string): default `pt`

### Optional fields
- `doc_date` (YYYY-MM-DD) and/or `doc_year` (int)
- `source_type` (pdf, txt, docx, md, etc.)
- `source_uri` (provenance)

### UX expectations
- Clear “where am I ingesting?” banner: KB name + environment.
- Confirm before ingesting large content.
- Show progress (at least “uploading” vs “ingesting” states).
- Show a request id (if available) and a copy-paste debug payload.

### Validation & guardrails
- Require non-empty title.
- Require non-empty content.
- Warn if `kb_id` looks wrong (e.g., contains spaces).
- If uploading:
  - validate file extension and MIME type (best effort)
  - enforce max upload size

## Security requirements (important)

Ingestion is a write operation and should not be publicly open.

Minimum protection (recommended):
- Add an API key requirement, e.g. `X-Admin-Key: ...`, checked server-side.
  - Store as env var (e.g. `ADMIN_API_KEY`).
  - UI includes a password field or expects the key in a browser session.

Other acceptable options:
- Basic auth behind a reverse proxy.
- Platform-level auth (Render private service + internal access).

Also recommended:
- CORS configured appropriately.
- CSRF not required if you do not use cookies; if you do, add CSRF.

## Performance & limits

### Known cost drivers
- Embeddings cost scales with tokens.
- Ingest time scales with number of chunks and external round-trips.

Suggested limits for synchronous ingestion endpoints:
- Max file size: start with 10–25 MB.
- Max pages (PDF): start with 50–100 pages for synchronous.
- Max request time: 60–180 seconds depending on hosting.

If you expect 300–800 page PDFs, plan for Option C (async jobs) early.

## Observability

Nice-to-have to simplify support:
- Log a request id for each ingestion.
- Include summary in response: number of chunks, chunk types counts, doc_id.
- Record ingestion failures with a reason (auth, extraction, embedding, upsert).

## Suggested UI routes

- `GET /ui/ingest` — admin ingestion form
- (optional) `GET /ui/ingest/jobs/{job_id}` — job progress page for async

## Suggested API additions (when you’re ready)

If implementing file upload ingestion:
- `POST /v1/kb/{kb_id}/ingest/file`
  - supports: PDF/TXT first
  - returns: `{success, doc_id, message, stats}`

If implementing async:
- `POST /v1/kb/{kb_id}/ingest/jobs`
- `GET /v1/ingest/jobs/{job_id}`

## Notes on DOCX support

DOCX support is feasible but adds dependency and complexity.

- Extraction library: `python-docx`
- Some DOCX contain tables; decide whether to keep tables as text or drop.

Recommendation:
- Add DOCX after PDF upload is stable.

## Integration with existing PDF ingestion tooling

The existing `pdf_ingestor.py` already supports batching/retries/resume and is useful for backfilling large PDFs.

Even after adding a UI, keep `pdf_ingestor.py` for:
- huge documents
- automated pipelines
- ingestion at scale

---

## Checklist for “ready to build”

- Decide Option A vs B vs C for v1.
- Decide whether UI is admin-only.
- Decide auth strategy (ADMIN_API_KEY header is simplest).
- Decide max upload size/timeouts.
- Decide initial supported file types (recommended: paste text + PDF).
