# Client UI — Ingestion Guide (Node.js)

This guide describes how to build a Node.js-based client UI that ingests content using the VisitAssist RAG API routes available in this repo.

## API base

- Local dev API base: `http://127.0.0.1:8000/v1`
- Health check: `GET /health`

All ingestion and query routes below are under the `/v1` prefix.

## Concepts

- **kb_id**: Knowledge base identifier. The UI must collect or select a `kb_id` (e.g. `itaipu`) and include it in the URL path.
- **Ingestion is text-first**: the backend ingest contract expects *plain text* plus metadata. “Source file type” (PDF/DOCX/HTML) is resolved in the UI or an upstream converter.
- **Provenance**: always provide `source_type` / `source_uri` (or `url`) so query snippets can trace back to the original source.

## Supported ingestion capabilities

### 1) Ingest plain text (generic)

**Route**

- `POST /kb/{kb_id}/ingest/text`

**Use when**

- The UI already has text extracted (pasted text, file-converted text, copied email, etc.).

**Request body**

```json
{
  "title": "string",
  "text": "string",
  "source_type": "txt",
  "source_uri": "",
  "language": "pt",
  "doc_date": "YYYY-MM-DD",
  "doc_year": 2026
}
```

Notes:
- `doc_date` is optional but recommended in `YYYY-MM-DD`.
- `doc_year` is optional. If you set `doc_date`, set `doc_year` to the same year.
- Large text is okay; the backend will chunk it.

**Response**

```json
{
  "success": true,
  "doc_id": "uuid",
  "message": null
}
```

### 2) URL ingest (deterministic fetch + extract) — preview

**Route**

- `POST /kb/{kb_id}/ingest/url/preview`

**Use when**

- You want the backend to fetch a URL and deterministically extract readable text, but you want to show the user a preview and any extraction warnings before ingesting.

**Request body**

```json
{
  "url": "https://example.com/page",
  "language": "pt"
}
```

**Response (preview)**

```json
{
  "url": "https://example.com/page",
  "final_url": "https://example.com/page",
  "title": "Page title",
  "text_preview": "...",
  "content_hash": "...",
  "char_count": 12345,
  "word_count": 2345,
  "link_count": 12,
  "link_density": 0.05,
  "warnings": ["..."]
}
```

UI guidance:
- Show `title`, `char_count`, `word_count`, and `warnings` prominently.
- Show `text_preview` in a read-only textarea.
- If the endpoint returns HTTP 400 (often due to 403/blocked sites), offer the **URL paste fallback** flow below.

### 3) URL ingest (deterministic fetch + extract) — confirm

**Route**

- `POST /kb/{kb_id}/ingest/url/confirm`

**Use when**

- The user reviewed the preview and chose to ingest the URL.

**Request body**

```json
{
  "url": "https://example.com/page",
  "title": "optional override title",
  "language": "pt",
  "doc_date": "YYYY-MM-DD",
  "doc_year": 2026
}
```

Notes:
- The backend **re-fetches** and **re-extracts** on confirm for auditable ingestion.
- Keep this deterministic: don’t run any LLM cleanup in the UI.

**Response**

```json
{
  "success": true,
  "doc_id": "uuid",
  "message": null
}
```

### 4) URL paste fallback (for blocked/unreachable URLs)x

**Route**

- `POST /kb/{kb_id}/ingest/url/paste`

**Use when**

- The URL preview/confirm fails (e.g., HTTP 403 due to bot protection) **or** the user prefers to paste the content they can access in a browser.

**Request body**

```json
{
  "url": "https://example.com/page",
  "text": "(copied page text)",
  "title": "optional",
  "language": "pt",
  "doc_date": "YYYY-MM-DD",
  "doc_year": 2026
}
```

Validation notes:
- `url` must start with `http://` or `https://`.
- `text` must be at least ~200 characters (otherwise the API returns HTTP 400).

**Response**

```json
{
  "success": true,
  "doc_id": "uuid",
  "message": null
}
```

## Suggested UI layout (ingestion)

Create an “Ingest” page with 3 tabs:

1. **Text**
   - Fields: `kb_id`, `title`, `language`, `doc_date`, `doc_year`, `source_type`, `source_uri`, `text`.
   - Button: “Ingest text” → calls `/ingest/text`.

2. **URL**
   - Fields: `kb_id`, `url`, `language`, `doc_date`, `doc_year`, optional `title override`.
   - Step 1: “Preview” → calls `/ingest/url/preview`.
   - Show preview (`title`, `warnings`, `text_preview`).
   - Step 2: “Confirm ingest” → calls `/ingest/url/confirm`.
   - Error path: if preview fails (400), show “Site blocked; use Paste tab”.

3. **URL (Paste)**
   - Fields: `kb_id`, `url`, `title`, `language`, `doc_date`, `doc_year`, `text`.
   - Button: “Ingest pasted text” → calls `/ingest/url/paste`.

### Text formatting rules (recommended)

Keep it simple and deterministic:
- Preserve line breaks (send plain text with `\n`).
- Normalize Windows newlines `\r\n` → `\n`.
- Trim leading/trailing whitespace.
- Optionally collapse runs of >2 blank lines to 2.

Avoid:
- LLM-based rewriting/cleaning.
- HTML conversion.

## Query page (verification)

After ingestion, the UI should let the user verify retrieval.

### Full payload query (explicative but grounded)

- `POST /kb/{kb_id}/query`

```json
{
  "question": "string",
  "language": "pt",
  "mode": "tourist_chat",
  "debug": false
}
```

Response contains:
- `answer`
- `snippets[]` (with `source` metadata like `doc_id`, `source_uri`, `source_type`, `doc_date`, etc.)

### Short answer query (strict)

- `POST /kb/{kb_id}/query/answer`

```json
{
  "question": "string",
  "language": "pt",
  "mode": "tourist_chat",
  "debug": false
}
```

Response:

```json
{ "answer": "...\nFonte: [S1]" }
```

UI tip:
- Provide a toggle: “Short answer (strict)” vs “Full (with snippets)”.

## Node.js implementation notes

### Configuration

- Store API base URL in an env var, e.g. `VISITASSIST_API_BASE_URL=http://127.0.0.1:8000/v1`.

### Fetch examples (TypeScript)

```ts
const API_BASE = process.env.VISITASSIST_API_BASE_URL ?? "http://127.0.0.1:8000/v1";

export async function ingestUrlPaste(kbId: string, payload: {
  url: string;
  text: string;
  title?: string;
  language?: string;
  doc_date?: string;
  doc_year?: number;
}) {
  const res = await fetch(`${API_BASE}/kb/${encodeURIComponent(kbId)}/ingest/url/paste`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json" },
    body: JSON.stringify(payload),
  });

  const bodyText = await res.text();
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${bodyText}`);
  return JSON.parse(bodyText);
}
```

### Error handling

- HTTP 400: user-correctable (missing URL, too-small text, blocked URL fetch, etc.). Show `detail`.
- HTTP 500: server/config issue (e.g., missing API keys). Show the message and suggest retry.

### CORS / deployment

If you host the Node UI on a different origin than the API, you’ll need one of:
- Configure CORS on the FastAPI server, or
- Serve the UI behind the same domain (reverse proxy), or
- Use a Next.js server action / API route as a proxy to the FastAPI backend.

## Recommended “minimum viable” ingest UX

- Require `kb_id`.
- For URL ingest:
  - Try preview → if blocked, automatically switch to Paste tab and keep the URL filled.
- Always show the returned `doc_id` after a successful ingest.
- Provide a “Test query” box right on the success screen.
