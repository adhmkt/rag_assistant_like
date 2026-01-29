# Consuming the VisitAssist RAG API

This document explains how to call the VisitAssist RAG service from your chatbot (or any client).

## Base URLs

- Local dev (typical): `http://localhost:8000`
- Production (example): `https://rag-assistant-like.onrender.com`

All endpoints below are shown with a `{BASE_URL}` placeholder.

## Quick test in the browser

- Swagger UI: `{BASE_URL}/docs`
- Test UI page: `{BASE_URL}/ui/test`
- Health check: `{BASE_URL}/health`

## Key concept: `kb_id`

This is a multi-tenant API.

- The **knowledge base identifier** is the `{kb_id}` path segment.
- **Do not send `kb_id` in the JSON body**. The API uses the path parameter as the source of truth.

Example (kb_id = `itaipu`):

- `POST {BASE_URL}/v1/kb/itaipu/query/answer`

## Recommended endpoint (chatbot): answer-only

### `POST /v1/kb/{kb_id}/query/answer`

Returns only the final answer.

**Request JSON**

```json
{
  "question": "Qual é o total de instrumentos e o total de drenos citados no documento?",
  "language": "pt",
  "mode": "tourist_chat",
  "debug": false
}
```

**Response JSON (success)**

```json
{
  "answer": "O total ...\nFonte: [S1]"
}
```

**Response JSON (error)**

Errors also return an `answer` field for UI consistency:

```json
{
  "answer": "OpenAI authentication failed. Check that OPENAI_API_KEY is set to a valid key in the server environment."
}
```

> Tip: in a chatbot, treat non-2xx as failure, but you can still display the returned `answer` as a friendly error message.

### cURL example

```bash
curl -X POST "{BASE_URL}/v1/kb/itaipu/query/answer" \
  -H "accept: application/json" \
  -H "content-type: application/json" \
  -d '{
    "question": "Qual é o total de instrumentos e o total de drenos citados no documento?",
    "language": "pt",
    "mode": "tourist_chat",
    "debug": false
  }'
```

### JavaScript (fetch) example

```js
const baseUrl = "https://rag-assistant-like.onrender.com";
const kbId = "itaipu";

const res = await fetch(`${baseUrl}/v1/kb/${encodeURIComponent(kbId)}/query/answer`, {
  method: "POST",
  headers: {
    "content-type": "application/json",
    "accept": "application/json",
  },
  body: JSON.stringify({
    question: "Qual é o total de instrumentos e o total de drenos citados no documento?",
    language: "pt",
    mode: "tourist_chat",
    debug: false,
  }),
});

const data = await res.json();
// Always present:
console.log(data.answer);
```

### Python (requests) example

```python
import requests

base_url = "https://rag-assistant-like.onrender.com"
kb_id = "itaipu"

payload = {
    "question": "Qual é o total de instrumentos e o total de drenos citados no documento?",
    "language": "pt",
    "mode": "tourist_chat",
    "debug": False,
}

r = requests.post(f"{base_url}/v1/kb/{kb_id}/query/answer", json=payload, timeout=60)
print(r.status_code)
print(r.json()["answer"])
```

## Full endpoint (debugging / admin tools)

### `POST /v1/kb/{kb_id}/query`

Returns `answer` plus supporting `snippets` (and `debug` if enabled).

**Response JSON (success)**

```json
{
  "answer": "...\nFonte: [S1]",
  "snippets": [
    {
      "type": "paragraph",
      "title": "...",
      "text": "...",
      "source": {"doc_title": "...", "chunk_type": "fine", "doc_year": 2026}
    }
  ],
  "debug": null
}
```

**Response JSON (error)**

```json
{
  "answer": "<error message>",
  "snippets": [],
  "debug": null
}
```

## Ingestion endpoint

### `POST /v1/kb/{kb_id}/ingest/text`

Ingests a text document into a knowledge base.

**Request JSON**

```json
{
  "title": "My Doc",
  "text": "...",
  "source_type": "txt",
  "source_uri": "",
  "language": "pt",
  "doc_date": "2026-01-01",
  "doc_year": 2026
}
```

## Request parameters

- `question` (string, required): user question.
- `language` (string, default `pt`): language hint.
- `mode` (string, default `tourist_chat`): affects retrieval filters.
  - Allowed: `tourist_chat`, `faq_first`, `events`, `directory`, `coupons`
- `debug` (boolean, default `false`): when true (on the full endpoint), returns `debug` diagnostics.

## Operational guidance

- **Timeouts**: For production clients, use timeouts like 30–90 seconds depending on your latency tolerance.
- **Retries**: Safe to retry on network failures and 5xx. Use exponential backoff.
- **Rate limits**: The service depends on external providers (OpenAI/Pinecone). If you see 429/5xx spikes, back off and retry.

## Troubleshooting

- `OpenAI authentication failed...`:
  - Ensure the server has a valid `OPENAI_API_KEY` set in its environment.
  - If you changed the key, restart/redeploy the service.

- Empty answer/snippets for a KB:
  - Confirm you’re calling the correct `{kb_id}` path.
  - Confirm the KB has been ingested and Pinecone metadata includes that `kb_id`.
