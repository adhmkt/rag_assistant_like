# Eval harness

This folder contains a lightweight evaluation harness to track RAG quality over time.

## Quickstart

Run with your normal environment (Pinecone/OpenAI configured) from the repo root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m visitassist_rag.eval.run_eval --cases visitassist_rag/eval/cases_itaipu.jsonl --kb-id itaipu --mode tourist_chat
```

## Case format (JSONL)

Each line is a JSON object:

- `id` (string): unique identifier
- `kb_id` (string, optional): overrides `--kb-id`
- `question` (string)
- `mode` (string, optional): overrides `--mode`
- `language` (string, optional, default `pt`)
- `expect` (object, optional): soft/strict checks
  - `must_contain_any` (list[string])
  - `must_contain_all` (list[string])
  - `must_not_contain` (list[string])
  - `min_citations` (int, default 1)
  - `max_snippets` (int, default 4)

Checks are intentionally simple: this harness is aimed at catching regressions (missing citations, too many snippets, wrong obvious values, etc.).
