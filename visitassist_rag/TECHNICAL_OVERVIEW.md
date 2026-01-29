# VisitAssist RAG Technical Overview

## Introduction
VisitAssist RAG is a Retrieval-Augmented Generation (RAG) system designed to ingest, index, and answer questions over large document collections (such as PDFs) using a combination of vector search (Pinecone), metadata storage (Supabase), and LLMs (OpenAI). It exposes a FastAPI-based HTTP API for ingestion and querying.

---

## Architecture Overview

- **API Layer**: FastAPI app with endpoints for document ingestion and querying.
- **Ingestion Pipeline**: Processes raw text, splits into sections/chunks, generates embeddings, and stores both metadata and vectors.
- **Retrieval Pipeline**: For a user query, retrieves relevant chunks using vector search, reranks with LLM, deduplicates, and generates a grounded answer.
- **Storage**:
  - **Pinecone**: Stores vector embeddings for semantic search.
  - **Supabase**: Stores document, section, and chunk metadata.

---

## Key Components

### 1. API Endpoints
- **/v1/kb/{kb_id}/ingest/text**: Ingests a text document into a knowledge base.
- **/v1/kb/{kb_id}/query**: Answers a question using the specified knowledge base.

### 2. Ingestion Flow
- **Entry Point**: `ingest_text_document` (rag/ingest.py)
- **Steps**:
  1. Normalize and preprocess text.
  2. Split into sections (by markdown headings or as a whole).
  3. Chunk sections by token count (with overlap for context).
  4. Generate embeddings for each chunk (OpenAI API).
  5. Store metadata in Supabase and vectors in Pinecone.

### 3. Query Flow
- **Entry Point**: `rag_query` (rag/engine.py)
- **Steps**:
  1. Build filters for Pinecone search (by KB, language, chunk type, etc.).
  2. Retrieve candidate chunks from Pinecone (summary, section, fine granularity).
  3. Deduplicate candidates.
  4. Rerank with LLM (OpenAI, rerank.py).
  5. Generate a grounded answer using only retrieved sources (grounding.py).

### 4. Chunking Logic
- **Tokenization**: Uses tiktoken for accurate chunk sizing.
- **Chunk Types**: Section-level (coarse) and fine-grained (smaller, overlapping).
- **Section Detection**: Markdown headings or fallback to whole document.

### 5. Storage Layer
- **Supabase**: Tables for documents, sections, and chunks. CRUD via stores/supabase_store.py.
- **Pinecone**: Vector upsert/query via stores/pinecone_store.py.

---

## File Structure

- `app.py` (root): Alternative API entrypoint using rag_core (legacy/simple mode).
- `visitassist_rag/app.py`: Main FastAPI app, includes all routers.
- `visitassist_rag/api/`: API route definitions.
- `visitassist_rag/models/`: Pydantic schemas for API and internal data.
- `visitassist_rag/rag/`: Core RAG logic (ingest, chunking, retrieval, rerank, grounding).
- `visitassist_rag/stores/`: Storage adapters for Pinecone and Supabase.
- `visitassist_rag/scripts/`: (Optional) CLI scripts for ingestion.
- `visitassist_rag/tests/`: Unit tests for chunking and query logic.

---

## Extending and Customizing
- **Chunking**: Adjust chunk sizes/overlap in `rag/chunking.py`.
- **Embeddings**: Change model in `rag/embeddings.py`.
- **Rerank/Answer**: Modify prompts or models in `rag/rerank.py` and `rag/grounding.py`.
- **Storage**: Swap out Pinecone/Supabase adapters as needed.
- **API**: Add new endpoints in `api/` and register in `app.py`.

---

## Running and Testing
1. Set up environment variables for Pinecone, Supabase, and OpenAI in `.env`.
2. Start the FastAPI server (e.g., `uvicorn visitassist_rag.app:app`).
3. Access API docs at `http://localhost:8000/docs`.
4. Run tests with `pytest`.

For client integration examples (cURL/JS/Python), see `CONSUME_API.md`.

---

## Example Ingestion/Query
- Ingest: POST to `/v1/kb/{kb_id}/ingest/text` with title, text, etc.
- Query: POST to `/v1/kb/{kb_id}/query` with question and options.

---

## Notes
- Designed for multi-language and multi-knowledge-base support.
- Handles large documents by batching and chunking.
- LLMs are used for both reranking and answer generation, ensuring grounded, source-cited responses.

---

## Contact & Contribution
For questions or contributions, see the repository README or contact the maintainers.
