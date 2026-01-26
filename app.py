from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import ingest_text_document, rag_query, KB_ID, DEFAULT_LANG

app = FastAPI(title="Assistant-like RAG (Pinecone Index)")

class IngestReq(BaseModel):
    title: str
    text: str
    source_type: str = "txt"
    source_uri: str = ""
    kb_id: str = KB_ID
    language: str = DEFAULT_LANG

class QueryReq(BaseModel):
    question: str
    kb_id: str = KB_ID
    language: str = DEFAULT_LANG

@app.post("/ingest/text")
def ingest_text(req: IngestReq):
    return ingest_text_document(
        title=req.title,
        text=req.text,
        source_type=req.source_type,
        source_uri=req.source_uri,
        kb_id=req.kb_id,
        lang=req.language,
    )

@app.post("/rag/query")
def query(req: QueryReq):
    return rag_query(req.question, kb_id=req.kb_id, lang=req.language)
