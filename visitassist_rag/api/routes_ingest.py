from fastapi import APIRouter
from visitassist_rag.models.schemas import IngestTextRequest, IngestResponse
from visitassist_rag.rag.ingest import ingest_text_document

router = APIRouter()

@router.post("/kb/{kb_id}/ingest/text", response_model=IngestResponse)
def ingest_text(kb_id: str, req: IngestTextRequest):
    return ingest_text_document(kb_id=kb_id, **req.dict())
