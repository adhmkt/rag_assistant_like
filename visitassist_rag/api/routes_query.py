from fastapi import APIRouter
from visitassist_rag.models.schemas import QueryRequest, QueryResponse
from visitassist_rag.rag.engine import rag_query

router = APIRouter()

@router.post("/kb/{kb_id}/query", response_model=QueryResponse)
def query_kb(kb_id: str, req: QueryRequest):
    return rag_query(kb_id=kb_id, **req.dict())
