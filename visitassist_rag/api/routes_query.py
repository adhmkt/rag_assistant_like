from fastapi import APIRouter
from fastapi.responses import JSONResponse

from visitassist_rag.models.schemas import AnswerOnlyResponse, QueryRequest, QueryResponse
from visitassist_rag.rag.engine import rag_query

router = APIRouter()

try:
    # openai>=1.x
    from openai import AuthenticationError  # type: ignore
except Exception:  # pragma: no cover
    AuthenticationError = None  # type: ignore


def _rag_query_with_nice_errors(*, kb_id: str, req: QueryRequest, answer_style: str | None = None):
    try:
        extra = {} if answer_style is None else {"answer_style": answer_style}
        return rag_query(kb_id=kb_id, **req.dict(), **extra)
    except RuntimeError as e:
        # Usually missing env vars like OPENAI_API_KEY.
        raise RuntimeError(str(e))
    except Exception as e:
        if AuthenticationError is not None and isinstance(e, AuthenticationError):
            raise RuntimeError(
                "OpenAI authentication failed. Check that OPENAI_API_KEY is set to a valid key in the server environment."
            )
        raise

@router.post("/kb/{kb_id}/query", response_model=QueryResponse)
def query_kb(kb_id: str, req: QueryRequest):
    try:
        # Full payload: allow a more explanatory (but still grounded) format.
        return _rag_query_with_nice_errors(kb_id=kb_id, req=req, answer_style="explicative")
    except RuntimeError as e:
        # Keep response shape consistent for clients.
        return JSONResponse(
            status_code=500,
            content={"answer": str(e), "snippets": [], "debug": None},
        )


@router.post("/kb/{kb_id}/query/answer", response_model=AnswerOnlyResponse)
def query_kb_answer_only(kb_id: str, req: QueryRequest):
    try:
        # Short answers: strict anti-inference.
        resp = _rag_query_with_nice_errors(kb_id=kb_id, req=req, answer_style="strict")
        return AnswerOnlyResponse(answer=resp.answer)
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"answer": str(e)})
