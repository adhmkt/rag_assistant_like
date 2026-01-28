from fastapi import FastAPI
from visitassist_rag.api.routes_ingest import router as ingest_router
from visitassist_rag.api.routes_query import router as query_router
from visitassist_rag.api.routes_admin import router as admin_router

app = FastAPI(title="VisitAssist RAG Engine")


@app.get("/health")
def health():
	return {"status": "ok"}

app.include_router(ingest_router, prefix="/v1")
app.include_router(query_router, prefix="/v1")
app.include_router(admin_router, prefix="/v1")
