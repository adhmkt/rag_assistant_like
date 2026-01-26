from visitassist_rag.rag.retrieval import pinecone_query
from visitassist_rag.rag.rerank import llm_rerank
from visitassist_rag.rag.grounding import grounded_answer
from visitassist_rag.settings import settings
from visitassist_rag.rag.dedupe import dedupe_snippets
from visitassist_rag.rag.ingest import fallback_kb_id
from visitassist_rag.models.schemas import QueryRequest, QueryResponse, Snippet

# Helper to build Pinecone filter

def build_filter(kb_id: str, lang: str, chunk_type: str, source_types: list[str] | None = None):
    flt = {"kb_id": kb_id, "language": lang, "chunk_type": chunk_type}
    if source_types:
        flt["source_type"] = {"$in": source_types}
    return flt

def rag_query(question: str, language: str = "pt", mode: str = "tourist_chat", kb_id: str = None, debug: bool = False, **kwargs):
    # Mode-based source_type selection
    source_types = None
    if mode == "events":
        source_types = ["events"]
    elif mode == "directory":
        source_types = ["directory"]
    elif mode == "coupons":
        source_types = ["coupon"]
    elif mode == "faq_first":
        source_types = ["faq"]

    cands = []
    cands += pinecone_query(question, 1,  build_filter(kb_id, language, "summary", source_types))
    cands += pinecone_query(question, 8,  build_filter(kb_id, language, "section", source_types))
    cands += pinecone_query(question, 18, build_filter(kb_id, language, "fine", source_types))

    # Fallback to city master KB if empty
    if not cands and fallback_kb_id(kb_id):
        kb_id2 = fallback_kb_id(kb_id)
        cands += pinecone_query(question, 1,  build_filter(kb_id2, language, "summary", source_types))
        cands += pinecone_query(question, 8,  build_filter(kb_id2, language, "section", source_types))
        cands += pinecone_query(question, 18, build_filter(kb_id2, language, "fine", source_types))

    cands = dedupe_snippets(cands)
    answer, snippets, trace = grounded_answer(question, cands, mode=mode, debug=debug)
    return QueryResponse(answer=answer, snippets=snippets, debug=trace if debug else None)
