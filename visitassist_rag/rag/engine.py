from visitassist_rag.rag.retrieval import pinecone_query
from visitassist_rag.rag.rerank import llm_rerank
from visitassist_rag.rag.grounding import grounded_answer
from visitassist_rag.settings import settings
from visitassist_rag.rag.dedupe import dedupe_snippets
from visitassist_rag.rag.ingest import fallback_kb_id
from visitassist_rag.models.schemas import QueryRequest, QueryResponse, Snippet

# Helper to build Pinecone filter

def build_filter(kb_id: str, lang: str, chunk_type: str, source_types: list[str] | None = None, debug_no_filter: bool = False, less_strict: bool = False):
    if debug_no_filter:
        return {}
    if less_strict:
        # Only filter by kb_id
        return {"kb_id": kb_id}
    flt = {"kb_id": kb_id, "language": lang, "chunk_type": chunk_type}
    if source_types:
        flt["source_type"] = {"$in": source_types}
    return flt

def rag_query(question: str, language: str = "pt", mode: str = "tourist_chat", kb_id: str = None, debug: bool = False, debug_no_filter: bool = False, less_strict: bool = False, **kwargs):
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
    cands += pinecone_query(question, 1,  build_filter(kb_id, language, "summary", source_types, debug_no_filter, less_strict))
    cands += pinecone_query(question, 8,  build_filter(kb_id, language, "section", source_types, debug_no_filter, less_strict))
    cands += pinecone_query(question, 18, build_filter(kb_id, language, "fine", source_types, debug_no_filter, less_strict))

    # Fallback to city master KB if empty
    if not cands and fallback_kb_id(kb_id):
        kb_id2 = fallback_kb_id(kb_id)
        cands += pinecone_query(question, 1,  build_filter(kb_id2, language, "summary", source_types, debug_no_filter, less_strict))
        cands += pinecone_query(question, 8,  build_filter(kb_id2, language, "section", source_types, debug_no_filter, less_strict))
        cands += pinecone_query(question, 18, build_filter(kb_id2, language, "fine", source_types, debug_no_filter, less_strict))

    cands = dedupe_snippets(cands)
    answer, snippets, trace = grounded_answer(question, cands, mode=mode, debug=debug)

    # Map each candidate to a Snippet object
    from visitassist_rag.models.schemas import Snippet
    allowed_types = {"event", "place", "coupon", "faq", "paragraph"}
    snippet_objs = []
    for c in snippets:
        meta = c.get('metadata', {}) if isinstance(c, dict) else getattr(c, 'metadata', {})
        chunk_type = meta.get('chunk_type', 'paragraph')
        snippet_type = chunk_type if chunk_type in allowed_types else 'paragraph'
        snippet_objs.append(Snippet(
            type=snippet_type,
            title=meta.get('doc_title', ''),
            text=meta.get('chunk_text', ''),
            source=meta
        ))

    return QueryResponse(answer=answer, snippets=snippet_objs, debug=trace if debug else None)
