from visitassist_rag.rag.retrieval import pinecone_query
from visitassist_rag.rag.rerank import llm_rerank
from visitassist_rag.rag.grounding import grounded_answer
from visitassist_rag.settings import settings
from visitassist_rag.rag.dedupe import dedupe_snippets
from visitassist_rag.rag.ingest import fallback_kb_id
from visitassist_rag.models.schemas import QueryRequest, QueryResponse, Snippet

import re

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


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _looks_like_pdf_table_or_toc(text: str) -> bool:
    """Heuristic filter for PDF-extracted noise (tables/TOC/labels).

    Intentionally conservative: it should remove obvious garbage, not semantics.
    """
    if not text:
        return True
    t = text.strip()
    if len(t) < 40:
        return False

    upper = t.upper()
    if "SUMÁRIO" in upper or "SUMARIO" in upper or "APRESENTAÇÃO" in upper or "APRESENTACAO" in upper:
        # Often the PDF TOC; usually not useful as grounding.
        return True

    # If it's mostly digits/punctuation and has lots of line breaks, it's likely a table/chart.
    digits = sum(ch.isdigit() for ch in t)
    letters = sum(ch.isalpha() for ch in t)
    newlines = t.count("\n")
    denom = max(1, digits + letters)
    digit_ratio = digits / denom

    if newlines >= 10 and digit_ratio >= 0.30:
        return True

    # Many very short lines is also a strong signal of tables/axis labels.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 15:
        short_lines = sum(1 for ln in lines if len(ln) <= 12)
        if (short_lines / len(lines)) >= 0.55 and digit_ratio >= 0.15:
            return True

    # If there are almost no letters, it's not good grounding text.
    if letters < 20 and digits > 50:
        return True

    return False


def _pick_grounding_candidates(question: str, ranked: list[dict], max_sources: int = 6) -> list[dict]:
    """Pick the best candidates to pass into grounding.

    Goals:
    - Keep grounding context small and relevant.
    - Prefer fine chunks over large section chunks.
    - Drop obvious PDF noise (tables/TOC).
    """
    picked: list[dict] = []
    seen_text: set[str] = set()
    fine_sections: set[str] = set()

    # Prefer fine chunks early without completely ignoring rerank order.
    def type_priority(c: dict) -> int:
        ct = (c.get("metadata", {}) or {}).get("chunk_type", "")
        return 0 if ct == "fine" else (1 if ct == "summary" else 2)

    ranked2 = sorted(list(ranked), key=lambda c: (type_priority(c), ranked.index(c)))

    for c in ranked2:
        md = c.get("metadata", {}) or {}
        text = md.get("chunk_text", "") or ""
        chunk_type = md.get("chunk_type", "")
        section_id = md.get("section_id", "")

        if chunk_type == "section" and section_id and section_id in fine_sections:
            continue

        if _looks_like_pdf_table_or_toc(text):
            continue

        key = _normalize_for_dedupe(text)
        if key in seen_text:
            continue
        seen_text.add(key)

        if chunk_type == "fine" and section_id:
            fine_sections.add(section_id)

        picked.append(c)
        if len(picked) >= max_sources:
            break

    # Fallback: if we filtered too aggressively, keep the top reranked items.
    if not picked:
        return ranked[:max_sources]

    return picked

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

    # Sort candidates by doc_year descending (latest first)
    def get_year(c):
        meta = c.get('metadata', {}) if isinstance(c, dict) else getattr(c, 'metadata', {})
        return meta.get('doc_year', 0) or 0
    cands = sorted(cands, key=get_year, reverse=True)
    cands = dedupe_snippets(cands)

    # Rerank and keep only the best few chunks before grounding.
    # This reduces noisy sources (tables/TOC/etc.) and improves answer quality.
    # If reranking fails for any reason, fall back to the original order.
    try:
        ranked = llm_rerank(question, cands, top_n=12)
    except Exception:
        ranked = cands[:12]

    grounding_cands = _pick_grounding_candidates(question, ranked, max_sources=6)
    answer, snippets, trace = grounded_answer(question, grounding_cands, mode=mode, debug=debug)

    # Map each candidate to a Snippet object
    from visitassist_rag.models.schemas import Snippet
    allowed_types = {"event", "place", "coupon", "faq", "paragraph"}
    snippet_objs = []
    max_snippet_chars = 900
    for c in snippets:
        meta = c.get('metadata', {}) if isinstance(c, dict) else getattr(c, 'metadata', {})
        chunk_type = meta.get('chunk_type', 'paragraph')
        snippet_type = chunk_type if chunk_type in allowed_types else 'paragraph'
        full_text = meta.get('chunk_text', '')
        preview_text = full_text
        if isinstance(preview_text, str) and len(preview_text) > max_snippet_chars:
            preview_text = preview_text[:max_snippet_chars].rstrip() + "…"
        snippet_objs.append(Snippet(
            type=snippet_type,
            title=meta.get('doc_title', ''),
            text=preview_text or "",
            source=meta
        ))

    return QueryResponse(answer=answer, snippets=snippet_objs, debug=trace if debug else None)
