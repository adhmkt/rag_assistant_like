from visitassist_rag.rag.retrieval import pinecone_query
from visitassist_rag.rag.rerank import llm_rerank
from visitassist_rag.rag.grounding import grounded_answer
from visitassist_rag.settings import settings
from visitassist_rag.rag.dedupe import dedupe_snippets
from visitassist_rag.rag.ingest import fallback_kb_id
from visitassist_rag.models.schemas import QueryRequest, QueryResponse, Snippet

import re
import time

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


def _filter_by_answer_citations(answer: str, snippets: list[dict]) -> list[dict]:
    """Keep only snippets that are actually cited in the answer.

    The grounding prompt asks the model to cite sources like [S1], [S2], ...
    This makes the returned snippet list tighter and avoids including unused context.
    """
    if not answer or not snippets:
        return snippets

    cited = {int(m) for m in re.findall(r"\[S(\d+)\]", answer) if m.isdigit()}
    if not cited:
        return snippets

    out: list[dict] = []
    for i, s in enumerate(snippets, start=1):
        if i in cited:
            out.append(s)

    return out or snippets


def _ensure_citation_footer(answer: str, language: str) -> str:
    """Normalize citations so the answer ends with a consistent footer.

    If the model emits inline citations (e.g., "... [S1]."), we strip them and
    append a final footer line:
      - Portuguese: "Fonte: [S1]"
      - Other:      "Sources: [S1]"

    If no citations exist, the answer is returned unchanged.
    """
    if not answer:
        return answer

    # Capture citations in order of appearance (unique).
    found = re.findall(r"\[S(\d+)\]", answer)
    if not found:
        return answer

    seen: set[str] = set()
    citations: list[str] = []
    for n in found:
        if n not in seen:
            seen.add(n)
            citations.append(f"[S{n}]")

    # Remove any existing footer lines to avoid duplicates.
    lines = [ln.rstrip() for ln in answer.strip().splitlines()]
    lines = [ln for ln in lines if not re.match(r"^(fonte|sources)\s*:\s*", ln.strip(), flags=re.IGNORECASE)]
    body = "\n".join(lines).strip()

    # Strip inline citations.
    body = re.sub(r"\s*\[S\d+\]", "", body).strip()
    # Clean up stray spaces before punctuation.
    body = re.sub(r"\s+([\.,;:!?])", r"\1", body)

    label = "Fonte" if (language or "").lower().startswith("pt") else "Sources"
    footer = f"{label}: " + ", ".join(citations)
    if body:
        return body.rstrip() + "\n" + footer
    return footer


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


def _clean_pdf_table_preview(question: str, text: str) -> str:
    """Make PDF-extracted table/list snippets more readable for the API preview.

    Notes:
    - This only affects the `Snippet.text` preview, never the stored `source.chunk_text`.
    - The cleanup is intentionally conservative and should not remove key evidence.
    """
    if not text:
        return text

    q = (question or "").lower()
    keep_totals = any(k in q for k in ["total", "dreno", "drenos"]) and "instrument" in q

    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return text

    # Drop common numeric header/row label like "10,00" when it precedes real text.
    if len(lines) >= 2 and re.fullmatch(r"\d{1,4}([.,]\d{1,4})", lines[0]) and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", lines[1]):
        lines = lines[1:]

    # Optionally drop trailing totals for non-total questions.
    if not keep_totals:
        drop_markers = {"total de instrumentos", "total de drenos"}
        filtered: list[str] = []
        i = 0
        while i < len(lines):
            low = lines[i].lower()
            if low in drop_markers:
                # Drop marker line and a following numeric line if present.
                i += 1
                if i < len(lines) and re.fullmatch(r"[\d.]+", lines[i]):
                    i += 1
                continue
            filtered.append(lines[i])
            i += 1
        lines = filtered

    # Reflow soft-wrapped lines common in PDF text extraction.
    joined: list[str] = []
    for ln in lines:
        if not joined:
            joined.append(ln)
            continue

        prev = joined[-1]
        ln_low = ln.lower()
        prev_low = prev.lower()

        starts_like_continuation = (
            ln.startswith("(")
            or ln_low.startswith(("no ", "na ", "de ", "do ", "da ", "dos ", "das ", "em ", "para ", "por "))
            or (len(ln.split()) <= 2 and ln[:1].islower())
        )
        ends_like_incomplete = prev.endswith(",") or prev_low.endswith((" de", " da", " do"))

        if starts_like_continuation or ends_like_incomplete:
            joined[-1] = (prev.rstrip(" ,") + " " + ln).strip()
        else:
            joined.append(ln)

    return "\n".join(joined).strip() or text


def _pick_grounding_candidates(question: str, ranked: list[dict], max_sources: int = 6, min_sources: int = 2) -> list[dict]:
    """Pick the best candidates to pass into grounding.

    Goals:
    - Keep grounding context small and relevant.
    - Prefer fine chunks over large section chunks.
    - Drop obvious PDF noise (tables/TOC).
    """
    picked: list[dict] = []
    seen_text: set[str] = set()
    fine_sections: set[str] = set()

    # Dynamic relevance gating using Pinecone similarity scores.
    # We keep at least `min_sources`, then require candidates to be reasonably close
    # to the best score to avoid pulling in unrelated table-like chunks.
    scores = []
    for c in ranked:
        try:
            scores.append(float(c.get("score", 0) or 0))
        except Exception:
            scores.append(0.0)
    max_score = max(scores) if scores else 0.0
    # Two-stage gate: allow a small set of sources, then become stricter to avoid
    # drifting into loosely-related PDF chunks.
    score_floor_loose = max_score * 0.90 if max_score > 0 else 0.0
    score_floor_strict = max_score * 0.95 if max_score > 0 else 0.0

    for c in ranked:
        md = c.get("metadata", {}) or {}
        text = md.get("chunk_text", "") or ""
        chunk_type = md.get("chunk_type", "")
        section_id = md.get("section_id", "")

        try:
            score = float(c.get("score", 0) or 0)
        except Exception:
            score = 0.0

        # After we have enough context, only keep near-top matches.
        if len(picked) >= min_sources and score < score_floor_strict:
            continue
        # Even before min_sources, drop very low scoring outliers.
        if len(picked) < min_sources and score < score_floor_loose:
            continue

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

    # Prune redundant long chunks that contain a smaller picked chunk verbatim.
    # This commonly happens when a large "section" chunk includes a concise "fine" chunk.
    texts = []
    for c in picked:
        md = c.get("metadata", {}) or {}
        texts.append((md.get("chunk_text", "") or "", c))

    keep = []
    for i, (ti, ci) in enumerate(texts):
        ni = _normalize_for_dedupe(ti)
        if not ni:
            continue
        redundant = False
        for j, (tj, _cj) in enumerate(texts):
            if i == j:
                continue
            nj = _normalize_for_dedupe(tj)
            if not nj:
                continue
            # If this chunk is much longer and contains another picked chunk, drop it.
            if len(ni) > len(nj) * 3 and nj in ni:
                redundant = True
                break
        if not redundant:
            keep.append(ci)

    # Keep ordering stable (preserve the original picked order)
    keep_ids = {c.get("id") for c in keep}
    keep = [c for c in picked if c.get("id") in keep_ids]

    return keep


def _candidate_debug_row(c: dict) -> dict:
    md = c.get("metadata", {}) or {}
    return {
        "id": c.get("id"),
        "score": c.get("score"),
        "chunk_type": md.get("chunk_type"),
        "doc_title": md.get("doc_title"),
        "section_path": md.get("section_path"),
        "chunk_index": md.get("chunk_index"),
        "source_type": md.get("source_type"),
    }

def rag_query(question: str, language: str = "pt", mode: str = "tourist_chat", kb_id: str = None, debug: bool = False, debug_no_filter: bool = False, less_strict: bool = False, **kwargs):
    t0 = time.perf_counter()
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
    t_retr0 = time.perf_counter()
    c_summary = pinecone_query(question, 1,  build_filter(kb_id, language, "summary", source_types, debug_no_filter, less_strict))
    c_section = pinecone_query(question, 8,  build_filter(kb_id, language, "section", source_types, debug_no_filter, less_strict))
    c_fine = pinecone_query(question, 18, build_filter(kb_id, language, "fine", source_types, debug_no_filter, less_strict))
    cands += c_summary
    cands += c_section
    cands += c_fine
    t_retr1 = time.perf_counter()

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
    t_rer0 = time.perf_counter()
    try:
        ranked = llm_rerank(question, cands, top_n=12)
        rerank_error = None
    except Exception as e:
        ranked = cands[:12]
        rerank_error = repr(e)
    t_rer1 = time.perf_counter()

    # If the reranker returns an empty list (e.g., bad/mismatched ids), fall back.
    if not ranked and cands:
        ranked = cands[:12]

    grounding_cands = _pick_grounding_candidates(question, ranked, max_sources=4)

    # Pre-compute score diagnostics for debug.
    ranked_scores: list[float] = []
    for c in ranked:
        try:
            ranked_scores.append(float(c.get("score", 0) or 0))
        except Exception:
            ranked_scores.append(0.0)
    ranked_max_score = max(ranked_scores) if ranked_scores else 0.0
    ranked_score_floor_loose = ranked_max_score * 0.90 if ranked_max_score > 0 else 0.0
    ranked_score_floor_strict = ranked_max_score * 0.95 if ranked_max_score > 0 else 0.0

    # Never ask the LLM to answer without sources.
    t_gnd0 = None
    t_gnd1 = None
    if not grounding_cands:
        answer = "Não encontrei informações relevantes na base para responder com segurança."
        snippets = []
        trace = {"reason": "no_sources"} if debug else None
    else:
        t_gnd0 = time.perf_counter()
        answer, snippets, trace = grounded_answer(question, grounding_cands, mode=mode, debug=debug)
        t_gnd1 = time.perf_counter()

    # Enforce consistent citation formatting (footer) for API consumers.
    answer = _ensure_citation_footer(answer, language)

    # Tighten snippet list to only what was cited.
    snippets = _filter_by_answer_citations(answer, snippets)

    # Merge structured debug info (without leaking full chunk text).
    if debug:
        dbg = trace if isinstance(trace, dict) else {}
        dbg.update({
            "kb_id": kb_id,
            "language": language,
            "mode": mode,
            "filters": {
                "debug_no_filter": debug_no_filter,
                "less_strict": less_strict,
                "source_types": source_types,
            },
            "retrieval": {
                "counts": {
                    "summary": len(c_summary),
                    "section": len(c_section),
                    "fine": len(c_fine),
                    "total": len(c_summary) + len(c_section) + len(c_fine),
                },
            },
            "rerank": {
                "error": rerank_error,
                "ranked_max_score": ranked_max_score,
                "score_floor_loose": ranked_score_floor_loose,
                "score_floor_strict": ranked_score_floor_strict,
            },
            "candidates": {
                "top_pre_rerank": [_candidate_debug_row(c) for c in cands[:8]],
                "top_reranked": [_candidate_debug_row(c) for c in ranked[:8]],
                "grounding_selected": [_candidate_debug_row(c) for c in grounding_cands],
            },
            "timings_ms": {
                "retrieval": round((t_retr1 - t_retr0) * 1000.0, 2),
                "rerank": round((t_rer1 - t_rer0) * 1000.0, 2),
                "grounding": None if (t_gnd0 is None or t_gnd1 is None) else round((t_gnd1 - t_gnd0) * 1000.0, 2),
                "total": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        })
        trace = dbg

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

        # Improve readability for PDF-extracted table-like chunks.
        if isinstance(preview_text, str) and preview_text.count("\n") >= 6:
            preview_text = _clean_pdf_table_preview(question, preview_text)

        if isinstance(preview_text, str) and len(preview_text) > max_snippet_chars:
            preview_text = preview_text[:max_snippet_chars].rstrip() + "…"
        snippet_objs.append(Snippet(
            type=snippet_type,
            title=meta.get('doc_title', ''),
            text=preview_text or "",
            source=meta
        ))

    return QueryResponse(answer=answer, snippets=snippet_objs, debug=trace if debug else None)

