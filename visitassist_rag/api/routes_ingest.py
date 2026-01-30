from fastapi import APIRouter, HTTPException
from visitassist_rag.models.schemas import (
    IngestResponse,
    IngestTextRequest,
    IngestUrlConfirmRequest,
    IngestUrlPasteRequest,
    IngestUrlPreviewRequest,
    IngestUrlPreviewResponse,
)
from visitassist_rag.rag.ingest import ingest_text_document
from visitassist_rag.rag.url_ingest import UrlIngestError, build_url_preview


def _normalize_source_url(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    u = (url or "").strip()
    if not u:
        return u
    parts = urlsplit(u)
    if parts.scheme and parts.netloc:
        q = parse_qsl(parts.query, keep_blank_values=True)
        drop_prefixes = ("utm_",)
        drop_exact = {"gclid", "fbclid", "mc_cid", "mc_eid"}
        q2 = [(k, v) for (k, v) in q if not k.lower().startswith(drop_prefixes) and k.lower() not in drop_exact]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q2), ""))
    return u

router = APIRouter()

@router.post("/kb/{kb_id}/ingest/text", response_model=IngestResponse)
def ingest_text(kb_id: str, req: IngestTextRequest):
    return ingest_text_document(kb_id=kb_id, **req.dict())


@router.post("/kb/{kb_id}/ingest/url/preview", response_model=IngestUrlPreviewResponse)
def ingest_url_preview(kb_id: str, req: IngestUrlPreviewRequest):
    # kb_id is kept in the route for symmetry/audit, but preview does not ingest.
    _ = kb_id
    try:
        preview = build_url_preview(req.url)
    except UrlIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    max_preview_chars = 5000
    text_preview = preview.text if len(preview.text) <= max_preview_chars else (preview.text[:max_preview_chars].rstrip() + "…")
    return IngestUrlPreviewResponse(
        url=preview.url,
        final_url=preview.final_url,
        title=preview.title,
        text_preview=text_preview,
        content_hash=preview.content_hash,
        char_count=preview.char_count,
        word_count=preview.word_count,
        link_count=preview.link_count,
        link_density=preview.link_density,
        warnings=preview.warnings,
    )


@router.post("/kb/{kb_id}/ingest/url/confirm", response_model=IngestResponse)
def ingest_url_confirm(kb_id: str, req: IngestUrlConfirmRequest):
    # Re-fetch + re-extract on the server for a deterministic, auditable ingest.
    try:
        preview = build_url_preview(req.url)
    except UrlIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))

    title = (req.title or "").strip() or preview.title
    return ingest_text_document(
        kb_id=kb_id,
        title=title,
        text=preview.text,
        source_type="url",
        source_uri=preview.final_url,
        language=req.language or "pt",
        doc_date=req.doc_date,
        doc_year=req.doc_year,
    )


@router.post("/kb/{kb_id}/ingest/url/paste", response_model=IngestResponse)
def ingest_url_paste(kb_id: str, req: IngestUrlPasteRequest):
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    text = (req.text or "").strip()
    if len(text) < 200:
        raise HTTPException(status_code=400, detail="Pasted text is too small to ingest")

    title = (req.title or "").strip() or _normalize_source_url(url)
    source_uri = _normalize_source_url(url)

    return ingest_text_document(
        kb_id=kb_id,
        title=title,
        text=text,
        source_type="url_paste",
        source_uri=source_uri,
        language=req.language or "pt",
        doc_date=req.doc_date,
        doc_year=req.doc_year,
    )
