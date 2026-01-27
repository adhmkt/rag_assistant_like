from visitassist_rag.stores.supabase_store import upsert_doc, insert_section, insert_chunk
from visitassist_rag.stores.pinecone_store import upsert_chunks
from visitassist_rag.rag.chunking import normalize_ws, build_sections, split_paragraphs, chunk_by_tokens, count_tokens
from visitassist_rag.rag.embeddings import embed_texts
from visitassist_rag.models.schemas import IngestTextRequest, IngestResponse
import uuid

def ingest_text_document(kb_id: str, title: str, text: str, source_type: str, source_uri: str, language: str, **kwargs):
    text = normalize_ws(text)
    doc_id = str(uuid.uuid4())
    upsert_doc(doc_id, kb_id, title, source_type, source_uri, language)
    sec_pairs = build_sections(text)
    all_chunks = []
    section_ids = []
    for sidx, (spath, stext) in enumerate(sec_pairs):
        section_id = str(uuid.uuid4())
        section_ids.append(section_id)
        insert_section(section_id, doc_id, spath, sidx, stext)
        if count_tokens(stext) <= 900:
            section_chunks = [stext]
        else:
            section_chunks = chunk_by_tokens(
                split_paragraphs(stext),
                target_tokens=900,
                overlap_tokens=100
            )
        for j, sc in enumerate(section_chunks):
            all_chunks.append(type('Chunk', (), {
                "doc_id": doc_id,
                "section_id": section_id,
                "chunk_type": "section",
                "chunk_index": j,
                "chunk_text": sc,
                "section_path": spath
            }))
        fine_chunks = chunk_by_tokens(split_paragraphs(stext), target_tokens=180, overlap_tokens=30)
        for k, fc in enumerate(fine_chunks):
            all_chunks.append(type('Chunk', (), {
                "doc_id": doc_id,
                "section_id": section_id,
                "chunk_type": "fine",
                "chunk_index": k,
                "chunk_text": fc,
                "section_path": spath
            }))
    texts = [c.chunk_text for c in all_chunks]
    embs = embed_texts(texts)
    pine_vectors = []
    doc_date = kwargs.get("doc_date")
    doc_year = kwargs.get("doc_year")
    for idx, (ch, emb) in enumerate(zip(all_chunks, embs)):
        chunk_id = str(uuid.uuid4())
        insert_chunk(chunk_id, ch)
        meta = {
            "kb_id": kb_id,
            "doc_id": ch.doc_id,
            "doc_title": title,
            "source_type": source_type,
            "source_uri": source_uri,
            "language": language,
            "chunk_type": ch.chunk_type,
            "section_path": ch.section_path or "",
            "section_id": ch.section_id or "",
            "chunk_index": ch.chunk_index,
            "ingest_version": "v1",
            "chunk_text": ch.chunk_text,
        }
        if doc_date:
            meta["doc_date"] = doc_date
        if doc_year:
            meta["doc_year"] = doc_year
        pine_vectors.append((chunk_id, emb, meta))
    upsert_chunks(pine_vectors)
    return IngestResponse(success=True, doc_id=doc_id)

def fallback_kb_id(kb_id: str):
    if "__" in kb_id and not kb_id.endswith("__default"):
        city = kb_id.split("__")[0]
        return f"{city}__default"
    return None
