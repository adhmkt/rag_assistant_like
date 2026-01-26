import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from supabase import create_client
from pinecone import Pinecone
from openai import OpenAI
import tiktoken

load_dotenv()

# --------- Clients ----------
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX"]
index = pc.Index(INDEX_NAME)

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

KB_ID = os.environ.get("KB_ID", "default-kb")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "pt")

EMBED_MODEL = "text-embedding-3-large"  # dim=1024
TOKENIZER = tiktoken.get_encoding("cl100k_base")
INGEST_VERSION = "v1"

# --------- Data types ----------
@dataclass
class Chunk:
    doc_id: str
    section_id: Optional[str]
    chunk_type: str  # summary|section|fine
    chunk_index: int
    chunk_text: str
    section_path: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

# --------- Utilities ----------
def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    # Basic paragraph split
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts

def chunk_by_tokens(paragraphs: List[str], target_tokens: int, overlap_tokens: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    for p in paragraphs:
        t = count_tokens(p)
        if t > target_tokens:
            # Hard split long paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                st = count_tokens(s)
                if buf_tokens + st > target_tokens and buf:
                    chunks.append(" ".join(buf).strip())
                    # overlap
                    if overlap_tokens > 0:
                        chunks[-1] = chunks[-1]  # keep
                    buf = []
                    buf_tokens = 0
                buf.append(s)
                buf_tokens += st
            continue

        if buf_tokens + t > target_tokens and buf:
            chunks.append(" ".join(buf).strip())
            # overlap: take tail tokens from last chunk
            if overlap_tokens > 0:
                tail = take_tail_tokens(chunks[-1], overlap_tokens)
                buf = [tail] if tail else []
                buf_tokens = count_tokens(tail) if tail else 0
            else:
                buf = []
                buf_tokens = 0

        buf.append(p)
        buf_tokens += t

    if buf:
        chunks.append(" ".join(buf).strip())

    return [c for c in chunks if c]

def take_tail_tokens(text: str, tail_tokens: int) -> str:
    toks = TOKENIZER.encode(text)
    tail = toks[-tail_tokens:] if len(toks) > tail_tokens else toks
    return TOKENIZER.decode(tail).strip()

# --------- Structure parsing (simple but works) ----------
def build_sections(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (section_path, section_text).
    Heuristic:
      - If markdown-style headings exist (#, ##, ###), use them.
      - Otherwise treat whole doc as one section.
    """
    lines = text.split("\n")
    has_md_headings = any(re.match(r"^\s{0,3}#{1,6}\s+\S+", ln) for ln in lines)

    if not has_md_headings:
        return [("Document", text)]

    sections: List[Tuple[str, List[str]]] = []
    current_path = "Document"
    buf: List[str] = []

    for ln in lines:
        m = re.match(r"^\s{0,3}(#{1,6})\s+(.+)$", ln)
        if m:
            # flush previous
            if buf:
                sections.append((current_path, buf))
                buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # naive path: use ">" with heading level
            current_path = f"H{level} > {title}"
        else:
            buf.append(ln)

    if buf:
        sections.append((current_path, buf))

    # join + normalize
    out: List[Tuple[str, str]] = []
    for path, blines in sections:
        st = normalize_ws("\n".join(blines))
        if st:
            out.append((path, st))
    return out if out else [("Document", text)]

# --------- Summarize doc (for summary chunks) ----------
def make_doc_summary(title: str, text: str, lang: str) -> str:
    # Keep it short and factual; this is used for retrieval, not user-facing.
    prompt = f"""
You create a short retrieval-oriented summary for RAG.
Language: {lang}
Title: {title}

Rules:
- 6 to 10 bullet points max.
- Include key entities, scope, and main sections/topics.
- Avoid fluff, no marketing.
- No citations.

Text:
{text[:12000]}
""".strip()

    resp = oai.chat.completions.create(
        model="gpt-4.1-mini",  # choose your preferred chat model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# --------- Embeddings ----------
def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = oai.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]

# --------- Supabase writes ----------
def upsert_doc(doc_id: str, kb_id: str, title: str, source_type: str, source_uri: str, lang: str):
    sb.table("rag_docs").upsert({
        "doc_id": doc_id,
        "kb_id": kb_id,
        "title": title,
        "source_type": source_type,
        "source_uri": source_uri,
        "language": lang
    }).execute()

def insert_section(section_id: str, doc_id: str, section_path: str, section_index: int, section_text: str):
    sb.table("rag_sections").insert({
        "section_id": section_id,
        "doc_id": doc_id,
        "section_path": section_path,
        "section_index": section_index,
        "section_text": section_text
    }).execute()

def insert_chunk(chunk_id: str, ch: Chunk):
    sb.table("rag_chunks").insert({
        "chunk_id": chunk_id,
        "doc_id": ch.doc_id,
        "section_id": ch.section_id,
        "chunk_type": ch.chunk_type,
        "chunk_index": ch.chunk_index,
        "start_char": ch.start_char,
        "end_char": ch.end_char,
        "chunk_text": ch.chunk_text,
        "ingest_version": INGEST_VERSION
    }).execute()

# --------- Pinecone upsert ----------
def pinecone_upsert(vectors: List[Tuple[str, List[float], Dict[str, Any]]]):
    # vectors: list of (id, embedding, metadata)
    # Upsert in batches
    B = 100
    for i in range(0, len(vectors), B):
        batch = vectors[i:i+B]
        index.upsert(vectors=batch)

# --------- Ingest a text document ----------
def ingest_text_document(
    title: str,
    text: str,
    source_type: str = "txt",
    source_uri: str = "",
    kb_id: str = KB_ID,
    lang: str = DEFAULT_LANG,
):
    text = normalize_ws(text)
    doc_id = str(uuid.uuid4())

    # 1) store doc metadata
    upsert_doc(doc_id, kb_id, title, source_type, source_uri, lang)

    # 2) build sections
    sec_pairs = build_sections(text)

    # 3) doc summary chunk
    summary_text = make_doc_summary(title, text, lang)
    summary_chunk = Chunk(
        doc_id=doc_id,
        section_id=None,
        chunk_type="summary",
        chunk_index=0,
        chunk_text=summary_text,
        section_path="Document Summary",
    )

    # 4) section chunks + fine chunks
    all_chunks: List[Chunk] = [summary_chunk]
    section_ids: List[str] = []

    for sidx, (spath, stext) in enumerate(sec_pairs):
        section_id = str(uuid.uuid4())
        section_ids.append(section_id)
        insert_section(section_id, doc_id, spath, sidx, stext)

        # Section-level chunk (coarse)
        if count_tokens(stext) <= 900:
            section_chunks = [stext]
        else:
            section_chunks = chunk_by_tokens(
                split_paragraphs(stext),
                target_tokens=900,
                overlap_tokens=100
            )
        # Usually 1; if long section, multiple "section pages"
        for j, sc in enumerate(section_chunks):
            all_chunks.append(Chunk(
                doc_id=doc_id,
                section_id=section_id,
                chunk_type="section",
                chunk_index=j,
                chunk_text=sc,
                section_path=spath
            ))

        # Fine chunks
        fine_chunks = chunk_by_tokens(split_paragraphs(stext), target_tokens=180, overlap_tokens=30)
        for k, fc in enumerate(fine_chunks):
            all_chunks.append(Chunk(
                doc_id=doc_id,
                section_id=section_id,
                chunk_type="fine",
                chunk_index=k,
                chunk_text=fc,
                section_path=spath
            ))

    # 5) write chunks to Supabase + embeddings to Pinecone
    # Prepare texts in same order to embed efficiently
    texts = [c.chunk_text for c in all_chunks]
    embs = embed_texts(texts)

    pine_vectors: List[Tuple[str, List[float], Dict[str, Any]]] = []

    for idx, (ch, emb) in enumerate(zip(all_chunks, embs)):
        chunk_id = str(uuid.uuid4())
        insert_chunk(chunk_id, ch)

        meta = {
            "kb_id": kb_id,
            "doc_id": ch.doc_id,
            "doc_title": title,
            "source_type": source_type,
            "source_uri": source_uri,
            "language": lang,
            "chunk_type": ch.chunk_type,
            "section_path": ch.section_path or "",
            "section_id": ch.section_id or "",
            "chunk_index": ch.chunk_index,
            "ingest_version": INGEST_VERSION,
        }
        pine_vectors.append((chunk_id, emb, meta))

    pinecone_upsert(pine_vectors)

    return {"doc_id": doc_id, "chunks_indexed": len(all_chunks)}

# --------- Retrieval ----------
def pinecone_query(question: str, top_k: int, flt: Dict[str, Any]):
    q_emb = embed_texts([question])[0]
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True, filter=flt)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches
    out = []
    for m in matches:
        md = m["metadata"] if isinstance(m, dict) else m.metadata
        score = m["score"] if isinstance(m, dict) else m.score
        out.append({"id": m["id"] if isinstance(m, dict) else m.id, "score": score, "metadata": md})
    return out

def dedupe_diversify(cands: List[Dict[str, Any]], per_doc: int = 5, per_section: int = 2, max_total: int = 30):
    # Sort high score first
    cands = sorted(cands, key=lambda x: x["score"], reverse=True)

    doc_counts: Dict[str, int] = {}
    sec_counts: Dict[str, int] = {}
    picked = []

    for c in cands:
        md = c["metadata"]
        doc_id = md.get("doc_id", "")
        sec_id = md.get("section_id", "")
        if doc_counts.get(doc_id, 0) >= per_doc:
            continue
        if sec_id and sec_counts.get(sec_id, 0) >= per_section:
            continue
        picked.append(c)
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        if sec_id:
            sec_counts[sec_id] = sec_counts.get(sec_id, 0) + 1
        if len(picked) >= max_total:
            break

    return picked

# --------- Rerank (LLM) ----------
def llm_rerank(question: str, cands: List[Dict[str, Any]], top_n: int = 8) -> List[Dict[str, Any]]:
    # Pull chunk text from Supabase for rerank
    ids = [c["id"] for c in cands]
    rows = sb.table("rag_chunks").select("chunk_id,chunk_text").in_("chunk_id", ids).execute().data
    text_by_id = {r["chunk_id"]: r["chunk_text"] for r in rows}

    items = []
    for i, c in enumerate(cands):
        cid = c["id"]
        md = c["metadata"]
        preview = (text_by_id.get(cid, "")[:800]).replace("\n", " ")
        items.append(f"{i+1}) id={cid} | type={md.get('chunk_type')} | section={md.get('section_path')}\n{preview}")

    prompt = f"""
You are reranking retrieval candidates for a RAG system.

Question:
{question}

Candidates:
{chr(10).join(items)}

Return ONLY a JSON array of the best ids in order, length {top_n}.
Example: ["id1","id2",...]
""".strip()

    resp = oai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()

    # Safe parse
    import json
    try:
        ordered_ids = json.loads(txt)
        ordered_ids = [x for x in ordered_ids if isinstance(x, str)]
    except Exception:
        # fallback: keep original order
        return cands[:top_n]

    c_by_id = {c["id"]: c for c in cands}
    ranked = [c_by_id[i] for i in ordered_ids if i in c_by_id]
    return ranked[:top_n]

# --------- Context compaction ----------
def compact_context(ranked: List[Dict[str, Any]], max_snippets: int = 6, max_tokens_per_snippet: int = 350):
    # Fetch chunk_text + metadata from Supabase
    ids = [c["id"] for c in ranked]
    rows = sb.table("rag_chunks").select("chunk_id,chunk_text,doc_id,section_id,chunk_type,chunk_index").in_("chunk_id", ids).execute().data
    row_by_id = {r["chunk_id"]: r for r in rows}

    snippets = []
    for c in ranked[:max_snippets]:
        md = c["metadata"]
        r = row_by_id.get(c["id"])
        if not r:
            continue

        txt = r["chunk_text"].strip()
        # Trim to token limit per snippet
        toks = TOKENIZER.encode(txt)
        if len(toks) > max_tokens_per_snippet:
            txt = TOKENIZER.decode(toks[:max_tokens_per_snippet]).strip()

        snippets.append({
            "chunk_id": c["id"],
            "doc_title": md.get("doc_title", ""),
            "section_path": md.get("section_path", ""),
            "chunk_type": md.get("chunk_type", ""),
            "text": txt
        })

    return snippets

# --------- Final grounded answer ----------
def grounded_answer(question: str, snippets: List[Dict[str, Any]], lang: str):
    sources = []
    for i, s in enumerate(snippets, start=1):
        sources.append(
            f"[S{i}] {s['doc_title']} — {s['section_path']} ({s['chunk_type']})\n{s['text']}"
        )

    prompt = f"""
Answer the question using ONLY the sources below.
If the sources do not contain enough information, say so clearly and ask what document or section is missing.

Language: {lang}

Question:
{question}

Sources:
{chr(10).join(sources)}

Rules:
- Cite the minimum number of sources needed to support each statement.
- If multiple sources say the same thing, cite only one.
- Do not invent facts.
- Be concise.
""".strip()

    resp = oai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# --------- Public function: query pipeline ----------
def rag_query(question: str, kb_id: str = KB_ID, lang: str = DEFAULT_LANG):
    # Multi-pass retrieval mirroring Assistants
    cands = []
    cands += pinecone_query(question, top_k=5,  flt={"kb_id": kb_id, "language": lang, "chunk_type": "summary"})
    cands += pinecone_query(question, top_k=10, flt={"kb_id": kb_id, "language": lang, "chunk_type": "section"})
    cands += pinecone_query(question, top_k=20, flt={"kb_id": kb_id, "language": lang, "chunk_type": "fine"})

    merged = dedupe_diversify(cands, per_doc=5, per_section=2, max_total=30)
    ranked = llm_rerank(question, merged, top_n=8)
    snippets = compact_context(ranked, max_snippets=6, max_tokens_per_snippet=350)
    snippets = dedupe_by_text(snippets)
    answer = grounded_answer(question, snippets, lang)

    

    return {"answer": answer, "snippets": snippets}

# --------- Public function: dedupe by text ----------
def dedupe_by_text(snippets):
    seen = set()
    out = []
    for s in snippets:
        key = re.sub(r"\s+", " ", s["text"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
