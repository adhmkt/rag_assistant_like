# Placeholder for Supabase CRUD

def insert_doc_and_chunks(kb_id, title, text, source_type, source_uri, language):
    # TODO: Implement this function
    pass

# --- Supabase CRUD ---
import os
import uuid
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

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

def insert_chunk(chunk_id: str, ch):
    sb.table("rag_chunks").insert({
        "chunk_id": chunk_id,
        "doc_id": ch.doc_id,
        "section_id": ch.section_id,
        "chunk_type": ch.chunk_type,
        "chunk_index": ch.chunk_index,
        "start_char": getattr(ch, 'start_char', None),
        "end_char": getattr(ch, 'end_char', None),
        "chunk_text": ch.chunk_text,
        "ingest_version": "v1"
    }).execute()
