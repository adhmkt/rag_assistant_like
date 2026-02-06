from visitassist_rag.rag.embeddings import embed_texts
from visitassist_rag.stores.pinecone_store import query_chunks

def pinecone_query(question, top_k, flt, *, namespace: str | None = None):
    q_emb = embed_texts([question])[0]
    return query_chunks(q_emb, top_k, flt, namespace=namespace)
