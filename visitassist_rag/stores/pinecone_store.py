from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX"]
index = pc.Index(INDEX_NAME)

def upsert_chunks(vectors, *, namespace: str | None = None):
    # vectors: list of (id, embedding, metadata)
    B = 100
    for i in range(0, len(vectors), B):
        batch = vectors[i:i+B]
        if namespace:
            index.upsert(vectors=batch, namespace=namespace)
        else:
            index.upsert(vectors=batch)

def query_chunks(vector, top_k, flt, *, namespace: str | None = None):
    if namespace:
        res = index.query(vector=vector, top_k=top_k, include_metadata=True, filter=flt, namespace=namespace)
    else:
        res = index.query(vector=vector, top_k=top_k, include_metadata=True, filter=flt)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches
    out = []
    for m in matches:
        md = m["metadata"] if isinstance(m, dict) else m.metadata
        score = m["score"] if isinstance(m, dict) else m.score
        out.append({"id": m["id"] if isinstance(m, dict) else m.id, "score": score, "metadata": md})
    return out
