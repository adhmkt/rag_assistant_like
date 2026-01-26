import os
from openai import OpenAI
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def llm_rerank(question, cands, top_n=8):
    ids = [c["id"] for c in cands]
    items = []
    for i, c in enumerate(cands):
        md = c["metadata"]
        preview = (md.get("chunk_text", "")[:800]).replace("\n", " ") if "chunk_text" in md else ""
        items.append(f"{i+1}) id={c['id']} | type={md.get('chunk_type')} | section={md.get('section_path')}\n{preview}")

    prompt = f"""
You are reranking retrieval candidates for a RAG system.

Question:
{question}

Candidates:
{chr(10).join(items)}

Return ONLY a JSON array of the best ids in order, length {top_n}.
Example: [\"id1\",\"id2\",...]
""".strip()

    resp = oai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()
    import json
    try:
        ordered_ids = json.loads(txt)
        ordered_ids = [x for x in ordered_ids if isinstance(x, str)]
    except Exception:
        return cands[:top_n]
    c_by_id = {c["id"]: c for c in cands}
    ranked = [c_by_id[i] for i in ordered_ids if i in c_by_id]
    return ranked[:top_n]
