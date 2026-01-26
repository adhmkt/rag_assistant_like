
import os
from openai import OpenAI
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def grounded_answer(question, snippets, mode="tourist_chat", debug=False):
    lang = "pt"  # You may want to pass this in
    sources = []
    for i, s in enumerate(snippets, start=1):
        doc_title = s["metadata"].get("doc_title", "") if "metadata" in s else s.get("doc_title", "")
        section_path = s["metadata"].get("section_path", "") if "metadata" in s else s.get("section_path", "")
        chunk_type = s["metadata"].get("chunk_type", "") if "metadata" in s else s.get("chunk_type", "")
        text = s["metadata"].get("chunk_text", "") if "metadata" in s else s.get("text", "")
        sources.append(
            f"[S{i}] {doc_title} — {section_path} ({chunk_type})\n{text}"
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
    answer = resp.choices[0].message.content.strip()
    trace = {"sources": sources} if debug else None
    return answer, snippets, trace
