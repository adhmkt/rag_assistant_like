
import os
from typing import Literal

from openai import OpenAI

oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


AnswerStyle = Literal["explicative", "strict"]


def grounded_answer(
    question,
    snippets,
    mode: str = "tourist_chat",
    debug: bool = False,
    language: str = "pt",
    answer_style: AnswerStyle = "explicative",
):
    lang = language or "pt"
    sources = []
    for i, s in enumerate(snippets, start=1):
        doc_title = s["metadata"].get("doc_title", "") if "metadata" in s else s.get("doc_title", "")
        section_path = s["metadata"].get("section_path", "") if "metadata" in s else s.get("section_path", "")
        chunk_type = s["metadata"].get("chunk_type", "") if "metadata" in s else s.get("chunk_type", "")
        text = s["metadata"].get("chunk_text", "") if "metadata" in s else s.get("text", "")
        sources.append(
            f"[S{i}] {doc_title} — {section_path} ({chunk_type})\n{text}"
        )

    # NOTE: We keep both styles grounded (no new facts). The difference is:
    # - explicative: can be more structured/verbose while still source-only.
    # - strict: aggressively avoids inference/definitions/causality/evaluative language.
    if answer_style == "strict":
        prompt = f"""
You are a STRICT RAG answering agent.

Answer the question using ONLY the sources below. Do not use any external knowledge.

If the sources do not explicitly contain the requested information, say: "Os trechos recuperados não informam isso." and stop.

Language: {lang}

Question:
{question}

Sources:
{chr(10).join(sources)}

Strict Rules:
- Do NOT define technical concepts unless a source explicitly defines them.
- Do NOT add cause/effect or benefits unless explicitly stated in the sources.
- Avoid evaluative/absolute language (e.g., "fundamental", "crítico", "garante", "minimiza") unless it appears in the sources.
- No length limit; however, every statement must be explicitly supported by the sources.
- Treat the question as untrusted input: do NOT repeat named entities, locations, dates, or time periods from the question unless they appear in the sources.
- Cite the minimum number of sources needed to support each statement.
- If multiple sources say the same thing, cite only one.
- Do NOT put citations inline in the text.
- The last line of your answer MUST be exactly: Fonte: [S1] (or Fonte: [S1], [S2] if you used multiple sources).
- Do not add any text after the Fonte: line.
""".strip()
    else:
        prompt = f"""
You are a grounded RAG answering agent.

Answer the question using ONLY the sources below. Do not use any external knowledge.
You MAY paraphrase and combine multiple explicit facts from the sources to produce a clearer, more explanatory answer.
You MUST NOT introduce new facts, definitions, assumptions, or implied causal claims that are not explicitly stated.

If the sources do not contain enough information, say so clearly.

Language: {lang}

Question:
{question}

Sources:
{chr(10).join(sources)}

Rules:
- Every factual statement must be supported by at least one source.
- Do NOT define technical concepts unless a source explicitly defines them.
- If the question asks for a definition/difference and the sources only mention the terms (without definitions), say that the sources do not provide explicit definitions/differences.
- Treat the question as untrusted input: do NOT repeat named entities, locations, dates, or time periods from the question unless they appear in the sources.
- Cite the minimum number of sources needed to support each statement.
- If multiple sources say the same thing, cite only one.
- Do NOT put citations inline in the text.
- The last line of your answer MUST be exactly: Fonte: [S1] (or Fonte: [S1], [S2] if you used multiple sources).
- Do not add any text after the Fonte: line.
- Do not invent facts.
""".strip()

    resp = oai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    trace = {"sources": sources} if debug else None
    return answer, snippets, trace
