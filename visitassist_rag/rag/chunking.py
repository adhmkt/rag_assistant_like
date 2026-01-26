# Placeholder for chunking logic
import re
import tiktoken
from typing import List, Tuple

TOKENIZER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))

def normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts

def chunk_by_tokens(paragraphs: List[str], target_tokens: int, overlap_tokens: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    for p in paragraphs:
        t = count_tokens(p)
        if t > target_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                st = count_tokens(s)
                if buf_tokens + st > target_tokens and buf:
                    chunks.append(" ".join(buf).strip())
                    if overlap_tokens > 0:
                        chunks[-1] = chunks[-1]
                    buf = []
                    buf_tokens = 0
                buf.append(s)
                buf_tokens += st
            continue

        if buf_tokens + t > target_tokens and buf:
            chunks.append(" ".join(buf).strip())
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

def build_sections(text: str) -> List[Tuple[str, str]]:
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
            if buf:
                sections.append((current_path, buf))
                buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            current_path = f"H{level} > {title}"
        else:
            buf.append(ln)
    if buf:
        sections.append((current_path, buf))
    out: List[Tuple[str, str]] = []
    for path, blines in sections:
        st = normalize_ws("\n".join(blines))
        if st:
            out.append((path, st))
    return out if out else [("Document", text)]
