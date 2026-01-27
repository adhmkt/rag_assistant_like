# Deduplicate snippets by normalized text
import re
def dedupe_snippets(snippets):
    seen = set()
    out = []
    for s in snippets:
        if isinstance(s, dict):
            text = s.get("text") or (s.get("metadata", {}).get("chunk_text"))
        else:
            text = getattr(s, "text", None) or getattr(s, "chunk_text", None)
        if not text:
            continue
        key = re.sub(r"\s+", " ", text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
