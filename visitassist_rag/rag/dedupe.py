# Deduplicate snippets by normalized text
import re
def dedupe_snippets(snippets):
    seen = set()
    out = []
    for s in snippets:
        key = re.sub(r"\s+", " ", s["text"].lower()).strip() if isinstance(s, dict) else re.sub(r"\s+", " ", s.text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
