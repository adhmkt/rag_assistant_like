import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class UrlPreview:
    url: str
    final_url: str
    title: str
    text: str
    content_hash: str
    char_count: int
    word_count: int
    link_count: int
    link_density: float
    warnings: list[str]


class UrlIngestError(RuntimeError):
    pass


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_tracking_params(url: str) -> str:
    # Keep it simple and deterministic; we don't want to get cute with URL parsing.
    # This strips common tracking params but preserves other query params.
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

    parts = urlsplit(url)
    q = parse_qsl(parts.query, keep_blank_values=True)
    drop_prefixes = ("utm_",)
    drop_exact = {"gclid", "fbclid", "mc_cid", "mc_eid"}
    q2 = [(k, v) for (k, v) in q if not k.lower().startswith(drop_prefixes) and k.lower() not in drop_exact]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q2), parts.fragment))


def fetch_url(url: str, *, timeout_s: float = 15.0, user_agent: Optional[str] = None) -> tuple[str, str]:
    if not url or not isinstance(url, str):
        raise UrlIngestError("Missing URL")

    url = url.strip()
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        raise UrlIngestError("URL must start with http:// or https://")

    # Some sites block default python user agents. Use a browser-like UA by default,
    # while keeping behavior deterministic.
    default_ua = os.getenv(
        "VISITASSIST_URL_INGEST_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    )
    accept_lang = os.getenv("VISITASSIST_URL_INGEST_ACCEPT_LANGUAGE", "pt-BR,pt;q=0.9,en;q=0.6")

    headers = {
        "User-Agent": user_agent or default_ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": accept_lang,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }

    # A Referer header sometimes helps with simple anti-bot rules.
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        headers.setdefault("Referer", f"{parts.scheme}://{parts.netloc}/")
    except Exception:
        pass

    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
    except requests.RequestException as e:
        raise UrlIngestError(f"Failed to fetch URL: {e}")

    if resp.status_code >= 400:
        msg = f"URL fetch failed with HTTP {resp.status_code}"
        if resp.status_code in (401, 403):
            msg += ". The site may block automated requests. Try another URL, use a page export (PDF), or ingest copied text instead."
        raise UrlIngestError(msg)

    ctype = (resp.headers.get("content-type") or "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        # Some servers omit content-type; allow empty but block obvious binaries.
        if ctype and not ("text/" in ctype or "xml" in ctype or "json" in ctype):
            raise UrlIngestError(f"Unsupported content-type: {ctype}")

    html = resp.text or ""
    final_url = _strip_tracking_params(str(resp.url))
    return html, final_url


def extract_main_text(html: str) -> tuple[str, str, int]:
    """Extract (title, text, link_count) from HTML.

    Deterministic and intentionally conservative.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:  # pragma: no cover
        raise UrlIngestError(
            "URL ingestion requires beautifulsoup4. Install it with: pip install beautifulsoup4"
        ) from e

    soup = BeautifulSoup(html or "", "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = _normalize_ws(soup.title.string)

    # Remove obvious noise.
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()

    # Remove common layout containers.
    for name in ["nav", "footer", "header", "aside", "form"]:
        for tag in soup.find_all(name):
            tag.decompose()

    # Remove elements by class/id heuristics.
    noise_markers = [
        "cookie",
        "consent",
        "banner",
        "modal",
        "popup",
        "subscribe",
        "newsletter",
        "nav",
        "footer",
        "header",
        "breadcrumbs",
        "breadcrumb",
        "menu",
        "sidebar",
    ]

    def looks_noisy(val: str) -> bool:
        v = (val or "").lower()
        return any(m in v for m in noise_markers)

    for tag in soup.find_all(True):
        tid = tag.get("id")
        if isinstance(tid, str) and looks_noisy(tid):
            tag.decompose()
            continue
        cls = tag.get("class")
        if isinstance(cls, list) and any(isinstance(c, str) and looks_noisy(c) for c in cls):
            tag.decompose()

    link_count = len(soup.find_all("a"))

    # Prefer <main> if present.
    main = soup.find("main")
    root = main if main is not None else soup.body if soup.body is not None else soup

    text = root.get_text("\n", strip=True) if root else ""
    # Collapse excessive blank lines.
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    text = "\n".join(lines)

    return title, text, link_count


def build_url_preview(
    url: str,
    *,
    timeout_s: float = 15.0,
    min_chars: int = 500,
    max_chars: int = 200_000,
) -> UrlPreview:
    html, final_url = fetch_url(url, timeout_s=timeout_s)
    title, text, link_count = extract_main_text(html)

    warnings: list[str] = []
    if not title:
        warnings.append("Missing <title>; using URL as title.")

    char_count = len(text)
    if char_count < min_chars:
        raise UrlIngestError(
            f"Extracted content is too small ({char_count} chars). This URL may be an index/login/landing page or blocked by the site."
        )

    if char_count > max_chars:
        text = text[:max_chars].rstrip() + "…"
        char_count = len(text)
        warnings.append(f"Content was truncated to {max_chars} characters.")

    words = re.findall(r"\w+", text, flags=re.UNICODE)
    word_count = len(words)
    link_density = (link_count / max(1, word_count))

    # Heuristic: link-heavy pages are often category/index pages with low semantic value.
    if link_density >= 0.12 and link_count >= 25:
        warnings.append(
            "High link density detected; this may be an index/category page and can add noise to the KB. Consider ingesting a more specific page."
        )

    # Content hash for dedupe/audit.
    content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    return UrlPreview(
        url=_strip_tracking_params(url.strip()),
        final_url=final_url,
        title=title or final_url,
        text=text,
        content_hash=content_hash,
        char_count=char_count,
        word_count=word_count,
        link_count=link_count,
        link_density=link_density,
        warnings=warnings,
    )
