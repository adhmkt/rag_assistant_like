import pytest


def test_extract_main_text_basic():
    from visitassist_rag.rag.url_ingest import extract_main_text

    html = """
    <html>
      <head><title>My Page</title></head>
      <body>
        <header>Header stuff</header>
        <nav><a href='/x'>Menu</a></nav>
        <main>
          <h1>Heading</h1>
          <p>Hello <b>world</b>.</p>
          <p>Second paragraph.</p>
          <a href='https://example.com'>link</a>
        </main>
        <footer>Footer stuff</footer>
      </body>
    </html>
    """

    title, text, link_count = extract_main_text(html)
    assert title == "My Page"
    assert "Header stuff" not in text
    assert "Footer stuff" not in text
    assert "Heading" in text
    assert "Hello" in text
    assert link_count >= 1


def test_build_url_preview_rejects_small_content(monkeypatch):
    from visitassist_rag.rag import url_ingest

    def fake_fetch_url(url: str, *, timeout_s: float = 15.0, user_agent=None):
        return "<html><head><title>T</title></head><body><main>Hi</main></body></html>", url

    monkeypatch.setattr(url_ingest, "fetch_url", fake_fetch_url)

    with pytest.raises(url_ingest.UrlIngestError):
        url_ingest.build_url_preview("https://example.com", min_chars=50)


def test_build_url_preview_emits_link_density_warning(monkeypatch):
    from visitassist_rag.rag import url_ingest

    # Build a page with lots of links and enough text to pass min_chars.
    many_links = "".join([f"<a href='/{i}'>L{i}</a>" for i in range(60)])
    text = "palavra " * 120  # ~960 chars
    html = f"<html><head><title>T</title></head><body><main>{many_links}<p>{text}</p></main></body></html>"

    def fake_fetch_url(url: str, *, timeout_s: float = 15.0, user_agent=None):
        return html, url

    monkeypatch.setattr(url_ingest, "fetch_url", fake_fetch_url)

    preview = url_ingest.build_url_preview("https://example.com", min_chars=200)
    assert any("link density" in w.lower() for w in preview.warnings)


def test_ingest_url_paste_rejects_small_text():
    from fastapi import HTTPException

    from visitassist_rag.api.routes_ingest import ingest_url_paste
    from visitassist_rag.models.schemas import IngestUrlPasteRequest

    req = IngestUrlPasteRequest(url="https://example.com/x", text="too small")
    with pytest.raises(HTTPException):
        ingest_url_paste("kb", req)
