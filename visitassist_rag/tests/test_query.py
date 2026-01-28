def test_query():
    from visitassist_rag.rag.engine import rag_query
    resp = rag_query("What is the weather?", kb_id="curitiba__default")
    assert hasattr(resp, "answer")


def test_ensure_citation_footer_pt():
    from visitassist_rag.rag.engine import _ensure_citation_footer

    out = _ensure_citation_footer("O total é 3.055 e 5.365 [S1].", "pt")
    assert "Fonte: [S1]" in out
    assert "[S1]." not in out


def test_ensure_citation_footer_multi_sources():
    from visitassist_rag.rag.engine import _ensure_citation_footer

    out = _ensure_citation_footer("A. [S2]\nB. [S1]", "pt")
    assert out.endswith("Fonte: [S2], [S1]")
