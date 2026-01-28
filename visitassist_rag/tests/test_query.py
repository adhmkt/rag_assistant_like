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


def test_get_doc_year_prefers_doc_year_over_date():
    from visitassist_rag.rag.engine import _get_doc_year

    assert _get_doc_year({"doc_year": 2026, "doc_date": "1990-01-01"}) == 2026
    assert _get_doc_year({"doc_date": "1990-01-01"}) == 1990
    assert _get_doc_year({"doc_year": "2001"}) == 2001


def test_sort_newest_first_is_stable_for_equal_years():
    from visitassist_rag.rag.engine import _sort_newest_first

    c1 = {"id": "a", "score": 0.9, "metadata": {"doc_year": 2020}}
    c2 = {"id": "b", "score": 0.8, "metadata": {"doc_year": 2020}}
    c3 = {"id": "c", "score": 0.7, "metadata": {"doc_year": 1990}}
    out = _sort_newest_first([c1, c2, c3])
    assert [c["id"] for c in out] == ["a", "b", "c"]
