import os

import pytest


@pytest.mark.skipif(
    os.getenv("VISITASSIST_RUN_INTEGRATION_TESTS") != "1",
    reason="Integration test (requires external services). Set VISITASSIST_RUN_INTEGRATION_TESTS=1 to run.",
)
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


def test_ensure_citation_footer_strips_dangling_fonte_label():
    from visitassist_rag.rag.engine import _ensure_citation_footer

    out = _ensure_citation_footer("X. Fonte:\nFonte: [S1]", "pt")
    assert "\nFonte:\n" not in out
    assert out.endswith("Fonte: [S1]")


def test_get_doc_year_prefers_doc_year_over_date():
    from visitassist_rag.rag.engine import _get_doc_year

    assert _get_doc_year({"doc_year": 2026, "doc_date": "1990-01-01"}) == 2026
    assert _get_doc_year({"doc_date": "1990-01-01"}) == 1990
    assert _get_doc_year({"doc_year": "2001"}) == 2001


def test_sort_newest_first_prefers_full_doc_date_within_year():
    from visitassist_rag.rag.engine import _sort_newest_first

    c1 = {"id": "old", "score": 0.9, "metadata": {"doc_date": "2026-01-01"}}
    c2 = {"id": "new", "score": 0.8, "metadata": {"doc_date": "2026-12-31"}}
    out = _sort_newest_first([c1, c2])
    assert [c["id"] for c in out] == ["new", "old"]


def test_sort_newest_first_is_stable_for_equal_years():
    from visitassist_rag.rag.engine import _sort_newest_first

    c1 = {"id": "a", "score": 0.9, "metadata": {"doc_year": 2020}}
    c2 = {"id": "b", "score": 0.8, "metadata": {"doc_year": 2020}}
    c3 = {"id": "c", "score": 0.7, "metadata": {"doc_year": 1990}}
    out = _sort_newest_first([c1, c2, c3])
    assert [c["id"] for c in out] == ["a", "b", "c"]


def test_definition_guard_explicative_returns_grounded_facts_when_no_definition():
    from visitassist_rag.rag.engine import _definition_guard

    question = "O que é café?"
    # Simulate an LLM trying to define (marker: 'consiste em'), with a citation already attached.
    answer = "Café consiste em algo.\nFonte: [S1]"
    snippets = [
        {
            "metadata": {
                "chunk_text": "Isto fez com que o café se deslocasse para o interior do São Paulo, mais precisamente para a região oeste do estado.",
                "chunk_type": "fine",
            }
        }
    ]

    out = _definition_guard(
        question=question,
        answer=answer,
        snippets=snippets,
        language="pt",
        answer_style="explicative",
    )

    assert "não trazem uma definição explícita" in out
    assert "Isto fez com que o café se deslocasse" in out
    assert out.endswith("Fonte: [S1]")


def test_definition_guard_strict_keeps_refusal_for_definition_questions():
    from visitassist_rag.rag.engine import _definition_guard

    question = "O que é café?"
    answer = "Café consiste em algo.\nFonte: [S1]"
    snippets = [
        {
            "metadata": {
                "chunk_text": "Isto fez com que o café se deslocasse para o interior do São Paulo.",
                "chunk_type": "fine",
            }
        }
    ]

    out = _definition_guard(
        question=question,
        answer=answer,
        snippets=snippets,
        language="pt",
        answer_style="strict",
    )

    assert "não contêm definição explícita suficiente" in out
    assert "Isto fez com que o café se deslocasse" not in out
    assert out.endswith("Fonte: [S1]")


def test_question_constraint_guard_blocks_question_only_entities_and_time():
    from visitassist_rag.rag.engine import _question_constraint_guard

    question = "Quais fatores explicam a mudança do eixo da produção de café do Rio de Janeiro para São Paulo no século XIX?"
    # Simulate an answer that repeats question-only constraints.
    answer = "No século XIX, a mudança do Rio de Janeiro para São Paulo ocorreu por X.\nFonte: [S1]"
    snippets = [
        {
            "metadata": {
                "chunk_text": "Isto fez com que o café se deslocasse para o interior do São Paulo, mais precisamente para a região oeste do estado. Uma das consequências deste fato foi o aumento dos custos de transporte, levando à construção de ferrovias.",
                "chunk_type": "fine",
            }
        }
    ]

    out = _question_constraint_guard(
        question=question,
        answer=answer,
        snippets=snippets,
        language="pt",
        answer_style="explicative",
    )

    assert "Rio de Janeiro" not in out
    assert "século XIX" not in out
    assert "Isto fez com que o café se deslocasse" in out
    assert out.endswith("Fonte: [S1]")
