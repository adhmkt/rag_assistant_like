from visitassist_rag.rag.ingest import _make_summary_chunk_text
from visitassist_rag.rag.chunking import count_tokens


def test_make_summary_chunk_text_is_bounded_and_nonempty():
    text = "\n".join(
        [
            "Este é um documento de teste.",
            "Ele contém várias frases para simular uma seção longa.",
        ]
        * 200
    )
    summary = _make_summary_chunk_text(text, max_tokens=120)
    assert isinstance(summary, str)
    assert summary.strip()
    assert count_tokens(summary) <= 120
