def test_chunking():
    from visitassist_rag.rag.chunking import chunk_by_tokens, split_paragraphs
    text = "This is a test. " * 100
    paragraphs = split_paragraphs(text)
    chunks = chunk_by_tokens(paragraphs, target_tokens=50, overlap_tokens=10)
    assert len(chunks) > 0
