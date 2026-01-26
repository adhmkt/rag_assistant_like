def test_query():
    from visitassist_rag.rag.engine import rag_query
    resp = rag_query("What is the weather?", kb_id="curitiba__default")
    assert hasattr(resp, "answer")
