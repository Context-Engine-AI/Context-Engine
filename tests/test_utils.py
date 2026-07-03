import re
from scripts.utils import sanitize_vector_name


def test_sanitize_vector_name_bge_alias():
    assert sanitize_vector_name("BAAI/bge-base-en-v1.5") == "fast-bge-base-en-v1.5"
    assert sanitize_vector_name("baai/BGE-BASE-en-v1.5") == "fast-bge-base-en-v1.5"


def test_sanitize_vector_name_minilm_alias():
    for name in (
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-minilm-l6-v2",
        "sentence-transformers/all-minilm-l-6-v2",
    ):
        assert sanitize_vector_name(name) == "fast-all-minilm-l6-v2"


def test_sanitize_vector_name_fallback_rules():
    out = sanitize_vector_name("org/model_name__v2")
    assert "/" not in out
    assert "_" not in out
    assert len(out) <= 64
    assert re.match(r"^[a-z0-9\-]+$", out)
