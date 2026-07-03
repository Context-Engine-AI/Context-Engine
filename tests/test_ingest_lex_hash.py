import math
import importlib

# Import the function from ingest_code
ing = importlib.import_module("scripts.ingest_code")


def test_lex_hash_vector_norm_non_empty():
    vec = ing._lex_hash_vector("foo bar baz foo")
    # Non-empty input should be L2-normalized to ~1.0
    norm = math.sqrt(sum(v * v for v in vec))
    assert 0.98 <= norm <= 1.02


def test_lex_hash_vector_deterministic():
    v1 = ing._lex_hash_vector("symbols_and_identifiers")
    v2 = ing._lex_hash_vector("symbols_and_identifiers")
    assert v1 == v2


def test_lex_hash_vector_dim():
    dim = int(getattr(ing, "LEX_VECTOR_DIM", 4096))
    vec = ing._lex_hash_vector("x")
    assert len(vec) == dim
