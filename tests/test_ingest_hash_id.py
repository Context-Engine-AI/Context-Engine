import importlib

ing = importlib.import_module("scripts.ingest_code")


def test_hash_id_stability():
    t = "some text"
    p = "path/file.py"
    s = 1
    e = 10
    h1 = ing.hash_id(t, p, s, e)
    h2 = ing.hash_id(t, p, s, e)
    assert h1 == h2


def test_hash_id_changes_on_input_change():
    t = "some text"
    p = "path/file.py"
    s = 1
    e = 10
    h1 = ing.hash_id(t, p, s, e)
    h2 = ing.hash_id(t + "!", p, s, e)
    assert h1 != h2
