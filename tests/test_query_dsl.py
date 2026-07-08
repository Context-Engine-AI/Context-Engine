"""Regression tests for the query DSL tokenizer.

The DSL must only trigger on the compact form (`key:value`, no whitespace
around the colon). Natural-language queries containing "file: upload flow"
or "repo: billing" previously became hard filters that emptied results or
redirected the search to another repo.
"""

from scripts.hybrid.filters import parse_query_dsl


def test_compact_dsl_tokens_extract():
    clean, tokens = parse_query_dsl(["auth file:scripts/upload.py lang:python"])
    assert tokens["under"] == "scripts/upload.py"
    assert tokens["language"] == "python"
    assert clean == ["auth"]


def test_prose_colon_is_not_dsl():
    query = "explain the file: upload flow and repo: billing behavior"
    clean, tokens = parse_query_dsl([query])
    assert tokens == {}
    assert clean == [query]


def test_repo_token_still_extracts_in_compact_form():
    clean, tokens = parse_query_dsl(["payment handler repo:billing"])
    assert tokens["repo"] == "billing"
    assert clean == ["payment handler"]
