import importlib


ing = importlib.import_module("scripts.ingest_code")


def test_chunk_semantic_go_without_tree_sitter(monkeypatch):
    # Even with tree-sitter disabled, Go symbol extraction is regex-based and should
    # enable semantic chunking around functions/types.
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    go = "\n".join(
        [
            "package main",
            "",
            "type Foo struct {",
            "  X int",
            "}",
            "",
            "func (f *Foo) Bar(a int) int {",
            "  return a + 1",
            "}",
            "",
            "func Baz() {",
            "  Bar(1)",
            "}",
        ]
    )
    chunks = ing.chunk_semantic(go, language="go", max_lines=50, overlap=3)
    assert any(c.get("symbol") in {"Foo", "Bar", "Baz"} for c in chunks)


def test_chunk_semantic_java_without_tree_sitter(monkeypatch):
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    java = "\n".join(
        [
            "package demo;",
            "import java.util.*;",
            "public class A {",
            "  public void foo() {",
            "    bar();",
            "  }",
            "  public void bar() {",
            "  }",
            "}",
        ]
    )
    chunks = ing.chunk_semantic(java, language="java", max_lines=50, overlap=3)
    assert any(c.get("symbol") in {"A", "foo", "bar"} for c in chunks)


def test_chunk_semantic_csharp_without_tree_sitter(monkeypatch):
    monkeypatch.setenv("USE_TREE_SITTER", "0")
    cs = "\n".join(
        [
            "using System;",
            "namespace Demo {",
            "  public class A {",
            "    public void Foo() {",
            "      Bar();",
            "    }",
            "    public void Bar() {}",
            "  }",
            "}",
        ]
    )
    chunks = ing.chunk_semantic(cs, language="csharp", max_lines=50, overlap=3)
    assert any(c.get("symbol") in {"A", "Foo", "Bar"} for c in chunks)


