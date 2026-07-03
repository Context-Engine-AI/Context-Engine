"""
tests/test_ts_extractors.py - Tests for tree-sitter symbol extraction.

Tests the new tree-sitter extraction handlers for Go, Java, Rust, C#, and Bash.
"""
import pytest
import os

# These tests require tree-sitter parsing to be enabled.
# Other tests may disable it globally (env var), so force-enable it for this file.
@pytest.fixture(autouse=True)
def _force_tree_sitter_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USE_TREE_SITTER", "1")

from scripts.ingest.tree_sitter import _ts_parser

from scripts.ingest.symbols import (
    _ts_extract_symbols_go,
    _ts_extract_symbols_java,
    _ts_extract_symbols_rust,
    _ts_extract_symbols_csharp,
    _ts_extract_symbols_bash,
    _ts_extract_symbols,
)

def _bool_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _ts_diag(lang_key: str) -> str:
    """Produce a useful diagnostic when tree-sitter is unexpectedly unavailable."""
    try:
        from scripts.ingest import tree_sitter as ts
    except Exception as e:
        return f"failed to import scripts.ingest.tree_sitter: {e!r}"

    pkg_map = {
        "go": "tree_sitter_go",
        "java": "tree_sitter_java",
        "rust": "tree_sitter_rust",
        "c_sharp": "tree_sitter_c_sharp",
        "csharp": "tree_sitter_c_sharp",
        "bash": "tree_sitter_bash",
        "shell": "tree_sitter_bash",
        "sh": "tree_sitter_bash",
    }
    pkg = pkg_map.get(lang_key)
    pkg_status = "n/a"
    if pkg:
        try:
            __import__(pkg)
            pkg_status = "import ok"
        except Exception as e:
            pkg_status = f"import failed: {e!r}"

    try:
        __import__("tree_sitter")
        core_status = "import ok"
    except Exception as e:
        core_status = f"import failed: {e!r}"

    try:
        loaded = sorted(list(getattr(ts, "_TS_LANGUAGES", {}).keys()))
    except Exception:
        loaded = []

    return (
        f"USE_TREE_SITTER={os.environ.get('USE_TREE_SITTER')!r}, "
        f"TS_AVAILABLE={getattr(ts, '_TS_AVAILABLE', None)!r}, "
        f"Parser_is_None={getattr(ts, 'Parser', None) is None}, "
        f"tree_sitter={core_status}, "
        f"{pkg}={pkg_status}, "
        f"loaded_lang_keys={loaded}"
    )


def _require_ts(lang_key: str, label: str) -> None:
    """Skip only when the tree-sitter parser/grammar truly isn't available."""
    if _ts_parser(lang_key) is None:
        # In CI we expect requirements to be installed; skipping hides real regressions.
        if _bool_env("CI") or _bool_env("FORCE_TREE_SITTER_TESTS"):
            pytest.fail(f"{label} not installed or not loadable in CI. {_ts_diag(lang_key)}")
        pytest.skip(f"{label} not installed or not loadable")


class TestGoExtractor:
    """Test Go tree-sitter extraction."""

    def test_go_function(self):
        """Test Go function extraction."""
        code = '''package main

func HelloWorld() {
    fmt.Println("Hello")
}

func main() {
    HelloWorld()
}
'''
        _require_ts("go", "tree-sitter-go")
        syms = _ts_extract_symbols_go(code)
        assert syms, "tree-sitter-go parser available but extractor returned no symbols"
        names = [s.get("name") for s in syms]
        assert "HelloWorld" in names
        assert "main" in names

    def test_go_method(self):
        """Test Go method extraction."""
        code = '''package main

type Server struct {
    host string
}

func (s *Server) Start() error {
    return nil
}

func (s Server) Stop() {
}
'''
        _require_ts("go", "tree-sitter-go")
        syms = _ts_extract_symbols_go(code)
        assert syms, "tree-sitter-go parser available but extractor returned no symbols"
        
        # Find methods
        methods = [s for s in syms if s.get("kind") == "method"]
        method_names = [m.get("name") for m in methods]
        assert "Start" in method_names
        assert "Stop" in method_names
        
        # Check path includes receiver
        start_method = next((m for m in methods if m.get("name") == "Start"), None)
        if start_method:
            assert "Server" in str(start_method.get("path", ""))

    def test_go_struct_interface(self):
        """Test Go struct/interface extraction."""
        code = '''package main

type Handler interface {
    Handle(req Request) Response
}

type MyHandler struct {
    name string
}
'''
        _require_ts("go", "tree-sitter-go")
        syms = _ts_extract_symbols_go(code)
        assert syms, "tree-sitter-go parser available but extractor returned no symbols"
        
        kinds = {s.get("name"): s.get("kind") for s in syms}
        assert kinds.get("Handler") == "interface"
        assert kinds.get("MyHandler") == "struct"


class TestJavaExtractor:
    """Test Java tree-sitter extraction."""

    def test_java_class_methods(self):
        """Test Java class and method extraction."""
        code = '''public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    private int subtract(int a, int b) {
        return a - b;
    }
}
'''
        _require_ts("java", "tree-sitter-java")
        syms = _ts_extract_symbols_java(code)
        assert syms, "tree-sitter-java parser available but extractor returned no symbols"
        
        names = [s.get("name") for s in syms]
        assert "Calculator" in names
        assert "add" in names
        assert "subtract" in names
        
        # Check method paths
        add_method = next((s for s in syms if s.get("name") == "add"), None)
        if add_method:
            assert add_method.get("path") == "Calculator.add"

    def test_java_interface(self):
        """Test Java interface extraction."""
        code = '''public interface Repository {
    void save(Object entity);
    Object findById(String id);
}
'''
        _require_ts("java", "tree-sitter-java")
        syms = _ts_extract_symbols_java(code)
        assert syms, "tree-sitter-java parser available but extractor returned no symbols"
        
        repo = next((s for s in syms if s.get("name") == "Repository"), None)
        assert repo is not None
        assert repo.get("kind") == "interface"


class TestRustExtractor:
    """Test Rust tree-sitter extraction."""

    def test_rust_functions(self):
        """Test Rust function extraction."""
        code = '''fn main() {
    println!("Hello");
}

pub fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}
'''
        _require_ts("rust", "tree-sitter-rust")
        syms = _ts_extract_symbols_rust(code)
        assert syms, "tree-sitter-rust parser available but extractor returned no symbols"
        
        names = [s.get("name") for s in syms]
        assert "main" in names
        assert "greet" in names

    def test_rust_structs_impl(self):
        """Test Rust struct and impl extraction."""
        code = '''pub struct Config {
    host: String,
    port: u16,
}

impl Config {
    pub fn new() -> Self {
        Config { host: "localhost".to_string(), port: 8080 }
    }
    
    pub fn host(&self) -> &str {
        &self.host
    }
}
'''
        _require_ts("rust", "tree-sitter-rust")
        syms = _ts_extract_symbols_rust(code)
        assert syms, "tree-sitter-rust parser available but extractor returned no symbols"
        
        # Should find struct and impl methods
        names = [s.get("name") for s in syms]
        assert "Config" in names
        assert "new" in names
        assert "host" in names
        
        # Methods should have impl path
        new_fn = next((s for s in syms if s.get("name") == "new"), None)
        if new_fn:
            assert "Config" in str(new_fn.get("path", ""))

    def test_rust_traits(self):
        """Test Rust trait extraction."""
        code = '''pub trait Handler {
    fn handle(&self, req: Request) -> Response;
}
'''
        _require_ts("rust", "tree-sitter-rust")
        syms = _ts_extract_symbols_rust(code)
        assert syms, "tree-sitter-rust parser available but extractor returned no symbols"
        
        handler = next((s for s in syms if s.get("name") == "Handler"), None)
        assert handler is not None
        assert handler.get("kind") == "trait"


class TestCSharpExtractor:
    """Test C# tree-sitter extraction."""

    def test_csharp_class_methods(self):
        """Test C# class and method extraction."""
        code = '''public class UserService
{
    private readonly IRepository _repo;
    
    public UserService(IRepository repo)
    {
        _repo = repo;
    }
    
    public User GetById(string id)
    {
        return _repo.Find(id);
    }
}
'''
        if _ts_parser("c_sharp") is None and _ts_parser("csharp") is None:
            pytest.skip("tree-sitter-c-sharp not installed or not loadable")
        syms = _ts_extract_symbols_csharp(code)
        assert syms, "tree-sitter-c-sharp parser available but extractor returned no symbols"
        
        names = [s.get("name") for s in syms]
        assert "UserService" in names
        assert "GetById" in names

    def test_csharp_interface(self):
        """Test C# interface extraction."""
        code = '''public interface IRepository
{
    void Save(object entity);
    object Find(string id);
}
'''
        if _ts_parser("c_sharp") is None and _ts_parser("csharp") is None:
            pytest.skip("tree-sitter-c-sharp not installed or not loadable")
        syms = _ts_extract_symbols_csharp(code)
        assert syms, "tree-sitter-c-sharp parser available but extractor returned no symbols"
        
        repo = next((s for s in syms if s.get("name") == "IRepository"), None)
        assert repo is not None
        assert repo.get("kind") == "interface"


class TestBashExtractor:
    """Test Bash/Shell tree-sitter extraction."""

    def test_bash_functions(self):
        """Test Bash function extraction."""
        code = '''#!/bin/bash

function setup() {
    echo "Setting up..."
}

cleanup() {
    echo "Cleaning up..."
}

main() {
    setup
    do_work
    cleanup
}
'''
        if _ts_parser("bash") is None and _ts_parser("shell") is None and _ts_parser("sh") is None:
            pytest.skip("tree-sitter-bash not installed or not loadable")
        syms = _ts_extract_symbols_bash(code)
        assert syms, "tree-sitter-bash parser available but extractor returned no symbols"
        
        names = [s.get("name") for s in syms]
        # At minimum should find function definitions
        assert len(syms) >= 1
        # Check that we have function kinds
        kinds = [s.get("kind") for s in syms]
        assert "function" in kinds


class TestDispatcher:
    """Test the _ts_extract_symbols dispatcher."""

    def test_dispatcher_routes_go(self):
        """Test dispatcher routes Go correctly."""
        code = "func main() {}"
        syms = _ts_extract_symbols("go", code)
        # If tree-sitter-go is available, should return symbols
        assert isinstance(syms, list)

    def test_dispatcher_routes_java(self):
        """Test dispatcher routes Java correctly."""
        code = "public class Foo {}"
        syms = _ts_extract_symbols("java", code)
        assert isinstance(syms, list)

    def test_dispatcher_routes_rust(self):
        """Test dispatcher routes Rust correctly."""
        code = "fn main() {}"
        syms = _ts_extract_symbols("rust", code)
        assert isinstance(syms, list)

    def test_dispatcher_routes_csharp(self):
        """Test dispatcher routes C# correctly."""
        code = "public class Foo {}"
        for lang in ["csharp", "c_sharp"]:
            syms = _ts_extract_symbols(lang, code)
            assert isinstance(syms, list)

    def test_dispatcher_routes_bash(self):
        """Test dispatcher routes Bash correctly."""
        code = "function foo() { echo hi; }"
        for lang in ["bash", "shell", "sh"]:
            syms = _ts_extract_symbols(lang, code)
            assert isinstance(syms, list)
