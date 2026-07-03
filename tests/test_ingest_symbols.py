import importlib
import textwrap

ing = importlib.import_module("scripts.ingest_code")


def test_symbols_python_basic():
    code = textwrap.dedent(
        """
        def foo():
            pass

        class Bar:
            def baz(self):
                return 1
        """
    )
    syms = ing._extract_symbols("python", code)
    kinds = {s.kind for s in syms}
    names = {s.name for s in syms}
    assert "function" in kinds
    assert "class" in kinds
    assert {"foo", "Bar"}.issubset(names)


def test_symbols_js_basic():
    code = textwrap.dedent(
        """
        export class A {}
        function f() {}
        const g = function() {}
        """
    )
    syms = ing._extract_symbols("javascript", code)
    names = {s.name for s in syms}
    assert {"A", "f", "g"}.issubset(names)


def test_symbols_go_basic():
    code = textwrap.dedent(
        """
        package p
        type T struct{}
        func (t *T) M() {}
        func F() {}
        """
    )
    syms = ing._extract_symbols("go", code)
    names = {s.name for s in syms}
    assert {"T", "M", "F"}.issubset(names)


def test_symbols_rust_basic():
    code = textwrap.dedent(
        """
        impl Foo { fn new() {} }
        struct Foo {}
        fn main() {}
        """
    )
    syms = ing._extract_symbols("rust", code)
    names = {s.name for s in syms}
    assert {"Foo", "main"}.issubset(names)


def test_symbols_java_basic():
    code = textwrap.dedent(
        """
        public class A {}
        public class B {}
        """
    )
    syms = ing._extract_symbols("java", code)
    names = [s.name for s in syms if s.kind == "class"]
    assert names == ["A", "B"]


def test_symbols_shell_basic():
    code = textwrap.dedent(
        """
        foo() {
          echo hi
        }
        function bar {
          echo hi
        }
        """
    )
    syms = ing._extract_symbols("shell", code)
    names = {s.name for s in syms}
    assert {"foo", "bar"}.issubset(names)


def test_symbols_terraform_basic():
    code = textwrap.dedent(
        """
        resource "aws_s3_bucket" "b" { }
        data "aws_ami" "x" { }
        module "m" { source = "." }
        variable "v" {}
        output "o" {}
        """
    )
    syms = ing._extract_symbols("terraform", code)
    kinds = {s.kind for s in syms}
    assert {"resource", "data", "module", "variable", "output"}.issubset(kinds)
