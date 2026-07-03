import importlib
import textwrap

ing = importlib.import_module("scripts.ingest_code")


def test_csharp_imports_basic():
    code = textwrap.dedent(
        """
        using System;
        using static System.Math;
        using MyApp.Services;
        using Alias = MyApp.Util.Helper;
        namespace MyApp.Core {
            public class Foo {
                public void Bar() { }
            }
        }
        """
    )
    imps = ing._extract_imports("csharp", code)
    # Should capture namespaces (not alias target after '=') and static namespace
    assert "System" in imps
    assert "System.Math" in imps
    assert "MyApp.Services" in imps


def test_csharp_symbols_basic():
    code = textwrap.dedent(
        """
        using System;
        namespace Demo {
            public class Greeter {
                public void SayHello() {}
                private static int Add(int a, int b) { return a + b; }
            }
            interface IFoo {}
            struct S {}
            enum E { A, B }
        }
        """
    )
    syms = ing._extract_symbols("csharp", code)
    kinds = {s.kind for s in syms}
    names = {s.name for s in syms}
    # Expect to see class, interface, struct, enum and methods
    assert {"class", "interface", "struct", "enum", "method"}.issubset(kinds)
    assert {"Greeter", "SayHello", "Add", "IFoo", "S", "E"}.issubset(names)

