import importlib
import textwrap

ing = importlib.import_module("scripts.ingest_code")


def test_php_imports_basic():
    code = textwrap.dedent(
        """
        <?php
        namespace My\\App;
        use Foo\\Bar;
        use function Baz\\qux;
        use const Baz\\QUX;
        require 'vendor/autoload.php';
        include_once "./lib/util.php";
        ?>
        """
    )
    imps = ing._extract_imports("php", code)
    assert "Foo\\Bar" in imps
    assert "Baz\\qux" in imps
    assert "Baz\\QUX" in imps
    assert "vendor/autoload.php" in imps
    assert "./lib/util.php" in imps


def test_php_symbols_basic():
    code = textwrap.dedent(
        """
        <?php
        namespace Demo;
        final class Greeter {
            public function sayHello() {}
            private static function add($a, $b) { return $a + $b; }
        }
        interface IFoo {}
        trait TT {}
        function top() {}
        ?>
        """
    )
    syms = ing._extract_symbols("php", code)
    kinds = {s.kind for s in syms}
    names = {s.name for s in syms}
    assert {"namespace", "class", "interface", "trait", "method", "function"}.issubset(kinds)
    assert {"Demo", "Greeter", "sayHello", "add", "IFoo", "TT", "top"}.issubset(names)

