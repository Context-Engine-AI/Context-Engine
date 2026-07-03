#!/usr/bin/env python3
"""
ingest/symbols.py - Symbol extraction for code analysis.

This module provides functions for extracting functions, classes, methods,
and other code symbols from source files using tree-sitter or regex fallbacks.
"""
from __future__ import annotations

import os
import re
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from scripts.ingest.tree_sitter import (
    _use_tree_sitter,
    _TS_LANGUAGES,
    _ts_parser,
)


# ---------------------------------------------------------------------------
# Symbol dict class for convenient attribute access
# ---------------------------------------------------------------------------
class _Sym(dict):
    """Symbol dict with attribute-style access."""
    __getattr__ = dict.get


# ---------------------------------------------------------------------------
# Python symbol extraction (using built-in ast)
# ---------------------------------------------------------------------------
def _extract_symbols_python(text: str) -> List[_Sym]:
    """Extract symbols from Python code using the ast module (fallback for tree-sitter)."""
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    out: List[_Sym] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(
                _Sym(
                    kind="function",
                    name=node.name,
                    start=getattr(node, "lineno", 0),
                    end=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                )
            )
        elif isinstance(node, ast.ClassDef):
            out.append(
                _Sym(
                    kind="class",
                    name=node.name,
                    start=getattr(node, "lineno", 0),
                    end=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                )
            )
    # Filter invalid
    return [s for s in out if s.start and s.end and s.end >= s.start]


# ---------------------------------------------------------------------------
# JavaScript/TypeScript symbol extraction (regex-based)
# ---------------------------------------------------------------------------
_JS_FUNC_PATTERNS = [
    r"^\s*export\s+function\s+([A-Za-z_$][\w$]*)\s*\(",
    r"^\s*function\s+([A-Za-z_$][\w$]*)\s*\(",
    r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>",
    r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*function\s*\(",
]
_JS_CLASS_PATTERNS = [r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\b"]


def _extract_symbols_js_like(text: str) -> List[_Sym]:
    """Extract symbols from JavaScript/TypeScript using regex."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        for pat in _JS_CLASS_PATTERNS:
            m = re.match(pat, line)
            if m:
                syms.append(_Sym(kind="class", name=m.group(1), start=idx, end=idx))
                break
        for pat in _JS_FUNC_PATTERNS:
            m = re.match(pat, line)
            if m:
                syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
                break
    # Approximate end by next symbol start-1
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        if i + 1 < len(syms):
            syms[i]["end"] = max(syms[i].start, syms[i + 1].start - 1)
        else:
            syms[i]["end"] = max(syms[i].start, len(lines))
    return syms


def _extract_symbols_go(text: str) -> List[_Sym]:
    """Extract symbols from Go source code."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*type\s+([A-Za-z_][\w]*)\s+struct\b", line)
        if m:
            syms.append(_Sym(kind="struct", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*type\s+([A-Za-z_][\w]*)\s+interface\b", line)
        if m:
            syms.append(_Sym(kind="interface", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(
            r"^\s*func\s*\(\s*[^)]+\s+\*?([A-Za-z_][\w]*)\s*\)\s*([A-Za-z_][\w]*)\s*\(",
            line,
        )
        if m:
            syms.append(
                _Sym(
                    kind="method",
                    name=m.group(2),
                    path=f"{m.group(1)}.{m.group(2)}",
                    start=idx,
                    end=idx,
                )
            )
            continue
        m = re.match(r"^\s*func\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols_java(text: str) -> List[_Sym]:
    """Extract symbols from Java source code."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_class = None
    for idx, line in enumerate(lines, 1):
        m = re.match(
            r"^\s*(?:public|protected|private)?\s*(?:final\s+|abstract\s+)?class\s+([A-Za-z_][\w]*)\b",
            line,
        )
        if m:
            current_class = m.group(1)
            syms.append(_Sym(kind="class", name=current_class, start=idx, end=idx))
            continue
        m = re.match(
            r"^\s*(?:public|protected|private)?\s*(?:static\s+)?[A-Za-z_<>,\[\]]+\s+([A-Za-z_][\w]*)\s*\(",
            line,
        )
        if m:
            name = m.group(1)
            path = f"{current_class}.{name}" if current_class else name
            syms.append(_Sym(kind="method", name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols_csharp(text: str) -> List[_Sym]:
    """Extract symbols from C# source code."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_type = None
    for idx, line in enumerate(lines, 1):
        # class / interface / struct / enum
        m = re.match(r"^\s*(?:public|protected|private|internal)?\s*(?:abstract\s+|sealed\s+|static\s+)?(class|interface|struct|enum)\s+([A-Za-z_][\w]*)\b", line)
        if m:
            kind, name = m.group(1), m.group(2)
            current_type = name
            kind_map = {"class": "class", "interface": "interface", "struct": "struct", "enum": "enum"}
            syms.append(_Sym(kind=kind_map.get(kind, "type"), name=name, start=idx, end=idx))
            continue
        # method (very heuristic)
        m = re.match(r"^\s*(?:public|protected|private|internal)?\s*(?:static\s+|virtual\s+|override\s+|async\s+)?[A-Za-z_<>,\[\]\.]+\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            name = m.group(1)
            path = f"{current_type}.{name}" if current_type else name
            syms.append(_Sym(kind="method", name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols_php(text: str) -> List[_Sym]:
    """Extract symbols from PHP source code."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_type = None
    depth = 0
    for idx, line in enumerate(lines, 1):
        # track simple brace depth to reset current_type when leaving class
        depth += line.count("{")
        depth -= line.count("}")
        if depth <= 0:
            current_type = None
        # namespace declaration (optional informational anchor)
        m = re.match(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_\\\\]*)\s*;", line)
        if m:
            ns = m.group(1).replace("\\\\", "\\")
            syms.append(_Sym(kind="namespace", name=ns, start=idx, end=idx))
            continue
        # class/interface/trait
        m = re.match(r"^\s*(?:final\s+|abstract\s+)?(class|interface|trait)\s+([A-Za-z_][\w]*)\b", line)
        if m:
            kind, name = m.group(1), m.group(2)
            current_type = name
            syms.append(_Sym(kind=kind, name=name, start=idx, end=idx))
            continue
        # methods or functions
        m = re.match(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?function\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            name = m.group(1)
            path = f"{current_type}.{name}" if current_type else name
            syms.append(_Sym(kind="method" if current_type else "function", name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols_shell(text: str) -> List[_Sym]:
    """Extract symbols from shell scripts."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*([A-Za-z_][\w]*)\s*\(\)\s*\{", line)
        if m:
            syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*function\s+([A-Za-z_][\w]*)\s*\{", line)
        if m:
            syms.append(_Sym(kind="function", name=m.group(1), start=idx, end=idx))
            continue
    return syms


def _extract_symbols_yaml(text: str) -> List[_Sym]:
    """Extract symbols from YAML files (headings)."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        # treat Markdown-style headings in YAML files as anchors
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            syms.append(
                _Sym(kind="heading", name=m.group(1).strip(), start=idx, end=idx)
            )
    return syms


def _extract_symbols_powershell(text: str) -> List[_Sym]:
    """Extract symbols from PowerShell scripts."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        if re.match(
            r"^\s*function\s+([A-Za-z_][\w-]*)\s*\{", line, flags=re.IGNORECASE
        ):
            name = (
                re.sub(r"^\s*function\s+", "", line, flags=re.IGNORECASE)
                .split("{")[0]
                .strip()
            )
            syms.append(_Sym(kind="function", name=name, start=idx, end=idx))
            continue
        m = re.match(r"^\s*class\s+([A-Za-z_][\w-]*)\s*\{", line, flags=re.IGNORECASE)
        if m:
            syms.append(_Sym(kind="class", name=m.group(1), start=idx, end=idx))
            continue
    return syms


def _extract_symbols_rust(text: str) -> List[_Sym]:
    """Extract symbols from Rust source code."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    current_impl = None
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*impl(?:\s*<[^>]+>)?\s*([A-Za-z_][\w:]*)", line)
        if m:
            current_impl = m.group(1)
            syms.append(_Sym(kind="impl", name=current_impl, start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][\w]*)\b", line)
        if m:
            syms.append(_Sym(kind="struct", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][\w]*)\b", line)
        if m:
            syms.append(_Sym(kind="enum", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][\w]*)\b", line)
        if m:
            syms.append(_Sym(kind="trait", name=m.group(1), start=idx, end=idx))
            continue
        m = re.match(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][\w]*)\s*\(", line)
        if m:
            name = m.group(1)
            path = f"{current_impl}::{name}" if current_impl else name
            kind = "method" if current_impl else "function"
            syms.append(_Sym(kind=kind, name=name, path=path, start=idx, end=idx))
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


def _extract_symbols_terraform(text: str) -> List[_Sym]:
    """Extract symbols from Terraform files."""
    lines = text.splitlines()
    syms: List[_Sym] = []
    for idx, line in enumerate(lines, 1):
        m = re.match(r"^\s*(resource)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s*\{", line)
        if m:
            t, name = m.group(2), m.group(3)
            syms.append(
                _Sym(kind="resource", name=name, path=f"{t}.{name}", start=idx, end=idx)
            )
            continue
        m = re.match(r"^\s*(data)\s+\"([^\"]+)\"\s+\"([^\"]+)\"\s*\{", line)
        if m:
            t, name = m.group(2), m.group(3)
            syms.append(
                _Sym(
                    kind="data", name=name, path=f"data.{t}.{name}", start=idx, end=idx
                )
            )
            continue
        m = re.match(r"^\s*(module)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(
                    kind="module", name=name, path=f"module.{name}", start=idx, end=idx
                )
            )
            continue
        m = re.match(r"^\s*(variable)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(kind="variable", name=name, path=f"var.{name}", start=idx, end=idx)
            )
            continue
        m = re.match(r"^\s*(output)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(
                    kind="output", name=name, path=f"output.{name}", start=idx, end=idx
                )
            )
            continue
        m = re.match(r"^\s*(provider)\s+\"([^\"]+)\"\s*\{", line)
        if m:
            name = m.group(2)
            syms.append(
                _Sym(
                    kind="provider",
                    name=name,
                    path=f"provider.{name}",
                    start=idx,
                    end=idx,
                )
            )
            continue
        m = re.match(r"^\s*(locals)\s*\{", line)
        if m:
            syms.append(
                _Sym(kind="locals", name="locals", path="locals", start=idx, end=idx)
            )
            continue
    syms.sort(key=lambda s: s.start)
    for i in range(len(syms)):
        syms[i]["end"] = (syms[i + 1].start - 1) if (i + 1 < len(syms)) else len(lines)
    return syms


# ---------------------------------------------------------------------------
# Tree-sitter based extraction
# ---------------------------------------------------------------------------
def _ts_extract_symbols_python(text: str) -> List[_Sym]:
    """Extract Python symbols using tree-sitter."""
    parser = _ts_parser("python")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        print(f"[WARN] Tree-sitter parse failed for Python: {e}")
        return []
    syms: List[_Sym] = []

    def node_text(n):
        return text.encode("utf-8")[n.start_byte : n.end_byte].decode(
            "utf-8", errors="ignore"
        )

    class_stack: List[str] = []

    def walk(n):
        t = n.type
        if t == "class_definition":
            name_node = n.child_by_field_name("name")
            cls = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            syms.append(_Sym(kind="class", name=cls, start=start, end=end))
            class_stack.append(cls)
            # Walk body
            for c in n.children:
                walk(c)
            class_stack.pop()
            return
        if t == "function_definition":
            name_node = n.child_by_field_name("name")
            fn = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            if class_stack:
                path = f"{class_stack[-1]}.{fn}"
                syms.append(
                    _Sym(kind="method", name=fn, path=path, start=start, end=end)
                )
            else:
                syms.append(_Sym(kind="function", name=fn, start=start, end=end))
        for c in n.children:
            walk(c)

    walk(root)
    return syms


def _ts_extract_symbols_js(text: str) -> List[_Sym]:
    """Extract JavaScript symbols using tree-sitter."""
    parser = _ts_parser("javascript")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        print(f"[WARN] Tree-sitter parse failed for JavaScript/TypeScript: {e}")
        return []
    syms: List[_Sym] = []

    def node_text(n):
        return text.encode("utf-8")[n.start_byte : n.end_byte].decode(
            "utf-8", errors="ignore"
        )

    class_stack: List[str] = []

    def walk(n):
        t = n.type
        if t == "class_declaration":
            name_node = n.child_by_field_name("name")
            cls = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            syms.append(_Sym(kind="class", name=cls, start=start, end=end))
            class_stack.append(cls)
            for c in n.children:
                walk(c)
            class_stack.pop()
            return
        if t in ("function_declaration",):
            name_node = n.child_by_field_name("name")
            fn = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            syms.append(_Sym(kind="function", name=fn, start=start, end=end))
        if t == "method_definition":
            name_node = n.child_by_field_name("name")
            m = node_text(name_node) if name_node else ""
            start = n.start_point[0] + 1
            end = n.end_point[0] + 1
            path = f"{class_stack[-1]}.{m}" if class_stack else m
            syms.append(_Sym(kind="method", name=m, path=path, start=start, end=end))
        # Handle variable declarations with function expressions or arrow functions
        if t == "variable_declarator":
            name_node = None
            value_node = None
            for c in n.children:
                if c.type == "identifier" and name_node is None:
                    name_node = c
                elif c.type in ("function_expression", "arrow_function"):
                    value_node = c
            if name_node and value_node:
                fn = node_text(name_node)
                start = n.start_point[0] + 1
                end = n.end_point[0] + 1
                syms.append(_Sym(kind="function", name=fn, start=start, end=end))
                return
        for c in n.children:
            walk(c)

    walk(root)
    return syms


def _ts_extract_symbols_yaml(text: str) -> List[_Sym]:
    """Tree-sitter based YAML symbol extraction."""
    parser = _ts_parser("yaml")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception):
        return []

    syms: List[_Sym] = []
    text_bytes = text.encode("utf-8")

    def _node_text(node) -> str:
        return text_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def walk(node, path: list[str] | None = None):
        path = path or []
        ntype = node.type if hasattr(node, "type") else ""

        if ntype == "block_mapping_pair":
            key_node = None
            for child in node.children:
                if hasattr(child, "type") and child.type == "flow_node":
                    key_node = child
                    break
                if hasattr(child, "type") and child.type in (
                    "plain_scalar",
                    "double_quote_scalar",
                    "single_quote_scalar",
                ):
                    key_node = child
                    break
            if key_node:
                key = _node_text(key_node).strip().strip('"').strip("'")
                if key:
                    full_path = ".".join(path + [key])
                    syms.append(
                        _Sym(
                            kind="key",
                            name=full_path,
                            start=node.start_point[0] + 1,
                            end=node.end_point[0] + 1,
                        )
                    )
                    for child in node.children:
                        walk(child, path + [key])
                    return

        if ntype in ("anchor", "alias"):
            name = _node_text(node)
            syms.append(
                _Sym(
                    kind="anchor" if ntype == "anchor" else "alias",
                    name=name,
                    start=node.start_point[0] + 1,
                    end=node.end_point[0] + 1,
                )
            )

        for child in node.children:
            walk(child, path)

    walk(root)
    return syms


def _ts_extract_symbols_go(text: str) -> List[_Sym]:
    """Extract Go symbols using tree-sitter."""
    parser = _ts_parser("go")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        print(f"[WARN] Tree-sitter parse failed for Go: {e}")
        return []

    syms: List[_Sym] = []
    text_bytes = text.encode("utf-8")

    def node_text(n):
        return text_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    def walk(node):
        ntype = node.type

        # Functions
        if ntype == "function_declaration":
            name_node = node.child_by_field_name("name")
            fn = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="function",
                name=fn,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Methods (func (r *Receiver) Name())
        elif ntype == "method_declaration":
            name_node = node.child_by_field_name("name")
            method_name = node_text(name_node) if name_node else ""
            receiver = node.child_by_field_name("receiver")
            receiver_type = ""
            if receiver:
                # Recursively find type_identifier in receiver subtree
                def find_type_identifier(n):
                    if n.type == "type_identifier":
                        return node_text(n)
                    for c in n.children:
                        result = find_type_identifier(c)
                        if result:
                            return result
                    return ""
                receiver_type = find_type_identifier(receiver)
            path = f"{receiver_type}.{method_name}" if receiver_type else method_name
            syms.append(_Sym(
                kind="method",
                name=method_name,
                path=path,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Types (struct, interface)
        elif ntype == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name_node = child.child_by_field_name("name")
                    type_name = node_text(name_node) if name_node else ""
                    type_node = child.child_by_field_name("type")
                    kind = "type"
                    if type_node:
                        if type_node.type == "struct_type":
                            kind = "struct"
                        elif type_node.type == "interface_type":
                            kind = "interface"
                    syms.append(_Sym(
                        kind=kind,
                        name=type_name,
                        start=child.start_point[0] + 1,
                        end=child.end_point[0] + 1,
                    ))

        for child in node.children:
            walk(child)

    walk(root)
    return syms


def _ts_extract_symbols_java(text: str) -> List[_Sym]:
    """Extract Java symbols using tree-sitter."""
    parser = _ts_parser("java")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        print(f"[WARN] Tree-sitter parse failed for Java: {e}")
        return []

    syms: List[_Sym] = []
    text_bytes = text.encode("utf-8")

    def node_text(n):
        return text_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    class_stack: List[str] = []

    def walk(node):
        ntype = node.type

        # Class declarations
        if ntype == "class_declaration":
            name_node = node.child_by_field_name("name")
            cls = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="class",
                name=cls,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))
            class_stack.append(cls)
            for child in node.children:
                walk(child)
            class_stack.pop()
            return

        # Interface declarations
        elif ntype == "interface_declaration":
            name_node = node.child_by_field_name("name")
            iface = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="interface",
                name=iface,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))
            class_stack.append(iface)
            for child in node.children:
                walk(child)
            class_stack.pop()
            return

        # Enum declarations
        elif ntype == "enum_declaration":
            name_node = node.child_by_field_name("name")
            enum = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="enum",
                name=enum,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Method declarations
        elif ntype == "method_declaration":
            name_node = node.child_by_field_name("name")
            method = node_text(name_node) if name_node else ""
            path = f"{class_stack[-1]}.{method}" if class_stack else method
            syms.append(_Sym(
                kind="method",
                name=method,
                path=path,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Constructor declarations
        elif ntype == "constructor_declaration":
            name_node = node.child_by_field_name("name")
            ctor = node_text(name_node) if name_node else ""
            path = f"{class_stack[-1]}.{ctor}" if class_stack else ctor
            syms.append(_Sym(
                kind="constructor",
                name=ctor,
                path=path,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        for child in node.children:
            walk(child)

    walk(root)
    return syms


def _ts_extract_symbols_rust(text: str) -> List[_Sym]:
    """Extract Rust symbols using tree-sitter."""
    parser = _ts_parser("rust")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        print(f"[WARN] Tree-sitter parse failed for Rust: {e}")
        return []

    syms: List[_Sym] = []
    text_bytes = text.encode("utf-8")

    def node_text(n):
        return text_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    impl_stack: List[str] = []

    def walk(node):
        ntype = node.type

        # Functions
        if ntype == "function_item":
            name_node = node.child_by_field_name("name")
            fn = node_text(name_node) if name_node else ""
            if impl_stack:
                path = f"{impl_stack[-1]}::{fn}"
                kind = "method"
            else:
                path = fn
                kind = "function"
            syms.append(_Sym(
                kind=kind,
                name=fn,
                path=path,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Impl blocks
        elif ntype == "impl_item":
            type_node = node.child_by_field_name("type")
            impl_type = node_text(type_node).split("<")[0].strip() if type_node else ""
            syms.append(_Sym(
                kind="impl",
                name=impl_type,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))
            impl_stack.append(impl_type)
            for child in node.children:
                walk(child)
            impl_stack.pop()
            return

        # Structs
        elif ntype == "struct_item":
            name_node = node.child_by_field_name("name")
            struct = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="struct",
                name=struct,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Enums
        elif ntype == "enum_item":
            name_node = node.child_by_field_name("name")
            enum = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="enum",
                name=enum,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Traits
        elif ntype == "trait_item":
            name_node = node.child_by_field_name("name")
            trait = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="trait",
                name=trait,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Modules
        elif ntype == "mod_item":
            name_node = node.child_by_field_name("name")
            mod = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="module",
                name=mod,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        for child in node.children:
            walk(child)

    walk(root)
    return syms


def _ts_extract_symbols_csharp(text: str) -> List[_Sym]:
    """Extract C# symbols using tree-sitter."""
    parser = _ts_parser("c_sharp")
    if not parser:
        # Try alias
        parser = _ts_parser("csharp")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        print(f"[WARN] Tree-sitter parse failed for C#: {e}")
        return []

    syms: List[_Sym] = []
    text_bytes = text.encode("utf-8")

    def node_text(n):
        return text_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    type_stack: List[str] = []

    def walk(node):
        ntype = node.type

        # Class declarations
        if ntype == "class_declaration":
            name_node = node.child_by_field_name("name")
            cls = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="class",
                name=cls,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))
            type_stack.append(cls)
            for child in node.children:
                walk(child)
            type_stack.pop()
            return

        # Interface declarations
        elif ntype == "interface_declaration":
            name_node = node.child_by_field_name("name")
            iface = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="interface",
                name=iface,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))
            type_stack.append(iface)
            for child in node.children:
                walk(child)
            type_stack.pop()
            return

        # Struct declarations
        elif ntype == "struct_declaration":
            name_node = node.child_by_field_name("name")
            struct = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="struct",
                name=struct,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))
            type_stack.append(struct)
            for child in node.children:
                walk(child)
            type_stack.pop()
            return

        # Enum declarations
        elif ntype == "enum_declaration":
            name_node = node.child_by_field_name("name")
            enum = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="enum",
                name=enum,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Method declarations
        elif ntype == "method_declaration":
            name_node = node.child_by_field_name("name")
            method = node_text(name_node) if name_node else ""
            path = f"{type_stack[-1]}.{method}" if type_stack else method
            syms.append(_Sym(
                kind="method",
                name=method,
                path=path,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Constructor declarations
        elif ntype == "constructor_declaration":
            name_node = node.child_by_field_name("name")
            ctor = node_text(name_node) if name_node else ""
            path = f"{type_stack[-1]}.{ctor}" if type_stack else ctor
            syms.append(_Sym(
                kind="constructor",
                name=ctor,
                path=path,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        # Property declarations
        elif ntype == "property_declaration":
            name_node = node.child_by_field_name("name")
            prop = node_text(name_node) if name_node else ""
            path = f"{type_stack[-1]}.{prop}" if type_stack else prop
            syms.append(_Sym(
                kind="property",
                name=prop,
                path=path,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        for child in node.children:
            walk(child)

    walk(root)
    return syms


def _ts_extract_symbols_bash(text: str) -> List[_Sym]:
    """Extract Bash/Shell symbols using tree-sitter."""
    parser = _ts_parser("bash")
    if not parser:
        parser = _ts_parser("shell")
    if not parser:
        parser = _ts_parser("sh")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        print(f"[WARN] Tree-sitter parse failed for Bash: {e}")
        return []

    syms: List[_Sym] = []
    text_bytes = text.encode("utf-8")

    def node_text(n):
        return text_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    def walk(node):
        ntype = node.type

        # Function definitions
        if ntype == "function_definition":
            name_node = node.child_by_field_name("name")
            fn = node_text(name_node) if name_node else ""
            syms.append(_Sym(
                kind="function",
                name=fn,
                start=node.start_point[0] + 1,
                end=node.end_point[0] + 1,
            ))

        for child in node.children:
            walk(child)

    walk(root)
    return syms


def _ts_extract_symbols(language: str, text: str) -> List[_Sym]:
    """Extract symbols using tree-sitter for supported languages."""
    if language == "python":
        return _ts_extract_symbols_python(text)
    if language == "javascript":
        return _ts_extract_symbols_js(text)
    if language == "typescript":
        if "typescript" in _TS_LANGUAGES:
            parser = _ts_parser("typescript")
            if parser:
                try:
                    tree = parser.parse(text.encode("utf-8"))
                    if tree is None:
                        return []
                    root = tree.root_node
                except (ValueError, Exception) as e:
                    print(f"[WARN] Tree-sitter parse failed for TypeScript: {e}")
                    return []

                syms: List[_Sym] = []

                def node_text(n):
                    return text.encode("utf-8")[n.start_byte : n.end_byte].decode(
                        "utf-8", errors="ignore"
                    )

                class_stack: List[str] = []

                def walk(n):
                    t = n.type
                    if t == "class_declaration":
                        name_node = n.child_by_field_name("name")
                        cls = node_text(name_node) if name_node else ""
                        start = n.start_point[0] + 1
                        end = n.end_point[0] + 1
                        syms.append(_Sym(kind="class", name=cls, start=start, end=end))
                        class_stack.append(cls)
                        for c in n.children:
                            walk(c)
                        class_stack.pop()
                        return
                    if t in ("function_declaration",):
                        name_node = n.child_by_field_name("name")
                        fn = node_text(name_node) if name_node else ""
                        start = n.start_point[0] + 1
                        end = n.end_point[0] + 1
                        syms.append(_Sym(kind="function", name=fn, start=start, end=end))
                    if t == "method_definition":
                        name_node = n.child_by_field_name("name")
                        m = node_text(name_node) if name_node else ""
                        start = n.start_point[0] + 1
                        end = n.end_point[0] + 1
                        path = f"{class_stack[-1]}.{m}" if class_stack else m
                        syms.append(_Sym(kind="method", name=m, path=path, start=start, end=end))
                    if t == "variable_declarator":
                        name_node = None
                        value_node = None
                        for c in n.children:
                            if c.type == "identifier" and name_node is None:
                                name_node = c
                            elif c.type in ("function_expression", "arrow_function"):
                                value_node = c
                        if name_node and value_node:
                            fn = node_text(name_node)
                            start = n.start_point[0] + 1
                            end = n.end_point[0] + 1
                            syms.append(_Sym(kind="function", name=fn, start=start, end=end))
                            return
                    for c in n.children:
                        walk(c)

                walk(root)
                return syms

        return _ts_extract_symbols_js(text)

    if language == "yaml":
        return _ts_extract_symbols_yaml(text)

    # New tree-sitter extractors
    if language == "go":
        return _ts_extract_symbols_go(text)
    if language == "java":
        return _ts_extract_symbols_java(text)
    if language == "rust":
        return _ts_extract_symbols_rust(text)
    if language in ("csharp", "c_sharp"):
        return _ts_extract_symbols_csharp(text)
    if language in ("shell", "bash", "sh"):
        return _ts_extract_symbols_bash(text)

    return []


# ---------------------------------------------------------------------------
# Main symbol extraction dispatcher
# ---------------------------------------------------------------------------
def _extract_symbols(language: str, text: str) -> List[_Sym]:
    """Extract symbols from source code.
    
    Prefers tree-sitter when enabled and supported; falls back to regex extractors.
    """
    if _use_tree_sitter():
        ts_syms = _ts_extract_symbols(language, text)
        if ts_syms:
            return ts_syms
    if language == "python":
        return _extract_symbols_python(text)
    if language in ("javascript", "typescript"):
        return _extract_symbols_js_like(text)
    if language == "go":
        return _extract_symbols_go(text)
    if language == "java":
        return _extract_symbols_java(text)
    if language == "rust":
        return _extract_symbols_rust(text)
    if language == "terraform":
        return _extract_symbols_terraform(text)
    if language == "shell":
        return _extract_symbols_shell(text)
    if language == "yaml":
        return _extract_symbols_yaml(text)
    if language == "powershell":
        return _extract_symbols_powershell(text)
    if language == "csharp":
        return _extract_symbols_csharp(text)
    if language == "php":
        return _extract_symbols_php(text)
    return []


def _choose_symbol_for_chunk(start: int, end: int, symbols: List[_Sym]):
    """Choose the most relevant symbol for a given chunk range."""
    if not symbols:
        return "", "", ""
    overlaps = [s for s in symbols if s.start <= end and s.end >= start]

    def pick(sym):
        name = sym.get("name") or ""
        path = sym.get("path") or name
        return sym.get("kind") or "", name, path

    if overlaps:
        overlaps.sort(key=lambda s: (-(s.start), (s.end - s.start)))
        return pick(overlaps[0])
    preceding = [s for s in symbols if s.start <= end]
    if preceding:
        s = max(preceding, key=lambda x: x.start)
        return pick(s)
    return "", "", ""


# ---------------------------------------------------------------------------
# Smart symbol reindexing support
# ---------------------------------------------------------------------------
def extract_symbols_with_tree_sitter(file_path: str) -> dict:
    """Extract functions, classes, methods from file using tree-sitter or fallback.

    Returns:
        dict: {symbol_id: {name, type, start_line, end_line, content_hash, pseudo, tags}}
    """
    from scripts.ingest.pipeline import detect_language
    
    try:
        # Read file content
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        language = detect_language(Path(file_path))

        # Use existing symbol extraction infrastructure
        symbols_list = _extract_symbols(language, text)

        # Convert to our expected dict format
        symbols = {}
        for sym in symbols_list:
            symbol_id = f"{sym['kind']}_{sym['name']}_{sym['start']}"

            # Extract actual content for hashing
            content_lines = text.split("\n")[sym["start"] - 1 : sym["end"]]
            content = "\n".join(content_lines)
            content_hash = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()

            symbols[symbol_id] = {
                "name": sym["name"],
                "type": sym["kind"],
                "start_line": sym["start"],
                "end_line": sym["end"],
                "content_hash": content_hash,
                "content": content,
                "pseudo": "",
                "tags": [],
                "qdrant_ids": [],
            }

        return symbols

    except Exception as e:
        print(f"[SYMBOL_EXTRACTION] Failed to extract symbols from {file_path}: {e}")
        return {}
