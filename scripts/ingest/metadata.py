#!/usr/bin/env python3
"""
ingest/metadata.py - Metadata extraction for indexed files.

This module provides functions for extracting git metadata, imports, calls,
and other file-level information for the indexing pipeline.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional


def _git_metadata(file_path: Path) -> Tuple[int, int, int]:
    """Return (last_modified_at, churn_count, author_count) using git when available.
    
    Falls back to fs mtime and zeros when not in a repo.
    """
    try:
        import subprocess

        fp = str(file_path)
        # last commit unix timestamp (%ct)
        ts = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout.strip()
        last_ts = int(ts) if ts.isdigit() else int(file_path.stat().st_mtime)
        # churn: number of commits touching this file (bounded)
        churn_s = subprocess.run(
            ["git", "rev-list", "--count", "HEAD", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout.strip()
        churn = int(churn_s) if churn_s.isdigit() else 0
        # author count
        authors = subprocess.run(
            ["git", "shortlog", "-s", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout
        author_count = len([ln for ln in authors.splitlines() if ln.strip()])
        return last_ts, churn, author_count
    except Exception:
        try:
            return int(file_path.stat().st_mtime), 0, 0
        except Exception:
            return int(time.time()), 0, 0


def _extract_imports(language: str, text: str) -> List[str]:
    """Lightweight import extraction per language (best-effort)."""
    lines = text.splitlines()
    imps: List[str] = []
    if language == "python":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.]+)", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*from\s+([\w\.]+)\s+import\s+", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language in ("javascript", "typescript"):
        for ln in lines:
            m = re.match(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            # Match require statements: require('x'), const x = require('x'), etc.
            m = re.search(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "go":
        block = False
        for ln in lines:
            if re.match(r"^\s*import\s*\(", ln):
                block = True
                continue
            if block:
                if ")" in ln:
                    block = False
                    continue
                m = re.match(r"^\s*\"([^\"]+)\"", ln)
                if m:
                    imps.append(m.group(1))
                    continue
            m = re.match(r"^\s*import\s+\"([^\"]+)\"", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "java":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.\*]+);", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "csharp":
        for ln in lines:
            # Match: using System; using static System.Math; using Alias = System.Text;
            m = re.match(r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\._]*)\s*;", ln)
            if m:
                imps.append(m.group(1))
                continue
            # Match alias: using Alias = Namespace.Type;
            m = re.match(r"^\s*using\s+\w+\s*=\s*([A-Za-z_][\w\._]*)\s*;", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "php":
        for ln in lines:
            m = re.match(r"^\s*use\s+(?:function\s+|const\s+)?([A-Za-z_][A-Za-z0-9_\\\\]*)\s*;", ln)
            if m:
                imps.append(m.group(1).replace("\\\\", "\\"))
                continue
        for ln in lines:
            m = re.match(r"^\s*(?:include|include_once|require|require_once)\s*\(?\s*['\"]([^'\"]+)['\"]\s*\)?\s*;", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "rust":
        for ln in lines:
            m = re.match(r"^\s*use\s+([^;]+);", ln)
            if m:
                imps.append(m.group(1).strip())
                continue
    elif language in ("c", "cpp"):
        for ln in lines:
            m = re.match(r'^\s*#include\s*[<"]([^>"]+)[>"]', ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "ruby":
        for ln in lines:
            m = re.match(r"^\s*require\s+['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*require_relative\s+['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*load\s+['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "kotlin":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.\*]+)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "swift":
        for ln in lines:
            m = re.match(r"^\s*import\s+(\w+)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "scala":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.\{\}\,\s_]+)", ln)
            if m:
                imps.append(m.group(1).strip())
                continue
    elif language == "terraform":
        for ln in lines:
            m = re.match(r"^\s*source\s*=\s*['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*provider\s*=\s*['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "powershell":
        for ln in lines:
            m = re.match(
                r"^\s*Import-Module\s+([A-Za-z0-9_.\-]+)", ln, flags=re.IGNORECASE
            )
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*using\s+module\s+([^\s;]+)", ln, flags=re.IGNORECASE)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*using\s+namespace\s+([^\s;]+)", ln, flags=re.IGNORECASE)
            if m:
                imps.append(m.group(1))
                continue
    return imps[:200]


def _extract_calls(language: str, text: str) -> List[str]:
    """Lightweight call-site extraction (best-effort, language-agnostic heuristics)."""
    names: List[str] = []
    # Simple heuristic: word followed by '(' that isn't a keyword
    kw = set([
        "if", "for", "while", "switch", "return", "new",
        "catch", "func", "def", "class", "match",
    ])
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = m.group(1)
        if name not in kw:
            names.append(name)
    # Deduplicate preserving order
    out: List[str] = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out[:200]


# Tree-sitter node type mappings per language
# Maps language -> (call_types, member_field_map)
# member_field_map: node_type -> (object_field, property_field)
_TS_LANG_CONFIG = {
    "python": {
        "calls": ["call"],
        "constructors": [],
        "member": {"attribute": ("object", "attribute")},
    },
    "javascript": {
        "calls": ["call_expression"],
        "constructors": ["new_expression"],
        "member": {"member_expression": ("object", "property")},
    },
    "typescript": {
        "calls": ["call_expression"],
        "constructors": ["new_expression"],
        "member": {"member_expression": ("object", "property")},
    },
    "tsx": {
        "calls": ["call_expression"],
        "constructors": ["new_expression"],
        "member": {"member_expression": ("object", "property")},
    },
    "jsx": {
        "calls": ["call_expression"],
        "constructors": ["new_expression"],
        "member": {"member_expression": ("object", "property")},
    },
    "go": {
        "calls": ["call_expression"],
        "constructors": [],
        "member": {"selector_expression": ("operand", "field")},
    },
    "rust": {
        "calls": ["call_expression", "macro_invocation"],
        "constructors": [],
        "member": {"field_expression": ("value", "field")},
    },
    "java": {
        "calls": ["method_invocation"],
        "constructors": ["object_creation_expression"],
        "member": {"method_invocation": ("object", "name")},
    },
    "c": {
        "calls": ["call_expression"],
        "constructors": [],
        "member": {"field_expression": ("argument", "field")},
    },
    "cpp": {
        "calls": ["call_expression"],
        "constructors": ["new_expression"],
        "member": {
            "field_expression": ("argument", "field"),
            "qualified_identifier": ("scope", "name"),
        },
    },
    "ruby": {
        "calls": ["call", "method_call"],
        "constructors": [],
        "member": {"call": ("receiver", "method")},
    },
    "c_sharp": {
        "calls": ["invocation_expression"],
        "constructors": ["object_creation_expression"],
        "member": {"member_access_expression": ("expression", "name")},
    },
    "csharp": {
        "calls": ["invocation_expression"],
        "constructors": ["object_creation_expression"],
        "member": {"member_access_expression": ("expression", "name")},
    },
    "bash": {
        "calls": ["command"],
        "constructors": [],
        "member": {},
    },
    "shell": {
        "calls": ["command"],
        "constructors": [],
        "member": {},
    },
    "sh": {
        "calls": ["command"],
        "constructors": [],
        "member": {},
    },
}

# Languages that have tree-sitter call extraction support
# Derived set of languages that support tree-sitter call extraction
_TS_CALL_LANGUAGES = set(_TS_LANG_CONFIG.keys())

# Default config for unknown languages
_TS_DEFAULT_CONFIG = {
    "calls": ["call_expression"],
    "constructors": ["new_expression"],
    "member": {"member_expression": ("object", "property")},
}

# Import node types per language for tree-sitter
_TS_IMPORT_CONFIG = {
    "python": {
        "nodes": ["import_statement", "import_from_statement"],
    },
    "javascript": {
        "nodes": ["import_statement"],
        "source_field": "source",
    },
    "typescript": {
        "nodes": ["import_statement"],
        "source_field": "source",
    },
    "tsx": {
        "nodes": ["import_statement"],
        "source_field": "source",
    },
    "go": {
        "nodes": ["import_declaration"],
    },
    "rust": {
        "nodes": ["use_declaration"],
    },
    "java": {
        "nodes": ["import_declaration"],
    },
    "c": {
        "nodes": ["preproc_include"],
    },
    "cpp": {
        "nodes": ["preproc_include"],
    },
    "c_sharp": {
        "nodes": ["using_directive"],
    },
    "csharp": {
        "nodes": ["using_directive"],
    },
    "ruby": {
        "nodes": ["call"],  # require/require_relative are method calls
    },
    "kotlin": {
        "nodes": ["import_header"],
    },
    "swift": {
        "nodes": ["import_declaration"],
    },
    "scala": {
        "nodes": ["import_declaration"],
    },
    "php": {
        "nodes": ["namespace_use_declaration"],
    },
}


def _ts_extract_imports(language: str, text: str) -> List[str]:
    """Extract imports using tree-sitter AST traversal.

    Uses proper AST node structure instead of regex.
    """
    from scripts.ingest.tree_sitter import _ts_parser

    if language not in _TS_IMPORT_CONFIG:
        return _extract_imports(language, text)

    parser = _ts_parser(language)
    if not parser:
        return _extract_imports(language, text)

    data = text.encode("utf-8")
    try:
        tree = parser.parse(data)
        if tree is None:
            return _extract_imports(language, text)
        root = tree.root_node
    except Exception:
        return _extract_imports(language, text)

    config = _TS_IMPORT_CONFIG[language]
    import_nodes = set(config["nodes"])

    def node_text(n):
        return data[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    def find_string_content(node) -> str:
        """Find string literal content from a node."""
        # Try to find string/string_literal child
        for child in node.children:
            if child.type in ("string", "string_literal", "interpreted_string_literal"):
                # Get the content without quotes
                text = node_text(child)
                # Strip quotes
                if len(text) >= 2 and text[0] in ('"', "'", "`"):
                    return text[1:-1]
                return text
            # Recurse for nested structures
            if child.type in ("import_spec", "import_clause"):
                result = find_string_content(child)
                if result:
                    return result
        return ""

    def extract_scoped_path(node) -> str:
        """Extract path from scoped_identifier (Rust: std::io)."""
        parts = []
        def collect(n):
            if n.type == "identifier":
                parts.append(node_text(n))
            elif n.type in ("crate", "self", "super"):
                parts.append(node_text(n))
            for c in n.children:
                if c.type not in ("::", ".", ";"):
                    collect(c)
        collect(node)
        return "::".join(parts) if parts else node_text(node)

    imports: List[str] = []

    def walk(n):
        ntype = n.type

        if ntype in import_nodes:
            if language == "python":
                # Python: import X or from X import Y
                if ntype == "import_statement":
                    # Look for dotted_name
                    for child in n.children:
                        if child.type == "dotted_name":
                            imports.append(node_text(child))
                        elif child.type == "aliased_import":
                            name = child.child_by_field_name("name")
                            if name:
                                imports.append(node_text(name))
                elif ntype == "import_from_statement":
                    module = n.child_by_field_name("module_name")
                    if module:
                        imports.append(node_text(module))
                    else:
                        for child in n.children:
                            if child.type == "dotted_name":
                                imports.append(node_text(child))
                                break

            elif language in ("javascript", "typescript", "tsx"):
                # JS/TS: import ... from "source"
                source = n.child_by_field_name("source")
                if source:
                    text = node_text(source)
                    # Strip quotes
                    if len(text) >= 2:
                        imports.append(text[1:-1])
                else:
                    # Fallback: find string child
                    result = find_string_content(n)
                    if result:
                        imports.append(result)

            elif language == "go":
                # Go: import "path" or import ( "path1" "path2" )
                for child in n.children:
                    if child.type == "import_spec":
                        result = find_string_content(child)
                        if result:
                            imports.append(result)
                    elif child.type == "import_spec_list":
                        for spec in child.children:
                            if spec.type == "import_spec":
                                result = find_string_content(spec)
                                if result:
                                    imports.append(result)
                    elif child.type == "interpreted_string_literal":
                        text = node_text(child)
                        if len(text) >= 2:
                            imports.append(text[1:-1])

            elif language == "rust":
                # Rust: use std::io;
                for child in n.children:
                    if child.type in ("scoped_identifier", "identifier", "use_wildcard"):
                        imports.append(extract_scoped_path(child))
                    elif child.type == "use_list":
                        # use std::{io, fs}
                        imports.append(node_text(child))

            elif language == "java":
                # Java: import java.util.List;
                for child in n.children:
                    if child.type == "scoped_identifier":
                        imports.append(node_text(child).replace(" ", ""))

            elif language in ("c", "cpp"):
                # C/C++: #include <file> or #include "file"
                path = n.child_by_field_name("path")
                if path:
                    text = node_text(path)
                    # Strip < > or " "
                    if len(text) >= 2:
                        imports.append(text[1:-1])
                else:
                    # Find string_literal or system_lib_string
                    for child in n.children:
                        if child.type in ("string_literal", "system_lib_string"):
                            text = node_text(child)
                            if len(text) >= 2:
                                imports.append(text[1:-1])

            elif language in ("c_sharp", "csharp"):
                # C#: using System.Text;
                for child in n.children:
                    if child.type in ("identifier", "qualified_name"):
                        imports.append(node_text(child))

            elif language == "ruby":
                # Ruby: require is a method call
                if ntype == "call":
                    # Check if it's require/require_relative
                    method = n.child_by_field_name("method")
                    if method and node_text(method) in ("require", "require_relative", "load"):
                        args = n.child_by_field_name("arguments")
                        if args:
                            result = find_string_content(args)
                            if result:
                                imports.append(result)

            elif language == "kotlin":
                # Kotlin: import java.util.List
                for child in n.children:
                    if child.type == "identifier":
                        imports.append(node_text(child))
                    elif child.type == "import_alias":
                        # import foo.Bar as Baz - get foo.Bar
                        continue
                # Try to get the full path from the node text
                full = node_text(n).replace("import ", "").strip()
                if full and not full.startswith("import"):
                    imports.append(full.split(" as ")[0].strip())

            elif language == "swift":
                # Swift: import Foundation
                for child in n.children:
                    if child.type == "identifier":
                        imports.append(node_text(child))

            elif language == "scala":
                # Scala: import java.util.List or import java.util._
                full = node_text(n).replace("import ", "").strip()
                if full:
                    imports.append(full)

            elif language == "php":
                # PHP: use Namespace\ClassName;
                for child in n.children:
                    if child.type == "namespace_name":
                        imports.append(node_text(child).replace("\\\\", "\\"))
                    elif child.type == "qualified_name":
                        imports.append(node_text(child).replace("\\\\", "\\"))

        for c in n.children:
            walk(c)

    walk(root)

    # Deduplicate
    seen = set()
    result = []
    for x in imports:
        if x and x not in seen:
            seen.add(x)
            result.append(x)
    return result[:200]


def _ts_extract_calls_generic(language: str, text: str) -> List[str]:
    """Extract function/method calls using tree-sitter AST traversal.

    Uses proper AST node structure - no regex parsing of code text.
    Captures both base names (method) and qualified names (obj.method).
    """
    from scripts.ingest.tree_sitter import _ts_parser

    parser = _ts_parser(language)
    if not parser:
        return _extract_calls(language, text)

    data = text.encode("utf-8")
    try:
        tree = parser.parse(data)
        if tree is None:
            return _extract_calls(language, text)
        root = tree.root_node
    except Exception:
        return _extract_calls(language, text)

    # Get language config
    config = _TS_LANG_CONFIG.get(language, _TS_DEFAULT_CONFIG)
    call_types = set(config["calls"])
    constructor_types = set(config["constructors"])
    member_map = config["member"]  # node_type -> (object_field, property_field)

    def node_text(n):
        return data[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    def is_valid_identifier(name: str) -> bool:
        """Check if name is a valid identifier (no operators, keywords check skipped for speed)."""
        if not name or len(name) > 100:
            return False
        first = name[0]
        if not (first.isalpha() or first == "_"):
            return False
        return all(c.isalnum() or c == "_" for c in name)

    def extract_from_member(node) -> List[str]:
        """Extract method name and qualified name from member access node using AST structure."""
        names = []
        node_type = node.type

        if node_type not in member_map:
            # Unknown member type - try common field names
            prop = (
                node.child_by_field_name("property") or
                node.child_by_field_name("field") or
                node.child_by_field_name("name") or
                node.child_by_field_name("attribute") or
                node.child_by_field_name("method")
            )
            if prop and prop.type == "identifier":
                method = node_text(prop)
                if is_valid_identifier(method):
                    names.append(method)
            return names

        obj_field, prop_field = member_map[node_type]
        prop_node = node.child_by_field_name(prop_field)
        obj_node = node.child_by_field_name(obj_field)

        # Get method/property name
        method = ""
        if prop_node:
            if prop_node.type in ("identifier", "property_identifier", "field_identifier"):
                method = node_text(prop_node)
            else:
                # Nested - recurse
                nested = extract_from_member(prop_node)
                if nested:
                    method = nested[0]

        if method and is_valid_identifier(method):
            names.append(method)

            # Get object name for qualified form (skip self/this/cls)
            if obj_node:
                obj_name = ""
                if obj_node.type in ("identifier", "this", "self"):
                    obj_name = node_text(obj_node)
                elif obj_node.type in member_map:
                    # Chained: a.b.c -> get "b" from a.b
                    nested = extract_from_member(obj_node)
                    if nested:
                        obj_name = nested[0]

                # Add qualified if object isn't self/this/cls
                if obj_name and obj_name not in ("self", "this", "cls", "super"):
                    if is_valid_identifier(obj_name):
                        names.append(f"{obj_name}.{method}")

        return names

    def extract_function_name(func_node) -> List[str]:
        """Extract function name(s) from the function part of a call expression."""
        if func_node is None:
            return []

        node_type = func_node.type

        # Simple identifier: foo()
        if node_type in ("identifier", "constant", "method_identifier"):
            name = node_text(func_node)
            if is_valid_identifier(name):
                return [name]
            return []

        # Member access: obj.method()
        if node_type in member_map:
            return extract_from_member(func_node)

        # Try common field names for unknown node types
        prop = (
            func_node.child_by_field_name("property") or
            func_node.child_by_field_name("field") or
            func_node.child_by_field_name("name")
        )
        if prop:
            name = node_text(prop)
            if is_valid_identifier(name):
                return [name]

        return []

    def extract_constructor_type(node) -> List[str]:
        """Extract class name from constructor call (new Foo())."""
        type_node = (
            node.child_by_field_name("constructor") or
            node.child_by_field_name("type") or
            node.child_by_field_name("name")
        )
        if not type_node:
            return []

        # Handle generic types: get the base identifier
        if type_node.type in ("identifier", "type_identifier"):
            name = node_text(type_node)
            if is_valid_identifier(name):
                return [name]

        # Generic type: ArrayList<String> - find the identifier child
        if type_node.type in ("generic_type", "parameterized_type"):
            for child in type_node.children:
                if child.type in ("identifier", "type_identifier"):
                    name = node_text(child)
                    if is_valid_identifier(name):
                        return [name]
                    break

        return []

    calls: List[str] = []

    def walk(n):
        ntype = n.type

        # Regular function/method calls
        if ntype in call_types:
            # Java/Kotlin: method_invocation has object+name directly on the node
            if ntype == "method_invocation":
                name_node = n.child_by_field_name("name")
                obj_node = n.child_by_field_name("object")
                if name_node and name_node.type == "identifier":
                    method = node_text(name_node)
                    if is_valid_identifier(method):
                        calls.append(method)
                        # Add qualified name
                        if obj_node and obj_node.type == "identifier":
                            obj = node_text(obj_node)
                            if is_valid_identifier(obj):
                                calls.append(f"{obj}.{method}")
            else:
                # Try standard field names for the function part
                func = (
                    n.child_by_field_name("function") or
                    n.child_by_field_name("method") or
                    n.child_by_field_name("name") or
                    n.child_by_field_name("callee") or
                    n.child_by_field_name("receiver")
                )
                if func:
                    calls.extend(extract_function_name(func))
                else:
                    # Ruby/other: first relevant child
                    for child in n.children:
                        if child.type in ("identifier", "constant", "method_identifier"):
                            name = node_text(child)
                            if is_valid_identifier(name):
                                calls.append(name)
                            break
                        elif child.type in member_map:
                            calls.extend(extract_from_member(child))
                            break

        # Constructor calls
        elif ntype in constructor_types:
            calls.extend(extract_constructor_type(n))

        # Rust macros
        elif ntype == "macro_invocation":
            macro = n.child_by_field_name("macro")
            if macro:
                name = node_text(macro)
                # Strip trailing ! if present in text
                name = name.rstrip("!")
                if is_valid_identifier(name):
                    calls.append(name)

        # Bash commands
        elif ntype == "command":
            cmd = n.child_by_field_name("name")
            if cmd:
                name = node_text(cmd)
                # Allow hyphens in command names
                if name and all(c.isalnum() or c in "_-" for c in name):
                    calls.append(name)

        for c in n.children:
            walk(c)

    walk(root)

    # Deduplicate preserving order
    seen = set()
    result = []
    for x in calls:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result[:200]


def _get_imports_calls(language: str, text: str) -> Tuple[List[str], List[str]]:
    """Get imports and calls for a file, using tree-sitter when available."""
    from scripts.ingest.tree_sitter import _use_tree_sitter

    # Use tree-sitter for Python (specialized) or generic for other supported languages
    if _use_tree_sitter():
        if language == "python":
            return _ts_extract_imports_calls_python(text)
        elif language in _TS_CALL_LANGUAGES:
            # Use tree-sitter for both imports and calls
            imports = _ts_extract_imports(language, text)
            calls = _ts_extract_calls_generic(language, text)
            return imports, calls

    return _extract_imports(language, text), _extract_calls(language, text)


def _ts_extract_imports_calls_python(text: str) -> Tuple[List[str], List[str]]:
    """Extract imports and calls from Python using tree-sitter AST traversal.

    Uses proper AST node structure - no regex parsing of code text for calls.
    """
    from scripts.ingest.tree_sitter import _ts_parser

    parser = _ts_parser("python")
    if not parser:
        return [], []
    data = text.encode("utf-8")
    try:
        tree = parser.parse(data)
        if tree is None:
            return [], []
        root = tree.root_node
    except (ValueError, Exception):
        return [], []

    def node_text(n):
        return data[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    def is_valid_identifier(name: str) -> bool:
        if not name or len(name) > 100:
            return False
        first = name[0]
        if not (first.isalpha() or first == "_"):
            return False
        return all(c.isalnum() or c == "_" for c in name)

    imports: List[str] = []
    calls: List[str] = []

    def extract_python_call(func_node) -> List[str]:
        """Extract call names from Python function node using AST structure."""
        if func_node is None:
            return []

        # Simple identifier: foo()
        if func_node.type == "identifier":
            name = node_text(func_node)
            if is_valid_identifier(name):
                return [name]
            return []

        # Attribute access: obj.method()
        if func_node.type == "attribute":
            names = []
            # Python attribute node has: object and attribute fields
            attr_node = func_node.child_by_field_name("attribute")
            value_node = func_node.child_by_field_name("object")

            method = ""
            if attr_node and attr_node.type == "identifier":
                method = node_text(attr_node)

            if method and is_valid_identifier(method):
                names.append(method)

                # Get object for qualified name
                if value_node:
                    obj_name = ""
                    if value_node.type == "identifier":
                        obj_name = node_text(value_node)
                    elif value_node.type == "attribute":
                        # Chained: a.b.method - get "b" from a.b
                        nested_attr = value_node.child_by_field_name("attribute")
                        if nested_attr and nested_attr.type == "identifier":
                            obj_name = node_text(nested_attr)

                    # Add qualified if not self/cls/super
                    if obj_name and obj_name not in ("self", "cls", "super"):
                        if is_valid_identifier(obj_name):
                            names.append(f"{obj_name}.{method}")

            return names

        return []

    def extract_import_module(node) -> str:
        """Extract module name from import node using AST structure."""
        # For import_statement: import foo.bar
        # For import_from_statement: from foo.bar import baz
        # Look for dotted_name or aliased_import children
        for child in node.children:
            if child.type == "dotted_name":
                return node_text(child)
            elif child.type == "aliased_import":
                # import foo as f -> get "foo"
                name_child = child.child_by_field_name("name")
                if name_child:
                    return node_text(name_child)
        return ""

    def walk(n):
        t = n.type
        if t == "import_statement":
            mod = extract_import_module(n)
            if mod:
                imports.append(mod)
        elif t == "import_from_statement":
            # from X import Y -> get X
            module_node = n.child_by_field_name("module_name")
            if module_node:
                imports.append(node_text(module_node))
            else:
                # Fallback: look for dotted_name
                for child in n.children:
                    if child.type == "dotted_name":
                        imports.append(node_text(child))
                        break
        elif t == "call":
            func = n.child_by_field_name("function")
            if func:
                calls.extend(extract_python_call(func))

        for c in n.children:
            walk(c)

    walk(root)

    # Deduplicate preserving order
    seen = set()
    calls_dedup = []
    for x in calls:
        if x not in seen:
            seen.add(x)
            calls_dedup.append(x)
    return imports[:200], calls_dedup[:200]


def _get_host_path_from_origin(workspace_path: str, repo_name: str = None) -> Optional[str]:
    """Get client host_path from origin source_path in workspace state."""
    try:
        from scripts.workspace_state import get_workspace_state
        state = get_workspace_state(workspace_path, repo_name)
        if state and state.get("origin", {}).get("source_path"):
            return state["origin"]["source_path"]
    except Exception:
        pass
    return None


def _compute_host_and_container_paths(cur_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Compute host_path and container_path for a given absolute path."""
    _host_root = str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
    if ":" in _host_root:
        _host_root = ""
    _host_path: Optional[str] = None
    _container_path: Optional[str] = None
    _origin_client_path: Optional[str] = None

    try:
        if cur_path.startswith("/work/"):
            _parts = cur_path[6:].split("/")
            if len(_parts) >= 2:
                _repo_name = _parts[0]
                _workspace_path = f"/work/{_repo_name}"
                _origin_client_path = _get_host_path_from_origin(
                    _workspace_path, _repo_name
                )
    except Exception:
        _origin_client_path = None

    try:
        if cur_path.startswith("/work/") and (_host_root or _origin_client_path):
            _rel = cur_path[len("/work/"):]
            if _origin_client_path:
                _parts = _rel.split("/", 1)
                _tail = _parts[1] if len(_parts) > 1 else ""
                _base = _origin_client_path.rstrip("/")
                _host_path = (
                    os.path.realpath(os.path.join(_base, _tail)) if _tail else _base
                )
            else:
                _host_path = os.path.realpath(os.path.join(_host_root, _rel))
            _container_path = cur_path
        else:
            _host_path = cur_path
            if (
                (_host_root or _origin_client_path)
                and cur_path.startswith(((_origin_client_path or _host_root) + "/"))
            ):
                _rel = cur_path[len((_origin_client_path or _host_root)) + 1:]
                _container_path = "/work/" + _rel
    except Exception:
        _host_path = cur_path
        _container_path = cur_path if cur_path.startswith("/work/") else None

    return _host_path, _container_path
