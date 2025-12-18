from __future__ import annotations


# Import repository detection from workspace_state to avoid duplication
def _detect_repo_name_from_path(path: Path) -> str:
    """Wrapper function to use workspace_state repository detection."""
    try:
        from scripts.workspace_state import _extract_repo_name_from_path as _ws_detect
        return _ws_detect(str(path))
    except ImportError:
        # Fallback for when workspace_state is not available
        return path.name if path.is_dir() else path.parent.name


#!/usr/bin/env python3
import os
import sys
import argparse
import hashlib
import re
import ast
import time
from pathlib import Path
from typing import List, Dict, Iterable, Any, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# Ensure project root is on sys.path when run as a script (so 'scripts' package imports work)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from qdrant_client import QdrantClient, models

# Use embedder factory for Qwen3 support; fallback to direct fastembed
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False

# Import TextEmbedding for type hints and backward compatibility with tests
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fastembed import TextEmbedding

# Always try to import TextEmbedding for backward compatibility with tests
try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # type: ignore


from datetime import datetime

# Import critical multi-repo functions first
try:
    from scripts.workspace_state import (
        is_multi_repo_mode,
        get_collection_name,
        logical_repo_reuse_enabled,
    )
except ImportError:
    is_multi_repo_mode = None  # type: ignore
    get_collection_name = None  # type: ignore

    def logical_repo_reuse_enabled() -> bool:  # type: ignore[no-redef]
        return False

# Import watcher's repo detection for surgical fix
try:
    from scripts.watch_index import _detect_repo_for_file, _get_collection_for_file
except ImportError:
    _detect_repo_for_file = None  # type: ignore
    _get_collection_for_file = None  # type: ignore

# Import other workspace state functions (optional)
try:
    from scripts.workspace_state import (
        log_activity,
        get_cached_file_hash,
        set_cached_file_hash,
        remove_cached_file,
        update_indexing_status,
        update_workspace_state,
        get_cached_symbols,
        set_cached_symbols,
        remove_cached_symbols,
        compare_symbol_changes,
        get_cached_pseudo,
        set_cached_pseudo,
        update_symbols_with_pseudo,
        get_workspace_state,
        get_cached_file_meta,
    )
except ImportError:
    # State integration is optional; continue if not available
    log_activity = None  # type: ignore
    get_cached_file_hash = None  # type: ignore
    set_cached_file_hash = None  # type: ignore
    remove_cached_file = None  # type: ignore
    update_indexing_status = None  # type: ignore
    update_workspace_state = None  # type: ignore
    get_cached_symbols = None  # type: ignore
    set_cached_symbols = None  # type: ignore
    remove_cached_symbols = None  # type: ignore
    get_cached_pseudo = None  # type: ignore
    set_cached_pseudo = None  # type: ignore
    update_symbols_with_pseudo = None  # type: ignore
    compare_symbol_changes = None  # type: ignore
    get_workspace_state = None  # type: ignore
    get_cached_file_meta = None  # type: ignore

# Bloom filter for fast index skip (feature flag: BLOOM_SKIP_ENABLED=1)
_BLOOM_SKIP_ENABLED = os.environ.get("BLOOM_SKIP_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
_bloom_index_instance = None

def _get_bloom_index():
    """Lazy-load bloom index singleton."""
    global _bloom_index_instance
    if _bloom_index_instance is None and _BLOOM_SKIP_ENABLED:
        try:
            from scripts.bloom_index import BloomIndex
            ws_path = os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work"
            _bloom_index_instance = BloomIndex.load_or_create(ws_path)
        except Exception:
            pass
    return _bloom_index_instance

def _bloom_might_contain(file_hash: str) -> bool:
    """Check if hash might be in bloom filter. Returns False if bloom disabled or hash definitely not present."""
    if not _BLOOM_SKIP_ENABLED or not file_hash:
        return True  # Conservative: assume might be present
    bloom = _get_bloom_index()
    if bloom is None:
        return True
    return bloom.might_contain(file_hash)

def _bloom_add(file_hash: str) -> None:
    """Add hash to bloom filter."""
    if not _BLOOM_SKIP_ENABLED or not file_hash:
        return
    bloom = _get_bloom_index()
    if bloom:
        bloom.add(file_hash)

def _bloom_save() -> None:
    """Persist bloom filter to disk."""
    if not _BLOOM_SKIP_ENABLED:
        return
    bloom = _get_bloom_index()
    if bloom:
        bloom.save()

# Optional Tree-sitter import (graceful fallback) - tree-sitter 0.25+ API
_TS_LANGUAGES: Dict[str, Any] = {}
_TS_AVAILABLE = False
try:
    from tree_sitter import Parser, Language  # type: ignore

    def _load_ts_language(mod: Any, *, preferred: list[str] | None = None) -> Any | None:
        """Return a tree-sitter Language instance from a per-language package.

        Different packages expose different entrypoints (e.g. language(),
        language_typescript(), language_tsx()).
        """
        preferred = preferred or []
        candidates: list[Any] = []
        if getattr(mod, "language", None) is not None and callable(getattr(mod, "language")):
            candidates.append(getattr(mod, "language"))
        for name in preferred:
            fn = getattr(mod, name, None)
            if fn is not None and callable(fn):
                candidates.append(fn)
        # Last resort: scan for any callable language* attribute
        for name in dir(mod):
            if not name.startswith("language"):
                continue
            fn = getattr(mod, name, None)
            if fn is not None and callable(fn):
                candidates.append(fn)

        for fn in candidates:
            try:
                raw_lang = fn()
                return raw_lang if isinstance(raw_lang, Language) else Language(raw_lang)
            except Exception:
                continue
        return None

    # Import all available language packages
    for lang_name, pkg_name in [
        ("python", "tree_sitter_python"),
        ("javascript", "tree_sitter_javascript"),
        ("typescript", "tree_sitter_typescript"),
        ("go", "tree_sitter_go"),
        ("rust", "tree_sitter_rust"),
        ("java", "tree_sitter_java"),
        ("c", "tree_sitter_c"),
        ("cpp", "tree_sitter_cpp"),
        ("ruby", "tree_sitter_ruby"),
        ("c_sharp", "tree_sitter_c_sharp"),
        ("bash", "tree_sitter_bash"),
        ("json", "tree_sitter_json"),
        ("yaml", "tree_sitter_yaml"),
        ("html", "tree_sitter_html"),
        ("css", "tree_sitter_css"),
    ]:
        try:
            mod = __import__(pkg_name)
            preferred: list[str] = []
            if lang_name == "typescript":
                preferred = ["language_typescript"]
            elif lang_name == "c_sharp":
                preferred = ["language_c_sharp", "language_csharp"]
            lang = _load_ts_language(mod, preferred=preferred)
            if lang is not None:
                _TS_LANGUAGES[lang_name] = lang
                # Also load TSX if provided by the typescript package
                if lang_name == "typescript":
                    tsx_lang = _load_ts_language(mod, preferred=["language_tsx"])
                    if tsx_lang is not None:
                        _TS_LANGUAGES["tsx"] = tsx_lang
        except Exception:
            pass  # Language package not installed

    # Add aliases
    if "javascript" in _TS_LANGUAGES:
        _TS_LANGUAGES["jsx"] = _TS_LANGUAGES["javascript"]
    if "c_sharp" in _TS_LANGUAGES:
        _TS_LANGUAGES["csharp"] = _TS_LANGUAGES["c_sharp"]
    if "bash" in _TS_LANGUAGES:
        _TS_LANGUAGES["shell"] = _TS_LANGUAGES["bash"]
        _TS_LANGUAGES["sh"] = _TS_LANGUAGES["bash"]

    _TS_AVAILABLE = len(_TS_LANGUAGES) > 0
except Exception:  # pragma: no cover
    Parser = None  # type: ignore
    Language = None  # type: ignore
    _TS_LANGUAGES = {}
    _TS_AVAILABLE = False

# Import AST analyzer for enhanced semantic chunking
try:
    from scripts.ast_analyzer import get_ast_analyzer, chunk_code_semantically
    _AST_ANALYZER_AVAILABLE = True
except ImportError:
    _AST_ANALYZER_AVAILABLE = False


_TS_WARNED = False


def _use_tree_sitter() -> bool:
    global _TS_WARNED
    val = os.environ.get("USE_TREE_SITTER")
    # Default ON when libs are available; allow explicit disable via 0/false
    if val is None or str(val).strip() == "":
        want = True
    else:
        want = str(val).strip().lower() in {"1", "true", "yes", "on"}
    if want and not _TS_AVAILABLE and not _TS_WARNED:
        print(
            "[WARN] USE_TREE_SITTER=1 but tree-sitter libs not available; falling back to regex heuristics"
        )
        _TS_WARNED = True
    return _TS_AVAILABLE and want


CODE_EXTS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
    ".sql": "sql",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".json": "json",
    ".tf": "terraform",
    ".csx": "csharp",
    ".cshtml": "razor",
    ".razor": "razor",
    ".csproj": "xml",
    ".config": "xml",
    ".resx": "xml",
}

# --- Named vector config ---
LEX_VECTOR_NAME = os.environ.get("LEX_VECTOR_NAME", "lex")
LEX_VECTOR_DIM = int(os.environ.get("LEX_VECTOR_DIM", "4096") or 4096)
# Optional mini vector (ReFRAG-style gating); conditionally created by REFRAG_MODE
MINI_VECTOR_NAME = os.environ.get("MINI_VECTOR_NAME", "mini")
MINI_VEC_DIM = int(os.environ.get("MINI_VEC_DIM", "64") or 64)

# Lightweight hashing-trick sparse vector (fixed-size dense) for lexical signals
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "to",
    "with",
    "by",
    "is",
    "are",
    "be",
    "this",
    "that",
}

# Random +/-1 projection (Gaussian-sign) for compact mini vectors; cached by (in_dim,out_dim,seed)
_MINI_PROJ_CACHE: dict[tuple[int, int, int], list[list[float]]] = {}


def _get_mini_proj(
    in_dim: int, out_dim: int, seed: int | None = None
) -> list[list[float]]:
    import math, random

    s = int(os.environ.get("MINI_VEC_SEED", "1337")) if seed is None else int(seed)
    key = (in_dim, out_dim, s)
    M = _MINI_PROJ_CACHE.get(key)
    if M is None:
        rnd = random.Random(s)
        scale = 1.0 / math.sqrt(out_dim)
        # Dense Rademacher matrix (+/-1) scaled; good enough for fast gating
        M = [
            [scale * (1.0 if rnd.random() < 0.5 else -1.0) for _ in range(out_dim)]
            for _ in range(in_dim)
        ]
        _MINI_PROJ_CACHE[key] = M
    return M


def project_mini(vec: list[float], out_dim: int | None = None) -> list[float]:
    import math

    if not vec:
        return [0.0] * (int(out_dim or MINI_VEC_DIM))
    od = int(out_dim or MINI_VEC_DIM)
    M = _get_mini_proj(len(vec), od)
    out = [0.0] * od
    # y = x @ M
    for i, val in enumerate(vec):
        if val == 0.0:
            continue
        row = M[i]
        for j in range(od):
            out[j] += val * row[j]
    # L2 normalize to keep scale consistent
    norm = (sum(x * x for x in out) or 0.0) ** 0.5 or 1.0
    return [x / norm for x in out]


def _split_ident_lex(s: str):
    parts = re.split(r"[^A-Za-z0-9]+", s)
    out = []
    for p in parts:
        if not p:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
        out.extend([x for x in segs if x])
    return [x.lower() for x in out if x and x.lower() not in _STOP]


def _lex_hash_vector(text: str, dim: int = LEX_VECTOR_DIM) -> list[float]:
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    # Tokenize identifiers & words
    toks = _split_ident_lex(text)
    if not toks:
        return vec
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize (avoid huge magnitudes)
    import math

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _git_metadata(file_path: Path) -> tuple[int, int, int]:
    """Return (last_modified_at, churn_count, author_count) using git when available.
    Fallbacks to fs mtime and zeros when not in a repo.
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


# --- Exclusions (configurable) ---
# Defaults can be overridden by .qdrantignore, env, or CLI
_DEFAULT_EXCLUDE_DIRS = [
    "/models",
    "/.vs",
    "/.vscode",
    "/.cache",
    "/.codebase",
    "/.remote-git",
    "/node_modules",
    "/dist",
    "/build",
    "/.venv",
    "/venv",
    "/py-venv",
    "/site-packages",
    "/__pycache__",
    "bin",
    "obj",
    "TestResults",
    "/.git",
]
# Glob patterns for directories (matched against basename)
_DEFAULT_EXCLUDE_DIR_GLOBS = [
    ".venv*",  # .venv, .venv311, .venv39, etc.
]
_DEFAULT_EXCLUDE_FILES = [
    "*.onnx",
    "*.bin",
    "*.safetensors",
    "tokenizer.json",
    "*.whl",
    "*.tar.gz",
]

_ANY_DEPTH_EXCLUDE_DIR_NAMES = {
    ".git",
    ".remote-git",
    ".codebase",
    "node_modules",
}

def _should_skip_explicit_file_by_excluder(file_path: Path) -> bool:
    try:
        p = file_path if isinstance(file_path, Path) else Path(str(file_path))
    except Exception:
        return False

    root = None
    try:
        parts = list(p.parts)
        if ".remote-git" in parts:
            i = parts.index(".remote-git")
            root = Path(*parts[:i]) if i > 0 else Path("/")
    except Exception:
        root = None

    if root is None:
        try:
            s = str(p)
            if s.startswith("/work/"):
                slug = s[len("/work/") :].split("/", 1)[0]
                root = (Path("/work") / slug) if slug else None
        except Exception:
            root = None

    if root is None:
        try:
            ws = (os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "").strip()
            if ws:
                ws_path = Path(ws).resolve()
                pr = p.resolve()
                if pr == ws_path or ws_path in pr.parents:
                    root = ws_path
        except Exception:
            root = None

    if root is None:
        try:
            pr = p.resolve()
            for anc in [pr.parent] + list(pr.parents):
                if (anc / ".codebase").exists():
                    root = anc
                    break
        except Exception:
            root = None

    if not root or str(root) == "/":
        return False

    try:
        rel = p.resolve().relative_to(root.resolve()).as_posix().lstrip("/")
    except Exception:
        return False
    if not rel:
        return False

    try:
        excl = _Excluder(root)
        cur = ""
        for seg in [x for x in rel.split("/") if x][:-1]:
            cur = cur + "/" + seg
            if excl.exclude_dir(cur):
                return True
        return excl.exclude_file(rel)
    except Exception:
        return False


def _env_truthy(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def hash_id(text: str, path: str, start: int, end: int) -> int:
    h = hashlib.sha1(
        f"{path}:{start}-{end}\n{text}".encode("utf-8", errors="ignore")
    ).hexdigest()
    return int(h[:16], 16)


class _Excluder:
    def __init__(self, root: Path):
        self.root = root
        self.dir_prefixes = []  # absolute like /path/sub
        self.dir_globs = []  # fnmatch patterns for directory names
        self.file_globs = []  # fnmatch patterns

        # Defaults
        use_defaults = _env_truthy(os.environ.get("QDRANT_DEFAULT_EXCLUDES"), True)
        if use_defaults:
            self.dir_prefixes.extend(_DEFAULT_EXCLUDE_DIRS)
            self.dir_globs.extend(_DEFAULT_EXCLUDE_DIR_GLOBS)
            self.file_globs.extend(_DEFAULT_EXCLUDE_FILES)

        # .qdrantignore
        ignore_file = os.environ.get("QDRANT_IGNORE_FILE", ".qdrantignore")
        ig_path = root / ignore_file
        if ig_path.exists():
            for raw in ig_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                self._add_pattern(line)

        # Extra excludes via env (comma separated)
        extra = os.environ.get("QDRANT_EXCLUDES", "").strip()
        if extra:
            for pat in [p.strip() for p in extra.split(",") if p.strip()]:
                self._add_pattern(pat)

    def _add_pattern(self, pat: str):
        # Normalize to leading-slash for prefixes
        has_wild = any(ch in pat for ch in "*?[")
        if pat.startswith("/") and not has_wild:
            # Treat as directory prefix if no wildcard
            self.dir_prefixes.append(pat.rstrip("/"))
        else:
            # Treat as file glob (match against relpath and basename)
            self.file_globs.append(pat.lstrip("/"))

    def exclude_dir(self, rel: str) -> bool:
        import fnmatch

        # rel like /a/b
        for pref in self.dir_prefixes:
            if rel == pref or rel.startswith(pref + "/"):
                return True

        base = rel.rsplit("/", 1)[-1]

        # Match directory name against dir_globs (e.g., .venv*)
        for g in self.dir_globs:
            if fnmatch.fnmatch(base, g):
                return True

        # Treat single-segment dir prefixes (e.g. "/.git", "/node_modules") as
        # "exclude this directory name anywhere". This matters when indexing a
        # workspace root that contains multiple repos, e.g. /work/<repo>/.git.
        try:
            if base in _ANY_DEPTH_EXCLUDE_DIR_NAMES and ("/" + base) in self.dir_prefixes:
                return True
        except Exception:
            pass

        # Also allow dir name-only patterns in file_globs (e.g., node_modules)
        for g in self.file_globs:
            # Match bare dir names without wildcards
            if g and all(ch not in g for ch in "*?[") and base == g:
                return True
        return False

    def exclude_file(self, rel: str) -> bool:
        import fnmatch

        # Try matching whole rel path and basename
        base = rel.rsplit("/", 1)[-1]
        for g in self.file_globs:
            if fnmatch.fnmatch(rel.lstrip("/"), g) or fnmatch.fnmatch(base, g):
                return True
        return False


def iter_files(root: Path) -> Iterable[Path]:
    # Allow passing a single file
    if root.is_file():
        if root.suffix.lower() in CODE_EXTS and not _should_skip_explicit_file_by_excluder(root):
            yield root
        return

    excl = _Excluder(root)
    # Use os.walk to prune directories for performance
    # NOTE: avoid Path.resolve()/realpath here; on network filesystems (e.g. CephFS)
    # it can trigger expensive metadata calls during large unchanged indexing runs.
    try:
        root_abs = os.path.abspath(str(root))
    except Exception:
        root_abs = str(root)

    for dirpath, dirnames, filenames in os.walk(root_abs):
        # Compute rel path like /a/b from root without resolving symlinks
        try:
            rel = os.path.relpath(dirpath, root_abs)
        except Exception:
            rel = "."
        if rel in (".", ""):
            rel_dir = "/"
        else:
            rel_dir = "/" + rel.replace(os.sep, "/")
        # Prune excluded directories in-place
        keep = []
        for d in dirnames:
            rel = (rel_dir.rstrip("/") + "/" + d).replace("//", "/")
            if excl.exclude_dir(rel):
                continue
            keep.append(d)
        dirnames[:] = keep

        for f in filenames:
            p = Path(dirpath) / f
            if p.suffix.lower() not in CODE_EXTS:
                continue
            relf = (rel_dir.rstrip("/") + "/" + f).replace("//", "/")
            if excl.exclude_file(relf):
                continue
            yield p


def chunk_lines(text: str, max_lines: int = 120, overlap: int = 20) -> List[Dict]:
    lines = text.splitlines()
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        j = min(n, i + max_lines)
        chunk = "\n".join(lines[i:j])
        chunks.append({"text": chunk, "start": i + 1, "end": j})
        if j == n:
            break

        i = max(j - overlap, i + 1)
    return chunks


def chunk_semantic(
    text: str, language: str, max_lines: int = 120, overlap: int = 20
) -> List[Dict]:
    """AST-aware chunking that tries to keep complete functions/classes together."""
    # Try enhanced AST analyzer first (if available)
    use_enhanced = os.environ.get("INDEX_USE_ENHANCED_AST", "1").lower() in {"1", "true", "yes", "on"}
    if use_enhanced and _AST_ANALYZER_AVAILABLE and language in ("python", "javascript", "typescript"):
        try:
            chunks = chunk_code_semantically(text, language, max_lines, overlap)
            # Convert to expected format
            return [
                {
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                    "is_semantic": c.get("is_semantic", True)
                }
                for c in chunks
            ]
        except Exception as e:
            if os.environ.get("DEBUG_INDEXING"):
                print(f"[DEBUG] Enhanced AST chunking failed, falling back: {e}")
    
    if not _use_tree_sitter() or language not in ("python", "javascript", "typescript"):
        # Fallback to line-based chunking
        return chunk_lines(text, max_lines, overlap)

    lines = text.splitlines()
    n = len(lines)

    # Extract symbols with line ranges
    symbols = _extract_symbols(language, text)
    if not symbols:
        return chunk_lines(text, max_lines, overlap)

    # Sort symbols by start line
    symbols.sort(key=lambda s: s.start)

    chunks = []
    i = 0  # Current line index (0-based)

    while i < n:
        chunk_start = i + 1  # 1-based for output
        chunk_end = min(n, i + max_lines)  # 1-based

        # Try to find a symbol that starts within our current window
        best_symbol = None
        for sym in symbols:
            if sym.start >= chunk_start and sym.start <= chunk_end:
                # Check if the entire symbol fits within max_lines from current position
                symbol_size = sym.end - sym.start + 1
                if symbol_size <= max_lines and sym.end <= i + max_lines:
                    best_symbol = sym
                    break

        if best_symbol:
            # Chunk this complete symbol
            chunk_text = "\n".join(lines[best_symbol.start - 1 : best_symbol.end])
            chunks.append(
                {
                    "text": chunk_text,
                    "start": best_symbol.start,
                    "end": best_symbol.end,
                    "symbol": best_symbol.name,
                    "kind": best_symbol.kind,
                }
            )
            # Move past this symbol with minimal overlap
            i = max(best_symbol.end - overlap, i + 1)
        else:
            # No suitable symbol found, fall back to line-based chunking
            chunk_text = "\n".join(lines[i : i + max_lines])
            actual_end = min(n, i + max_lines)
            chunks.append({"text": chunk_text, "start": i + 1, "end": actual_end})
            i = max(actual_end - overlap, i + 1)

    return chunks


# --- Token-based micro-chunking (ReFRAG-lite) ---
# Produces tiny fixed-size token windows with stride, maps back to original line ranges.
def chunk_by_tokens(
    text: str, k_tokens: int = None, stride_tokens: int = None
) -> List[Dict]:
    try:
        from tokenizers import Tokenizer  # lightweight, already in requirements
    except Exception:
        Tokenizer = None  # type: ignore

    try:
        k = int(os.environ.get("MICRO_CHUNK_TOKENS", str(k_tokens or 16)) or 16)
    except Exception:
        k = 16
    try:
        s = int(
            os.environ.get("MICRO_CHUNK_STRIDE", str(stride_tokens or max(1, k // 2)))
            or max(1, k // 2)
        )
    except Exception:
        s = max(1, k // 2)

    # Helper: simple regex-based token offsets when HF tokenizer JSON is unavailable
    def _simple_offsets(txt: str):
        import re
        offs = []
        for m in re.finditer(r"\S+", txt):
            offs.append((m.start(), m.end()))
        return offs

    offsets = []
    # Load tokenizer; default to local model file if present
    tok_path = os.environ.get(
        "TOKENIZER_JSON", str((ROOT_DIR / "models" / "tokenizer.json"))
    )
    if Tokenizer is not None:
        try:
            tokenizer = Tokenizer.from_file(tok_path)
            try:
                enc = tokenizer.encode(text)
                offsets = getattr(enc, "offsets", None) or []
            except Exception:
                offsets = []
        except Exception:
            offsets = []

    if not offsets:
        # Fallback to simple regex tokenization; avoids degrading to 120-line chunks
        if os.environ.get("DEBUG_CHUNKING"):
            print("[ingest] tokenizers missing/unusable -> using simple regex tokenization")
        offsets = _simple_offsets(text)

    if not offsets:
        return chunk_lines(text, max_lines=120, overlap=20)

    # Precompute line starts for fast char->line mapping
    lines = text.splitlines(keepends=True)
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)
    total_chars = len(text)

    def char_to_line(c: int) -> int:
        # Binary search line_starts to find 1-based line number
        lo, hi = 0, len(line_starts) - 1
        if c <= 0:
            return 1
        if c >= total_chars:
            return len(lines)
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= c:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans + 1  # 1-based

    chunks: List[Dict] = []
    i = 0
    n = len(offsets)
    while i < n:
        j = min(n, i + k)
        start_char = offsets[i][0]
        end_char = offsets[j - 1][1] if j - 1 < n else offsets[-1][1]
        start_char = max(0, start_char)
        end_char = min(total_chars, max(start_char, end_char))
        chunk_text = text[start_char:end_char]
        if chunk_text:
            start_line = char_to_line(start_char)
            end_line = (
                char_to_line(end_char - 1) if end_char > start_char else start_line
            )
            chunks.append(
                {
                    "text": chunk_text,
                    "start": start_line,
                    "end": end_line,
                }
            )
        if j == n:
            break
        i = i + s if s > 0 else i + 1
    return chunks


from scripts.utils import sanitize_vector_name as _sanitize_vector_name
from scripts.utils import lex_hash_vector_text as _lex_hash_vector_text


# Optional index-time pseudo descriptions for micro-chunks
# Enabled via REFRAG_PSEUDO_DESCRIBE=1 and requires REFRAG_DECODER=1

def _pseudo_describe_enabled() -> bool:
    try:
        return str(os.environ.get("REFRAG_PSEUDO_DESCRIBE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


# ===== Symbol Extraction for Smart Reindexing =====

def _smart_symbol_reindexing_enabled() -> bool:
    """Check if symbol-aware reindexing is enabled."""
    try:
        return str(os.environ.get("SMART_SYMBOL_REINDEXING", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def extract_symbols_with_tree_sitter(file_path: str) -> dict:
    """Extract functions, classes, methods from file using tree-sitter or fallback.

    Returns:
        dict: {symbol_id: {name, type, start_line, end_line, content_hash, pseudo, tags}}
    """
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
                # These will be populated during processing
                "pseudo": "",
                "tags": [],
                "qdrant_ids": [],  # Will store Qdrant point IDs for this symbol
            }

        return symbols

    except Exception as e:
        print(f"[SYMBOL_EXTRACTION] Failed to extract symbols from {file_path}: {e}")
        return {}


def should_use_smart_reindexing(file_path: str, file_hash: str) -> tuple[bool, str]:
    """Determine if smart reindexing should be used for a file.

    Returns:
        (use_smart, reason)
    """
    if not _smart_symbol_reindexing_enabled():
        return False, "smart_reindexing_disabled"

    if not get_cached_symbols or not set_cached_symbols:
        return False, "symbol_cache_unavailable"

    # Load cached symbols
    cached_symbols = get_cached_symbols(file_path)
    if not cached_symbols:
        return False, "no_cached_symbols"

    # Extract current symbols
    current_symbols = extract_symbols_with_tree_sitter(file_path)
    if not current_symbols:
        return False, "no_current_symbols"

    # Compare symbols
    unchanged_symbols, changed_symbols = compare_symbol_changes(cached_symbols, current_symbols)

    total_symbols = len(current_symbols)
    changed_ratio = len(changed_symbols) / max(total_symbols, 1)

    # Use thresholds to decide strategy
    max_changed_ratio = float(os.environ.get("MAX_CHANGED_SYMBOLS_RATIO", "0.3"))
    if changed_ratio > max_changed_ratio:
        return False, f"too_many_changes_{changed_ratio:.2f}"

    print(f"[SMART_REINDEX] {file_path}: {len(unchanged_symbols)} unchanged, {len(changed_symbols)} changed")
    return True, f"smart_reindex_{len(changed_symbols)}/{total_symbols}"


def generate_pseudo_tags(text: str) -> tuple[str, list[str]]:
    """Best-effort: ask local decoder to produce a short label and 3-6 tags.
    Returns (pseudo, tags). On failure returns ("", [])."""
    pseudo: str = ""
    tags: list[str] = []
    if not _pseudo_describe_enabled() or not text.strip():
        return pseudo, tags
    try:
        from scripts.refrag_llamacpp import (  # type: ignore
            LlamaCppRefragClient,
            is_decoder_enabled,
            get_runtime_kind,
        )
        if not is_decoder_enabled():
            return "", []
        runtime = get_runtime_kind()
        # Keep decoding tight/fast – this is only enrichment for retrieval.
        # Preserve original llama.cpp prompt semantics, and use a stricter
        # JSON-only prompt only for the GLM runtime.
        if runtime == "glm":
            prompt = (
                "You are a JSON-only function that labels code spans for search enrichment.\n"
                "Respond with a single JSON object and nothing else (no prose, no markdown).\n"
                "Exact format: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
                "Code:\n" + text[:2000]
            )
            from scripts.refrag_glm import GLMRefragClient  # type: ignore
            client = GLMRefragClient()
            out = client.generate_with_soft_embeddings(
                prompt=prompt,
                max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
                temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
                top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
                stop=["\n\n"],
                force_json=True,
            )
        else:
            prompt = (
                "You label code spans for search enrichment.\n"
                "Return strictly JSON: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
                "Code:\n" + text[:2000]
            )
            client = LlamaCppRefragClient()
            out = client.generate_with_soft_embeddings(
                prompt=prompt,
                max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
                temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
                top_k=int(os.environ.get("PSEUDO_TOP_K", "30") or 30),
                top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
                stop=["\n\n"],
            )
        import json as _json
        try:
            obj = _json.loads(out)
            if isinstance(obj, dict):
                p = obj.get("pseudo")
                t = obj.get("tags")
                if isinstance(p, str):
                    pseudo = p.strip()[:256]
                if isinstance(t, list):
                    tags = [str(x).strip() for x in t if str(x).strip()][:6]
        except Exception:
            pass
    except Exception:
        return "", []
    return pseudo, tags


def should_process_pseudo_for_chunk(
    file_path: str, chunk: dict, changed_symbols: set
) -> tuple[bool, str, list[str]]:
    """Determine if a chunk needs pseudo processing based on symbol changes AND pseudo cache.

    Uses existing symbol change detection and pseudo cache lookup for optimal performance.

    Args:
        file_path: Path to the file containing this chunk
        chunk: Chunk dict with symbol information
        changed_symbols: Set of symbol IDs that changed (from compare_symbol_changes)

    Returns:
        (needs_processing, cached_pseudo, cached_tags)
    """
    # For chunks without symbol info, process them (fallback - no symbol to reuse from)
    symbol_name = chunk.get("symbol", "")
    if not symbol_name:
        return True, "", []

    # Create symbol ID matching the format used in symbol cache
    kind = chunk.get("kind", "unknown")
    start_line = chunk.get("start", 0)
    symbol_id = f"{kind}_{symbol_name}_{start_line}"

    # If we don't have any change information, best effort: try reusing cached pseudo when present
    if not changed_symbols and get_cached_pseudo:
        try:
            cached_pseudo, cached_tags = get_cached_pseudo(file_path, symbol_id)
            if cached_pseudo or cached_tags:
                return False, cached_pseudo, cached_tags
        except Exception:
            pass
        return True, "", []

    # Unchanged symbol: prefer reuse when cached pseudo/tags exist
    if symbol_id not in changed_symbols:
        if get_cached_pseudo:
            try:
                cached_pseudo, cached_tags = get_cached_pseudo(file_path, symbol_id)
                if cached_pseudo or cached_tags:
                    return False, cached_pseudo, cached_tags
            except Exception:
                pass
        # Unchanged but no cached data yet – process once
        return True, "", []

    # Symbol content changed: always re-run pseudo; do not reuse stale cached values
    return True, "", []


class CollectionNeedsRecreateError(Exception):
    """Raised when a collection needs to be recreated to add new vector types."""
    pass


def ensure_collection(client: QdrantClient, name: str, dim: int, vector_name: str):
    """Ensure collection exists with named vectors.
    Always includes dense (vector_name) and lexical (LEX_VECTOR_NAME).
    When REFRAG_MODE=1, also includes a compact mini vector (MINI_VECTOR_NAME).
    """
    # Track backup file path for this ensure_collection call (per-collection, per-process)
    backup_file = None
    try:
        info = client.get_collection(name)
        # Prevent I/O storm - only update vectors if they actually don't exist
        try:
            cfg = getattr(info.config.params, "vectors", None)
            if isinstance(cfg, dict):
                # Check if collection already has required vectors before updating
                has_lex = LEX_VECTOR_NAME in cfg
                has_mini = MINI_VECTOR_NAME in cfg

                # Only add to missing if vector doesn't already exist
                missing = {}
                if not has_lex:
                    missing[LEX_VECTOR_NAME] = models.VectorParams(
                        size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
                    )

                try:
                    refrag_on = os.environ.get("REFRAG_MODE", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                except Exception:
                    refrag_on = False

                if refrag_on and not has_mini:
                    missing[MINI_VECTOR_NAME] = models.VectorParams(
                        size=int(
                            os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM
                        ),
                        distance=models.Distance.COSINE,
                    )

                # Only update collection if vectors are actually missing
                # Previous behavior: always called update_collection() causing I/O storms
                if missing:
                    try:
                        client.update_collection(
                            collection_name=name, vectors_config=missing
                        )
                        print(f"[COLLECTION_SUCCESS] Successfully updated collection {name} with missing vectors")
                    except Exception as update_e:
                        # Qdrant doesn't support adding new vector names to existing collections
                        # Fall back to recreating the collection with the correct vector configuration
                        print(f"[COLLECTION_WARNING] Cannot add missing vectors to {name} ({update_e}). Recreating collection...")

                        # Backup memories before recreating collection using dedicated backup script
                        backup_file = None
                        try:
                            import tempfile
                            import subprocess
                            import sys

                            # Create temporary backup file
                            with tempfile.NamedTemporaryFile(mode='w', suffix='_memories_backup.json', delete=False) as f:
                                backup_file = f.name

                            print(f"[MEMORY_BACKUP] Backing up memories from {name} to {backup_file}")

                            # Use battle-tested backup script
                            backup_script = Path(__file__).parent / "memory_backup.py"
                            result = subprocess.run([
                                sys.executable, str(backup_script),
                                "--collection", name,
                                "--output", backup_file
                            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)

                            if result.returncode == 0:
                                print(f"[MEMORY_BACKUP] Successfully backed up memories using {backup_script.name}")
                            else:
                                print(f"[MEMORY_BACKUP_WARNING] Backup script failed: {result.stderr}")
                                backup_file = None

                        except Exception as backup_e:
                            print(f"[MEMORY_BACKUP_WARNING] Failed to backup memories: {backup_e}")
                            backup_file = None

                        try:
                            client.delete_collection(name)
                            print(f"[COLLECTION_INFO] Deleted existing collection {name}")
                        except Exception:
                            pass

                        # Store backup info for restoration
                        # backup_file remains bound for this function call; used after collection creation

                        # Proceed to recreate with full vector configuration
                        raise CollectionNeedsRecreateError(f"Collection {name} needs recreation for new vectors")
        except CollectionNeedsRecreateError:
            # Let this fall through to collection creation logic
            print(f"[COLLECTION_INFO] Collection {name} needs recreation - proceeding...")
            raise
        except Exception as e:
            print(f"[COLLECTION_ERROR] Failed to update collection {name}: {e}")
            pass
        return
    except Exception as e:
        # Collection doesn't exist - proceed to create it
        print(f"[COLLECTION_INFO] Creating new collection {name}: {type(e).__name__}")
        pass
    vectors_cfg = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(
            size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    # Conditionally add mini vector for ReFRAG gating
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            vectors_cfg[MINI_VECTOR_NAME] = models.VectorParams(
                size=int(os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM),
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )
    print(f"[COLLECTION_INFO] Successfully created new collection {name} with vectors: {list(vectors_cfg.keys())}")

    # Restore memories if we have a backup from recreation using dedicated restore script
    strict_restore = False
    try:
        val = os.environ.get("STRICT_MEMORY_RESTORE", "")
        strict_restore = str(val or "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        strict_restore = False

    try:
        if backup_file and os.path.exists(backup_file):
            print(f"[MEMORY_RESTORE] Restoring memories from {backup_file}")
            import subprocess
            import sys

            # Use battle-tested restore script (skip collection creation since ingest_code.py already handles it)
            restore_script = Path(__file__).parent / "memory_restore.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(restore_script),
                    "--backup",
                    backup_file,
                    "--collection",
                    name,
                    "--skip-collection-creation",
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            if result.returncode == 0:
                print(f"[MEMORY_RESTORE] Successfully restored memories using {restore_script.name}")
            else:
                msg = result.stderr or result.stdout or "unknown error"
                print(f"[MEMORY_RESTORE_WARNING] Restore script failed: {msg}")
                if strict_restore:
                    raise RuntimeError(f"Memory restore failed for collection {name}: {msg}")

            # Clean up backup file once we've attempted restore
            try:
                os.unlink(backup_file)
                print(f"[MEMORY_RESTORE] Cleaned up backup file {backup_file}")
            except Exception:
                pass
            finally:
                backup_file = None

        elif backup_file:
            print(f"[MEMORY_RESTORE_WARNING] Backup file {backup_file} not found")
            backup_file = None

    except Exception as restore_e:
        print(f"[MEMORY_RESTORE_ERROR] Failed to restore memories: {restore_e}")
        # Optionally fail hard when STRICT_MEMORY_RESTORE is enabled
        if strict_restore:
            raise


def recreate_collection(client: QdrantClient, name: str, dim: int, vector_name: str):
    """Drop and recreate collection with named vectors (dense + lex [+ mini when REFRAG_MODE=1])."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    vectors_cfg = {
        vector_name: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        LEX_VECTOR_NAME: models.VectorParams(
            size=LEX_VECTOR_DIM, distance=models.Distance.COSINE
        ),
    }
    try:
        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            vectors_cfg[MINI_VECTOR_NAME] = models.VectorParams(
                size=int(os.environ.get("MINI_VEC_DIM", MINI_VEC_DIM) or MINI_VEC_DIM),
                distance=models.Distance.COSINE,
            )
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_cfg,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=256),
    )


def ensure_payload_indexes(client: QdrantClient, collection: str):
    """Create helpful payload indexes if they don't exist (idempotent)."""
    for field in (
        "metadata.language",
        "metadata.path_prefix",
        "metadata.repo_id",
        "metadata.repo_rel_path",
        "metadata.repo",
        "metadata.kind",
        "metadata.symbol",
        "metadata.symbol_path",
        "metadata.imports",
        "metadata.calls",
        "metadata.file_hash",
        "metadata.ingested_at",
        "metadata.last_modified_at",
        "metadata.churn_count",
        "metadata.author_count",
        "pid_str",
    ):
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

ENSURED_COLLECTIONS: set[str] = set()
ENSURED_COLLECTIONS_LAST_CHECK: dict[str, float] = {}


def pseudo_backfill_tick(
    client: QdrantClient,
    collection: str,
    repo_name: str | None = None,
    *,
    max_points: int = 256,
    dim: int | None = None,
    vector_name: str | None = None,
) -> int:
    """Best-effort pseudo/tag backfill for a collection.

    Scans up to max_points points for a given repo (when provided) that have not yet
    been marked as pseudo-enriched and updates them in-place with pseudo/tags and
    refreshed lexical vectors. Does not touch cache.json or hash-based skip logic;
    operates purely on Qdrant payloads/vectors.
    """

    if not collection or max_points <= 0:
        return 0

    try:
        from qdrant_client import models as _models
    except Exception:
        return 0

    must_conditions: list[Any] = []
    if repo_name:
        try:
            must_conditions.append(
                _models.FieldCondition(
                    key="metadata.repo",
                    match=_models.MatchValue(value=repo_name),
                )
            )
        except Exception:
            pass

    flt = None
    try:
        # Prefer server-side filtering for points missing pseudo/tags when supported
        null_cond = getattr(_models, "IsNullCondition", None)
        empty_cond = getattr(_models, "IsEmptyCondition", None)
        if null_cond is not None:
            should_conditions = []
            try:
                should_conditions.append(null_cond(is_null="pseudo"))
            except Exception:
                pass
            try:
                should_conditions.append(null_cond(is_null="tags"))
            except Exception:
                pass
            if empty_cond is not None:
                try:
                    should_conditions.append(empty_cond(is_empty="tags"))
                except Exception:
                    pass
            flt = _models.Filter(
                must=must_conditions or None,
                should=should_conditions or None,
            )
        else:
            # Fallback: only scope by repo, rely on Python-side pseudo/tags checks
            flt = _models.Filter(must=must_conditions or None)
    except Exception:
        flt = None

    processed = 0
    debug_enabled = (os.environ.get("PSEUDO_BACKFILL_DEBUG") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    debug_stats = {
        "scanned": 0,
        "glm_calls": 0,
        "glm_success": 0,
        "filled_new": 0,
        "updated_existing": 0,
        "skipped_no_code": 0,
        "skipped_after_glm": 0,
    }
    next_offset = None

    def _maybe_ensure_collection() -> bool:
        if not dim or not vector_name:
            return False
        try:
            ensure_collection_and_indexes_once(client, collection, int(dim), vector_name)
            return True
        except Exception:
            return False

    while processed < max_points:
        batch_limit = max(1, min(64, max_points - processed))
        try:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=batch_limit,
                with_payload=True,
                with_vectors=True,
                offset=next_offset,
            )
        except Exception:
            if _maybe_ensure_collection():
                try:
                    points, next_offset = client.scroll(
                        collection_name=collection,
                        scroll_filter=flt,
                        limit=batch_limit,
                        with_payload=True,
                        with_vectors=True,
                        offset=next_offset,
                    )
                except Exception:
                    break
            else:
                break

        if not points:
            break

        new_points: list[Any] = []
        for rec in points:
            try:
                if debug_enabled:
                    debug_stats["scanned"] += 1
                payload = rec.payload or {}
                md = payload.get("metadata") or {}
                code = md.get("code") or ""
                if not code:
                    if debug_enabled:
                        debug_stats["skipped_no_code"] += 1
                    continue

                pseudo = payload.get("pseudo") or ""
                tags_val = payload.get("tags") or []
                tags: list[str] = list(tags_val) if isinstance(tags_val, list) else []
                had_existing = bool(pseudo or tags)

                # If pseudo/tags are missing, generate them once
                if not pseudo and not tags:
                    try:
                        if debug_enabled:
                            debug_stats["glm_calls"] += 1
                        pseudo, tags = generate_pseudo_tags(code)
                        if debug_enabled and (pseudo or tags):
                            debug_stats["glm_success"] += 1
                    except Exception:
                        pseudo, tags = "", []

                if not pseudo and not tags:
                    if debug_enabled:
                        debug_stats["skipped_after_glm"] += 1
                    continue

                # Update payload and lexical vector with pseudo/tags
                payload["pseudo"] = pseudo
                payload["tags"] = tags
                if debug_enabled:
                    if had_existing:
                        debug_stats["updated_existing"] += 1
                    else:
                        debug_stats["filled_new"] += 1

                aug_text = f"{code} {pseudo} {' '.join(tags)}".strip()
                lex_vec = _lex_hash_vector_text(aug_text)

                vec = rec.vector
                if isinstance(vec, dict):
                    vecs = dict(vec)
                    vecs[LEX_VECTOR_NAME] = lex_vec
                    new_vec = vecs
                else:
                    # Fallback: collections without named vectors - leave dense vector as-is
                    new_vec = vec

                new_points.append(
                    models.PointStruct(
                        id=rec.id,
                        vector=new_vec,
                        payload=payload,
                    )
                )
                processed += 1
            except Exception:
                continue

        if new_points:
            try:
                upsert_points(client, collection, new_points)
            except Exception:
                if _maybe_ensure_collection():
                    try:
                        upsert_points(client, collection, new_points)
                    except Exception:
                        # Best-effort: on failure, stop this tick
                        break
                else:
                    # Best-effort: on failure, stop this tick
                    break

        if next_offset is None:
            break

    return processed


def ensure_collection_and_indexes_once(
    client: QdrantClient,
    collection: str,
    dim: int,
    vector_name: str | None,
) -> None:
    if not collection:
        return
    if collection in ENSURED_COLLECTIONS:
        # By default we do NOT ping Qdrant repeatedly for an already-ensured collection.
        # This avoids a steady stream of GET /collections/<name> requests during large runs.
        # Opt-in: set ENSURED_COLLECTION_PING_SECONDS to a positive float (e.g. 60).
        try:
            ping_seconds = float(os.environ.get("ENSURED_COLLECTION_PING_SECONDS", "0") or 0)
        except Exception:
            ping_seconds = 0.0

        if ping_seconds <= 0:
            return

        try:
            now = time.time()
            last = ENSURED_COLLECTIONS_LAST_CHECK.get(collection, 0.0)
            if (now - last) < ping_seconds:
                return
            client.get_collection(collection)
            ENSURED_COLLECTIONS_LAST_CHECK[collection] = now
            return
        except Exception:
            try:
                ENSURED_COLLECTIONS.discard(collection)
            except Exception:
                pass
            try:
                ENSURED_COLLECTIONS_LAST_CHECK.pop(collection, None)
            except Exception:
                pass
    ensure_collection(client, collection, dim, vector_name)
    ensure_payload_indexes(client, collection)
    ENSURED_COLLECTIONS.add(collection)
    try:
        ENSURED_COLLECTIONS_LAST_CHECK[collection] = time.time()
    except Exception:
        pass


# Lightweight import extraction per language (best-effort)
def _extract_imports(language: str, text: str) -> list:
    lines = text.splitlines()
    imps = []
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
            m = re.match(r"^\s*require\(\s*['\"]([^'\"]+)['\"]\s*\)", ln)
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
            # using Namespace.Sub; using static System.Math; using Alias = Namespace.Type;
            m = re.match(r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\._]*)(?:\s*;|\s*=)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "php":
        for ln in lines:
            # Namespaced uses: use Foo\Bar; use function Foo\bar; use const Foo\BAR;
            m = re.match(r"^\s*use\s+(?:function\s+|const\s+)?([A-Za-z_][A-Za-z0-9_\\\\]*)\s*;", ln)
            if m:
                imps.append(m.group(1).replace("\\\\", "\\"))
                continue
        for ln in lines:
            # include/require path-like imports
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
    elif language == "terraform":
        # modules/providers are most relevant cross-file references
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


# Lightweight call-site extraction (best-effort, language-agnostic heuristics)
def _extract_calls(language: str, text: str) -> list:
    names = []
    # Simple heuristic: word followed by '(' that isn't a keyword
    kw = set(
        [
            "if",
            "for",
            "while",
            "switch",
            "return",
            "new",
            "catch",
            "func",
            "def",
            "class",
            "match",
        ]
    )
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = m.group(1)
        if name not in kw:
            names.append(name)
    # Deduplicate preserving order
    out = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out[:200]


def get_indexed_file_hash(
    client: QdrantClient,
    collection: str,
    file_path: str,
    *,
    repo_id: str | None = None,
    repo_rel_path: str | None = None,
) -> str:
    """Return previously indexed file hash for this logical path, or empty string.

    Prefers logical identity (repo_id + repo_rel_path) when available so that
    worktrees sharing a logical repo can reuse existing index state, but falls
    back to metadata.path for backwards compatibility.
    """
    # Prefer logical identity when both repo_id and repo_rel_path are provided
    if logical_repo_reuse_enabled() and repo_id and repo_rel_path:
        try:
            filt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.repo_id", match=models.MatchValue(value=repo_id)
                    ),
                    models.FieldCondition(
                        key="metadata.repo_rel_path",
                        match=models.MatchValue(value=repo_rel_path),
                    ),
                ]
            )
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=filt,
                with_payload=True,
                limit=1,
            )
            if points:
                md = (points[0].payload or {}).get("metadata") or {}
                fh = md.get("file_hash")
                if fh:
                    return str(fh)
        except Exception:
            # Fall back to path-based lookup below
            pass

    # Backwards-compatible path-based lookup
    try:
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path", match=models.MatchValue(value=file_path)
                )
            ]
        )
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            with_payload=True,
            limit=1,
        )
        if points:
            md = (points[0].payload or {}).get("metadata") or {}
            fh = md.get("file_hash")
            if fh:
                return str(fh)
    except Exception:
        return ""
    return ""


def delete_points_by_path(client: QdrantClient, collection: str, file_path: str):
    try:
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.path", match=models.MatchValue(value=file_path)
                )
            ]
        )
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=filt),
            wait=True,
        )
    except Exception:
        pass


def embed_batch(model: "TextEmbedding", texts: List[str]) -> List[List[float]]:
    # fastembed returns a generator of numpy arrays
    return [vec.tolist() for vec in model.embed(texts)]


def upsert_points(
    client: QdrantClient, collection: str, points: List[models.PointStruct]
):
    if not points:
        return
    # Safer upsert for large payloads: chunk + retry with backoff
    try:
        bsz = int(os.environ.get("INDEX_UPSERT_BATCH", "256") or 256)
    except Exception:
        bsz = 256
    try:
        retries = int(os.environ.get("INDEX_UPSERT_RETRIES", "3") or 3)
    except Exception:
        retries = 3
    try:
        backoff = float(os.environ.get("INDEX_UPSERT_BACKOFF", "0.5") or 0.5)
    except Exception:
        backoff = 0.5

    for i in range(0, len(points), max(1, bsz)):
        batch = points[i : i + max(1, bsz)]
        attempt = 0
        while True:
            try:
                client.upsert(collection_name=collection, points=batch, wait=True)
                break
            except Exception:
                attempt += 1
                if attempt >= retries:
                    # Last-resort: try smaller sub-batches to avoid dropping updates entirely
                    sub_size = max(1, bsz // 4)
                    for j in range(0, len(batch), sub_size):
                        sub = batch[j : j + sub_size]
                        try:
                            client.upsert(
                                collection_name=collection, points=sub, wait=True
                            )
                        except Exception:
                            # Give up on this tiny sub-batch; continue with the rest
                            pass
                    break
                else:
                    try:
                        time.sleep(backoff * attempt)
                    except Exception:
                        pass


def detect_language(path: Path) -> str:
    return CODE_EXTS.get(path.suffix.lower(), "unknown")


def build_information(
    language: str, path: Path, start: int, end: int, first_line: str
) -> str:
    first_line = (first_line or "").strip()
    if len(first_line) > 160:
        first_line = first_line[:160] + "…"
    return f"{language} code from {path} lines {start}-{end}. {first_line}"


# ---- Symbol extraction helpers ----
class _Sym(dict):
    __getattr__ = dict.get


def _extract_symbols_python(text: str) -> List[_Sym]:
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


_JS_FUNC_PATTERNS = [
    r"^\s*export\s+function\s+([A-Za-z_$][\w$]*)\s*\(",
    r"^\s*function\s+([A-Za-z_$][\w$]*)\s*\(",
    r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>",
    r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*function\s*\(",
]
_JS_CLASS_PATTERNS = [r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\b"]


def _extract_symbols_js_like(text: str) -> List[_Sym]:
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


# ----- Tree-sitter based extraction (currently Python, JS/TS) -----
def _ts_parser(lang_key: str):
    """Return a tree-sitter Parser for the given language key.

    Uses tree-sitter 0.25+ API with pre-loaded Language objects.
    """
    if not _use_tree_sitter():
        return None

    if Parser is None or lang_key not in _TS_LANGUAGES:
        return None

    try:
        lang = _TS_LANGUAGES[lang_key]
        return Parser(lang)
    except Exception:
        return None


def _ts_extract_symbols_python(text: str) -> List[_Sym]:
    parser = _ts_parser("python")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        # Parsing can fail on malformed code - fallback to empty symbols
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
    # Works for javascript/typescript using the most-specific available grammar
    parser = _ts_parser("javascript")
    if not parser:
        return []
    try:
        tree = parser.parse(text.encode("utf-8"))
        if tree is None:
            return []
        root = tree.root_node
    except (ValueError, Exception) as e:
        # Parsing can fail on malformed code - fallback to empty symbols
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
        # e.g., const g = function() {}, const h = () => {}, var j = function() {}
        if t == "variable_declarator":
            # Check if the value is a function expression or arrow function
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
                # Don't recurse into the function expression to avoid duplicates
                return
        for c in n.children:
            walk(c)

    walk(root)
    return syms


def _ts_extract_symbols(language: str, text: str) -> List[_Sym]:
    if language == "python":
        return _ts_extract_symbols_python(text)
    if language == "javascript":
        return _ts_extract_symbols_js(text)
    if language == "typescript":
        # Prefer TypeScript grammar when available; otherwise fall back to JS grammar.
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
                    # Handle variable declarations with function expressions or arrow functions
                    # e.g., const g = function() {}, const h = () => {}, var j = function() {}
                    if t == "variable_declarator":
                        # Check if the value is a function expression or arrow function
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
                            # Don't recurse into the function expression to avoid duplicates
                            return
                    for c in n.children:
                        walk(c)

                walk(root)
                return syms

        return _ts_extract_symbols_js(text)

    return []


def _ts_extract_imports_calls_python(text: str):
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
        return data[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")

    imports: List[str] = []
    calls: List[str] = []

    def walk(n):
        t = n.type
        if t == "import_statement":
            s = node_text(n)
            m = re.search(r"\bimport\s+([\w\.]+)", s)
            if m:
                imports.append(m.group(1))
        elif t == "import_from_statement":
            s = node_text(n)
            m = re.search(r"\bfrom\s+([\w\.]+)\s+import\b", s)
            if m:
                imports.append(m.group(1))
        elif t == "call":
            func = n.child_by_field_name("function")
            if func:
                name = node_text(func)
                # Take the last attribute part if dotted
                base = re.split(r"[\.:]", name)[-1]
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base):
                    calls.append(base)
        for c in n.children:
            walk(c)

    walk(root)
    # Deduplicate preserving order
    seen = set()
    calls_dedup = []
    for x in calls:
        if x not in seen:
            calls_dedup.append(x)
            seen.add(x)
    return imports[:200], calls_dedup[:200]


def _get_imports_calls(language: str, text: str):
    if _use_tree_sitter() and language == "python":
        return _ts_extract_imports_calls_python(text)
    return _extract_imports(language, text), _extract_calls(language, text)


def _extract_symbols_terraform(text: str) -> List[_Sym]:
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


def _extract_symbols(language: str, text: str) -> List[_Sym]:
    # Prefer tree-sitter when enabled and supported; fallback to existing extractors
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


def _compute_host_and_container_paths(cur_path: str) -> tuple[Optional[str], Optional[str]]:
    """Compute host_path and container_path for a given absolute path.

    Behavior:
    - path field in metadata continues to use cur_path as-is (container view).
    - host_path prefers origin.source_path (client workspace root) when available.
    - When indexing under /work/<slug>/..., drop the slug when mapping back to host.
    - HOST_INDEX_PATH is used only as a best-effort fallback and ignored when it
      looks like a Windows path (contains a drive letter).
    """
    _host_root = str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
    if ":" in _host_root:
        _host_root = ""
    _host_path: Optional[str] = None
    _container_path: Optional[str] = None
    _origin_client_path: Optional[str] = None

    # Try to get client workspace root from origin metadata first.
    try:
        if cur_path.startswith("/work/"):
            # Extract workspace from container path
            _parts = cur_path[6:].split("/")  # Remove "/work/" prefix
            if len(_parts) >= 2:
                _repo_name = _parts[0]  # First part is repo name / slug
                _workspace_path = f"/work/{_repo_name}"
                _origin_client_path = _get_host_path_from_origin(
                    _workspace_path, _repo_name
                )
    except Exception:
        _origin_client_path = None

    try:
        if cur_path.startswith("/work/") and (_host_root or _origin_client_path):
            _rel = cur_path[len("/work/") :]
            # Prioritize client path from origin metadata over HOST_INDEX_PATH.
            if _origin_client_path:
                # Drop the leading repo slug (e.g. Context-Engine-<hash>) when mapping
                # /work paths back to the client workspace root, so host_path is
                # /home/.../Context-Engine/<rel-path-inside-repo> instead of including
                # the slug directory.
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
            # Likely indexing on the host directly
            _host_path = cur_path
            if (
                (_host_root or _origin_client_path)
                and cur_path.startswith(((_origin_client_path or _host_root) + "/"))
            ):
                _rel = cur_path[len((_origin_client_path or _host_root)) + 1 :]
                _container_path = "/work/" + _rel
    except Exception:
        _host_path = cur_path
        _container_path = cur_path if cur_path.startswith("/work/") else None

    return _host_path, _container_path


def index_single_file(
    client: QdrantClient,
    model: "TextEmbedding",
    collection: str,
    vector_name: str,
    file_path: Path,
    *,
    dedupe: bool = True,
    skip_unchanged: bool = True,
    pseudo_mode: str = "full",
    trust_cache: bool | None = None,
    repo_name_for_cache: str | None = None,
) -> bool:
    """Index a single file path. Returns True if indexed, False if skipped.

    When trust_cache is enabled (via argument or INDEX_TRUST_CACHE=1), rely solely on the
    local .codebase/cache.json for unchanged detection and skip Qdrant per-file hash checks.
    This is a debug-only escape hatch and is unsafe for normal operation: enabling it may
    hide index/cache drift, especially with git worktree reuse or collection rebuilds.
    """

    try:
        if _should_skip_explicit_file_by_excluder(file_path):
            try:
                delete_points_by_path(client, collection, str(file_path))
            except Exception:
                pass
            print(f"Skipping excluded file: {file_path}")
            return False
    except Exception:
        return False

    # Resolve trust_cache from env when not explicitly provided. INDEX_TRUST_CACHE is intended
    # for debugging only and should not be enabled in normal indexing runs.
    if trust_cache is None:
        try:
            trust_cache = os.environ.get("INDEX_TRUST_CACHE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        except Exception:
            trust_cache = False

    fast_fs = _env_truthy(os.environ.get("INDEX_FS_FASTPATH"), False)
    if skip_unchanged and fast_fs and get_cached_file_meta is not None:
        try:
            repo_for_cache = repo_name_for_cache or _detect_repo_name_from_path(file_path)
            meta = get_cached_file_meta(str(file_path), repo_for_cache) or {}
            size = meta.get("size")
            mtime = meta.get("mtime")
            if size is not None and mtime is not None:
                st = file_path.stat()
                if int(getattr(st, "st_size", 0)) == int(size) and int(
                    getattr(st, "st_mtime", 0)
                ) == int(mtime):
                    print(f"Skipping unchanged file (fs-meta): {file_path}")
                    return False
        except Exception:
            pass

    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return False

    language = detect_language(file_path)
    file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    repo_tag = repo_name_for_cache or _detect_repo_name_from_path(file_path)

    # Derive logical repo identity and repo-relative path for cross-worktree reuse.
    repo_id: str | None = None
    repo_rel_path: str | None = None
    if logical_repo_reuse_enabled() and get_workspace_state is not None:
        try:
            ws_root = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
            # Resolve workspace state for this repo to read logical_repo_id
            state = get_workspace_state(ws_root, repo_tag)
            lrid = state.get("logical_repo_id") if isinstance(state, dict) else None
            if isinstance(lrid, str) and lrid:
                repo_id = lrid
            # Compute repo-relative path within the current workspace tree
            try:
                fp = file_path.resolve()
            except Exception:
                fp = file_path
            try:
                ws_base = Path(os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work").resolve()
                repo_root = ws_base
                if repo_tag:
                    # In multi-repo scenarios, repos live under /work/<repo_tag>
                    candidate = ws_base / repo_tag
                    if candidate.exists():
                        repo_root = candidate
                rel = fp.relative_to(repo_root)
                repo_rel_path = rel.as_posix()
            except Exception:
                repo_rel_path = None
        except Exception as e:
            print(f"[logical_repo] Failed to derive logical identity for {file_path}: {e}")

    # Get changed symbols for pseudo processing optimization
    changed_symbols = set()
    if get_cached_symbols and set_cached_symbols:
        cached_symbols = get_cached_symbols(str(file_path))
        if cached_symbols:
            current_symbols = extract_symbols_with_tree_sitter(str(file_path))
            _, changed = compare_symbol_changes(cached_symbols, current_symbols)
            # Convert symbol names to IDs for lookup
            for symbol_data in current_symbols.values():
                symbol_id = f"{symbol_data['type']}_{symbol_data['name']}_{symbol_data['start_line']}"
                if symbol_id in changed:
                    changed_symbols.add(symbol_id)

    if skip_unchanged:
        # Prefer local workspace cache to avoid Qdrant lookups
        ws_path = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
        try:
            if get_cached_file_hash:
                prev_local = get_cached_file_hash(str(file_path), repo_tag)
                if prev_local and file_hash and prev_local == file_hash:
                    # When fs fast-path is enabled, refresh cache entry with size/mtime
                    if fast_fs and set_cached_file_hash:
                        try:
                            set_cached_file_hash(str(file_path), file_hash, repo_tag)
                        except Exception:
                            pass
                    print(f"Skipping unchanged file (cache): {file_path}")
                    return False
        except Exception:
            pass

        # Optional Qdrant-backed unchanged detection; disabled when trust_cache is enabled
        # Bloom filter fast-path: if hash definitely not in bloom, skip Qdrant check
        if not trust_cache:
            # When BLOOM_SKIP_ENABLED=1, check bloom filter first
            if not _bloom_might_contain(file_hash):
                # Hash definitely not indexed - no need to check Qdrant
                pass  # Continue to indexing
            else:
                # Hash might be indexed - do Qdrant check to confirm
                prev = get_indexed_file_hash(
                    client,
                    collection,
                    str(file_path),
                    repo_id=repo_id,
                    repo_rel_path=repo_rel_path,
                )
                if prev and prev == file_hash:
                    # When fs fast-path is enabled, refresh cache entry with size/mtime
                    if fast_fs and set_cached_file_hash:
                        try:
                            set_cached_file_hash(str(file_path), file_hash, repo_tag)
                        except Exception:
                            pass
                    print(f"Skipping unchanged file: {file_path}")
                    return False

    if dedupe:
        delete_points_by_path(client, collection, str(file_path))

    symbols = _extract_symbols(language, text)
    imports, calls = _get_imports_calls(language, text)
    last_mod, churn_count, author_count = _git_metadata(file_path)

    CHUNK_LINES = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    CHUNK_OVERLAP = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)
    # Micro-chunking (token-based) takes precedence; else semantic; else line-based
    use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if use_micro:
        chunks = chunk_by_tokens(text)
        try:
            _cap = int(os.environ.get("MAX_MICRO_CHUNKS_PER_FILE", "500") or 500)
            if _cap > 0 and len(chunks) > _cap:
                _before = len(chunks)
                # Priority: keep ALL_CAPS assignments near top so constants survive capping
                try:
                    import re as _re
                    if os.environ.get("INGEST_CONSTANT_PRIORITY", "1").lower() not in {"0", "false", "off", "no"}:
                        _pat = _re.compile(r"\b[A-Z_]{2,}\s*=\s*")
                        _important = []
                        _rest = []
                        for c in chunks:
                            txt = c.get("text", "")
                            (_important if _pat.search(txt) else _rest).append(c)
                        # Preserve relative order within groups
                        chunks = (_important + _rest)[:_cap]
                    else:
                        chunks = chunks[:_cap]
                except Exception:
                    chunks = chunks[:_cap]
                try:
                    print(
                        f"[ingest] micro-chunks capped path={file_path} count={_before}->{len(chunks)} cap={_cap}"
                    )
                except Exception:
                    pass
        except Exception:
            pass
    elif use_semantic:
        chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
    else:
        chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)
    batch_texts: List[str] = []
    batch_meta: List[Dict] = []
    batch_ids: List[int] = []
    batch_lex: List[list[float]] = []

    def make_point(pid, dense_vec, lex_vec, payload):
        if vector_name:
            vecs = {vector_name: dense_vec, LEX_VECTOR_NAME: lex_vec}
            try:
                if os.environ.get("REFRAG_MODE", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    vecs[MINI_VECTOR_NAME] = project_mini(list(dense_vec), MINI_VEC_DIM)
            except Exception:
                pass
            return models.PointStruct(id=pid, vector=vecs, payload=payload)
        else:
            # unnamed collection: store dense only
            return models.PointStruct(id=pid, vector=dense_vec, payload=payload)

    for ch in chunks:
        info = build_information(
            language,
            file_path,
            ch["start"],
            ch["end"],
            ch["text"].splitlines()[0] if ch["text"] else "",
        )
        kind, sym, sym_path = _choose_symbol_for_chunk(ch["start"], ch["end"], symbols)
        # Prefer embedded symbol metadata from semantic chunker when present
        if "kind" in ch and ch.get("kind"):
            kind = ch.get("kind") or kind
        if "symbol" in ch and ch.get("symbol"):
            sym = ch.get("symbol") or sym
        if "symbol_path" in ch and ch.get("symbol_path"):
            sym_path = ch.get("symbol_path") or sym_path
        # Ensure chunks always carry symbol metadata so pseudo gating can work for all chunking modes
        if not ch.get("kind") and kind:
            ch["kind"] = kind
        if not ch.get("symbol") and sym:
            ch["symbol"] = sym
        if not ch.get("symbol_path") and sym_path:
            ch["symbol_path"] = sym_path
        # Track both container path (/work mirror) and original host path for clarity across environments
        _cur_path = str(file_path)
        # upload_service writes origin.source_path from the client --path flag so we can
        # reconstruct host paths even when indexing inside a slugged /work/<repo-hash> tree.
        _host_path, _container_path = _compute_host_and_container_paths(_cur_path)

        payload = {
            "document": info,
            "information": info,
            "metadata": {
                "path": str(file_path),
                "path_prefix": str(file_path.parent),
                "ext": str(file_path.suffix).lstrip(".").lower(),
                "language": language,
                "kind": kind,
                "symbol": sym,
                "symbol_path": sym_path,
                "repo": repo_tag,
                "start_line": ch["start"],
                "end_line": ch["end"],
                "code": ch["text"],
                "file_hash": file_hash,
                "imports": imports,
                "calls": calls,
                "ingested_at": int(time.time()),
                "last_modified_at": int(last_mod),
                "churn_count": int(churn_count),
                "author_count": int(author_count),
                # Logical identity for cross-worktree reuse
                "repo_id": repo_id,
                "repo_rel_path": repo_rel_path,
                # New: explicit dual-path tracking
                "host_path": _host_path,
                "container_path": _container_path,
            },
        }
        # Optional LLM enrichment for lexical retrieval: pseudo + tags per micro-chunk
        # Use symbol-aware gating and cached pseudo/tags where possible
        pseudo = ""
        tags = []
        if pseudo_mode != "off":
            needs_pseudo, cached_pseudo, cached_tags = should_process_pseudo_for_chunk(
                str(file_path), ch, changed_symbols
            )
            pseudo, tags = cached_pseudo, cached_tags
            if pseudo_mode == "full" and needs_pseudo:
                try:
                    pseudo, tags = generate_pseudo_tags(ch.get("text") or "")
                    if pseudo or tags:
                        # Cache the pseudo data for this symbol
                        symbol_name = ch.get("symbol", "")
                        if symbol_name:
                            kind = ch.get("kind", "unknown")
                            start_line = ch.get("start", 0)
                            symbol_id = f"{kind}_{symbol_name}_{start_line}"

                            if set_cached_pseudo:
                                set_cached_pseudo(str(file_path), symbol_id, pseudo, tags, file_hash)
                except Exception:
                    # Fall back to cached values (if any) or empty pseudo/tags
                    pass
        # Attach whichever pseudo/tags we ended up with (cached or freshly generated)
        if pseudo:
            payload["pseudo"] = pseudo
        if tags:
            payload["tags"] = tags
        batch_texts.append(info)
        batch_meta.append(payload)
        batch_ids.append(hash_id(ch["text"], str(file_path), ch["start"], ch["end"]))
        aug_lex_text = (ch.get("text") or "") + (" " + pseudo if pseudo else "") + (" " + " ".join(tags) if tags else "")
        batch_lex.append(_lex_hash_vector_text(aug_lex_text))

    if batch_texts:
        vectors = embed_batch(model, batch_texts)
        # Inject pid_str into payloads for server-side gating

        for _idx, _m in enumerate(batch_meta):
            try:
                _m["pid_str"] = str(batch_ids[_idx])
            except Exception:
                pass
        points = [
            make_point(i, v, lx, m)
            for i, v, lx, m in zip(batch_ids, vectors, batch_lex, batch_meta)
        ]
        upsert_points(client, collection, points)
        # Update local file-hash cache only after successful upsert
        try:
            ws = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
            if set_cached_file_hash:
                file_repo_tag = repo_tag
                set_cached_file_hash(str(file_path), file_hash, file_repo_tag)
        except Exception:
            pass
        # Add to bloom filter for future fast-path checks
        _bloom_add(file_hash)
        return True
    return False

def index_repo(
    root: Path,
    qdrant_url: str,
    api_key: str,
    collection: str,
    model_name: str,
    recreate: bool,
    *,
    dedupe: bool = True,
    skip_unchanged: bool = True,
    pseudo_mode: str = "full",
):
    # Optional fast no-change precheck: when INDEX_FS_FASTPATH is enabled, use
    # fs metadata + cache.json to exit early before model/Qdrant setup when all
    # files are unchanged.
    fast_fs = _env_truthy(os.environ.get("INDEX_FS_FASTPATH"), False)
    if skip_unchanged and not recreate and fast_fs and get_cached_file_meta is not None:
        try:
            is_multi_repo = bool(is_multi_repo_mode and is_multi_repo_mode())
            root_repo_for_cache = (
                _detect_repo_name_from_path(root)
                if (not is_multi_repo and _detect_repo_name_from_path)
                else None
            )
            all_unchanged = True
            for file_path in iter_files(root):
                per_file_repo_for_cache = (
                    root_repo_for_cache
                    if root_repo_for_cache is not None
                    else (
                        _detect_repo_name_from_path(file_path)
                        if _detect_repo_name_from_path
                        else None
                    )
                )
                meta = get_cached_file_meta(str(file_path), per_file_repo_for_cache) or {}
                size = meta.get("size")
                mtime = meta.get("mtime")
                if size is None or mtime is None:
                    all_unchanged = False
                    break
                st = file_path.stat()
                if int(getattr(st, "st_size", 0)) != int(size) or int(getattr(st, "st_mtime", 0)) != int(mtime):
                    all_unchanged = False
                    break
            if all_unchanged:
                try:
                    print("[fast_index] No changes detected via fs metadata; skipping model and Qdrant setup")
                except Exception:
                    pass
                return
        except Exception:
            pass

    # Use centralized embedder factory if available (supports Qwen3 feature flag)
    try:
        from scripts.embedder import get_embedding_model, get_model_dimension
        model = get_embedding_model(model_name)
        dim = get_model_dimension(model_name)
    except ImportError:
        # Fallback to direct fastembed initialization
        model = TextEmbedding(model_name=model_name)
        dim = len(next(model.embed(["dimension probe"])))

    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key or None,
        timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20),
    )

    # Determine vector name
    if recreate:
        vector_name = _sanitize_vector_name(model_name)
    else:
        vector_name = None
        try:
            info = client.get_collection(collection)
            cfg = info.config.params.vectors
            if isinstance(cfg, dict) and cfg:
                # Prefer named vector whose size matches current embedding dim
                for name, params in cfg.items():
                    psize = getattr(params, "size", None) or getattr(
                        params, "dim", None
                    )
                    if psize and int(psize) == int(dim):
                        vector_name = name
                        break
                # Otherwise, if a LEX vector exists, pick a different name as dense
                if vector_name is None and LEX_VECTOR_NAME in cfg:
                    for name in cfg.keys():
                        if name != LEX_VECTOR_NAME:
                            vector_name = name
                            break
        except Exception:
            pass
        if vector_name is None:
            vector_name = _sanitize_vector_name(model_name)

    use_per_repo_collections = False

    # Workspace state: derive collection and persist metadata
    try:
        ws_path = str(root)
        repo_tag = _detect_repo_name_from_path(root) if _detect_repo_name_from_path else None

        is_multi_repo = bool(is_multi_repo_mode and is_multi_repo_mode())
        use_per_repo_collections = bool(is_multi_repo and _get_collection_for_file)

        if use_per_repo_collections:
            collection = None  # Determined per file later
            print("[multi_repo] Using per-repo collections for root")
        else:
            if 'get_collection_name' in globals() and get_collection_name:
                try:
                    resolved = get_collection_name(ws_path)
                    placeholders = {"", "default-collection", "my-collection", "codebase"}
                    if resolved and collection in placeholders:
                        collection = resolved
                except Exception:
                    pass

        if update_workspace_state and not use_per_repo_collections:
            update_workspace_state(
                workspace_path=ws_path,
                updates={"qdrant_collection": collection},
                repo_name=repo_tag,
            )
        if update_indexing_status and repo_tag:
            update_indexing_status(
                workspace_path=ws_path,
                status={
                    "state": "indexing",
                    "started_at": datetime.now().isoformat(),
                    "progress": {"files_processed": 0, "total_files": None},
                },
                repo_name=repo_tag,
            )
    except Exception as e:
        # Log state update errors instead of silent failure
        import traceback
        print(f"[ERROR] Failed to update workspace state during indexing: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")


    print(
        f"Indexing root={root} -> {qdrant_url} collection={collection} model={model_name} recreate={recreate}"
    )

    # Health check: detect cache/collection sync issues before indexing (single-collection mode only)
    # TODO: In future, consider a dedicated "health-check-only" mode/command that runs these
    # expensive Qdrant probes without doing a full index pass, so that "nothing changed" runs
    # can stay as cheap as possible while still offering an explicit way to validate collections.
    # Skip with SKIP_HEALTH_CHECK=1 for large collections where scroll is slow
    _skip_health = os.environ.get("SKIP_HEALTH_CHECK", "").strip().lower() in {"1", "true", "yes"}
    if not _skip_health and not recreate and skip_unchanged and not use_per_repo_collections and collection:
        try:
            from scripts.collection_health import auto_heal_if_needed

            print("[health_check] Checking collection health...")
            heal_result = auto_heal_if_needed(str(root), collection, qdrant_url, dry_run=False)
            if heal_result["action_taken"] == "cleared_cache":
                print("[health_check] Cache cleared due to sync issue - forcing full reindex")
            elif not heal_result["health_check"]["healthy"]:
                print(f"[health_check] Issue detected: {heal_result['health_check']['issue']}")
            else:
                print("[health_check] Collection health OK")
        except Exception as e:
            print(f"[health_check] Warning: health check failed: {e}")
    elif _skip_health:
        print("[health_check] Skipped (SKIP_HEALTH_CHECK=1)")

    # Skip single collection setup in multi-repo mode
    if not use_per_repo_collections:
        if recreate:
            recreate_collection(client, collection, dim, vector_name)
        # Ensure useful payload indexes exist (idempotent)
        ensure_collection_and_indexes_once(client, collection, dim, vector_name)
    else:
        print("[multi_repo] Skipping single collection setup - will create per-repo collections during indexing")
    # Repo tag for filtering: auto-detect from git or folder name
    repo_tag = _detect_repo_name_from_path(root)
    # TODO: Long-term, in upload-server mode repo identity should come from workspace metadata
    # (state.json/origin), but keep bindmount/git fallbacks for back-compat.
    workspace_root = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
    touched_repos: set[str] = set()
    repo_roots: dict[str, str] = {}

    # Batch and scaling config (env/CLI overridable)
    batch_texts: list[str] = []
    batch_meta: list[dict] = []
    batch_ids: list[int] = []
    batch_lex: list[list[float]] = []
    BATCH_SIZE = int(os.environ.get("INDEX_BATCH_SIZE", "256") or 256)
    CHUNK_LINES = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
    CHUNK_OVERLAP = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)
    PROGRESS_EVERY = int(os.environ.get("INDEX_PROGRESS_EVERY", "200") or 200)
    # Trust-cache mode: skip Qdrant hash lookups when local cache says unchanged
    _trust_cache = os.environ.get("INDEX_TRUST_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if _trust_cache:
        print("[trust_cache] INDEX_TRUST_CACHE enabled - skipping Qdrant per-file hash checks")
    # Semantic chunking toggle
    use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Debug chunking mode
    if os.environ.get("DEBUG_CHUNKING"):
        print(f"[DEBUG] INDEX_SEMANTIC_CHUNKS={os.environ.get('INDEX_SEMANTIC_CHUNKS', 'NOT_SET')} -> use_semantic={use_semantic}")
        print(f"[DEBUG] INDEX_MICRO_CHUNKS={os.environ.get('INDEX_MICRO_CHUNKS', 'NOT_SET')}")

    files_seen = 0
    files_indexed = 0
    points_indexed = 0

    skipped_fsmeta = 0
    skipped_cache = 0
    skipped_qdrant = 0

    def make_point(pid, dense_vec, lex_vec, payload):
        # Use named vectors if collection has names: store dense + lexical (+ mini if REFRAG_MODE)
        if vector_name:
            vecs = {vector_name: dense_vec, LEX_VECTOR_NAME: lex_vec}
            try:
                if os.environ.get("REFRAG_MODE", "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    vecs[MINI_VECTOR_NAME] = project_mini(list(dense_vec), MINI_VEC_DIM)
            except Exception:
                pass
            return models.PointStruct(id=pid, vector=vecs, payload=payload)
        else:
            # unnamed collection: store dense only
            return models.PointStruct(id=pid, vector=dense_vec, payload=payload)

    # Track per-file hashes across the entire run for cache updates on any flush
    batch_file_hashes = {}

    fast_fs = _env_truthy(os.environ.get("INDEX_FS_FASTPATH"), False)

    # Collect files for progress bar (fast: just list paths, no I/O)
    all_files = list(iter_files(root))
    total_files = len(all_files)
    print(f"Found {total_files} files to process")

    # Use tqdm progress bar if available, otherwise simple iteration
    # When progress bar is active, suppress per-file skip messages
    _use_progress_bar = tqdm is not None
    if _use_progress_bar:
        file_iter = tqdm(
            all_files,
            desc="Indexing",
            unit="file",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )
    else:
        file_iter = all_files

    def _maybe_update_postfix() -> None:
        if not _use_progress_bar:
            return
        try:
            if files_seen % 25 != 0:
                return
        except Exception:
            return
        try:
            file_iter.set_postfix_str(
                f"fsmeta={skipped_fsmeta} cache={skipped_cache} qdrant={skipped_qdrant}",
                refresh=False,
            )
        except Exception:
            pass

    for file_path in file_iter:
        files_seen += 1

        # Determine collection per-file in multi-repo mode (use watcher's exact logic)
        current_collection = collection
        if use_per_repo_collections:
            if _get_collection_for_file:
                current_collection = _get_collection_for_file(file_path)
                # Ensure collection exists on first use
                ensure_collection_and_indexes_once(client, current_collection, dim, vector_name)
            else:
                current_collection = get_collection_name(ws_path) if get_collection_name else "default-collection"

        # Optional fs-metadata fast-path: skip files whose size/mtime match cache
        if skip_unchanged and fast_fs and get_cached_file_meta is not None:
            try:
                per_file_repo_for_cache = (
                    _detect_repo_name_from_path(file_path)
                    if use_per_repo_collections
                    else repo_tag
                )
                meta = get_cached_file_meta(str(file_path), per_file_repo_for_cache) or {}
                size = meta.get("size")
                mtime = meta.get("mtime")
                if size is not None and mtime is not None:
                    st = file_path.stat()
                    if int(getattr(st, "st_size", 0)) == int(size) and int(
                        getattr(st, "st_mtime", 0)
                    ) == int(mtime):
                        if not _use_progress_bar:
                            print(f"Skipping unchanged file (fs-meta): {file_path}")
                        else:
                            skipped_fsmeta += 1
                            _maybe_update_postfix()
                        continue
            except Exception:
                pass

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        # Skip empty files
        if not text or not text.strip():
            continue

        language = detect_language(file_path)
        file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

        per_file_repo = repo_tag
        if use_per_repo_collections:
            per_file_repo = (
                _detect_repo_name_from_path(file_path)
                if _detect_repo_name_from_path
                else repo_tag
            )
        if per_file_repo:
            touched_repos.add(per_file_repo)
            repo_roots.setdefault(
                per_file_repo,
                str(Path(workspace_root).resolve() / per_file_repo),
            )

        # Derive logical repo identity and repo-relative path for cross-worktree reuse.
        repo_id: str | None = None
        repo_rel_path: str | None = None
        try:
            if get_workspace_state is not None:
                ws_root = os.environ.get("WATCH_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"
                state = get_workspace_state(ws_root, per_file_repo)
                lrid = state.get("logical_repo_id") if isinstance(state, dict) else None
                if isinstance(lrid, str) and lrid:
                    repo_id = lrid
            try:
                fp_resolved = file_path.resolve()
            except Exception:
                fp_resolved = file_path
            try:
                ws_base = Path(workspace_root).resolve()
                repo_root = ws_base
                if per_file_repo:
                    candidate = ws_base / per_file_repo
                    if candidate.exists():
                        repo_root = candidate
                rel = fp_resolved.relative_to(repo_root)
                repo_rel_path = rel.as_posix()
            except Exception:
                repo_rel_path = None
        except Exception:
            repo_id = None
            repo_rel_path = None

        # Skip unchanged files if enabled (default)
        if skip_unchanged:
            # Prefer local workspace cache to avoid Qdrant lookups
            try:
                if get_cached_file_hash:
                    prev_local = get_cached_file_hash(str(file_path), per_file_repo)
                    if prev_local and file_hash and prev_local == file_hash:
                        # When fs fast-path is enabled, refresh cache entry with size/mtime
                        if fast_fs and set_cached_file_hash:
                            try:
                                need_refresh = True
                                try:
                                    if get_cached_file_meta:
                                        _m = get_cached_file_meta(str(file_path), per_file_repo) or {}
                                        if _m.get("size") is not None and _m.get("mtime") is not None:
                                            need_refresh = False
                                except Exception:
                                    need_refresh = True
                                if need_refresh:
                                    set_cached_file_hash(str(file_path), file_hash, per_file_repo)
                            except Exception:
                                pass
                        # Only print skip messages if no progress bar
                        if not _use_progress_bar:
                            if PROGRESS_EVERY <= 0 and files_seen % 50 == 0:
                                print(f"... processed {files_seen} files (skipping unchanged, cache)")
                                try:
                                    if update_indexing_status:
                                        target_workspace = (
                                            ws_path if not use_per_repo_collections else str(file_path.parent)
                                        )
                                        target_repo = (
                                            repo_tag if not use_per_repo_collections else per_file_repo
                                        )
                                        update_indexing_status(
                                            workspace_path=target_workspace,
                                            status={
                                                "state": "indexing",
                                                "progress": {
                                                    "files_processed": files_seen,
                                                    "total_files": None,
                                                    "current_file": str(file_path),
                                                },
                                            },
                                            repo_name=target_repo,
                                        )
                                except Exception:
                                    pass
                            else:
                                print(f"Skipping unchanged file (cache): {file_path}")
                        else:
                            skipped_cache += 1
                            _maybe_update_postfix()
                        continue
            except Exception:
                pass

            # Check existing indexed hash in Qdrant (logical identity when available)
            # Skip this when INDEX_TRUST_CACHE is enabled - rely solely on local cache
            if not _trust_cache:
                prev = get_indexed_file_hash(
                    client,
                    current_collection,
                    str(file_path),
                    repo_id=repo_id,
                    repo_rel_path=repo_rel_path,
                )
                if prev and file_hash and prev == file_hash:
                    # File exists in Qdrant with same hash - cache it locally for next time
                    try:
                        if set_cached_file_hash:
                            set_cached_file_hash(str(file_path), file_hash, per_file_repo)
                    except Exception:
                        pass
                    # Only print skip messages if no progress bar
                    if not _use_progress_bar:
                        if PROGRESS_EVERY <= 0 and files_seen % 50 == 0:
                            print(f"... processed {files_seen} files (skipping unchanged)")
                        else:
                            print(f"Skipping unchanged file: {file_path}")
                    else:
                        skipped_qdrant += 1
                        _maybe_update_postfix()
                    continue

            # At this point, file content has changed vs previous index; attempt smart reindex when enabled
            if _smart_symbol_reindexing_enabled():
                try:
                    use_smart, smart_reason = should_use_smart_reindexing(str(file_path), file_hash)
                    if use_smart:
                        print(f"[SMART_REINDEX] Using smart reindexing for {file_path} ({smart_reason})")
                        status = process_file_with_smart_reindexing(
                            file_path,
                            text,
                            language,
                            client,
                            current_collection,
                            per_file_repo,
                            model,
                            vector_name,
                        )
                        if status == "success":
                            files_indexed += 1
                            # Smart path handles point counts internally; skip full reindex for this file
                            continue
                        else:
                            print(
                                f"[SMART_REINDEX] Smart reindex failed for {file_path} (status={status}), falling back to full reindex"
                            )
                    else:
                        print(f"[SMART_REINDEX] Using full reindexing for {file_path} ({smart_reason})")
                except Exception as e:
                    print(f"[SMART_REINDEX] Smart reindexing failed, falling back to full reindex: {e}")

        # Dedupe per-file by deleting previous points for this path (default)
        if dedupe:
            delete_points_by_path(client, current_collection, str(file_path))

        files_indexed += 1
        # Progress: show each file being indexed
        print(f"Indexing [{files_indexed}]: {file_path}")
        symbols = _extract_symbols(language, text)
        imports, calls = _get_imports_calls(language, text)
        last_mod, churn_count, author_count = _git_metadata(file_path)

        # Get changed symbols for pseudo processing optimization (reuse existing pattern)
        changed_symbols = set()
        if get_cached_symbols and set_cached_symbols:
            cached_symbols = get_cached_symbols(str(file_path))
            if cached_symbols:
                current_symbols = extract_symbols_with_tree_sitter(str(file_path))
                _, changed = compare_symbol_changes(cached_symbols, current_symbols)
                # Convert symbol names to IDs for lookup
                for symbol_data in current_symbols.values():
                    symbol_id = f"{symbol_data['type']}_{symbol_data['name']}_{symbol_data['start_line']}"
                    if symbol_id in changed:
                        changed_symbols.add(symbol_id)

        # Micro-chunking (token-based) takes precedence; else semantic; else line-based
        use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_micro:
            chunks = chunk_by_tokens(text)
            try:
                _cap = int(os.environ.get("MAX_MICRO_CHUNKS_PER_FILE", "500") or 500)
                if _cap > 0 and len(chunks) > _cap:
                    _before = len(chunks)
                    # Priority: keep ALL_CAPS assignments near top so constants survive capping
                    try:
                        import re as _re
                        if os.environ.get("INGEST_CONSTANT_PRIORITY", "1").lower() not in {"0", "false", "off", "no"}:
                            _pat = _re.compile(r"\b[A-Z_]{2,}\s*=\s*")
                            _important = []
                            _rest = []
                            for c in chunks:
                                txt = c.get("text", "")
                                (_important if _pat.search(txt) else _rest).append(c)
                            chunks = (_important + _rest)[:_cap]
                        else:
                            chunks = chunks[:_cap]
                    except Exception:
                        chunks = chunks[:_cap]
                    try:
                        print(
                            f"[ingest] micro-chunks capped path={file_path} count={_before}->{len(chunks)} cap={_cap}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        elif use_semantic:
            chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
        else:
            chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)
        for ch in chunks:
            info = build_information(
                language,
                file_path,
                ch["start"],
                ch["end"],
                ch["text"].splitlines()[0] if ch["text"] else "",
            )
            kind, sym, sym_path = _choose_symbol_for_chunk(
                ch["start"], ch["end"], symbols
            )
            # If chunk_semantic returned embedded symbol/kind, prefer it
            if "kind" in ch and ch.get("kind"):
                kind = ch.get("kind") or kind
            if "symbol" in ch and ch.get("symbol"):
                sym = ch.get("symbol") or sym
            if "symbol_path" in ch and ch.get("symbol_path"):
                sym_path = ch.get("symbol_path") or sym_path
            # Ensure chunks carry symbol metadata so pseudo gating works across all chunking modes
            if not ch.get("kind") and kind:
                ch["kind"] = kind
            if not ch.get("symbol") and sym:
                ch["symbol"] = sym
            if not ch.get("symbol_path") and sym_path:
                ch["symbol_path"] = sym_path
            # Track both container path (/work mirror) and original host path
            _cur_path = str(file_path)
            _host_path, _container_path = _compute_host_and_container_paths(_cur_path)

            payload = {
                "document": info,
                "information": info,
                "metadata": {
                    "path": str(file_path),
                    "path_prefix": str(file_path.parent),
                    "ext": str(file_path.suffix).lstrip(".").lower(),
                    "language": language,
                    "kind": kind,
                    "symbol": sym,
                    "symbol_path": sym_path or "",
                    "repo": per_file_repo,
                    "start_line": ch["start"],
                    "end_line": ch["end"],
                    "code": ch["text"],
                    "file_hash": file_hash,
                    "imports": imports,
                    "calls": calls,
                    "ingested_at": int(time.time()),
                    "last_modified_at": int(last_mod),
                    "churn_count": int(churn_count),
                    "author_count": int(author_count),
                    # Logical identity for cross-worktree reuse
                    "repo_id": repo_id,
                    "repo_rel_path": repo_rel_path,
                    # New: dual-path tracking
                    "host_path": _host_path,
                    "container_path": _container_path,
                },
            }
            # Optional LLM enrichment for lexical retrieval: pseudo + tags per micro-chunk
            # Use symbol-aware gating and cached pseudo/tags where possible
            pseudo = ""
            tags: list[str] = []
            if pseudo_mode != "off":
                needs_pseudo, cached_pseudo, cached_tags = should_process_pseudo_for_chunk(
                    str(file_path), ch, changed_symbols
                )
                pseudo, tags = cached_pseudo, cached_tags
                if pseudo_mode == "full" and needs_pseudo:
                    try:
                        pseudo, tags = generate_pseudo_tags(ch.get("text") or "")
                        if pseudo or tags:
                            symbol_name = ch.get("symbol", "")
                            if symbol_name:
                                kind = ch.get("kind", "unknown")
                                start_line = ch.get("start", 0)
                                symbol_id = f"{kind}_{symbol_name}_{start_line}"
                                if set_cached_pseudo:
                                    set_cached_pseudo(str(file_path), symbol_id, pseudo, tags, file_hash)
                    except Exception:
                        pass
            if pseudo:
                payload["pseudo"] = pseudo
            if tags:
                payload["tags"] = tags
            batch_texts.append(info)
            batch_meta.append(payload)
            # Track per-file latest hash once we add the first chunk to any batch
            try:
                batch_file_hashes[str(file_path)] = file_hash
            except Exception:
                pass

            batch_ids.append(
                hash_id(ch["text"], str(file_path), ch["start"], ch["end"])
            )
            aug_lex_text = (ch.get("text") or "") + (" " + pseudo if pseudo else "") + (" " + " ".join(tags) if tags else "")
            batch_lex.append(_lex_hash_vector_text(aug_lex_text))
            points_indexed += 1
            if len(batch_texts) >= BATCH_SIZE:
                vectors = embed_batch(model, batch_texts)
                # Inject pid_str into payloads for server-side gating
                for _idx, _m in enumerate(batch_meta):
                    try:
                        _m["pid_str"] = str(batch_ids[_idx])
                    except Exception:
                        pass
                points = [
                    make_point(i, v, lx, m)
                    for i, v, lx, m in zip(batch_ids, vectors, batch_lex, batch_meta)
                ]
                upsert_points(client, current_collection, points)
                # Update local file-hash cache for any files that had chunks in this flush
                try:
                    if set_cached_file_hash:
                        for _p, _h in list(batch_file_hashes.items()):
                            try:
                                if _p and _h:
                                    file_repo_tag = (
                                        _detect_repo_name_from_path(Path(_p))
                                        if use_per_repo_collections
                                        else repo_tag
                                    )
                                    repos_touched_name = file_repo_tag or per_file_repo
                                    if repos_touched_name:
                                        touched_repos.add(repos_touched_name)
                                        repo_roots.setdefault(
                                            repos_touched_name,
                                            str(Path(workspace_root).resolve() / repos_touched_name),
                                        )
                                    set_cached_file_hash(_p, _h, file_repo_tag)
                                    # Add to bloom filter for future fast-path checks
                                    _bloom_add(_h)
                            except Exception:
                                continue
                except Exception:
                    pass

                batch_texts, batch_meta, batch_ids, batch_lex = [], [], [], []

        if PROGRESS_EVERY > 0 and files_seen % PROGRESS_EVERY == 0:
            print(
                f"Progress: files_seen={files_seen}, files_indexed={files_indexed}, chunks_indexed={points_indexed}"
            )
            try:
                if update_indexing_status:
                    per_file_repo = repo_tag
                    if use_per_repo_collections:
                        per_file_repo = (
                            _detect_repo_name_from_path(file_path)
                            if _detect_repo_name_from_path
                            else repo_tag
                        )
                    if per_file_repo:
                        update_indexing_status(
                            workspace_path=str(file_path.parent),
                            status={
                                "state": "indexing",
                                "progress": {
                                    "files_processed": files_indexed,
                                    "total_files": files_seen,
                                    "current_file": str(file_path),
                                },
                            },
                            repo_name=per_file_repo,
                        )
            except Exception as e:
                # Log progress update errors instead of silent failure
                import traceback
                print(f"[ERROR] Failed to update indexing progress: {e}")
                print(f"[ERROR] Traceback: {traceback.format_exc()}")

    if batch_texts:
        vectors = embed_batch(model, batch_texts)
        # Inject pid_str into payloads for server-side gating (final batch)
        for _idx, _m in enumerate(batch_meta):
            try:
                _m["pid_str"] = str(batch_ids[_idx])
            except Exception:
                pass
        points = [
            make_point(i, v, lx, m)
            for i, v, lx, m in zip(batch_ids, vectors, batch_lex, batch_meta)
        ]
        upsert_points(client, current_collection, points)
        # Update local file-hash cache for any files that had chunks during this run (final flush)
        try:
            if set_cached_file_hash:
                for _p, _h in list(batch_file_hashes.items()):
                    try:
                        if _p and _h:
                            per_file_repo = (
                                _detect_repo_name_from_path(Path(_p))
                                if use_per_repo_collections
                                else repo_tag
                            )
                            if per_file_repo:
                                set_cached_file_hash(_p, _h, per_file_repo)
                            # Add to bloom filter for future fast-path checks
                            _bloom_add(_h)
                    except Exception:
                        continue

            # NEW: Update symbol cache for files that were processed
            if set_cached_symbols and _smart_symbol_reindexing_enabled():
                try:
                    # Process files that had chunks and extract/update their symbol cache
                    processed_files = set(str(Path(_p).resolve()) for _p in batch_file_hashes.keys())

                    for file_path_str in processed_files:
                        try:
                            # Extract current symbols for this file
                            current_symbols = extract_symbols_with_tree_sitter(file_path_str)
                            if current_symbols:
                                # Generate file hash for this file
                                with open(file_path_str, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                file_hash = hashlib.sha1(content.encode('utf-8', errors='ignore')).hexdigest()

                                # Save symbol cache
                                set_cached_symbols(file_path_str, current_symbols, file_hash)
                                print(f"[SYMBOL_CACHE] Updated symbols for {Path(file_path_str).name}: {len(current_symbols)} symbols")
                        except Exception as e:
                            print(f"[SYMBOL_CACHE] Failed to update symbols for {Path(_p).name}: {e}")
                except Exception as e:
                    print(f"[SYMBOL_CACHE] Symbol cache update failed: {e}")
        except Exception:
            pass

    print(
        f"Indexing complete. files_seen={files_seen}, files_indexed={files_indexed}, chunks_indexed={points_indexed}"
    )

    # Workspace state: mark completion
    try:
        if log_activity:
            # Extract repo name from workspace path for log_activity
            repo_name = None
            if use_per_repo_collections:
                # In multi-repo mode, we need to determine which repo this activity belongs to
                # For scan completion, we use the workspace path as the repo identifier
                repo_name = _detect_repo_name_from_path(Path(ws_path))

            log_activity(
                repo_name=repo_name,
                action="scan-completed",
                file_path="",
                details={
                    "files_seen": files_seen,
                    "files_indexed": files_indexed,
                    "chunks_indexed": points_indexed,
                },
            )
        if update_indexing_status:
            for repo_name in touched_repos or ({repo_tag} if repo_tag else set()):
                try:
                    target_ws = repo_roots.get(repo_name) or ws_path
                    update_indexing_status(
                        workspace_path=target_ws,
                        status={
                            "state": "idle",
                            "progress": {"files_processed": files_indexed, "total_files": None},
                        },
                        repo_name=repo_name,
                    )
                except Exception:
                    continue
    except Exception as e:
        # Log the error instead of silently swallowing it
        import traceback
        print(f"[ERROR] Failed to update workspace state after indexing completion: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")

    # Persist bloom filter to disk at end of indexing run
    _bloom_save()


def process_file_with_smart_reindexing(
    file_path,
    text: str,
    language: str,
    client: QdrantClient,
    current_collection: str,
    per_file_repo,
    model: "TextEmbedding",
    vector_name: str | None,
) -> str:
    """Smart, chunk-level reindexing for a single file.

    Rebuilds all points for the file with *accurate* line numbers while:
    - Reusing existing embeddings/lexical vectors for unchanged chunks (by code content), and
    - Re-embedding only for changed chunks.

    Symbol cache is used to gate pseudo/tag generation, but embedding reuse is decided
    at the chunk level by matching previous chunk code.

    TODO(logical_repo): consider loading existing points by logical identity
    (repo_id + repo_rel_path) instead of metadata.path so worktrees/branches
    sharing a repo can reuse embeddings across slugs, not just per-path.
    """
    try:
        try:
            p = Path(str(file_path))
            if _should_skip_explicit_file_by_excluder(p):
                try:
                    delete_points_by_path(client, current_collection, str(p))
                except Exception:
                    pass
                print(f"[SMART_REINDEX] Skipping excluded file: {file_path}")
                return "skipped"
        except Exception:
            return "skipped"

        print(f"[SMART_REINDEX] Processing {file_path} with chunk-level reindexing")

        # Normalize path / types
        try:
            fp = str(file_path)
        except Exception:
            fp = str(file_path)
        try:
            if not isinstance(file_path, Path):
                file_path = Path(fp)
        except Exception:
            file_path = Path(fp)

        # Compute current file hash
        file_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

        # Extract current symbols for diffing (dict) and for chunk mapping (List[_Sym])
        symbol_meta = extract_symbols_with_tree_sitter(fp)
        if not symbol_meta:
            print(f"[SMART_REINDEX] No symbols found in {file_path}, falling back to full reindex")
            return "failed"

        # Use the dict-style symbol_meta for cache diffing
        cached_symbols = get_cached_symbols(fp) if get_cached_symbols else {}
        unchanged_symbols: list[str] = []
        changed_symbols: list[str] = []
        if cached_symbols and compare_symbol_changes:
            try:
                unchanged_symbols, changed_symbols = compare_symbol_changes(
                    cached_symbols, symbol_meta
                )
            except Exception:
                # On failure, treat everything as changed
                unchanged_symbols = []
                changed_symbols = list(symbol_meta.keys())
        else:
            changed_symbols = list(symbol_meta.keys())
        changed_set = set(changed_symbols)

        # Load existing points for this file (for embedding reuse)
        existing_points = []
        try:
            filt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.path", match=models.MatchValue(value=fp)
                    )
                ]
            )
            next_offset = None
            while True:
                pts, next_offset = client.scroll(
                    collection_name=current_collection,
                    scroll_filter=filt,
                    with_payload=True,
                    with_vectors=True,
                    limit=256,
                    offset=next_offset,
                )
                if not pts:
                    break
                existing_points.extend(pts)
                if next_offset is None:
                    break
        except Exception as e:
            print(f"[SMART_REINDEX] Failed to load existing points for {file_path}: {e}")
            existing_points = []

        # Index existing points by (symbol_id, code, embedding_text) for reuse.
        # Important: the dense embedding is computed from `info` (payload['information']).
        # If line ranges change, `info` changes; reusing an old dense vector would be wrong.
        points_by_code: dict[tuple[str, str, str], list[models.Record]] = {}
        try:
            for rec in existing_points:
                payload = rec.payload or {}
                md = payload.get("metadata") or {}
                code_text = md.get("code") or ""
                embed_text = payload.get("information") or payload.get("document") or ""
                kind = md.get("kind") or ""
                sym_name = md.get("symbol") or ""
                start_line = md.get("start_line") or 0
                symbol_id = (
                    f"{kind}_{sym_name}_{start_line}"
                    if kind and sym_name and start_line
                    else ""
                )
                key = (symbol_id, code_text, embed_text) if symbol_id else ("", code_text, embed_text)
                points_by_code.setdefault(key, []).append(rec)
        except Exception:
            points_by_code = {}

        # Chunk current file using the same strategy as normal indexing
        CHUNK_LINES = int(os.environ.get("INDEX_CHUNK_LINES", "120") or 120)
        CHUNK_OVERLAP = int(os.environ.get("INDEX_CHUNK_OVERLAP", "20") or 20)
        use_micro = os.environ.get("INDEX_MICRO_CHUNKS", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        use_semantic = os.environ.get("INDEX_SEMANTIC_CHUNKS", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if use_micro:
            chunks = chunk_by_tokens(text)
            symbol_spans: list[_Sym] = _extract_symbols(language, text)
        elif use_semantic:
            chunks = chunk_semantic(text, language, CHUNK_LINES, CHUNK_OVERLAP)
            symbol_spans = _extract_symbols(language, text)
        else:
            chunks = chunk_lines(text, CHUNK_LINES, CHUNK_OVERLAP)
            symbol_spans = _extract_symbols(language, text)

        # Prepare collections for reused vs newly embedded points
        reused_points: list[models.PointStruct] = []
        embed_texts: list[str] = []
        embed_payloads: list[dict] = []
        embed_ids: list[int] = []
        embed_lex: list[list[float]] = []

        imports, calls = _get_imports_calls(language, text)
        last_mod, churn_count, author_count = _git_metadata(file_path)

        for ch in chunks:
            info = build_information(
                language,
                file_path,
                ch["start"],
                ch["end"],
                ch["text"].splitlines()[0] if ch["text"] else "",
            )
            # Use span-style symbols for mapping chunks to symbols
            kind, sym, sym_path = _choose_symbol_for_chunk(
                ch["start"], ch["end"], symbol_spans
            )
            # Prefer embedded symbol metadata from semantic chunker when present
            if "kind" in ch and ch.get("kind"):
                kind = ch.get("kind") or kind
            if "symbol" in ch and ch.get("symbol"):
                sym = ch.get("symbol") or sym
            if "symbol_path" in ch and ch.get("symbol_path"):
                sym_path = ch.get("symbol_path") or sym_path
            # Ensure chunks carry symbol metadata so pseudo gating works
            if not ch.get("kind") and kind:
                ch["kind"] = kind
            if not ch.get("symbol") and sym:
                ch["symbol"] = sym
            if not ch.get("symbol_path") and sym_path:
                ch["symbol_path"] = sym_path

            # Basic metadata payload
            _cur_path = str(file_path)
            _host_path, _container_path = _compute_host_and_container_paths(_cur_path)

            payload = {
                "document": info,
                "information": info,
                "metadata": {
                    "path": str(file_path),
                    "path_prefix": str(file_path.parent),
                    "ext": str(file_path.suffix).lstrip(".").lower(),
                    "language": language,
                    "kind": kind,
                    "symbol": sym,
                    "symbol_path": sym_path or "",
                    "repo": per_file_repo,
                    "start_line": ch["start"],
                    "end_line": ch["end"],
                    "code": ch["text"],
                    "file_hash": file_hash,
                    "imports": imports,
                    "calls": calls,
                    "ingested_at": int(time.time()),
                    "last_modified_at": int(last_mod),
                    "churn_count": int(churn_count),
                    "author_count": int(author_count),
                    "host_path": _host_path,
                    "container_path": _container_path,
                },
            }

            # Pseudo / tags with symbol-aware gating
            needs_pseudo, cached_pseudo, cached_tags = should_process_pseudo_for_chunk(
                fp, ch, changed_set
            )
            pseudo, tags = cached_pseudo, cached_tags
            if needs_pseudo:
                try:
                    pseudo, tags = generate_pseudo_tags(ch.get("text") or "")
                    if pseudo or tags:
                        symbol_name = ch.get("symbol", "")
                        if symbol_name:
                            k = ch.get("kind", "unknown")
                            start_line = ch.get("start", 0)
                            sid = f"{k}_{symbol_name}_{start_line}"
                            if set_cached_pseudo:
                                set_cached_pseudo(fp, sid, pseudo, tags, file_hash)
                except Exception:
                    pass
            if pseudo:
                payload["pseudo"] = pseudo
            if tags:
                payload["tags"] = tags

            # Decide whether we can reuse an existing embedding for this chunk
            code_text = ch.get("text") or ""
            chunk_symbol_id = ""
            if sym and kind:
                chunk_symbol_id = f"{kind}_{sym}_{ch['start']}"

            reuse_key = (chunk_symbol_id, code_text, info)
            fallback_key = ("", code_text, info)
            reused_rec = None
            used_key = None
            bucket = points_by_code.get(reuse_key)
            if bucket is not None:
                used_key = reuse_key
            else:
                bucket = points_by_code.get(fallback_key)
                if bucket is not None:
                    used_key = fallback_key
            if bucket:
                try:
                    reused_rec = bucket.pop()
                    if not bucket:
                        # Clean up empty bucket
                        if used_key is not None:
                            points_by_code.pop(used_key, None)
                except Exception:
                    reused_rec = None

            if reused_rec is not None:
                try:
                    vec = reused_rec.vector
                    # Validate vector shape before reuse.
                    if vector_name and isinstance(vec, dict) and vector_name not in vec:
                        raise ValueError("reused vector missing dense key")
                    # If we're reusing an existing embedding, we still need to refresh
                    # the lexical vector because it depends on pseudo/tags (and can drift).
                    aug_lex_text = (code_text or "") + (" " + pseudo if pseudo else "") + (
                        " " + " ".join(tags) if tags else ""
                    )
                    refreshed_lex = _lex_hash_vector_text(aug_lex_text)
                    if vector_name:
                        if isinstance(vec, dict):
                            # Named vectors: keep dense/mini as-is, overwrite lex.
                            vec = dict(vec)
                            vec[LEX_VECTOR_NAME] = refreshed_lex
                        else:
                            # Unexpected shape: treat as dense and rebuild named vectors.
                            vecs = {vector_name: vec, LEX_VECTOR_NAME: refreshed_lex}
                            try:
                                if os.environ.get("REFRAG_MODE", "").strip().lower() in {
                                    "1",
                                    "true",
                                    "yes",
                                    "on",
                                }:
                                    vecs[MINI_VECTOR_NAME] = project_mini(
                                        list(vec), MINI_VEC_DIM
                                    )
                            except Exception:
                                pass
                            vec = vecs
                    else:
                        # Unnamed vectors collection: ensure we pass dense-only vector.
                        if isinstance(vec, dict):
                            # Prefer any non-lex/non-mini vector as dense.
                            dense = None
                            try:
                                for k, v in vec.items():
                                    if k not in {LEX_VECTOR_NAME, MINI_VECTOR_NAME}:
                                        dense = v
                                        break
                            except Exception:
                                dense = None
                            if dense is None:
                                raise ValueError("reused vector has no dense component")
                            vec = dense
                    pid = hash_id(code_text, fp, ch["start"], ch["end"])
                    reused_points.append(
                        models.PointStruct(id=pid, vector=vec, payload=payload)
                    )
                    continue
                except Exception:
                    # Fall through to re-embedding path
                    pass

            # Need to embed this chunk
            embed_texts.append(info)
            embed_payloads.append(payload)
            embed_ids.append(
                hash_id(code_text, fp, ch["start"], ch["end"])
            )
            aug_lex_text = (code_text or "") + (
                " " + pseudo if pseudo else ""
            ) + (" " + " ".join(tags) if tags else "")
            embed_lex.append(_lex_hash_vector_text(aug_lex_text))

        # Embed changed/new chunks and build final point set
        new_points: list[models.PointStruct] = []
        if embed_texts:
            vectors = embed_batch(model, embed_texts)
            for pid, v, lx, pl in zip(
                embed_ids,
                vectors,
                embed_lex,
                embed_payloads,
            ):
                if vector_name:
                    vecs = {vector_name: v, LEX_VECTOR_NAME: lx}
                    try:
                        if os.environ.get("REFRAG_MODE", "").strip().lower() in {
                            "1",
                            "true",
                            "yes",
                            "on",
                        }:
                            vecs[MINI_VECTOR_NAME] = project_mini(
                                list(v), MINI_VEC_DIM
                            )
                    except Exception:
                        pass
                    new_points.append(
                        models.PointStruct(id=pid, vector=vecs, payload=pl)
                    )
                else:
                    new_points.append(
                        models.PointStruct(id=pid, vector=v, payload=pl)
                    )

        all_points = reused_points + new_points

        # Replace existing points for this file with the new set
        try:
            delete_points_by_path(client, current_collection, fp)
        except Exception as e:
            print(f"[SMART_REINDEX] Failed to delete old points for {file_path}: {e}")

        if all_points:
            upsert_points(client, current_collection, all_points)

        # Update caches with the new state
        try:
            if set_cached_symbols:
                set_cached_symbols(fp, symbol_meta, file_hash)
        except Exception as e:
            print(f"[SMART_REINDEX] Failed to update symbol cache for {file_path}: {e}")
        try:
            if set_cached_file_hash:
                set_cached_file_hash(fp, file_hash, per_file_repo)
        except Exception:
            pass

        print(
            f"[SMART_REINDEX] Completed {file_path}: chunks={len(chunks)}, reused_points={len(reused_points)}, embedded_points={len(new_points)}"
        )
        return "success"

    except Exception as e:
        print(f"[SMART_REINDEX] Failed to process {file_path}: {e}")
        import traceback
        print(f"[SMART_REINDEX] Traceback: {traceback.format_exc()}")
        return "failed"

def main():
    parser = argparse.ArgumentParser(
        description="Index code into Qdrant with metadata for MCP code search."
    )
    parser.add_argument("--root", type=str, default=".", help="Root directory to index")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection before indexing",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not delete existing points for each file before inserting",
    )
    parser.add_argument(
        "--no-skip-unchanged",
        action="store_true",
        help="Do not skip files whose content hash matches existing index",
    )
    # Exclusion controls
    parser.add_argument(
        "--ignore-file",
        type=str,
        default=None,
        help="Path to a .qdrantignore-style file of patterns to exclude",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Disable default exclusions (models, node_modules, build, venv, .git, etc.)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Additional exclude pattern(s); can be used multiple times or comma-separated",
    )
    # Scaling controls
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding/upsert batch size (default 64)",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=None,
        help="Max lines per chunk (default 120)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap lines between chunks (default 20)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Print progress every N files (default 200; 0 disables)",
    )
    # GLM psueo tag test - # TODO: Remove GLM psuedo tag test harness after confirming 100% stable and not needed
    parser.add_argument(
        "--test-pseudo",
        type=str,
        default=None,
        help="Test generate_pseudo_tags on the given code snippet and print result, then exit",
    )
    parser.add_argument(
        "--test-pseudo-file",
        type=str,
        default=None,
        help="Test generate_pseudo_tags on the contents of the given file and print result, then exit",
    )
    # End

    args = parser.parse_args()

    # Map CLI overrides to env so downstream helpers pick them up
    if args.ignore_file:
        os.environ["QDRANT_IGNORE_FILE"] = args.ignore_file
    if args.no_default_excludes:
        os.environ["QDRANT_DEFAULT_EXCLUDES"] = "0"
    if args.exclude:
        # allow comma-separated and repeated flags
        parts = []
        for e in args.exclude:
            parts.extend([p.strip() for p in str(e).split(",") if p.strip()])
        if parts:
            os.environ["QDRANT_EXCLUDES"] = ",".join(parts)
    if args.batch_size is not None:
        os.environ["INDEX_BATCH_SIZE"] = str(args.batch_size)
    if args.chunk_lines is not None:
        os.environ["INDEX_CHUNK_LINES"] = str(args.chunk_lines)
    if args.chunk_overlap is not None:
        os.environ["INDEX_CHUNK_OVERLAP"] = str(args.chunk_overlap)
    if args.progress_every is not None:
        os.environ["INDEX_PROGRESS_EVERY"] = str(args.progress_every)

    # TODO: Remove GLM psuedo tag test harness after confirming 100% stable and not needed
    # # Optional test mode: exercise generate_pseudo_tags (including GLM runtime) and exit
    if args.test_pseudo or args.test_pseudo_file:
        import json as _json

        code_text = ""
        if args.test_pseudo:
            code_text = args.test_pseudo
        if args.test_pseudo_file:
            try:
                code_text = Path(args.test_pseudo_file).read_text(
                    encoding="utf-8", errors="ignore"
                )
            except Exception as e:
                print(f"[TEST_PSEUDO] Failed to read file {args.test_pseudo_file}: {e}")
                return
        if not code_text.strip():
            print("[TEST_PSEUDO] No code text provided")
            return

        # Use the normal generate_pseudo_tags path so behavior matches indexing.
        try:
            from scripts.refrag_llamacpp import get_runtime_kind  # type: ignore

            runtime = get_runtime_kind()
        except Exception:
            runtime = "unknown"

        pseudo, tags = "", []
        try:
            pseudo, tags = generate_pseudo_tags(code_text)
        except Exception as e:
            print(f"[TEST_PSEUDO] Error while generating pseudo tags: {e}")

        print(
            _json.dumps(
                {
                    "runtime": runtime,
                    "pseudo": pseudo,
                    "tags": tags,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    collection = os.environ.get("COLLECTION_NAME") or os.environ.get("DEFAULT_COLLECTION") or "codebase"
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    # Resolve collection name based on multi-repo mode
    multi_repo = bool(is_multi_repo_mode and is_multi_repo_mode())
    if multi_repo:
        # Multi-repo mode: pass collection=None to trigger per-repo collection resolution
        collection = None
        print("[multi_repo] Multi-repo mode enabled - will create separate collections per repository")
    else:
        # Single-repo mode: use environment variable
        if 'get_collection_name' in globals() and get_collection_name:
            try:
                resolved = get_collection_name(str(Path(args.root).resolve()))
                placeholders = {"", "default-collection", "my-collection", "codebase"}
                if resolved and collection in placeholders:
                    collection = resolved
            except Exception:
                pass
        if not collection:
            collection = os.environ.get("COLLECTION_NAME", "codebase")
        print(f"[single_repo] Single-repo mode enabled - using collection: {collection}")

    flag = (os.environ.get("PSEUDO_BACKFILL_ENABLED") or "").strip().lower()
    pseudo_mode = "off" if flag in {"1", "true", "yes", "on"} else "full"

    index_repo(
        Path(args.root).resolve(),
        qdrant_url,
        api_key,
        collection,
        model_name,
        args.recreate,
        dedupe=(not args.no_dedupe),
        skip_unchanged=(not args.no_skip_unchanged),
        # Pseudo/tags are inlined by default; when PSEUDO_BACKFILL_ENABLED=1 we run
        # base-only and rely on the background backfill worker to add pseudo/tags.
        pseudo_mode=pseudo_mode,
    )


if __name__ == "__main__":
    main()
