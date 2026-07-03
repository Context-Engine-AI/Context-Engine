#!/usr/bin/env python3
import os
import argparse
import subprocess
import shlex
import hashlib
from typing import List, Dict, Any
import re
import time
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
API_KEY = os.environ.get("QDRANT_API_KEY")
REPO_NAME = os.environ.get("REPO_NAME", "workspace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import TextEmbedding for type hints and fallback
from fastembed import TextEmbedding

# Use embedder factory for Qwen3 support; fallback to direct fastembed
try:
    from scripts.embedder import get_embedding_model as _get_embedding_model
    _EMBEDDER_FACTORY = True
except ImportError:
    _EMBEDDER_FACTORY = False

from scripts.utils import sanitize_vector_name as _sanitize_vector_name


def _manifest_run_id(manifest_path: str) -> str:
    try:
        stem = Path(str(manifest_path)).name
        if stem.endswith(".json"):
            stem = stem[: -len(".json")]
        stem = stem.strip()
        return stem or "git_history"
    except Exception:
        return "git_history"


def _history_prune_enabled() -> bool:
    try:
        return str(os.environ.get("GIT_HISTORY_PRUNE", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return True


def _prune_old_commit_points(
    client: QdrantClient,
    run_id: str,
    *,
    mode: str,
) -> None:
    try:
        if mode != "snapshot" or not _history_prune_enabled():
            return
    except Exception:
        return

    try:
        keep_cond = models.FieldCondition(
            key="metadata.git_history_run_id", match=models.MatchValue(value=run_id)
        )
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.kind", match=models.MatchValue(value="git_message")
                ),
                models.FieldCondition(
                    key="metadata.repo", match=models.MatchValue(value=REPO_NAME)
                ),
            ],
            must_not=[keep_cond],
        )
        client.delete(
            collection_name=COLLECTION,
            points_selector=models.FilterSelector(filter=flt),
            wait=True,
        )
    except Exception:
        return


def _cleanup_manifest_files(manifest_path: str) -> None:
    try:
        p = Path(str(manifest_path))
    except Exception:
        return

    try:
        delete_self = str(os.environ.get("GIT_HISTORY_DELETE_MANIFEST", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        delete_self = False

    if delete_self:
        try:
            p.unlink()
        except Exception:
            pass

    try:
        raw = str(os.environ.get("GIT_HISTORY_MANIFEST_MAX_FILES", "0")).strip()
        max_keep = int(raw) if raw else 0
    except Exception:
        max_keep = 0

    if max_keep <= 0:
        return

    try:
        parent = p.parent
        files = []
        for cand in parent.glob("git_history_*.json"):
            try:
                files.append((cand.stat().st_mtime, cand))
            except Exception:
                continue
        files.sort(key=lambda t: t[0])
        excess = files[:-max_keep] if len(files) > max_keep else []
        for _ts, fp in excess:
            try:
                fp.unlink()
            except Exception:
                continue
    except Exception:
        return


def run(cmd: str) -> str:
    p = subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{p.stderr}")
    return p.stdout


def try_run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def ensure_base_ref(args) -> str:
    # Prefer local HEAD when present
    proc = try_run("git rev-parse --verify HEAD")
    if proc.returncode == 0 and proc.stdout.strip():
        return "HEAD"
    # Fallback: fetch shallow history from remote and use its HEAD
    remote = args.remote or "origin"
    depth = args.fetch_depth or 1000
    try_run(f"git fetch --all --tags --prune --depth {depth}")
    ref = try_run(f"git symbolic-ref -q {remote}/HEAD")
    if ref.returncode == 0 and ref.stdout.strip():
        return ref.stdout.strip()
    # Last resort: try common branch names
    for b in (f"{remote}/main", f"{remote}/master"):
        if try_run(f"git rev-parse --verify {b}").returncode == 0:
            return b
    raise RuntimeError("Could not determine a base revision (HEAD or remote HEAD)")


def list_commits(args) -> List[str]:
    cmd = ["git", "rev-list", "--no-merges"]
    if args.since:
        cmd += [f"--since={args.since}"]
    if args.until:
        cmd += [f"--until={args.until}"]
    if args.author:
        cmd += [f"--author={args.author}"]
    if args.grep:
        cmd += [f"--grep={args.grep}"]
    base = ensure_base_ref(args)
    if args.path:
        cmd += [base, "--", args.path]
    else:
        cmd += [base]
    out = run(" ".join(shlex.quote(c) for c in cmd))
    commits = [l.strip() for l in out.splitlines() if l.strip()]
    if args.max_commits and len(commits) > args.max_commits:
        commits = commits[: args.max_commits]
    return commits


def _redact_emails(text: str) -> str:
    return re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<redacted>", text or ""
    )


def commit_metadata(commit: str) -> Dict[str, Any]:
    fmt = "%H%x1f%an%x1f%ae%x1f%ad%x1f%s%x1f%b"
    out = run(f"git show -s --format={fmt} {commit}")
    parts = out.strip().split("\x1f")
    sha, an, ae, ad, subj, body = (parts + [""] * 6)[:6]
    files_out = run(f"git diff-tree --no-commit-id --name-only -r {commit}")
    files = [f for f in files_out.splitlines() if f]
    message = _redact_emails((subj + ("\n" + body if body else "")).strip())
    if len(message) > 2000:
        message = message[:2000] + "…"
    return {
        "commit_id": sha,
        "author_name": an,
        # email stripped for privacy
        "authored_date": ad,
        "message": message,
        "files": files,
    }


def stable_id(commit_id: str) -> int:
    h = hashlib.sha1(commit_id.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _commit_summary_enabled() -> bool:
    """Check REFRAG_COMMIT_DESCRIBE to decide if commit summarization is enabled.

    This is an opt-in feature: set REFRAG_COMMIT_DESCRIBE=1 (and enable the decoder)
    to generate per-commit lineage summaries at ingest time.
    """
    try:
        return str(os.environ.get("REFRAG_COMMIT_DESCRIBE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return False


def generate_commit_summary(md: Dict[str, Any], diff_text: str) -> tuple[str, list[str], list[str]]:
    """Best-effort: ask local decoder to summarize a git commit.

    Returns (goal, symbols, tags). On failure returns ("", [], []).

    The summary is designed to be compact and search-friendly, mirroring the
    Context Lineage goals: high-level intent, key symbols, and short tags.
    """
    goal: str = ""
    symbols: list[str] = []
    tags: list[str] = []
    if not _commit_summary_enabled() or not diff_text.strip():
        return goal, symbols, tags
    try:
        from scripts.refrag_llamacpp import (  # type: ignore
            LlamaCppRefragClient,
            is_decoder_enabled,
            get_runtime_kind,
        )

        if not is_decoder_enabled():
            return "", [], []
        runtime = get_runtime_kind()
        commit_id = str(md.get("commit_id") or "")
        message = str(md.get("message") or "")
        files = md.get("files") or []
        try:
            files_str = "\n".join(str(f) for f in files[:50])
        except Exception:
            files_str = ""
        # Truncate diff text to keep summarization fast/token-efficient
        try:
            max_chars = int(os.environ.get("COMMIT_SUMMARY_DIFF_CHARS", "6000") or 6000)
        except Exception:
            max_chars = 6000
        body = diff_text[:max_chars]

        if runtime == "glm":
            from scripts.refrag_glm import GLMRefragClient  # type: ignore

            client = GLMRefragClient()
            prompt = (
                "You are a JSON-only function that summarizes git commits for search enrichment.\n"
                "Respond with a single JSON object and nothing else (no prose, no markdown).\n"
                "Exact format: {\"goal\": string (<=200 chars), \"symbols\": [1-6 short strings], \"tags\": [3-6 short strings]}.\n"
                f"Commit id: {commit_id}\n"
                f"Message:\n{message}\n"
                f"Files:\n{files_str}\n"
                "Diff:\n" + body
            )
            out = client.generate_with_soft_embeddings(
                prompt=prompt,
                max_tokens=int(os.environ.get("COMMIT_SUMMARY_MAX_TOKENS", "128") or 128),
                temperature=float(os.environ.get("COMMIT_SUMMARY_TEMPERATURE", "0.10") or 0.10),
                top_p=float(os.environ.get("COMMIT_SUMMARY_TOP_P", "0.9") or 0.9),
                stop=["\n\n"],
                force_json=True,
            )
        else:
            client = LlamaCppRefragClient()
            prompt = (
                "You summarize git commits for search enrichment.\n"
                "Return strictly JSON: {\"goal\": string (<=200 chars), \"symbols\": [1-6 short strings], \"tags\": [3-6 short strings]}.\n"
                f"Commit id: {commit_id}\n"
                f"Message:\n{message}\n"
                f"Files:\n{files_str}\n"
                "Diff:\n" + body
            )
            out = client.generate_with_soft_embeddings(
                prompt=prompt,
                max_tokens=int(os.environ.get("COMMIT_SUMMARY_MAX_TOKENS", "128") or 128),
                temperature=float(os.environ.get("COMMIT_SUMMARY_TEMPERATURE", "0.10") or 0.10),
                top_k=int(os.environ.get("COMMIT_SUMMARY_TOP_K", "30") or 30),
                top_p=float(os.environ.get("COMMIT_SUMMARY_TOP_P", "0.9") or 0.9),
                stop=["\n\n"],
            )
        import json as _json
        from scripts.llm_utils import strip_markdown_fences
        try:
            obj = _json.loads(strip_markdown_fences(out))
            if isinstance(obj, dict):
                g = obj.get("goal")
                s = obj.get("symbols")
                t = obj.get("tags")
                if isinstance(g, str):
                    goal = g.strip()[:200]
                if isinstance(s, list):
                    symbols = [str(x).strip() for x in s if str(x).strip()][:6]
                if isinstance(t, list):
                    tags = [str(x).strip() for x in t if str(x).strip()][:6]
        except Exception:
            pass
    except Exception:
        return "", [], []
    return goal, symbols, tags


def build_text(
    md: Dict[str, Any], max_files: int = 200, include_body: bool = True
) -> str:
    msg = md.get("message", "")
    files = md.get("files", [])
    if not include_body:
        msg = msg.splitlines()[0] if msg else ""
    head = msg.strip()
    files_part = "\n".join(files[:max_files])
    return (head + "\n\nFiles:\n" + files_part).strip()


def _ingest_from_manifest(
    manifest_path: str,
    model: Any,
    client: QdrantClient,
    vec_name: str,
    include_body: bool,
    per_batch: int,
) -> int:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read manifest {manifest_path}: {e}")
        return 0

    commits = data.get("commits") or []
    if not commits:
        print("No commits in manifest.")
        return 0

    run_id = _manifest_run_id(manifest_path)
    mode = str(data.get("mode") or "delta").strip().lower() or "delta"

    points: List[models.PointStruct] = []
    count = 0
    for c in commits:
        try:
            if not isinstance(c, dict):
                continue
            commit_id = str(c.get("commit_id") or "").strip()
            if not commit_id:
                continue
            author_name = str(c.get("author_name") or "")
            authored_date = str(c.get("authored_date") or "")
            message = str(c.get("message") or "")
            files = c.get("files") or []
            if not isinstance(files, list):
                files = []
            md: Dict[str, Any] = {
                "commit_id": commit_id,
                "author_name": author_name,
                "authored_date": authored_date,
                "message": message,
                "files": files,
            }
            text = build_text(md, include_body=include_body)
            try:
                vec = next(model.embed([text])).tolist()
            except Exception:
                continue

            goal: str = ""
            sym: List[str] = []
            tgs: List[str] = []
            diff_text = str(c.get("diff") or "")
            if diff_text.strip():
                try:
                    goal, sym, tgs = generate_commit_summary(md, diff_text)
                except Exception:
                    goal, sym, tgs = "", [], []

            md_payload: Dict[str, Any] = {
                "language": "git",
                "kind": "git_message",
                "symbol": commit_id,
                "symbol_path": commit_id,
                "repo": REPO_NAME,
                "commit_id": commit_id,
                "git_history_run_id": run_id,
                "git_history_mode": mode,
                "author_name": author_name,
                "authored_date": authored_date,
                "message": message,
                "files": files,
                "path": ".git",
                "path_prefix": ".git",
                "ingested_at": int(time.time()),
            }
            if goal:
                md_payload["lineage_goal"] = goal
            if sym:
                md_payload["lineage_symbols"] = sym
            if tgs:
                md_payload["lineage_tags"] = tgs

            payload = {
                "document": (message.splitlines()[0] if message else commit_id),
                "information": text[:512],
                "metadata": md_payload,
            }
            pid = stable_id(commit_id)
            pt = models.PointStruct(id=pid, vector={vec_name: vec}, payload=payload)
            points.append(pt)
            count += 1
            if len(points) >= per_batch:
                client.upsert(collection_name=COLLECTION, points=points)
                points.clear()
        except Exception:
            continue

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    try:
        _prune_old_commit_points(client, run_id, mode=mode)
    except Exception:
        pass
    try:
        _cleanup_manifest_files(manifest_path)
    except Exception:
        pass
    print(f"Ingested {count} commits into {COLLECTION} from manifest {manifest_path}.")
    return count


def main():
    ap = argparse.ArgumentParser(
        description="Ingest Git history into Qdrant deterministically"
    )
    ap.add_argument(
        "--since", type=str, default=None, help="e.g., '2 years ago' or '2023-01-01'"
    )
    ap.add_argument("--until", type=str, default=None)
    ap.add_argument("--author", type=str, default=None)
    ap.add_argument(
        "--grep", type=str, default=None, help="Filter commit messages by regex"
    )
    ap.add_argument(
        "--path", type=str, default=None, help="Limit history to a path subtree"
    )
    ap.add_argument("--max-commits", type=int, default=500)
    ap.add_argument("--per-batch", type=int, default=128)
    ap.add_argument(
        "--include-body", action="store_true", help="Include full body in text to embed"
    )
    ap.add_argument(
        "--remote",
        type=str,
        default="origin",
        help="Remote to fetch from if no local HEAD is present",
    )
    ap.add_argument(
        "--manifest-json",
        type=str,
        default=None,
        help="Path to git history manifest JSON produced by upload client",
    )
    ap.add_argument(
        "--fetch-depth",
        type=int,
        default=1000,
        help="Shallow fetch depth for history, when needed",
    )
    args = ap.parse_args()

    # Use embedder factory for Qwen3 support
    if _EMBEDDER_FACTORY:
        model = _get_embedding_model(MODEL_NAME)
    else:
        model = TextEmbedding(model_name=MODEL_NAME)
    vec_name = _sanitize_vector_name(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY or None)

    if args.manifest_json:
        _ingest_from_manifest(
            args.manifest_json,
            model,
            client,
            vec_name,
            args.include_body,
            args.per_batch,
        )
        return

    commits = list_commits(args)
    if not commits:
        print("No commits matched filters.")
        return

    points: List[models.PointStruct] = []
    for sha in commits:
        md = commit_metadata(sha)
        text = build_text(md, include_body=args.include_body)
        vec = next(model.embed([text])).tolist()
        goal, sym, tgs = "", [], []
        try:
            diff = run(f"git show --stat --patch --unified=3 {sha}")
            goal, sym, tgs = generate_commit_summary(md, diff)
        except Exception:
            pass

        md_payload: Dict[str, Any] = {
            "language": "git",
            "kind": "git_message",
            "symbol": md["commit_id"],
            "symbol_path": md["commit_id"],
            "repo": REPO_NAME,
            "commit_id": md["commit_id"],
            "author_name": md["author_name"],
            "authored_date": md["authored_date"],
            "message": md["message"],
            "files": md["files"],
            "path": ".git",
            "path_prefix": ".git",
            "ingested_at": int(time.time()),
        }
        if goal:
            md_payload["lineage_goal"] = goal
        if sym:
            md_payload["lineage_symbols"] = sym
        if tgs:
            md_payload["lineage_tags"] = tgs

        payload = {
            "document": (
                md.get("message", "").splitlines()[0]
                if md.get("message")
                else md["commit_id"]
            ),
            "information": text[:512],
            "metadata": md_payload,
        }
        pid = stable_id(md["commit_id"])  # deterministic per-commit
        point = models.PointStruct(id=pid, vector={vec_name: vec}, payload=payload)
        points.append(point)
        if len(points) >= args.per_batch:
            client.upsert(collection_name=COLLECTION, points=points)
            points.clear()
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(commits)} commits into {COLLECTION}.")


if __name__ == "__main__":
    main()
