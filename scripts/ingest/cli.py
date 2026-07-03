#!/usr/bin/env python3
"""
ingest/cli.py - Command-line interface for code indexing.

This module provides the CLI argument parsing and main() function for running
the indexer as a standalone script.
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

from scripts.ingest.config import (
    is_multi_repo_mode,
    get_collection_name,
)
from scripts.ingest.pipeline import index_repo
from scripts.ingest.pseudo import generate_pseudo_tags


def parse_args():
    """Parse command-line arguments."""
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
    parser.add_argument(
        "--schema-mode",
        type=str,
        default=None,
        choices=["validate", "create", "migrate"],
        help=(
            "Schema handling mode: validate (read-only), create (create if missing), "
            "migrate (add missing vectors/indexes). Default preserves legacy behavior."
        ),
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
    # GLM pseudo tag test
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

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    # Load .env file for REFRAG_*, GLM_*, and other settings
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent / ".env")
    except ImportError:
        pass  # python-dotenv not installed, rely on exported env vars
    
    args = parse_args()

    # Map CLI overrides to env so downstream helpers pick them up
    if args.ignore_file:
        os.environ["QDRANT_IGNORE_FILE"] = args.ignore_file
    if args.no_default_excludes:
        os.environ["QDRANT_DEFAULT_EXCLUDES"] = "0"
    if args.exclude:
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

    # Optional test mode: exercise generate_pseudo_tags and exit
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

        try:
            from scripts.refrag_llamacpp import get_runtime_kind
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
        print("[multi_repo] Multi-repo mode enabled - will create separate collections per repository")

        root_path = Path(args.root).resolve()
        repos = []
        try:
            if root_path.is_dir():
                for child in sorted(root_path.iterdir()):
                    try:
                        if not child.is_dir():
                            continue
                        if child.name.startswith("."):
                            continue
                        if child.name in {".codebase", "__pycache__"}:
                            continue
                        repos.append(child)
                    except Exception:
                        continue
        except Exception:
            repos = []

        if not repos:
            print(f"[multi_repo] No repo directories found under: {root_path}")
            return

        for repo_root in repos:
            repo_name = repo_root.name
            repo_collection = collection
            if get_collection_name:
                try:
                    resolved = get_collection_name(repo_name)
                    if resolved:
                        repo_collection = resolved
                except Exception:
                    pass
            if not repo_collection:
                repo_collection = "codebase"

            index_repo(
                repo_root,
                qdrant_url,
                api_key,
                repo_collection,
                model_name,
                args.recreate,
                dedupe=(not args.no_dedupe),
                skip_unchanged=(not args.no_skip_unchanged),
                pseudo_mode="off" if (os.environ.get("PSEUDO_DEFER_TO_WORKER") or "").strip().lower() in {"1", "true", "yes", "on"} else "full",
                schema_mode=args.schema_mode,
            )
        return
    else:
        if get_collection_name:
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

    flag = (os.environ.get("PSEUDO_DEFER_TO_WORKER") or "").strip().lower()
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
        pseudo_mode=pseudo_mode,
        schema_mode=args.schema_mode,
    )


if __name__ == "__main__":
    main()
