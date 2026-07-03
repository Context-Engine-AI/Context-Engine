#!/usr/bin/env python3
"""
Clone public repos at pinned refs for reproducible benchmarks.

Usage:
    python bench/clone_snapshot.py --manifest bench/datasets/public_v1.json
    python bench/clone_snapshot.py --manifest bench/datasets/public_v1.json --repo kubernetes/kubernetes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"


def clone_repo(
    name: str,
    ref: str,
    dest: Path,
    shallow: bool = True,
) -> Optional[str]:
    """Clone a repo at a specific ref, return resolved SHA."""
    if dest.exists():
        print(f"[clone] {name} already exists at {dest}, fetching ref...")
        # Fetch and checkout the ref
        subprocess.run(
            ["git", "fetch", "--depth=1", "origin", f"refs/tags/{ref}:refs/tags/{ref}"],
            cwd=dest,
            capture_output=True,
        )
        proc = subprocess.run(
            ["git", "checkout", ref],
            cwd=dest,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[clone] WARNING: checkout {ref} failed: {proc.stderr[:200]}")
    else:
        print(f"[clone] Cloning {name}@{ref} to {dest}...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        clone_cmd = [
            "git", "clone",
            "--branch", ref,
            f"https://github.com/{name}.git",
            str(dest),
        ]
        if shallow:
            clone_cmd.insert(2, "--depth=1")
        
        proc = subprocess.run(clone_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[clone] ERROR cloning {name}: {proc.stderr[:500]}")
            return None
    
    # Get resolved SHA
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=dest,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        sha = proc.stdout.strip()
        print(f"[clone] {name}@{ref} → {sha[:12]}")
        return sha
    return None


def count_files(path: Path, extensions: List[str]) -> int:
    """Count files with given extensions."""
    count = 0
    for ext in extensions:
        count += len(list(path.rglob(f"*{ext}")))
    return count


def main():
    parser = argparse.ArgumentParser(description="Clone benchmark snapshot repos")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BENCH_DIR / "datasets" / "public_v1.json"),
        help="Path to dataset manifest JSON",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Clone only this repo (owner/name format)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory to clone repos into",
    )
    parser.add_argument(
        "--full-clone",
        action="store_true",
        help="Do full clone instead of shallow (for history analysis)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without cloning",
    )
    
    args = parser.parse_args()
    
    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    data_dir = Path(args.data_dir)
    repos = manifest.get("repos", [])
    
    # Filter to specific repo if requested
    if args.repo:
        repos = [r for r in repos if r["name"] == args.repo]
        if not repos:
            print(f"ERROR: Repo '{args.repo}' not found in manifest")
            sys.exit(1)
    
    print(f"[clone] Dataset: {manifest.get('id', 'unknown')}")
    print(f"[clone] Repos: {len(repos)}")
    print(f"[clone] Data dir: {data_dir}")
    
    if args.dry_run:
        for repo in repos:
            name = repo["name"]
            ref = repo["ref"]
            dest = data_dir / name.replace("/", "_")
            print(f"  Would clone: {name}@{ref} → {dest}")
        return
    
    # Clone each repo and update manifest with SHAs
    resolved: List[Dict[str, Any]] = []
    
    for repo in repos:
        name = repo["name"]
        ref = repo["ref"]
        dest = data_dir / name.replace("/", "_")
        
        sha = clone_repo(name, ref, dest, shallow=not args.full_clone)
        
        repo_info = dict(repo)
        repo_info["sha"] = sha
        repo_info["local_path"] = str(dest)
        
        # Count files
        ext_map = {
            "go": [".go"],
            "python": [".py"],
            "typescript": [".ts", ".tsx"],
            "javascript": [".js", ".jsx"],
            "yaml": [".yaml", ".yml"],
        }
        file_counts = {}
        for lang in repo.get("language", []):
            exts = ext_map.get(lang, [])
            if exts:
                file_counts[lang] = count_files(dest, exts)
        repo_info["file_counts"] = file_counts
        
        resolved.append(repo_info)
    
    # Write resolved manifest
    resolved_manifest = {
        "id": manifest.get("id", "unknown"),
        "version": manifest.get("version", "1.0.0"),
        "cloned_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "repos": resolved,
    }
    
    resolved_path = manifest_path.with_suffix(".resolved.json")
    with open(resolved_path, "w") as f:
        json.dump(resolved_manifest, f, indent=2)
    
    print(f"\n[clone] Resolved manifest written to: {resolved_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SNAPSHOT SUMMARY")
    print("=" * 60)
    for repo in resolved:
        name = repo["name"]
        sha = repo.get("sha", "?")[:12]
        counts = repo.get("file_counts", {})
        count_str = ", ".join(f"{k}:{v}" for k, v in counts.items())
        print(f"  {name}@{sha} → {count_str or 'no counts'}")


if __name__ == "__main__":
    main()
