#!/usr/bin/env python3
"""
Kubernetes Repository Uploader

Upload local repositories to a Kubernetes cluster running Context Engine
and trigger indexing via the MCP Indexer API.

Usage:
    python scripts/k8s_uploader.py /path/to/repo --namespace context-engine
    python scripts/k8s_uploader.py /path/to/repo --collection my-project --recreate
    python scripts/k8s_uploader.py /path/to/repo --pod mcp-indexer-abc123 --skip-index
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List


def run_command(cmd: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if capture and e.stderr:
            print(f"stderr: {e.stderr}")
        raise


def get_indexer_pod(namespace: str, pod_name: Optional[str] = None) -> str:
    """Get the name of an MCP indexer pod."""
    if pod_name:
        return pod_name
    
    # Find a running mcp-indexer pod
    result = run_command([
        "kubectl", "get", "pods",
        "-n", namespace,
        "-l", "component=mcp-indexer",
        "-o", "jsonpath={.items[0].metadata.name}"
    ])
    
    pod = result.stdout.strip()
    if not pod:
        raise RuntimeError(f"No mcp-indexer pod found in namespace {namespace}")
    
    print(f"Using pod: {pod}")
    return pod


def create_tar_archive(source_path: Path, exclude_patterns: Optional[List[str]] = None) -> Path:
    """Create a tar.gz archive of the source directory."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    
    if not source_path.is_dir():
        raise ValueError(f"Source path must be a directory: {source_path}")
    
    # Default exclusions
    if exclude_patterns is None:
        exclude_patterns = [
            ".git",
            ".codebase",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            "*.log"
        ]
    
    # Create temporary tar file
    temp_dir = Path(tempfile.mkdtemp())
    tar_path = temp_dir / f"{source_path.name}.tar.gz"
    
    print(f"Creating archive: {tar_path}")
    print(f"Source: {source_path}")
    print(f"Excluding: {', '.join(exclude_patterns)}")
    
    def should_exclude(path: Path) -> bool:
        """Check if a path should be excluded."""
        for pattern in exclude_patterns:
            if pattern.startswith("*."):
                # File extension pattern
                if path.suffix == pattern[1:]:
                    return True
            elif path.name == pattern:
                return True
            elif pattern in str(path):
                return True
        return False
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in source_path.rglob("*"):
            if should_exclude(item):
                continue
            
            arcname = item.relative_to(source_path.parent)
            try:
                tar.add(item, arcname=arcname)
            except Exception as e:
                print(f"Warning: Could not add {item}: {e}")
    
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"Archive created: {tar_path} ({size_mb:.2f} MB)")
    
    return tar_path


def upload_to_pod(tar_path: Path, namespace: str, pod_name: str, target_dir: str = "/work") -> str:
    """Upload tar archive to a pod and extract it."""
    repo_name = tar_path.stem.replace(".tar", "")
    target_path = f"{target_dir}/{repo_name}"
    
    print(f"Uploading to pod {pod_name}:{target_path}")
    
    # Create target directory in pod
    run_command([
        "kubectl", "exec", "-n", namespace, pod_name, "--",
        "mkdir", "-p", target_path
    ])
    
    # Copy tar file to pod
    temp_tar = f"/tmp/{tar_path.name}"
    run_command([
        "kubectl", "cp", str(tar_path),
        f"{namespace}/{pod_name}:{temp_tar}"
    ])
    
    # Extract in pod
    print(f"Extracting archive in pod...")
    run_command([
        "kubectl", "exec", "-n", namespace, pod_name, "--",
        "tar", "-xzf", temp_tar, "-C", target_dir
    ])
    
    # Clean up temp tar in pod
    run_command([
        "kubectl", "exec", "-n", namespace, pod_name, "--",
        "rm", temp_tar
    ], check=False)
    
    print(f"Upload complete: {target_path}")
    return target_path


def trigger_indexing(
    namespace: str,
    pod_name: str,
    repo_path: str,
    collection: Optional[str] = None,
    recreate: bool = False
) -> Dict[str, Any]:
    """Trigger indexing via the MCP indexer server."""
    print(f"Triggering indexing for {repo_path}")

    # Build Python command to call qdrant_index via MCP server
    # Use qdrant_index with subdir parameter to index specific repo
    python_cmd = f"""
import sys
sys.path.insert(0, '/app')
from scripts.mcp_indexer_server import qdrant_index
import asyncio
import json

# Extract subdir from repo_path (e.g., /work/test-repo -> test-repo)
repo_path = '{repo_path}'
if repo_path.startswith('/work/'):
    subdir = repo_path[6:]  # Remove '/work/' prefix
else:
    subdir = repo_path

# Call indexing
result = asyncio.run(qdrant_index(
    subdir=subdir,
    recreate={str(recreate)},
    collection={repr(collection) if collection else 'None'}
))
print(json.dumps(result, indent=2))
"""
    
    # Execute in pod
    result = run_command([
        "kubectl", "exec", "-n", namespace, pod_name, "--",
        "python", "-c", python_cmd
    ], check=False)

    # Parse result
    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode

    # Extract return code from output
    for line in stdout.split("\n"):
        if line.startswith("RETURNCODE:"):
            try:
                returncode = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    return {
        "ok": returncode == 0,
        "code": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "collection": collection or "codebase",
        "repo_path": repo_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Upload repositories to Kubernetes Context Engine cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload and index a repository
  python scripts/k8s_uploader.py /path/to/my-repo

  # Upload to specific namespace and collection
  python scripts/k8s_uploader.py /path/to/my-repo --namespace prod --collection my-project

  # Upload and recreate collection (drops existing data)
  python scripts/k8s_uploader.py /path/to/my-repo --recreate

  # Upload only (skip indexing)
  python scripts/k8s_uploader.py /path/to/my-repo --skip-index

  # Upload to specific pod
  python scripts/k8s_uploader.py /path/to/my-repo --pod mcp-indexer-abc123
        """
    )
    
    parser.add_argument("source", type=str, help="Path to repository to upload")
    parser.add_argument("--namespace", "-n", default="context-engine", help="Kubernetes namespace (default: context-engine)")
    parser.add_argument("--pod", "-p", help="Specific pod name (default: auto-detect mcp-indexer pod)")
    parser.add_argument("--collection", "-c", help="Qdrant collection name (default: codebase)")
    parser.add_argument("--target-dir", default="/work", help="Target directory in pod (default: /work)")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection (drops existing data)")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing after upload")
    parser.add_argument("--exclude", action="append", help="Additional exclude patterns")
    parser.add_argument("--keep-archive", action="store_true", help="Keep temporary archive file")
    
    args = parser.parse_args()
    
    source_path = Path(args.source).resolve()
    
    try:
        # Get target pod
        pod_name = get_indexer_pod(args.namespace, args.pod)
        
        # Create archive
        tar_path = create_tar_archive(source_path, args.exclude)
        
        # Upload to pod
        repo_path = upload_to_pod(tar_path, args.namespace, pod_name, args.target_dir)
        
        # Trigger indexing
        if not args.skip_index:
            result = trigger_indexing(
                args.namespace,
                pod_name,
                repo_path,
                args.collection,
                args.recreate
            )
            
            print("\n" + "="*60)
            print("INDEXING RESULT:")
            print("="*60)
            print(json.dumps(result, indent=2))

            if result.get("ok") and result.get("code") == 0:
                print("\n[SUCCESS] Upload and indexing completed successfully!")
            else:
                print("\n[WARNING] Indexing completed with warnings or errors")
                sys.exit(1)
        else:
            print("\n[SUCCESS] Upload completed successfully (indexing skipped)")
        
        # Clean up
        if not args.keep_archive:
            tar_path.unlink()
            tar_path.parent.rmdir()
            print(f"Cleaned up temporary archive")
        else:
            print(f"Archive kept at: {tar_path}")
    
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

