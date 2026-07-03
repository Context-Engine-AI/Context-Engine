#!/usr/bin/env python3
"""
SWE-bench Dataset Loader.

Downloads and processes SWE-bench instances from HuggingFace.
Extracts ground-truth file paths from patches.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# Cache directory for datasets
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "swe-bench"


@dataclass
class SWEInstance:
    """A single SWE-bench instance (issue + ground-truth files)."""
    
    instance_id: str
    repo: str  # e.g., "django/django"
    base_commit: str  # Commit SHA to checkout before applying patch
    problem_statement: str  # The GitHub issue text
    patch: str  # The ground-truth fix (unified diff)
    
    # Extracted from patch
    ground_truth_files: list[str]  # Files modified in the patch
    ground_truth_functions: list[str]  # Functions modified (if extractable)
    
    # Metadata
    created_at: str = ""
    version: str = ""
    
    @classmethod
    def from_hf_row(cls, row: dict) -> "SWEInstance":
        """Create instance from HuggingFace dataset row."""
        patch = row.get("patch", "")
        files = extract_files_from_patch(patch)
        functions = extract_functions_from_patch(patch)
        
        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            patch=patch,
            ground_truth_files=files,
            ground_truth_functions=functions,
            created_at=row.get("created_at", ""),
            version=row.get("version", ""),
        )


def extract_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a unified diff patch.

    Uses only the NEW file path (b/...) from the patch header.
    Filters out:
    - /dev/null (file deletions)
    - Old paths from renames (--- a/... lines show old path)

    For a patch that modifies existing files:
        diff --git a/foo.py b/foo.py
        +++ b/foo.py  <- this is the target file

    For a rename:
        diff --git a/old.py b/new.py
        --- a/old.py  <- old path (excluded)
        +++ b/new.py  <- new path (included)

    For a deletion:
        diff --git a/foo.py b/dev/null
        +++ /dev/null  <- excluded (file deleted)
    """
    files = set()

    # Match "diff --git a/... b/..." - use the 'b' path (new/destination)
    for match in re.finditer(r"diff --git a/(.+?) b/(.+?)(?:\n|$)", patch):
        new_path = match.group(2)
        # Skip /dev/null (file deletions)
        if new_path != "/dev/null" and not new_path.endswith("/dev/null"):
            files.add(new_path)

    # Also parse "+++ b/..." lines for robustness (actual new file path)
    # This is more reliable than "--- a/..." which shows the OLD path
    for match in re.finditer(r"^\+\+\+ b/(.+?)$", patch, re.MULTILINE):
        new_path = match.group(1)
        # Skip /dev/null
        if new_path != "/dev/null" and new_path != "dev/null":
            files.add(new_path)

    return sorted(files)


def extract_functions_from_patch(patch: str) -> list[str]:
    """Extract function/method names modified in the patch.
    
    Looks for @@ hunk headers with function context:
    @@ -10,5 +10,6 @@ def some_function(...)
    """
    functions = set()
    
    # Match @@ ... @@ optional_function_context
    for match in re.finditer(r"^@@[^@]+@@\s*(?:def|class|async def)\s+(\w+)", patch, re.MULTILINE):
        functions.add(match.group(1))
    
    return sorted(functions)


def load_swe_bench(
    subset: str = "lite",
    cache_dir: Optional[Path] = None,
) -> list[SWEInstance]:
    """Load SWE-bench dataset from HuggingFace.
    
    Args:
        subset: "lite" (300 instances) or "full" (2,294 instances)
        cache_dir: Where to cache the dataset
        
    Returns:
        List of SWEInstance objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package required: pip install datasets"
        )
    
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset names on HuggingFace
    if subset == "lite":
        dataset_name = "princeton-nlp/SWE-bench_Lite"
    elif subset == "full":
        dataset_name = "princeton-nlp/SWE-bench"
    else:
        raise ValueError(f"Unknown subset: {subset}. Use 'lite' or 'full'.")
    
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="test", cache_dir=str(cache_dir))
    
    instances = []
    for row in ds:
        try:
            instance = SWEInstance.from_hf_row(row)
            if instance.ground_truth_files:  # Skip if no files extracted
                instances.append(instance)
        except Exception as e:
            print(f"Warning: Failed to parse instance {row.get('instance_id', '?')}: {e}")
    
    print(f"Loaded {len(instances)} instances from {subset} subset")
    return instances


def filter_by_repo(
    instances: list[SWEInstance],
    repos: list[str],
) -> list[SWEInstance]:
    """Filter instances to specific repositories."""
    repo_set = set(repos)
    return [i for i in instances if i.repo in repo_set]


def get_unique_repos(instances: list[SWEInstance]) -> list[str]:
    """Get list of unique repositories in the dataset."""
    return sorted(set(i.repo for i in instances))

