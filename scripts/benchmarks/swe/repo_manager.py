#!/usr/bin/env python3
"""
Repository Manager for SWE-bench Evaluation.

Handles cloning, caching, and checking out specific commits
for SWE-bench repository evaluation.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "swe-bench" / "repos"

# Repository URLs
REPO_URLS = {
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "django/django": "https://github.com/django/django.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
}

# Pattern for valid repo names (org/repo format)
_REPO_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def _decode_output(data: Optional[bytes]) -> str:
    """Safely decode subprocess output."""
    if data is None:
        return ""
    try:
        return data.decode("utf-8", errors="replace").strip()
    except Exception:
        return "<decode error>"


def _safe_rmtree(path: Path) -> None:
    """Safely remove a path (file or directory)."""
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception as e:
        logger.warning("Failed to remove %s: %s", path, e)


def _sanitize_repo_name(repo: str) -> str:
    """Sanitize repository name for safe filesystem use."""
    # Normalize backslashes to forward slashes
    repo = repo.replace("\\", "/")
    # Strip leading/trailing whitespace and slashes
    repo = repo.strip().strip("/")
    return repo


class RepoManager:
    """Manages repository clones for SWE-bench evaluation."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._current_checkouts: dict[str, str] = {}  # repo -> current commit

    def get_repo_path(self, repo: str) -> Path:
        """Get the local path for a repository."""
        # Sanitize and convert "django/django" to "django__django"
        repo = _sanitize_repo_name(repo)
        if not _REPO_NAME_PATTERN.match(repo):
            raise ValueError(f"Invalid repository name format: {repo}")
        safe_name = repo.replace("/", "__")
        return self.cache_dir / safe_name

    def ensure_repo(self, repo: str, max_retries: int = 3) -> Path:
        """Ensure repository is cloned locally.

        Returns the local path to the repository.

        Args:
            repo: Repository identifier (e.g., "django/django")
            max_retries: Number of retries for network operations
        """
        repo = _sanitize_repo_name(repo)
        repo_path = self.get_repo_path(repo)

        if repo_path.exists():
            # Check if it's a directory first (could be a file)
            if not repo_path.is_dir():
                logger.warning("Removing non-directory at %s", repo_path)
                _safe_rmtree(repo_path)
            elif (repo_path / ".git").exists():
                # Already cloned, fetch latest
                logger.info("Updating %s...", repo)
                try:
                    subprocess.run(
                        ["git", "fetch", "--all", "--quiet"],
                        cwd=repo_path,
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        "Failed to fetch %s (may be offline): %s",
                        repo, _decode_output(e.stderr)
                    )
                return repo_path
            else:
                # Directory exists but isn't a git repo - remove it
                logger.warning("Removing non-git directory %s", repo_path)
                _safe_rmtree(repo_path)

        # Clone fresh with retry logic
        url = REPO_URLS.get(repo)
        if not url:
            # Try to construct GitHub URL
            url = f"https://github.com/{repo}.git"

        logger.info("Cloning %s...", repo)
        repo_path.parent.mkdir(parents=True, exist_ok=True)

        last_error = None
        for attempt in range(max_retries):
            try:
                subprocess.run(
                    ["git", "clone", "--quiet", url, str(repo_path)],
                    check=True,
                    capture_output=True,
                )
                return repo_path
            except subprocess.CalledProcessError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        "Clone failed (%s), retrying in %ds...",
                        _decode_output(e.stderr), wait_time
                    )
                    time.sleep(wait_time)
                    # Clean up partial clone if any
                    _safe_rmtree(repo_path)

        raise RuntimeError(
            f"Failed to clone {repo} after {max_retries} attempts: "
            f"{_decode_output(last_error.stderr) if last_error else 'unknown error'}"
        )
    
    def checkout_commit(self, repo: str, commit: str) -> Path:
        """Checkout a specific commit in the repository.

        Returns the repo path after checkout.
        """
        repo = _sanitize_repo_name(repo)
        repo_path = self.ensure_repo(repo)

        # Check if already at this commit
        if self._current_checkouts.get(repo) == commit:
            return repo_path

        # Clean and checkout
        logger.info("Checking out %s @ %s", repo, commit[:12])
        try:
            subprocess.run(
                ["git", "checkout", "--force", "--quiet", commit],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "clean", "-fdx", "--quiet"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            self._current_checkouts[repo] = commit
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to checkout {commit} in {repo}: {_decode_output(e.stderr)}"
            ) from e

        return repo_path

    def get_current_commit(self, repo: str) -> Optional[str]:
        """Get the currently checked out commit for a repo."""
        repo = _sanitize_repo_name(repo)
        return self._current_checkouts.get(repo)

    def list_cached_repos(self) -> list[str]:
        """List all cached repositories."""
        repos = []
        if not self.cache_dir.exists():
            return repos
        for path in self.cache_dir.iterdir():
            if path.is_dir() and (path / ".git").exists():
                # Convert "django__django" back to "django/django"
                repos.append(path.name.replace("__", "/"))
        return sorted(repos)

    def clear_cache(self, repo: Optional[str] = None):
        """Clear cached repositories.

        If repo is specified, only clear that repo.
        Otherwise, clear all cached repos.
        """
        if repo:
            repo = _sanitize_repo_name(repo)
            repo_path = self.get_repo_path(repo)
            _safe_rmtree(repo_path)
            self._current_checkouts.pop(repo, None)
        else:
            _safe_rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._current_checkouts.clear()

