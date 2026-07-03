import importlib
import os


def _import_runner_with_env_restored():
    before = dict(os.environ)
    try:
        return importlib.import_module("scripts.benchmarks.swe.runner")
    finally:
        os.environ.clear()
        os.environ.update(before)


def test_normalize_result_path_handles_dunder_repo():
    runner = _import_runner_with_env_restored()
    repo_path = "/tmp/org__repo__special"
    repo_name = "org__repo__special"
    path = "/work/org/repo__special/src/main.py"

    assert runner._normalize_result_path(path, repo_path, repo_name) == "src/main.py"


def test_normalize_result_path_org_repo_basic():
    runner = _import_runner_with_env_restored()
    repo_path = "/tmp/django__django"
    repo_name = "django__django"
    path = "/work/django/django/app/models.py"

    assert runner._normalize_result_path(path, repo_path, repo_name) == "app/models.py"
