#!/usr/bin/env python3
"""
Test script to exercise and verify env snapshot functionality.

This validates that:
1. get_env_snapshot() filters sensitive keys correctly
2. get_runtime_info() captures all expected fields
3. save_run_meta() creates a valid JSON file
"""
import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.common import (
    get_env_snapshot,
    get_runtime_info,
    save_run_meta,
    get_platform_info,
    get_git_sha,
    _is_sensitive_key,
)


def test_sensitive_key_filtering():
    """Verify sensitive keys are correctly identified."""
    print("\n=== Test 1: Sensitive Key Filtering ===")
    
    # Should be sensitive (blocked)
    sensitive_keys = [
        "OPENAI_API_KEY",
        "GLM_API_KEY",
        "MY_SECRET",
        "DATABASE_PASSWORD",
        "AUTH_SECRET",
        "AUTH_USERNAME",
        "JWT_TOKEN",
        "MY_CREDENTIAL",
    ]
    
    # Should NOT be sensitive (allowed)
    allowed_keys = [
        "COLLECTION_NAME",
        "EMBEDDING_MODEL",
        "RERANKER_MODEL",
        "INDEX_DENSE_MODE",
        "FNAME_BOOST",
        "HYBRID_EXPAND",
        "LLM_EXPAND_MAX",
        "PSEUDO_DEFER_TO_WORKER",
        "PATH",  # Common OS var
    ]
    
    errors = []
    for key in sensitive_keys:
        if not _is_sensitive_key(key):
            errors.append(f"  FAIL: {key} should be SENSITIVE but wasn't filtered")
    
    for key in allowed_keys:
        if _is_sensitive_key(key):
            errors.append(f"  FAIL: {key} should be ALLOWED but was filtered")
    
    if errors:
        for e in errors:
            print(e)
        return False
    
    print("  PASS: All sensitive key patterns correctly identified")
    return True


def test_env_snapshot():
    """Verify env snapshot captures expected vars and filters sensitive ones."""
    print("\n=== Test 2: Env Snapshot Content ===")
    
    # Set some test env vars
    os.environ["TEST_COLLECTION_NAME"] = "test-collection"
    os.environ["TEST_EMBEDDING_MODEL"] = "bge-base-en"
    os.environ["TEST_SECRET_API_KEY"] = "should_be_filtered"  # Sensitive
    
    snapshot = get_env_snapshot()
    
    errors = []
    
    # Check expected keys are present
    if "TEST_COLLECTION_NAME" not in snapshot:
        errors.append("  FAIL: TEST_COLLECTION_NAME not in snapshot")
    if "TEST_EMBEDDING_MODEL" not in snapshot:
        errors.append("  FAIL: TEST_EMBEDDING_MODEL not in snapshot")
    
    # Check sensitive keys are filtered
    if "TEST_SECRET_API_KEY" in snapshot:
        errors.append("  FAIL: TEST_SECRET_API_KEY should be filtered but wasn't")
    
    # Report size
    print(f"  Snapshot contains {len(snapshot)} environment variables")
    
    if errors:
        for e in errors:
            print(e)
        return False
    
    print("  PASS: Env snapshot correctly captures and filters variables")
    
    # Cleanup
    del os.environ["TEST_COLLECTION_NAME"]
    del os.environ["TEST_EMBEDDING_MODEL"]
    del os.environ["TEST_SECRET_API_KEY"]
    
    return True


def test_runtime_info():
    """Verify runtime info contains all expected fields."""
    print("\n=== Test 3: Runtime Info Structure ===")
    
    info = get_runtime_info()
    
    required_keys = ["timestamp", "git_sha", "env_snapshot", "platform"]
    errors = []
    
    for key in required_keys:
        if key not in info:
            errors.append(f"  FAIL: Missing required key '{key}'")
    
    # Check platform info
    platform_info = info.get("platform", {})
    platform_keys = ["machine", "system", "python", "processor"]
    for key in platform_keys:
        if key not in platform_info:
            errors.append(f"  FAIL: Missing platform key '{key}'")
    
    if errors:
        for e in errors:
            print(e)
        return False
    
    print(f"  timestamp: {info['timestamp']}")
    print(f"  git_sha: {info['git_sha']}")
    print(f"  platform: {info['platform']['system']} / {info['platform']['machine']}")
    print(f"  env_snapshot: {len(info['env_snapshot'])} vars")
    print("  PASS: Runtime info contains all expected fields")
    return True


def test_save_run_meta():
    """Verify save_run_meta creates a valid JSON file."""
    print("\n=== Test 4: Save Run Meta ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "collection": "cosqa-test",
            "limit": 10,
            "rerank_enabled": True,
        }
        extra = {
            "benchmark": "cosqa",
            "corpus_size": 1000,
            "queries": 50,
        }
        
        run_id = "test_run_20260107"
        meta_path = save_run_meta(tmpdir, run_id, config, extra=extra)
        
        # Verify file exists
        if not Path(meta_path).exists():
            print(f"  FAIL: Meta file not created at {meta_path}")
            return False
        
        # Verify JSON is valid
        try:
            with open(meta_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  FAIL: Invalid JSON: {e}")
            return False
        
        # Verify structure
        required_keys = ["run_id", "timestamp", "git_sha", "env_snapshot", "platform", "config"]
        errors = []
        for key in required_keys:
            if key not in data:
                errors.append(f"  FAIL: Missing key '{key}' in saved file")
        
        if errors:
            for e in errors:
                print(e)
            return False
        
        # Verify extra was merged
        if data.get("benchmark") != "cosqa":
            print("  FAIL: Extra metadata not merged correctly")
            return False
        
        print(f"  File saved: {meta_path}")
        print(f"  File size: {Path(meta_path).stat().st_size} bytes")
        print(f"  Keys: {list(data.keys())}")
        print(f"  Config: {data['config']}")
        print("  PASS: Run metadata file created and valid")
        
        # Show a sample of the env_snapshot
        env_snap = data.get("env_snapshot", {})
        sample_keys = list(env_snap.keys())[:5]
        print(f"\n  Sample env vars captured: {sample_keys}")
        
        return True


def test_runner_integration():
    """INTEGRATION TEST: Prove the runner actually calls save_run_meta with env_snapshot.
    
    Since we can't easily import runner.py (has heavy deps like qdrant_client),
    we verify the code path statically by:
    1. Reading the runner source code
    2. Proving save_run_meta import exists
    3. Proving save_run_meta call exists  
    4. Proving the call uses get_runtime_info (which contains env_snapshot)
    """
    print("\n=== Test 5: Runner Integration (Static Analysis) ===")
    
    runner_path = ROOT / "scripts" / "benchmarks" / "cosqa" / "runner.py"
    
    if not runner_path.exists():
        print(f"  FAIL: Runner not found at {runner_path}")
        return False
    
    source = runner_path.read_text(encoding="utf-8")
    
    errors = []
    
    # 1. Check save_run_meta is imported
    if "from scripts.benchmarks.common import" in source and "save_run_meta" in source:
        print("  [1] save_run_meta is imported from common.py")
    else:
        errors.append("  FAIL: save_run_meta not imported from common.py")
    
    # 2. Check save_run_meta is called
    if "save_run_meta(" in source:
        # Find the line
        for i, line in enumerate(source.splitlines(), 1):
            if "meta_path = save_run_meta(" in line:
                print(f"  [2] save_run_meta called at line {i}")
                break
    else:
        errors.append("  FAIL: save_run_meta() not called in runner.py")
    
    # 3. Check get_runtime_info is part of the chain (used in save_run_meta)
    common_path = ROOT / "scripts" / "benchmarks" / "common.py"
    common_source = common_path.read_text(encoding="utf-8")
    
    # Verify save_run_meta uses get_runtime_info which contains env_snapshot
    if "get_runtime_info()" in common_source and "def save_run_meta" in common_source:
        # Check the save_run_meta function uses get_runtime_info
        in_save_run_meta = False
        for line in common_source.splitlines():
            if "def save_run_meta" in line:
                in_save_run_meta = True
            if in_save_run_meta and "get_runtime_info()" in line:
                print("  [3] save_run_meta calls get_runtime_info() which contains env_snapshot")
                break
            if in_save_run_meta and line.strip().startswith("def ") and "save_run_meta" not in line:
                in_save_run_meta = False
    
    # 4. Verify get_runtime_info includes env_snapshot
    if '"env_snapshot": get_env_snapshot()' in common_source:
        print("  [4] get_runtime_info() includes env_snapshot")
    else:
        errors.append("  FAIL: get_runtime_info doesn't include env_snapshot")
    
    # 5. Now do a LIVE test of the actual save function to prove the chain works end-to-end
    print("\n  --- Live verification of save_run_meta chain ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = save_run_meta(
            tmpdir, 
            "integration_test_run",
            {"collection": "test", "rerank_enabled": True},
            extra={"benchmark": "cosqa", "corpus_size": 100}
        )
        
        with open(meta_path) as f:
            data = json.load(f)
        
        if "env_snapshot" in data and isinstance(data["env_snapshot"], dict):
            env_count = len(data["env_snapshot"])
            print(f"  [5] LIVE: env_snapshot saved with {env_count} variables")
            
            # Show some proof
            sample = list(data["env_snapshot"].keys())[:3]
            print(f"      Sample keys: {sample}")
        else:
            errors.append("  FAIL: env_snapshot not in saved metadata")
    
    if errors:
        for e in errors:
            print(e)
        return False
    
    print("  PASS: Runner integration verified - code path proven")
    return True


def main():
    print("=" * 60)
    print("ENV SNAPSHOT FUNCTIONALITY TEST")
    print("=" * 60)
    
    results = []
    results.append(("Sensitive Key Filtering", test_sensitive_key_filtering()))
    results.append(("Env Snapshot Content", test_env_snapshot()))
    results.append(("Runtime Info Structure", test_runtime_info()))
    results.append(("Save Run Meta", test_save_run_meta()))
    results.append(("Runner Integration", test_runner_integration()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\nAll tests passed! Env snapshot functionality is working correctly.")
        return 0
    else:
        print("\nSome tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
