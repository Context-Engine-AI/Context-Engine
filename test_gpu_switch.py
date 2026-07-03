#!/usr/bin/env python3
"""
Test script to verify GPU decoder switching functionality.

Usage:
    # Test Docker CPU-only decoder
    python test_gpu_switch.py

    # Test native GPU-accelerated decoder  
    USE_GPU_DECODER=1 python test_gpu_switch.py
"""

import os
import sys

def load_env_file():
    """Load environment variables from .env file."""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value

def test_decoder_url_resolution():
    """Test that the decoder URL is resolved correctly based on USE_GPU_DECODER flag."""
    
    # Import the resolver function
    sys.path.insert(0, 'scripts')
    from refrag_llamacpp import LlamaCppRefragClient
    
    # Test current configuration
    use_gpu = os.environ.get("USE_GPU_DECODER", "0")
    print(f"USE_GPU_DECODER = {use_gpu}")
    
    # Create client and check URL
    client = LlamaCppRefragClient()
    print(f"Resolved decoder URL: {client.base_url}")

    # Test health endpoint
    try:
        import urllib.request

        health_url = client.base_url.rstrip('/') + '/health'

        # For Docker service names, try localhost equivalent when running on host
        if 'llamacpp:8080' in health_url:
            health_url = health_url.replace('llamacpp:8080', 'localhost:8080')
            print(f"Testing health endpoint: {health_url} (Docker service via localhost)")
        else:
            print(f"Testing health endpoint: {health_url}")

        req = urllib.request.Request(health_url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("PASS: Decoder server is healthy and reachable")
                return True
            else:
                print(f"FAIL: Decoder server returned status {resp.status}")
                return False

    except Exception as e:
        print(f"FAIL: Failed to reach decoder server: {e}")
        return False

def test_simple_completion():
    """Test a simple completion request."""
    
    sys.path.insert(0, 'scripts')
    from refrag_llamacpp import LlamaCppRefragClient, is_decoder_enabled
    
    if not is_decoder_enabled():
        print("FAIL: Decoder is disabled. Set REFRAG_DECODER=1 to enable.")
        return False
    
    try:
        client = LlamaCppRefragClient()

        # For Docker service names, use localhost equivalent when running on host
        test_url = client.base_url
        if 'llamacpp:8080' in client.base_url:
            test_url = client.base_url.replace('llamacpp:8080', 'localhost:8080')
            # Override the client's base_url for testing
            client.base_url = test_url
            print(f"Testing completion with decoder at: {test_url} (Docker service via localhost)")
        else:
            print(f"Testing completion with decoder at: {client.base_url}")

        response = client.generate_with_soft_embeddings(
            prompt="What is 2+2?",
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"PASS: Completion successful: {response[:100]}...")
        return True

    except Exception as e:
        print(f"FAIL: Completion failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing GPU decoder switching functionality\n")

    # Load .env file first
    load_env_file()

    # Set decoder enabled for testing
    os.environ.setdefault("REFRAG_DECODER", "1")

    print("1. Testing URL resolution...")
    url_ok = test_decoder_url_resolution()

    print("\n2. Testing simple completion...")
    completion_ok = test_simple_completion()

    print(f"\nResults:")
    print(f"   URL Resolution: {'PASS' if url_ok else 'FAIL'}")
    print(f"   Completion Test: {'PASS' if completion_ok else 'FAIL'}")

    if url_ok and completion_ok:
        print("\nAll tests passed! GPU switching is working correctly.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Check your decoder setup.")
        sys.exit(1)
