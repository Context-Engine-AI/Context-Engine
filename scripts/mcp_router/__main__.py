#!/usr/bin/env python3
"""
Allow running as: python -m scripts.mcp_router "query"
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
