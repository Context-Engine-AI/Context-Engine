#!/usr/bin/env python3
"""Backward-compatibility shim. See scripts/rerank_tools/local.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rerank_tools.local import *

if __name__ == "__main__":
    main()
