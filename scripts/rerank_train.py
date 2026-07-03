#!/usr/bin/env python3
"""Backward-compatibility shim. See scripts/rerank_tools/train.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rerank_tools.train import *

if __name__ == "__main__":
    main()
