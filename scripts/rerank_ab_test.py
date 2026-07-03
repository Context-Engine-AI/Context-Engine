#!/usr/bin/env python3
"""Backward-compatibility shim. See scripts/rerank_tools/ab_test.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rerank_tools.ab_test import *

if __name__ == "__main__":
    simulate_ab_test(n_sessions=100, n_queries_per_session=5)
