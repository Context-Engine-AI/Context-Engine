#!/usr/bin/env python3
"""Backward-compatibility shim. See scripts/rerank_tools/events.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rerank_tools.events import *
