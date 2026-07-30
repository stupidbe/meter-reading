#!/usr/bin/env python3
"""Entry point: run pipeline on mounted data directory."""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import main

if __name__ == "__main__":
    main()
