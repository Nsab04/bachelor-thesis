#!/usr/bin/env python3
"""
Compatibility launcher for the clumPyLens workflow.

The canonical implementation now lives in apps/clumPyLens/clumPyLens.py so the
workflow can keep its code, inputs, and examples together in one app folder.
"""

from __future__ import annotations

from clumPyLens.clumPyLens import main


if __name__ == "__main__":
    main()
