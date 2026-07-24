"""
__main__.py — lets this package be run directly as `python -m irt`.

Deliberately two lines of logic: argument parsing, repository
construction, pipeline execution, and reporting all live in cli.py
(see its module docstring) so that logic stays testable by calling
`irt.cli.main(argv)` directly, without going through a subprocess.
This file exists only to satisfy Python's `python -m irt` convention.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
