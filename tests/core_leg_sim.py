# -*- coding: utf-8 -*-
"""Simulate the CI `core` job locally: no vector backend installed.

The CI matrix has a `core` leg that installs only `.[dev]` — deliberately, to
exercise the degradation path — while `vector` installs the lancedb stack. On a
dev machine every dependency is present, so the core leg is the one configuration
nobody can reproduce by accident. It has been red for days.

This plugin makes the dev machine look like that leg by making the optional
imports fail, so the leg can be reproduced locally before pushing:

    PYTHONPATH=tests python -m pytest tests/ -q -p core_leg_sim

(It is not named test_* on purpose, so pytest does not collect it.)

Raising from `find_spec` (rather than returning None) matters: returning None
means "not mine, keep looking", so the real module would still be imported.
"""
import importlib.abc
import sys

#: What `pip install -e .[dev]` does not bring in.
BLOCKED = ("lancedb", "pyarrow", "fastembed", "sentence_transformers", "pandas")


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(f"[core-sim] '{fullname}' is not installed in the core leg")
        return None


if not any(isinstance(f, _Blocker) for f in sys.meta_path):
    sys.meta_path.insert(0, _Blocker())
