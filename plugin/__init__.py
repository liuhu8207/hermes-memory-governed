# -*- coding: utf-8 -*-
"""Hermes plugin package.

Contains :mod:`plugin.memory_governed`, the governed memory provider.

This ``__init__.py`` must exist so that ``plugin`` is a real package: the
``pyproject.toml`` entry point is ``plugin.memory_governed:register``, and a
wheel install without this file leaves ``plugin`` as an implicit namespace
directory that some installers do not package at all, making the entry point
unimportable.
"""

from __future__ import annotations
