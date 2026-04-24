"""Compatibility wrapper for local repository imports."""

try:  # package import
    from .pymetaheuristic import *  # noqa: F401,F403
except ImportError:  # direct-path import during local test collection
    from pymetaheuristic import *  # noqa: F401,F403
