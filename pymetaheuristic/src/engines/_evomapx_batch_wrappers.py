"""Deprecated compatibility module.

Batch wrapper engines were retired because they made different algorithms share
synthetic family dynamics.  EvoMapX is now collected by passive probes attached
to the faithful engines.
"""
from __future__ import annotations

class DeprecatedEvoMapXBatchWrapper:  # pragma: no cover - compatibility guard only
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "EvoMapX batch wrappers are retired. Use faithful engines plus "
            "pymetaheuristic.src.evomapx_probe.EvoMapXProbe."
        )
