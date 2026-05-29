"""Deprecated compatibility module.

Generic guessed EvoMapX engines were removed from the active architecture.
Faithful engine implementations should be observed through
``pymetaheuristic.src.evomapx_probe.EvoMapXProbe`` instead of inheriting a
synthetic optimizer.
"""
from __future__ import annotations

class GuessedEvoMapXEngine:  # pragma: no cover - compatibility guard only
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "GuessedEvoMapXEngine has been retired. Use faithful engines with "
            "the passive EvoMapXProbe instrumentation layer."
        )
