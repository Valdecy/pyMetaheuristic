"""pyMetaheuristic src — BIPOP-CMA-ES Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._restart_common import RestartCMAESBase


class BIPOPCMAESEngine(RestartCMAESBase):
    """BI-population CMA-ES with alternating large and small restart regimes."""

    algorithm_id = "bipop_cmaes"
    algorithm_name = "BIPOP-CMA-ES"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1145/1570256.1570333",
        "title": "Benchmarking a BI-population CMA-ES on the BBOB-2009 function testbed",
        "authors": "Nikolaus Hansen",
        "year": 2009,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = {
        **RestartCMAESBase._DEFAULTS,
        "population_multiplier": 2.0,
        "small_population_probability": 0.5,
    }

    def _next_lambda(self, restart_index: int, payload: dict | None = None) -> int:
        if restart_index <= 0:
            return self._base_lambda
        p_small = float(self._params.get("small_population_probability", 0.5))
        p_small = min(1.0, max(0.0, p_small))
        large_count = int((restart_index + 1) // 2)
        large_lambda = int(round(self._base_lambda * (self._population_multiplier ** max(1, large_count))))
        if self._rng.random() < p_small:
            upper = max(self._base_lambda, large_lambda // 2)
            return int(self._rng.integers(max(4, self._base_lambda // 2), upper + 1))
        return large_lambda
