"""pyMetaheuristic src — IPOP-CMA-ES Engine"""
from __future__ import annotations

from .protocol import CapabilityProfile
from ._restart_common import RestartCMAESBase


class IPOPCMAESEngine(RestartCMAESBase):
    """CMA-ES with increasing population-size restarts (IPOP-CMA-ES)."""

    algorithm_id = "ipop_cmaes"
    algorithm_name = "IPOP-CMA-ES"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2005.1554902",
        "title": "A restart CMA evolution strategy with increasing population size",
        "authors": "Anne Auger, Nikolaus Hansen",
        "year": 2005,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = {
        **RestartCMAESBase._DEFAULTS,
        "population_multiplier": 2.0,
    }
