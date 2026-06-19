"""pyMetaheuristic src — iLSHADE-RSP Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class ILSHADERspEngine(ShadeVariantEngine):
    """iLSHADE-RSP with Cauchy perturbation of target vectors."""

    algorithm_id = "ilshade_rsp"
    algorithm_name = "iLSHADE-RSP"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.48550/arXiv.2006.02591",
        "title": "An Improved LSHADE-RSP Algorithm with the Cauchy Perturbation: iLSHADE-RSP",
        "authors": "Tae Jong Choi and Chang Wook Ahn",
        "year": 2020,
    }
    _DEFAULTS = dict(
        ShadeVariantEngine._DEFAULTS,
        population_size=100,
        min_population_size=4,
        hist_mem_size=5,
        pbest_max=0.20,
        pbest_min=0.05,
        rank_greediness=3.0,
        archive_rate=2.0,
        jumping_rate=0.2,
        cauchy_scale=0.1,
    )
    _VARIANT_OVERRIDES = dict(rank_pressure=True, cauchy_perturb=True)
