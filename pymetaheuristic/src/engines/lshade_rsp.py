"""pyMetaheuristic src — LSHADE-RSP Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class LSHADERspEngine(ShadeVariantEngine):
    """L-SHADE with rank-based selective pressure mutation sampling."""

    algorithm_id = "lshade_rsp"
    algorithm_name = "LSHADE-RSP"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2018.8477957",
        "title": "LSHADE Algorithm with Rank-Based Selective Pressure Strategy for Solving CEC 2017 Benchmark Problems",
        "authors": "Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin",
        "year": 2018,
    }
    _DEFAULTS = dict(
        ShadeVariantEngine._DEFAULTS,
        population_size=100,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        pbest_max=0.20,
        pbest_min=0.05,
        rank_greediness=3.0,
        archive_rate=2.0,
    )
    _VARIANT_OVERRIDES = dict(rank_pressure=True)
