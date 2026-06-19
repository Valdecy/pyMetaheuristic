"""pyMetaheuristic src — jSO Differential Evolution Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class JSODEEngine(ShadeVariantEngine):
    """jSO: improved iL-SHADE with weighted current-to-pbest mutation."""

    algorithm_id = "jso_de"
    algorithm_name = "jSO Differential Evolution"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2017.7969362",
        "title": "Single Objective Real-Parameter Optimization: Algorithm jSO",
        "authors": "Janez Brest, Mirjam Sepesy Maucec, Borko Boskovic",
        "year": 2017,
    }
    _DEFAULTS = dict(
        ShadeVariantEngine._DEFAULTS,
        population_size=100,
        min_population_size=4,
        hist_mem_size=6,
        memory_f_init=0.5,
        memory_cr_init=0.8,
        pbest_factor=0.11,
        archive_rate=1.4,
    )
    _VARIANT_OVERRIDES = dict(weighted_pbest=True)
