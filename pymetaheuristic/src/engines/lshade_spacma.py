"""pyMetaheuristic src — LSHADE-SPACMA Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class LSHADESpacmaEngine(ShadeVariantEngine):
    """L-SHADE-SPACMA: semi-parameter-adaptive L-SHADE hybridized with CMA-style sampling."""

    algorithm_id = "lshade_spacma"
    algorithm_name = "LSHADE-SPACMA"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2017.7969307",
        "title": "LSHADE with Semi-Parameter Adaptation Hybrid with CMA-ES for Solving CEC 2017 Benchmark Problems",
        "authors": "Ali W. Mohamed, Anas A. Hadi, Anas M. Fattouh, Kamal M. Jambi",
        "year": 2017,
    }
    _DEFAULTS = dict(
        ShadeVariantEngine._DEFAULTS,
        population_size=100,
        min_population_size=4,
        hist_mem_size=6,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        pbest_factor=0.11,
        archive_rate=2.0,
    )
    _VARIANT_OVERRIDES = dict(spacma=True)
