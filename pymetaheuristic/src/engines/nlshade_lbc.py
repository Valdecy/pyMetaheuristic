"""pyMetaheuristic src — NL-SHADE-LBC Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class NLSHADELbcEngine(ShadeVariantEngine):
    """NL-SHADE-LBC with linear parameter adaptation bias change."""

    algorithm_id = "nlshade_lbc"
    algorithm_name = "NL-SHADE-LBC"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC55065.2022.9870295",
        "title": "NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization",
        "authors": "Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin",
        "year": 2022,
    }
    _DEFAULTS = dict(
        ShadeVariantEngine._DEFAULTS,
        population_size=120,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.5,
        memory_cr_init=0.8,
        pbest_max=0.25,
        pbest_min=0.05,
        rank_greediness=3.0,
        archive_rate=2.0,
        archive_use_probability=0.5,
        nlpsr_exponent=1.5,
    )
    _VARIANT_OVERRIDES = dict(rank_pressure=True, nonlinear_reduction=True, nlshade=True, lbc=True, resample_bounds=True)
