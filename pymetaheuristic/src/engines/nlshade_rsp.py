"""pyMetaheuristic src — NL-SHADE-RSP Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class NLSHADERspEngine(ShadeVariantEngine):
    """NL-SHADE-RSP with non-linear population reduction, archive use, and selective pressure."""

    algorithm_id = "nlshade_rsp"
    algorithm_name = "NL-SHADE-RSP"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC45853.2021.9504959",
        "title": "NL-SHADE-RSP Algorithm with Adaptive Archive and Selective Pressure for CEC 2021 Numerical Optimization",
        "authors": "Vladimir Stanovov, Shakhnaz Akhmedova, Eugene Semenkin",
        "year": 2021,
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
    _VARIANT_OVERRIDES = dict(rank_pressure=True, nonlinear_reduction=True, nlshade=True)
