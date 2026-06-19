"""pyMetaheuristic src — NL-SHADE-RSP-Midpoint Engine."""
from __future__ import annotations

from dataclasses import replace

from ._shade_variants import ShadeVariantEngine


class NLSHADERspMidpointEngine(ShadeVariantEngine):
    """NL-SHADE-RSP with population midpoint estimation, two-cluster midpoint, and restart trigger."""

    algorithm_id = "nlshade_rsp_midpoint"
    algorithm_name = "NL-SHADE-RSP-Midpoint"
    family = "evolutionary"
    capabilities = replace(ShadeVariantEngine.capabilities, supports_restart=True)
    _REFERENCE = {
        "doi": "10.1109/CEC55065.2022.9870220",
        "title": "A Version of NL-SHADE-RSP Algorithm with Midpoint for CEC 2022 Single Objective Bound Constrained Problems",
        "authors": "Rafal Biedrzycki, Jaroslaw Arabas, Eryk Warchulski",
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
        nlpsr_exponent=1.4,
        midpoint_restart_patience=9,
        midpoint_restart_tol=1e-8,
    )
    _VARIANT_OVERRIDES = dict(
        rank_pressure=True,
        nonlinear_reduction=True,
        nlshade=True,
        midpoint=True,
        midpoint_cluster=True,
        restart=True,
        resample_bounds=True,
    )
