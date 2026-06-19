"""pyMetaheuristic src — LSHADE-EpSin Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class LSHADEEpSinEngine(ShadeVariantEngine):
    """L-SHADE with ensemble sinusoidal F adaptation and late Gaussian walk."""

    algorithm_id = "lshade_epsin"
    algorithm_name = "LSHADE-EpSin"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2016.7744313",
        "title": "An Ensemble Sinusoidal Parameter Adaptation incorporated with L-SHADE for Solving CEC2014 Benchmark Problems",
        "authors": "Noor H. Awad, Mostafa Z. Ali, Ponnuthurai N. Suganthan, Robert G. Reynolds",
        "year": 2016,
    }
    _DEFAULTS = dict(
        ShadeVariantEngine._DEFAULTS,
        population_size=100,
        min_population_size=4,
        hist_mem_size=6,
        memory_f_init=0.5,
        memory_cr_init=0.5,
        pbest_factor=0.11,
        epsin_frequency=0.5,
        local_search_start=0.85,
        local_search_elites=2,
        local_search_sigma=0.01,
    )
    _VARIANT_OVERRIDES = dict(epsin=True)
