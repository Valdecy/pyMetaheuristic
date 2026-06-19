"""pyMetaheuristic src — Reconstructed Differential Evolution Engine."""
from __future__ import annotations

from ._shade_variants import ShadeVariantEngine


class RDEEngine(ShadeVariantEngine):
    """RDE: reconstructed DE with order-pbest/current-to-pbest strategy allocation."""

    algorithm_id = "rde"
    algorithm_name = "Reconstructed Differential Evolution"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.48550/arXiv.2404.16280",
        "title": "An Efficient Reconstructed Differential Evolution Variant by Some of the Current State-of-the-art Strategies for Solving Single Objective Bound Constrained Problems",
        "authors": "Sichen Tao, Ruihan Zhao, Kaiyu Wang, Shangce Gao",
        "year": 2024,
    }
    _DEFAULTS = dict(
        ShadeVariantEngine._DEFAULTS,
        population_size=108,
        min_population_size=4,
        hist_mem_size=5,
        memory_f_init=0.3,
        memory_cr_init=0.8,
        pmax=0.25,
        pbest_min=0.05,
        rank_greediness=3.0,
        archive_rate=1.0,
        strategy_ratios=[0.5, 0.5],
        jumping_rate=0.2,
        cauchy_scale=0.1,
    )
    _VARIANT_OVERRIDES = dict(rank_pressure=True, rde=True, use_order_pbest=True, cauchy_perturb=True)
