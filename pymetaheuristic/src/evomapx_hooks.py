"""EvoMapX attribution gate and compatibility helpers.

This module no longer fabricates family-template operator telemetry. Engines
that emit complete signed native operator logs keep those logs. Engines that do
not emit complete native logs explicitly opt out with
``evomapx_delta_f == "unavailable"`` and ``evomapx_fidelity_runtime ==
"no_attribution"``.

Lineage metadata is never overwritten when an engine has already populated
``state.payload["lineage"]``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


DE_VARIANTS: set[str] = {
    "de", "jade", "sade", "sap_de", "hde", "jde", "shade", "ilshade",
    "lshade_cnepsin", "imode", "sade_amss", "sade_atdsc", "sade_sammon",
}

PSO_VARIANTS: set[str] = {
    "pso", "aiw_pso", "aesspso", "gpso",
}

GWO_VARIANTS: set[str] = {
    "gwo", "acgwo", "cg_gwo", "chaotic_gwo", "ds_gwo", "er_gwo", "ex_gwo",
    "fuzzy_gwo", "gwo_woa", "i_gwo", "iagwo", "incremental_gwo", "iobl_gwo",
    "ogwo",
}

WOA_VARIANTS: set[str] = {
    "woa", "i_woa", "hi_woa", "gwo_woa", "whale_foa", "nwoa",
}

HHO_VARIANTS: set[str] = {
    "hho",
}

BAT_VARIANTS: set[str] = {
    "bat_a", "hba", "hsaba", "saba", "dvba", "plba",
}

FIREFLY_VARIANTS: set[str] = {
    "firefly_a",
}

CUCKOO_LEVY_VARIANTS: set[str] = {
    "cuckoo_s", "cco", "fpa", "lfd", "levy_ja", "laro",
}

ABC_BEE_VARIANTS: set[str] = {
    "abco", "bea",
}

ACO_ANT_VARIANTS: set[str] = {
    "aco", "acor", "misaco",
}

ANTLION_VARIANTS: set[str] = {
    "alo",
}

# Broad Phase 4 leader-guided swarm family.  These methods share the
# common pattern of generating new candidates from exploration/exploitation
# rules around leaders, elites, prey, groups, or social references.  More
# specialized families are checked before this generic leader template.
LEADER_GUIDED_SWARM_VARIANTS: set[str] = {
    "aaa", "aao", "afsa", "agto", "aha", "aho", "ala", "ao", "aoa",
    "aoo", "apo", "aro", "aso", "avoa", "bboa", "bbso", "bes", "bfo",
    "bka", "bmo", "boa", "bono", "bps", "bsa", "camel", "capsa",
    "cat_so", "cddo", "cdo", "cfoa", "chameleon_sa", "chicken_so",
    "choa", "coa", "coati_oa", "cockroach_so", "coot", "cpo",
    "crayfish_oa", "csa", "csbo", "cso", "da", "dbo", "deo_dolphin",
    "dfo", "dhole_oa", "dmoa", "ecological_cycle_o", "eefo",
    "eel_grouper_o", "eho", "elk_ho", "eoa", "epc", "esoa", "fda",
    "fdo", "ffa", "ffo", "flo", "foa", "foa_fossa", "fox", "fss",
    "fwa", "gazelle_oa", "ggo", "gja", "gjo", "gkso", "gmo",
    "go_growth", "goa", "gpoo", "gso", "gso_glider_snake", "gto",
    "hba_honey", "hgs", "ho_hippo", "horse_oa", "hus", "iaro",
    "improved_tlo", "jso", "jy", "kha", "kma", "loa", "loa_lyrebird",
    "mbo", "mfa", "mfo", "mgo", "mpa", "mrfo", "msa_e", "mshoa",
    "mvo", "ngo", "nmra", "ofa", "ooa", "parrot_o", "pdo", "pfa",
    "pfa_polar_fox", "pko", "poa", "puma_o", "qle_sca", "rbmo",
    "rfo", "rhso", "roa", "rsa", "rso", "sacoso", "samso", "sbo",
    "sboa", "scso", "seaho", "serval_oa", "sfo", "sfoa", "shio",
    "shio_success", "sho", "sine_cosine_a", "slo", "smo", "so_snake",
    "soa", "sos", "sparrow_sa", "spbo", "squirrel_sa", "srsr",
    "srsr_robotics", "ssa", "sso", "sspider_a", "sto", "superb_foa",
    "tdo", "tlbo", "tlco", "tsa", "tso", "vcs", "waoa", "who",
    "wmqimrfo", "wooa", "wso", "zoa",
}


# Phase 5 physics-based and equilibrium/force-field algorithms.  These methods
# share mechanics based on forces, fields, equilibrium pools, waves, flows,
# physical coefficients, or energy-state updates.  More specific native engine
# logs are preserved when available; this family hook provides passive operator
# attribution from already-computed pre/post fitness.
PHYSICS_FORCE_FIELD_VARIANTS: set[str] = {
    "adaptive_eo", "aefa", "arch_oa", "aso_atom", "cdo_chernobyl",
    "ceo_cosmic", "ddao", "do_dandelion", "ecpo", "efo",
    "enhanced_two", "eo", "eso", "evo", "fata", "fla", "flood_a",
    "gea", "gsa", "hgso", "ikoa", "liwo", "lso_spectrum",
    "modified_eo", "mso", "nro", "plo", "rcco", "rime", "snow_oa",
    "soo", "tfwo", "toc", "two", "wdo", "wo_wave", "ydse",
}


# Phase 6 human/social/teaching/competition algorithms.  These methods
# update candidates through teaching/learning, social imitation, role or team
# competition, assimilation/revolution, queueing, rescue/investigation, or
# supply-demand style social interaction mechanisms.
TEACHING_LEARNING_VARIANTS: set[str] = {
    "tlbo", "improved_tlo", "petio", "toa", "eco", "spbo",
}

COMPETITION_ROLE_VARIANTS: set[str] = {
    "btoa", "bro", "bso", "chio", "gco", "gska", "hbo", "ica",
    "mgoa_market", "mvpa", "political_o", "pro", "qsa", "improved_qsa",
    "saro", "thro", "warso",
}

HUMAN_SOCIAL_VARIANTS: set[str] = {
    "aft", "aeo", "btoa", "bro", "bso", "cddo_child", "chio", "doa",
    "dra", "dream_oa", "dso", "eco", "enhanced_aeo", "esc", "fbio",
    "gco", "gska", "hbo", "hiking_oa", "hco", "heoa", "ica",
    "improved_aeo", "improved_qsa", "lco", "mgoa_market", "modified_aeo",
    "mtbo", "mvpa", "petio", "political_o", "pro", "qsa", "saro",
    "singer_oa", "ssdo", "supply_do", "thro", "toa", "warso",
    # Included because the method is explicitly teaching/student/social even
    # when the source table classifies it outside the human family.
    "tlbo", "improved_tlo", "spbo",
}


# Phase 7 distribution/model-based, surrogate-assisted, CMA-ES/EDA, and
# trajectory/local-search algorithms.  These methods often expose different
# internal objects (probability vectors, covariance matrices, acquisition
# models, neighborhoods, tabu memories, or restarts).  The hooks below keep
# the same passive-budget rule: no objective call and no random-number use.
DISTRIBUTION_MODEL_VARIANTS: set[str] = {
    "cem", "compact_ga", "ego", "pbil", "sopt",
}

CMAES_EDA_VARIANTS: set[str] = {
    "cmaes", "bipop_cmaes", "ipop_cmaes", "nlapsmjso_eda",
}

SURROGATE_MODEL_VARIANTS: set[str] = {
    "et_bo", "gp_bo", "gbrt_bo", "rf_bo",
}

SURROGATE_ASSISTED_VARIANTS: set[str] = {
    "l2smea", "misaco", "sacc_eam2", "sacoso", "sade_amss",
    "sade_atdsc", "sapo",
}

TRAJECTORY_LOCAL_VARIANTS: set[str] = {
    "ars", "basin_hopping", "grasp", "hsa", "hc", "ils", "msls",
    "mts", "nmm", "random_s", "sa", "ts", "vns",
}

GRADIENT_LOCAL_VARIANTS: set[str] = {
    "adam", "bfgs", "frcg", "rmsprop", "sd", "sqp",
}


# Phase 8 remaining evolutionary, immune/clonal, genetic/memetic,
# cultural/biogeography, evolutionary programming/strategy, and
# multi-factorial algorithms.  DE variants, CMA-ES/EDA variants, and
# surrogate-assisted evolutionary methods were already assigned to earlier
# phases and are intentionally not duplicated here.
GENETIC_MEMETIC_VARIANTS: set[str] = {
    "autov", "bspga", "bwo", "ga", "memetic_a", "pcx", "ssio_rl",
}

IMMUNE_CULTURAL_EVOLUTIONARY_VARIANTS: set[str] = {
    "bbo", "clonalg", "cro", "ca", "ocro", "mke", "bco", "bacterial_colony_o",
}

EVOLUTIONARY_PROGRAMMING_STRATEGY_VARIANTS: set[str] = {
    "es", "ep", "fep",
}

MULTIFACTORIAL_EVOLUTIONARY_VARIANTS: set[str] = {
    "mfea", "mfea2", "nndrea_so", "frofi",
}

PHASE8_EVOLUTIONARY_VARIANTS: set[str] = (
    GENETIC_MEMETIC_VARIANTS
    | IMMUNE_CULTURAL_EVOLUTIONARY_VARIANTS
    | EVOLUTIONARY_PROGRAMMING_STRATEGY_VARIANTS
    | MULTIFACTORIAL_EVOLUTIONARY_VARIANTS
)


# Phase 9 remaining nature/biology/growth and mathematical-transform
# optimizers.  Biological/nature methods already handled as immune or
# bacterial evolutionary methods in Phase 8 are intentionally not repeated
# here.  Gradient/local mathematical methods were handled in Phase 7; Phase 9
# covers transform/interpolation/distribution-style mathematical optimizers.
NATURE_BIOLOGY_GROWTH_VARIANTS: set[str] = {
    "artemisinin_o", "eao", "ivya", "iwo", "lca", "lpo", "moss_go",
    "sma", "tpo", "tree_seed_a", "wca", "wutp",
}

MATH_TRANSFORM_VARIANTS: set[str] = {
    "cgo", "circle_sa", "edo", "eto", "gbo", "gndo", "info", "nca",
    "noa", "pss", "qio", "run", "scho", "ttao",
}

PHASE9_NATURE_MATH_VARIANTS: set[str] = (
    NATURE_BIOLOGY_GROWTH_VARIANTS | MATH_TRANSFORM_VARIANTS
)

PHASE2_NATIVE_FAMILY_ALGORITHMS: set[str] = DE_VARIANTS | PSO_VARIANTS | GWO_VARIANTS
PHASE3_NATIVE_FAMILY_ALGORITHMS: set[str] = (
    WOA_VARIANTS | HHO_VARIANTS | BAT_VARIANTS | FIREFLY_VARIANTS | CUCKOO_LEVY_VARIANTS
)
PHASE4_NATIVE_FAMILY_ALGORITHMS: set[str] = (
    ABC_BEE_VARIANTS | ACO_ANT_VARIANTS | ANTLION_VARIANTS | LEADER_GUIDED_SWARM_VARIANTS
)
PHASE5_NATIVE_FAMILY_ALGORITHMS: set[str] = PHYSICS_FORCE_FIELD_VARIANTS
PHASE6_NATIVE_FAMILY_ALGORITHMS: set[str] = HUMAN_SOCIAL_VARIANTS
PHASE7_NATIVE_FAMILY_ALGORITHMS: set[str] = (
    DISTRIBUTION_MODEL_VARIANTS
    | CMAES_EDA_VARIANTS
    | SURROGATE_MODEL_VARIANTS
    | SURROGATE_ASSISTED_VARIANTS
    | TRAJECTORY_LOCAL_VARIANTS
    | GRADIENT_LOCAL_VARIANTS
)
UNCATEGORIZED_VARIANTS: set[str] = set()

NATIVE_FAMILY_ALGORITHMS: set[str] = (
    PHASE2_NATIVE_FAMILY_ALGORITHMS
    | PHASE3_NATIVE_FAMILY_ALGORITHMS
    | PHASE4_NATIVE_FAMILY_ALGORITHMS
    | PHASE5_NATIVE_FAMILY_ALGORITHMS
    | PHASE6_NATIVE_FAMILY_ALGORITHMS
    | PHASE7_NATIVE_FAMILY_ALGORITHMS
    | PHASE8_EVOLUTIONARY_VARIANTS
    | PHASE9_NATURE_MATH_VARIANTS
    | UNCATEGORIZED_VARIANTS
)


def _safe_copy(value: Any) -> Any:
    """Copy arrays cheaply and fall back to deepcopy for small metadata."""
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return deepcopy(value)
    except Exception:
        return value


def _payload_array(payload: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, np.ndarray):
            return value.copy()
    return None


def _population_key_from_state(engine: Any, state: Any) -> str | None:
    try:
        key = engine._population_payload_key(state)
        if key is not None:
            return str(key)
    except Exception:
        pass
    payload = getattr(state, "payload", {}) or {}
    for key in ("population", "positions", "sources", "memories"):
        value = payload.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return key
    return None


def _fitness_column(pop: np.ndarray | None, dim: int | None = None) -> np.ndarray | None:
    if not isinstance(pop, np.ndarray) or pop.ndim != 2 or pop.shape[1] < 2:
        return None
    if dim is not None and pop.shape[1] < int(dim) + 1:
        return None
    return np.asarray(pop[:, -1], dtype=float).copy()


def _positive_scalar(before: float | None, after: float | None, objective: str) -> float:
    if before is None or after is None:
        return 0.0
    try:
        b = float(before)
        a = float(after)
    except Exception:
        return 0.0
    if not (np.isfinite(b) and np.isfinite(a)):
        return 0.0
    if str(objective).lower() == "max":
        return float(max(0.0, a - b))
    return float(max(0.0, b - a))


def _positive_array_gain(before: np.ndarray | None, after: np.ndarray | None, objective: str) -> float:
    if before is None or after is None:
        return 0.0
    b = np.asarray(before, dtype=float).reshape(-1)
    a = np.asarray(after, dtype=float).reshape(-1)
    if b.size == 0 or a.size == 0:
        return 0.0
    n = min(b.size, a.size)
    b = b[:n]
    a = a[:n]
    if str(objective).lower() == "max":
        gain = a - b
    else:
        gain = b - a
    gain = gain[np.isfinite(gain)]
    if gain.size == 0:
        return 0.0
    return float(np.maximum(gain, 0.0).sum())


def _best_gain(before: dict[str, Any] | None, after_state: Any, objective: str) -> float:
    if not before:
        return 0.0
    return _positive_scalar(before.get("best_fitness"), getattr(after_state, "best_fitness", None), objective)


def _total_observed_gain(before: dict[str, Any] | None, after_state: Any, engine: Any) -> tuple[float, dict[str, float]]:
    """Return macro gain and diagnostic gain components from observed data only."""
    if not before:
        return 0.0, {"population_gain": 0.0, "best_gain": 0.0, "memory_gain": 0.0}
    objective = getattr(getattr(engine, "problem", None), "objective", "min")
    dim = getattr(getattr(engine, "problem", None), "dimension", None)
    payload = getattr(after_state, "payload", {}) or {}
    pop_key = before.get("population_key") or _population_key_from_state(engine, after_state)
    after_pop = payload.get(pop_key) if pop_key else None
    before_fit = _fitness_column(before.get("population"), dim)
    after_fit = _fitness_column(after_pop, dim)
    pop_gain = _positive_array_gain(before_fit, after_fit, objective)
    best_gain = _best_gain(before, after_state, objective)

    memory_gain = 0.0
    before_i_best = _fitness_column(before.get("i_best"), dim)
    after_i_best = _fitness_column(payload.get("i_best"), dim)
    memory_gain += _positive_array_gain(before_i_best, after_i_best, objective)
    before_g_best = before.get("g_best")
    after_g_best = payload.get("g_best")
    if isinstance(before_g_best, np.ndarray) and before_g_best.ndim == 1 and before_g_best.size >= 1:
        b = float(before_g_best[-1])
        if isinstance(after_g_best, np.ndarray) and after_g_best.ndim == 1 and after_g_best.size >= 1:
            memory_gain += _positive_scalar(b, float(after_g_best[-1]), objective)

    total = max(float(pop_gain), float(best_gain), float(memory_gain))
    return total, {"population_gain": float(pop_gain), "best_gain": float(best_gain), "memory_gain": float(memory_gain)}


def capture_phase2_state(engine: Any, state: Any) -> dict[str, Any] | None:
    """Capture minimal pre-step data for Phase 2 family hooks."""
    algorithm_id = str(getattr(engine, "algorithm_id", "") or "").lower()
    if algorithm_id not in NATIVE_FAMILY_ALGORITHMS:
        return None
    payload = getattr(state, "payload", {}) or {}
    pop_key = _population_key_from_state(engine, state)
    captured: dict[str, Any] = {
        "algorithm_id": algorithm_id,
        "step": int(getattr(state, "step", 0) or 0),
        "best_fitness": _safe_copy(getattr(state, "best_fitness", None)),
        "population_key": pop_key,
        "population": _safe_copy(payload.get(pop_key)) if pop_key else None,
        "velocities": _payload_array(payload, ("velocities", "velocity")),
        "i_best": _payload_array(payload, ("i_best", "p_best", "personal_best", "personal_bests")),
        "g_best": _payload_array(payload, ("g_best", "global_best")),
        "elite": _payload_array(payload, ("elite", "best", "leader")),
        "lineage": _safe_copy(payload.get("lineage")),
    }
    return captured


def _weighted_contributions(total_gain: float, weights: dict[str, float]) -> dict[str, float]:
    total_gain = float(total_gain or 0.0)
    denom = float(sum(max(0.0, float(v)) for v in weights.values())) or 1.0
    return {str(k): float(total_gain * max(0.0, float(v)) / denom) for k, v in weights.items()}


def _de_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    # Differential mutation/crossover are not separately evaluated in most DE
    # variants.  The observed trial/replacement gain is therefore assigned to a
    # native-family variation pipeline without adding objective calls.
    weights = {
        "differential mutation": 0.40,
        "crossover/recombination": 0.35,
        "greedy selection/replacement": 0.25,
    }
    if algorithm_id in {"jade", "sade", "sap_de", "jde", "shade", "ilshade", "lshade_cnepsin", "imode"}:
        weights["parameter adaptation/archive"] = 0.0
    if algorithm_id in {"sade_amss", "sade_atdsc", "sade_sammon"}:
        weights["surrogate/subspace assistance"] = 0.0
    return _weighted_contributions(total_gain, weights)


def _pso_contributions(total_gain: float, diagnostics: dict[str, float]) -> dict[str, float]:
    memory_gain = float(diagnostics.get("memory_gain", 0.0) or 0.0)
    if total_gain <= 0.0:
        return {
            "velocity/social update": 0.0,
            "position update": 0.0,
            "personal/global memory": 0.0,
        }
    memory_share = min(0.35, memory_gain / (total_gain + 1.0e-12)) if memory_gain > 0.0 else 0.20
    remaining = max(0.0, 1.0 - memory_share)
    weights = {
        "velocity/social update": 0.60 * remaining,
        "position update": 0.40 * remaining,
        "personal/global memory": memory_share,
    }
    return _weighted_contributions(total_gain, weights)


def _gwo_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    weights = {
        "alpha/beta/delta guidance": 0.50,
        "encircling/diversification": 0.30,
        "position update/replacement": 0.20,
    }
    if algorithm_id == "gwo_woa":
        weights["spiral/whale exploitation"] = 0.0
    if algorithm_id in {"acgwo", "chaotic_gwo", "cg_gwo"}:
        weights["chaotic/adaptive control"] = 0.0
    if algorithm_id in {"iobl_gwo", "ogwo"}:
        weights["opposition learning"] = 0.0
    return _weighted_contributions(total_gain, weights)


def _woa_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    weights = {
        "encircling/search": 0.35,
        "spiral exploitation": 0.30,
        "leader guidance": 0.20,
        "replacement": 0.15,
    }
    if algorithm_id == "gwo_woa":
        weights["wolf hierarchy guidance"] = 0.0
    if algorithm_id == "whale_foa":
        weights["fruit-fly sensory search"] = 0.0
    return _weighted_contributions(total_gain, weights)


def _hho_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "exploration perch/search": 0.30,
        "soft/hard besiege": 0.35,
        "rapid dive/exploitation": 0.20,
        "replacement": 0.15,
    }
    return _weighted_contributions(total_gain, weights)


def _bat_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "velocity/frequency update": 0.35,
        "local random walk": 0.25,
        "acceptance/replacement": 0.25,
        "pulse/loudness adaptation": 0.15,
    }
    return _weighted_contributions(total_gain, weights)


def _firefly_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "brightness attraction": 0.45,
        "randomization": 0.30,
        "selection/replacement": 0.25,
    }
    return _weighted_contributions(total_gain, weights)


def _cuckoo_levy_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    weights = {
        "levy/global pollination": 0.45,
        "local pollination/random walk": 0.30,
        "replacement/selection": 0.25,
    }
    if algorithm_id in {"cuckoo_s", "cco"}:
        weights["abandonment/reinitialization"] = 0.10
    if algorithm_id == "laro":
        weights["opposition learning"] = 0.0
    return _weighted_contributions(total_gain, weights)


def _abc_bee_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "employed/foraging search": 0.35,
        "onlooker/elite selection": 0.25,
        "scout/reinitialization": 0.20,
        "replacement/memorization": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _aco_ant_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "solution construction": 0.40,
        "pheromone/model update": 0.20,
        "local/global exploitation": 0.25,
        "selection/replacement": 0.15,
    }
    return _weighted_contributions(total_gain, weights)


def _antlion_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "random walk around antlion": 0.35,
        "elite antlion guidance": 0.30,
        "trap/boundary adaptation": 0.15,
        "selection/replacement": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _physics_equilibrium_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "equilibrium pool guidance": 0.35,
        "generation/control-rate update": 0.20,
        "position update": 0.25,
        "selection/replacement": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _physics_force_field_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "force/field interaction": 0.35,
        "acceleration/mass update": 0.20,
        "position update": 0.25,
        "selection/replacement": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _physics_wave_flow_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "flow/wave propagation": 0.35,
        "physical coefficient update": 0.20,
        "position transport/update": 0.25,
        "selection/replacement": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _physics_energy_state_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "energy/state transition": 0.30,
        "force/equilibrium guidance": 0.30,
        "position update": 0.25,
        "selection/replacement": 0.15,
    }
    return _weighted_contributions(total_gain, weights)


def _physics_force_field_family_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    equilibrium = {"adaptive_eo", "eo", "modified_eo", "hgso"}
    force_field = {"aefa", "aso_atom", "ecpo", "efo", "gsa", "nro", "two", "enhanced_two"}
    wave_flow = {"fla", "flood_a", "liwo", "lso_spectrum", "rcco", "tfwo", "toc", "wdo", "wo_wave", "ydse"}
    if algorithm_id in equilibrium:
        return _physics_equilibrium_contributions(total_gain)
    if algorithm_id in force_field:
        return _physics_force_field_contributions(total_gain)
    if algorithm_id in wave_flow:
        return _physics_wave_flow_contributions(total_gain)
    return _physics_energy_state_contributions(total_gain)



def _teaching_learning_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "teacher/leader phase": 0.35,
        "learner/social phase": 0.30,
        "competition/evaluation": 0.15,
        "selection/replacement": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _competition_role_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "role/team competition": 0.35,
        "social learning/assimilation": 0.25,
        "movement/update": 0.20,
        "selection/replacement": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _human_social_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "social learning": 0.30,
        "competition/role update": 0.25,
        "movement/update": 0.25,
        "selection/replacement": 0.20,
    }
    return _weighted_contributions(total_gain, weights)


def _human_social_family_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id in TEACHING_LEARNING_VARIANTS:
        return _teaching_learning_contributions(total_gain)
    if algorithm_id in COMPETITION_ROLE_VARIANTS:
        return _competition_role_contributions(total_gain)
    return _human_social_contributions(total_gain)



def _distribution_model_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id in {"compact_ga", "pbil"}:
        weights = {
            "probability-vector sampling": 0.35,
            "elite/model selection": 0.25,
            "probability-model update": 0.25,
            "replacement/incumbent update": 0.15,
        }
    elif algorithm_id == "cem":
        weights = {
            "sampling": 0.35,
            "elite selection": 0.30,
            "distribution update": 0.20,
            "replacement/incumbent update": 0.15,
        }
    elif algorithm_id == "ego":
        weights = {
            "surrogate/model fit": 0.25,
            "acquisition search": 0.35,
            "candidate evaluation": 0.25,
            "incumbent/model update": 0.15,
        }
    else:
        weights = {
            "model sampling": 0.35,
            "elite/model selection": 0.25,
            "model update": 0.25,
            "replacement/incumbent update": 0.15,
        }
    return _weighted_contributions(total_gain, weights)


def _cmaes_eda_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    weights = {
        "multivariate sampling": 0.30,
        "elite/parent selection": 0.25,
        "mean/covariance update": 0.25,
        "step-size/restart control": 0.20,
    }
    if algorithm_id == "nlapsmjso_eda":
        weights = {
            "success-history sampling": 0.25,
            "EDA model update": 0.25,
            "local/adaptive search": 0.25,
            "selection/replacement": 0.25,
        }
    return _weighted_contributions(total_gain, weights)


def _surrogate_model_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "surrogate fit": 0.25,
        "acquisition search": 0.35,
        "candidate evaluation": 0.25,
        "model/incumbent update": 0.15,
    }
    return _weighted_contributions(total_gain, weights)


def _surrogate_assisted_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    weights = {
        "surrogate screening/modeling": 0.25,
        "evolutionary/swarm variation": 0.30,
        "candidate evaluation": 0.25,
        "selection/model update": 0.20,
    }
    if algorithm_id in {"sade_amss", "sade_atdsc"}:
        weights = {
            "DE mutation/crossover": 0.30,
            "surrogate/subspace screening": 0.30,
            "candidate evaluation": 0.20,
            "selection/parameter update": 0.20,
        }
    if algorithm_id == "misaco":
        weights = {
            "ant solution construction": 0.30,
            "surrogate assistance": 0.30,
            "pheromone/model update": 0.20,
            "selection/replacement": 0.20,
        }
    return _weighted_contributions(total_gain, weights)


def _trajectory_local_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id in {"sa"}:
        weights = {
            "neighborhood proposal": 0.35,
            "temperature/metropolis acceptance": 0.30,
            "cooling/step adaptation": 0.15,
            "incumbent update/restart": 0.20,
        }
    elif algorithm_id in {"ts"}:
        weights = {
            "neighborhood proposal": 0.30,
            "tabu memory/filtering": 0.25,
            "aspiration/acceptance": 0.20,
            "incumbent update": 0.25,
        }
    elif algorithm_id in {"nmm"}:
        weights = {
            "reflection/expansion": 0.35,
            "contraction/shrink": 0.25,
            "simplex ranking": 0.20,
            "incumbent update": 0.20,
        }
    elif algorithm_id in {"basin_hopping", "ils", "msls", "grasp", "vns"}:
        weights = {
            "construction/perturbation": 0.30,
            "local search": 0.35,
            "acceptance/replacement": 0.20,
            "restart/neighborhood change": 0.15,
        }
    elif algorithm_id == "hsa":
        weights = {
            "memory consideration": 0.30,
            "pitch adjustment": 0.25,
            "random improvisation": 0.25,
            "selection/replacement": 0.20,
        }
    else:
        weights = {
            "proposal/neighborhood move": 0.35,
            "move acceptance": 0.30,
            "step-size/adaptation": 0.15,
            "incumbent update": 0.20,
        }
    return _weighted_contributions(total_gain, weights)


def _gradient_local_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id in {"adam", "rmsprop"}:
        weights = {
            "gradient estimate": 0.25,
            "moment/adaptive scaling": 0.30,
            "parameter step": 0.30,
            "acceptance/incumbent update": 0.15,
        }
    elif algorithm_id in {"bfgs", "sqp"}:
        weights = {
            "model/curvature update": 0.30,
            "search direction": 0.25,
            "line-search/step": 0.25,
            "acceptance/incumbent update": 0.20,
        }
    else:
        weights = {
            "descent direction": 0.35,
            "step-size/line search": 0.25,
            "parameter step": 0.25,
            "acceptance/incumbent update": 0.15,
        }
    return _weighted_contributions(total_gain, weights)




def _genetic_memetic_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id == "memetic_a":
        weights = {
            "parent selection": 0.20,
            "crossover/recombination": 0.25,
            "mutation/diversification": 0.20,
            "local improvement": 0.20,
            "elitist replacement": 0.15,
        }
    elif algorithm_id == "pcx":
        weights = {
            "parent-centric crossover": 0.40,
            "offspring generation": 0.20,
            "selection/replacement": 0.25,
            "diversification control": 0.15,
        }
    elif algorithm_id == "autov":
        weights = {
            "operator design/selection": 0.25,
            "adaptive variation": 0.30,
            "candidate evaluation": 0.20,
            "selection/replacement": 0.25,
        }
    elif algorithm_id == "ssio_rl":
        weights = {
            "operator policy/action": 0.35,
            "variation/update": 0.25,
            "reward/fitness feedback": 0.20,
            "selection/replacement": 0.20,
        }
    else:
        weights = {
            "parent selection": 0.25,
            "crossover/recombination": 0.30,
            "mutation/diversification": 0.25,
            "elitist replacement": 0.20,
        }
    return _weighted_contributions(total_gain, weights)


def _immune_cultural_evolutionary_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id == "clonalg":
        weights = {
            "affinity selection": 0.25,
            "cloning/reproduction": 0.25,
            "hypermutation": 0.30,
            "receptor editing/replacement": 0.20,
        }
    elif algorithm_id in {"bco", "bacterial_colony_o"}:
        weights = {
            "chemotaxis/foraging move": 0.30,
            "reproduction/spread": 0.25,
            "elimination/dispersal": 0.20,
            "selection/replacement": 0.25,
        }
    elif algorithm_id == "bbo":
        weights = {
            "migration/immigration": 0.35,
            "mutation/diversification": 0.25,
            "habitat suitability selection": 0.20,
            "elitist replacement": 0.20,
        }
    elif algorithm_id in {"cro", "ocro"}:
        weights = {
            "larvae settlement/reproduction": 0.30,
            "mutation/depredation": 0.25,
            "reef competition": 0.25,
            "selection/replacement": 0.20,
        }
    elif algorithm_id == "ca":
        weights = {
            "population variation": 0.30,
            "belief-space influence": 0.25,
            "belief update": 0.20,
            "selection/replacement": 0.25,
        }
    else:
        weights = {
            "evolutionary variation": 0.35,
            "adaptive diversification": 0.25,
            "selection pressure": 0.20,
            "replacement": 0.20,
        }
    return _weighted_contributions(total_gain, weights)


def _evolutionary_programming_strategy_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "mutation/self-adaptation": 0.35,
        "offspring generation": 0.25,
        "survivor selection": 0.25,
        "strategy-parameter update": 0.15,
    }
    return _weighted_contributions(total_gain, weights)


def _multifactorial_evolutionary_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id == "frofi":
        weights = {
            "feasibility-rule filtering": 0.30,
            "objective-informed selection": 0.25,
            "variation/update": 0.25,
            "replacement": 0.20,
        }
    elif algorithm_id == "nndrea_so":
        weights = {
            "dimensionality-reduction model": 0.25,
            "latent-space variation": 0.30,
            "candidate reconstruction/evaluation": 0.25,
            "selection/replacement": 0.20,
        }
    else:
        weights = {
            "skill-factor assignment": 0.20,
            "assortative mating/transfer": 0.30,
            "mutation/diversification": 0.25,
            "selection/replacement": 0.25,
        }
    return _weighted_contributions(total_gain, weights)


def _phase8_evolutionary_family_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id in GENETIC_MEMETIC_VARIANTS:
        return _genetic_memetic_contributions(algorithm_id, total_gain)
    if algorithm_id in IMMUNE_CULTURAL_EVOLUTIONARY_VARIANTS:
        return _immune_cultural_evolutionary_contributions(algorithm_id, total_gain)
    if algorithm_id in EVOLUTIONARY_PROGRAMMING_STRATEGY_VARIANTS:
        return _evolutionary_programming_strategy_contributions(total_gain)
    return _multifactorial_evolutionary_contributions(algorithm_id, total_gain)


def _nature_biology_growth_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id == "iwo":
        weights = {
            "seed dispersal/germination": 0.30,
            "spatial spread/growth": 0.25,
            "competitive exclusion": 0.25,
            "selection/replacement": 0.20,
        }
    elif algorithm_id in {"tree_seed_a", "tpo", "wca", "wutp", "moss_go"}:
        weights = {
            "growth/branching update": 0.30,
            "resource/water transport": 0.25,
            "foraging/spread move": 0.25,
            "selection/replacement": 0.20,
        }
    elif algorithm_id in {"sma", "artemisinin_o", "eao", "ivya", "lca", "lpo"}:
        weights = {
            "biological activity/growth": 0.30,
            "adaptive foraging/spread": 0.30,
            "reproduction/diversification": 0.20,
            "selection/replacement": 0.20,
        }
    else:
        weights = {
            "growth/foraging move": 0.35,
            "reproduction/spread": 0.25,
            "adaptive diversification": 0.20,
            "selection/replacement": 0.20,
        }
    return _weighted_contributions(total_gain, weights)


def _math_transform_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id in {"qio", "ttao"}:
        weights = {
            "interpolation/geometric transform": 0.35,
            "candidate update": 0.25,
            "exploitation refinement": 0.20,
            "selection/replacement": 0.20,
        }
    elif algorithm_id in {"edo", "gndo", "nca", "scho"}:
        weights = {
            "probability/distribution transform": 0.35,
            "candidate generation": 0.25,
            "adaptive scaling": 0.20,
            "selection/replacement": 0.20,
        }
    elif algorithm_id in {"run", "gbo"}:
        weights = {
            "mathematical search direction": 0.30,
            "runge-kutta/gradient transform": 0.30,
            "candidate update": 0.20,
            "selection/replacement": 0.20,
        }
    elif algorithm_id in {"cgo", "circle_sa", "eto", "info", "noa", "pss"}:
        weights = {
            "mathematical transform": 0.35,
            "candidate update": 0.30,
            "diversification/control": 0.15,
            "selection/replacement": 0.20,
        }
    else:
        weights = {
            "mathematical transform": 0.35,
            "candidate update": 0.30,
            "adaptive control": 0.15,
            "selection/replacement": 0.20,
        }
    return _weighted_contributions(total_gain, weights)


def _phase9_nature_math_family_contributions(algorithm_id: str, total_gain: float) -> dict[str, float]:
    if algorithm_id in NATURE_BIOLOGY_GROWTH_VARIANTS:
        return _nature_biology_growth_contributions(algorithm_id, total_gain)
    return _math_transform_contributions(algorithm_id, total_gain)

def _leader_guided_swarm_contributions(total_gain: float) -> dict[str, float]:
    weights = {
        "exploration move": 0.30,
        "exploitation move": 0.30,
        "leader/social guidance": 0.25,
        "selection/replacement": 0.15,
    }
    return _weighted_contributions(total_gain, weights)



def _signed_scalar_delta(before: float | None, after: float | None, objective: str) -> float:
    if before is None or after is None:
        return 0.0
    try:
        b = float(before)
        a = float(after)
    except Exception:
        return 0.0
    if not (np.isfinite(b) and np.isfinite(a)):
        return 0.0
    if str(objective).lower() == "max":
        return float(a - b)
    return float(b - a)


def _signed_array_deltas(before: np.ndarray | None, after: np.ndarray | None, objective: str) -> np.ndarray:
    if before is None or after is None:
        return np.asarray([], dtype=float)
    b = np.asarray(before, dtype=float).reshape(-1)
    a = np.asarray(after, dtype=float).reshape(-1)
    if b.size == 0 or a.size == 0:
        return np.asarray([], dtype=float)
    n = min(b.size, a.size)
    if str(objective).lower() == "max":
        d = a[:n] - b[:n]
    else:
        d = b[:n] - a[:n]
    d = d[np.isfinite(d)]
    return d.astype(float, copy=False)


def _signed_observed_delta(before: dict[str, Any] | None, after_state: Any, engine: Any) -> tuple[float, int, dict[str, float]]:
    """Measure signed macro-step Δf from already-computed pre/post state only."""
    if not before:
        return 0.0, 0, {"population_delta": 0.0, "best_delta": 0.0, "memory_delta": 0.0}
    objective = getattr(getattr(engine, "problem", None), "objective", "min")
    dim = getattr(getattr(engine, "problem", None), "dimension", None)
    payload = getattr(after_state, "payload", {}) or {}
    pop_key = before.get("population_key") or _population_key_from_state(engine, after_state)
    after_pop = payload.get(pop_key) if pop_key else None
    before_fit = _fitness_column(before.get("population"), dim)
    after_fit = _fitness_column(after_pop, dim)
    deltas = _signed_array_deltas(before_fit, after_fit, objective)
    pop_delta = float(np.sum(deltas)) if deltas.size else 0.0
    pop_count = int(deltas.size)

    best_delta = _signed_scalar_delta(before.get("best_fitness"), getattr(after_state, "best_fitness", None), objective)

    # Rejected/unsuccessful proposal accounting: some engines evaluate proposals
    # but only store accepted candidates in the population. When a step exposes
    # the trial fitness vector, measure the signed proposal Δf against the
    # corresponding incumbent fitness without adding objective evaluations.
    trial_fit = payload.get("evomapx_trial_fitness")
    if trial_fit is None:
        trial_fit = payload.get("evomapx_candidate_fitness")
    trial_deltas = _signed_array_deltas(before_fit, trial_fit, objective)
    trial_delta = float(np.sum(trial_deltas)) if trial_deltas.size else 0.0
    trial_count = int(trial_deltas.size)

    # Sequential model-based optimizers store only an archive or observed design.
    # If the newly evaluated candidate is not a best-so-far improvement, the
    # signed proposal Δf relative to the previous incumbent is still meaningful.
    last_delta = 0.0
    last_fit = payload.get("last_fitness")
    if last_fit is not None:
        last_delta = _signed_scalar_delta(before.get("best_fitness"), last_fit, objective)

    memory_delta = 0.0
    before_i_best = _fitness_column(before.get("i_best"), dim)
    after_i_best = _fitness_column(payload.get("i_best"), dim)
    memory_d = _signed_array_deltas(before_i_best, after_i_best, objective)
    if memory_d.size:
        memory_delta += float(np.sum(memory_d))
    before_g_best = before.get("g_best")
    after_g_best = payload.get("g_best")
    if isinstance(before_g_best, np.ndarray) and before_g_best.ndim == 1 and before_g_best.size >= 1:
        if isinstance(after_g_best, np.ndarray) and after_g_best.ndim == 1 and after_g_best.size >= 1:
            memory_delta += _signed_scalar_delta(float(before_g_best[-1]), float(after_g_best[-1]), objective)

    signals = [
        ("population_delta", pop_delta, pop_count),
        ("trial_delta", trial_delta, trial_count),
        ("best_delta", best_delta, 1),
        ("last_candidate_delta", last_delta, 1),
        ("memory_delta", memory_delta, 1),
    ]
    name, total, count = max(signals, key=lambda item: abs(float(item[1])))
    if abs(float(total)) <= 0.0:
        count = max(1, pop_count, trial_count)
    return float(total), int(max(1, count)), {
        "population_delta": float(pop_delta),
        "trial_delta": float(trial_delta),
        "best_delta": float(best_delta),
        "last_candidate_delta": float(last_delta),
        "memory_delta": float(memory_delta),
        "selected_delta_source": name,
    }


def _family_schema_for(algorithm_id: str) -> tuple[list[str], str, str]:
    """Return mandatory labels, primary measured label, and family description."""
    aid = str(algorithm_id or "").lower()
    if aid in DE_VARIANTS:
        labels = ["de_mutation", "de_crossover", "de_selection"]
        if aid != "de":
            labels.append("de_parameter_adaptation")
        return labels, "de_selection", "DE-variant signed native composite"
    if aid in PSO_VARIANTS:
        labels = ["pso_inertia_component", "pso_cognitive_component", "pso_social_component"]
        return labels, "pso_social_component", "PSO-variant signed native composite"
    if aid in GWO_VARIANTS:
        labels = ["gwo_alpha_guidance", "gwo_beta_guidance", "gwo_delta_guidance", "gwo_position_update"]
        return labels, "gwo_position_update", "GWO-variant signed native composite"
    if aid in WOA_VARIANTS:
        labels = ["woa_encircling", "woa_spiral_exploitation", "woa_leader_guidance", "woa_replacement"]
        if aid == "whale_foa":
            labels.append("whale_foa_sensory_search")
        return labels, "woa_replacement", "WOA/whale-search signed native composite"
    if aid in HHO_VARIANTS:
        labels = ["hho_exploration", "hho_soft_besiege", "hho_hard_besiege", "hho_rapid_dive", "hho_replacement"]
        return labels, "hho_replacement", "HHO signed native composite"
    if aid in BAT_VARIANTS:
        labels = ["bat_velocity_update", "bat_local_random_walk", "bat_acceptance", "bat_pulse_loudness_adaptation"]
        return labels, "bat_acceptance", "BAT signed native composite"
    if aid in FIREFLY_VARIANTS:
        labels = ["firefly_attraction", "firefly_randomization", "firefly_replacement"]
        return labels, "firefly_replacement", "Firefly signed native composite"
    if aid in CUCKOO_LEVY_VARIANTS:
        labels = ["cs_levy_flight", "cs_successful_replacement", "cs_unsuccessful_attempt", "cs_abandoned_nest", "cs_random_init"]
        if aid == "fpa":
            labels.extend(["fpa_global_pollination", "fpa_local_pollination"])
        return labels, "cs_levy_flight", "Cuckoo/Levy signed native composite"
    if aid in ABC_BEE_VARIANTS:
        labels = ["bee_employed", "bee_onlooker", "bee_scout", "bee_replacement"]
        return labels, "bee_replacement", "ABC/bee-search signed native composite"
    if aid in ACO_ANT_VARIANTS:
        labels = ["ant_solution_construction", "ant_pheromone_update", "ant_local_exploitation", "ant_replacement"]
        return labels, "ant_replacement", "ACO/ant-search signed native composite"
    if aid in ANTLION_VARIANTS:
        labels = ["antlion_random_walk", "antlion_elite_guidance", "antlion_trap_adaptation", "antlion_replacement"]
        return labels, "antlion_replacement", "Antlion signed native composite"
    # Leader-guided swarm is checked before broader/overlapping families
    # (e.g. tlbo/improved_tlo/spbo and sacoso) because the batch plan assigns
    # those overlapping engines to the per-engine <aid>_* LGS schema.
    if aid in LEADER_GUIDED_SWARM_VARIANTS:
        labels = [f"{aid}_leader_guidance", f"{aid}_diversification", f"{aid}_position_update", f"{aid}_replacement"]
        return labels, f"{aid}_position_update", "Leader-guided swarm signed native composite"
    if aid in PHYSICS_FORCE_FIELD_VARIANTS:
        labels = ["phys_force_interaction", "phys_coefficient_update", "phys_position_update", "phys_replacement"]
        return labels, "phys_position_update", "Physics/force-field signed native composite"
    if aid in HUMAN_SOCIAL_VARIANTS:
        labels = ["hs_teacher_phase", "hs_learner_phase", "hs_competition", "hs_replacement"]
        return labels, "hs_replacement", "Human/social signed native composite"
    if aid in DISTRIBUTION_MODEL_VARIANTS:
        labels = ["dist_sampling", "dist_elite_selection", "dist_model_update", "dist_replacement"]
        return labels, "dist_replacement", "Distribution/model signed native composite"
    if aid in CMAES_EDA_VARIANTS:
        labels = ["cma_sampling", "cma_elite_selection", "cma_covariance_update", "cma_step_size_control"]
        return labels, "cma_sampling", "CMA-ES/EDA signed native composite"
    if aid in SURROGATE_MODEL_VARIANTS:
        labels = ["surr_acquisition", "surr_model_update", "surr_replacement"]
        return labels, "surr_acquisition", "Surrogate-model signed native composite"
    if aid in SURROGATE_ASSISTED_VARIANTS:
        labels = ["surr_acquisition", "surr_model_update", "surr_replacement"]
        if aid in DE_VARIANTS or aid in {"sade_amss", "sade_atdsc"}:
            labels.extend(["de_mutation", "de_crossover", "de_selection", "de_parameter_adaptation"])
            return labels, "de_selection", "Surrogate-assisted DE signed native composite"
        return labels, "surr_acquisition", "Surrogate-assisted signed native composite"
    if aid in TRAJECTORY_LOCAL_VARIANTS:
        labels = [f"{aid}_proposal", f"{aid}_acceptance", f"{aid}_step_or_temperature_update", f"{aid}_restart"]
        return labels, f"{aid}_acceptance", "Trajectory/local-search signed native composite"
    if aid in GRADIENT_LOCAL_VARIANTS:
        labels = [f"{aid}_gradient_step", f"{aid}_line_search"]
        return labels, f"{aid}_gradient_step", "Gradient/local signed native composite"
    if aid in GENETIC_MEMETIC_VARIANTS:
        labels = ["gm_crossover", "gm_mutation", "gm_local_search", "gm_selection"]
        return labels, "gm_selection", "Genetic/memetic signed native composite"
    if aid in IMMUNE_CULTURAL_EVOLUTIONARY_VARIANTS:
        labels = ["ice_variation", "ice_selection", "ice_cultural_or_immune_update", "ice_replacement"]
        return labels, "ice_replacement", "Immune/cultural/evolutionary signed native composite"
    if aid in EVOLUTIONARY_PROGRAMMING_STRATEGY_VARIANTS:
        labels = ["es_mutation", "es_recombination_if_present", "es_selection", "es_strategy_parameter_update"]
        return labels, "es_selection", "Evolutionary programming/strategy signed native composite"
    if aid in MULTIFACTORIAL_EVOLUTIONARY_VARIANTS:
        labels = ["mfea_assortative_mating", "mfea_vertical_cultural_transmission", "mfea_selection", "mfea_skill_factor_update"]
        return labels, "mfea_selection", "Multifactorial evolutionary signed native composite"
    if aid in NATURE_BIOLOGY_GROWTH_VARIANTS:
        labels = ["nbg_growth", "nbg_dispersal", "nbg_competition", "nbg_replacement"]
        return labels, "nbg_replacement", "Nature/biology/growth signed native composite"
    if aid in MATH_TRANSFORM_VARIANTS:
        labels = ["math_primary_transform", "math_secondary_transform", "math_replacement"]
        return labels, "math_replacement", "Mathematical-transform signed native composite"
    labels = [f"{aid}_update"]
    return labels, labels[0], "Unclassified signed native composite"


def _mandatory_expected_for(algorithm_id: str) -> set[str]:
    return set(_family_schema_for(algorithm_id)[0])


def _is_complete_signed_native(observation: dict[str, Any], expected: set[str]) -> bool:
    contribs = observation.get("operator_contributions")
    counts = observation.get("operator_counts")
    return (
        isinstance(contribs, dict) and bool(contribs)
        and isinstance(counts, dict)
        and observation.get("evomapx_delta_f") == "signed"
        and expected.issubset(set(contribs))
        and expected.issubset(set(counts))
        and any(isinstance(v, int) and v > 0 for v in counts.values())
    )


def augment_phase2_observation(engine: Any, before: dict[str, Any] | None, state: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Preserve complete native EvoMapX logs or mark attribution unavailable.

    This function intentionally refuses to manufacture per-operator Δf from a
    single macro-step gain. A run is marked ``native_engine`` only when the
    engine itself emitted signed operator contributions, operator counts, and
    lineage metadata. Otherwise the observation is explicitly opted out.
    """
    if observation is None:
        observation = {}
    algorithm_id = str(getattr(engine, "algorithm_id", "") or "").lower()
    if algorithm_id not in NATIVE_FAMILY_ALGORITHMS or not before:
        return observation

    labels, primary, family = _family_schema_for(algorithm_id)
    expected = set(labels)

    payload = getattr(state, "payload", {}) if state is not None else {}
    lineage = payload.get("lineage") if isinstance(payload, dict) else None
    has_lineage = isinstance(lineage, list) and len(lineage) > 0 and all(
        isinstance(x, dict) and x.get("id") and "parent_ids" in x and x.get("operator")
        for x in lineage
    )

    # Generated individual reconstructions are explicitly marked as guessed.
    # They are not paper-faithful native ports, but they do emit real signed
    # per-operator deltas and lineage from the engine itself. Preserve them
    # instead of wiping their telemetry for not matching the old family template.
    if (
        str(observation.get("evomapx_fidelity", "")).startswith("guessed_from_code_profiles")
        and isinstance(observation.get("operator_contributions"), dict)
        and bool(observation.get("operator_contributions"))
        and isinstance(observation.get("operator_counts"), dict)
        and observation.get("evomapx_delta_f") == "signed"
        and has_lineage
    ):
        observation.setdefault("evomapx_fidelity_runtime", "guessed_individual_engine")
        observation.setdefault("evomapx_phase_runtime", "preserved_guessed_engine_log")
        observation.setdefault("evomapx_family_template", family)
        observation.setdefault("evomapx_attribution_available", True)
        return observation

    if _is_complete_signed_native(observation, expected) and has_lineage:
        observation.setdefault("evomapx_fidelity_runtime", "native_engine")
        observation.setdefault("evomapx_phase_runtime", "preserved_native_engine_log")
        observation.setdefault("evomapx_family_template", family)
        return observation

    # Honest opt-out. Do not assign macro-step gain to a fake operator label;
    # do not synthesize positional lineage; do not claim native fidelity.
    observation["operator_contributions"] = {}
    observation["operator_counts"] = {}
    observation.pop("operator", None)
    observation["evomapx_delta_f"] = "unavailable"
    observation["evomapx_fidelity_runtime"] = "no_attribution"
    observation["evomapx_phase_runtime"] = "no_native_operator_attribution"
    observation["evomapx_family_template"] = family
    observation["evomapx_attribution_available"] = False
    observation["evomapx_attribution_reason"] = (
        "engine did not emit complete signed per-operator Δf and lineage; "
        "centralized fabrication is disabled"
    )
    return observation

__all__ = [
    "DE_VARIANTS",
    "PSO_VARIANTS",
    "GWO_VARIANTS",
    "WOA_VARIANTS",
    "HHO_VARIANTS",
    "BAT_VARIANTS",
    "FIREFLY_VARIANTS",
    "CUCKOO_LEVY_VARIANTS",
    "ABC_BEE_VARIANTS",
    "ACO_ANT_VARIANTS",
    "ANTLION_VARIANTS",
    "LEADER_GUIDED_SWARM_VARIANTS",
    "PHYSICS_FORCE_FIELD_VARIANTS",
    "TEACHING_LEARNING_VARIANTS",
    "COMPETITION_ROLE_VARIANTS",
    "HUMAN_SOCIAL_VARIANTS",
    "DISTRIBUTION_MODEL_VARIANTS",
    "CMAES_EDA_VARIANTS",
    "SURROGATE_MODEL_VARIANTS",
    "SURROGATE_ASSISTED_VARIANTS",
    "TRAJECTORY_LOCAL_VARIANTS",
    "GRADIENT_LOCAL_VARIANTS",
    "GENETIC_MEMETIC_VARIANTS",
    "IMMUNE_CULTURAL_EVOLUTIONARY_VARIANTS",
    "EVOLUTIONARY_PROGRAMMING_STRATEGY_VARIANTS",
    "MULTIFACTORIAL_EVOLUTIONARY_VARIANTS",
    "PHASE8_EVOLUTIONARY_VARIANTS",
    "NATURE_BIOLOGY_GROWTH_VARIANTS",
    "MATH_TRANSFORM_VARIANTS",
    "PHASE9_NATURE_MATH_VARIANTS",
    "PHASE2_NATIVE_FAMILY_ALGORITHMS",
    "PHASE3_NATIVE_FAMILY_ALGORITHMS",
    "PHASE4_NATIVE_FAMILY_ALGORITHMS",
    "PHASE5_NATIVE_FAMILY_ALGORITHMS",
    "PHASE6_NATIVE_FAMILY_ALGORITHMS",
    "PHASE7_NATIVE_FAMILY_ALGORITHMS",
    "UNCATEGORIZED_VARIANTS",
    "NATIVE_FAMILY_ALGORITHMS",
    "capture_phase2_state",
    "augment_phase2_observation",
]
