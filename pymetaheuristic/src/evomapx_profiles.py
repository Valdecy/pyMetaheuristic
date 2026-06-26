"""Central EvoMapX operator profiles for pyMetaheuristic.

This module is intentionally organized as a single final profile catalog.  Earlier
package revisions appended phase overrides and paper addenda after the public API,
which made the effective profile for an algorithm hard to audit.  The current file
keeps each algorithm ID in exactly one section, grouped by the implementation phase
that introduced or last refined its EvoMapX taxonomy.

The profiles are declarative only.  They do not evaluate objective functions, alter
random-number consumption, or change optimizer behavior.  Runtime hooks and engines
may emit richer operator-contribution data, but this module remains the auditable
metadata source for profile/family/fidelity/operator declarations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvoMapXProfile:
    """Declared EvoMapX operator profile for an optimizer."""

    algorithm_id: str
    family: str = "unknown"
    operators: tuple[str, ...] = field(default_factory=tuple)
    fidelity: str = "profiled"  # native | native-family | profiled | family | macro
    phase: str = "profile_catalog"
    notes: str = "Operator taxonomy declared; native hooks are added progressively by family."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operators"] = list(self.operators)
        return data


FAMILY_DEFAULTS: dict[str, tuple[str, ...]] = {
    "evolutionary": ("selection", "variation", "mutation/recombination", "replacement"),
    "swarm": ("exploration move", "exploitation move", "leader/social guidance", "replacement"),
    "physics": ("interaction/force update", "field/equilibrium guidance", "position update", "replacement"),
    "human": ("social learning", "competition/role update", "movement/update", "selection/replacement"),
    "math": ("mathematical transform", "candidate update", "selection/replacement"),
    "trajectory": ("neighborhood/proposal", "move acceptance", "step adaptation"),
    "distribution": ("sampling", "elite/model selection", "distribution/model update", "replacement"),
    "surrogate": ("surrogate fit", "acquisition search", "candidate evaluation", "model update"),
    "nature": ("growth/foraging move", "reproduction/spread", "selection/replacement"),
    "unknown": ("candidate generation", "selection/replacement"),
}


EVOMAPX_OPERATOR_PROFILES: dict[str, EvoMapXProfile] = {}


def _profile(
    algorithm_id: str,
    family: str,
    operators: tuple[str, ...],
    fidelity: str,
    phase: str,
    notes: str,
) -> EvoMapXProfile:
    """Create one normalized profile row."""
    return EvoMapXProfile(
        algorithm_id=algorithm_id,
        family=family,
        operators=operators,
        fidelity=fidelity,
        phase=phase,
        notes=notes,
    )


def _register_profiles(*profiles: EvoMapXProfile) -> None:
    """Register profiles and fail fast on accidental duplicate algorithm IDs."""
    for profile in profiles:
        if profile.algorithm_id in EVOMAPX_OPERATOR_PROFILES:
            raise ValueError(f"Duplicate EvoMapX profile for {profile.algorithm_id!r}")
        EVOMAPX_OPERATOR_PROFILES[profile.algorithm_id] = profile


# ---------------------------------------------------------------------------
# Final profile catalog
# ---------------------------------------------------------------------------

# Phase 2 — DE, PSO, and GWO native-family profiles
_register_profiles(
    _profile('de', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('hde', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('ilshade', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('imode', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('jade', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('jde', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('lshade_cnepsin', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('sade', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('sade_sammon', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('sap_de', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('shade', 'evolutionary', ('differential mutation', 'crossover/recombination', 'greedy selection/replacement', 'parameter adaptation/archive'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.'),
    _profile('acgwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement', 'chaotic/adaptive control'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('aesspso', 'swarm', ('velocity/social update', 'position update', 'personal/global memory'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs PSO-style velocity, position, and memory mechanisms without extra objective evaluations.'),
    _profile('aiw_pso', 'swarm', ('velocity/social update', 'position update', 'personal/global memory'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs PSO-style velocity, position, and memory mechanisms without extra objective evaluations.'),
    _profile('cg_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement', 'chaotic/adaptive control'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('chaotic_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement', 'chaotic/adaptive control'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('ds_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('er_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('ex_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('fuzzy_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('gpso', 'swarm', ('velocity/social update', 'position update', 'personal/global memory'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs PSO-style velocity, position, and memory mechanisms without extra objective evaluations.'),
    _profile('gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('gwo_woa', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement', 'spiral/whale exploitation'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('i_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('iagwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('incremental_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('iobl_gwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement', 'opposition learning'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('ogwo', 'swarm', ('alpha/beta/delta guidance', 'encircling/diversification', 'position update/replacement', 'opposition learning'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.'),
    _profile('pso', 'swarm', ('velocity/social update', 'position update', 'personal/global memory'), 'native-family', 'phase_2_de_pso_gwo_variants', 'Phase 2 native-family EvoMapX hook logs PSO-style velocity, position, and memory mechanisms without extra objective evaluations.'),
)

# Phase 3 — WOA, HHO, BAT, Firefly, and Cuckoo/Levy profiles
_register_profiles(
    _profile('levy_ja', 'distribution', ('levy/global pollination', 'local pollination/random walk', 'replacement/selection'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs Cuckoo/Levy-pollination search mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('lfd', 'distribution', ('levy/global pollination', 'local pollination/random walk', 'replacement/selection'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs Cuckoo/Levy-pollination search mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('bat_a', 'swarm', ('velocity/frequency update', 'local random walk', 'acceptance/replacement', 'pulse/loudness adaptation'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs BAT acoustic-search mechanisms without extra objective evaluations.'),
    _profile('cco', 'swarm', ('levy/global pollination', 'local pollination/random walk', 'replacement/selection', 'abandonment/reinitialization'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs Cuckoo/Levy-pollination search mechanisms without extra objective evaluations.'),
    _profile('cuckoo_s', 'swarm', ('levy/global pollination', 'local pollination/random walk', 'replacement/selection', 'abandonment/reinitialization'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs Cuckoo/Levy-pollination search mechanisms without extra objective evaluations.'),
    _profile('dvba', 'swarm', ('velocity/frequency update', 'local random walk', 'acceptance/replacement', 'pulse/loudness adaptation'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs BAT acoustic-search mechanisms without extra objective evaluations.'),
    _profile('firefly_a', 'swarm', ('brightness attraction', 'randomization', 'selection/replacement'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs Firefly attraction, randomization, and replacement mechanisms without extra objective evaluations.'),
    _profile('fpa', 'swarm', ('levy/global pollination', 'local pollination/random walk', 'replacement/selection'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs Cuckoo/Levy-pollination search mechanisms without extra objective evaluations.'),
    _profile('hba', 'swarm', ('velocity/frequency update', 'local random walk', 'acceptance/replacement', 'pulse/loudness adaptation'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs BAT acoustic-search mechanisms without extra objective evaluations.'),
    _profile('hho', 'swarm', ('exploration perch/search', 'soft/hard besiege', 'rapid dive/exploitation', 'replacement'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs HHO exploration, besiege, rapid-dive, and replacement mechanisms without extra objective evaluations.'),
    _profile('hi_woa', 'swarm', ('encircling/search', 'spiral exploitation', 'leader guidance', 'replacement'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs WOA/whale-search mechanisms without extra objective evaluations.'),
    _profile('hsaba', 'swarm', ('velocity/frequency update', 'local random walk', 'acceptance/replacement', 'pulse/loudness adaptation'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs BAT acoustic-search mechanisms without extra objective evaluations.'),
    _profile('i_woa', 'swarm', ('encircling/search', 'spiral exploitation', 'leader guidance', 'replacement'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs WOA/whale-search mechanisms without extra objective evaluations.'),
    _profile('laro', 'swarm', ('levy/global pollination', 'local pollination/random walk', 'replacement/selection', 'opposition learning'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs Cuckoo/Levy-pollination search mechanisms without extra objective evaluations.'),
    _profile('nwoa', 'swarm', ('encircling/search', 'spiral exploitation', 'leader guidance', 'replacement'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs WOA/whale-search mechanisms without extra objective evaluations.'),
    _profile('plba', 'swarm', ('velocity/frequency update', 'local random walk', 'acceptance/replacement', 'pulse/loudness adaptation'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs BAT acoustic-search mechanisms without extra objective evaluations.'),
    _profile('saba', 'swarm', ('velocity/frequency update', 'local random walk', 'acceptance/replacement', 'pulse/loudness adaptation'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs BAT acoustic-search mechanisms without extra objective evaluations.'),
    _profile('whale_foa', 'swarm', ('encircling/search', 'spiral exploitation', 'leader guidance', 'replacement', 'fruit-fly sensory search'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs WOA/whale-search mechanisms without extra objective evaluations.'),
    _profile('woa', 'swarm', ('encircling/search', 'spiral exploitation', 'leader guidance', 'replacement'), 'native-family', 'phase_3_woa_hho_bat_firefly_cuckoo_swarm', 'Phase 3 native-family EvoMapX hook logs WOA/whale-search mechanisms without extra objective evaluations.'),
)

# Phase 4 — ABC, ACO, Antlion, and leader-guided swarm profiles
_register_profiles(
    _profile('gmo', 'math', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('jy', 'math', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('qle_sca', 'math', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('sine_cosine_a', 'math', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('csbo', 'nature', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('hgs', 'nature', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('aaa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('aao', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('abco', 'swarm', ('employed/foraging search', 'onlooker/elite selection', 'scout/reinitialization', 'replacement/memorization'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs ABC/bee-search foraging, onlooker, scout, and replacement mechanisms without extra objective evaluations.'),
    _profile('aco', 'swarm', ('solution construction', 'pheromone/model update', 'local/global exploitation', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs ACO/ant-search construction, pheromone/model update, local/global exploitation, and replacement mechanisms without extra objective evaluations.'),
    _profile('acor', 'swarm', ('solution construction', 'pheromone/model update', 'local/global exploitation', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs ACO/ant-search construction, pheromone/model update, local/global exploitation, and replacement mechanisms without extra objective evaluations.'),
    _profile('afsa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('agto', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('aha', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('aho', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ala', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('alo', 'swarm', ('random walk around antlion', 'elite antlion guidance', 'trap/boundary adaptation', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs Ant-lion random walk, elite guidance, trap/boundary adaptation, and replacement mechanisms without extra objective evaluations.'),
    _profile('ao', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('aoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('aoo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('apo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('aro', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('aso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('avoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bboa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bbso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bea', 'swarm', ('employed/foraging search', 'onlooker/elite selection', 'scout/reinitialization', 'replacement/memorization'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs ABC/bee-search foraging, onlooker, scout, and replacement mechanisms without extra objective evaluations.'),
    _profile('bes', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bfo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bka', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bmo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('boa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bono', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bps', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('bsa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('camel', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('capsa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('cat_so', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('cddo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('cdo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('cfoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('chameleon_sa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('chicken_so', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('choa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('coa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('coati_oa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('cockroach_so', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('coot', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('cpo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('crayfish_oa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('csa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('cso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('da', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('dbo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('deo_dolphin', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('dfo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('dhole_oa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('dmoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ecological_cycle_o', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('eefo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('eel_grouper_o', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('eho', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('elk_ho', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('eoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('epc', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('esoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('fda', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('fdo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ffa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ffo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('flo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('foa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('foa_fossa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('fox', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('fss', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('fwa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gazelle_oa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ggo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gja', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gjo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gkso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('go_growth', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('goa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gpoo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gso_glider_snake', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('gto', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('hba_honey', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ho_hippo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('horse_oa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('hus', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('iaro', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('jso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('kha', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('kma', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('loa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('loa_lyrebird', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mbo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mfa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mfo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mgo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mpa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mrfo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('msa_e', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mshoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('mvo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ngo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('nmra', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ofa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ooa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('parrot_o', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('pdo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('pfa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('pfa_polar_fox', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('pko', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('poa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('puma_o', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('rbmo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('rfo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('rhso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('roa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('rsa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('rso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('samso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sbo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sboa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('scso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('seaho', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('serval_oa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sfo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sfoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('shio', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('shio_success', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sho', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('slo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('smo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('so_snake', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('soa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sos', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sparrow_sa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('squirrel_sa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('srsr', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('srsr_robotics', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('ssa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sspider_a', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('sto', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('superb_foa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('tdo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('tlco', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('tsa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('tso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('vcs', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('waoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('who', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('wmqimrfo', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('wooa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('wso', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
    _profile('zoa', 'swarm', ('exploration move', 'exploitation move', 'leader/social guidance', 'selection/replacement'), 'native-family', 'phase_4_abc_aco_antlion_leader_swarm', 'Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.'),
)

# Phase 5 — physics, equilibrium, force-field, wave/flow, and energy-state profiles
_register_profiles(
    _profile('adaptive_eo', 'physics', ('equilibrium pool guidance', 'generation/control-rate update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs equilibrium-pool guidance, control-rate update, position update, and replacement without extra objective evaluations.'),
    _profile('aefa', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('arch_oa', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('aso_atom', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('cdo_chernobyl', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('ceo_cosmic', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('ddao', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('ecpo', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('efo', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('enhanced_two', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('eo', 'physics', ('equilibrium pool guidance', 'generation/control-rate update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs equilibrium-pool guidance, control-rate update, position update, and replacement without extra objective evaluations.'),
    _profile('eso', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('evo', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('fata', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('fla', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('flood_a', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('gea', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('gsa', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('hgso', 'physics', ('equilibrium pool guidance', 'generation/control-rate update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs equilibrium-pool guidance, control-rate update, position update, and replacement without extra objective evaluations.'),
    _profile('ikoa', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('liwo', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('lso_spectrum', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('modified_eo', 'physics', ('equilibrium pool guidance', 'generation/control-rate update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs equilibrium-pool guidance, control-rate update, position update, and replacement without extra objective evaluations.'),
    _profile('mso', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('nro', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('plo', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('rcco', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('rime', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('snow_oa', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('soo', 'physics', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.'),
    _profile('tfwo', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('toc', 'physics', ('fitness_proportional_assignment', 'coriolis_velocity_update', 'windstorm_to_tornado_evolution', 'windstorm_to_thunderstorm_evolution', 'thunderstorm_to_tornado_evolution', 'random_windstorm_formation', 'role_exchange_replacement'), 'native', 'addendum_toc_native', 'Native TOC instrumentation logs paper-level tornado, thunderstorm, windstorm evolution and random windstorm formation using already-computed fitness values. Runtime labels are fully qualified with the toc prefix in evomapx_operator_catalog. Direct operators are windstorm_to_tornado_evolution, windstorm_to_thunderstorm_evolution, thunderstorm_to_tornado_evolution, and random_windstorm_formation. Diagnostic operators are fitness_proportional_assignment, coriolis_velocity_update, and role_exchange_replacement, which do not receive direct objective-improvement credit.'),
    _profile('two', 'physics', ('force/field interaction', 'acceleration/mass update', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.'),
    _profile('wdo', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('wo_wave', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('ydse', 'physics', ('flow/wave propagation', 'physical coefficient update', 'position transport/update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.'),
    _profile('do_dandelion', 'swarm', ('energy/state transition', 'force/equilibrium guidance', 'position update', 'selection/replacement'), 'native-family', 'phase_5_physics_equilibrium_force_field', 'Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
)

# Phase 6 — human/social, teaching/learning, and competition profiles
_register_profiles(
    _profile('aft', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('bro', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('bso', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('btoa', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('cddo_child', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('chio', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('doa', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('dra', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('dream_oa', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('dso', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('eco', 'human', ('teacher/leader phase', 'learner/social phase', 'competition/evaluation', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs teaching/leader phase, learner/social phase, competition/evaluation, and replacement without extra objective evaluations.'),
    _profile('esc', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('fbio', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('gco', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('gska', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('hbo', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('hco', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('heoa', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('hiking_oa', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('ica', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('improved_qsa', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('improved_tlo', 'human', ('teacher/leader phase', 'learner/social phase', 'competition/evaluation', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs teaching/leader phase, learner/social phase, competition/evaluation, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('lco', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('mgoa_market', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('mtbo', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('mvpa', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('petio', 'human', ('teacher/leader phase', 'learner/social phase', 'competition/evaluation', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs teaching/leader phase, learner/social phase, competition/evaluation, and replacement without extra objective evaluations.'),
    _profile('political_o', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('pro', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('qsa', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('saro', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('singer_oa', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('spbo', 'human', ('teacher/leader phase', 'learner/social phase', 'competition/evaluation', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs teaching/leader phase, learner/social phase, competition/evaluation, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('ssdo', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('supply_do', 'human', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.'),
    _profile('thro', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('tlbo', 'human', ('teacher/leader phase', 'learner/social phase', 'competition/evaluation', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs teaching/leader phase, learner/social phase, competition/evaluation, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('toa', 'human', ('teacher/leader phase', 'learner/social phase', 'competition/evaluation', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs teaching/leader phase, learner/social phase, competition/evaluation, and replacement without extra objective evaluations.'),
    _profile('warso', 'human', ('role/team competition', 'social learning/assimilation', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.'),
    _profile('aeo', 'nature', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('enhanced_aeo', 'nature', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('improved_aeo', 'nature', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('modified_aeo', 'nature', ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'), 'native-family', 'phase_6_human_social_teaching_competition', 'Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations. Family synchronized with engine registry metadata.'),
)

# Phase 7 — distribution/model, surrogate, trajectory, and local-search profiles
_register_profiles(
    _profile('cem', 'distribution', ('model sampling', 'elite/model selection', 'model update', 'replacement/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs model sampling, elite/model selection, model update, and incumbent replacement without extra objective evaluations.'),
    _profile('compact_ga', 'distribution', ('model sampling', 'elite/model selection', 'model update', 'replacement/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs model sampling, elite/model selection, model update, and incumbent replacement without extra objective evaluations.'),
    _profile('ego', 'distribution', ('model sampling', 'elite/model selection', 'model update', 'replacement/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs model sampling, elite/model selection, model update, and incumbent replacement without extra objective evaluations.'),
    _profile('pbil', 'distribution', ('model sampling', 'elite/model selection', 'model update', 'replacement/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs model sampling, elite/model selection, model update, and incumbent replacement without extra objective evaluations.'),
    _profile('sopt', 'distribution', ('model sampling', 'elite/model selection', 'model update', 'replacement/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs model sampling, elite/model selection, model update, and incumbent replacement without extra objective evaluations.'),
    _profile('bipop_cmaes', 'evolutionary', ('multivariate sampling', 'elite/parent selection', 'mean/covariance update', 'step-size/restart control'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs CMA-ES/EDA sampling, parent selection, distribution update, and restart/step-size control without extra objective evaluations.'),
    _profile('cmaes', 'evolutionary', ('multivariate sampling', 'elite/parent selection', 'mean/covariance update', 'step-size/restart control'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs CMA-ES/EDA sampling, parent selection, distribution update, and restart/step-size control without extra objective evaluations.'),
    _profile('ipop_cmaes', 'evolutionary', ('cmaes_sampling', 'elite_recombination', 'distribution_update', 'step_size_adaptation', 'population_restart', 'boundary_penalty', 'candidate_injection', 'initialization'), 'native', 'phase_7_distribution_surrogate_trajectory', 'Native IPOP-CMA-ES telemetry logs CMA-ES sampling, elite recombination, covariance/distribution update, step-size adaptation, increasing-population restart, boundary penalty, candidate injection, and initialization without EvoMapX-side objective evaluations. Direct-improvement operators are sampling, elite recombination, and candidate injection; model/restart/boundary operators are diagnostic.'),
    _profile('l2smea', 'evolutionary', ('surrogate screening/modeling', 'evolutionary/swarm variation', 'candidate evaluation', 'selection/model update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate-assisted modeling/screening, variation, candidate evaluation, and selection/model update without extra objective evaluations.'),
    _profile('nlapsmjso_eda', 'evolutionary', ('multivariate sampling', 'elite/parent selection', 'mean/covariance update', 'step-size/restart control'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs CMA-ES/EDA sampling, parent selection, distribution update, and restart/step-size control without extra objective evaluations.'),
    _profile('sacc_eam2', 'evolutionary', ('surrogate screening/modeling', 'evolutionary/swarm variation', 'candidate evaluation', 'selection/model update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate-assisted modeling/screening, variation, candidate evaluation, and selection/model update without extra objective evaluations.'),
    _profile('sade_amss', 'evolutionary', ('surrogate screening/modeling', 'evolutionary/swarm variation', 'candidate evaluation', 'selection/model update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate-assisted modeling/screening, variation, candidate evaluation, and selection/model update without extra objective evaluations.'),
    _profile('sade_atdsc', 'evolutionary', ('surrogate screening/modeling', 'evolutionary/swarm variation', 'candidate evaluation', 'selection/model update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate-assisted modeling/screening, variation, candidate evaluation, and selection/model update without extra objective evaluations.'),
    _profile('sapo', 'evolutionary', ('surrogate screening/modeling', 'evolutionary/swarm variation', 'candidate evaluation', 'selection/model update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate-assisted modeling/screening, variation, candidate evaluation, and selection/model update without extra objective evaluations.'),
    _profile('adam', 'math', ('descent/gradient direction', 'scaling/curvature update', 'parameter step', 'acceptance/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs local mathematical descent/gradient direction, scaling or curvature update, parameter step, and incumbent update without extra objective evaluations.'),
    _profile('bfgs', 'math', ('descent/gradient direction', 'scaling/curvature update', 'parameter step', 'acceptance/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs local mathematical descent/gradient direction, scaling or curvature update, parameter step, and incumbent update without extra objective evaluations.'),
    _profile('et_bo', 'surrogate', ('extra_trees_surrogate_fit', 'random_cutpoint_screening', 'acquisition_search', 'candidate_evaluation', 'incumbent_update'), 'native', 'paper_faithful_et_surrogate', 'Native ET-BO telemetry exposes the paper-specific Extra-Trees surrogate fit and random cut-point screening plus BO acquisition, candidate evaluation, and incumbent update without extra objective evaluations.'),
    _profile('frcg', 'math', ('descent/gradient direction', 'scaling/curvature update', 'parameter step', 'acceptance/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs local mathematical descent/gradient direction, scaling or curvature update, parameter step, and incumbent update without extra objective evaluations.'),
    _profile('gbrt_bo', 'math', ('surrogate fit', 'acquisition search', 'candidate evaluation', 'model/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate fit, acquisition search, candidate evaluation, and model/incumbent update without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('gp_bo', 'math', ('surrogate fit', 'acquisition search', 'candidate evaluation', 'model/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate fit, acquisition search, candidate evaluation, and model/incumbent update without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('rf_bo', 'math', ('surrogate fit', 'acquisition search', 'candidate evaluation', 'model/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate fit, acquisition search, candidate evaluation, and model/incumbent update without extra objective evaluations. Family synchronized with engine registry metadata.'),
    _profile('rmsprop', 'math', ('descent/gradient direction', 'scaling/curvature update', 'parameter step', 'acceptance/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs local mathematical descent/gradient direction, scaling or curvature update, parameter step, and incumbent update without extra objective evaluations.'),
    _profile('sd', 'math', ('descent/gradient direction', 'scaling/curvature update', 'parameter step', 'acceptance/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs local mathematical descent/gradient direction, scaling or curvature update, parameter step, and incumbent update without extra objective evaluations.'),
    _profile('sqp', 'math', ('descent/gradient direction', 'scaling/curvature update', 'parameter step', 'acceptance/incumbent update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs local mathematical descent/gradient direction, scaling or curvature update, parameter step, and incumbent update without extra objective evaluations.'),
    _profile('misaco', 'swarm', ('lhs_initialization', 'acomv_offspring_generation', 'rbf_fit_selection', 'lsbt_fit_selection', 'random_selection', 'sqp_rbf_local_search', 'expensive_candidate_evaluation', 'archive_update'), 'native', 'paper_faithful_misaco', 'Native MiSACO telemetry exposes LHS initialization, ACOMV offspring generation, multisurrogate-assisted selection with RBF/LSBT/random choice, SQP-RBF local search, exact candidate evaluation, and archive update without extra objective evaluations.'),
    _profile('sacoso', 'swarm', ('surrogate screening/modeling', 'evolutionary/swarm variation', 'candidate evaluation', 'selection/model update'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs surrogate-assisted modeling/screening, variation, candidate evaluation, and selection/model update without extra objective evaluations.'),
    _profile('ars', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('basin_hopping', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('grasp', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('hc', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('hsa', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('ils', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('msls', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('mts', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('nmm', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('random_s', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('sa', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('ts', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
    _profile('vns', 'trajectory', ('proposal/neighborhood move', 'move acceptance', 'step-size/adaptation', 'incumbent update/restart'), 'native-family', 'phase_7_distribution_surrogate_trajectory', 'Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.'),
)

# Phase 8 — evolutionary, immune/clonal, genetic/memetic, and multifactorial profiles
_register_profiles(
    _profile('autov', 'evolutionary', ('parent/operator selection', 'crossover/recombination', 'mutation/diversification', 'elitist replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs genetic/memetic selection, recombination, mutation/diversification, and elitist replacement without extra objective evaluations.'),
    _profile('bbo', 'evolutionary', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
    _profile('bspga', 'evolutionary', ('parent/operator selection', 'crossover/recombination', 'mutation/diversification', 'elitist replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs genetic/memetic selection, recombination, mutation/diversification, and elitist replacement without extra objective evaluations.'),
    _profile('bwo', 'evolutionary', ('parent/operator selection', 'crossover/recombination', 'mutation/diversification', 'elitist replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs genetic/memetic selection, recombination, mutation/diversification, and elitist replacement without extra objective evaluations.'),
    _profile('ca', 'evolutionary', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
    _profile('clonalg', 'evolutionary', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
    _profile('cro', 'evolutionary', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
    _profile('ep', 'evolutionary', ('mutation/self-adaptation', 'offspring generation', 'survivor selection', 'strategy-parameter update'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs evolutionary programming/strategy mutation, offspring generation, survivor selection, and strategy-parameter update without extra objective evaluations.'),
    _profile('es', 'evolutionary', ('mutation/self-adaptation', 'offspring generation', 'survivor selection', 'strategy-parameter update'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs evolutionary programming/strategy mutation, offspring generation, survivor selection, and strategy-parameter update without extra objective evaluations.'),
    _profile('fep', 'evolutionary', ('mutation/self-adaptation', 'offspring generation', 'survivor selection', 'strategy-parameter update'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs evolutionary programming/strategy mutation, offspring generation, survivor selection, and strategy-parameter update without extra objective evaluations.'),
    _profile('frofi', 'evolutionary', ('task/skill assignment', 'assortative mating/transfer', 'mutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs multi-factorial or objective-informed evolutionary assignment/transfer, variation, diversification, and replacement without extra objective evaluations.'),
    _profile('ga', 'evolutionary', ('candidate_generation', 'selection', 'breed', 'mutate'), 'native', 'phase_8_evolutionary_immune_genetic_memetic', 'Native GA EvoMapX profile aligned with catalog labels: candidate generation, selection, breeding/crossover, and mutation without extra objective evaluations.'),
    _profile('memetic_a', 'evolutionary', ('parent/operator selection', 'crossover/recombination', 'mutation/diversification', 'elitist replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs genetic/memetic selection, recombination, mutation/diversification, and elitist replacement without extra objective evaluations.'),
    _profile('mfea', 'evolutionary', ('unified random-key initialization', 'factorial evaluation/ranking', 'skill-factor assignment', 'assortative mating and rmp transfer', 'vertical cultural transmission', 'SBX/Gaussian variation', 'scalar-fitness elitist replacement'), 'native', 'phase_8_evolutionary_immune_genetic_memetic', 'Native MFEA telemetry logs unified random-key representation, factorial ranks, skill factors, assortative mating, vertical cultural transmission, selective task evaluation, and elitist scalar-fitness replacement without extra objective evaluations.'),
    _profile('mfea2', 'evolutionary', ('task/skill assignment', 'assortative mating/transfer', 'mutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs multi-factorial or objective-informed evolutionary assignment/transfer, variation, diversification, and replacement without extra objective evaluations.'),
    _profile('mke', 'evolutionary', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
    _profile('nndrea_so', 'evolutionary', ('task/skill assignment', 'assortative mating/transfer', 'mutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs multi-factorial or objective-informed evolutionary assignment/transfer, variation, diversification, and replacement without extra objective evaluations.'),
    _profile('ocro', 'evolutionary', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
    _profile('pcx', 'evolutionary', ('parent/operator selection', 'crossover/recombination', 'mutation/diversification', 'elitist replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs genetic/memetic selection, recombination, mutation/diversification, and elitist replacement without extra objective evaluations.'),
    _profile('ssio_rl', 'evolutionary', ('parent/operator selection', 'crossover/recombination', 'mutation/diversification', 'elitist replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs genetic/memetic selection, recombination, mutation/diversification, and elitist replacement without extra objective evaluations.'),
    _profile('bacterial_colony_o', 'nature', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
    _profile('bco', 'nature', ('affinity/migration selection', 'cloning/reproduction', 'hypermutation/diversification', 'selection/replacement'), 'native-family', 'phase_8_evolutionary_immune_genetic_memetic', 'Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.'),
)

# Phase 9 — nature/biology/growth and mathematical-transform profiles
_register_profiles(
    _profile('cgo', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('circle_sa', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('edo', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('eto', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('gbo', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('gndo', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('info', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('nca', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('noa', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('pss', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('qio', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('run', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('scho', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('ttao', 'math', ('mathematical transform', 'candidate update', 'adaptive control/diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.'),
    _profile('artemisinin_o', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('eao', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('ivya', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('iwo', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('lca', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('lpo', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('moss_go', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('sma', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('tpo', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('tree_seed_a', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('wca', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
    _profile('wutp', 'nature', ('biological growth/foraging', 'reproduction/spread', 'adaptive diversification', 'selection/replacement'), 'native-family', 'phase_9_nature_biology_math_transform', 'Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.'),
)

# Addendum — New Caledonian Crow Learning Algorithm
_register_profiles(
    _profile('nccla', 'swarm', ('vertical_social_learning', 'horizontal_social_learning', 'individual_learning', 'juvenile_reinforcement', 'parent_reinforcement', 'parent_selection'), 'native', 'addendum_nccla', 'Native NCCLA hook logs parent reinforcement and juvenile vertical social, horizontal social, individual learning, and reinforcement transitions. Parent selection is diagnostic metadata based on top-two sorted crows. Family synchronized with engine registry metadata.'),
)

# Addendum — 2024–2025 paper ports
_register_profiles(
    _profile('lea', 'human', ('stimulus_matching', 'value_phase', 'reflection_operation', 'role_phase', 'generational_replacement'), 'native', 'addendum_2025_metaheuristics', 'Native LEA hook labels pairwise value, reflection, and role-phase transformations for EvoMapX lineage. Family synchronized with engine registry metadata.'),
    _profile('agdo', 'math', ('progressive_gradient_momentum_integration', 'dynamic_gradient_interaction', 'trust_region_selection', 'system_optimization_operator'), 'native', 'addendum_2025_metaheuristics', 'Native AGDO hook labels Adam-inspired dynamic gradient interaction and logistic system-optimization replacements without extra objective evaluations.'),
    _profile('dp', 'math', ('delta_operation', 'realtime_learning_vector', 'inertial_learning_vector', 'greedy_selection'), 'native', 'addendum_2025_metaheuristics', 'Native Delta Plus hook labels accepted Delta-operation moves; realtime and inertial learning vectors are computed inside the engine state update.'),
    _profile('ppo', 'swarm', ('escape_sexual_cannibalism_juvenile_generation', 'escape_predation_local_search'), 'native', 'addendum_2025_metaheuristics', 'Native PPO EvoMapX profile aligned with the two catalogued accepted move labels: cannibalism/juvenile generation and predation local search.'),
    _profile('rrto', 'swarm', ('adaptive_step_size_wandering', 'absolute_difference_step', 'boundary_based_step', 'collision_boundary_handling'), 'native', 'addendum_2025_metaheuristics', 'Native RRTO hook labels accepted moves from its three RRT-inspired adaptive step-size strategies.'),
)

# Addendum — Yukthi Opus
_register_profiles(
    _profile('yo', 'trajectory', ('mcmc_burn_in', 'post_burnin_selection', 'mcmc_proposal', 'greedy_refinement', 'simulated_annealing_acceptance', 'blacklist_filter', 'adaptive_reheating', 'elite_update'), 'native', 'addendum_yukthi_opus', 'Native Yukthi Opus instrumentation logs MCMC burn-in, post-burnin selection, MCMC proposal, greedy refinement, and SA acceptance as direct objective-improvement operators. Blacklist filtering, adaptive reheating, and elite updates are diagnostic operators with zero direct improvement unless the engine reports already-computed gains. Direct and diagnostic separation is documented in this note and in result metadata.'),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_evomapx_profile(algorithm_id: str | None, family: str | None = None) -> EvoMapXProfile:
    """Return the declared EvoMapX operator profile for an algorithm ID."""
    aid = str(algorithm_id or "").strip().lower()
    if aid in EVOMAPX_OPERATOR_PROFILES:
        return EVOMAPX_OPERATOR_PROFILES[aid]
    fam = str(family or "unknown").strip().lower() or "unknown"
    ops = FAMILY_DEFAULTS.get(fam, FAMILY_DEFAULTS["unknown"])
    return EvoMapXProfile(algorithm_id=aid or "unknown", family=fam, operators=ops, fidelity="macro")


def get_evomapx_operators(algorithm_id: str | None, family: str | None = None) -> tuple[str, ...]:
    """Return the declared operator labels for an algorithm ID."""
    return get_evomapx_profile(algorithm_id, family).operators


def list_evomapx_profiles() -> list[dict[str, Any]]:
    """Return all declared profiles as dictionaries, sorted by algorithm ID."""
    return [EVOMAPX_OPERATOR_PROFILES[k].to_dict() for k in sorted(EVOMAPX_OPERATOR_PROFILES)]

# Addendum — L-SHADE family engines added from supplied papers.
EVOMAPX_OPERATOR_PROFILES["lshade"] = _profile(
    "lshade",
    "evolutionary",
    (
        "success-history parameter adaptation",
        "current-to-pbest differential mutation",
        "binomial crossover",
        "greedy selection/replacement",
        "archive update",
        "linear population reduction",
    ),
    "native",
    "lshade_family_addendum",
    "Native engine telemetry logs L-SHADE mutation, crossover, selection, archive update, success-history update, and population reduction without instrumentation-side objective evaluations.",
)
EVOMAPX_OPERATOR_PROFILES["mlshade_rl"] = _profile(
    "mlshade_rl",
    "evolutionary",
    (
        "multi-operator differential mutation",
        "covariance/binomial crossover",
        "greedy selection/replacement",
        "strategy probability adaptation",
        "success-history parameter adaptation",
        "restart",
        "local search",
        "linear population reduction",
    ),
    "native",
    "lshade_family_addendum",
    "Native engine telemetry logs mLSHADE-RL mutation-strategy contributions, crossover, selection, strategy/parameter adaptation, restart, local search, archive update, and population reduction. Local-search objective calls are part of the optimizer budget, not EvoMapX instrumentation.",
)

# Addendum — SHADE-family and Secant engines from supplied papers.
_SHADE_SUPPLIED_OPERATORS = (
    "success-history parameter adaptation",
    "current-to-pbest/order-pbest differential mutation",
    "binomial crossover",
    "rank-biased sampling/archive update",
    "greedy selection/replacement",
    "population reduction",
)
for _aid, _name, _extra in [
    ("jso_de", "jSO Differential Evolution", ("weighted pbest mutation",)),
    ("lshade_epsin", "LSHADE-EpSin", ("ensemble sinusoidal adaptation", "Gaussian-walk local search")),
    ("lshade_rsp", "LSHADE-RSP", ("rank-based selective pressure",)),
    ("lshade_spacma", "LSHADE-SPACMA", ("CMA-ES-style elite covariance sampling",)),
    ("ilshade_rsp", "iLSHADE-RSP", ("Cauchy target perturbation",)),
    ("nlshade_lbc", "NL-SHADE-LBC", ("linear parameter-bias change", "nonlinear population reduction")),
    ("nlshade_rsp", "NL-SHADE-RSP", ("nonlinear population reduction", "adaptive archive use")),
    ("nlshade_rsp_midpoint", "NL-SHADE-RSP-Midpoint", ("population/two-cluster midpoint", "midpoint restart trigger")),
    ("rde", "Reconstructed Differential Evolution", ("order-pbest strategy allocation", "strategy-ratio update")),
]:
    EVOMAPX_OPERATOR_PROFILES[_aid] = _profile(
        _aid,
        "evolutionary",
        tuple(dict.fromkeys(_SHADE_SUPPLIED_OPERATORS + tuple(_extra))),
        "native",
        "supplied_shade_family_addendum",
        f"Native engine telemetry logs {_name} operator contributions without EvoMapX-side objective evaluations.",
    )

EVOMAPX_OPERATOR_PROFILES["secant_oa"] = _profile(
    "secant_oa",
    "math",
    (
        "secant derivative-free update",
        "stochastic exploitation around closest/farthest solutions",
        "mutation gate",
        "greedy selection/replacement",
    ),
    "native",
    "supplied_secant_addendum",
    "Native engine telemetry logs SOA secant and stochastic exploitation phases without EvoMapX-side objective evaluations.",
)

# Addendum — RDEx-SOP.
EVOMAPX_OPERATOR_PROFILES["rdex_sop"] = _profile(
    "rdex_sop",
    "evolutionary",
    (
        "standard_branch_mutation",
        "exploitation_biased_mutation",
        "binomial_crossover",
        "cauchy_local_perturbation",
        "greedy_selection",
        "dynamic_pbest_selection",
        "hybrid_rate_update",
        "success_history_update",
        "linear_population_reduction",
        "bound_resampling",
    ),
    "native",
    "rdex_sop_addendum",
    "Native RDEx-SOP telemetry logs the standard and exploitation-biased mutation branches, binomial crossover, accepted Cauchy local perturbation, greedy selection, dynamic pbest pressure, hybrid-rate adaptation, success-history adaptation, population reduction, and bound resampling without EvoMapX-side objective evaluations. Direct-improvement operators are the two mutation branches, crossover, Cauchy local perturbation, and greedy selection; control/update operators are diagnostic.",
)


# Addendum — BIPOP-CMA-ES native operator telemetry.
EVOMAPX_OPERATOR_PROFILES["bipop_cmaes"] = _profile(
    "bipop_cmaes",
    "evolutionary",
    (
        "cmaes_sampling",
        "elite_recombination",
        "distribution_update",
        "step_size_adaptation",
        "large_population_restart",
        "small_population_restart",
        "budget_regime_selection",
        "termination_check",
        "boundary_repair",
        "candidate_injection",
    ),
    "native",
    "bipop_cmaes_native_addendum",
    "Native BIPOP-CMA-ES telemetry logs CMA-ES sampling/recombination, model and step-size updates, budget-controlled large/small restarts, termination checks, bound repairs, and candidate injection without EvoMapX-side objective evaluations.",
)

# Addendum — L-SRTDE native operator telemetry.
EVOMAPX_OPERATOR_PROFILES["l_srtde"] = _profile(
    "l_srtde",
    "evolutionary",
    (
        "success-rate F adaptation",
        "success-rate pbest control",
        "rank-selective pressure",
        "r-new-to-ptop differential mutation",
        "binomial crossover",
        "bound resampling",
        "selection",
        "newest population update",
        "top population update",
        "crossover-memory update",
        "linear population reduction",
    ),
    "native",
    "l_srtde_addendum",
    "Native L-SRTDE telemetry logs success-rate F and pbest control, rank-selective pressure, r-new-to-ptop mutation, binomial crossover, bound resampling, selection, newest/top population updates, crossover-memory adaptation, and linear population reduction without EvoMapX-side objective evaluations.",
)

# ---------------------------------------------------------------------------
# README-synchronized operator overrides
# ---------------------------------------------------------------------------
# Generated from README.md algorithm table.  This block keeps the public
# EvoMapX profile API synchronized with the operator labels shown to users.
# It intentionally preserves each profile's family/fidelity/phase/notes and
# only replaces the operator tuple.
_README_OPERATOR_OVERRIDES: dict[str, tuple[str, ...]] = {
    'aaa': (
        'aaa.recombination',
        'aaa.selection',
        'aaa.adaptation_most_starving_colony_moves_toward',
        'aaa.is_replaced_by_corresponding_cell_biggest',
    ),
    'aao': (
        'aao.adaptive_aquila_guidance',
        'aao.position_update',
        'aao.elite_local_refinement',
        'aao.selection',
    ),
    'abco': (
        'abco.employed',
        'abco.onlooker',
        'abco.scout',
    ),
    'acgwo': (
        'acgwo.selection',
        'acgwo.adaptive_weighted_pack_update',
        'acgwo.alpha_guidance_trial',
        'acgwo.beta_guidance_trial',
        'acgwo.delta_guidance_trial',
    ),
    'aco': (
        'aco.pheromone_weighted_perturbation_in_each_dimension',
    ),
    'acor': (
        'acor.archive_kernel_sampling_update',
    ),
    'adam': (
        'adam.candidate_generation',
        'adam.selection',
        'adam.search_direction',
        'adam.step_acceptance',
        'adam.initialization',
    ),
    'adaptive_eo': (
        'adaptive_eo.selection',
        'adaptive_eo.adaptive_local_refinement',
        'adaptive_eo.equilibrium_pool_guided_update',
    ),
    'aefa': (
        'aefa.electric_field_force_update',
    ),
    'aeo': (
        'aeo.selection',
        'aeo.consumer_decomposer_update',
        'aeo.production_worst_agent',
    ),
    'aesspso': (
        'aesspso.adaptive_velocity_position_update',
    ),
    'afsa': (
        'afsa.leap',
    ),
    'aft': (
        'aft.best_guided_tracking',
        'aft.random_treasure_search',
        'aft.opposition_tracking',
    ),
    'agdo': (
        'agdo.progressive_gradient_momentum_dynamic_interaction',
        'agdo.system_optimization_operator',
    ),
    'agto': (
        'agto.migration',
        'agto.exploration',
        'agto.state_update',
        'agto.exploitation',
    ),
    'aha': (
        'aha.guided_foraging',
        'aha.territorial_foraging',
        'aha.migration',
    ),
    'aho': (
        'aho.single_shot_prey_projection',
        'aho.double_shot_prey_projection',
        'aho.levy_stagnation_rescue',
    ),
    'aiw_pso': (
        'aiw_pso.position_update',
        'aiw_pso.selection',
        'aiw_pso.velocity_update',
        'aiw_pso.elite_local_refinement',
    ),
    'ala': (
        'ala.high_energy_digging_walk',
        'ala.high_energy_lemming_migration',
        'ala.low_energy_spiral_foraging',
        'ala.low_energy_levy_escape',
    ),
    'alo': (
        'alo.random_walk',
        'alo.state_update',
        'alo.candidate_generation',
        'alo.selection',
        'alo.combine',
    ),
    'ao': (
        'ao.high_soar_vertical_stoop',
        'ao.contour_flight_exploration',
        'ao.low_flight_attack',
        'ao.walk_and_grab_prey',
    ),
    'aoa': (
        'aoa.arithmetic_operator_position_update',
    ),
    'aoo': (
        'aoo.mean_wind_animation_update',
        'aoo.best_wind_animation_update',
        'aoo.self_wind_animation_update',
        'aoo.rolling_levy_animation_update',
        'aoo.projectile_jump_animation_update',
    ),
    'apo': (
        'apo.dormancy_random_restart',
        'apo.dormancy_local_perturbation',
        'apo.foraging_reproduction_update',
        'apo.autotrophic_foraging_update',
    ),
    'arch_oa': (
        'arch_oa.archimedes_density_volume_acceleration_update',
    ),
    'aro': (
        'aro.detour_foraging',
        'aro.random_hiding',
    ),
    'ars': (
        'ars.small_step',
        'ars.large_step',
    ),
    'artemisinin_o': (
        'artemisinin_o.self_growth_update',
        'artemisinin_o.best_growth_update',
        'artemisinin_o.differential_mutation_update',
        'artemisinin_o.self_reset_mutation',
        'artemisinin_o.best_reset_mutation',
        'artemisinin_o.boundary_best_repair',
    ),
    'aso': (
        'aso.anarchic_social_position_update',
    ),
    'aso_atom': (
        'aso_atom.do_not_move_current_elites_unless',
    ),
    'autov': (
        'autov.learned_variation_operator_update',
    ),
    'avoa': (
        'avoa.exploration_vulture_soaring',
        'avoa.random_roost_exploration',
        'avoa.convergent_competition_exploitation',
        'avoa.levy_food_exploitation',
        'avoa.aggressive_siege_exploitation',
        'avoa.spiral_siege_exploitation',
    ),
    'bacterial_colony_o': (
        'bacterial_colony_o.migration',
        'bacterial_colony_o.position_update',
        'bacterial_colony_o.recombination',
        'bacterial_colony_o.selection',
        'bacterial_colony_o.current_colony_best_accept_only_it',
        'bacterial_colony_o.implementation_but_only_as_bounded_macro',
    ),
    'basin_hopping': (
        'basin_hopping.update',
    ),
    'bat_a': (
        'bat_a.candidate_generation',
        'bat_a.selection',
        'bat_a.force_or_velocity_update',
        'bat_a.position_update',
        'bat_a.acceptance',
        'bat_a.state_update',
        'bat_a.initialization',
    ),
    'bbo': (
        'bbo.migration_mutation_selection_update',
    ),
    'bboa': (
        'bboa.selection',
        'bboa.2_sniffing',
        'bboa.pedal_marking_update',
    ),
    'bbso': (
        'bbso.coordinated_following_trial',
        'bbso.self_following_trial',
    ),
    'bco': (
        'bco.swim_refinement_update',
    ),
    'bea': (
        'bea.elite_site_neighbourhood_search',
        'bea.selected_site_neighbourhood_search',
        'bea.scout_site_global_search',
    ),
    'bes': (
        'bes.candidate_generation',
        'bes.selection',
        'bes.candidate_search',
    ),
    'bfgs': (
        'bfgs.update',
    ),
    'bfo': (
        'bfo.chemotaxis_tumble_update',
        'bfo.selection',
    ),
    'bipop_cmaes': (
        'bipop_cmaes.cmaes_sampling',
        'bipop_cmaes.elite_recombination',
        'bipop_cmaes.distribution_update',
        'bipop_cmaes.step_size_adaptation',
        'bipop_cmaes.large_population_restart',
        'bipop_cmaes.small_population_restart',
        'bipop_cmaes.budget_regime_selection',
        'bipop_cmaes.termination_check',
        'bipop_cmaes.boundary_repair',
        'bipop_cmaes.candidate_injection',
    ),
    'bka': (
        'bka.sine_soaring_update',
        'bka.random_soaring_update',
        'bka.peer_repulsion_cauchy_update',
        'bka.leader_attraction_cauchy_update',
    ),
    'bmo': (
        'bmo.barnacle_recombination',
        'bmo.random_barnacle_drift',
    ),
    'boa': (
        'boa.global_fragrance_attraction',
        'boa.local_fragrance_random_walk',
    ),
    'bono': (
        'bono.social_guidance_phase',
        'bono.exploratory_directional_move',
    ),
    'bps': (
        'bps.long_distance_flight',
        'bps.local_tree_movement',
        'bps.best_tree_attraction',
    ),
    'bro': (
        'bro.find_nearest_neighbour',
        'bro.battle_damage_relocation_update',
        'bro.selection',
    ),
    'bsa': (
        'bsa.foraging_flight_update',
        'bsa.vigilance_flight_update',
        'bsa.producer_guided_flight_update',
        'bsa.scrounger_random_flight_update',
    ),
    'bso': (
        'bso.single_cluster_center_idea',
        'bso.single_cluster_member_idea',
        'bso.empty_cluster_center_idea',
        'bso.two_cluster_center_blend',
        'bso.two_cluster_member_blend',
    ),
    'bspga': (
        'bspga.binary_partition_tree_variation_update',
    ),
    'btoa': (
        'btoa.position_update',
        'btoa.selection',
        'btoa.defensive_play_refinement',
        'btoa.dynamic_position_candidate',
        'btoa.offensive_play_update',
    ),
    'bwo': (
        'bwo.crossover',
        'bwo.mutation',
        'bwo.procreation',
        'bwo.candidate_generation',
        'bwo.selection',
    ),
    'ca': (
        'ca.cultural_belief_guided_update',
    ),
    'camel': (
        'camel.endurance_temperature_update',
        'camel.selection',
    ),
    'capsa': (
        'capsa.jumping_global_motion',
        'capsa.long_jump_global_motion',
        'capsa.velocity_swing_update',
        'capsa.best_swing_update',
        'capsa.velocity_memory_update',
        'capsa.random_tree_leap',
        'capsa.group_following_update',
    ),
    'cat_so': (
        'cat_so.seeking_mode_expansive_copy_update',
        'cat_so.seeking_mode_contracting_copy_update',
        'cat_so.tracing_mode_velocity_update',
    ),
    'cco': (
        'cco.candidate_search',
        'cco.selection',
        'cco.candidate_generation',
    ),
    'cddo': (
        'cddo.cheetah_chase_position_update',
    ),
    'cddo_child': (
        'cddo_child.child_drawing_development_update',
    ),
    'cdo': (
        'cdo.alpha_cheetah_attack_component',
        'cdo.beta_cheetah_attack_component',
        'cdo.gamma_cheetah_attack_component',
    ),
    'cdo_chernobyl': (
        'cdo_chernobyl.alpha_beta_gamma_radiation_update',
        'cdo_chernobyl.cdo_chernobyl_position_update',
        'cdo_chernobyl.selection',
    ),
    'cem': (
        'cem.model_sampling_elite_distribution_update',
    ),
    'ceo_cosmic': (
        'ceo_cosmic.exploration_attraction_alignment',
        'ceo_cosmic.global_collision_update',
        'ceo_cosmic.resonance_refinement_update',
    ),
    'cfoa': (
        'cfoa.individual_foraging_update',
        'cfoa.group_foraging_update',
        'cfoa.late_gaussian_capture_update',
    ),
    'cg_gwo': (
        'cg_gwo.selection',
        'cg_gwo.elite_local_refinement',
        'cg_gwo.leader_guided_population_update',
    ),
    'cgo': (
        'cgo.current_seed_attractor',
        'cgo.best_seed_attractor',
        'cgo.mean_group_seed_attractor',
        'cgo.dimension_mutation_seed',
    ),
    'chameleon_sa': (
        'chameleon_sa.social_pbest_gbest_update',
        'chameleon_sa.random_global_exploration',
    ),
    'chaotic_gwo': (
        'chaotic_gwo.selection',
        'chaotic_gwo.elite_local_refinement',
        'chaotic_gwo.leader_guided_population_update',
    ),
    'chicken_so': (
        'chicken_so.selection',
        'chicken_so.chicken_so_semantic_update',
    ),
    'chio': (
        'chio.infected_contact_update',
        'chio.susceptible_contact_update',
        'chio.immune_contact_update',
    ),
    'choa': (
        'choa.chimp_hunting_position_update',
    ),
    'circle_sa': (
        'circle_sa.circle_position_update',
    ),
    'clonalg': (
        'clonalg.candidate_generation',
        'clonalg.selection',
        'clonalg.cloning',
        'clonalg.hypermutation',
    ),
    'cmaes': (
        'cmaes.offspring_sampling',
        'cmaes.parent_selection',
        'cmaes.evolution_path_update',
        'cmaes.covariance_update',
        'cmaes.step_size_update',
        'cmaes.boundary_repair',
        'cmaes.initialization',
        'cmaes.candidate_injection',
    ),
    'coa': (
        'coa.alpha_social_condition_update',
        'coa.tendency_social_condition_update',
        'coa.pup_birth_replacement',
        'coa.migration_exchange',
    ),
    'coati_oa': (
        'coati_oa.candidate_generation',
        'coati_oa.selection',
        'coati_oa.behavioral_move',
    ),
    'cockroach_so': (
        'cockroach_so.dispersal',
        'cockroach_so.replacement',
        'cockroach_so.state_update',
    ),
    'compact_ga': (
        'compact_ga.model_update',
        'compact_ga.sampling',
        'compact_ga.selection',
        'compact_ga.state_update',
        'compact_ga.compact_genetic_algorithm_semantic_update',
    ),
    'coot': (
        'coot.chain_movement_update',
    ),
    'cpo': (
        'cpo.aroma_luring_trial',
        'cpo.predation_feeding_trial',
    ),
    'crayfish_oa': (
        'crayfish_oa.high_temperature_shelter_update',
        'crayfish_oa.high_temperature_competition_update',
        'crayfish_oa.food_competition_update',
        'crayfish_oa.food_intake_update',
    ),
    'cro': (
        'cro.broadcast_spawning_recombination',
        'cro.brooding_clone_mutation',
        'cro.depredation_random_reseeding',
    ),
    'csa': (
        'csa.memory_following_update',
        'csa.awareness_random_relocation',
        'csa.mixed_memory_random_update',
    ),
    'csbo': (
        'csbo.systolic',
        'csbo.diastolic',
    ),
    'cso': (
        'cso.mean_all_positions',
    ),
    'cuckoo_s': (
        'cuckoo_s.levy_flight',
        'cuckoo_s.replacement',
        'cuckoo_s.candidate_generation',
        'cuckoo_s.selection',
    ),
    'da': (
        'da.neighbour_alignment_update',
        'da.levy_flight_exploration',
        'da.food_enemy_swarm_update',
    ),
    'dbo': (
        'dbo.foraging',
        'dbo.selection',
        'dbo.state_update',
        'dbo.ball_rolling_dance_update',
    ),
    'ddao': (
        'ddao.exploration',
        'ddao.selection',
        'ddao.state_update',
        'ddao.dynamic_annealed_refinement_update',
    ),
    'de': (
        'de.mutation',
        'de.crossover',
        'de.selection',
        'de.bound_repair',
    ),
    'deo_dolphin': (
        'deo_dolphin.elite_reference_echo_guidance',
        'deo_dolphin.elite_jitter_echo_guidance',
        'deo_dolphin.peer_reference_echo_guidance',
        'deo_dolphin.peer_jitter_echo_guidance',
    ),
    'dfo': (
        'dfo.dispersive_fly_neighbour_update',
        'dfo.elite_disturbance_update',
        'dfo.selection',
    ),
    'dhole_oa': (
        'dhole_oa.searching_stage',
        'dhole_oa.encircling_stage',
        'dhole_oa.large_prey_attack',
        'dhole_oa.small_prey_kill',
    ),
    'dmoa': (
        'dmoa.selection',
        'dmoa.3_baby_sitter_eviction',
        'dmoa.scalar_broadcast',
        'dmoa.scout_phase',
    ),
    'do_dandelion': (
        'do_dandelion.rising_seed_phase',
        'do_dandelion.descent_diffusion_phase',
        'do_dandelion.elite_landing_phase',
        'do_dandelion.candidate_generation',
        'do_dandelion.selection',
    ),
    'doa': (
        'doa.hunting',
        'doa.search',
        'doa.state_update',
        'doa.exploitation_move',
        'doa.replacement',
    ),
    'dp': (
        'dp.delta_operation',
    ),
    'dra': (
        'dra.selection',
        'dra.dialectic_interaction_update',
    ),
    'dream_oa': (
        'dream_oa.dream_generation_refinement_update',
    ),
    'ds_gwo': (
        'ds_gwo.selection',
        'ds_gwo.elite_local_refinement',
        'ds_gwo.leader_guided_population_update',
    ),
    'dso': (
        'dso.deep_sleep_decay_update',
        'dso.slow_wave_recovery_update',
    ),
    'dvba': (
        'dvba.force_or_velocity_update',
        'dvba.position_update',
        'dvba.random_walk',
        'dvba.state_update',
        'dvba.candidate_generation',
        'dvba.selection',
    ),
    'eao': (
        'eao.sinusoidal_best_substrate_update',
        'eao.vector_scaled_differential_substrate_update',
        'eao.scalar_scaled_differential_substrate_update',
    ),
    'eco': (
        'eco.primary_competition_update',
        'eco.sine_cosine_learning_update',
        'eco.best_weighted_learning_update',
        'eco.levy_exam_update',
    ),
    'ecological_cycle_o': (
        'ecological_cycle_o.selection',
        'ecological_cycle_o.ecological_cycle_transition_update',
        'ecological_cycle_o.eval_accept_group',
    ),
    'ecpo': (
        'ecpo.electric_charge_random_perturbation',
    ),
    'edo': (
        'edo.distribution_update',
        'edo.candidate_generation',
        'edo.state_update',
    ),
    'eefo': (
        'eefo.interaction_migration',
        'eefo.resting_area_update',
        'eefo.levy_hunting_update',
        'eefo.prey_capture_update',
    ),
    'eel_grouper_o': (
        'eel_grouper_o.eel_weighted_hunting_update',
        'eel_grouper_o.grouper_weighted_hunting_update',
    ),
    'efo': (
        'efo.electromagnetic_field_update',
        'efo.random_field_reinitialization',
        'efo.dimension_reset_mutation',
    ),
    'ego': (
        'ego.expected_improvement_candidate_generation',
    ),
    'eho': (
        'eho.long_range_clan_best_guided_update',
        'eho.short_range_clan_best_guided_update',
        'eho.matriarch_center_update',
        'eho.separating_random_relocation',
    ),
    'elk_ho': (
        'elk_ho.selection',
        'elk_ho.family_mating_position_update',
    ),
    'enhanced_aeo': (
        'enhanced_aeo.selection',
        'enhanced_aeo.ecosystem_producer_consumer_update',
        'enhanced_aeo.enhanced_decomposition_refinement',
    ),
    'enhanced_two': (
        'enhanced_two.candidate_generation',
        'enhanced_two.selection',
        'enhanced_two.force_update',
        'enhanced_two.state_update',
        'enhanced_two.initialization',
    ),
    'eo': (
        'eo.equilibrium_position_update',
    ),
    'eoa': (
        'eoa.crossover',
        'eoa.state_update',
        'eoa.mutation',
        'eoa.candidate_generation',
        'eoa.selection',
        'eoa.reproduction',
    ),
    'ep': (
        'ep.parent_survivor',
        'ep.large_strategy_mutation_offspring',
        'ep.small_strategy_mutation_offspring',
    ),
    'epc': (
        'epc.spiral_attraction_update',
        'epc.thermal_mutation_update',
    ),
    'er_gwo': (
        'er_gwo.selection',
        'er_gwo.elite_local_refinement',
        'er_gwo.leader_guided_population_update',
    ),
    'es': (
        'es.parent_survivor',
        'es.large_step_mutation_offspring',
        'es.small_step_mutation_offspring',
    ),
    'esc': (
        'esc.escape_from_worst_update',
        'esc.move_toward_best_update',
        'esc.random_exploration_update',
    ),
    'eso': (
        'eso.electric_storm_field_update',
    ),
    'esoa': (
        'esoa.behavioral_move',
        'esoa.selection',
        'esoa.egret_sit_and_wait_update',
    ),
    'et_bo': (
        'et_bo.extra_trees_surrogate_fit',
        'et_bo.random_cutpoint_screening',
        'et_bo.acquisition_search',
        'et_bo.candidate_evaluation',
        'et_bo.incumbent_update',
    ),
    'eto': (
        'eto.exponential_orbit_update',
        'eto.trigonometric_orbit_update',
    ),
    'evo': (
        'evo.exploration',
        'evo.state_update',
        'evo.exploitation',
    ),
    'ex_gwo': (
        'ex_gwo.selection',
        'ex_gwo.elite_local_refinement',
        'ex_gwo.leader_guided_population_update',
    ),
    'fata': (
        'fata.random_refraction_update',
        'fata.best_refraction_update',
        'fata.peer_refraction_update',
    ),
    'fbio': (
        'fbio.candidate_generation',
        'fbio.selection',
        'fbio.exploration',
    ),
    'fda': (
        'fda.downhill_flow_direction_update',
        'fda.neighbour_flow_direction_update',
        'fda.elite_flow_direction_update',
    ),
    'fdo': (
        'fdo.fitness_weighted_pace_update',
        'fdo.best_guided_position_update',
        'fdo.selection',
    ),
    'fep': (
        'fep.fast_mutation_tournament_selection_update',
    ),
    'ffa': (
        'ffa.fruitfly_smell_search_update',
    ),
    'ffo': (
        'ffo.exploration',
        'ffo.state_update',
        'ffo.exploitation',
    ),
    'firefly_a': (
        'firefly_a.attraction_dominant_move',
        'firefly_a.randomization_dominant_move',
    ),
    'fla': (
        'fla.forward_diffusion_transfer',
        'fla.source_fluid_diffusion',
        'fla.receiver_fluid_diffusion',
        'fla.reverse_diffusion_transfer',
        'fla.equilibrium_exploitation_update',
    ),
    'flo': (
        'flo.update',
    ),
    'flood_a': (
        'flood_a.flood_flow_direction_update',
        'flood_a.flood_recession_refinement_update',
        'flood_a.selection',
    ),
    'foa': (
        'foa.local_seeding_growth_update',
        'foa.selection',
    ),
    'foa_fossa': (
        'foa_fossa.prey_pursuit_update',
        'foa_fossa.defensive_escape_update',
    ),
    'fox': (
        'fox.prey_jump_exploitation',
        'fox.current_to_random_walk_update',
        'fox.best_radius_random_walk',
    ),
    'fpa': (
        'fpa.global_levy_pollination',
        'fpa.local_pollination',
    ),
    'frcg': (
        'frcg.update',
    ),
    'frofi': (
        'frofi.current_to_rand_de',
        'frofi.rand_to_best_crossover_de',
        'frofi.no_crossover_de',
        'frofi.targeted_mutation',
    ),
    'fss': (
        'fss.collective_volitive_movement',
        'fss.selection',
    ),
    'fuzzy_gwo': (
        'fuzzy_gwo.selection',
        'fuzzy_gwo.elite_local_refinement',
        'fuzzy_gwo.leader_guided_population_update',
    ),
    'fwa': (
        'fwa.selection',
        'fwa.state_update',
    ),
    'ga': (
        'ga.candidate_generation',
        'ga.selection',
        'ga.breed',
        'ga.mutate',
    ),
    'gazelle_oa': (
        'gazelle_oa.brownian_foraging_update',
        'gazelle_oa.levy_elite_transition_update',
        'gazelle_oa.levy_foraging_update',
        'gazelle_oa.random_patch_avoidance_update',
        'gazelle_oa.peer_difference_escape_update',
    ),
    'gbo': (
        'gbo.gradient_search_rule_update',
        'gbo.local_escaping_operator_update',
    ),
    'gbrt_bo': (
        'gbrt_bo.update',
    ),
    'gco': (
        'gco.dark_zone_mutation_update',
    ),
    'gea': (
        'gea.neighbour_geyser_eruption_update',
        'gea.pressure_random_eruption_update',
    ),
    'ggo': (
        'ggo.initialization',
        'ggo.dynamic_group_update',
        'ggo.exploration_leader_move_eq1',
        'ggo.exploration_paddling_mutation_eq2',
        'ggo.exploration_spiral_move_eq4',
        'ggo.flock_local_search_eq7',
        'ggo.exploitation_sentry_guidance_eq5_6',
        'ggo.elitist_selection',
        'ggo.boundary_repair',
        'ggo.role_shuffle',
        'ggo.stagnation_group_boost',
        'ggo.candidate_injection',
    ),
    'gja': (
        'gja.levy_wall_search',
        'gja.gaussian_wall_search',
    ),
    'gjo': (
        'gjo.male_female_exploitation',
        'gjo.male_female_exploration',
    ),
    'gkso': (
        'gkso.genghis_khan_crossover_exploration',
        'gkso.shark_hunting_pso_update',
    ),
    'gmo': (
        'gmo.marketing_guidance_update',
    ),
    'gndo': (
        'gndo.generalized_normal_local_update',
        'gndo.difference_vector_global_update',
    ),
    'go_growth': (
        'go_growth.growth_phase_update',
        'go_growth.maturity_phase_update',
        'go_growth.selection',
    ),
    'goa': (
        'goa.grasshopper_social_force_update',
    ),
    'gp_bo': (
        'gp_bo.update',
    ),
    'gpoo': (
        'gpoo.octopus_tentacle_prey_position_update',
    ),
    'gpso': (
        'gpso.velocity_position_update',
    ),
    'grasp': (
        'grasp.update',
    ),
    'gsa': (
        'gsa.gravitational_force_acceleration_update',
    ),
    'gska': (
        'gska.gaining_sharing_knowledge_update',
    ),
    'gso': (
        'gso.glowworm_luciferin_movement_update',
    ),
    'gso_glider_snake': (
        'gso_glider_snake.glider_snake_position_update',
    ),
    'gto': (
        'gto.candidate_search',
        'gto.selection',
        'gto.candidate_generation',
        'gto.behavioral_move',
    ),
    'gwo': (
        'gwo.alpha_guidance',
        'gwo.beta_guidance',
        'gwo.delta_guidance',
        'gwo.position_update',
    ),
    'gwo_woa': (
        'gwo_woa.selection',
        'gwo_woa.elite_local_refinement',
        'gwo_woa.leader_guided_population_update',
    ),
    'hba': (
        'hba.bat_frequency_movement',
        'hba.de_local_search',
    ),
    'hba_honey': (
        'hba_honey.digging_phase_update',
        'hba_honey.honey_phase_update',
    ),
    'hbo': (
        'hbo.heap_rank_pressure_update',
    ),
    'hc': (
        'hc.update',
    ),
    'hco': (
        'hco.conception_growth_update',
    ),
    'hde': (
        'hde.candidate_search',
        'hde.selection',
        'hde.differential_evolution_update',
    ),
    'heoa': (
        'heoa.elite_local_refinement',
        'heoa.learner_levy_best_attraction',
        'heoa.explorer_centroid_escape',
        'heoa.follower_best_contraction',
        'heoa.risk_taker_best_sampling',
    ),
    'hgs': (
        'hgs.random_hunger_exploration',
        'hgs.hunger_weighted_approach',
        'hgs.hunger_weighted_retreat',
    ),
    'hgso': (
        'hgso.cluster_best_solubility_update',
        'hgso.global_best_solubility_update',
        'hgso.worst_agent_random_reset',
    ),
    'hho': (
        'hho.exploration',
        'hho.soft_besiege',
        'hho.hard_besiege',
        'hho.soft_besiege_rapid_dive',
        'hho.hard_besiege_rapid_dive',
        'hho.levy_rapid_dive_refinement',
    ),
    'hi_woa': (
        'hi_woa.selection',
        'hi_woa.elite_local_refinement',
        'hi_woa.whale_position_update',
    ),
    'hiking_oa': (
        'hiking_oa.hiking_slope_velocity_update',
    ),
    'ho_hippo': (
        'ho_hippo.exploitation',
        'ho_hippo.selection',
        'ho_hippo.state_update',
        'ho_hippo.group_defense_position_update',
        'ho_hippo.predator_defense_update',
        'ho_hippo.river_pond_position_update',
    ),
    'horse_oa': (
        'horse_oa.dominant_stallion_update',
        'horse_oa.experienced_horse_social_update',
        'horse_oa.middle_rank_grazing_update',
        'horse_oa.foal_exploration_update',
    ),
    'hsa': (
        'hsa.harmony_memory_improvisation_update',
    ),
    'hsaba': (
        'hsaba.local_bat_random_walk',
        'hsaba.velocity_bat_update',
        'hsaba.differential_evolution_refinement',
    ),
    'hus': (
        'hus.update',
    ),
    'i_gwo': (
        'i_gwo.selection',
        'i_gwo.alpha_guidance_trial',
        'i_gwo.beta_guidance_trial',
        'i_gwo.delta_guidance_trial',
        'i_gwo.mean_leader_position_update',
    ),
    'i_woa': (
        'i_woa.polynomial_breeding_refinement',
    ),
    'iagwo': (
        'iagwo.adaptive_alpha_beta_delta_update',
    ),
    'iaro': (
        'iaro.improved_rabbit_global_update',
        'iaro.elite_local_refinement',
        'iaro.selection',
    ),
    'ica': (
        'ica.assimilation',
        'ica.imperialist_revolution',
        'ica.colony_revolution',
        'ica.intra_empire_competition',
    ),
    'ikoa': (
        'ikoa.selection',
        'ikoa.assignment_matching_position_update',
        'ikoa.improved_matching_refinement_update',
    ),
    'ils': (
        'ils.update',
    ),
    'ilshade': (
        'ilshade.current_to_pbest_mutation',
        'ilshade.binomial_crossover',
        'ilshade.greedy_selection',
        'ilshade.external_archive_update',
        'ilshade.success_history_update',
        'ilshade.linear_population_size_reduction',
        'ilshade.pbest_schedule',
        'ilshade.fixed_memory_cell',
        'ilshade.early_parameter_control',
        'ilshade.midpoint_bound_repair',
    ),
    'ilshade_rsp': (
        'ilshade_rsp.mutation',
        'ilshade_rsp.crossover',
        'ilshade_rsp.selection',
        'ilshade_rsp.archive_update',
        'ilshade_rsp.success_history_update',
        'ilshade_rsp.population_reduction',
        'ilshade_rsp.rank_selective_pressure',
        'ilshade_rsp.weighted_pbest_scaling',
        'ilshade_rsp.cauchy_target_perturbation',
    ),
    'imode': (
        'imode.candidate_generation',
        'imode.selection',
        'imode.state_update',
        'imode.initialization',
        'imode.mutation',
        'imode.crossover',
    ),
    'improved_aeo': (
        'improved_aeo.selection',
        'improved_aeo.ecosystem_producer_consumer_update',
        'improved_aeo.improved_decomposition_refinement',
    ),
    'improved_qsa': (
        'improved_qsa.selection',
        'improved_qsa.queue_business_one_update',
        'improved_qsa.queue_business_two_refinement',
    ),
    'improved_tlo': (
        'improved_tlo.selection',
        'improved_tlo.elite_local_refinement',
        'improved_tlo.teacher_learner_population_update',
    ),
    'incremental_gwo': (
        'incremental_gwo.selection',
        'incremental_gwo.elite_local_refinement',
        'incremental_gwo.leader_guided_population_update',
    ),
    'info': (
        'info.best_weighted_mean_rule',
        'info.random_weighted_mean_rule',
    ),
    'iobl_gwo': (
        'iobl_gwo.selection',
        'iobl_gwo.elite_local_refinement',
        'iobl_gwo.leader_guided_population_update',
    ),
    'ipop_cmaes': (
        'ipop_cmaes.initialization',
        'ipop_cmaes.cmaes_sampling',
        'ipop_cmaes.elite_recombination',
        'ipop_cmaes.distribution_update',
        'ipop_cmaes.population_restart',
        'ipop_cmaes.boundary_penalty',
        'ipop_cmaes.candidate_injection',
    ),
    'ivya': (
        'ivya.neighbor_growth_update',
        'ivya.best_growth_update',
    ),
    'iwo': (
        'iwo.seed_dispersal_colonization_update',
    ),
    'jade': (
        'jade.candidate_generation',
        'jade.selection',
        'jade.mutation',
        'jade.crossover',
        'jade.initialization',
    ),
    'jde': (
        'jde.de_trial',
        'jde.f_self_adaptation_trial',
        'jde.cr_self_adaptation_trial',
        'jde.f_cr_self_adaptation_trial',
    ),
    'jso': (
        'jso.ocean_current_swarm_motion_update',
    ),
    'jso_de': (
        'jso_de.mutation',
        'jso_de.weighted_pbest_scaling',
        'jso_de.crossover',
        'jso_de.selection',
        'jso_de.archive_update',
        'jso_de.success_history_update',
        'jso_de.population_reduction',
        'jso_de.bound_resampling',
    ),
    'jy': (
        'jy.best_away_from_worst_update',
    ),
    'kha': (
        'kha.crossover',
        'kha.diffusion',
        'kha.mutation',
        'kha.selection',
        'kha.state_update',
        'kha.induced_movement_update',
    ),
    'kma': (
        'kma.update',
    ),
    'l2smea': (
        'l2smea.update',
    ),
    'l_srtde': (
        'l_srtde.success_rate_f_adaptation',
        'l_srtde.success_rate_pbest_control',
        'l_srtde.rank_selective_pressure',
        'l_srtde.r_new_to_ptop_mutation',
        'l_srtde.binomial_crossover',
        'l_srtde.bound_resampling',
        'l_srtde.selection',
        'l_srtde.newest_population_update',
        'l_srtde.top_population_update',
        'l_srtde.crossover_memory_update',
        'l_srtde.linear_population_reduction',
    ),
    'laro': (
        'laro.candidate_search',
        'laro.selection',
        'laro.candidate_generation',
        'laro.initialization',
    ),
    'lca': (
        'lca.best_cell_replication',
        'lca.peer_lateral_invasion',
        'lca.angiogenesis_mutation',
    ),
    'lco': (
        'lco.life_choice_boundary_reflection_update',
    ),
    'lea': (
        'lea.reflection_operation',
        'lea.value_phase_reflection_operation',
        'lea.value_phase_role_phase',
    ),
    'levy_ja': (
        'levy_ja.candidate_search',
        'levy_ja.selection',
        'levy_ja.candidate_generation',
        'levy_ja.initialization',
    ),
    'lfd': (
        'lfd.levy_flight_search',
    ),
    'liwo': (
        'liwo.breeze_spiral_translation',
        'liwo.strong_wind_displacement',
    ),
    'loa': (
        'loa.nomad_roaming_update',
        'loa.pride_mating_recombination',
        'loa.pride_leader_roaming_update',
        'loa.nomad_roaming_update.mutation',
        'loa.pride_mating_recombination.mutation',
        'loa.pride_leader_roaming_update.mutation',
        'loa.territorial_takeover_exchange',
    ),
    'loa_lyrebird': (
        'loa_lyrebird.better_bird_imitation_update',
        'loa_lyrebird.escape_step_update',
    ),
    'lpo': (
        'lpo.lichen_growth_propagation_update',
    ),
    'lshade': (
        'lshade.parameter_sampling',
        'lshade.current_to_pbest_mutation',
        'lshade.midpoint_bound_repair',
        'lshade.binomial_crossover',
        'lshade.greedy_selection',
        'lshade.external_archive_update',
        'lshade.success_history_update',
        'lshade.linear_population_size_reduction',
    ),
    'lshade_cnepsin': (
        'lshade_cnepsin.cn_epsin_mutation_crossover_selection',
    ),
    'lshade_epsin': (
        'lshade_epsin.mutation',
        'lshade_epsin.crossover',
        'lshade_epsin.selection',
        'lshade_epsin.archive_update',
        'lshade_epsin.success_history_update',
        'lshade_epsin.population_reduction',
        'lshade_epsin.sinusoidal_decreasing_f',
        'lshade_epsin.sinusoidal_increasing_f',
        'lshade_epsin.adaptive_frequency_update',
        'lshade_epsin.lshade_second_phase_adaptation',
        'lshade_epsin.gaussian_walk_local_search',
        'lshade_epsin.bound_repair',
    ),
    'lshade_rsp': (
        'lshade_rsp.mutation',
        'lshade_rsp.weighted_pbest_scaling',
        'lshade_rsp.crossover',
        'lshade_rsp.selection',
        'lshade_rsp.archive_update',
        'lshade_rsp.success_history_update',
        'lshade_rsp.population_reduction',
        'lshade_rsp.rank_selective_pressure',
        'lshade_rsp.bound_resampling',
    ),
    'lshade_spacma': (
        'lshade_spacma.mutation',
        'lshade_spacma.crossover',
        'lshade_spacma.selection',
        'lshade_spacma.archive_update',
        'lshade_spacma.success_history_update',
        'lshade_spacma.population_reduction',
        'lshade_spacma.cma_es_sampling',
        'lshade_spacma.cma_es_update',
        'lshade_spacma.semi_parameter_adaptation',
        'lshade_spacma.fcp_assignment',
        'lshade_spacma.fcp_memory_update',
        'lshade_spacma.lshade_branch',
        'lshade_spacma.cma_branch',
        'lshade_spacma.bound_repair',
    ),
    'lso_spectrum': (
        'lso_spectrum.light_spectrum_position_update',
    ),
    'mbo': (
        'mbo.monarch_migration_adjusting_update',
    ),
    'memetic_a': (
        'memetic_a.candidate_generation',
        'memetic_a.selection',
        'memetic_a.recombination',
        'memetic_a.mutation',
        'memetic_a.mutate',
        'memetic_a.xhc',
    ),
    'mfa': (
        'mfa.moth_flame_spiral_update',
    ),
    'mfea': (
        'mfea.unified_initialization',
        'mfea.factorial_evaluation',
        'mfea.factorial_rank_update',
        'mfea.skill_factor_assignment',
        'mfea.assortative_mating',
        'mfea.intratask_sbx_crossover',
        'mfea.intertask_sbx_transfer',
        'mfea.parent_centric_gaussian_mutation',
        'mfea.vertical_cultural_transmission',
        'mfea.scalar_fitness_selection',
        'mfea.elitist_replacement',
        'mfea.boundary_repair',
        'mfea.candidate_injection',
    ),
    'mfea2': (
        'mfea2.unified_initialization',
        'mfea2.skill_factor_assignment',
        'mfea2.scalar_fitness_selection',
        'mfea2.univariate_model_building',
        'mfea2.online_rmp_matrix_learning',
        'mfea2.intratask_sbx_crossover',
        'mfea2.intertask_sbx_transfer',
        'mfea2.parent_centric_polynomial_mutation',
        'mfea2.elitist_scalar_replacement',
        'mfea2.boundary_repair',
        'mfea2.candidate_injection',
    ),
    'mfo': (
        'mfo.exploration_move',
        'mfo.exploitation_move',
        'mfo.replacement',
    ),
    'mgo': (
        'mgo.territory_mountain_herding_update',
    ),
    'mgoa_market': (
        'mgoa_market.market_gradient_position_update',
    ),
    'misaco': (
        'misaco.lhs_initialization',
        'misaco.acomv_offspring_generation',
        'misaco.rbf_fit_selection',
        'misaco.lsbt_fit_selection',
        'misaco.random_selection',
        'misaco.sqp_rbf_local_search',
        'misaco.expensive_candidate_evaluation',
        'misaco.archive_update',
    ),
    'mke': (
        'mke.king_learning_fluctuation_update',
        'mke.peer_knowledge_difference_update',
    ),
    'mlshade_rl': (
        'mlshade_rl.ms1_current_to_pbest_weight_archive',
        'mlshade_rl.ms2_current_to_pbest_no_archive',
        'mlshade_rl.ms3_current_to_ordpbest_weight',
        'mlshade_rl.crossover',
        'mlshade_rl.selection',
        'mlshade_rl.strategy_probability_update',
        'mlshade_rl.parameter_adaptation',
        'mlshade_rl.archive_update',
        'mlshade_rl.population_reduction',
        'mlshade_rl.restart',
        'mlshade_rl.local_search',
    ),
    'modified_aeo': (
        'modified_aeo.selection',
        'modified_aeo.ecosystem_producer_consumer_update',
        'modified_aeo.modified_decomposition_refinement',
    ),
    'modified_eo': (
        'modified_eo.selection',
        'modified_eo.modified_equilibrium_pool_update',
        'modified_eo.modified_local_refinement',
    ),
    'moss_go': (
        'moss_go.water_dispersal_growth_update',
    ),
    'mpa': (
        'mpa.brownian_exploration',
        'mpa.brownian_transition',
        'mpa.levy_transition',
        'mpa.levy_exploitation',
        'mpa.fads',
    ),
    'mrfo': (
        'mrfo.chain_foraging',
        'mrfo.cyclone_random_foraging',
        'mrfo.cyclone_best_foraging',
        'mrfo.somersault_foraging',
    ),
    'msa_e': (
        'msa_e.golden_ratio_exploitation_update',
    ),
    'mshoa': (
        'mshoa.smasher_attack_update',
        'mshoa.spearer_circular_attack_update',
        'mshoa.defense_position_update',
    ),
    'msls': (
        'msls.update',
    ),
    'mso': (
        'mso.superior_mirage_search_update',
        'mso.inferior_mirage_search_update',
    ),
    'mtbo': (
        'mtbo.team_leader_coordinated_movement',
        'mtbo.avalanche_worst_avoidance',
        'mtbo.team_mean_movement',
        'mtbo.random_relocation_phase',
        'mtbo.candidate_generation',
        'mtbo.selection',
    ),
    'mts': (
        'mts.multiple_trajectory_local_search_update',
    ),
    'mvo': (
        'mvo.candidate_generation',
        'mvo.selection',
        'mvo.exploitation_move',
        'mvo.replacement',
    ),
    'mvpa': (
        'mvpa.mvp_guided_player_update',
    ),
    'nca': (
        'nca.acceleration_hyperbolic_contraction_random_subset_components',
    ),
    'nccla': (
        'nccla.vertical_social_learning',
        'nccla.horizontal_social_learning',
        'nccla.individual_learning',
        'nccla.juvenile_reinforcement',
        'nccla.parent_reinforcement',
        'nccla.parent_selection',
    ),
    'ngo': (
        'ngo.phase_one_update',
        'ngo.pursuit_exploitation_update',
        'ngo.selection',
    ),
    'nlapsmjso_eda': (
        'nlapsmjso_eda.sampling',
        'nlapsmjso_eda.selection',
        'nlapsmjso_eda.state_update',
        'nlapsmjso_eda.non_linear_population_analysis_update',
    ),
    'nlshade_lbc': (
        'nlshade_lbc.mutation',
        'nlshade_lbc.crossover',
        'nlshade_lbc.selection',
        'nlshade_lbc.archive_update',
        'nlshade_lbc.success_history_update',
        'nlshade_lbc.population_reduction',
        'nlshade_lbc.rank_selective_pressure',
        'nlshade_lbc.linear_bias_change',
        'nlshade_lbc.bound_resampling',
        'nlshade_lbc.crossover_rate_sorting',
    ),
    'nlshade_rsp': (
        'nlshade_rsp.mutation',
        'nlshade_rsp.crossover_binomial',
        'nlshade_rsp.crossover_exponential',
        'nlshade_rsp.crossover_rate_sorting',
        'nlshade_rsp.selection',
        'nlshade_rsp.archive_update',
        'nlshade_rsp.adaptive_archive_probability',
        'nlshade_rsp.success_history_update',
        'nlshade_rsp.nonlinear_population_reduction',
        'nlshade_rsp.rank_selective_pressure',
        'nlshade_rsp.bound_random_repair',
    ),
    'nlshade_rsp_midpoint': (
        'nlshade_rsp_midpoint.mutation',
        'nlshade_rsp_midpoint.crossover_binomial',
        'nlshade_rsp_midpoint.crossover_exponential',
        'nlshade_rsp_midpoint.crossover_rate_sorting',
        'nlshade_rsp_midpoint.selection',
        'nlshade_rsp_midpoint.archive_update',
        'nlshade_rsp_midpoint.adaptive_archive_probability',
        'nlshade_rsp_midpoint.success_history_update',
        'nlshade_rsp_midpoint.nonlinear_population_reduction',
        'nlshade_rsp_midpoint.rank_selective_pressure',
        'nlshade_rsp_midpoint.bound_resampling',
        'nlshade_rsp_midpoint.bound_random_repair_fallback',
        'nlshade_rsp_midpoint.midpoint_evaluation',
        'nlshade_rsp_midpoint.midpoint_replacement',
        'nlshade_rsp_midpoint.kmeans_midpoint',
        'nlshade_rsp_midpoint.midpoint_restart',
        'nlshade_rsp_midpoint.bounds_restart',
        'nlshade_rsp_midpoint.restart',
    ),
    'nmm': (
        'nmm.reflection_update',
        'nmm.expansion_update',
        'nmm.contraction_update',
        'nmm.shrink_update',
    ),
    'nmra': (
        'nmra.breeder_exploitation_update',
        'nmra.worker_exploration_update',
    ),
    'nndrea_so': (
        'nndrea_so.nn_weight_de_stage',
        'nndrea_so.solution_de_stage',
    ),
    'noa': (
        'noa.newton_position_update',
    ),
    'nro': (
        'nro.nuclear_fission_update',
        'nro.nuclear_fusion_update',
        'nro.selection',
    ),
    'nwoa': (
        'nwoa.exploration_move',
        'nwoa.exploitation_move',
        'nwoa.replacement',
    ),
    'ocro': (
        'ocro.candidate_generation',
        'ocro.selection',
        'ocro.position_update',
        'ocro.state_update',
        'ocro.initialization',
    ),
    'ofa': (
        'ofa.owl_neighbour_flight_update',
    ),
    'ogwo': (
        'ogwo.selection',
        'ogwo.elite_local_refinement',
        'ogwo.leader_guided_population_update',
    ),
    'ooa': (
        'ooa.hunting',
        'ooa.search',
        'ooa.selection',
        'ooa.state_update',
        'ooa.fish_carrying_local_update',
    ),
    'parrot_o': (
        'parrot_o.flight_area_search_update',
    ),
    'pbil': (
        'pbil.update',
    ),
    'pcx': (
        'pcx.parent_centric_crossover_update',
    ),
    'pdo': (
        'pdo.prairie_dog_burrow_alarm_update',
    ),
    'petio': (
        'petio.performance_evaluation_teaching_update',
    ),
    'pfa': (
        'pfa.pathfinder_position_update',
    ),
    'pfa_polar_fox': (
        'pfa_polar_fox.exploitation',
        'pfa_polar_fox.selection',
        'pfa_polar_fox.state_update',
        'pfa_polar_fox.experience_phase',
        'pfa_polar_fox.leader_guided_refinement_update',
        'pfa_polar_fox.leader_phase',
    ),
    'pko': (
        'pko.diving_beating_rate_update',
        'pko.crest_angle_foraging_update',
        'pko.hovering_attack_update',
        'pko.population_escape_update',
    ),
    'plba': (
        'plba.path_looping_bat_update',
    ),
    'plo': (
        'plo.aurora_global_local_update',
        'plo.polar_light_collision_update',
    ),
    'poa': (
        'poa.prey_pursuit_update',
        'poa.water_surface_winging_update',
    ),
    'political_o': (
        'political_o.candidate_generation',
        'political_o.selection',
        'political_o.parliamentary',
    ),
    'ppo': (
        'ppo.escape_sexual_cannibalism_juvenile_generation',
        'ppo.escape_predation_local_search',
    ),
    'pro': (
        'pro.learning',
        'pro.state_update',
        'pro.candidate_generation',
        'pro.selection',
    ),
    'pso': (
        'pso.inertia_velocity_update',
        'pso.cognitive_memory_update',
        'pso.social_global_update',
    ),
    'pss': (
        'pss.prominent_domain_sampling_update',
        'pss.full_domain_sampling_update',
        'pss.mixed_domain_sampling_update',
    ),
    'puma_o': (
        'puma_o.update',
    ),
    'qio': (
        'qio.three_point_quadratic_interpolation',
        'qio.two_point_reflection_interpolation',
    ),
    'qle_sca': (
        'qle_sca.candidate_generation',
        'qle_sca.selection',
        'qle_sca.learning',
        'qle_sca.state_update',
        'qle_sca.initialization',
    ),
    'qsa': (
        'qsa.business1',
        'qsa.business2',
        'qsa.business3',
    ),
    'random_s': (
        'random_s.random_sampling_update',
    ),
    'rbmo': (
        'rbmo.update',
    ),
    'rcco': (
        'rcco.rain_cloud_convection_update',
        'rcco.cloud_collision_local_update',
        'rcco.selection',
    ),
    'rde': (
        'rde.mutation_current_to_pbest',
        'rde.mutation_current_to_order_pbest',
        'rde.strategy_resource_allocation',
        'rde.extended_rank_selective_pressure',
        'rde.crossover',
        'rde.cauchy_target_perturbation',
        'rde.selection',
        'rde.archive_update',
        'rde.success_history_update',
        'rde.linear_population_reduction',
        'rde.bound_repair',
    ),
    'rdex_sop': (
        'rdex_sop.standard_branch_mutation',
        'rdex_sop.exploitation_biased_mutation',
        'rdex_sop.binomial_crossover',
        'rdex_sop.cauchy_local_perturbation',
        'rdex_sop.greedy_selection',
        'rdex_sop.dynamic_pbest_selection',
        'rdex_sop.hybrid_rate_update',
        'rdex_sop.success_history_update',
        'rdex_sop.linear_population_reduction',
        'rdex_sop.bound_resampling',
    ),
    'rf_bo': (
        'rf_bo.update',
    ),
    'rfo': (
        'rfo.red_fox_smell_search_update',
    ),
    'rhso': (
        'rhso.rhinoceros_herd_position_update',
    ),
    'rime': (
        'rime.hard_rime_puncture_update',
    ),
    'rmsprop': (
        'rmsprop.candidate_generation',
        'rmsprop.selection',
        'rmsprop.search_direction',
        'rmsprop.step_acceptance',
        'rmsprop.initialization',
    ),
    'roa': (
        'roa.remora_attempt_update',
    ),
    'rrto': (
        'rrto.adaptive_step_size_wandering',
        'rrto.absolute_difference_step',
        'rrto.boundary_based_step',
    ),
    'rsa': (
        'rsa.reptile_hunting_encircling_update',
    ),
    'rso': (
        'rso.long_chasing_update',
        'rso.short_chasing_update',
    ),
    'run': (
        'run.selection',
        'run.enhanced_solution_quality_update',
        'run.runge_kutta_position_update',
    ),
    'sa': (
        'sa.update',
    ),
    'saba': (
        'saba.self_adaptive_bat_update',
    ),
    'sacc_eam2': (
        'sacc_eam2.even_subcomponent_de_update',
        'sacc_eam2.odd_subcomponent_de_update',
    ),
    'sacoso': (
        'sacoso.cognitive_swarm_update',
        'sacoso.social_swarm_update',
    ),
    'sade': (
        'sade.selection',
        'sade.adaptive_strategy_de_update',
        'sade.elite_local_refinement',
    ),
    'sade_amss': (
        'sade_amss.adaptive_multistrategy_subspace_de_update',
    ),
    'sade_atdsc': (
        'sade_atdsc.adaptive_trial_distribution_selection_update',
    ),
    'sade_sammon': (
        'sade_sammon.sammon_surrogate_de_selection_update',
    ),
    'samso': (
        'samso.self_adaptive_migratory_swarm_update',
    ),
    'sap_de': (
        'sap_de.selection',
        'sap_de.elite_local_refinement',
        'sap_de.self_adaptive_parameter_de_update',
    ),
    'sapo': (
        'sapo.update',
    ),
    'saro': (
        'saro.candidate_generation',
        'saro.selection',
        'saro.individual',
        'saro.candidate_search',
        'saro.social',
    ),
    'sbo': (
        'sbo.bowerbird_mutation_update',
    ),
    'sboa': (
        'sboa.update',
    ),
    'scho': (
        'scho.scholar_chess_position_update',
    ),
    'scso': (
        'scso.exploration_move',
        'scso.replacement',
        'scso.selection',
        'scso.exploitation_move',
        'scso.candidate_generation',
    ),
    'sd': (
        'sd.candidate_generation',
        'sd.selection',
        'sd.search_direction',
        'sd.step_acceptance',
        'sd.initialization',
    ),
    'seaho': (
        'seaho.candidate_generation',
        'seaho.selection',
        'seaho.recombination',
    ),
    'secant_oa': (
        'secant_oa.secant_update',
        'secant_oa.stochastic_exploitation',
        'secant_oa.mutation_gate',
        'secant_oa.selection',
    ),
    'serval_oa': (
        'serval_oa.hunting',
        'serval_oa.selection',
        'serval_oa.state_update',
    ),
    'sfo': (
        'sfo.behavioral_move',
        'sfo.selection',
    ),
    'sfoa': (
        'sfoa.foraging',
        'sfoa.state_update',
        'sfoa.exploitation_move',
        'sfoa.replacement',
        'sfoa.exploration',
    ),
    'shade': (
        'shade.success_history_mutation_crossover_selection',
    ),
    'shio': (
        'shio.first_iguana_guidance',
        'shio.second_iguana_guidance',
        'shio.third_iguana_guidance',
    ),
    'shio_success': (
        'shio_success.best_history_guidance',
        'shio_success.second_history_guidance',
        'shio_success.third_history_guidance',
    ),
    'sho': (
        'sho.spotted_hyena_hunting_update',
    ),
    'sine_cosine_a': (
        'sine_cosine_a.sine_position_update',
        'sine_cosine_a.cosine_position_update',
    ),
    'singer_oa': (
        'singer_oa.imitation_mimicry_phase',
        'singer_oa.creation_perturbation_phase',
    ),
    'slo': (
        'slo.best_encircling_update',
        'slo.random_peer_encircling_update',
        'slo.spiral_attack_update',
    ),
    'sma': (
        'sma.random_dispersion_update',
        'sma.best_weighted_oscillation_update',
        'sma.contracting_vibration_update',
    ),
    'smo': (
        'smo.local_leader_phase',
        'smo.global_leader_phase',
        'smo.local_leader_decision',
    ),
    'snow_oa': (
        'snow_oa.exploration_group_update',
        'snow_oa.development_group_update',
    ),
    'so_snake': (
        'so_snake.male_snake_update',
        'so_snake.female_snake_update',
        'so_snake.selection',
    ),
    'soa': (
        'soa.seagull_spiral_attack_update',
    ),
    'soo': (
        'soo.selection',
        'soo.1_oscillatory_position',
        'soo.2_top_3_average_oscillatory',
    ),
    'sopt': (
        'sopt.statistical_population_selection_update',
    ),
    'sos': (
        'sos.candidate_generation',
        'sos.selection',
        'sos.mutualism',
        'sos.comensalism',
        'sos.parasitism',
    ),
    'sparrow_sa': (
        'sparrow_sa.producer_safe_foraging',
        'sparrow_sa.producer_alarm_random_walk',
        'sparrow_sa.scrounger_worst_avoidance',
        'sparrow_sa.scrounger_best_following',
        'sparrow_sa.awareness_best_escape',
        'sparrow_sa.awareness_worst_escape',
    ),
    'spbo': (
        'spbo.groups',
        'spbo.selection',
        'spbo.average_student_phase_update',
        'spbo.best_student',
        'spbo.excellent_student_phase_update',
    ),
    'sqp': (
        'sqp.update',
    ),
    'squirrel_sa': (
        'squirrel_sa.acorn_to_hickory_glide',
        'squirrel_sa.normal_to_acorn_glide',
        'squirrel_sa.normal_to_hickory_glide',
        'squirrel_sa.predator_random_relocation',
    ),
    'srsr': (
        'srsr.exploration',
        'srsr.selection',
        'srsr.1_accumulation_new_positions_via_gaussian',
    ),
    'srsr_robotics': (
        'srsr_robotics.candidate_generation',
        'srsr_robotics.selection',
        'srsr_robotics.guidance',
        'srsr_robotics.state_update',
        'srsr_robotics.exploration',
    ),
    'ssa': (
        'ssa.leader_plus_food_guidance',
        'ssa.leader_minus_food_guidance',
        'ssa.follower_front_chain_update',
        'ssa.follower_rear_chain_update',
    ),
    'ssdo': (
        'ssdo.sine_velocity_update',
        'ssdo.cosine_velocity_update',
    ),
    'ssio_rl': (
        'ssio_rl.update',
    ),
    'sso': (
        'sso.female_spider_position_update',
        'sso.male_spider_position_update',
    ),
    'sspider_a': (
        'sspider_a.social_spider_vibration_update',
    ),
    'sto': (
        'sto.prey_hunting_update',
        'sto.range_reduction_update',
    ),
    'superb_foa': (
        'superb_foa.global_smell_random_update',
        'superb_foa.levy_food_attraction',
        'superb_foa.best_food_convergence',
    ),
    'supply_do': (
        'supply_do.quantity_equilibrium_update',
        'supply_do.price_equilibrium_update',
    ),
    'tdo': (
        'tdo.exploration',
        'tdo.state_update',
        'tdo.hunting',
        'tdo.exploitation',
    ),
    'tfwo': (
        'tfwo.effect_of_objects',
        'tfwo.random_object_relocation',
        'tfwo.effect_of_whirlpools',
        'tfwo.best_whirlpool_preservation',
        'tfwo.object_whirlpool_exchange',
        'tfwo.state_structure_update',
    ),
    'thro': (
        'thro.throwing_race_update',
    ),
    'tlbo': (
        'tlbo.teacher_phase',
        'tlbo.learner_phase',
    ),
    'tlco': (
        'tlco.teacher_phase_update',
        'tlco.learner_phase_update',
        'tlco.selection',
    ),
    'toa': (
        'toa.stage_1_supervisor_guidance',
        'toa.learning',
        'toa.state_update',
        'toa.candidate_generation',
        'toa.selection',
    ),
    'toc': (
        'toc.fitness_proportional_assignment',
        'toc.coriolis_velocity_update',
        'toc.windstorm_to_tornado_evolution',
        'toc.windstorm_to_thunderstorm_evolution',
        'toc.thunderstorm_to_tornado_evolution',
        'toc.random_windstorm_formation',
        'toc.role_exchange_replacement',
    ),
    'tpo': (
        'tpo.carbon_nutrient_leaf_update',
    ),
    'tree_seed_a': (
        'tree_seed_a.toward_best_seed',
        'tree_seed_a.away_random_seed',
    ),
    'ts': (
        'ts.update',
    ),
    'tsa': (
        'tsa.toward_best_tunicate_update',
        'tsa.away_best_tunicate_update',
        'tsa.swarm_chain_averaging_update',
    ),
    'tso': (
        'tso.leader_spiral_update',
        'tso.random_migration_update',
        'tso.spiral_following_update',
        'tso.parabolic_foraging_update',
    ),
    'ttao': (
        'ttao.crossover',
        'ttao.selection',
        'ttao.state_update',
        'ttao.extra_candidate_diversification_update',
        'ttao.random_population_refresh_update',
    ),
    'two': (
        'two.tug_of_war_weight_force_update',
    ),
    'vcs': (
        'vcs.virus_diffusion',
        'vcs.host_cell_infection',
        'vcs.immune_response',
    ),
    'vns': (
        'vns.update',
    ),
    'waoa': (
        'waoa.feeding_exploration_update',
        'waoa.range_narrowing_exploitation',
    ),
    'warso': (
        'warso.attack_strategy_update',
        'warso.defense_strategy_update',
    ),
    'wca': (
        'wca.stream_toward_river',
        'wca.stream_river_exchange',
        'wca.river_toward_sea',
        'wca.evaporation_raining',
    ),
    'wdo': (
        'wdo.wind_velocity_position_update',
    ),
    'whale_foa': (
        'whale_foa.selection',
        'whale_foa.elite_local_refinement',
        'whale_foa.whale_position_update',
    ),
    'who': (
        'who.selection',
        'who.1_local_movement_milling',
        'who.2_herd_instinct',
        'who.social_memory',
    ),
    'wmqimrfo': (
        'wmqimrfo.selection',
        'wmqimrfo.elite_local_refinement',
        'wmqimrfo.weighted_multi_quadratic_mrfo_update',
    ),
    'wo_wave': (
        'wo_wave.wave_propagation_position_update',
    ),
    'woa': (
        'woa.search_for_prey',
        'woa.encircling_prey',
        'woa.spiral_bubble_net',
    ),
    'wooa': (
        'wooa.scavenging_predator_following',
        'wooa.prey_attack_update',
        'wooa.fight_chase_local_update',
    ),
    'wso': (
        'wso.white_shark_swarm_position_update',
    ),
    'wutp': (
        'wutp.horizontal_water_transport_update',
    ),
    'ydse': (
        'ydse.central_bright_fringe_update',
        'ydse.bright_fringe_interference_update',
        'ydse.dark_fringe_interference_update',
    ),
    'yo': (
        'yo.mcmc_burn_in',
        'yo.post_burnin_selection',
        'yo.mcmc_proposal',
        'yo.greedy_refinement',
        'yo.simulated_annealing_acceptance',
        'yo.blacklist_filter',
        'yo.adaptive_reheating',
        'yo.elite_update',
    ),
    'zoa': (
        'zoa.behavioral_move',
        'zoa.selection',
        'zoa.candidate_generation',
    ),
}

for _aid, _operators in _README_OPERATOR_OVERRIDES.items():
    _profile_current = EVOMAPX_OPERATOR_PROFILES.get(_aid)
    if _profile_current is not None:
        EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
            algorithm_id=_profile_current.algorithm_id,
            family=_profile_current.family,
            operators=tuple(_operators),
            fidelity=_profile_current.fidelity,
            phase=_profile_current.phase,
            notes=_profile_current.notes,
        )


# ---------------------------------------------------------------------------
# Paper-faithful native telemetry overrides for SAMSO / L2SMEA / SAPO
# ---------------------------------------------------------------------------
EVOMAPX_OPERATOR_PROFILES['samso'] = EvoMapXProfile(
    algorithm_id='samso',
    family='swarm',
    operators=(
        'samso.lhs_initialization',
        'samso.rbf_model_fit',
        'samso.rbf_optimum_infill',
        'samso.s_swarm_pso_update',
        'samso.l_swarm_tlbo_learner_update',
        'samso.prescreen_exact_evaluation',
        'samso.archive_update',
    ),
    fidelity='native',
    phase='paper_faithful_samso_l2smea_sapo',
    notes='Native SAMSO telemetry: LHS archive initialization, RBF modeling/infill, dynamic SPSO/TLBO multiswarm updates, prescreened exact evaluations, and archive updates.',
)
EVOMAPX_OPERATOR_PROFILES['l2smea'] = EvoMapXProfile(
    algorithm_id='l2smea',
    family='evolutionary',
    operators=(
        'l2smea.lhs_initialization',
        'l2smea.gaussian_subspace_construction',
        'l2smea.linear_subspace_surrogate_fit',
        'l2smea.multi_task_candidate_search',
        'l2smea.bi_criteria_infill_selection',
        'l2smea.expensive_evaluation_archive_update',
        'l2smea.gaussian_parameter_update',
    ),
    fidelity='native',
    phase='paper_faithful_samso_l2smea_sapo',
    notes='Native L2SMEA telemetry: 2D LHS archive, Gaussian linear-subspace construction, subspace surrogate fit, subproblem search, bi-criterion infill, expensive evaluation, and Gaussian parameter update.',
)
EVOMAPX_OPERATOR_PROFILES['sapo'] = EvoMapXProfile(
    algorithm_id='sapo',
    family='evolutionary',
    operators=(
        'sapo.lhs_initialization',
        'sapo.partial_selection_f_g_to_g',
        'sapo.partial_selection_g_to_f',
        'sapo.de_rand_1_binomial',
        'sapo.de_best_1_binomial',
        'sapo.reflection_bound_repair',
        'sapo.cubic_rbf_fit_predict',
        'sapo.feasibility_rule_selection',
        'sapo.expensive_evaluation_archive_update',
    ),
    fidelity='native',
    phase='paper_faithful_samso_l2smea_sapo',
    notes='Native SAPO telemetry: alternating partial objective/constraint selection, DE trial generation, reflection repair, cubic RBF screening, feasibility-rule selection, and archive update.',
)


# Addendum — native Differential Evolution telemetry refinement.
EVOMAPX_OPERATOR_PROFILES["de"] = EvoMapXProfile(
    algorithm_id="de",
    family="evolutionary",
    operators=("de.mutation", "de.crossover", "de.selection", "de.bound_repair"),
    fidelity="native",
    phase="supplied_de_audit_refinement",
    notes="Native DE telemetry reports synchronous differential mutation, binomial crossover, greedy selection/replacement, and framework bound repair without EvoMapX-side objective evaluations.",
)
