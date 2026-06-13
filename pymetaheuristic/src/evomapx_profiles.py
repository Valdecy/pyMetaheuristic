"""EvoMapX operator profiles for pyMetaheuristic.

Phase 1 of package-wide EvoMapX support: a centralized, auditable
operator taxonomy for every algorithm ID in the public table. Native
hooks can then be added family by family while preserving consistent names.
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
    fidelity: str = "profiled"  # native | profiled | macro
    phase: str = "phase_1_profile"
    notes: str = "Operator taxonomy declared; native hooks are added progressively by family."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["operators"] = list(self.operators)
        return data

FAMILY_DEFAULTS: dict[str, tuple[str, ...]] = {
    'evolutionary': ('selection', 'variation', 'mutation/recombination', 'replacement'),
    'swarm': ('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'),
    'physics': ('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'),
    'human': ('social learning', 'competition/role update', 'movement/update', 'selection/replacement'),
    'math': ('mathematical transform', 'candidate update', 'selection/replacement'),
    'trajectory': ('neighborhood/proposal', 'move acceptance', 'step adaptation'),
    'distribution': ('sampling', 'elite/model selection', 'distribution/model update', 'replacement'),
    'surrogate': ('surrogate fit', 'acquisition search', 'candidate evaluation', 'model update'),
    'nature': ('growth/foraging move', 'reproduction/spread', 'selection/replacement'),
    'unknown': ('candidate generation', 'selection/replacement'),
}

EVOMAPX_OPERATOR_PROFILES: dict[str, EvoMapXProfile] = {
    'aaa': EvoMapXProfile(algorithm_id='aaa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement', 'restart'), fidelity='profiled'),
    'aao': EvoMapXProfile(algorithm_id='aao', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'abco': EvoMapXProfile(algorithm_id='abco', family='swarm', operators=('employed search', 'onlooker selection', 'scout/reinitialization', 'replacement'), fidelity='profiled'),
    'acgwo': EvoMapXProfile(algorithm_id='acgwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'aco': EvoMapXProfile(algorithm_id='aco', family='swarm', operators=('pheromone/model update', 'solution construction', 'local/global update', 'replacement'), fidelity='profiled'),
    'acor': EvoMapXProfile(algorithm_id='acor', family='swarm', operators=('pheromone/model update', 'solution construction', 'local/global update', 'replacement'), fidelity='profiled'),
    'adam': EvoMapXProfile(algorithm_id='adam', family='math', operators=('descent direction', 'step update', 'acceptance'), fidelity='profiled'),
    'adaptive_eo': EvoMapXProfile(algorithm_id='adaptive_eo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'aefa': EvoMapXProfile(algorithm_id='aefa', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'aeo': EvoMapXProfile(algorithm_id='aeo', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'aesspso': EvoMapXProfile(algorithm_id='aesspso', family='swarm', operators=('velocity/social update', 'personal/global memory', 'position update', 'replacement'), fidelity='profiled'),
    'afsa': EvoMapXProfile(algorithm_id='afsa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'aft': EvoMapXProfile(algorithm_id='aft', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'agto': EvoMapXProfile(algorithm_id='agto', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'aha': EvoMapXProfile(algorithm_id='aha', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'aho': EvoMapXProfile(algorithm_id='aho', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'aiw_pso': EvoMapXProfile(algorithm_id='aiw_pso', family='swarm', operators=('velocity/social update', 'personal/global memory', 'position update', 'replacement'), fidelity='profiled'),
    'ala': EvoMapXProfile(algorithm_id='ala', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'alo': EvoMapXProfile(algorithm_id='alo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ao': EvoMapXProfile(algorithm_id='ao', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'aoa': EvoMapXProfile(algorithm_id='aoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'aoo': EvoMapXProfile(algorithm_id='aoo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'apo': EvoMapXProfile(algorithm_id='apo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'arch_oa': EvoMapXProfile(algorithm_id='arch_oa', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'aro': EvoMapXProfile(algorithm_id='aro', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ars': EvoMapXProfile(algorithm_id='ars', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation'), fidelity='profiled'),
    'artemisinin_o': EvoMapXProfile(algorithm_id='artemisinin_o', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'aso': EvoMapXProfile(algorithm_id='aso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'aso_atom': EvoMapXProfile(algorithm_id='aso_atom', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'autov': EvoMapXProfile(algorithm_id='autov', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'avoa': EvoMapXProfile(algorithm_id='avoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bacterial_colony_o': EvoMapXProfile(algorithm_id='bacterial_colony_o', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'basin_hopping': EvoMapXProfile(algorithm_id='basin_hopping', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation', 'restart'), fidelity='profiled'),
    'bat_a': EvoMapXProfile(algorithm_id='bat_a', family='swarm', operators=('velocity/frequency update', 'local search', 'acceptance', 'pulse/loudness update'), fidelity='profiled'),
    'bbo': EvoMapXProfile(algorithm_id='bbo', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'bboa': EvoMapXProfile(algorithm_id='bboa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bbso': EvoMapXProfile(algorithm_id='bbso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bco': EvoMapXProfile(algorithm_id='bco', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'bea': EvoMapXProfile(algorithm_id='bea', family='swarm', operators=('employed search', 'onlooker selection', 'scout/reinitialization', 'replacement'), fidelity='profiled'),
    'bes': EvoMapXProfile(algorithm_id='bes', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bfgs': EvoMapXProfile(algorithm_id='bfgs', family='math', operators=('descent direction', 'step update', 'acceptance'), fidelity='profiled'),
    'bfo': EvoMapXProfile(algorithm_id='bfo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bipop_cmaes': EvoMapXProfile(algorithm_id='bipop_cmaes', family='evolutionary', operators=('sampling', 'elite selection', 'mean update', 'covariance update', 'restart'), fidelity='profiled'),
    'bka': EvoMapXProfile(algorithm_id='bka', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bmo': EvoMapXProfile(algorithm_id='bmo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'boa': EvoMapXProfile(algorithm_id='boa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bono': EvoMapXProfile(algorithm_id='bono', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bps': EvoMapXProfile(algorithm_id='bps', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bro': EvoMapXProfile(algorithm_id='bro', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'bsa': EvoMapXProfile(algorithm_id='bsa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'bso': EvoMapXProfile(algorithm_id='bso', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'bspga': EvoMapXProfile(algorithm_id='bspga', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'btoa': EvoMapXProfile(algorithm_id='btoa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'bwo': EvoMapXProfile(algorithm_id='bwo', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'ca': EvoMapXProfile(algorithm_id='ca', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'camel': EvoMapXProfile(algorithm_id='camel', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'capsa': EvoMapXProfile(algorithm_id='capsa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cat_so': EvoMapXProfile(algorithm_id='cat_so', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cco': EvoMapXProfile(algorithm_id='cco', family='swarm', operators=('levy flight', 'replacement', 'abandonment/reinitialization'), fidelity='profiled'),
    'cddo': EvoMapXProfile(algorithm_id='cddo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cddo_child': EvoMapXProfile(algorithm_id='cddo_child', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'cdo': EvoMapXProfile(algorithm_id='cdo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cdo_chernobyl': EvoMapXProfile(algorithm_id='cdo_chernobyl', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'cem': EvoMapXProfile(algorithm_id='cem', family='distribution', operators=('sampling', 'elite selection', 'distribution update'), fidelity='native'),
    'ceo_cosmic': EvoMapXProfile(algorithm_id='ceo_cosmic', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'cfoa': EvoMapXProfile(algorithm_id='cfoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cg_gwo': EvoMapXProfile(algorithm_id='cg_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'cgo': EvoMapXProfile(algorithm_id='cgo', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'chameleon_sa': EvoMapXProfile(algorithm_id='chameleon_sa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'chaotic_gwo': EvoMapXProfile(algorithm_id='chaotic_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'chicken_so': EvoMapXProfile(algorithm_id='chicken_so', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'chio': EvoMapXProfile(algorithm_id='chio', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'choa': EvoMapXProfile(algorithm_id='choa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'circle_sa': EvoMapXProfile(algorithm_id='circle_sa', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'clonalg': EvoMapXProfile(algorithm_id='clonalg', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'cmaes': EvoMapXProfile(algorithm_id='cmaes', family='evolutionary', operators=('sampling', 'elite selection', 'mean update', 'covariance update', 'restart'), fidelity='profiled'),
    'coa': EvoMapXProfile(algorithm_id='coa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'coati_oa': EvoMapXProfile(algorithm_id='coati_oa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cockroach_so': EvoMapXProfile(algorithm_id='cockroach_so', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'compact_ga': EvoMapXProfile(algorithm_id='compact_ga', family='distribution', operators=('sampling', 'elite/model selection', 'distribution/model update', 'replacement'), fidelity='profiled'),
    'coot': EvoMapXProfile(algorithm_id='coot', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cpo': EvoMapXProfile(algorithm_id='cpo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'crayfish_oa': EvoMapXProfile(algorithm_id='crayfish_oa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cro': EvoMapXProfile(algorithm_id='cro', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'csa': EvoMapXProfile(algorithm_id='csa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'csbo': EvoMapXProfile(algorithm_id='csbo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'cso': EvoMapXProfile(algorithm_id='cso', family='swarm', operators=('velocity/social update', 'personal/global memory', 'position update', 'replacement'), fidelity='profiled'),
    'cuckoo_s': EvoMapXProfile(algorithm_id='cuckoo_s', family='swarm', operators=('levy flight', 'replacement', 'abandonment/reinitialization'), fidelity='profiled'),
    'da': EvoMapXProfile(algorithm_id='da', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'dbo': EvoMapXProfile(algorithm_id='dbo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ddao': EvoMapXProfile(algorithm_id='ddao', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'de': EvoMapXProfile(algorithm_id='de', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='native'),
    'deo_dolphin': EvoMapXProfile(algorithm_id='deo_dolphin', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'dfo': EvoMapXProfile(algorithm_id='dfo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'dhole_oa': EvoMapXProfile(algorithm_id='dhole_oa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'dmoa': EvoMapXProfile(algorithm_id='dmoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'do_dandelion': EvoMapXProfile(algorithm_id='do_dandelion', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'doa': EvoMapXProfile(algorithm_id='doa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'dra': EvoMapXProfile(algorithm_id='dra', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'dream_oa': EvoMapXProfile(algorithm_id='dream_oa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'ds_gwo': EvoMapXProfile(algorithm_id='ds_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'dso': EvoMapXProfile(algorithm_id='dso', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'dvba': EvoMapXProfile(algorithm_id='dvba', family='swarm', operators=('velocity/frequency update', 'local search', 'acceptance', 'pulse/loudness update'), fidelity='profiled'),
    'eao': EvoMapXProfile(algorithm_id='eao', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'eco': EvoMapXProfile(algorithm_id='eco', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'ecological_cycle_o': EvoMapXProfile(algorithm_id='ecological_cycle_o', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ecpo': EvoMapXProfile(algorithm_id='ecpo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'edo': EvoMapXProfile(algorithm_id='edo', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'eefo': EvoMapXProfile(algorithm_id='eefo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'eel_grouper_o': EvoMapXProfile(algorithm_id='eel_grouper_o', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'efo': EvoMapXProfile(algorithm_id='efo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'ego': EvoMapXProfile(algorithm_id='ego', family='distribution', operators=('sampling', 'elite/model selection', 'distribution/model update', 'replacement'), fidelity='profiled'),
    'eho': EvoMapXProfile(algorithm_id='eho', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'elk_ho': EvoMapXProfile(algorithm_id='elk_ho', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'enhanced_aeo': EvoMapXProfile(algorithm_id='enhanced_aeo', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'enhanced_two': EvoMapXProfile(algorithm_id='enhanced_two', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'eo': EvoMapXProfile(algorithm_id='eo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'eoa': EvoMapXProfile(algorithm_id='eoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ep': EvoMapXProfile(algorithm_id='ep', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'epc': EvoMapXProfile(algorithm_id='epc', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'er_gwo': EvoMapXProfile(algorithm_id='er_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'es': EvoMapXProfile(algorithm_id='es', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'esc': EvoMapXProfile(algorithm_id='esc', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'eso': EvoMapXProfile(algorithm_id='eso', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'esoa': EvoMapXProfile(algorithm_id='esoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'et_bo': EvoMapXProfile(algorithm_id='et_bo', family='surrogate', operators=('surrogate fit', 'acquisition search', 'candidate evaluation', 'model update'), fidelity='profiled'),
    'eto': EvoMapXProfile(algorithm_id='eto', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'evo': EvoMapXProfile(algorithm_id='evo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'ex_gwo': EvoMapXProfile(algorithm_id='ex_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'fata': EvoMapXProfile(algorithm_id='fata', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'fbio': EvoMapXProfile(algorithm_id='fbio', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'fda': EvoMapXProfile(algorithm_id='fda', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'fdo': EvoMapXProfile(algorithm_id='fdo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'fep': EvoMapXProfile(algorithm_id='fep', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'ffa': EvoMapXProfile(algorithm_id='ffa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ffo': EvoMapXProfile(algorithm_id='ffo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'firefly_a': EvoMapXProfile(algorithm_id='firefly_a', family='swarm', operators=('attraction move', 'randomization', 'selection/replacement'), fidelity='profiled'),
    'fla': EvoMapXProfile(algorithm_id='fla', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'flo': EvoMapXProfile(algorithm_id='flo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'flood_a': EvoMapXProfile(algorithm_id='flood_a', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'foa': EvoMapXProfile(algorithm_id='foa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'foa_fossa': EvoMapXProfile(algorithm_id='foa_fossa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'fox': EvoMapXProfile(algorithm_id='fox', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'fpa': EvoMapXProfile(algorithm_id='fpa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'frcg': EvoMapXProfile(algorithm_id='frcg', family='math', operators=('descent direction', 'step update', 'acceptance'), fidelity='profiled'),
    'frofi': EvoMapXProfile(algorithm_id='frofi', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'fss': EvoMapXProfile(algorithm_id='fss', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'fuzzy_gwo': EvoMapXProfile(algorithm_id='fuzzy_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'fwa': EvoMapXProfile(algorithm_id='fwa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ga': EvoMapXProfile(algorithm_id='ga', family='evolutionary', operators=('selection', 'crossover', 'mutation', 'elitism', 'replacement'), fidelity='native'),
    'gazelle_oa': EvoMapXProfile(algorithm_id='gazelle_oa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gbo': EvoMapXProfile(algorithm_id='gbo', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'gbrt_bo': EvoMapXProfile(algorithm_id='gbrt_bo', family='surrogate', operators=('surrogate fit', 'acquisition search', 'candidate evaluation', 'model update'), fidelity='profiled'),
    'gco': EvoMapXProfile(algorithm_id='gco', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'gea': EvoMapXProfile(algorithm_id='gea', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'ggo': EvoMapXProfile(algorithm_id='ggo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gja': EvoMapXProfile(algorithm_id='gja', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gjo': EvoMapXProfile(algorithm_id='gjo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gkso': EvoMapXProfile(algorithm_id='gkso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gmo': EvoMapXProfile(algorithm_id='gmo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gndo': EvoMapXProfile(algorithm_id='gndo', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'go_growth': EvoMapXProfile(algorithm_id='go_growth', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'goa': EvoMapXProfile(algorithm_id='goa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gp_bo': EvoMapXProfile(algorithm_id='gp_bo', family='surrogate', operators=('surrogate fit', 'acquisition search', 'candidate evaluation', 'model update'), fidelity='profiled'),
    'gpoo': EvoMapXProfile(algorithm_id='gpoo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gpso': EvoMapXProfile(algorithm_id='gpso', family='swarm', operators=('velocity/social update', 'personal/global memory', 'position update', 'replacement'), fidelity='profiled'),
    'grasp': EvoMapXProfile(algorithm_id='grasp', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation', 'restart'), fidelity='profiled'),
    'gsa': EvoMapXProfile(algorithm_id='gsa', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'gska': EvoMapXProfile(algorithm_id='gska', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'gso': EvoMapXProfile(algorithm_id='gso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gso_glider_snake': EvoMapXProfile(algorithm_id='gso_glider_snake', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gto': EvoMapXProfile(algorithm_id='gto', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'gwo': EvoMapXProfile(algorithm_id='gwo', family='swarm', operators=('alpha guidance', 'beta guidance', 'delta guidance', 'position update'), fidelity='native'),
    'gwo_woa': EvoMapXProfile(algorithm_id='gwo_woa', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'hba': EvoMapXProfile(algorithm_id='hba', family='swarm', operators=('velocity/frequency update', 'local search', 'acceptance', 'pulse/loudness update'), fidelity='profiled'),
    'hba_honey': EvoMapXProfile(algorithm_id='hba_honey', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'hbo': EvoMapXProfile(algorithm_id='hbo', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'hc': EvoMapXProfile(algorithm_id='hc', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation'), fidelity='profiled'),
    'hco': EvoMapXProfile(algorithm_id='hco', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'hde': EvoMapXProfile(algorithm_id='hde', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'heoa': EvoMapXProfile(algorithm_id='heoa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'hgs': EvoMapXProfile(algorithm_id='hgs', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'hgso': EvoMapXProfile(algorithm_id='hgso', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'hho': EvoMapXProfile(algorithm_id='hho', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'hi_woa': EvoMapXProfile(algorithm_id='hi_woa', family='swarm', operators=('encircling/search', 'spiral update', 'leader guidance', 'replacement'), fidelity='profiled'),
    'hiking_oa': EvoMapXProfile(algorithm_id='hiking_oa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'ho_hippo': EvoMapXProfile(algorithm_id='ho_hippo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'horse_oa': EvoMapXProfile(algorithm_id='horse_oa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'hsa': EvoMapXProfile(algorithm_id='hsa', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation'), fidelity='profiled'),
    'hsaba': EvoMapXProfile(algorithm_id='hsaba', family='swarm', operators=('velocity/frequency update', 'local search', 'acceptance', 'pulse/loudness update'), fidelity='profiled'),
    'hus': EvoMapXProfile(algorithm_id='hus', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'i_gwo': EvoMapXProfile(algorithm_id='i_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'i_woa': EvoMapXProfile(algorithm_id='i_woa', family='swarm', operators=('encircling/search', 'spiral update', 'leader guidance', 'replacement'), fidelity='profiled'),
    'iagwo': EvoMapXProfile(algorithm_id='iagwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'iaro': EvoMapXProfile(algorithm_id='iaro', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ica': EvoMapXProfile(algorithm_id='ica', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'ikoa': EvoMapXProfile(algorithm_id='ikoa', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'ils': EvoMapXProfile(algorithm_id='ils', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation', 'restart'), fidelity='profiled'),
    'ilshade': EvoMapXProfile(algorithm_id='ilshade', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'imode': EvoMapXProfile(algorithm_id='imode', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'improved_aeo': EvoMapXProfile(algorithm_id='improved_aeo', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'improved_qsa': EvoMapXProfile(algorithm_id='improved_qsa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'improved_tlo': EvoMapXProfile(algorithm_id='improved_tlo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'incremental_gwo': EvoMapXProfile(algorithm_id='incremental_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'info': EvoMapXProfile(algorithm_id='info', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'iobl_gwo': EvoMapXProfile(algorithm_id='iobl_gwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'ipop_cmaes': EvoMapXProfile(algorithm_id='ipop_cmaes', family='evolutionary', operators=('sampling', 'elite selection', 'mean update', 'covariance update', 'restart'), fidelity='profiled'),
    'ivya': EvoMapXProfile(algorithm_id='ivya', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'iwo': EvoMapXProfile(algorithm_id='iwo', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'jade': EvoMapXProfile(algorithm_id='jade', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'jde': EvoMapXProfile(algorithm_id='jde', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'jso': EvoMapXProfile(algorithm_id='jso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'jy': EvoMapXProfile(algorithm_id='jy', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'kha': EvoMapXProfile(algorithm_id='kha', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'kma': EvoMapXProfile(algorithm_id='kma', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'l2smea': EvoMapXProfile(algorithm_id='l2smea', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'laro': EvoMapXProfile(algorithm_id='laro', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'lca': EvoMapXProfile(algorithm_id='lca', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'lco': EvoMapXProfile(algorithm_id='lco', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'levy_ja': EvoMapXProfile(algorithm_id='levy_ja', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'lfd': EvoMapXProfile(algorithm_id='lfd', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'liwo': EvoMapXProfile(algorithm_id='liwo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'loa': EvoMapXProfile(algorithm_id='loa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'loa_lyrebird': EvoMapXProfile(algorithm_id='loa_lyrebird', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'lpo': EvoMapXProfile(algorithm_id='lpo', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'lshade_cnepsin': EvoMapXProfile(algorithm_id='lshade_cnepsin', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'lso_spectrum': EvoMapXProfile(algorithm_id='lso_spectrum', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'mbo': EvoMapXProfile(algorithm_id='mbo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'memetic_a': EvoMapXProfile(algorithm_id='memetic_a', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'mfa': EvoMapXProfile(algorithm_id='mfa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'mfea': EvoMapXProfile(algorithm_id='mfea', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'mfea2': EvoMapXProfile(algorithm_id='mfea2', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'mfo': EvoMapXProfile(algorithm_id='mfo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'mgo': EvoMapXProfile(algorithm_id='mgo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'mgoa_market': EvoMapXProfile(algorithm_id='mgoa_market', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'misaco': EvoMapXProfile(algorithm_id='misaco', family='swarm', operators=('pheromone/model update', 'solution construction', 'local/global update', 'replacement'), fidelity='profiled'),
    'mke': EvoMapXProfile(algorithm_id='mke', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'modified_aeo': EvoMapXProfile(algorithm_id='modified_aeo', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'modified_eo': EvoMapXProfile(algorithm_id='modified_eo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'moss_go': EvoMapXProfile(algorithm_id='moss_go', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'mpa': EvoMapXProfile(algorithm_id='mpa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'mrfo': EvoMapXProfile(algorithm_id='mrfo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'msa_e': EvoMapXProfile(algorithm_id='msa_e', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'mshoa': EvoMapXProfile(algorithm_id='mshoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'msls': EvoMapXProfile(algorithm_id='msls', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation', 'restart'), fidelity='profiled'),
    'mso': EvoMapXProfile(algorithm_id='mso', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'mtbo': EvoMapXProfile(algorithm_id='mtbo', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'mts': EvoMapXProfile(algorithm_id='mts', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation'), fidelity='profiled'),
    'mvo': EvoMapXProfile(algorithm_id='mvo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'mvpa': EvoMapXProfile(algorithm_id='mvpa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'nca': EvoMapXProfile(algorithm_id='nca', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'ngo': EvoMapXProfile(algorithm_id='ngo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'nlapsmjso_eda': EvoMapXProfile(algorithm_id='nlapsmjso_eda', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'nmm': EvoMapXProfile(algorithm_id='nmm', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation'), fidelity='profiled'),
    'nmra': EvoMapXProfile(algorithm_id='nmra', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'nndrea_so': EvoMapXProfile(algorithm_id='nndrea_so', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'noa': EvoMapXProfile(algorithm_id='noa', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'nro': EvoMapXProfile(algorithm_id='nro', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'nwoa': EvoMapXProfile(algorithm_id='nwoa', family='swarm', operators=('encircling/search', 'spiral update', 'leader guidance', 'replacement'), fidelity='profiled'),
    'ocro': EvoMapXProfile(algorithm_id='ocro', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'ofa': EvoMapXProfile(algorithm_id='ofa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ogwo': EvoMapXProfile(algorithm_id='ogwo', family='swarm', operators=('leader hierarchy guidance', 'encircling/diversification', 'position update', 'replacement'), fidelity='profiled'),
    'ooa': EvoMapXProfile(algorithm_id='ooa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'parrot_o': EvoMapXProfile(algorithm_id='parrot_o', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'pbil': EvoMapXProfile(algorithm_id='pbil', family='distribution', operators=('sampling', 'elite/model selection', 'distribution/model update', 'replacement'), fidelity='profiled'),
    'pcx': EvoMapXProfile(algorithm_id='pcx', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'pdo': EvoMapXProfile(algorithm_id='pdo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'petio': EvoMapXProfile(algorithm_id='petio', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'pfa': EvoMapXProfile(algorithm_id='pfa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'pfa_polar_fox': EvoMapXProfile(algorithm_id='pfa_polar_fox', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'pko': EvoMapXProfile(algorithm_id='pko', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'plba': EvoMapXProfile(algorithm_id='plba', family='swarm', operators=('velocity/frequency update', 'local search', 'acceptance', 'pulse/loudness update'), fidelity='profiled'),
    'plo': EvoMapXProfile(algorithm_id='plo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'poa': EvoMapXProfile(algorithm_id='poa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'political_o': EvoMapXProfile(algorithm_id='political_o', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'pro': EvoMapXProfile(algorithm_id='pro', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'pso': EvoMapXProfile(algorithm_id='pso', family='swarm', operators=('velocity update', 'personal memory', 'social guidance', 'position update'), fidelity='native'),
    'pss': EvoMapXProfile(algorithm_id='pss', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'puma_o': EvoMapXProfile(algorithm_id='puma_o', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'qio': EvoMapXProfile(algorithm_id='qio', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'qle_sca': EvoMapXProfile(algorithm_id='qle_sca', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'qsa': EvoMapXProfile(algorithm_id='qsa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'random_s': EvoMapXProfile(algorithm_id='random_s', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation'), fidelity='profiled'),
    'rbmo': EvoMapXProfile(algorithm_id='rbmo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'rcco': EvoMapXProfile(algorithm_id='rcco', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'rf_bo': EvoMapXProfile(algorithm_id='rf_bo', family='surrogate', operators=('surrogate fit', 'acquisition search', 'candidate evaluation', 'model update'), fidelity='profiled'),
    'rfo': EvoMapXProfile(algorithm_id='rfo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'rhso': EvoMapXProfile(algorithm_id='rhso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'rime': EvoMapXProfile(algorithm_id='rime', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'rmsprop': EvoMapXProfile(algorithm_id='rmsprop', family='math', operators=('descent direction', 'step update', 'acceptance'), fidelity='profiled'),
    'roa': EvoMapXProfile(algorithm_id='roa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'rsa': EvoMapXProfile(algorithm_id='rsa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'rso': EvoMapXProfile(algorithm_id='rso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'run': EvoMapXProfile(algorithm_id='run', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'sa': EvoMapXProfile(algorithm_id='sa', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation', 'restart'), fidelity='profiled'),
    'saba': EvoMapXProfile(algorithm_id='saba', family='swarm', operators=('velocity/frequency update', 'local search', 'acceptance', 'pulse/loudness update'), fidelity='profiled'),
    'sacc_eam2': EvoMapXProfile(algorithm_id='sacc_eam2', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'sacoso': EvoMapXProfile(algorithm_id='sacoso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sade': EvoMapXProfile(algorithm_id='sade', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'sade_amss': EvoMapXProfile(algorithm_id='sade_amss', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'sade_atdsc': EvoMapXProfile(algorithm_id='sade_atdsc', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'sade_sammon': EvoMapXProfile(algorithm_id='sade_sammon', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'samso': EvoMapXProfile(algorithm_id='samso', family='swarm', operators=('velocity/social update', 'personal/global memory', 'position update', 'replacement'), fidelity='profiled'),
    'sap_de': EvoMapXProfile(algorithm_id='sap_de', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'sapo': EvoMapXProfile(algorithm_id='sapo', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'saro': EvoMapXProfile(algorithm_id='saro', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'sbo': EvoMapXProfile(algorithm_id='sbo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sboa': EvoMapXProfile(algorithm_id='sboa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'scho': EvoMapXProfile(algorithm_id='scho', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'scso': EvoMapXProfile(algorithm_id='scso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sd': EvoMapXProfile(algorithm_id='sd', family='math', operators=('descent direction', 'step update', 'acceptance'), fidelity='profiled'),
    'seaho': EvoMapXProfile(algorithm_id='seaho', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'serval_oa': EvoMapXProfile(algorithm_id='serval_oa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sfo': EvoMapXProfile(algorithm_id='sfo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sfoa': EvoMapXProfile(algorithm_id='sfoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'shade': EvoMapXProfile(algorithm_id='shade', family='evolutionary', operators=('mutation', 'crossover', 'selection', 'parameter adaptation'), fidelity='profiled'),
    'shio': EvoMapXProfile(algorithm_id='shio', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'shio_success': EvoMapXProfile(algorithm_id='shio_success', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sho': EvoMapXProfile(algorithm_id='sho', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sine_cosine_a': EvoMapXProfile(algorithm_id='sine_cosine_a', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'singer_oa': EvoMapXProfile(algorithm_id='singer_oa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'slo': EvoMapXProfile(algorithm_id='slo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sma': EvoMapXProfile(algorithm_id='sma', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'smo': EvoMapXProfile(algorithm_id='smo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'snow_oa': EvoMapXProfile(algorithm_id='snow_oa', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'so_snake': EvoMapXProfile(algorithm_id='so_snake', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'soa': EvoMapXProfile(algorithm_id='soa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'soo': EvoMapXProfile(algorithm_id='soo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'sopt': EvoMapXProfile(algorithm_id='sopt', family='distribution', operators=('sampling', 'elite/model selection', 'distribution/model update', 'replacement'), fidelity='profiled'),
    'sos': EvoMapXProfile(algorithm_id='sos', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sparrow_sa': EvoMapXProfile(algorithm_id='sparrow_sa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'spbo': EvoMapXProfile(algorithm_id='spbo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sqp': EvoMapXProfile(algorithm_id='sqp', family='math', operators=('descent direction', 'step update', 'acceptance'), fidelity='profiled'),
    'squirrel_sa': EvoMapXProfile(algorithm_id='squirrel_sa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'srsr': EvoMapXProfile(algorithm_id='srsr', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'srsr_robotics': EvoMapXProfile(algorithm_id='srsr_robotics', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ssa': EvoMapXProfile(algorithm_id='ssa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ssdo': EvoMapXProfile(algorithm_id='ssdo', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'ssio_rl': EvoMapXProfile(algorithm_id='ssio_rl', family='evolutionary', operators=('selection', 'variation', 'mutation/recombination', 'replacement'), fidelity='profiled'),
    'sso': EvoMapXProfile(algorithm_id='sso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sspider_a': EvoMapXProfile(algorithm_id='sspider_a', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'sto': EvoMapXProfile(algorithm_id='sto', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'superb_foa': EvoMapXProfile(algorithm_id='superb_foa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'supply_do': EvoMapXProfile(algorithm_id='supply_do', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'tdo': EvoMapXProfile(algorithm_id='tdo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'tfwo': EvoMapXProfile(algorithm_id='tfwo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'thro': EvoMapXProfile(algorithm_id='thro', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'tlbo': EvoMapXProfile(algorithm_id='tlbo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'tlco': EvoMapXProfile(algorithm_id='tlco', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'toa': EvoMapXProfile(algorithm_id='toa', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'toc': EvoMapXProfile(algorithm_id='toc', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'tpo': EvoMapXProfile(algorithm_id='tpo', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'tree_seed_a': EvoMapXProfile(algorithm_id='tree_seed_a', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'ts': EvoMapXProfile(algorithm_id='ts', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation'), fidelity='profiled'),
    'tsa': EvoMapXProfile(algorithm_id='tsa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'tso': EvoMapXProfile(algorithm_id='tso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'ttao': EvoMapXProfile(algorithm_id='ttao', family='math', operators=('mathematical transform', 'candidate update', 'selection/replacement'), fidelity='profiled'),
    'two': EvoMapXProfile(algorithm_id='two', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'vcs': EvoMapXProfile(algorithm_id='vcs', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'vns': EvoMapXProfile(algorithm_id='vns', family='trajectory', operators=('neighborhood/proposal', 'move acceptance', 'step adaptation', 'restart'), fidelity='profiled'),
    'waoa': EvoMapXProfile(algorithm_id='waoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'warso': EvoMapXProfile(algorithm_id='warso', family='human', operators=('social learning', 'competition/role update', 'movement/update', 'replacement'), fidelity='profiled'),
    'wca': EvoMapXProfile(algorithm_id='wca', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'wdo': EvoMapXProfile(algorithm_id='wdo', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'whale_foa': EvoMapXProfile(algorithm_id='whale_foa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'who': EvoMapXProfile(algorithm_id='who', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'wmqimrfo': EvoMapXProfile(algorithm_id='wmqimrfo', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'wo_wave': EvoMapXProfile(algorithm_id='wo_wave', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'woa': EvoMapXProfile(algorithm_id='woa', family='swarm', operators=('encircling/search', 'spiral update', 'leader guidance', 'replacement'), fidelity='profiled'),
    'wooa': EvoMapXProfile(algorithm_id='wooa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'wso': EvoMapXProfile(algorithm_id='wso', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
    'wutp': EvoMapXProfile(algorithm_id='wutp', family='nature', operators=('growth/foraging move', 'reproduction/spread', 'selection/replacement'), fidelity='profiled'),
    'ydse': EvoMapXProfile(algorithm_id='ydse', family='physics', operators=('interaction/force update', 'field/equilibrium guidance', 'position update', 'replacement'), fidelity='profiled'),
    'zoa': EvoMapXProfile(algorithm_id='zoa', family='swarm', operators=('exploration move', 'exploitation move', 'leader/social guidance', 'replacement'), fidelity='profiled'),
}

# ---------------------------------------------------------------------------
# Phase 2 native-family profiles
# ---------------------------------------------------------------------------
# These algorithms now receive budget-preserving operator-level telemetry from
# evomapx_hooks.  Engines with more precise native logs keep their own
# operator_contributions; the phase-2 hook only fills missing logs.

_PHASE2_DE_VARIANTS = (
    "de", "jade", "sade", "sap_de", "hde", "jde", "shade", "ilshade",
    "lshade_cnepsin", "imode", "sade_amss", "sade_atdsc", "sade_sammon",
)
_PHASE2_PSO_VARIANTS = ("pso", "aiw_pso", "aesspso", "gpso")
_PHASE2_GWO_VARIANTS = (
    "gwo", "acgwo", "cg_gwo", "chaotic_gwo", "ds_gwo", "er_gwo", "ex_gwo",
    "fuzzy_gwo", "gwo_woa", "i_gwo", "iagwo", "incremental_gwo", "iobl_gwo",
    "ogwo",
)

for _aid in _PHASE2_DE_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="evolutionary",
        operators=(
            "differential mutation",
            "crossover/recombination",
            "greedy selection/replacement",
            "parameter adaptation/archive",
        ),
        fidelity="native-family",
        phase="phase_2_de_pso_gwo_variants",
        notes="Phase 2 native-family EvoMapX hook logs DE-style variation and replacement without extra objective evaluations.",
    )

for _aid in _PHASE2_PSO_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "velocity/social update",
            "position update",
            "personal/global memory",
        ),
        fidelity="native-family",
        phase="phase_2_de_pso_gwo_variants",
        notes="Phase 2 native-family EvoMapX hook logs PSO-style velocity, position, and memory mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE2_GWO_VARIANTS:
    _ops = (
        "alpha/beta/delta guidance",
        "encircling/diversification",
        "position update/replacement",
    )
    if _aid == "gwo_woa":
        _ops = _ops + ("spiral/whale exploitation",)
    if _aid in {"acgwo", "chaotic_gwo", "cg_gwo"}:
        _ops = _ops + ("chaotic/adaptive control",)
    if _aid in {"iobl_gwo", "ogwo"}:
        _ops = _ops + ("opposition learning",)
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=_ops,
        fidelity="native-family",
        phase="phase_2_de_pso_gwo_variants",
        notes="Phase 2 native-family EvoMapX hook logs GWO-style leader hierarchy, encircling, and replacement mechanisms without extra objective evaluations.",
    )


# ---------------------------------------------------------------------------
# Phase 3 native-family profiles
# ---------------------------------------------------------------------------
# WOA/HHO/BAT/Firefly/Cuckoo-Levy families now receive runtime operator
# contributions from evomapx_hooks.  The hooks are budget-preserving:
# they use already-computed pre/post fitness and do not evaluate the objective.

_PHASE3_WOA_VARIANTS = ("woa", "i_woa", "hi_woa", "whale_foa", "nwoa")
_PHASE3_HHO_VARIANTS = ("hho",)
_PHASE3_BAT_VARIANTS = ("bat_a", "hba", "hsaba", "saba", "dvba", "plba")
_PHASE3_FIREFLY_VARIANTS = ("firefly_a",)
_PHASE3_CUCKOO_LEVY_VARIANTS = ("cuckoo_s", "cco", "fpa", "lfd", "levy_ja", "laro")

for _aid in _PHASE3_WOA_VARIANTS:
    _ops = (
        "encircling/search",
        "spiral exploitation",
        "leader guidance",
        "replacement",
    )
    if _aid == "whale_foa":
        _ops = _ops + ("fruit-fly sensory search",)
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=_ops,
        fidelity="native-family",
        phase="phase_3_woa_hho_bat_firefly_cuckoo_swarm",
        notes="Phase 3 native-family EvoMapX hook logs WOA/whale-search mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE3_HHO_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "exploration perch/search",
            "soft/hard besiege",
            "rapid dive/exploitation",
            "replacement",
        ),
        fidelity="native-family",
        phase="phase_3_woa_hho_bat_firefly_cuckoo_swarm",
        notes="Phase 3 native-family EvoMapX hook logs HHO exploration, besiege, rapid-dive, and replacement mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE3_BAT_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "velocity/frequency update",
            "local random walk",
            "acceptance/replacement",
            "pulse/loudness adaptation",
        ),
        fidelity="native-family",
        phase="phase_3_woa_hho_bat_firefly_cuckoo_swarm",
        notes="Phase 3 native-family EvoMapX hook logs BAT acoustic-search mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE3_FIREFLY_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "brightness attraction",
            "randomization",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_3_woa_hho_bat_firefly_cuckoo_swarm",
        notes="Phase 3 native-family EvoMapX hook logs Firefly attraction, randomization, and replacement mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE3_CUCKOO_LEVY_VARIANTS:
    _ops = (
        "levy/global pollination",
        "local pollination/random walk",
        "replacement/selection",
    )
    if _aid in {"cuckoo_s", "cco"}:
        _ops = _ops + ("abandonment/reinitialization",)
    if _aid == "laro":
        _ops = _ops + ("opposition learning",)
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=_ops,
        fidelity="native-family",
        phase="phase_3_woa_hho_bat_firefly_cuckoo_swarm",
        notes="Phase 3 native-family EvoMapX hook logs Cuckoo/Levy-pollination search mechanisms without extra objective evaluations.",
    )


# ---------------------------------------------------------------------------
# Phase 4 native-family profiles
# ---------------------------------------------------------------------------
# ABC/bee-search, ACO/ant-search, Ant-lion, and broad leader-guided swarm
# families now receive runtime operator contributions from evomapx_hooks.
# The hooks remain budget-preserving: they compare already-computed pre/post
# fitness values and do not evaluate the objective function.

_PHASE4_ABC_BEE_VARIANTS = ("abco", "bea")
_PHASE4_ACO_ANT_VARIANTS = ("aco", "acor", "misaco")
_PHASE4_ANTLION_VARIANTS = ("alo",)
_PHASE4_LEADER_GUIDED_SWARM_VARIANTS = (
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
)

for _aid in _PHASE4_ABC_BEE_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "employed/foraging search",
            "onlooker/elite selection",
            "scout/reinitialization",
            "replacement/memorization",
        ),
        fidelity="native-family",
        phase="phase_4_abc_aco_antlion_leader_swarm",
        notes="Phase 4 native-family EvoMapX hook logs ABC/bee-search foraging, onlooker, scout, and replacement mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE4_ACO_ANT_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "solution construction",
            "pheromone/model update",
            "local/global exploitation",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_4_abc_aco_antlion_leader_swarm",
        notes="Phase 4 native-family EvoMapX hook logs ACO/ant-search construction, pheromone/model update, local/global exploitation, and replacement mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE4_ANTLION_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "random walk around antlion",
            "elite antlion guidance",
            "trap/boundary adaptation",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_4_abc_aco_antlion_leader_swarm",
        notes="Phase 4 native-family EvoMapX hook logs Ant-lion random walk, elite guidance, trap/boundary adaptation, and replacement mechanisms without extra objective evaluations.",
    )

for _aid in _PHASE4_LEADER_GUIDED_SWARM_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="swarm",
        operators=(
            "exploration move",
            "exploitation move",
            "leader/social guidance",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_4_abc_aco_antlion_leader_swarm",
        notes="Phase 4 native-family EvoMapX hook logs general leader-guided swarm exploration, exploitation, guidance, and replacement mechanisms without extra objective evaluations.",
    )



# ---------------------------------------------------------------------------
# Phase 5 native-family profiles
# ---------------------------------------------------------------------------
# Physics-based, equilibrium, force-field, wave/flow, and energy-state
# optimizers receive runtime operator contributions from evomapx_hooks.
# The hooks remain budget-preserving and passive: they use only already observed
# pre/post fitness values and never call the objective function.

_PHASE5_EQUILIBRIUM_VARIANTS = ("adaptive_eo", "eo", "modified_eo", "hgso")
_PHASE5_FORCE_FIELD_VARIANTS = ("aefa", "aso_atom", "ecpo", "efo", "gsa", "nro", "two", "enhanced_two")
_PHASE5_WAVE_FLOW_VARIANTS = ("fla", "flood_a", "liwo", "lso_spectrum", "rcco", "tfwo", "toc", "wdo", "wo_wave", "ydse")
_PHASE5_ENERGY_STATE_VARIANTS = (
    "arch_oa", "cdo_chernobyl", "ceo_cosmic", "ddao", "do_dandelion",
    "eso", "evo", "fata", "gea", "ikoa", "mso", "plo", "rime",
    "snow_oa", "soo",
)

for _aid in _PHASE5_EQUILIBRIUM_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="physics",
        operators=(
            "equilibrium pool guidance",
            "generation/control-rate update",
            "position update",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_5_physics_equilibrium_force_field",
        notes="Phase 5 native-family EvoMapX hook logs equilibrium-pool guidance, control-rate update, position update, and replacement without extra objective evaluations.",
    )

for _aid in _PHASE5_FORCE_FIELD_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="physics",
        operators=(
            "force/field interaction",
            "acceleration/mass update",
            "position update",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_5_physics_equilibrium_force_field",
        notes="Phase 5 native-family EvoMapX hook logs force/field interaction, acceleration or mass update, position update, and replacement without extra objective evaluations.",
    )

for _aid in _PHASE5_WAVE_FLOW_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="physics",
        operators=(
            "flow/wave propagation",
            "physical coefficient update",
            "position transport/update",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_5_physics_equilibrium_force_field",
        notes="Phase 5 native-family EvoMapX hook logs wave/flow propagation, physical coefficient update, position transport, and replacement without extra objective evaluations.",
    )

for _aid in _PHASE5_ENERGY_STATE_VARIANTS:
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family="physics",
        operators=(
            "energy/state transition",
            "force/equilibrium guidance",
            "position update",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_5_physics_equilibrium_force_field",
        notes="Phase 5 native-family EvoMapX hook logs energy/state transition, force or equilibrium guidance, position update, and replacement without extra objective evaluations.",
    )


# ---------------------------------------------------------------------------
# Phase 6 overrides — human/social/teaching/competition algorithms
# ---------------------------------------------------------------------------
_PHASE6_TEACHING_LEARNING_VARIANTS = {
    "tlbo", "improved_tlo", "petio", "toa", "eco", "spbo",
}

_PHASE6_COMPETITION_ROLE_VARIANTS = {
    "btoa", "bro", "bso", "chio", "gco", "gska", "hbo", "ica",
    "mgoa_market", "mvpa", "political_o", "pro", "qsa", "improved_qsa",
    "saro", "thro", "warso",
}

_PHASE6_HUMAN_SOCIAL_VARIANTS = {
    "aft", "aeo", "btoa", "bro", "bso", "cddo_child", "chio", "doa",
    "dra", "dream_oa", "dso", "eco", "enhanced_aeo", "esc", "fbio",
    "gco", "gska", "hbo", "hiking_oa", "hco", "heoa", "ica",
    "improved_aeo", "improved_qsa", "lco", "mgoa_market", "modified_aeo",
    "mtbo", "mvpa", "petio", "political_o", "pro", "qsa", "saro",
    "singer_oa", "ssdo", "supply_do", "thro", "toa", "warso",
    "tlbo", "improved_tlo", "spbo",
}

for _aid in _PHASE6_TEACHING_LEARNING_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "human",
        operators=(
            "teacher/leader phase",
            "learner/social phase",
            "competition/evaluation",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_6_human_social_teaching_competition",
        notes="Phase 6 native-family EvoMapX hook logs teaching/leader phase, learner/social phase, competition/evaluation, and replacement without extra objective evaluations.",
    )

for _aid in _PHASE6_COMPETITION_ROLE_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "human",
        operators=(
            "role/team competition",
            "social learning/assimilation",
            "movement/update",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_6_human_social_teaching_competition",
        notes="Phase 6 native-family EvoMapX hook logs role/team competition, social learning or assimilation, movement/update, and replacement without extra objective evaluations.",
    )

for _aid in sorted(_PHASE6_HUMAN_SOCIAL_VARIANTS - _PHASE6_TEACHING_LEARNING_VARIANTS - _PHASE6_COMPETITION_ROLE_VARIANTS):
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "human",
        operators=(
            "social learning",
            "competition/role update",
            "movement/update",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_6_human_social_teaching_competition",
        notes="Phase 6 native-family EvoMapX hook logs social learning, role/competition update, movement/update, and replacement without extra objective evaluations.",
    )


# ---------------------------------------------------------------------------
# Phase 7 overrides — distribution/model, CMA-ES/EDA, surrogate-assisted,
# trajectory/local-search, and classic gradient-local algorithms
# ---------------------------------------------------------------------------
_PHASE7_DISTRIBUTION_MODEL_VARIANTS = {
    "cem", "compact_ga", "ego", "pbil", "sopt",
}

_PHASE7_CMAES_EDA_VARIANTS = {
    "cmaes", "bipop_cmaes", "ipop_cmaes", "nlapsmjso_eda",
}

_PHASE7_SURROGATE_MODEL_VARIANTS = {
    "et_bo", "gp_bo", "gbrt_bo", "rf_bo",
}

_PHASE7_SURROGATE_ASSISTED_VARIANTS = {
    "l2smea", "misaco", "sacc_eam2", "sacoso", "sade_amss",
    "sade_atdsc", "sapo",
}

_PHASE7_TRAJECTORY_LOCAL_VARIANTS = {
    "ars", "basin_hopping", "grasp", "hsa", "hc", "ils", "msls",
    "mts", "nmm", "random_s", "sa", "ts", "vns",
}

_PHASE7_GRADIENT_LOCAL_VARIANTS = {
    "adam", "bfgs", "frcg", "rmsprop", "sd", "sqp",
}

for _aid in _PHASE7_DISTRIBUTION_MODEL_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "distribution",
        operators=(
            "model sampling",
            "elite/model selection",
            "model update",
            "replacement/incumbent update",
        ),
        fidelity="native-family",
        phase="phase_7_distribution_surrogate_trajectory",
        notes="Phase 7 native-family EvoMapX hook logs model sampling, elite/model selection, model update, and incumbent replacement without extra objective evaluations.",
    )

for _aid in _PHASE7_CMAES_EDA_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "evolutionary",
        operators=(
            "multivariate sampling",
            "elite/parent selection",
            "mean/covariance update",
            "step-size/restart control",
        ),
        fidelity="native-family",
        phase="phase_7_distribution_surrogate_trajectory",
        notes="Phase 7 native-family EvoMapX hook logs CMA-ES/EDA sampling, parent selection, distribution update, and restart/step-size control without extra objective evaluations.",
    )

for _aid in _PHASE7_SURROGATE_MODEL_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "surrogate",
        operators=(
            "surrogate fit",
            "acquisition search",
            "candidate evaluation",
            "model/incumbent update",
        ),
        fidelity="native-family",
        phase="phase_7_distribution_surrogate_trajectory",
        notes="Phase 7 native-family EvoMapX hook logs surrogate fit, acquisition search, candidate evaluation, and model/incumbent update without extra objective evaluations.",
    )

for _aid in _PHASE7_SURROGATE_ASSISTED_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "evolutionary",
        operators=(
            "surrogate screening/modeling",
            "evolutionary/swarm variation",
            "candidate evaluation",
            "selection/model update",
        ),
        fidelity="native-family",
        phase="phase_7_distribution_surrogate_trajectory",
        notes="Phase 7 native-family EvoMapX hook logs surrogate-assisted modeling/screening, variation, candidate evaluation, and selection/model update without extra objective evaluations.",
    )

for _aid in _PHASE7_TRAJECTORY_LOCAL_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "trajectory",
        operators=(
            "proposal/neighborhood move",
            "move acceptance",
            "step-size/adaptation",
            "incumbent update/restart",
        ),
        fidelity="native-family",
        phase="phase_7_distribution_surrogate_trajectory",
        notes="Phase 7 native-family EvoMapX hook logs proposal/neighborhood move, acceptance, adaptation, and incumbent/restart updates without extra objective evaluations.",
    )

for _aid in _PHASE7_GRADIENT_LOCAL_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "math",
        operators=(
            "descent/gradient direction",
            "scaling/curvature update",
            "parameter step",
            "acceptance/incumbent update",
        ),
        fidelity="native-family",
        phase="phase_7_distribution_surrogate_trajectory",
        notes="Phase 7 native-family EvoMapX hook logs local mathematical descent/gradient direction, scaling or curvature update, parameter step, and incumbent update without extra objective evaluations.",
    )


# ---------------------------------------------------------------------------
# Phase 8 overrides — remaining evolutionary, immune/clonal,
# genetic/memetic, cultural/biogeography, EP/ES, and multi-factorial methods
# ---------------------------------------------------------------------------
_PHASE8_GENETIC_MEMETIC_VARIANTS = {
    "autov", "bspga", "bwo", "ga", "memetic_a", "pcx", "ssio_rl",
}

_PHASE8_IMMUNE_CULTURAL_EVOLUTIONARY_VARIANTS = {
    "bbo", "clonalg", "cro", "ca", "ocro", "mke", "bco", "bacterial_colony_o",
}

_PHASE8_EP_ES_VARIANTS = {
    "es", "ep", "fep",
}

_PHASE8_MULTIFACTORIAL_VARIANTS = {
    "mfea", "mfea2", "nndrea_so", "frofi",
}

for _aid in _PHASE8_GENETIC_MEMETIC_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "evolutionary",
        operators=(
            "parent/operator selection",
            "crossover/recombination",
            "mutation/diversification",
            "elitist replacement",
        ),
        fidelity="native-family" if _aid != "ga" else "native",
        phase="phase_8_evolutionary_immune_genetic_memetic",
        notes="Phase 8 EvoMapX hook logs genetic/memetic selection, recombination, mutation/diversification, and elitist replacement without extra objective evaluations.",
    )

for _aid in _PHASE8_IMMUNE_CULTURAL_EVOLUTIONARY_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "evolutionary",
        operators=(
            "affinity/migration selection",
            "cloning/reproduction",
            "hypermutation/diversification",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_8_evolutionary_immune_genetic_memetic",
        notes="Phase 8 EvoMapX hook logs immune/cultural/biogeography evolutionary selection, reproduction, hypermutation/diversification, and replacement without extra objective evaluations.",
    )

for _aid in _PHASE8_EP_ES_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "evolutionary",
        operators=(
            "mutation/self-adaptation",
            "offspring generation",
            "survivor selection",
            "strategy-parameter update",
        ),
        fidelity="native-family",
        phase="phase_8_evolutionary_immune_genetic_memetic",
        notes="Phase 8 EvoMapX hook logs evolutionary programming/strategy mutation, offspring generation, survivor selection, and strategy-parameter update without extra objective evaluations.",
    )

for _aid in _PHASE8_MULTIFACTORIAL_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "evolutionary",
        operators=(
            "task/skill assignment",
            "assortative mating/transfer",
            "mutation/diversification",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_8_evolutionary_immune_genetic_memetic",
        notes="Phase 8 EvoMapX hook logs multi-factorial or objective-informed evolutionary assignment/transfer, variation, diversification, and replacement without extra objective evaluations.",
    )


# ---------------------------------------------------------------------------
# Phase 9 overrides — remaining nature/biology/growth and mathematical-transform optimizers
# ---------------------------------------------------------------------------
_PHASE9_NATURE_BIOLOGY_GROWTH_VARIANTS = {
    "artemisinin_o", "eao", "ivya", "iwo", "lca", "lpo", "moss_go",
    "sma", "tpo", "tree_seed_a", "wca", "wutp",
}

_PHASE9_MATH_TRANSFORM_VARIANTS = {
    "cgo", "circle_sa", "edo", "eto", "gbo", "gndo", "info", "nca",
    "noa", "pss", "qio", "run", "scho", "ttao",
}

for _aid in _PHASE9_NATURE_BIOLOGY_GROWTH_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "nature",
        operators=(
            "biological growth/foraging",
            "reproduction/spread",
            "adaptive diversification",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_9_nature_biology_math_transform",
        notes="Phase 9 native-family EvoMapX hook logs nature/biology/growth movement, reproduction/spread, adaptive diversification, and selection/replacement without extra objective evaluations.",
    )

for _aid in _PHASE9_MATH_TRANSFORM_VARIANTS:
    _family = EVOMAPX_OPERATOR_PROFILES.get(_aid, EvoMapXProfile(_aid)).family
    EVOMAPX_OPERATOR_PROFILES[_aid] = EvoMapXProfile(
        algorithm_id=_aid,
        family=_family if _family != "unknown" else "math",
        operators=(
            "mathematical transform",
            "candidate update",
            "adaptive control/diversification",
            "selection/replacement",
        ),
        fidelity="native-family",
        phase="phase_9_nature_biology_math_transform",
        notes="Phase 9 native-family EvoMapX hook logs mathematical-transform search, candidate update, adaptive control/diversification, and selection/replacement without extra objective evaluations.",
    )

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
    """Return all declared profiles as dictionaries."""
    return [EVOMAPX_OPERATOR_PROFILES[k].to_dict() for k in sorted(EVOMAPX_OPERATOR_PROFILES)]

# ---------------------------------------------------------------------------
# Addendum — New Caledonian Crow Learning Algorithm
# ---------------------------------------------------------------------------
EVOMAPX_OPERATOR_PROFILES["nccla"] = EvoMapXProfile(
    algorithm_id="nccla",
    family="human",
    operators=(
        "vertical_social_learning",
        "horizontal_social_learning",
        "individual_learning",
        "juvenile_reinforcement",
        "parent_reinforcement",
        "parent_selection",
    ),
    fidelity="native",
    phase="addendum_nccla",
    notes=(
        "Native NCCLA hook logs parent reinforcement and juvenile vertical social, "
        "horizontal social, individual learning, and reinforcement transitions. "
        "Parent selection is diagnostic metadata based on top-two sorted crows."
    ),
)

# ---------------------------------------------------------------------------
# Addendum — 2024–2025 metaheuristic paper ports
# ---------------------------------------------------------------------------
EVOMAPX_OPERATOR_PROFILES["agdo"] = EvoMapXProfile(
    algorithm_id="agdo",
    family="math",
    operators=(
        "progressive_gradient_momentum_integration",
        "dynamic_gradient_interaction",
        "trust_region_selection",
        "system_optimization_operator",
    ),
    fidelity="native",
    phase="addendum_2025_metaheuristics",
    notes="Native AGDO hook labels Adam-inspired dynamic gradient interaction and logistic system-optimization replacements without extra objective evaluations.",
)

EVOMAPX_OPERATOR_PROFILES["dp"] = EvoMapXProfile(
    algorithm_id="dp",
    family="math",
    operators=(
        "delta_operation",
        "realtime_learning_vector",
        "inertial_learning_vector",
        "greedy_selection",
    ),
    fidelity="native",
    phase="addendum_2025_metaheuristics",
    notes="Native Delta Plus hook labels accepted Delta-operation moves; realtime and inertial learning vectors are computed inside the engine state update.",
)

EVOMAPX_OPERATOR_PROFILES["lea"] = EvoMapXProfile(
    algorithm_id="lea",
    family="evolutionary",
    operators=(
        "stimulus_matching",
        "value_phase",
        "reflection_operation",
        "role_phase",
        "generational_replacement",
    ),
    fidelity="native",
    phase="addendum_2025_metaheuristics",
    notes="Native LEA hook labels pairwise value, reflection, and role-phase transformations for EvoMapX lineage.",
)

EVOMAPX_OPERATOR_PROFILES["ppo"] = EvoMapXProfile(
    algorithm_id="ppo",
    family="swarm",
    operators=(
        "escape_ejection",
        "sexual_cannibalism_juvenile_generation",
        "predation_local_search",
        "historical_food_update",
    ),
    fidelity="native",
    phase="addendum_2025_metaheuristics",
    notes="Native PPO hook labels accepted cannibalism/juvenile-generation and predation local-search moves while preserving historical food guidance.",
)

EVOMAPX_OPERATOR_PROFILES["rrto"] = EvoMapXProfile(
    algorithm_id="rrto",
    family="swarm",
    operators=(
        "adaptive_step_size_wandering",
        "absolute_difference_step",
        "boundary_based_step",
        "collision_boundary_handling",
    ),
    fidelity="native",
    phase="addendum_2025_metaheuristics",
    notes="Native RRTO hook labels accepted moves from its three RRT-inspired adaptive step-size strategies.",
)

# ---------------------------------------------------------------------------
# Addendum — Yukthi Opus
# ---------------------------------------------------------------------------
EVOMAPX_OPERATOR_PROFILES["yo"] = EvoMapXProfile(
    algorithm_id="yo",
    family="trajectory",
    operators=(
        "mcmc_burn_in",
        "post_burnin_selection",
        "mcmc_proposal",
        "greedy_refinement",
        "simulated_annealing_acceptance",
        "blacklist_filter",
        "adaptive_reheating",
        "elite_update",
    ),
    fidelity="native",
    phase="addendum_yukthi_opus",
    notes=(
        "Native Yukthi Opus instrumentation logs MCMC burn-in, post-burnin selection, "
        "MCMC proposal, greedy refinement, and SA acceptance as direct objective-improvement "
        "operators. Blacklist filtering, adaptive reheating, and elite updates are diagnostic "
        "operators with zero direct improvement unless the engine reports already-computed gains. "
        "The current EvoMapXProfile schema has no direct_operators/diagnostic_operators fields, "
        "so the separation is documented here and in result metadata."
    ),
)
