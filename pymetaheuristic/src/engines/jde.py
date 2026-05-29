"""pyMetaheuristic src — Self-Adaptive Differential Evolution Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class JDEEngine(PortedPopulationEngine):
    """Self-Adaptive Differential Evolution — individual F and CR adaptation."""
    algorithm_id = "jde"
    algorithm_name = "Self-Adaptive Differential Evolution"
    family = "evolutionary"
    _REFERENCE     = {"doi": "10.1109/TEVC.2006.872133"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, f_lower=0.0, f_upper=1.0, tao1=0.4, tao2=0.2, crossover_probability=0.9, differential_weight=0.5)

    def _initialize_payload(self, pop):
        n = pop.shape[0]
        return {"F": np.full(n, float(self._params.get("differential_weight", 0.5))), "CR": np.full(n, float(self._params.get("crossover_probability", 0.9)))}

    def _step_impl(self, state, pop):
        n = pop.shape[0]
        F = np.asarray(state.payload.get("F", np.full(n, 0.5)), dtype=float)
        CR = np.asarray(state.payload.get("CR", np.full(n, 0.9)), dtype=float)
        if F.shape[0] != n: F = np.full(n, 0.5)
        if CR.shape[0] != n: CR = np.full(n, 0.9)
        newF, newCR = F.copy(), CR.copy()
        trials = []
        attempted_labels = ["jde.de_trial"] * n
        for i in range(n):
            f_adapted = False
            cr_adapted = False
            if np.random.rand() < float(self._params.get("tao1", 0.4)):
                newF[i] = float(self._params.get("f_lower", 0.0)) + np.random.rand() * float(self._params.get("f_upper", 1.0))
                f_adapted = True
            if np.random.rand() < float(self._params.get("tao2", 0.2)):
                newCR[i] = np.random.rand()
                cr_adapted = True
            if f_adapted and cr_adapted:
                attempted_labels[i] = "jde.f_cr_self_adaptation_trial"
            elif f_adapted:
                attempted_labels[i] = "jde.f_self_adaptation_trial"
            elif cr_adapted:
                attempted_labels[i] = "jde.cr_self_adaptation_trial"
            trials.append(de_trial(self, pop, i, newF[i], newCR[i]))
        trial_pop = self._pop_from_positions(np.asarray(trials))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        pop[mask] = trial_pop[mask]
        F[mask], CR[mask] = newF[mask], newCR[mask]
        operator_labels = [attempted_labels[i] if bool(mask[i]) else "carryover" for i in range(n)]
        return pop, n, {"F": F, "CR": CR, "operator_labels": operator_labels}
