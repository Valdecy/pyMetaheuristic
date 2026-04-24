"""pyMetaheuristic src — Success-History Adaptive Differential Evolution Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine, de_trial


class SHADEEngine(PortedPopulationEngine):
    """Success-History Adaptive Differential Evolution."""
    algorithm_id = "shade"
    algorithm_name = "Success-History Adaptive Differential Evolution"
    family = "evolutionary"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True, supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=100, extern_arc_rate=2.6, pbest_factor=0.11, hist_mem_size=6)

    def _initialize_payload(self, pop):
        h = int(self._params.get("hist_mem_size", 6))
        return {"M_F": np.full(h, 0.5), "M_CR": np.full(h, 0.5), "k": 0, "archive": np.empty((0, self.problem.dimension))}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        M_F = np.asarray(state.payload.get("M_F", np.full(6, 0.5)), dtype=float)
        M_CR = np.asarray(state.payload.get("M_CR", np.full(6, 0.5)), dtype=float)
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim)
        k = int(state.payload.get("k", 0)) % len(M_F)
        order = self._order(pop[:, -1])
        pnum = max(2, int(float(self._params.get("pbest_factor", 0.11)) * n))
        union = np.vstack((pop[:, :-1], archive)) if archive.size else pop[:, :-1]
        trials, Fs, CRs = [], [], []
        for i in range(n):
            r = np.random.randint(len(M_F))
            F = np.clip(np.random.standard_cauchy() * 0.1 + M_F[r], 0.05, 1.0)
            CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
            pbest = pop[np.random.choice(order[:pnum]), :-1]
            r1 = pop[self._rand_indices(n, i, 1)[0], :-1]
            r2 = union[np.random.randint(union.shape[0])]
            mutant = pop[i, :-1] + F * (pbest - pop[i, :-1]) + F * (r1 - r2)
            cross = np.random.rand(dim) < CR; cross[np.random.randint(dim)] = True
            trials.append(np.clip(np.where(cross, mutant, pop[i, :-1]), self._lo, self._hi)); Fs.append(F); CRs.append(CR)
        trial_pop = self._pop_from_positions(np.asarray(trials))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        if np.any(mask):
            archive = np.vstack((archive, pop[mask, :-1])) if archive.size else pop[mask, :-1].copy()
            max_arc = max(1, int(float(self._params.get("extern_arc_rate", 2.6)) * n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            sf, scr = np.asarray(Fs)[mask], np.asarray(CRs)[mask]
            df = np.abs(pop[mask, -1] - trial_pop[mask, -1]); w = df / (df.sum() + 1e-30)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % len(M_F)
            pop[mask] = trial_pop[mask]
        return pop, n, {"M_F": M_F, "M_CR": M_CR, "k": k, "archive": archive}
