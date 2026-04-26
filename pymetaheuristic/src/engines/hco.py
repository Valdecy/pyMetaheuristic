"""pyMetaheuristic src — Human Conception Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class HCOEngine(PortedPopulationEngine):
    """Human Conception Optimizer — PSO-like velocity with fitness-potential weighting."""
    algorithm_id   = "hco"
    algorithm_name = "Human Conception Optimizer"
    family         = "human"
    _REFERENCE     = {"doi": "10.1038/s41598-022-25031-6"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, wfp=0.65, wfv=0.05, c1=1.4, c2=1.4)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n, dim = pop.shape[0], self.problem.dimension
        return {"personal_best": pop.copy(),
                "velocity":      np.zeros((n, dim))}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        wfp     = float(self._params.get("wfp", 0.65))
        wfv     = float(self._params.get("wfv", 0.05))
        c1      = float(self._params.get("c1",  1.4))
        c2      = float(self._params.get("c2",  1.4))

        pb  = np.asarray(state.payload.get("personal_best", pop.copy()), dtype=float)
        vel = np.asarray(state.payload.get("velocity",      np.zeros((n, dim))), dtype=float)

        order    = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        best_fit = float(pop[order[0], -1])

        fits     = pop[:, -1]
        fit_mean = float(np.mean(fits))
        lam      = np.random.random()
        neu      = 2.0
        denom    = (best_fit - fit_mean) + 1e-30
        RR       = (best_fit - fits) ** 2
        rr       = (fit_mean - fits) ** 2
        ll       = RR - rr
        VV       = lam * (ll / (4.0 * neu * denom))

        new_pos = np.empty_like(pop[:, :-1])
        for i in range(n):
            a1 = pb[i, :-1] - pop[i, :-1]
            a2 = best_pos   - pop[i, :-1]
            s  = 2.0 * np.pi * t / T
            vel[i] = (wfv * (VV[i] + vel[i])
                      + c1 * a1 * np.sin(s)
                      + c2 * a2 * np.sin(s))
            pos = pop[i, :-1] + vel[i]
            new_pos[i] = np.clip(pos, self._lo, self._hi)

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack([new_pos, new_fit[:, None]])
        for i in range(n):
            if self._is_better(float(new_fit[i]), float(pop[i, -1])):
                pop[i] = new_pop[i]
                if self._is_better(float(new_fit[i]), float(pb[i, -1])):
                    pb[i] = new_pop[i]
        return pop, n, {"personal_best": pb, "velocity": vel}
