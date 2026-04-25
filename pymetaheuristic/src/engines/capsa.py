"""pyMetaheuristic src — Capuchin Search Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class CapSAEngine(PortedPopulationEngine):
    algorithm_id   = "capsa"
    algorithm_name = "Capuchin Search Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00521-020-05066-5"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        return {"v": 0.1*pop[:, :-1].copy(), "v0": np.zeros_like(pop[:, :-1]),
                "pbest": pop[:, :-1].copy(), "pfit": pop[:, -1].copy(),
                "gbest": pop[self._best_index(pop[:, -1]), :-1].copy()}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step; max_iter = self._params.get("max_iterations", 1000)
        v = state.payload["v"]; v0 = state.payload["v0"]
        pbest = state.payload["pbest"]; pfit  = state.payload["pfit"]
        gbest = state.payload["gbest"]
        bf=0.70; cr=11.0; g=9.81; a1=1.25; a2=1.5
        tau = 2*np.exp(-11*t/max_iter)**2
        w   = 0.8 - 0.7*(t/max_iter)
        wmax=0.8; wmin=0.1; w = wmax-(wmax-wmin)*(t/max_iter)
        fol = np.random.randint(0, n, n)
        v += a1*(pbest-pop[:, :-1])*np.random.random((n,d)) + a2*(gbest-pop[:, :-1])*np.random.random((n,d))
        new_pos = pop[:, :-1].copy()
        for i in range(n):
            r = np.random.random()
            if i < n//2:
                if np.random.random() >= 0.1:
                    if r <= 0.15:
                        new_pos[i] = gbest + bf*(v[i]**2*np.sin(2*np.random.random()*1.5))/g
                    elif r <= 0.30:
                        new_pos[i] = gbest + cr*bf*(v[i]**2*np.sin(2*np.random.random()*1.5))/g
                    elif r <= 0.90:
                        new_pos[i] = pop[i, :-1] + v[i]
                    elif r <= 0.95:
                        new_pos[i] = gbest + bf*np.sin(np.random.random()*1.5)
                    else:
                        new_pos[i] = gbest + bf*(v[i]-v0[i])
                else:
                    new_pos[i] = tau*(lo + np.random.random(d)*(hi-lo))
            else:
                eps = ((np.random.random()+2*np.random.random())-(3*np.random.random()))/(1+np.random.random())
                fi = fol[i]
                prev = max(0, i-1)
                pos2 = gbest + 2*(pbest[fi]-pop[i, :-1])*eps + 2*(pop[i, :-1]-pbest[i])*eps
                new_pos[i] = (pos2+new_pos[prev])/2
        v0 = v.copy()
        new_pos = np.clip(new_pos, lo, hi)
        new_fits = self._evaluate_population(new_pos); evals += n
        mask = self._better_mask(new_fits, pfit)
        pbest[mask] = new_pos[mask]; pfit[mask] = new_fits[mask]
        pop = np.hstack([new_pos, new_fits[:, None]])
        bi = self._best_index(pfit)
        if self._is_better(pfit[bi], float(self._evaluate_population(gbest[None])[0])):
            gbest = pbest[bi].copy()
        state.payload.update({"v":v,"v0":v0,"pbest":pbest,"pfit":pfit,"gbest":gbest})
        return pop, evals, {}
