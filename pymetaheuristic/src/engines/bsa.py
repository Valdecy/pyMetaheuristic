"""pyMetaheuristic src — Bird Swarm Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class BSAEngine(PortedPopulationEngine):
    """Bird Swarm Algorithm — foraging/vigilance flight with periodic producer-scrounger split."""
    algorithm_id = "bsa"; algorithm_name = "Bird Swarm Algorithm"; family = "swarm"
    _REFERENCE   = {"doi": "10.1080/0952813X.2015.1042530"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, ff=10, pff=0.8, c1=1.5, c2=1.5, a1=1.0, a2=1.0)

    def _initialize_payload(self, pop):
        return {"local_best": pop.copy()}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        t = state.step + 1
        ff = max(1, int(self._params.get("ff", 10)))
        pff = float(self._params.get("pff", 0.8))
        c1 = float(self._params.get("c1", 1.5)); c2 = float(self._params.get("c2", 1.5))
        a1 = float(self._params.get("a1", 1.0)); a2 = float(self._params.get("a2", 1.0))
        lb = np.asarray(state.payload.get("local_best", pop.copy()), dtype=float)
        order = self._order(pop[:, -1])
        best_pos = pop[order[0], :-1].copy()
        EPS = 1e-30

        if t % ff != 0:
            pos_mean = pop[:, :-1].mean(axis=0)
            fit_sum  = float(np.sum(pop[:, -1])) + EPS
            new_pos  = np.empty_like(pop[:, :-1])
            for i in range(n):
                prob = np.random.random() * 0.2 + pff
                if np.random.random() < prob:
                    pos = pop[i, :-1] + c1*np.random.random()*(lb[i,:-1]-pop[i,:-1]) + c2*np.random.random()*(best_pos-pop[i,:-1])
                else:
                    A1 = a1*np.exp(-n*float(pop[i,-1])/(EPS+fit_sum))
                    k  = np.random.choice([x for x in range(n) if x!=i])
                    t1 = (float(pop[i,-1])-float(pop[k,-1])) / (abs(float(pop[i,-1])-float(pop[k,-1]))+EPS)
                    A2 = a2*np.exp(t1*n*float(pop[k,-1])/(fit_sum+EPS))
                    pos = pop[i,:-1] + A1*np.random.random()*(pos_mean-pop[i,:-1]) + A2*np.random.uniform(-1,1)*(best_pos-pop[i,:-1])
                new_pos[i] = np.clip(pos, self._lo, self._hi)
            new_fit = self._evaluate_population(new_pos)
            new_pop  = np.hstack([new_pos, new_fit[:,None]])
            mask = self._better_mask(new_fit, pop[:,-1])
            pop[mask] = new_pop[mask]
            for i in range(n):
                if self._is_better(float(pop[i,-1]), float(lb[i,-1])): lb[i] = pop[i]
        else:
            fit_list = pop[:,-1]
            min_i = int(np.argmin(fit_list)); max_i = int(np.argmax(fit_list))
            choose = 1 + (min_i >= n//2) + 2*(max_i >= n//2)
            for i in range(n//2+1, n):
                pos = pop[i,:-1] + np.random.random(dim)*self._span + self._lo
                fit = float(self.problem.evaluate(np.clip(pos,self._lo,self._hi)))
                if self._is_better(fit, float(pop[i,-1])):
                    pop[i,:-1] = np.clip(pos,self._lo,self._hi); pop[i,-1] = fit
            for i in range(0, n//2):
                pos = pop[i,:-1] + np.random.random(dim)*(best_pos - pop[i,:-1]) + np.random.uniform(-1,1,dim)*(pop[min_i,:-1]-pop[i,:-1])
                pos = np.clip(pos,self._lo,self._hi)
                fit = float(self.problem.evaluate(pos))
                if self._is_better(fit, float(pop[i,-1])): pop[i,:-1]=pos; pop[i,-1]=fit
        return pop, n, {"local_best": lb}
