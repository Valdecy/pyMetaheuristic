"""pyMetaheuristic src — Pied Kingfisher Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class PKOEngine(PortedPopulationEngine):
    algorithm_id   = "pko"
    algorithm_name = "Pied Kingfisher Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1007/s00521-024-09879-5"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30, BF=8)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        BF=int(self._params.get("BF",8))
        best_idx=self._best_index(pop[:,-1]); best_pos=pop[best_idx,:-1].copy(); best_fit=pop[best_idx,-1]
        Crest_angles=2*np.pi*np.random.random()
        o=np.exp(-t/max_iter)**2; PEmax=0.5; PEmin=0
        for i in range(n):
            if np.random.random()<0.8:
                j=i
                while j==i: j=np.random.randint(n)
                beatingRate=np.random.random()*pop[j,-1]/(pop[i,-1]+1e-300)
                alpha=2*np.random.randn(d)-1
                if np.random.random()<0.5:
                    T=beatingRate-(t**(1/BF)/(max_iter**(1/BF)))
                    X_1=pop[i,:-1]+alpha*T*(pop[j,:-1]-pop[i,:-1])
                else:
                    T=(np.e-np.e**((t-1)/max_iter)**(1/BF))*np.cos(Crest_angles)
                    X_1=pop[i,:-1]+alpha*T*(pop[j,:-1]-pop[i,:-1])
            else:
                alpha=2*np.random.randn(d)-1
                b=pop[i,:-1]+o**2*np.random.randn()*best_pos
                HA=np.random.random()*pop[i,-1]/(best_fit+1e-300)
                X_1=pop[i,:-1]+HA*o*alpha*(b-best_pos)
            X_1=np.clip(X_1,lo,hi)
            new_fit=float(self._evaluate_population(X_1[None])[0]); evals+=1
            if self._is_better(new_fit,pop[i,-1]):
                pop[i]=np.append(X_1,new_fit)
            if self._is_better(pop[i,-1],best_fit):
                best_fit=pop[i,-1]; best_pos=pop[i,:-1].copy()
        PE=PEmax-(PEmax-PEmin)*(t/max_iter)
        for i in range(n):
            alpha=2*np.random.randn(d)-1
            if np.random.random()>(1-PE):
                X_1=pop[np.random.randint(n),:-1]+o*alpha*np.abs(pop[i,:-1]-pop[np.random.randint(n),:-1])
                X_1=np.clip(X_1,lo,hi)
                new_fit=float(self._evaluate_population(X_1[None])[0]); evals+=1
                if self._is_better(new_fit,pop[i,-1]):
                    pop[i]=np.append(X_1,new_fit)
                if self._is_better(pop[i,-1],best_fit):
                    best_fit=pop[i,-1]; best_pos=pop[i,:-1].copy()
        return pop, evals, {}
