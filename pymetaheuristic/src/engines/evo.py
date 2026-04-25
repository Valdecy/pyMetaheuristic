"""pyMetaheuristic src — Energy Valley Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class EVOEngine(PortedPopulationEngine):
    """Energy Valley Optimizer — team/global-mean guided update scaled by stability level."""
    algorithm_id = "evo"; algorithm_name = "Energy Valley Optimizer"; family = "physics"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        EPS=1e-30; evals=0
        order = self._order(pop[:,-1]); best_pos=pop[order[0],:-1]; best_fit=float(pop[order[0],-1])
        worst_fit=float(pop[order[-1],-1]); pos_list=pop[:,:-1]; fit_list=pop[:,-1]
        pos_mean=pos_list.mean(axis=0)
        dists_sq=np.sum((pos_list[:,None,:]-pos_list[None,:,:])**2,axis=2)
        sorted_d=np.argsort(dists_sq,axis=1)
        eb=float(np.mean(fit_list))
        new_pop=pop.copy()
        for i in range(n):
            fi=float(fit_list[i]); sl=(fi-best_fit)/(worst_fit-best_fit+EPS)
            cn_pt=np.random.choice(list(set(range(2,n))-{i})) if n>3 else max(1,n-1)
            x_team=pos_list[sorted_d[i,1:cn_pt+1]].mean(axis=0) if cn_pt>0 else pos_mean
            if self._is_better(eb, fi):
                if np.random.random()>sl:
                    a1=np.random.randint(dim); a2idx=np.random.randint(0,dim,max(1,a1))
                    p1=pop[i,:-1].copy(); p1[a2idx]=best_pos[a2idx]
                    g1=np.random.randint(dim); g2idx=np.random.randint(0,dim,max(1,g1))
                    p2=pop[i,:-1].copy(); p2[g2idx]=x_team[g2idx]
                else:
                    ir=np.random.random(2); jr=np.random.random(dim)
                    p1=pop[i,:-1]+jr*(ir[0]*best_pos-ir[1]*pos_mean)/max(sl,EPS)
                    ir2=np.random.random(2); jr2=np.random.random(dim)
                    p2=pop[i,:-1]+jr2*(ir2[0]*best_pos-ir2[1]*x_team)
                for pp in [np.clip(p1,self._lo,self._hi), np.clip(p2,self._lo,self._hi)]:
                    ff=float(self.problem.evaluate(pp)); evals+=1
                    if self._is_better(ff, float(new_pop[i,-1])): new_pop[i,:-1]=pp; new_pop[i,-1]=ff
            else:
                pos3=np.clip(pop[i,:-1]+np.random.random()*sl*np.random.uniform(self._lo,self._hi), self._lo, self._hi)
                ff=float(self.problem.evaluate(pos3)); evals+=1
                if self._is_better(ff, float(new_pop[i,-1])): new_pop[i,:-1]=pos3; new_pop[i,-1]=ff
        return new_pop, evals, {}
