"""pyMetaheuristic src — Mirage-Search Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class MSOEngine(PortedPopulationEngine):
    """Mirage-Search Optimizer — superior/inferior mirage search with triangle-angle geometry."""
    algorithm_id = "mso"; algorithm_name = "Mirage-Search Optimizer"; family = "physics"
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50)

    @staticmethod
    def _sind(deg): return np.sin(np.radians(deg))

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 500); t = state.step+1
        order = self._order(pop[:,-1]); best_pos=pop[order[0],:-1]
        ac=np.random.permutation(n-1)+1
        cv=max(1,int(np.ceil((n*(2/3))*((T-t+1)/T))))
        new_pop=pop.copy(); evals=0
        # Superior mirage search
        for idx in ac[:cv]:
            pos=np.empty(dim)
            for k in range(dim):
                h=float(best_pos[k]-pop[idx,k])*np.random.random()
                cmax=1; hmax=5*np.log(max(1e-9,abs(1-(t/T))))+cmax; h=np.clip(h,cmax,hmax)
                zf=np.random.choice([-1,1]); a=np.random.random()*20; b=np.random.random()*(45-a/2)
                z=np.random.randint(1,4); A=B=C=D=90
                if z==1: C=b+90;D=180-C-a;B=180-2*D;A=180-B+a-90
                elif z==2 and a<b: C=90-b;D=90+a-b;B=180-2*D;A=180-B-a-90
                elif z==2: C=90-b;D=180-C-a;B=180-2*D;A=180-B-90+a
                else: zf=0
                s_d=self._sind(A); dx=(self._sind(B)*h*self._sind(C))/(self._sind(D)*s_d+1e-30)*zf if s_d else 0
                pos[k]=pop[idx,k]+dx
            pos=np.clip(pos,self._lo,self._hi)
            fit=float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit,float(new_pop[idx,-1])): new_pop[idx,:-1]=pos; new_pop[idx,-1]=fit
        # Inferior mirage search (remaining)
        for idx in ac[cv:]:
            pos=np.clip(pop[idx,:-1]+np.random.uniform(-1,1,dim)*self._span*0.1, self._lo, self._hi)
            fit=float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit,float(new_pop[idx,-1])): new_pop[idx,:-1]=pos; new_pop[idx,-1]=fit
        return new_pop, evals, {}
