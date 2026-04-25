"""pyMetaheuristic src — Snake Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class SnakeOptimizerEngine(PortedPopulationEngine):
    algorithm_id   = "so_snake"
    algorithm_name = "Snake Optimizer"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2022.108320"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        vec_flag=[-1,1]; Threshold=0.25; Thresold2=0.6
        C1=0.5; C2=0.05; C3=2
        Temp=np.exp(-t/max_iter); Q=min(1,C1*np.exp((t-max_iter)/max_iter))
        Nm=n//2; Nf=n-Nm
        order=self._order(pop[:,-1])
        food=pop[order[0],:-1].copy(); best_f=pop[order[0],-1]
        Xm=pop[:Nm,:-1].copy(); Xf=pop[Nm:,:-1].copy()
        fm=pop[:Nm,-1].copy(); ff=pop[Nm:,-1].copy()
        bm_idx=np.argmin(fm); bf_idx=np.argmin(ff)
        Xbest_m=Xm[bm_idx].copy(); Xbest_f=Xf[bf_idx].copy()
        fbest_m=fm[bm_idx]; fbest_f=ff[bf_idx]
        Xnewm=np.empty_like(Xm); Xnewf=np.empty_like(Xf)
        if Q<Threshold:
            for i in range(Nm):
                for j in range(d):
                    ri=np.random.randint(Nm); Am=np.exp(-fm[ri]/(fm[i]+1e-300))*C2
                    Xnewm[i,j]=Xm[ri,j]+np.random.choice(vec_flag)*Am*((hi[j]-lo[j])*np.random.random()+lo[j])
            for i in range(Nf):
                for j in range(d):
                    ri=np.random.randint(Nf); Af=np.exp(-ff[ri]/(ff[i]+1e-300))*C2
                    Xnewf[i,j]=Xf[ri,j]+np.random.choice(vec_flag)*Af*((hi[j]-lo[j])*np.random.random()+lo[j])
        else:
            if Temp>Thresold2:
                for i in range(Nm):
                    flag=np.random.choice(vec_flag)
                    for j in range(d):
                        Xnewm[i,j]=food[j]+C3*flag*Temp*np.random.random()*(food[j]-Xm[i,j])
                for i in range(Nf):
                    flag=np.random.choice(vec_flag)
                    for j in range(d):
                        Xnewf[i,j]=food[j]+flag*C3*Temp*np.random.random()*(food[j]-Xf[i,j])
            else:
                if np.random.random()>0.6:
                    for i in range(Nm):
                        for j in range(d):
                            FM=np.exp(-fbest_f/(fm[i]+1e-300))
                            Xnewm[i,j]=Xm[i,j]+C3*FM*np.random.random()*(Q*Xbest_f[j]-Xm[i,j])
                    for i in range(Nf):
                        for j in range(d):
                            FF=np.exp(-fbest_m/(ff[i]+1e-300))
                            Xnewf[i,j]=Xf[i,j]+C3*FF*np.random.random()*(Q*Xbest_m[j]-Xf[i,j])
                else:
                    for i in range(Nm):
                        for j in range(d):
                            Mm=np.exp(-ff[min(i,Nf-1)]/(fm[i]+1e-300))
                            Xnewm[i,j]=Xm[i,j]+C3*np.random.random()*Mm*(Q*Xf[min(i,Nf-1),j]-Xm[i,j])
                    for i in range(Nf):
                        for j in range(d):
                            Mf=np.exp(-fm[min(i,Nm-1)]/(ff[i]+1e-300))
                            Xnewf[i,j]=Xf[i,j]+C3*np.random.random()*Mf*(Q*Xm[min(i,Nm-1),j]-Xf[i,j])
                    if np.random.choice(vec_flag)==1:
                        gworst_m=int(np.argmax(fm))
                        Xnewm[gworst_m]=np.random.uniform(lo,hi)
                        gworst_f=int(np.argmax(ff))
                        Xnewf[gworst_f]=np.random.uniform(lo,hi)
        Xnewm=np.clip(Xnewm,lo,hi); Xnewf=np.clip(Xnewf,lo,hi)
        fm_new=self._evaluate_population(Xnewm); evals+=Nm
        ff_new=self._evaluate_population(Xnewf); evals+=Nf
        maskm=self._better_mask(fm_new,fm); maskf=self._better_mask(ff_new,ff)
        Xm[maskm]=Xnewm[maskm]; fm[maskm]=fm_new[maskm]
        Xf[maskf]=Xnewf[maskf]; ff[maskf]=ff_new[maskf]
        pop=np.vstack([np.hstack([Xm,fm[:,None]]),np.hstack([Xf,ff[:,None]])])
        return pop, evals, {}
