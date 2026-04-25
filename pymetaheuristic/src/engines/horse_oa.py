"""pyMetaheuristic src — Horse Herd Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class HorseOAEngine(PortedPopulationEngine):
    algorithm_id   = "horse_oa"
    algorithm_name = "Horse Herd Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2020.106711"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _initialize_payload(self, pop):
        n,d=pop.shape[0],self.problem.dimension
        VelMax=0.1*(self._hi-self._lo); VelMin=-VelMax
        return {"vel":np.zeros((n,d)),"pbest":pop[:,:-1].copy(),"pfit":pop[:,-1].copy(),"VelMax":VelMax,"VelMin":VelMin}

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi; evals = 0
        vel=state.payload["vel"]; pbest=state.payload["pbest"]
        pfit=state.payload["pfit"]; VelMax=state.payload["VelMax"]; VelMin=state.payload["VelMin"]
        gbest_idx=self._best_index(pfit); GlobalBest=pbest[gbest_idx].copy()
        order=self._order(pop[:,-1]); n10=max(1,round(0.1*n)); n30=max(1,round(0.3*n)); n60=max(1,round(0.6*n))
        MeanPos=np.mean(pop[:n,:-1],axis=0)
        BadPos=np.mean(pop[order[n-max(1,round(0.05*n)):],:-1],axis=0)
        GoodPos=np.mean(pop[order[:max(1,round(0.05*n))],:-1],axis=0)
        for i,oi in enumerate(order):
            rank=i+1
            if rank<=n10:
                vel[oi]=1.5*np.random.random(d)*(GlobalBest-pop[oi,:-1])-0.5*np.random.random(d)*pop[oi,:-1]+1.5*(0.95+0.1*np.random.random())*(pbest[oi]-pop[oi,:-1])
            elif rank<=n30:
                vel[oi]=0.2*np.random.random(d)*(MeanPos-pop[oi,:-1])-0.2*np.random.random(d)*(BadPos-pop[oi,:-1])+0.9*np.random.random(d)*(GlobalBest-pop[oi,:-1])+1.5*(0.95+0.1*np.random.random())*(pbest[oi]-pop[oi,:-1])
            elif rank<=n60:
                vel[oi]=0.1*np.random.random(d)*(MeanPos-pop[oi,:-1])+0.05*np.random.random(d)*pop[oi,:-1]-0.1*np.random.random(d)*(BadPos-pop[oi,:-1])+0.5*np.random.random(d)*(GlobalBest-pop[oi,:-1])+0.3*np.random.random(d)*(GoodPos-pop[oi,:-1])+1.5*(0.95+0.1*np.random.random())*(pbest[oi]-pop[oi,:-1])
            else:
                vel[oi]=0.05*np.random.random(d)*pop[oi,:-1]+1.5*(0.95+0.1*np.random.random())*(pbest[oi]-pop[oi,:-1])
            vel[oi]=np.clip(vel[oi],VelMin,VelMax)
            new_pos=pop[oi,:-1]+vel[oi]
            out=(new_pos<lo)|(new_pos>hi)
            vel[oi][out]*=-1
            new_pos=np.clip(new_pos,lo,hi)
            new_fit=float(self._evaluate_population(new_pos[None])[0]); evals+=1
            pop[oi]=np.append(new_pos,new_fit)
            if self._is_better(new_fit,pfit[oi]):
                pbest[oi]=new_pos.copy(); pfit[oi]=new_fit
                if self._is_better(new_fit,pfit[gbest_idx]):
                    gbest_idx=oi; GlobalBest=new_pos.copy()
        state.payload.update({"vel":vel,"pbest":pbest,"pfit":pfit})
        return pop, evals, {}
