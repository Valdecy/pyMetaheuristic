"""pyMetaheuristic src — Hippopotamus Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy(n,d,beta=1.5):
    num=gamma(1+beta)*np.sin(np.pi*beta/2)
    den=gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma=(num/den)**(1/beta)
    u=np.random.randn(n,d)*sigma; v=np.random.randn(n,d)
    return u/np.abs(v)**(1/beta)

class HippoEngine(PortedPopulationEngine):
    algorithm_id   = "ho_hippo"
    algorithm_name = "Hippopotamus Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1038/s41598-024-55040-6"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=30)

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        lo, hi = self._lo, self._hi
        evals = 0
        t = state.step; max_iter=self._params.get("max_iterations",1000)
        best_idx=self._best_index(pop[:,-1]); dom=pop[best_idx,:-1].copy()
        RL=_levy(n,d,1.5)
        for i in range(n//2):
            I1=np.random.randint(1,3); I2=np.random.randint(1,3)
            k=np.random.randint(n)
            grp=np.random.choice(n,k+1,replace=False)
            mean_g=np.mean(pop[grp,:-1],axis=0)
            A=np.random.random(d)*I2; B=2*np.random.random(d)-1
            rnd=np.random.random(d); rnd2=np.random.random()
            alfs=[I2*np.random.random(d)+~np.random.randint(0,2,d).astype(bool),
                  2*np.random.random(d)-1, np.random.random(d),
                  I1*np.random.random(d)+~np.random.randint(0,2,d).astype(bool),
                  np.array([np.random.random()])]
            Af=alfs[np.random.randint(len(alfs))]; Bf=alfs[np.random.randint(len(alfs))]
            X_P1=np.clip(pop[i,:-1]+np.random.random()*(dom-I1*pop[i,:-1]),lo,hi)
            T=np.exp(-t/max_iter)
            if T>0.6:
                X_P2=np.clip(pop[i,:-1]+Af*(dom-I2*mean_g),lo,hi)
            else:
                if np.random.random()>0.5:
                    X_P2=np.clip(pop[i,:-1]+Bf*(mean_g-dom),lo,hi)
                else:
                    X_P2=np.random.uniform(lo,hi)
            f1=float(self._evaluate_population(X_P1[None])[0]); evals+=1
            f2=float(self._evaluate_population(X_P2[None])[0]); evals+=1
            if self._is_better(f1,pop[i,-1]): pop[i]=np.append(X_P1,f1)
            if self._is_better(f2,pop[i,-1]): pop[i]=np.append(X_P2,f2)
        for i in range(n//2,n):
            pred=np.random.uniform(lo,hi)
            f_pred=float(self._evaluate_population(pred[None])[0]); evals+=1
            dist=np.abs(pred-pop[i,:-1])
            b=np.random.uniform(2,4); c=np.random.uniform(1,1.5)
            dd=np.random.uniform(2,3); ll=np.random.uniform(-2*np.pi,2*np.pi)
            if self._is_better(pop[i,-1],f_pred):
                X_P3=np.clip(RL[i]*pred+(b/(c-dd*np.cos(ll)))*(1/(dist+1e-300)),lo,hi)
            else:
                X_P3=np.clip(RL[i]*pred+(b/(c-dd*np.cos(ll)))*(1/(2*dist+np.random.random(d)+1e-300)),lo,hi)
            f3=float(self._evaluate_population(X_P3[None])[0]); evals+=1
            if self._is_better(f3,pop[i,-1]): pop[i]=np.append(X_P3,f3)
        LO_LOC=lo/(t+1); HI_LOC=hi/(t+1)
        for i in range(n):
            D=np.random.random(d)*2-1
            X_P4=np.clip(pop[i,:-1]+np.random.random()*(LO_LOC+D*(HI_LOC-LO_LOC)),lo,hi)
            f4=float(self._evaluate_population(X_P4[None])[0]); evals+=1
            if self._is_better(f4,pop[i,-1]): pop[i]=np.append(X_P4,f4)
        return pop, evals, {}
