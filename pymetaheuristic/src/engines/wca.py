"""pyMetaheuristic src — Water Cycle Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class WCAEngine(PortedPopulationEngine):
    """Water Cycle Algorithm — sea/river/stream hierarchy with evaporation raining."""
    algorithm_id = "wca"; algorithm_name = "Water Cycle Algorithm"; family = "nature"
    _REFERENCE   = {"doi": "10.1016/j.compstruc.2012.07.010"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, nsr=4, wc=2.0, ecc=0.001)

    def _initialize_payload(self, pop):
        n=pop.shape[0]; nsr=max(2,min(int(self._params.get("nsr",4)),n//2))
        order=self._order(pop[:,-1])
        rivers=[int(order[i]) for i in range(1,nsr)]
        streams_idx=[int(order[i]) for i in range(nsr,n)]
        np.random.shuffle(streams_idx)
        chunk=max(1,len(streams_idx)//(nsr-1))
        stream_map={r:[] for r in rivers}
        for k,s in enumerate(streams_idx): stream_map[rivers[k//(chunk+1) if chunk else 0]].append(s)
        return {"sea":int(order[0]),"rivers":rivers,"stream_map":stream_map,"nsr":nsr}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        wc=float(self._params.get("wc",2.0)); ecc=float(self._params.get("ecc",0.001)); evals=0
        sea_idx=int(state.payload.get("sea",0))
        rivers=list(state.payload.get("rivers",[]))
        stream_map=dict(state.payload.get("stream_map",{}))
        nsr=int(state.payload.get("nsr",4))
        sea_pos=pop[sea_idx,:-1].copy()
        # Update streams toward their river
        for r_idx,streams in stream_map.items():
            r_pos=pop[int(r_idx),:-1]
            for s in streams:
                pos=np.clip(pop[s,:-1]+np.random.random()*wc*(r_pos-pop[s,:-1]), self._lo, self._hi)
                fit=float(self.problem.evaluate(pos)); evals+=1
                if self._is_better(fit,float(pop[s,-1])): pop[s,:-1]=pos; pop[s,-1]=fit
            # Best stream replaces river if better
            if streams:
                best_s=streams[self._best_index(pop[np.array(streams,int),-1])]
                if self._is_better(float(pop[best_s,-1]),float(pop[int(r_idx),-1])):
                    pop[[best_s,int(r_idx)]]=pop[[int(r_idx),best_s]]
            # Update river toward sea
            pos=np.clip(pop[int(r_idx),:-1]+np.random.random()*wc*(sea_pos-pop[int(r_idx),:-1]), self._lo, self._hi)
            fit=float(self.problem.evaluate(pos)); evals+=1
            if self._is_better(fit,float(pop[int(r_idx),-1])): pop[int(r_idx),:-1]=pos; pop[int(r_idx),-1]=fit
            # Evaporation / raining
            d=float(np.linalg.norm(sea_pos-pop[int(r_idx),:-1]))
            if d<ecc or np.random.random()<0.1:
                new_pos=np.random.uniform(self._lo,self._hi)
                new_fit=float(self.problem.evaluate(new_pos)); evals+=1
                pop[int(r_idx),:-1]=new_pos; pop[int(r_idx),-1]=new_fit
        # Check if any river is better than sea
        order=self._order(pop[:,-1]); sea_idx=int(order[0])
        return pop, evals, {"sea":sea_idx,"rivers":rivers,"stream_map":stream_map,"nsr":nsr}
