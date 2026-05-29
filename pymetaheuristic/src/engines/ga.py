"""pyMetaheuristic src — Genetic Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)

class GAEngine(BaseEngine):
    algorithm_id   = "ga"
    algorithm_name = "Genetic Algorithm"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.7551/mitpress/1090.001.0001"}
    capabilities   = CapabilityProfile(has_population=True,supports_candidate_injection=True)
    _DEFAULTS = dict(population_size=25, mutation_rate=0.1, elite=1, eta=1, mu=1)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p={**self._DEFAULTS,**config.params}
        self._n=int(p["population_size"]); self._mr=float(p["mutation_rate"])
        self._elite=int(p["elite"]); self._eta=float(p["eta"]); self._mu=float(p["mu"])
        if config.seed is not None: np.random.seed(config.seed)

    def _init_pop(self):
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        pos=np.random.uniform(lo,hi,(self._n,self.problem.dimension))
        fit=self._evaluate_population(pos)
        return np.hstack((pos,fit[:,np.newaxis]))

    def _fitness_fn(self, pop):
        mn=abs(pop[:,-1].min()); fc=1/(1+pop[:,-1]+mn)
        cs=np.cumsum(fc); cs/=cs[-1]; return np.column_stack((fc,cs))

    def _roulette(self, fit):
        r=np.random.rand()
        for i in range(fit.shape[0]):
            if r<=fit[i,1]: return i
        return fit.shape[0]-1

    def _breed(self, pop, fit):
        """Crossover pass.  Returns (offspring_pop, evals, lineage_records)."""
        off=np.copy(pop); lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values)
        lineage = []
        # Elite individuals carry over unchanged; record them as such.
        if self._elite>0:
            prs=pop[pop[:,-1].argsort()]
            off[:self._elite,:]=prs[:self._elite,:]
            for k in range(self._elite):
                lineage.append({
                    "id": f"crossover:elite:{k}",
                    "index": k,
                    "operator": "crossover",
                    "parent_ids": [f"parent:{k}"],
                    "parent_index": k,
                    "parent_fitness": float(prs[k,-1]),
                    "child_fitness": float(prs[k,-1]),
                })
        evals=0
        for i in range(self._elite,self._n):
            p1=self._roulette(fit); p2=self._roulette(fit)
            while p1==p2: p2=np.random.choice(self._n-1)
            parent_fit = min(float(pop[p1,-1]), float(pop[p2,-1]))
            for j in range(self.problem.dimension):
                r=np.random.rand(); rb=np.random.rand(); rc=np.random.rand()
                b=2*rb**(1/(self._mu+1)) if r<=0.5 else (1/(2*(1-rb)))**(1/(self._mu+1))
                if rc>=0.5: off[i,j]=np.clip(((1+b)*pop[p1,j]+(1-b)*pop[p2,j])/2,lo[j],hi[j])
                else: off[i,j]=np.clip(((1-b)*pop[p1,j]+(1+b)*pop[p2,j])/2,lo[j],hi[j])
            off[i,-1]=self.problem.evaluate(off[i,:-1]); evals+=1
            lineage.append({
                "id": f"crossover:{i}",
                "index": i,
                "operator": "crossover",
                "parent_ids": [f"parent:{p1}", f"parent:{p2}"],
                "parent_index": p1,
                "parent_fitness": parent_fit,
                "child_fitness": float(off[i,-1]),
            })
        return off, evals, lineage

    def _mutate(self, off, breed_lineage):
        """Mutation pass.  Updates lineage records in-place when a candidate changes."""
        lo=np.array(self.problem.min_values); hi=np.array(self.problem.max_values); evals=0
        # Build index→lineage map for O(1) lookup.
        lin_by_index = {rec["index"]: rec for rec in breed_lineage}
        for i in range(self._n):
            changed=False
            pre_fitness = float(off[i,-1])
            for j in range(self.problem.dimension):
                if np.random.rand()<self._mr:
                    r=np.random.rand(); rd=np.random.rand()
                    d=2*rd**(1/(self._eta+1))-1 if r<=0.5 else 1-2*(1-rd)**(1/(self._eta+1))
                    off[i,j]=np.clip(off[i,j]+d,lo[j],hi[j]); changed=True
            if changed:
                off[i,-1]=self.problem.evaluate(off[i,:-1]); evals+=1
                # Overwrite the lineage entry: mutation is now the last
                # operator that touched this candidate.
                rec = lin_by_index.get(i)
                if rec is not None:
                    rec["operator"] = "mutation"
                    rec["id"] = f"mutation:{i}"
                    rec["parent_fitness"] = pre_fitness
                    rec["child_fitness"] = float(off[i,-1])
                else:
                    breed_lineage.append({
                        "id": f"mutation:{i}",
                        "index": i,
                        "operator": "mutation",
                        "parent_ids": [f"parent:{i}"],
                        "parent_index": i,
                        "parent_fitness": pre_fitness,
                        "child_fitness": float(off[i,-1]),
                    })
        return off, evals

    def initialize(self):
        pop=self._init_pop(); bi=np.argmin(pop[:,-1])
        return EngineState(step=0,evaluations=self._n,
            best_position=pop[bi,:-1].tolist(),best_fitness=float(pop[bi,-1]),
            initialized=True,payload=dict(population=pop,elite=pop[bi,:].copy()))

    def step(self, state):
        pop=state.payload["population"]; elite=state.payload["elite"]
        fit=self._fitness_fn(pop)
        off, e1, lineage = self._breed(pop, fit)
        pop, e2 = self._mutate(off, lineage)
        bi=pop[pop[:,-1].argsort()][0,:]
        if bi[-1]<elite[-1]: elite=bi.copy()
        state.step+=1; state.evaluations+=e1+e2
        state.payload=dict(population=pop, elite=elite, lineage=lineage)
        if self.problem.is_better(float(elite[-1]),state.best_fitness):
            state.best_fitness=float(elite[-1]); state.best_position=elite[:-1].tolist()
        return state

    def observe(self, state):
        pop = state.payload["population"]
        fitness = pop[:, -1]
        pos = pop[:, :-1]
        lo = np.array(self.problem.min_values)
        hi = np.array(self.problem.max_values)
        denom = np.linalg.norm(hi - lo) or 1.0
        centroid = pos.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        lineage = state.payload.get("lineage", [])
        contrib: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rec in lineage:
            op = str(rec.get("operator") or "crossover")
            pf = rec.get("parent_fitness")
            cf = rec.get("child_fitness")
            if pf is not None and cf is not None:
                delta = float(pf) - float(cf)  # positive = improvement (minimisation)
            else:
                delta = 0.0
            contrib[op] = contrib.get(op, 0.0) + delta
            counts[op] = counts.get(op, 0) + 1
        obs = dict(
            step=state.step,
            evaluations=state.evaluations,
            best_fitness=state.best_fitness,
            mean_fitness=float(np.mean(fitness)),
            std_fitness=float(np.std(fitness)),
            diversity=diversity,
        )
        if contrib:
            obs["operator_contributions"] = contrib
            obs["operator_counts"] = counts
            obs["evomapx_delta_f"] = "signed"
            obs["evomapx_fidelity"] = "guessed_from_code_profiles_ga"
        return obs

    def get_best_candidate(self,state):
        return CandidateRecord(position=list(state.best_position),fitness=state.best_fitness,
            source_algorithm=self.algorithm_id,source_step=state.step,role="best")

    def finalize(self,state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),best_fitness=state.best_fitness,
            steps=state.step,evaluations=state.evaluations,
            termination_reason=state.termination_reason,capabilities=self.capabilities,
            metadata=dict(algorithm_name=self.algorithm_name,elapsed_time=state.elapsed_time))

    def get_population(self,state):
        pop=state.payload["population"]
        return [CandidateRecord(position=pop[i,:-1].tolist(),fitness=float(pop[i,-1]),
            source_algorithm=self.algorithm_id,source_step=state.step,role="current")
            for i in range(pop.shape[0])]
