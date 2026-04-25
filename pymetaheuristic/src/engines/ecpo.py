"""
pyMetaheuristic src — Electric Charged Particles Optimization Engine
=====================================================================
Native macro-step: Coulomb-force interaction between randomly selected charged particles,
                   archive-based crossover, truncation selection
payload keys: population (ndarray [N, D+1])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class ECPOEngine(BaseEngine):
    algorithm_id   = "ecpo"
    algorithm_name = "Electric Charged Particles Optimization"
    family         = "physics"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, strategy=2, n_interact=3, archive_frac=None)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p          = {**self._DEFAULTS, **config.params}
        self._n    = max(4, int(p["population_size"]))
        self._strat= int(p["strategy"])
        self._ni   = int(p["n_interact"])
        af         = p["archive_frac"]
        self._naECP= max(1, int(af * self._n) if af else self._n // 3)
        if config.seed is not None:
            np.random.seed(config.seed)

    def _operator(self, pop: np.ndarray, archive: np.ndarray) -> np.ndarray:
        N, D       = pop.shape[0], self.problem.dimension
        nECPI      = self._ni
        strat      = self._strat
        naECP      = archive.shape[0]
        new_dec    = []

        if strat == 1:
            pop_fac = 2 * (nECPI * (nECPI - 1) // 2)
        elif strat == 2:
            pop_fac = nECPI
        else:
            pop_fac = 2 * (nECPI * (nECPI - 1) // 2) + nECPI

        for _ in range(max(1, N // pop_fac)):
            Force  = np.random.normal(0.7, 0.2)
            SP     = np.sort(np.random.choice(N, size=nECPI, replace=False))
            batch  = []
            if strat == 1:
                for ii in range(nECPI):
                    for jj in range(nECPI):
                        s1 = pop[SP[ii]] + Force * (pop[0] - pop[SP[ii]])
                        if jj < ii:
                            s1 = s1 + Force * (pop[SP[jj]] - pop[SP[ii]])
                            batch.append(s1)
                        elif jj > ii:
                            s1 = s1 - Force * (pop[SP[jj]] - pop[SP[ii]])
                            batch.append(s1)
            elif strat == 2:
                for ii in range(nECPI):
                    s1 = pop[SP[ii]].copy()
                    for jj in range(nECPI):
                        if jj < ii:
                            s1 = s1 + Force * (pop[SP[jj]] - pop[SP[ii]])
                        elif jj > ii:
                            s1 = s1 - Force * (pop[SP[jj]] - pop[SP[ii]])
                    batch.append(s1)
            else:
                b1, b2 = [], []
                for ii in range(nECPI):
                    s2 = pop[SP[ii]] + Force * (pop[0] - pop[SP[ii]])
                    for jj in range(nECPI):
                        s1 = pop[SP[ii]] + Force * (pop[0] - pop[SP[ii]])
                        if jj < ii:
                            s1 = s1 + Force * (pop[SP[jj]] - pop[SP[ii]])
                            s2 = s2 + Force * (pop[SP[jj]] - pop[SP[ii]])
                            b1.append(s1)
                        elif jj > ii:
                            s1 = s1 - Force * (pop[SP[jj]] - pop[SP[ii]])
                            s2 = s2 - Force * (pop[SP[jj]] - pop[SP[ii]])
                            b1.append(s1)
                    b2.append(s2)
                batch = b1 + b2
            new_dec.extend(batch)

        if not new_dec:
            return np.empty((0, D))
        # pop rows are [D+1] arrays; extract only the position part (first D elements)
        arr = np.array([row[:-1] if len(row) == D + 1 else row for row in new_dec])

        # Archive crossover (20% chance per gene)
        if naECP > 0:
            for i in range(arr.shape[0]):
                mask = np.random.rand(D) < 0.2
                if mask.any():
                    pos = np.random.randint(naECP)
                    arr[i, mask] = archive[pos, mask]

        return arr

    def initialize(self) -> EngineState:
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        order = np.argsort(pop[:, -1])
        pop   = pop[order]
        bi    = 0
        return EngineState(step=0, evaluations=self._n,
                           best_position=pop[bi, :-1].tolist(), best_fitness=float(pop[bi, -1]),
                           initialized=True, payload=dict(population=pop))

    def step(self, state: EngineState) -> EngineState:
        pop  = np.array(state.payload["population"])
        lo   = np.array(self.problem.min_values, dtype=float)
        hi   = np.array(self.problem.max_values, dtype=float)
        N    = pop.shape[0]
        D    = self.problem.dimension

        # Sort: best first
        order = np.argsort(pop[:, -1])
        pop   = pop[order]
        archive = pop[:self._naECP, :-1]

        new_pos = self._operator(pop, archive)
        if new_pos.shape[0] == 0:
            # fallback: random perturbation
            new_pos = np.random.uniform(lo, hi, (N, D))

        new_pos = np.clip(new_pos[:, :D], lo, hi)
        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack((new_pos, new_fit[:, None]))

        combined = np.vstack((archive[:, :D], new_pos))
        combined_full = np.vstack((pop[:self._naECP], new_pop))
        combined_all  = np.vstack((pop, new_pop))
        order2  = np.argsort(combined_all[:, -1])
        pop     = combined_all[order2[:N]]

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop)
        state.evaluations += new_pos.shape[0]
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations,
                    best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])))

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(position=list(state.best_position),
                               fitness=state.best_fitness,
                               source_algorithm=self.algorithm_id,
                               source_step=state.step, role="best")

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position),
                                  best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason,
                                  capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name,
                                                elapsed_time=state.elapsed_time))

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = state.payload["population"]
        return [CandidateRecord(position=pop[i, :-1].tolist(), fitness=float(pop[i, -1]),
                                source_algorithm=self.algorithm_id,
                                source_step=state.step, role="current")
                for i in range(pop.shape[0])]
