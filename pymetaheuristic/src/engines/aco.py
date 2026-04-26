"""
pyMetaheuristic src — Ant Colony Optimization Engine
=====================================================
Native macro-step: ants build tours using pheromone + heuristic → pheromone update
Cost-matrix warning: no pairwise distance matrix; cost approximated from objective function.
payload keys: population (ndarray [N, D+1]), tau (ndarray [D, D])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class ACOEngine(BaseEngine):
    algorithm_id   = "aco"
    algorithm_name = "Ant Colony Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/CEC.1999.782657"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=False)
    _DEFAULTS      = dict(population_size=20, rho=0.5, alpha=1.0, beta=2.0)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._rho= float(p["rho"])
        self._a  = float(p["alpha"])
        self._b  = float(p["beta"])
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once("cost_matrix",
            f"[{self.algorithm_id}] This algorithm is designed for permutation problems with a pairwise cost matrix. "
            "The cost matrix is approximated from the objective function, which may not reflect true tour cost.")
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension
        # Random real-valued initialization
        pos = np.random.uniform(lo, hi, (self._n, D))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        tau = np.ones((D, D))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop, tau=tau))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        tau = np.array(state.payload["tau"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension
        N   = pop.shape[0]

        new_pos = np.zeros((N, D))
        for i in range(N):
            # ACO-style real-valued random walk guided by pheromone
            x = np.random.uniform(lo, hi)
            for j in range(D):
                # pheromone-weighted perturbation in each dimension
                ph = tau[j, :].copy(); ph[j] = 0
                if ph.sum() > 0:
                    ph /= ph.sum()
                    ref_dim = np.random.choice(D, p=ph)
                    x[j] = np.clip(pop[np.random.randint(N), ref_dim] +
                                   np.random.randn() * (hi[j] - lo[j]) * 0.1, lo[j], hi[j])
            new_pos[i] = x

        new_fit = self._evaluate_population(new_pos)
        new_pop = np.hstack((new_pos, new_fit[:, None]))

        # Pheromone update: deposit on best solutions' dimension correlations
        dTau = np.full((D, D), 1e-6)
        order = np.argsort(new_fit)
        for i in order[:max(1, N // 4)]:
            obj_val = abs(new_fit[i]) + 1e-30
            for j in range(D - 1):
                d1 = int((new_pos[i, j] - lo[j]) / max(hi[j] - lo[j], 1e-10) * (D - 1))
                d2 = int((new_pos[i, j + 1] - lo[j + 1]) / max(hi[j + 1] - lo[j + 1], 1e-10) * (D - 1))
                d1 = max(0, min(d1, D - 1))
                d2 = max(0, min(d2, D - 1))
                dTau[d1, d2] += 1.0 / obj_val

        tau = self._rho * tau + dTau

        # Keep best survivors
        combined = np.vstack((pop, new_pop))
        order2   = np.argsort(combined[:, -1])
        pop      = combined[order2[:N]]

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, tau=tau)
        state.evaluations += N
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])))

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), state.best_fitness, self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        pop = state.payload["population"]
        return [CandidateRecord(pop[i, :-1].tolist(), float(pop[i, -1]),
                                self.algorithm_id, state.step, "current") for i in range(pop.shape[0])]
