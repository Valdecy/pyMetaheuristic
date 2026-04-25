"""
pyMetaheuristic src — Adaptive Exploration State-Space Particle Swarm Optimization Engine
==========================================================================================
Native macro-step: compute adaptive inertia weight w via bisection on eigenvalue criterion,
                   then PSO velocity/position update
payload keys: population (ndarray [N, D+1]), velocity (ndarray [N, D]),
              pbest (ndarray [N, D+1])
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


def _w_adapter(N: int, beta: np.ndarray, gamma: np.ndarray, tol: float = 1e-4) -> float:
    """Bisection search for stability-maximising inertia weight (per AESSPSO paper)."""
    W_min, W_max = -10.0, 10.0
    diff = 1.0
    while diff > tol:
        omega = (W_min + W_max) / 2.0
        # Eigenvalue criterion: min singular value of control matrix
        # A = [[diag(1-β-γ), ωI], [-diag(β+γ), ωI]]
        # B = [[diag(β), γ], [diag(β), γ]]   (simplified N=1 per-particle)
        b_arr = np.array([beta.mean(), gamma.mean()])
        # Use scalar approximation (mean across particles) for efficiency
        b1, g1 = b_arr[0], b_arr[1]
        # B column stack, A columns
        B = np.array([[b1, g1], [b1, g1]])
        A = np.array([[1 - b1 - g1, omega], [-(b1 + g1), omega]])
        Q = np.hstack([B, A @ B])
        sv_om = np.linalg.svd(Q, compute_uv=False).min()

        omega_l = omega - 1e-7
        A_l = np.array([[1 - b1 - g1, omega_l], [-(b1 + g1), omega_l]])
        Q_l = np.hstack([B, A_l @ B])
        sv_left = np.linalg.svd(Q_l, compute_uv=False).min()

        if sv_om > sv_left:
            diff = omega - W_min
            W_min = omega
        else:
            diff = W_max - omega
            W_max = omega
    return (W_min + W_max) / 2.0


class AESSPSOEngine(BaseEngine):
    algorithm_id   = "aesspso"
    algorithm_name = "Adaptive Exploration State-Space Particle Swarm Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, beta=2.05, gamma=2.05)
    _REFERENCE     = dict(doi="10.1016/j.swevo.2025.101868")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p          = {**self._DEFAULTS, **config.params}
        self._n    = max(4, int(p["population_size"]))
        self._beta = float(p["beta"])
        self._gamma= float(p["gamma"])
        if config.seed is not None:
            np.random.seed(config.seed)

    def initialize(self) -> EngineState:
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        vel = np.zeros((self._n, self.problem.dimension))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True,
                           payload=dict(population=pop, velocity=vel, pbest=pop.copy()))

    def step(self, state: EngineState) -> EngineState:
        pop   = np.array(state.payload["population"])
        vel   = np.array(state.payload["velocity"])
        pbest = np.array(state.payload["pbest"])
        lo    = np.array(self.problem.min_values, dtype=float)
        hi    = np.array(self.problem.max_values, dtype=float)
        N, D  = pop.shape[0], self.problem.dimension

        # Global best from pbest
        gi    = int(np.argmin(pbest[:, -1]))
        gbest = pbest[gi, :-1]

        # Per-particle adaptive weights
        r1 = self._beta  * np.random.rand(N, 1)
        r2 = self._gamma * np.random.rand(N, 1)

        # Scalar w (stability-adapted)
        try:
            w = _w_adapter(N, r1.ravel(), r2.ravel())
        except Exception:
            w = 0.5  # safe fallback

        new_vel = (w * vel
                   + r1 * (pbest[:, :-1] - pop[:, :-1])
                   + r2 * (gbest - pop[:, :-1]))
        new_pos = np.clip(pop[:, :-1] + new_vel, lo, hi)
        new_fit = self._evaluate_population(new_pos)

        # Update pbest
        better = new_fit < pbest[:, -1] if self.problem.objective == "min" else new_fit > pbest[:, -1]
        pbest[better, :-1] = new_pos[better]
        pbest[better, -1]  = new_fit[better]

        pop[:, :-1] = new_pos
        pop[:, -1]  = new_fit

        bi  = int(np.argmin(pbest[:, -1]))
        bf  = float(pbest[bi, -1])
        bp  = pbest[bi, :-1].tolist()

        state.payload      = dict(population=pop, velocity=new_vel, pbest=pbest)
        state.evaluations += N
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
