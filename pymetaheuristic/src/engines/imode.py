"""
pyMetaheuristic src — Improved Multi-Operator Differential Evolution Engine
============================================================================
Native macro-step: linear pop-size reduction, adaptive CR/F/operator probabilities,
three DE mutation operators selected probabilistically, archive-assisted mutation
payload keys: population (ndarray [N,D+1]), archive (ndarray [A,D+1]),
              MCR (ndarray [M]), MF (ndarray [M]), MOP (ndarray [3]), k (int)
"""
from __future__ import annotations
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class IMODEEngine(BaseEngine):
    algorithm_id   = "imode"
    algorithm_name = "Improved Multi-Operator Differential Evolution"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1109/CEC48606.2020.9185577"}
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=100, min_population=4, archive_rate=2.6, memory_size=None)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p          = {**self._DEFAULTS, **config.params}
        self._n0   = max(8, int(p["population_size"]))
        self._minN = max(4, int(p["min_population"]))
        self._aRate= float(p["archive_rate"])
        self._maxFE= getattr(config, "max_evaluations", None)
        M          = 20 * problem.dimension
        if config.seed is not None:
            np.random.seed(config.seed)
        self._M = M

    def initialize(self) -> EngineState:
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N   = self._n0
        pos = np.random.uniform(lo, hi, (N, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        M   = self._M
        return EngineState(step=0, evaluations=N,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True,
                           payload=dict(
                               population=pop,
                               archive=np.empty((0, self.problem.dimension + 1)),
                               MCR=np.full(M, 0.2),
                               MF=np.full(M, 0.2),
                               MOP=np.full(3, 1.0 / 3),
                               k=0,
                           ))

    def _sample_cr_f(self, N: int, MCR: np.ndarray, MF: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M   = len(MCR)
        idx = np.random.randint(M, size=N)
        CR  = np.clip(np.random.randn(N) * np.sqrt(0.1) + MCR[idx], 0, 1)
        CR  = np.sort(CR)
        F   = np.minimum(1.0, np.random.standard_t(1, N) * np.sqrt(0.1) + MF[idx])
        while np.any(F <= 0):
            bad = F <= 0
            F[bad] = np.minimum(1.0, np.random.standard_t(1, bad.sum()) * np.sqrt(0.1) + MF[idx[bad]])
        return CR, F

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        arc = np.array(state.payload["archive"])
        MCR = np.array(state.payload["MCR"])
        MF  = np.array(state.payload["MF"])
        MOP = np.array(state.payload["MOP"])
        k   = int(state.payload["k"])
        M   = len(MCR)

        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension

        # Linear population-size reduction
        FE = state.evaluations
        if self._maxFE is not None and self._maxFE > 0:
            N_target = int(np.ceil((self._minN - self._n0) * FE / self._maxFE) + self._n0)
            N_target = max(self._minN, min(N_target, pop.shape[0]))
        else:
            N_target = pop.shape[0]

        order = np.argsort(pop[:, -1])
        pop   = pop[order[:N_target]]
        N     = pop.shape[0]

        # Archive trimming
        aMax = max(1, int(np.ceil(self._aRate * N)))
        if arc.shape[0] > aMax:
            arc = arc[np.random.permutation(arc.shape[0])[:aMax]]

        # Sample CR, F, operator assignment
        CR, F = self._sample_cr_f(N, MCR, MF)
        # Operator probabilities cumulative
        op_cum = np.cumsum(MOP); op_cum /= op_cum[-1]
        OP_assign = np.array([np.searchsorted(op_cum, np.random.rand()) for _ in range(N)])
        op_groups = [np.where(OP_assign == i)[0] for i in range(3)]

        # Helper index arrays
        Xp1 = pop[np.random.randint(max(1, N // 4), size=N), :-1]
        Xp2 = pop[np.random.randint(max(2, N // 2), size=N), :-1]
        Xr1 = pop[np.random.randint(N, size=N), :-1]
        Xr3 = pop[np.random.randint(N, size=N), :-1]
        combined = pop if arc.shape[0] == 0 else np.vstack((pop, arc))
        Xr2 = combined[np.random.randint(combined.shape[0], size=N), :-1]

        F_mat  = F[:, None] * np.ones((N, D))
        CR_mat = CR[:, None] * np.ones((N, D))

        # Mutation
        off = pop[:, :-1].copy()
        g0  = op_groups[0]
        g1  = op_groups[1]
        g2  = op_groups[2]
        if len(g0) > 0:
            off[g0] = pop[g0, :-1] + F_mat[g0] * (Xp1[g0] - pop[g0, :-1] + Xr1[g0] - Xr2[g0])
        if len(g1) > 0:
            off[g1] = pop[g1, :-1] + F_mat[g1] * (Xp1[g1] - pop[g1, :-1] + Xr1[g1] - Xr3[g1])
        if len(g2) > 0:
            off[g2] = F_mat[g2] * (Xr1[g2] + Xp2[g2] - Xr3[g2])

        # Crossover
        if np.random.rand() < 0.4:
            mask = np.random.rand(N, D) > CR_mat
            off[mask] = pop[:, :-1][mask]
        else:
            for i in range(N):
                p1 = np.random.randint(D)
                p2 = next((j for j in range(D + 1) if np.random.rand() > CR[i]), D)
                site = list(range(p1)) + list(range(p1 + p2, D))
                if site:
                    off[i, site] = pop[i, site]

        off = np.clip(off, lo, hi)
        off_fit = self._evaluate_population(off)

        # Update archive and population
        pop_fit = pop[:, -1]
        delta   = pop_fit - off_fit
        replace = delta > 0

        if replace.sum() > 0:
            arc = np.vstack((arc, pop[replace])) if arc.shape[0] > 0 else pop[replace].copy()
        if arc.shape[0] > aMax:
            arc = arc[np.random.permutation(arc.shape[0])[:aMax]]

        pop[replace, :-1] = off[replace]
        pop[replace, -1]  = off_fit[replace]

        # Adapt MCR, MF
        if replace.sum() > 0:
            w_arr = delta[replace] / max(delta[replace].sum(), 1e-30)
            MCR[k] = (w_arr @ CR[replace] ** 2) / max(w_arr @ CR[replace], 1e-30)
            MF[k]  = (w_arr @ F[replace] ** 2) / max(w_arr @ F[replace], 1e-30)
            k = (k + 1) % M
        else:
            MCR[k] = 0.5; MF[k] = 0.5

        # Adapt MOP
        pop_fit_new = pop[:, -1]
        delta_norm  = np.maximum(0, delta / np.abs(pop_fit_new + 1e-30))
        if any(len(g) == 0 for g in op_groups):
            MOP = np.full(3, 1.0 / 3)
        else:
            MOP = np.array([delta_norm[g].mean() if len(g) > 0 else 0.0 for g in op_groups])
            MOP = np.clip(MOP / max(MOP.sum(), 1e-30), 0.1, 0.9)
            MOP /= MOP.sum()

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, archive=arc, MCR=MCR, MF=MF, MOP=MOP, k=k)
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
                    mean_fitness=float(np.mean(pop[:, -1])),
                    population_size=pop.shape[0])

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
