"""
pyMetaheuristic src — Neural Network-Based Dimensionality Reduction Evolutionary Algorithm (SO) Engine
=======================================================================================================
Native macro-step: Stage 1 — evolve NN weights mapping low→high-dim binary; Stage 2 — direct GA on solutions
Binary encoding warning: designed for binary problems; real-valued proxy used.
payload keys: population (ndarray [N, D+1]), weights (ndarray [N, W]), stage (int)
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


def _fc_forward(weights: np.ndarray, x: np.ndarray, structure: list[int]) -> np.ndarray:
    """Fully-connected forward pass, sigmoid activations, output rounded to {0,1}."""
    out = x.copy()
    ptr = 0
    for i in range(len(structure) - 1):
        n_in, n_out = structure[i], structure[i + 1]
        W = weights[ptr: ptr + n_in * n_out].reshape(n_in, n_out); ptr += n_in * n_out
        b = weights[ptr: ptr + n_out]; ptr += n_out
        out = 1.0 / (1.0 + np.exp(-np.clip(out @ W + b, -50, 50)))
    return np.round(out)


class NNDREASOEngine(BaseEngine):
    algorithm_id   = "nndrea_so"
    algorithm_name = "Neural Network-Based Dimensionality Reduction Evolutionary Algorithm (SO)"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30, delta=0.5, lower=-1.0, upper=1.0)
    _REFERENCE     = dict(doi="10.1109/TEVC.2024.3400398")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p         = {**self._DEFAULTS, **config.params}
        self._n   = max(4, int(p["population_size"]))
        self._delta = float(p["delta"])
        self._wlo = float(p["lower"])
        self._whi = float(p["upper"])
        D = problem.dimension
        # NN structure: input=4 (proxy features), hidden=4, output=D
        self._structure = [4, 4, D]
        self._W_dim = 4 * 4 + 4 + 4 * D + D  # weights + biases
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once("binary",
            f"[{self.algorithm_id}] This algorithm operates on binary strings. "
            "Decision variables are rounded to 0/1 internally and may not respect continuous bounds.")
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension

        # Stage 1: evolve NN weights; proxy input = 4 random features
        pop_w = np.random.uniform(self._wlo, self._whi, (self._n, self._W_dim))
        inp   = np.random.rand(self._n, 4)
        pos   = np.array([_fc_forward(pop_w[i], inp[i], self._structure) for i in range(self._n)])
        pos   = np.clip(pos, lo, hi)
        fit   = self._evaluate_population(pos)
        pop   = np.hstack((pos, fit[:, None]))
        bi    = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True,
                           payload=dict(population=pop, weights=pop_w, stage=1,
                                        max_evals_stage1=None))

    def step(self, state: EngineState) -> EngineState:
        pop   = np.array(state.payload["population"])
        pop_w = np.array(state.payload["weights"])
        stage = int(state.payload["stage"])
        lo    = np.array(self.problem.min_values, dtype=float)
        hi    = np.array(self.problem.max_values, dtype=float)
        N, D  = pop.shape[0], self.problem.dimension

        # Decide stage (delta fraction of evaluations in stage 1)
        max_eval_s1 = state.payload.get("max_eval_stage1", None)

        if stage == 1:
            # Evolve NN weights via DE
            F  = 0.8; CR = 0.9
            off_w = pop_w.copy()
            for i in range(N):
                idxs = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
                mutant = pop_w[idxs[0]] + F * (pop_w[idxs[1]] - pop_w[idxs[2]])
                cross  = np.random.rand(self._W_dim) < CR
                cross[np.random.randint(self._W_dim)] = True
                trial  = np.where(cross, mutant, pop_w[i])
                trial  = np.clip(trial, self._wlo, self._whi)
                # Decode through NN
                inp    = np.random.rand(4)
                pos_t  = np.clip(_fc_forward(trial, inp, self._structure), lo, hi)
                ft     = self.problem.evaluate(pos_t)
                if self.problem.is_better(ft, pop[i, -1]):
                    off_w[i]     = trial
                    pop[i, :-1]  = pos_t
                    pop[i, -1]   = ft

            # Check if we should switch to stage 2
            if state.evaluations >= int(state.step * N * self._delta + N):
                stage = 2
            pop_w = off_w
            evals = N
        else:
            # Stage 2: direct GA on binary decision variables
            F = 0.8; CR = 0.9
            for i in range(N):
                idxs = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
                mutant = pop[idxs[0], :-1] + F * (pop[idxs[1], :-1] - pop[idxs[2], :-1])
                cross  = np.random.rand(D) < CR
                cross[np.random.randint(D)] = True
                trial  = np.clip(np.round(np.where(cross, mutant, pop[i, :-1])), lo, hi)
                ft     = self.problem.evaluate(trial)
                if self.problem.is_better(ft, pop[i, -1]):
                    pop[i, :-1] = trial
                    pop[i, -1]  = ft
            evals = N

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, weights=pop_w, stage=stage)
        state.evaluations += evals
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    stage=state.payload["stage"], mean_fitness=float(np.mean(pop[:, -1])))

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


class SACCEAMIIEngine(BaseEngine):
    """Surrogate-Assisted Cooperative Co-Evolutionary Algorithm — variable decomposition + RBF surrogate fallback."""
    algorithm_id   = "sacc_eam2"
    algorithm_name = "Surrogate-Assisted Cooperative Co-Evolutionary Algorithm of Minamo II"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=20, s=50)
    _REFERENCE     = dict(doi="10.1007/978-3-319-97773-7_4")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._s  = max(1, int(p["s"]))  # number of subcomponents
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once("surrogate",
            f"[{self.algorithm_id}] This algorithm normally delegates expensive evaluations to a surrogate model. "
            "No surrogate is registered; all evaluations use the true objective function, "
            "so the budget may be exhausted faster than intended.")
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension

        # Variable grouping: sequential round-robin
        groups = [[] for _ in range(min(self._s, D))]
        for d in range(D):
            groups[d % len(groups)].append(d)

        pos = np.random.uniform(lo, hi, (self._n, D))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True,
                           payload=dict(population=pop, groups=groups, global_best=pos[bi].copy()))

    def step(self, state: EngineState) -> EngineState:
        pop    = np.array(state.payload["population"])
        groups = state.payload["groups"]
        gbest  = np.array(state.payload["global_best"])
        lo     = np.array(self.problem.min_values, dtype=float)
        hi     = np.array(self.problem.max_values, dtype=float)
        N, D   = pop.shape[0], self.problem.dimension
        evals  = 0

        # Optimise each subcomponent independently with DE
        for grp in groups:
            grp  = list(grp)
            sub_n= min(N, len(grp) + 1)
            F    = 0.8; CR = 0.9

            # DE on subcomponent, keeping global context from gbest
            sub_best = gbest.copy()
            for _ in range(min(2, N)):
                i    = np.random.randint(N)
                idxs = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
                trial= gbest.copy()
                for d in grp:
                    if np.random.rand() < CR:
                        trial[d] = pop[idxs[0], d] + F * (pop[idxs[1], d] - pop[idxs[2], d])
                trial = np.clip(trial, lo, hi)
                ft    = self.problem.evaluate(trial)
                evals += 1
                if self.problem.is_better(ft, pop[i, -1]):
                    pop[i, :-1] = trial
                    pop[i, -1]  = ft
                    if self.problem.is_better(ft, float(self.problem.evaluate(sub_best))):
                        sub_best = trial.copy()
                        evals   += 1
            # Update global best from subcomponent optimisation
            candidate = gbest.copy()
            for d in grp:
                candidate[d] = sub_best[d]
            fc = self.problem.evaluate(candidate)
            evals += 1
            if self.problem.is_better(fc, self.problem.evaluate(gbest)):
                gbest = candidate.copy()
                evals += 1

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        # Also check gbest
        gf = self.problem.evaluate(gbest)
        evals += 1
        if self.problem.is_better(gf, bf):
            bf = gf; bp = gbest.tolist()
        else:
            bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, groups=groups, global_best=gbest)
        state.evaluations += evals
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


class SSIORLEngine(BaseEngine):
    """Search Space Independent Operator RL — requires trained RL agent; falls back to adaptive DE."""
    algorithm_id   = "ssio_rl"
    algorithm_name = "Search Space Independent Operator Based Deep Reinforcement Learning"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=30)
    _REFERENCE     = dict(doi="https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125444")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p        = {**self._DEFAULTS, **config.params}
        self._n  = max(4, int(p["population_size"]))
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def initialize(self) -> EngineState:
        self._warn_once("rl_agent",
            f"[{self.algorithm_id}] This algorithm normally selects variation operators using a trained "
            "deep reinforcement learning agent loaded from an external weight file. No agent is available; "
            "falling back to adaptive DE with random operator selection.")
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop, F=0.8, CR=0.9))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        F   = float(state.payload["F"])
        CR  = float(state.payload["CR"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N, D = pop.shape[0], self.problem.dimension
        bi_g  = int(np.argmin(pop[:, -1]))
        best  = pop[bi_g, :-1]

        # SSIO-RL fallback: adaptive DE/current-to-best/1/bin
        off = np.empty((N, D))
        for i in range(N):
            idxs   = np.random.choice([j for j in range(N) if j != i], 2, replace=False)
            r1, r2 = pop[idxs[0], :-1], pop[idxs[1], :-1]
            mutant = pop[i, :-1] + F * (best - pop[i, :-1]) + F * (r1 - r2)
            cross  = np.random.rand(D) < CR
            cross[np.random.randint(D)] = True
            off[i] = np.clip(np.where(cross, mutant, pop[i, :-1]), lo, hi)

        off_fit = self._evaluate_population(off)
        better  = off_fit < pop[:, -1] if self.problem.objective == "min" else off_fit > pop[:, -1]
        pop[better, :-1] = off[better]
        pop[better, -1]  = off_fit[better]

        # Adapt F and CR slightly
        F  = np.clip(F  + 0.1 * (np.random.rand() - 0.5), 0.3, 1.0)
        CR = np.clip(CR + 0.1 * (np.random.rand() - 0.5), 0.1, 1.0)

        bi = int(np.argmin(pop[:, -1]))
        bf = float(pop[bi, -1])
        bp = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop, F=F, CR=CR)
        state.evaluations += N
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        pop = state.payload["population"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    mean_fitness=float(np.mean(pop[:, -1])), F=state.payload["F"], CR=state.payload["CR"])

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
