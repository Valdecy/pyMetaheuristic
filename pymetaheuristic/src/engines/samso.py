"""
pyMetaheuristic src — Multiswarm-Assisted Surrogate Optimization Engine
========================================================================
Surrogate warning: no RBF surrogate; PSO runs on true objective.
payload keys: population (ndarray [N,D+1]), velocity (ndarray [N,D]), pbest (ndarray [N,D+1])
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class SAMSOEngine(BaseEngine):
    algorithm_id   = "samso"
    algorithm_name = "Multiswarm-Assisted Expensive Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=40, Wnc=1, Pr=0.5)
    _REFERENCE     = dict(doi="10.1109/TCYB.2019.2950169")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p       = {**self._DEFAULTS, **config.params}
        self._n = max(4, int(p["population_size"]))
        self._w = float(p["Wnc"])
        self._pr= float(p["Pr"])
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
        pos = np.random.uniform(lo, hi, (self._n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        vel = (np.random.rand(self._n, self.problem.dimension) - 0.5) * (hi - lo)
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

        gi    = int(np.argmin(pbest[:, -1]))
        gbest = pbest[gi, :-1]
        c1, c2 = 2.0, 2.0
        r1 = np.random.rand(N, D); r2 = np.random.rand(N, D)
        vel = self._w * vel + c1 * r1 * (pbest[:, :-1] - pop[:, :-1]) + c2 * r2 * (gbest - pop[:, :-1])
        new_pos = np.clip(pop[:, :-1] + vel, lo, hi)
        new_fit = self._evaluate_population(new_pos)

        better = new_fit < pbest[:, -1] if self.problem.objective == "min" else new_fit > pbest[:, -1]
        pbest[better, :-1] = new_pos[better]
        pbest[better, -1]  = new_fit[better]
        pop[:, :-1] = new_pos; pop[:, -1] = new_fit

        bi = int(np.argmin(pbest[:, -1]))
        state.payload      = dict(population=pop, velocity=vel, pbest=pbest)
        state.evaluations += N
        state.step        += 1
        if self.problem.is_better(float(pbest[bi, -1]), state.best_fitness):
            state.best_fitness  = float(pbest[bi, -1])
            state.best_position = pbest[bi, :-1].tolist()
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


class L2SMEAEngine(BaseEngine):
    """Linear Subspace Surrogate Modeling EA — fallback: CMA-ES style steps without Kriging."""
    algorithm_id   = "l2smea"
    algorithm_name = "Linear Subspace Surrogate Modeling Evolutionary Algorithm"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=None, NLinear=8)
    _REFERENCE     = dict(doi="10.1109/TEVC.2024.3354543")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p         = {**self._DEFAULTS, **config.params}
        D         = problem.dimension
        n0        = p["population_size"]
        self._n   = max(4, int(n0) if n0 else max(8, 2 * D))
        self._NL  = int(p["NLinear"])
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
        # Latin-hypercube style init
        pos = lo + (np.random.permutation(self._n)[:, None] + np.random.rand(self._n, D)) / self._n * (hi - lo)
        pos = np.clip(pos, lo, hi)
        fit = self._evaluate_population(pos)
        arc = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        sigma = 0.5
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True,
                           payload=dict(archive=arc, sigma=sigma,
                                        mean=pos[bi].copy(), C=np.eye(D)))

    def step(self, state: EngineState) -> EngineState:
        arc   = np.array(state.payload["archive"])
        sigma = float(state.payload["sigma"])
        mean  = np.array(state.payload["mean"])
        C     = np.array(state.payload["C"])
        lo    = np.array(self.problem.min_values, dtype=float)
        hi    = np.array(self.problem.max_values, dtype=float)
        D     = self.problem.dimension
        lam   = 4 * self._NL

        # CMA-ES style sampling on linear subspaces
        eigv, B = np.linalg.eigh(C)
        eigv    = np.maximum(eigv, 1e-10)
        z       = np.random.randn(lam, D)
        steps   = (B * np.sqrt(eigv)) @ z.T
        cands   = np.clip(mean + sigma * steps.T, lo, hi)
        c_fit   = self._evaluate_population(cands)

        # Best mu
        mu      = lam // 2
        order   = np.argsort(c_fit)
        w       = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w       /= w.sum()
        mstep   = w @ steps.T[order[:mu]]
        mean    = mean + sigma * mstep

        arc     = np.vstack((arc, np.hstack((cands, c_fit[:, None]))))
        bi      = int(np.argmin(arc[:, -1]))
        bf      = float(arc[bi, -1])
        bp      = arc[bi, :-1].tolist()

        state.payload      = dict(archive=arc, sigma=sigma, mean=mean, C=C)
        state.evaluations += lam
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        arc = state.payload["archive"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    archive_size=arc.shape[0])

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), state.best_fitness, self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        arc = state.payload["archive"]
        return [CandidateRecord(arc[i, :-1].tolist(), float(arc[i, -1]),
                                self.algorithm_id, state.step, "current") for i in range(arc.shape[0])]


class MiSACOEngine(BaseEngine):
    """Multi-Surrogate Assisted Ant Colony Optimization — fallback: ACO-style without surrogate."""
    algorithm_id   = "misaco"
    algorithm_name = "Multi-Surrogate-Assisted Ant Colony Optimization"
    family         = "swarm"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=60, No=100)
    _REFERENCE     = dict(doi="10.1109/TCYB.2020.3035521")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p       = {**self._DEFAULTS, **config.params}
        self._n = max(4, int(p["population_size"]))
        self._No= max(4, int(p["No"]))
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
        pos = lo + (np.random.permutation(self._n)[:, None] + np.random.rand(self._n, D)) / self._n * (hi - lo)
        pos = np.clip(pos, lo, hi)
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(population=pop))

    def step(self, state: EngineState) -> EngineState:
        pop = np.array(state.payload["population"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N, D = pop.shape[0], self.problem.dimension

        order = np.argsort(pop[:, -1])
        pop   = pop[order]

        # ACO-style Gaussian sampling around ranked solutions
        sigma = np.std(pop[:, :-1], axis=0) / max(N, 1) + 1e-12
        q, xi = 1.0, 0.85
        w     = np.exp(-np.arange(N) ** 2 / (2 * q ** 2 * N ** 2))
        w    /= w.sum()
        cands = []
        for _ in range(min(self._No, 3)):  # evaluate 3 candidates per step to keep budget sane
            i      = np.random.choice(N, p=w)
            cand   = pop[i, :-1] + xi * sigma * np.random.randn(D)
            cands.append(np.clip(cand, lo, hi))

        cands    = np.array(cands)
        cand_fit = self._evaluate_population(cands)
        # Best evaluates truly
        bi_c    = int(np.argmin(cand_fit))
        new_row = np.hstack((cands[bi_c], [cand_fit[bi_c]]))
        pop     = np.vstack((pop, new_row))
        order2  = np.argsort(pop[:, -1])
        pop     = pop[order2[:N]]

        bi  = int(np.argmin(pop[:, -1]))
        bf  = float(pop[bi, -1])
        bp  = pop[bi, :-1].tolist()

        state.payload      = dict(population=pop)
        state.evaluations += len(cands)
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


class SAPOEngine(BaseEngine):
    """Surrogate-Assisted Partial Optimization — fallback: DE with alternating feasibility/objective focus."""
    algorithm_id   = "sapo"
    algorithm_name = "Surrogate-Assisted Partial Optimization"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=100, F=0.5, CR=0.9, initsize1=100, initsize2=200)
    _REFERENCE     = dict(doi="10.1007/978-3-031-70085-9_22")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p         = {**self._DEFAULTS, **config.params}
        D         = problem.dimension
        n0        = p["population_size"]
        init1     = int(p["initsize1"]); init2 = int(p["initsize2"])
        self._n0  = init2 if D >= 100 else init1
        self._n   = max(4, int(n0))
        self._F   = float(p["F"])
        self._CR  = float(p["CR"])
        self._warned: set[str] = set()
        if config.seed is not None:
            np.random.seed(config.seed)

    def _warn_once(self, key, msg):
        if key not in self._warned:
            warnings.warn(msg, stacklevel=3)
            self._warned.add(key)

    def _lhs_init(self, n: int) -> np.ndarray:
        """Stratified uniform LHS approximation."""
        lo = np.array(self.problem.min_values, dtype=float)
        hi = np.array(self.problem.max_values, dtype=float)
        D  = self.problem.dimension
        pos = np.zeros((n, D))
        for d in range(D):
            pos[:, d] = lo[d] + (np.random.permutation(n) + np.random.rand(n)) / n * (hi[d] - lo[d])
        return np.clip(pos, lo, hi)

    def initialize(self) -> EngineState:
        self._warn_once("surrogate",
            f"[{self.algorithm_id}] This algorithm normally delegates expensive evaluations to a surrogate model. "
            "No surrogate is registered; all evaluations use the true objective function, "
            "so the budget may be exhausted faster than intended.")
        pos = self._lhs_init(self._n0)
        fit = self._evaluate_population(pos)
        arc = np.hstack((pos, fit[:, None]))
        bi  = int(np.argmin(fit))
        return EngineState(step=0, evaluations=self._n0,
                           best_position=pos[bi].tolist(), best_fitness=float(fit[bi]),
                           initialized=True, payload=dict(archive=arc, gen=0))

    def step(self, state: EngineState) -> EngineState:
        arc = np.array(state.payload["archive"])
        gen = int(state.payload["gen"])
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        D   = self.problem.dimension
        N   = min(self._n, arc.shape[0])
        F, CR = self._F, self._CR

        # Take best N as working population
        order = np.argsort(arc[:, -1])
        pop   = arc[order[:N]]

        # Alternating DE mutation strategies (partial optimization mimic)
        bi_g = int(np.argmin(pop[:, -1]))
        best = pop[bi_g, :-1]

        if gen % 2 == 0:
            # current-to-best/1
            idxs  = np.array([np.random.choice([j for j in range(N) if j != i], 2, replace=False)
                               for i in range(N)])
            r1, r2 = pop[idxs[:, 0], :-1], pop[idxs[:, 1], :-1]
            mutant = pop[:, :-1] + F * (best - pop[:, :-1]) + F * (r1 - r2)
        else:
            # rand/1
            idxs  = np.array([np.random.choice([j for j in range(N) if j != i], 3, replace=False)
                               for i in range(N)])
            r1, r2, r3 = pop[idxs[:, 0], :-1], pop[idxs[:, 1], :-1], pop[idxs[:, 2], :-1]
            mutant = r1 + F * (r2 - r3)

        cross = np.random.rand(N, D) < CR
        cross[np.arange(N), np.random.randint(D, size=N)] = True
        trial = np.where(cross, mutant, pop[:, :-1])
        trial = np.clip(trial, lo, hi)
        t_fit = self._evaluate_population(trial)

        # Greedy replacement
        better = t_fit < pop[:, -1] if self.problem.objective == "min" else t_fit > pop[:, -1]
        pop[better, :-1] = trial[better]
        pop[better, -1]  = t_fit[better]

        arc = np.vstack((arc, pop[better]))
        bi  = int(np.argmin(arc[:, -1]))
        bf  = float(arc[bi, -1])
        bp  = arc[bi, :-1].tolist()

        state.payload      = dict(archive=arc, gen=gen + 1)
        state.evaluations += N
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state):
        arc = state.payload["archive"]
        return dict(step=state.step, evaluations=state.evaluations, best_fitness=state.best_fitness,
                    archive_size=arc.shape[0])

    def get_best_candidate(self, state):
        return CandidateRecord(list(state.best_position), state.best_fitness, self.algorithm_id, state.step, "best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position), best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason, capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time))

    def get_population(self, state):
        arc = state.payload["archive"]
        return [CandidateRecord(arc[i, :-1].tolist(), float(arc[i, -1]),
                                self.algorithm_id, state.step, "current") for i in range(arc.shape[0])]
