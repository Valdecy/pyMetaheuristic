"""pyMetaheuristic src — Differential Evolution Engine"""
from __future__ import annotations
import numpy as np

from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class DEEngine(BaseEngine):
    algorithm_id   = "de"
    algorithm_name = "Differential Evolution"
    family         = "evolutionary"
    _REFERENCE     = {"doi": "10.1023/A:1008202821328"}
    capabilities   = CapabilityProfile(has_population=True)

    _DEFAULTS = dict(size=30, F=0.8, Cr=0.9, strategy="rand1bin", jitter=0.0)

    def __init__(self, problem, config):
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = max(4, int(p["size"]), 5 * int(problem.dimension))
        self._F = float(p["F"])
        self._Cr = float(p["Cr"])
        self._strategy = str(p.get("strategy", "rand1bin")).lower()
        self._jitter = max(0.0, float(p.get("jitter", 0.0)))
        if not (0.0 < self._F <= 2.0):
            raise ValueError("de F must be in (0, 2].")
        if not (0.0 <= self._Cr <= 1.0):
            raise ValueError("de Cr must be in [0, 1].")
        if self._strategy not in {"rand1bin", "best1bin", "current-to-best1bin"}:
            raise ValueError("de strategy must be 'rand1bin', 'best1bin', or 'current-to-best1bin'.")
        if config.seed is not None:
            np.random.seed(config.seed)

    @property
    def _lo(self):
        return np.asarray(self.problem.min_values, dtype=float)

    @property
    def _hi(self):
        return np.asarray(self.problem.max_values, dtype=float)

    def _init_pop(self, n=None):
        if n is None:
            n = self._n
        pos = np.random.uniform(self._lo, self._hi, (n, self.problem.dimension))
        fit = self._evaluate_population(pos)
        return np.hstack((pos, fit[:, np.newaxis]))

    def initialize(self):
        pop = self._init_pop()
        bi = int(np.argmin(pop[:, -1]) if self.problem.objective == "min" else np.argmax(pop[:, -1]))
        elite = pop[bi, :].copy()
        return EngineState(
            step=0,
            evaluations=self._n,
            best_position=elite[:-1].tolist(),
            best_fitness=float(elite[-1]),
            initialized=True,
            payload=dict(population=pop, elite=elite),
        )

    def _sample_donors(self, n: int, exclude: int, k: int) -> np.ndarray:
        pool = np.array([idx for idx in range(n) if idx != exclude], dtype=int)
        replace = pool.size < k
        return np.random.choice(pool, size=k, replace=replace)

    def _trial_vector(self, pop: np.ndarray, i: int, best: np.ndarray) -> np.ndarray:
        dim = self.problem.dimension
        ids = self._sample_donors(pop.shape[0], i, 3)
        x = pop[i, :-1]
        r1, r2, r3 = pop[ids[0], :-1], pop[ids[1], :-1], pop[ids[2], :-1]
        F = self._F
        if self._jitter > 0.0:
            F = float(np.clip(F + np.random.uniform(-self._jitter, self._jitter), 1.0e-12, 2.0))

        if self._strategy == "best1bin":
            mutant = best + F * (r1 - r2)
        elif self._strategy == "current-to-best1bin":
            mutant = x + F * (best - x) + F * (r1 - r2)
        else:
            mutant = r1 + F * (r2 - r3)

        cross = np.random.rand(dim) <= self._Cr
        cross[np.random.randint(dim)] = True
        trial = np.where(cross, mutant, x)
        return np.clip(trial, self._lo, self._hi)

    def step(self, state):
        pop = np.asarray(state.payload["population"], dtype=float).copy()
        elite = np.asarray(state.payload["elite"], dtype=float).copy()
        order = np.argsort(pop[:, -1])
        if self.problem.objective == "max":
            order = order[::-1]
        best_vec = pop[order[0], :-1].copy()

        evals = 0
        lineage = []
        for i in range(pop.shape[0]):
            parent_fitness = float(pop[i, -1])
            trial_pos = self._trial_vector(pop, i, best_vec)
            trial_fit = float(self.problem.evaluate(trial_pos.copy()))
            evals += 1
            accepted = self.problem.is_better(trial_fit, parent_fitness) or trial_fit == parent_fitness
            if accepted:
                pop[i, :-1] = trial_pos
                pop[i, -1] = trial_fit
                child_fitness = trial_fit
                # The trial vector is produced by mutation+crossover; crossover
                # selects which dimensions come from the mutant.  We record the
                # full pipeline as "de_crossover" because crossover is the final
                # mixing step that determines the accepted candidate's genes,
                # consistent with DE family schema labels.
                operator = "de_crossover"
            else:
                child_fitness = parent_fitness
                operator = "de_selection"  # target retained by greedy selection
            lineage.append({
                "id": f"de:{i}",
                "index": i,
                "operator": operator,
                "parent_ids": [f"parent:{i}"],
                "parent_index": i,
                "parent_fitness": parent_fitness,
                "child_fitness": child_fitness,
            })

        bi = int(np.argmin(pop[:, -1]) if self.problem.objective == "min" else np.argmax(pop[:, -1]))
        if self.problem.is_better(float(pop[bi, -1]), float(elite[-1])):
            elite = pop[bi, :].copy()

        state.step += 1
        state.evaluations += evals
        state.payload = dict(population=pop, elite=elite, lineage=lineage)
        if self.problem.is_better(float(elite[-1]), state.best_fitness):
            state.best_fitness = float(elite[-1])
            state.best_position = elite[:-1].tolist()
        return state

    def observe(self, state):
        pop = state.payload["population"]
        fitness = pop[:, -1]
        pos = pop[:, :-1]
        denom = np.linalg.norm(self._hi - self._lo) or 1.0
        centroid = pos.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)
        lineage = state.payload.get("lineage", [])
        contrib: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rec in lineage:
            op = str(rec.get("operator") or "de_crossover")
            pf = rec.get("parent_fitness")
            cf = rec.get("child_fitness")
            delta = (float(pf) - float(cf)) if (pf is not None and cf is not None) else 0.0
            contrib[op] = contrib.get(op, 0.0) + delta
            counts[op] = counts.get(op, 0) + 1
        # Mutation is the vector-difference step; crossover mixes it with the
        # target.  Both operators share the total gain of accepted trials equally
        # so that both appear in the OAM, matching the paper's analysis.
        if "de_crossover" in contrib:
            mutation_gain = contrib["de_crossover"] * 0.5
            contrib["de_mutation"] = contrib.get("de_mutation", 0.0) + mutation_gain
            counts["de_mutation"] = counts.get("de_mutation", 0) + counts.get("de_crossover", 0)
            contrib["de_crossover"] *= 0.5
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
            obs["evomapx_fidelity"] = "guessed_from_code_profiles_de"
        return obs

    def get_best_candidate(self, state):
        return CandidateRecord(position=list(state.best_position), fitness=state.best_fitness,
            source_algorithm=self.algorithm_id, source_step=state.step, role="best")

    def finalize(self, state):
        return OptimizationResult(algorithm_id=self.algorithm_id,
            best_position=list(state.best_position), best_fitness=state.best_fitness,
            steps=state.step, evaluations=state.evaluations,
            termination_reason=state.termination_reason, capabilities=self.capabilities,
            metadata=dict(algorithm_name=self.algorithm_name, elapsed_time=state.elapsed_time,
                          strategy=self._strategy, F=self._F, Cr=self._Cr))

    def get_population(self, state):
        pop = state.payload["population"]
        return [CandidateRecord(position=pop[i, :-1].tolist(), fitness=float(pop[i, -1]),
            source_algorithm=self.algorithm_id, source_step=state.step, role="current")
            for i in range(pop.shape[0])]
