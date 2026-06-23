"""
pyMetaheuristic src — Multifactorial Evolutionary Algorithm Engine
===================================================================
Native macro-step: unified multitask population → assortative mating / rmp
cross-task transfer → vertical cultural transmission → selective evaluation →
elitist scalar-fitness replacement.

This file intentionally contains only MFEAEngine.  MFEA-II is kept in the
separate mfea2.py engine module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import warnings
import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


@dataclass
class _MFEATask:
    target_function: Callable[[list[float]], float]
    min_values: np.ndarray
    max_values: np.ndarray
    objective: str = "min"
    name: str = "task"
    problem_spec: ProblemSpec | None = None

    @property
    def dimension(self) -> int:
        return int(self.min_values.size)


class MFEAEngine(BaseEngine):
    """Multifactorial Evolutionary Algorithm (MFEA).

    The implementation follows Gupta, Ong and Feng's MFEA: individuals live in
    a unified random-key space [0, 1]^D, are ranked factorially across tasks,
    mate assortatively by skill factor with random mating probability ``rmp``,
    and offspring are selectively evaluated only on the parental skill factor
    they vertically imitate.
    """

    algorithm_id = "mfea"
    algorithm_name = "Multifactorial Evolutionary Algorithm"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/TEVC.2015.2458037",
        "title": "Multifactorial Evolution: Toward Evolutionary Multitasking",
        "authors": "Abhishek Gupta, Yew-Soon Ong, Liang Feng",
        "year": 2016,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_discrete=True,
        supports_integer=True,
        supports_mixed=True,
    )
    _DEFAULTS = dict(
        population_size=100,
        rmp=0.3,
        sbx_eta=15.0,
        mutation_sigma=0.02,
        mutate_after_crossover=True,
        primary_task=0,
    )
    _OPERATOR_LABELS = (
        "mfea.unified_initialization",
        "mfea.factorial_evaluation",
        "mfea.factorial_rank_update",
        "mfea.skill_factor_assignment",
        "mfea.assortative_mating",
        "mfea.intratask_sbx_crossover",
        "mfea.intertask_sbx_transfer",
        "mfea.parent_centric_gaussian_mutation",
        "mfea.vertical_cultural_transmission",
        "mfea.scalar_fitness_selection",
        "mfea.elitist_replacement",
        "mfea.boundary_repair",
        "mfea.candidate_injection",
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        self._params = {**self._DEFAULTS, **config.params}
        self._warned: set[str] = set()
        self._tasks = self._build_tasks(problem, self._params.get("tasks"))
        self._K = len(self._tasks)
        self._D = max(t.dimension for t in self._tasks)
        self._n = max(4, int(self._params.get("population_size", 100)))
        self._rmp = float(np.clip(float(self._params.get("rmp", 0.3)), 0.0, 1.0))
        self._sbx_eta = max(0.0, float(self._params.get("sbx_eta", 15.0)))
        self._mutation_sigma = max(0.0, float(self._params.get("mutation_sigma", 0.02)))
        self._mutate_after_crossover = bool(self._params.get("mutate_after_crossover", True))
        self._primary_task = int(np.clip(int(self._params.get("primary_task", 0)), 0, self._K - 1))
        self._last_operator_counts = {label: 0 for label in self._OPERATOR_LABELS}
        self._last_operator_contributions = {label: 0.0 for label in self._OPERATOR_LABELS}
        if config.seed is not None:
            np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # task handling and random-key decoding
    # ------------------------------------------------------------------

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            warnings.warn(message, stacklevel=3)
            self._warned.add(key)

    def _build_tasks(self, default_problem: ProblemSpec, tasks_param: Any) -> list[_MFEATask]:
        raw_tasks = tasks_param if tasks_param is not None else [default_problem]
        if not isinstance(raw_tasks, (list, tuple)):
            raw_tasks = [raw_tasks]
        tasks: list[_MFEATask] = []
        for idx, raw in enumerate(raw_tasks):
            if isinstance(raw, ProblemSpec):
                tasks.append(
                    _MFEATask(
                        target_function=raw.target_function,
                        min_values=np.asarray(raw.min_values, dtype=float),
                        max_values=np.asarray(raw.max_values, dtype=float),
                        objective=str(raw.objective).lower(),
                        name=str((raw.metadata or {}).get("name", f"task_{idx}")),
                        problem_spec=raw,
                    )
                )
            elif isinstance(raw, dict):
                fn = raw.get("target_function") or raw.get("function") or raw.get("objective_function")
                if not callable(fn):
                    raise ValueError("Each MFEA task dictionary must provide a callable target_function/function.")
                lo = np.asarray(raw.get("min_values", raw.get("lower_bounds")), dtype=float)
                hi = np.asarray(raw.get("max_values", raw.get("upper_bounds")), dtype=float)
                if lo.size == 0 or hi.size == 0 or lo.size != hi.size:
                    raise ValueError("Each MFEA task dictionary must provide equal-length min_values and max_values.")
                ps = ProblemSpec(
                    target_function=fn,
                    min_values=lo.tolist(),
                    max_values=hi.tolist(),
                    objective=str(raw.get("objective", "min")).lower(),
                    constraints=raw.get("constraints"),
                    constraint_handler=raw.get("constraint_handler"),
                    variable_types=raw.get("variable_types"),
                    metadata=dict(raw.get("metadata", {}) or {}),
                )
                tasks.append(
                    _MFEATask(
                        target_function=fn,
                        min_values=lo,
                        max_values=hi,
                        objective=ps.objective,
                        name=str(raw.get("name", f"task_{idx}")),
                        problem_spec=ps,
                    )
                )
            else:
                raise TypeError("MFEA tasks must be ProblemSpec objects or task dictionaries.")
        if len(tasks) == 1:
            self._warn_once(
                "single_task",
                f"[{self.algorithm_id}] Running the K=1 degeneracy of MFEA; cross-task transfer is inactive. "
                "Pass config.params['tasks'] for true multifactorial optimization.",
            )
        return tasks

    def _decode(self, chromosome: np.ndarray, task_idx: int) -> np.ndarray:
        task = self._tasks[int(task_idx)]
        u = np.asarray(chromosome[: task.dimension], dtype=float)
        pos = task.min_values + u * (task.max_values - task.min_values)
        if task.problem_spec is not None:
            pos = task.problem_spec.apply_variable_types(pos)
        return np.clip(pos, task.min_values, task.max_values)

    def _encode_primary(self, position: list[float] | np.ndarray) -> np.ndarray:
        task = self._tasks[self._primary_task]
        pos = np.asarray(position, dtype=float)[: task.dimension]
        span = np.where(task.max_values - task.min_values == 0.0, 1.0, task.max_values - task.min_values)
        chrom = np.random.rand(self._D)
        chrom[: task.dimension] = np.clip((pos - task.min_values) / span, 0.0, 1.0)
        return chrom

    def _evaluate_task(self, chromosome: np.ndarray, task_idx: int) -> float:
        task = self._tasks[int(task_idx)]
        pos = self._decode(chromosome, int(task_idx))
        if task.problem_spec is not None:
            return float(task.problem_spec.evaluate(pos))
        return float(task.target_function(pos.tolist()))

    def _worst_value(self, task_idx: int) -> float:
        return float("inf") if self._tasks[int(task_idx)].objective == "min" else float("-inf")

    def _population_matrix_for_primary(self, chromosomes: np.ndarray, costs: np.ndarray) -> np.ndarray:
        positions = np.vstack([self._decode(ch, self._primary_task) for ch in chromosomes])
        fit = costs[:, self._primary_task].copy()
        missing = ~np.isfinite(fit)
        if np.any(missing):
            # Compatibility payload for generic framework methods. Missing primary
            # task values are rare for K=1, but in real multitask runs we use the
            # scalar skill-fitness surrogate to keep a valid population matrix.
            fit[missing] = 1.0 / np.maximum(1.0e-12, self._scalar_fitness_cache[missing])
        return np.hstack((positions, fit[:, None]))

    # ------------------------------------------------------------------
    # ranking, selection, genetic operators
    # ------------------------------------------------------------------

    def _rank_skill_scalar(self, costs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k = costs.shape
        ranks = np.empty((n, k), dtype=float)
        for task_idx, task in enumerate(self._tasks):
            noise = np.random.random(n) * 1.0e-12
            values = costs[:, task_idx].copy()
            if task.objective == "min":
                order = np.lexsort((noise, values))
            else:
                order = np.lexsort((noise, -values))
            task_ranks = np.empty(n, dtype=float)
            task_ranks[order] = np.arange(1, n + 1, dtype=float)
            ranks[:, task_idx] = task_ranks
        skill = np.argmin(ranks, axis=1).astype(int)
        best_ranks = ranks[np.arange(n), skill]
        scalar = 1.0 / np.maximum(best_ranks, 1.0)
        return ranks, skill, scalar

    def _sbx(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.random.random(self._D)
        eta = self._sbx_eta
        beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
        c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
        c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
        return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)

    def _mutate(self, parent: np.ndarray) -> np.ndarray:
        child = np.asarray(parent, dtype=float).copy()
        pm = 1.0 / max(1, self._D)
        mask = np.random.random(self._D) < pm
        if not np.any(mask):
            mask[np.random.randint(self._D)] = True
        child[mask] += np.random.normal(0.0, self._mutation_sigma, int(np.count_nonzero(mask)))
        return np.clip(child, 0.0, 1.0)

    def _new_operator_trackers(self) -> tuple[dict[str, int], dict[str, float]]:
        return ({label: 0 for label in self._OPERATOR_LABELS}, {label: 0.0 for label in self._OPERATOR_LABELS})

    def _update_global_bests(
        self,
        best_chromosomes: np.ndarray,
        best_fitness: np.ndarray,
        chromosomes: np.ndarray,
        costs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        best_chromosomes = np.asarray(best_chromosomes, dtype=float).copy()
        best_fitness = np.asarray(best_fitness, dtype=float).copy()
        for task_idx, task in enumerate(self._tasks):
            finite = np.isfinite(costs[:, task_idx])
            if not np.any(finite):
                continue
            idxs = np.where(finite)[0]
            vals = costs[idxs, task_idx]
            local_pos = int(np.argmin(vals) if task.objective == "min" else np.argmax(vals))
            local = int(idxs[local_pos])
            candidate_value = float(vals[local_pos])
            improved = (
                candidate_value < best_fitness[task_idx]
                if task.objective == "min"
                else candidate_value > best_fitness[task_idx]
            )
            if not np.isfinite(best_fitness[task_idx]) or improved:
                best_fitness[task_idx] = candidate_value
                best_chromosomes[task_idx] = chromosomes[local].copy()
        return best_chromosomes, best_fitness

    def _task_best_positions(self, best_chromosomes: np.ndarray) -> list[list[float]]:
        return [self._decode(best_chromosomes[k], k).tolist() for k in range(self._K)]

    # ------------------------------------------------------------------
    # BaseEngine API
    # ------------------------------------------------------------------

    def initialize(self) -> EngineState:
        counts, contrib = self._new_operator_trackers()
        chromosomes = np.random.random((self._n, self._D))
        costs = np.empty((self._n, self._K), dtype=float)
        for k in range(self._K):
            counts["mfea.factorial_evaluation"] += self._n
            for i in range(self._n):
                costs[i, k] = self._evaluate_task(chromosomes[i], k)
        ranks, skill, scalar = self._rank_skill_scalar(costs)
        self._scalar_fitness_cache = scalar.copy()
        counts["mfea.unified_initialization"] = self._n
        counts["mfea.factorial_rank_update"] = self._K
        counts["mfea.skill_factor_assignment"] = self._n
        contrib["mfea.unified_initialization"] = float(self._n)
        best_chromosomes = np.zeros((self._K, self._D), dtype=float)
        best_fitness = np.array([self._worst_value(k) for k in range(self._K)], dtype=float)
        best_chromosomes, best_fitness = self._update_global_bests(best_chromosomes, best_fitness, chromosomes, costs)
        primary_fit = float(best_fitness[self._primary_task])
        primary_pos = self._decode(best_chromosomes[self._primary_task], self._primary_task).tolist()
        population = self._population_matrix_for_primary(chromosomes, costs)
        self._last_operator_counts = counts
        self._last_operator_contributions = contrib
        return EngineState(
            step=0,
            evaluations=int(self._n * self._K),
            best_position=primary_pos,
            best_fitness=primary_fit,
            initialized=True,
            payload=dict(
                chromosomes=chromosomes,
                factorial_costs=costs,
                factorial_ranks=ranks,
                skill_factors=skill,
                scalar_fitness=scalar,
                population=population,
                best_chromosomes=best_chromosomes,
                best_fitness_by_task=best_fitness,
                best_positions_by_task=self._task_best_positions(best_chromosomes),
                rmp=self._rmp,
            ),
        )

    def step(self, state: EngineState) -> EngineState:
        counts, contrib = self._new_operator_trackers()
        chromosomes = np.asarray(state.payload["chromosomes"], dtype=float)
        costs = np.asarray(state.payload["factorial_costs"], dtype=float)
        skill = np.asarray(state.payload["skill_factors"], dtype=int)
        n = chromosomes.shape[0]
        offspring: list[np.ndarray] = []
        offspring_skill: list[int] = []

        while len(offspring) < n:
            a, b = np.random.choice(n, size=2, replace=False)
            pa, pb = chromosomes[a], chromosomes[b]
            sa, sb = int(skill[a]), int(skill[b])
            same_skill = sa == sb
            do_crossover = same_skill or (np.random.random() < self._rmp)
            counts["mfea.assortative_mating"] += 1
            if do_crossover:
                ca, cb = self._sbx(pa, pb)
                if self._mutate_after_crossover:
                    ca = self._mutate(ca)
                    cb = self._mutate(cb)
                    counts["mfea.parent_centric_gaussian_mutation"] += 2
                counts["mfea.boundary_repair"] += 2
                if same_skill:
                    counts["mfea.intratask_sbx_crossover"] += 2
                else:
                    counts["mfea.intertask_sbx_transfer"] += 2
                # Vertical cultural transmission: each child imitates one parent.
                ta = sa if np.random.random() < 0.5 else sb
                tb = sa if np.random.random() < 0.5 else sb
                counts["mfea.vertical_cultural_transmission"] += 2
                offspring.extend([ca, cb])
                offspring_skill.extend([ta, tb])
            else:
                ca = self._mutate(pa)
                cb = self._mutate(pb)
                counts["mfea.parent_centric_gaussian_mutation"] += 2
                counts["mfea.boundary_repair"] += 2
                counts["mfea.vertical_cultural_transmission"] += 2
                offspring.extend([ca, cb])
                offspring_skill.extend([sa, sb])

        offspring_arr = np.asarray(offspring[:n], dtype=float)
        offspring_skill_arr = np.asarray(offspring_skill[:n], dtype=int)
        offspring_costs = np.empty((n, self._K), dtype=float)
        for k in range(self._K):
            offspring_costs[:, k] = self._worst_value(k)
        for i in range(n):
            k = int(offspring_skill_arr[i])
            offspring_costs[i, k] = self._evaluate_task(offspring_arr[i], k)
            counts["mfea.factorial_evaluation"] += 1

        combined_chrom = np.vstack((chromosomes, offspring_arr))
        combined_costs = np.vstack((costs, offspring_costs))
        ranks, combined_skill, scalar = self._rank_skill_scalar(combined_costs)
        order = np.argsort(-scalar)
        keep = order[:n]
        selected_chrom = combined_chrom[keep].copy()
        selected_costs = combined_costs[keep].copy()
        selected_ranks = ranks[keep].copy()
        selected_skill = combined_skill[keep].copy()
        selected_scalar = scalar[keep].copy()
        self._scalar_fitness_cache = selected_scalar.copy()

        counts["mfea.factorial_rank_update"] = self._K
        counts["mfea.skill_factor_assignment"] = combined_chrom.shape[0]
        counts["mfea.scalar_fitness_selection"] = combined_chrom.shape[0]
        counts["mfea.elitist_replacement"] = n
        contrib["mfea.scalar_fitness_selection"] = float(np.mean(selected_scalar))
        contrib["mfea.elitist_replacement"] = float(np.max(selected_scalar))
        if counts["mfea.intratask_sbx_crossover"]:
            contrib["mfea.intratask_sbx_crossover"] = float(counts["mfea.intratask_sbx_crossover"])
        if counts["mfea.intertask_sbx_transfer"]:
            contrib["mfea.intertask_sbx_transfer"] = float(counts["mfea.intertask_sbx_transfer"])
        if counts["mfea.parent_centric_gaussian_mutation"]:
            contrib["mfea.parent_centric_gaussian_mutation"] = float(counts["mfea.parent_centric_gaussian_mutation"])

        best_chrom = np.asarray(state.payload["best_chromosomes"], dtype=float)
        best_fit = np.asarray(state.payload["best_fitness_by_task"], dtype=float)
        best_chrom, best_fit = self._update_global_bests(best_chrom, best_fit, offspring_arr, offspring_costs)
        population = self._population_matrix_for_primary(selected_chrom, selected_costs)
        state.payload = dict(
            chromosomes=selected_chrom,
            factorial_costs=selected_costs,
            factorial_ranks=selected_ranks,
            skill_factors=selected_skill,
            scalar_fitness=selected_scalar,
            population=population,
            best_chromosomes=best_chrom,
            best_fitness_by_task=best_fit,
            best_positions_by_task=self._task_best_positions(best_chrom),
            rmp=self._rmp,
        )
        state.evaluations += n
        state.step += 1
        state.best_fitness = float(best_fit[self._primary_task])
        state.best_position = self._decode(best_chrom[self._primary_task], self._primary_task).tolist()
        self._last_operator_counts = {k: int(v) for k, v in counts.items()}
        self._last_operator_contributions = {k: float(v) for k, v in contrib.items()}
        return state

    def observe(self, state: EngineState) -> dict:
        scalar = np.asarray(state.payload["scalar_fitness"], dtype=float)
        skills = np.asarray(state.payload["skill_factors"], dtype=int)
        skill_counts = {str(k): int(np.count_nonzero(skills == k)) for k in range(self._K)}
        return dict(
            step=state.step,
            evaluations=state.evaluations,
            best_fitness=state.best_fitness,
            task_count=self._K,
            unified_dimension=self._D,
            rmp=self._rmp,
            mean_scalar_fitness=float(np.mean(scalar)),
            skill_factor_counts=skill_counts,
            best_fitness_by_task=[float(x) for x in state.payload["best_fitness_by_task"]],
            operator_contributions=dict(self._last_operator_contributions),
            operator_counts=dict(self._last_operator_counts),
            evomapx_delta_f="scalar_fitness_surrogate",
            evomapx_fidelity="native",
        )

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
            metadata={
                "task_index": self._primary_task,
                "task_name": self._tasks[self._primary_task].name,
                "rmp": self._rmp,
            },
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=state.step,
            evaluations=state.evaluations,
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata=dict(
                algorithm_name=self.algorithm_name,
                elapsed_time=state.elapsed_time,
                task_count=self._K,
                primary_task=self._primary_task,
                rmp=self._rmp,
                best_fitness_by_task=[float(x) for x in state.payload.get("best_fitness_by_task", [])],
                best_positions_by_task=state.payload.get("best_positions_by_task", []),
                operator_contributions=dict(self._last_operator_contributions),
                operator_counts=dict(self._last_operator_counts),
            ),
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        chrom = np.asarray(state.payload["chromosomes"], dtype=float)
        costs = np.asarray(state.payload["factorial_costs"], dtype=float)
        skills = np.asarray(state.payload["skill_factors"], dtype=int)
        scalar = np.asarray(state.payload["scalar_fitness"], dtype=float)
        out: list[CandidateRecord] = []
        for i in range(chrom.shape[0]):
            task_idx = int(skills[i])
            fit = costs[i, task_idx]
            if not np.isfinite(fit):
                fit = 1.0 / max(1.0e-12, float(scalar[i]))
            out.append(
                CandidateRecord(
                    position=self._decode(chrom[i], task_idx).tolist(),
                    fitness=float(fit),
                    source_algorithm=self.algorithm_id,
                    source_step=state.step,
                    role="current",
                    metadata={"task_index": task_idx, "scalar_fitness": float(scalar[i])},
                )
            )
        return out

    def inject_candidates(self, state: EngineState, candidates: list[CandidateRecord], policy: str = "native") -> EngineState:
        if not candidates:
            return state
        chromosomes = np.asarray(state.payload["chromosomes"], dtype=float).copy()
        costs = np.asarray(state.payload["factorial_costs"], dtype=float).copy()
        scalar = np.asarray(state.payload["scalar_fitness"], dtype=float)
        worst = np.argsort(scalar)[: len(candidates)]
        for idx, cand in zip(worst, candidates):
            chrom = self._encode_primary(cand.position)
            row = np.array([self._worst_value(k) for k in range(self._K)], dtype=float)
            row[self._primary_task] = self._evaluate_task(chrom, self._primary_task)
            chromosomes[int(idx)] = chrom
            costs[int(idx)] = row
            state.evaluations += 1
        ranks, skill, scalar = self._rank_skill_scalar(costs)
        self._scalar_fitness_cache = scalar.copy()
        best_chrom = np.asarray(state.payload["best_chromosomes"], dtype=float)
        best_fit = np.asarray(state.payload["best_fitness_by_task"], dtype=float)
        best_chrom, best_fit = self._update_global_bests(best_chrom, best_fit, chromosomes, costs)
        state.payload.update(
            chromosomes=chromosomes,
            factorial_costs=costs,
            factorial_ranks=ranks,
            skill_factors=skill,
            scalar_fitness=scalar,
            population=self._population_matrix_for_primary(chromosomes, costs),
            best_chromosomes=best_chrom,
            best_fitness_by_task=best_fit,
            best_positions_by_task=self._task_best_positions(best_chrom),
        )
        state.best_fitness = float(best_fit[self._primary_task])
        state.best_position = self._decode(best_chrom[self._primary_task], self._primary_task).tolist()
        self._last_operator_counts["mfea.candidate_injection"] = len(candidates)
        self._last_operator_contributions["mfea.candidate_injection"] = float(len(candidates))
        return state
