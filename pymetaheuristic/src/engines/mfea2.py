"""
pyMetaheuristic src — MFEA-II Engine
====================================
Native macro-step: unified multitask population → online RMP learning →
assortative/intertask SBX transfer → polynomial mutation → elitist scalar-fitness replacement.

The engine supports the package's ordinary single-ProblemSpec interface as the
K=1 degeneracy of MFEA-II.  For actual multitasking, pass a list of task
ProblemSpec objects or task dictionaries through config.params["tasks"].
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
    variable_types: Any = None

    @property
    def dimension(self) -> int:
        return int(self.min_values.size)


class MFEA2Engine(BaseEngine):
    """Multifactorial Evolutionary Algorithm II with online RMP matrix learning."""

    algorithm_id = "mfea2"
    algorithm_name = "Multifactorial Evolutionary Algorithm II"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/TEVC.2019.2906927",
        "title": "Multifactorial Evolutionary Algorithm With Online Transfer Parameter Estimation: MFEA-II",
        "authors": "Kavitesh Kumar Bali, Yew-Soon Ong, Abhishek Gupta, Puay Siew Tan",
        "year": 2020,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=50,       # per-task N used in the synthetic continuous study
        sbx_probability=1.0,
        sbx_eta=15.0,
        mutation_eta=15.0,
        mutation_probability=None, # paper: pm = 1 / d
        model_std_floor=0.05,      # simple marginal normal model, avoiding overfit collapse
        rmp_grid_size=11,
        rmp_learning_passes=2,
        primary_task=0,
    )

    _OPERATOR_LABELS = (
        "mfea2.unified_initialization",
        "mfea2.skill_factor_assignment",
        "mfea2.scalar_fitness_selection",
        "mfea2.univariate_model_building",
        "mfea2.online_rmp_matrix_learning",
        "mfea2.intratask_sbx_crossover",
        "mfea2.intertask_sbx_transfer",
        "mfea2.parent_centric_polynomial_mutation",
        "mfea2.elitist_scalar_replacement",
        "mfea2.boundary_repair",
        "mfea2.candidate_injection",
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        params = {**self._DEFAULTS, **(config.params or {})}
        self._params = params
        self._tasks = self._normalize_tasks(params.get("tasks"))
        self._k_tasks = len(self._tasks)
        self._unified_dim = max(t.dimension for t in self._tasks)
        self._n_per_task = max(2, int(params["population_size"]))
        self._total_n = self._n_per_task * self._k_tasks
        self._pc = float(params.get("sbx_probability", 1.0))
        self._eta_c = float(params.get("sbx_eta", 15.0))
        self._eta_m = float(params.get("mutation_eta", 15.0))
        pm = params.get("mutation_probability", None)
        self._pm = None if pm is None else float(pm)
        self._model_std_floor = max(1.0e-12, float(params.get("model_std_floor", 0.05)))
        self._rmp_grid_size = max(3, int(params.get("rmp_grid_size", 11)))
        self._rmp_learning_passes = max(1, int(params.get("rmp_learning_passes", 2)))
        self._primary_task = int(np.clip(int(params.get("primary_task", 0)), 0, self._k_tasks - 1))
        self._last_operator_counts = self._blank_counts()
        self._last_operator_contributions = self._blank_contribs()
        if self._k_tasks == 1:
            warnings.warn(
                "[mfea2] MFEA-II is a multitask optimizer. No config.params['tasks'] list was supplied, "
                "so the engine is running the K=1 single-task degeneracy with RMP=[[1]].",
                RuntimeWarning,
                stacklevel=2,
            )
        if config.seed is not None:
            np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # Task handling and unified-space decoding
    # ------------------------------------------------------------------
    def _normalize_tasks(self, tasks: Any) -> list[_MFEATask]:
        if not tasks:
            return [_MFEATask(
                target_function=self.problem.target_function,
                min_values=np.asarray(self.problem.min_values, dtype=float),
                max_values=np.asarray(self.problem.max_values, dtype=float),
                objective=str(self.problem.objective or "min").lower(),
                name=str((self.problem.metadata or {}).get("name", "task_0")),
                variable_types=self.problem.variable_types,
            )]
        out: list[_MFEATask] = []
        for idx, task in enumerate(tasks):
            if isinstance(task, ProblemSpec):
                out.append(_MFEATask(
                    target_function=task.target_function,
                    min_values=np.asarray(task.min_values, dtype=float),
                    max_values=np.asarray(task.max_values, dtype=float),
                    objective=str(task.objective or "min").lower(),
                    name=str((task.metadata or {}).get("name", f"task_{idx}")),
                    variable_types=task.variable_types,
                ))
            elif isinstance(task, dict):
                if "target_function" not in task:
                    raise ValueError("Each MFEA-II task dictionary needs a 'target_function'.")
                out.append(_MFEATask(
                    target_function=task["target_function"],
                    min_values=np.asarray(task.get("min_values", self.problem.min_values), dtype=float),
                    max_values=np.asarray(task.get("max_values", self.problem.max_values), dtype=float),
                    objective=str(task.get("objective", "min")).lower(),
                    name=str(task.get("name", f"task_{idx}")),
                    variable_types=task.get("variable_types", None),
                ))
            else:
                raise TypeError("MFEA-II tasks must be ProblemSpec objects or dictionaries.")
        if not out:
            raise ValueError("At least one task is required for MFEA-II.")
        return out

    def _decode_for_task(self, x_unified: np.ndarray, task_id: int) -> np.ndarray:
        task = self._tasks[int(task_id)]
        z = np.asarray(x_unified, dtype=float)[: task.dimension]
        decoded = task.min_values + np.clip(z, 0.0, 1.0) * (task.max_values - task.min_values)
        if task.variable_types is not None:
            # Reuse ProblemSpec's projection semantics for the primary package task only; for
            # auxiliary task dictionaries, apply the common integer/binary cases locally.
            vt = task.variable_types
            if isinstance(vt, str):
                vt = [vt] * task.dimension
            if len(vt) == 1 and task.dimension != 1:
                vt = list(vt) * task.dimension
            for j, desc in enumerate(vt):
                kind = str(desc.get("type", "float") if isinstance(desc, dict) else desc).lower()
                if kind in {"int", "integer", "discrete", "ordinal"}:
                    decoded[j] = np.round(decoded[j])
                elif kind in {"binary", "bool", "boolean"}:
                    decoded[j] = 1.0 if decoded[j] >= 0.5 * (task.min_values[j] + task.max_values[j]) else 0.0
            decoded = np.clip(decoded, task.min_values, task.max_values)
        return decoded

    def _encode_primary(self, position: list[float] | np.ndarray) -> np.ndarray:
        task = self._tasks[self._primary_task]
        pos = np.asarray(position, dtype=float)[: task.dimension]
        denom = np.maximum(task.max_values - task.min_values, 1.0e-30)
        z = (np.clip(pos, task.min_values, task.max_values) - task.min_values) / denom
        x = np.zeros(self._unified_dim, dtype=float) + 0.5
        x[: task.dimension] = z
        return np.clip(x, 0.0, 1.0)

    def _evaluate_task(self, x_unified: np.ndarray, task_id: int) -> float:
        task = self._tasks[int(task_id)]
        decoded = self._decode_for_task(x_unified, int(task_id))
        return float(task.target_function(decoded.tolist()))

    def _task_order(self, fitness: np.ndarray, skill: np.ndarray, task_id: int) -> np.ndarray:
        idx = np.where(skill == int(task_id))[0]
        if idx.size == 0:
            return idx
        vals = fitness[idx]
        reverse = self._tasks[int(task_id)].objective == "max"
        order = np.argsort(vals)
        if reverse:
            order = order[::-1]
        return idx[order]

    def _better_task(self, a: float, b: float, task_id: int) -> bool:
        return a < b if self._tasks[int(task_id)].objective == "min" else a > b

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def _sbx_pair(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self._pc:
            return a.copy(), b.copy()
        u = np.random.rand(self._unified_dim)
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (self._eta_c + 1.0)),
            (1.0 / np.maximum(2.0 * (1.0 - u), 1.0e-30)) ** (1.0 / (self._eta_c + 1.0)),
        )
        c1 = 0.5 * ((1.0 + beta) * a + (1.0 - beta) * b)
        c2 = 0.5 * ((1.0 - beta) * a + (1.0 + beta) * b)
        return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)

    def _polynomial_mutation(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        pm = (1.0 / max(1, self._unified_dim)) if self._pm is None else self._pm
        mutate = np.random.rand(self._unified_dim) < pm
        if not np.any(mutate):
            return np.clip(y, 0.0, 1.0)
        u = np.random.rand(self._unified_dim)
        eta = self._eta_m
        for j in np.where(mutate)[0]:
            yl, yu = 0.0, 1.0
            delta1 = (y[j] - yl) / (yu - yl)
            delta2 = (yu - y[j]) / (yu - yl)
            mut_pow = 1.0 / (eta + 1.0)
            if u[j] <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u[j] + (1.0 - 2.0 * u[j]) * (xy ** (eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u[j]) + 2.0 * (u[j] - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val ** mut_pow
            y[j] += deltaq * (yu - yl)
        return np.clip(y, 0.0, 1.0)

    def _build_models(self, pop: np.ndarray, skill: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.full((self._k_tasks, self._unified_dim), 0.5, dtype=float)
        stds = np.full((self._k_tasks, self._unified_dim), self._model_std_floor, dtype=float)
        for k in range(self._k_tasks):
            sub = pop[skill == k]
            if sub.size:
                means[k] = np.mean(sub, axis=0)
                if sub.shape[0] >= 2:
                    stds[k] = np.maximum(np.std(sub, axis=0, ddof=1), self._model_std_floor)
        return means, stds

    def _log_density_all_models(self, samples: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        # Returns shape (n_samples, K), variable-wise marginal normal product in log-domain.
        samples = np.asarray(samples, dtype=float)
        out = np.empty((samples.shape[0], self._k_tasks), dtype=float)
        log2pi = np.log(2.0 * np.pi)
        for k in range(self._k_tasks):
            z = (samples - means[k]) / stds[k]
            out[:, k] = -0.5 * np.sum(z * z + log2pi + 2.0 * np.log(stds[k]), axis=1)
        return out

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
        m = np.max(a, axis=axis, keepdims=True)
        return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis) + 1.0e-300)

    def _rmp_log_likelihood(self, rmp: np.ndarray, grouped_samples: list[np.ndarray], means: np.ndarray, stds: np.ndarray) -> float:
        if self._k_tasks == 1:
            return 0.0
        k_total = float(self._k_tasks)
        ll = 0.0
        for k, samples in enumerate(grouped_samples):
            if samples.size == 0:
                continue
            weights = np.zeros(self._k_tasks, dtype=float)
            for j in range(self._k_tasks):
                if j == k:
                    continue
                weights[j] = 0.5 * float(rmp[k, j]) / k_total
            weights[k] = 1.0 - 0.5 * float(np.sum(rmp[k]) - rmp[k, k]) / k_total
            weights = np.maximum(weights, 1.0e-300)
            logdens = self._log_density_all_models(samples, means, stds)
            ll += float(np.sum(self._logsumexp(logdens + np.log(weights)[None, :], axis=1)))
        return ll

    def _learn_rmp(self, pop: np.ndarray, skill: np.ndarray, previous: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means, stds = self._build_models(pop, skill)
        if self._k_tasks == 1:
            return np.ones((1, 1), dtype=float), means, stds
        if previous is None or np.asarray(previous).shape != (self._k_tasks, self._k_tasks):
            rmp = np.full((self._k_tasks, self._k_tasks), 0.5, dtype=float)
            np.fill_diagonal(rmp, 1.0)
        else:
            rmp = np.asarray(previous, dtype=float).copy()
            rmp = np.clip(0.5 * (rmp + rmp.T), 0.0, 1.0)
            np.fill_diagonal(rmp, 1.0)
        grouped = [pop[skill == k] for k in range(self._k_tasks)]
        grid = np.linspace(0.0, 1.0, self._rmp_grid_size)
        for _ in range(self._rmp_learning_passes):
            for k in range(self._k_tasks):
                for j in range(k + 1, self._k_tasks):
                    best_val = rmp[k, j]
                    best_ll = -float("inf")
                    for val in grid:
                        trial = rmp.copy()
                        trial[k, j] = trial[j, k] = float(val)
                        ll = self._rmp_log_likelihood(trial, grouped, means, stds)
                        if ll > best_ll:
                            best_ll, best_val = ll, float(val)
                    rmp[k, j] = rmp[j, k] = best_val
        return rmp, means, stds

    # ------------------------------------------------------------------
    # Fitness bookkeeping
    # ------------------------------------------------------------------
    def _scalar_fitness(self, fitness: np.ndarray, skill: np.ndarray) -> np.ndarray:
        scalar = np.zeros(fitness.shape[0], dtype=float)
        for k in range(self._k_tasks):
            order = self._task_order(fitness, skill, k)
            for rank, idx in enumerate(order, start=1):
                scalar[idx] = 1.0 / float(rank)
        return scalar

    def _task_bests_from_population(self, pop: np.ndarray, fitness: np.ndarray, skill: np.ndarray,
                                    old_pos: np.ndarray | None = None,
                                    old_fit: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        best_pos = np.zeros((self._k_tasks, self._unified_dim), dtype=float) + 0.5
        best_fit = np.empty(self._k_tasks, dtype=float)
        for k in range(self._k_tasks):
            if old_pos is not None and old_fit is not None:
                best_pos[k] = old_pos[k]
                best_fit[k] = old_fit[k]
            else:
                best_fit[k] = float("inf") if self._tasks[k].objective == "min" else float("-inf")
            ids = np.where(skill == k)[0]
            for idx in ids:
                if self._better_task(float(fitness[idx]), float(best_fit[k]), k):
                    best_fit[k] = float(fitness[idx])
                    best_pos[k] = pop[idx].copy()
        return best_pos, best_fit

    def _blank_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._OPERATOR_LABELS}

    def _blank_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._OPERATOR_LABELS}

    # ------------------------------------------------------------------
    # Engine protocol
    # ------------------------------------------------------------------
    def initialize(self) -> EngineState:
        pop = np.random.rand(self._total_n, self._unified_dim)
        skill = np.arange(self._total_n, dtype=int) % self._k_tasks
        fitness = np.array([self._evaluate_task(pop[i], int(skill[i])) for i in range(self._total_n)], dtype=float)
        best_pos, best_fit = self._task_bests_from_population(pop, fitness, skill)
        rmp = np.eye(self._k_tasks, dtype=float)
        if self._k_tasks > 1:
            rmp[:] = 0.5
            np.fill_diagonal(rmp, 1.0)
        matrix = np.hstack((pop, fitness[:, None]))
        self._last_operator_counts = self._blank_counts()
        self._last_operator_counts["mfea2.unified_initialization"] = self._total_n
        self._last_operator_counts["mfea2.skill_factor_assignment"] = self._total_n
        self._last_operator_contributions = self._blank_contribs()
        primary = self._primary_task
        return EngineState(
            step=0,
            evaluations=self._total_n,
            best_position=self._decode_for_task(best_pos[primary], primary).tolist(),
            best_fitness=float(best_fit[primary]),
            initialized=True,
            payload=dict(
                population=matrix,
                skill_factors=skill,
                rmp_matrix=rmp,
                task_best_unified=best_pos,
                task_best_fitness=best_fit,
                task_names=[t.name for t in self._tasks],
                models_mean=np.full((self._k_tasks, self._unified_dim), 0.5),
                models_std=np.full((self._k_tasks, self._unified_dim), self._model_std_floor),
            ),
        )

    def step(self, state: EngineState) -> EngineState:
        matrix = np.asarray(state.payload["population"], dtype=float)
        pop = matrix[:, : self._unified_dim]
        fitness = matrix[:, -1]
        skill = np.asarray(state.payload["skill_factors"], dtype=int)
        counts = self._blank_counts()
        contrib = self._blank_contribs()

        scalar = self._scalar_fitness(fitness, skill)
        parent_count = max(self._k_tasks, int(np.ceil(self._total_n / 2.0)))
        parents_idx = np.argsort(-scalar)[:parent_count]
        parents = pop[parents_idx]
        parent_skill = skill[parents_idx]
        counts["mfea2.scalar_fitness_selection"] = int(parent_count)

        rmp_prev = np.asarray(state.payload.get("rmp_matrix"), dtype=float)
        rmp, means, stds = self._learn_rmp(parents, parent_skill, rmp_prev)
        counts["mfea2.univariate_model_building"] = int(self._k_tasks)
        counts["mfea2.online_rmp_matrix_learning"] = int(max(0, self._k_tasks * (self._k_tasks - 1) // 2))

        offspring: list[np.ndarray] = []
        offspring_skill: list[int] = []
        task_offspring_count = np.zeros(self._k_tasks, dtype=int)
        attempts = 0
        max_attempts = max(1000, 20 * self._total_n)
        while np.any(task_offspring_count < self._n_per_task) and attempts < max_attempts:
            attempts += 1
            if parents.shape[0] >= 2:
                a_idx, b_idx = np.random.choice(parents.shape[0], 2, replace=False)
            else:
                a_idx = b_idx = 0
            x_i, x_j = parents[a_idx], parents[b_idx]
            t_i, t_j = int(parent_skill[a_idx]), int(parent_skill[b_idx])
            children: list[tuple[np.ndarray, int, str]] = []
            if t_i == t_j:
                c1, c2 = self._sbx_pair(x_i, x_j)
                children = [(c1, t_i, "mfea2.intratask_sbx_crossover"), (c2, t_i, "mfea2.intratask_sbx_crossover")]
            elif np.random.rand() <= rmp[t_i, t_j]:
                c1, c2 = self._sbx_pair(x_i, x_j)
                children = [
                    (c1, int(np.random.choice([t_i, t_j])), "mfea2.intertask_sbx_transfer"),
                    (c2, int(np.random.choice([t_i, t_j])), "mfea2.intertask_sbx_transfer"),
                ]
            else:
                # Algorithm 3 fallback: mate each parent inside its own skill factor.
                for x, t in ((x_i, t_i), (x_j, t_j)):
                    same = np.where(parent_skill == t)[0]
                    if same.size:
                        mate = parents[int(np.random.choice(same))]
                        c, _ = self._sbx_pair(x, mate)
                    else:
                        c = x.copy()
                    children.append((c, t, "mfea2.intratask_sbx_crossover"))

            for child, task_id, label in children:
                if task_offspring_count[task_id] >= self._n_per_task:
                    continue
                child = self._polynomial_mutation(child)
                offspring.append(child)
                offspring_skill.append(task_id)
                task_offspring_count[task_id] += 1
                counts[label] += 1
                counts["mfea2.parent_centric_polynomial_mutation"] += 1
                counts["mfea2.boundary_repair"] += 1

        # Fallback in the unlikely event that cap-aware child acceptance starved a task.
        for k in range(self._k_tasks):
            while task_offspring_count[k] < self._n_per_task:
                same = np.where(parent_skill == k)[0]
                seed = parents[int(np.random.choice(same))] if same.size else np.random.rand(self._unified_dim)
                child = self._polynomial_mutation(seed)
                offspring.append(child)
                offspring_skill.append(k)
                task_offspring_count[k] += 1
                counts["mfea2.parent_centric_polynomial_mutation"] += 1
                counts["mfea2.boundary_repair"] += 1

        off = np.asarray(offspring, dtype=float)
        off_skill = np.asarray(offspring_skill, dtype=int)
        off_fitness = np.array([self._evaluate_task(off[i], int(off_skill[i])) for i in range(off.shape[0])], dtype=float)

        old_best_pos = np.asarray(state.payload["task_best_unified"], dtype=float)
        old_best_fit = np.asarray(state.payload["task_best_fitness"], dtype=float)
        combined_pop = np.vstack((pop, off))
        combined_skill = np.concatenate((skill, off_skill))
        combined_fitness = np.concatenate((fitness, off_fitness))
        combined_scalar = self._scalar_fitness(combined_fitness, combined_skill)
        keep = np.argsort(-combined_scalar)[: self._total_n]
        new_pop = combined_pop[keep]
        new_skill = combined_skill[keep]
        new_fitness = combined_fitness[keep]
        counts["mfea2.elitist_scalar_replacement"] = int(self._total_n)

        best_pos, best_fit = self._task_bests_from_population(new_pop, new_fitness, new_skill, old_best_pos, old_best_fit)
        primary = self._primary_task
        previous_primary = float(old_best_fit[primary])
        new_primary = float(best_fit[primary])
        gain = 0.0
        if np.isfinite(previous_primary) and self._better_task(new_primary, previous_primary, primary):
            gain = abs(previous_primary - new_primary)
        if gain > 0.0:
            # Attribute primary-task improvement to the variation/transfer and replacement chain.
            active = [k for k in (
                "mfea2.online_rmp_matrix_learning",
                "mfea2.intratask_sbx_crossover",
                "mfea2.intertask_sbx_transfer",
                "mfea2.parent_centric_polynomial_mutation",
                "mfea2.elitist_scalar_replacement",
            ) if counts.get(k, 0) > 0]
            for label in active:
                contrib[label] = float(gain / len(active))

        matrix = np.hstack((new_pop, new_fitness[:, None]))
        self._last_operator_counts = {k: int(v) for k, v in counts.items()}
        self._last_operator_contributions = {k: float(v) for k, v in contrib.items()}
        state.payload = dict(
            population=matrix,
            skill_factors=new_skill,
            rmp_matrix=rmp,
            task_best_unified=best_pos,
            task_best_fitness=best_fit,
            task_names=[t.name for t in self._tasks],
            models_mean=means,
            models_std=stds,
        )
        state.evaluations += int(off.shape[0])
        state.step += 1
        state.best_position = self._decode_for_task(best_pos[primary], primary).tolist()
        state.best_fitness = float(best_fit[primary])
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        matrix = np.asarray(state.payload["population"], dtype=float)
        skill = np.asarray(state.payload["skill_factors"], dtype=int)
        task_best = np.asarray(state.payload["task_best_fitness"], dtype=float)
        rmp = np.asarray(state.payload["rmp_matrix"], dtype=float)
        return dict(
            step=int(state.step),
            evaluations=int(state.evaluations),
            best_fitness=float(state.best_fitness),
            mean_fitness=float(np.mean(matrix[:, -1])),
            n_tasks=int(self._k_tasks),
            unified_dimension=int(self._unified_dim),
            per_task_population_size=int(self._n_per_task),
            skill_factor_counts={str(k): int(np.sum(skill == k)) for k in range(self._k_tasks)},
            task_best_fitness={str(k): float(task_best[k]) for k in range(self._k_tasks)},
            rmp_matrix=rmp.tolist(),
            mean_offdiag_rmp=float(np.mean(rmp[~np.eye(self._k_tasks, dtype=bool)])) if self._k_tasks > 1 else 1.0,
            operator_counts=dict(self._last_operator_counts),
            operator_contributions=dict(self._last_operator_contributions),
            evomapx_delta_f="primary_task_best_improvement",
            evomapx_fidelity="native",
        )

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=int(state.step),
            role="best",
            metadata={"task": int(self._primary_task)},
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        task_best_positions = []
        best_unified = np.asarray(state.payload.get("task_best_unified"), dtype=float)
        for k in range(self._k_tasks):
            task_best_positions.append(self._decode_for_task(best_unified[k], k).tolist())
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata=dict(
                algorithm_name=self.algorithm_name,
                elapsed_time=state.elapsed_time,
                n_tasks=self._k_tasks,
                primary_task=self._primary_task,
                task_names=[t.name for t in self._tasks],
                task_best_fitness=np.asarray(state.payload.get("task_best_fitness"), dtype=float).tolist(),
                task_best_positions=task_best_positions,
                final_rmp_matrix=np.asarray(state.payload.get("rmp_matrix"), dtype=float).tolist(),
                per_task_population_size=self._n_per_task,
                unified_dimension=self._unified_dim,
            ),
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        matrix = np.asarray(state.payload["population"], dtype=float)
        skill = np.asarray(state.payload["skill_factors"], dtype=int)
        out: list[CandidateRecord] = []
        for i in range(matrix.shape[0]):
            k = int(skill[i])
            out.append(CandidateRecord(
                position=self._decode_for_task(matrix[i, : self._unified_dim], k).tolist(),
                fitness=float(matrix[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=int(state.step),
                role="current",
                metadata={"skill_factor": k, "unified_position": matrix[i, : self._unified_dim].tolist()},
            ))
        return out

    def inject_candidates(self, state: EngineState, candidates: list[CandidateRecord], policy: str = "native") -> EngineState:
        if not candidates:
            return state
        matrix = np.asarray(state.payload["population"], dtype=float).copy()
        skill = np.asarray(state.payload["skill_factors"], dtype=int).copy()
        scalar = self._scalar_fitness(matrix[:, -1], skill)
        worst = np.argsort(scalar)[: min(len(candidates), matrix.shape[0])]
        for slot, cand in zip(worst, candidates):
            task_id = int((cand.metadata or {}).get("task", self._primary_task))
            task_id = int(np.clip(task_id, 0, self._k_tasks - 1))
            x = self._encode_primary(cand.position) if task_id == self._primary_task else np.clip(np.asarray(cand.position, dtype=float), 0.0, 1.0)
            if x.size != self._unified_dim:
                tmp = np.zeros(self._unified_dim, dtype=float) + 0.5
                tmp[: min(x.size, self._unified_dim)] = x[: min(x.size, self._unified_dim)]
                x = tmp
            fit = self._evaluate_task(x, task_id)
            matrix[slot, : self._unified_dim] = x
            matrix[slot, -1] = fit
            skill[slot] = task_id
        best_pos, best_fit = self._task_bests_from_population(
            matrix[:, : self._unified_dim], matrix[:, -1], skill,
            np.asarray(state.payload["task_best_unified"], dtype=float),
            np.asarray(state.payload["task_best_fitness"], dtype=float),
        )
        state.payload["population"] = matrix
        state.payload["skill_factors"] = skill
        state.payload["task_best_unified"] = best_pos
        state.payload["task_best_fitness"] = best_fit
        state.best_position = self._decode_for_task(best_pos[self._primary_task], self._primary_task).tolist()
        state.best_fitness = float(best_fit[self._primary_task])
        self._last_operator_counts = self._blank_counts()
        self._last_operator_counts["mfea2.candidate_injection"] = int(len(worst))
        self._last_operator_contributions = self._blank_contribs()
        return state


__all__ = ["MFEA2Engine"]
