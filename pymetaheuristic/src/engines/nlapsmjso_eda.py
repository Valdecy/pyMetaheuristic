"""pyMetaheuristic src — NLAPSMjSO-EDA Engine"""
from __future__ import annotations

import numpy as np

from .protocol import BaseEngine, CandidateRecord, CapabilityProfile, EngineConfig, EngineState, OptimizationResult, ProblemSpec


class NLAPSMJSOEDAEngine(BaseEngine):
    """Nonlinear APSM-jSO with EDA elite sampling.

    Assumptions documented for faithfulness:
    * The paper builds on APSM-jSO / LSHADE-RSP and references inherited DE
      details. Where the PDF points to those equations but does not restate every
      implementation nuance, this engine follows the standard interpretation of
      the cited equations.
    * The algorithm is evaluation-budget driven. When ``max_evaluations`` is not
      supplied by the framework, a deterministic surrogate budget is inferred from
      ``max_steps`` and the initial population size so that the nonlinear
      shrinking schedule remains well defined.
    * Equation (64) can yield zero elite samples for tiny populations; this port
      uses a minimum of one EDA sample whenever an EDA phase is entered.
    """

    algorithm_id = "nlapsmjso_eda"
    algorithm_name = "NLAPSMjSO-EDA"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.3390/sym17020153",
        "title": "NLAPSMjSO-EDA: A Nonlinear Shrinking Population Strategy Algorithm for Elite Group Exploration with Symmetry Applications",
        "authors": "Yong Shen, Jiaxuan Liang, Hongwei Kang, Xingping Sun, Qingyi Chen",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=None,
        min_population=4,
        memory_size=6,
        p_min=0.085,
        p_max=0.17,
        archive_rate=1.3,
        tau=0.9,
        rank_greediness=3.0,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        params = {**self._DEFAULTS, **config.params}
        requested_pop = params.get("population_size")
        if requested_pop is None:
            requested_pop = int(np.ceil(150.0 * (problem.dimension ** (2.0 / 3.0))))
        self._np_init = max(8, int(requested_pop))
        self._np_min = max(4, int(params.get("min_population", 4)))
        self._H = max(2, int(params.get("memory_size", 6)))
        self._p_min = float(params.get("p_min", 0.085))
        self._p_max = float(params.get("p_max", 0.17))
        self._archive_rate = float(params.get("archive_rate", 1.3))
        self._tau = float(params.get("tau", 0.9))
        self._rank_greediness = float(params.get("rank_greediness", 3.0))
        if config.seed is not None:
            np.random.seed(config.seed)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._np_init < 4:
            raise ValueError("nlapsmjso_eda requires population_size >= 4.")
        if self._np_min < 4:
            raise ValueError("nlapsmjso_eda min_population must be at least 4.")
        if self._np_min > self._np_init:
            raise ValueError("nlapsmjso_eda min_population cannot exceed population_size.")
        if self._H < 2:
            raise ValueError("nlapsmjso_eda memory_size must be at least 2.")
        if not 0.0 < self._p_min <= self._p_max <= 1.0:
            raise ValueError("nlapsmjso_eda requires 0 < p_min <= p_max <= 1.")
        if self._archive_rate <= 0.0:
            raise ValueError("nlapsmjso_eda archive_rate must be positive.")
        if not 0.0 < self._tau <= 1.0:
            raise ValueError("nlapsmjso_eda tau must be in (0, 1].")
        if self._rank_greediness <= 0.0:
            raise ValueError("nlapsmjso_eda rank_greediness must be positive.")

    @property
    def _lo(self) -> np.ndarray:
        return np.asarray(self.problem.min_values, dtype=float)

    @property
    def _hi(self) -> np.ndarray:
        return np.asarray(self.problem.max_values, dtype=float)

    def _evaluate_population(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        return np.asarray([float(self.problem.evaluate(row.copy())) for row in positions], dtype=float)

    def _best_index(self, fit: np.ndarray) -> int:
        return int(np.argmin(fit) if self.problem.objective == "min" else np.argmax(fit))

    def _order(self, fit: np.ndarray) -> np.ndarray:
        idx = np.argsort(np.asarray(fit, dtype=float))
        return idx if self.problem.objective == "min" else idx[::-1]

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _better_mask(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a < b if self.problem.objective == "min" else a > b

    def _weighted_lehmer(self, values: list[float], improvements: list[float], fallback: float) -> float:
        if not values:
            return float(fallback)
        vals = np.asarray(values, dtype=float)
        imp = np.asarray(improvements, dtype=float)
        w = imp / (np.sum(imp) + 1.0e-30)
        denom = np.sum(w * vals) + 1.0e-30
        return float(np.sum(w * vals * vals) / denom)

    def _fes_max(self) -> int:
        if self.config.max_evaluations is not None:
            return max(1, int(self.config.max_evaluations))
        steps = max(1, int(self.config.max_steps or 50))
        approx_npnew = max(1, int(np.ceil(self._tau * self._p_max * self._np_init)))
        return max(1, steps * (self._np_init + approx_npnew))

    def _rank_probabilities(self, n: int) -> np.ndarray:
        ranks = self._rank_greediness * (n - np.arange(1, n + 1)) + 1.0
        probs = ranks / (np.sum(ranks) + 1.0e-30)
        return probs.astype(float)

    def _sample_ranked_index(self, probs: np.ndarray, exclude: set[int] | None = None) -> int:
        exclude = exclude or set()
        idx = np.arange(probs.size)
        mask = np.array([i not in exclude for i in idx], dtype=bool)
        active = idx[mask]
        if active.size == 0:
            return int(np.random.randint(probs.size))
        p = probs[mask]
        p = p / (np.sum(p) + 1.0e-30)
        return int(np.random.choice(active, p=p))

    def _sample_F(self, mean: float, fes: int, fesmax: int) -> float:
        F = -1.0
        while F <= 0.0:
            F = mean + 0.1 * np.random.standard_cauchy()
        if fes < 0.6 * fesmax:
            return float(min(F, 0.7))
        return float(min(F, 1.0))

    def _sample_CR(self, mean: float, fes: int, fesmax: int) -> float:
        CR = mean + 0.1 * np.random.normal()
        if CR < 0.0:
            return 0.0
        if fes < 0.25 * fesmax:
            return float(max(CR, 0.7))
        if fes < 0.5 * fesmax:
            return float(max(CR, 0.6))
        return float(min(CR, 1.0))

    def _fw(self, F: float, fes: int, fesmax: int) -> float:
        if fes < 0.2 * fesmax:
            return 0.7 * F
        if fes < 0.4 * fesmax:
            return 0.8 * F
        return 1.2 * F

    def _repair_mutant(self, mutant: np.ndarray, base: np.ndarray) -> np.ndarray:
        out = mutant.copy()
        lo, hi = self._lo, self._hi
        below = out < lo
        above = out > hi
        out[below] = 0.5 * (lo[below] + base[below])
        out[above] = 0.5 * (hi[above] + base[above])
        return out

    def initialize(self) -> EngineState:
        pos = np.random.uniform(self._lo, self._hi, (self._np_init, self.problem.dimension))
        fit = self._evaluate_population(pos)
        pop = np.hstack((pos, fit[:, None]))
        best_idx = self._best_index(fit)
        MCR = np.full(self._H, 0.8, dtype=float)
        MF = np.full(self._H, 0.3, dtype=float)
        MCR[-1] = 0.9
        MF[-1] = 0.9
        PR = np.full(self._H, 1.0 / self._H, dtype=float)
        return EngineState(
            step=0,
            evaluations=pop.shape[0],
            best_position=pop[best_idx, :-1].tolist(),
            best_fitness=float(pop[best_idx, -1]),
            initialized=True,
            payload={
                "population": pop,
                "archive": np.empty((0, self.problem.dimension), dtype=float),
                "MCR": MCR,
                "MF": MF,
                "PR": PR,
                "k": 0,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        pop = np.asarray(state.payload["population"], dtype=float)
        archive = np.asarray(state.payload.get("archive", np.empty((0, self.problem.dimension))), dtype=float).reshape(-1, self.problem.dimension)
        MCR = np.asarray(state.payload.get("MCR"), dtype=float).copy()
        MF = np.asarray(state.payload.get("MF"), dtype=float).copy()
        PR = np.asarray(state.payload.get("PR"), dtype=float).copy()
        k = int(state.payload.get("k", 0)) % self._H

        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        pos = pop[:, :-1].copy()
        fit = pop[:, -1].copy()
        n, dim = pos.shape
        fesmax = self._fes_max()
        fes = int(state.evaluations)
        p = self._p_min + (self._p_max - self._p_min) * min(1.0, fes / fesmax)
        pbest_count = max(2, int(np.ceil(p * n)))
        rank_probs = self._rank_probabilities(n)
        union = pos if archive.size == 0 else np.vstack((pos, archive))

        trials = np.empty_like(pos)
        trial_fit = np.empty(n, dtype=float)
        Fi_used: list[float] = []
        CR_used: list[float] = []
        mem_used: list[int] = []
        evals = 0

        for i in range(n):
            h = int(np.random.choice(self._H, p=PR / (np.sum(PR) + 1.0e-30)))
            Fi = self._sample_F(MF[h], fes + evals, fesmax)
            Fwi = self._fw(Fi, fes + evals, fesmax)
            CRi = self._sample_CR(MCR[h], fes + evals, fesmax)
            xpbest = pos[np.random.randint(pbest_count)]
            pr1_idx = self._sample_ranked_index(rank_probs, exclude={i})
            xpr1 = pos[pr1_idx]
            if union.shape[0] == 1:
                xr2 = union[0]
            else:
                xr2 = union[np.random.randint(union.shape[0])]
                tries = 0
                while tries < 10 and (np.allclose(xr2, pos[i]) or np.allclose(xr2, xpbest) or np.allclose(xr2, xpr1)):
                    xr2 = union[np.random.randint(union.shape[0])]
                    tries += 1
            mutant = pos[i] + Fwi * (xpbest - pos[i]) + Fi * (xpr1 - xr2)
            mutant = self._repair_mutant(mutant, pos[i])
            cross = np.random.rand(dim) < CRi
            cross[np.random.randint(dim)] = True
            ui = np.where(cross, mutant, pos[i])
            trials[i] = ui
            trial_fit[i] = float(self.problem.evaluate(ui))
            Fi_used.append(Fi)
            CR_used.append(CRi)
            mem_used.append(h)
            evals += 1

        improved = self._better_mask(trial_fit, fit)
        SCR_by_h = [[] for _ in range(self._H)]
        SF_by_h = [[] for _ in range(self._H)]
        IMP_by_h = [[] for _ in range(self._H)]
        Nh = np.zeros(self._H, dtype=int)
        SNh = np.zeros(self._H, dtype=int)
        for h in mem_used:
            Nh[h] += 1

        if np.any(improved):
            improved_idx = np.where(improved)[0]
            old_positions = pos[improved_idx].copy()
            archive = np.vstack((archive, old_positions)) if archive.size else old_positions.copy()
            for idx in improved_idx.tolist():
                h = mem_used[idx]
                SNh[h] += 1
                SCR_by_h[h].append(CR_used[idx])
                SF_by_h[h].append(Fi_used[idx])
                improvement = abs(float(fit[idx]) - float(trial_fit[idx]))
                IMP_by_h[h].append(improvement if improvement > 0.0 else 1.0e-12)
            pos[improved] = trials[improved]
            fit[improved] = trial_fit[improved]

        pop = np.hstack((pos, fit[:, None]))
        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        pos = pop[:, :-1].copy()
        fit = pop[:, -1].copy()

        npeda = n if n < 2 * dim else max(1, int(np.floor(0.5 * n)))
        pd = pos[:npeda].copy()
        mu = np.mean(pd, axis=0)
        centered = pd - mu
        cov = (centered.T @ centered) / max(1, pd.shape[0])
        cov = cov + 1.0e-9 * np.eye(dim)
        npnew = max(1, int(round(self._tau * p * n)))
        pe = np.random.multivariate_normal(mu, cov, size=npnew)
        out_of_box = (pe < self._lo) | (pe > self._hi)
        if np.any(out_of_box):
            random_fill = np.random.uniform(self._lo, self._hi, pe.shape)
            pe[out_of_box] = random_fill[out_of_box]
        pe_fit = self._evaluate_population(pe)
        evals += npnew
        pop = np.vstack((pop, np.hstack((pe, pe_fit[:, None]))))

        if np.sum(SNh) == 0:
            PR = np.full(self._H, 1.0 / self._H, dtype=float)
        else:
            sr = np.divide(SNh, np.maximum(Nh, 1), dtype=float)
            if np.sum(sr) <= 0.0:
                PR = np.full(self._H, 1.0 / self._H, dtype=float)
            else:
                PR = sr / np.sum(sr)

        for h in range(self._H - 1):
            flat_scr = SCR_by_h[h]
            flat_sf = SF_by_h[h]
            flat_imp = IMP_by_h[h]
            if flat_sf:
                MF[h] = 0.5 * (MF[h] + self._weighted_lehmer(flat_sf, flat_imp, MF[h]))
            if not flat_scr or max(flat_scr) <= 0.0:
                MCR[h] = 0.0
            else:
                MCR[h] = 0.5 * (MCR[h] + self._weighted_lehmer(flat_scr, flat_imp, MCR[h]))
        MF[-1] = 0.9
        MCR[-1] = 0.9
        k = (k + 1) % self._H

        total_fes = min(fesmax, state.evaluations + evals)
        ratio = min(1.0, total_fes / fesmax)
        np_target = int(round((ratio ** (1.0 - ratio)) * (self._np_min - self._np_init) + self._np_init))
        np_target = max(self._np_min, min(np_target, pop.shape[0]))
        pop = pop[self._order(pop[:, -1])][:np_target].copy()
        max_archive = max(1, int(round(self._archive_rate * np_target)))
        if archive.shape[0] > max_archive:
            archive = archive[-max_archive:]

        best_idx = self._best_index(pop[:, -1])
        best_fit = float(pop[best_idx, -1])
        best_pos = pop[best_idx, :-1].tolist()

        state.payload = {
            "population": pop,
            "archive": archive,
            "MCR": MCR,
            "MF": MF,
            "PR": PR,
            "k": k,
        }
        state.step += 1
        state.evaluations += int(evals)
        if state.best_fitness is None or self._is_better(best_fit, state.best_fitness):
            state.best_fitness = best_fit
            state.best_position = best_pos
        return state

    def observe(self, state: EngineState) -> dict:
        pop = np.asarray(state.payload["population"], dtype=float)
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": float(np.mean(pop[:, -1])),
            "std_fitness": float(np.std(pop[:, -1])),
            "population_size": int(pop.shape[0]),
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=state.step,
            role="best",
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
            metadata={
                "algorithm_name": self.algorithm_name,
                "population_size": int(np.asarray(state.payload["population"]).shape[0]),
                "elapsed_time": state.elapsed_time,
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"], dtype=float)
        return [
            CandidateRecord(
                position=pop[i, :-1].tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=state.step,
                role="current",
            )
            for i in range(pop.shape[0])
        ]
