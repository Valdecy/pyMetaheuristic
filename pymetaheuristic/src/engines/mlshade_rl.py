"""pyMetaheuristic src — mLSHADE-RL Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class MLSHADERLEngine(PortedPopulationEngine):
    """Multi-operator Ensemble LSHADE with Restart and Local Search.

    Implements the core mechanisms described for mLSHADE-RL: three adaptive DE
    mutation strategies, LSHADE-style success memories, linear population size
    reduction, stagnation-triggered horizontal/vertical restart, and optional
    late-stage SLSQP local search with a safe derivative-free fallback.
    """

    algorithm_id = "mlshade_rl"
    algorithm_name = "mLSHADE-RL"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.48550/arXiv.2409.15994",
        "title": "A Multi-operator Ensemble LSHADE with Restart and Local Search Mechanisms for Single-objective Optimization",
        "authors": "Dikshit Chauhan, Anupam Trivedi, and Shivani",
        "year": 2024,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_restart=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        has_archive=True,
    )
    _DEFAULTS = dict(
        population_size=None,
        min_population_size=4,
        hist_mem_size=6,
        extern_arc_rate=2.6,
        pbest_factor=0.11,
        covariance_rate=0.25,
        crossover_probability=0.4,
        neighborhood_fraction=0.5,
        sinusoid_frequency=0.5,
        learning_period=20,
        local_search_probability=0.01,
        local_search_success_probability=0.1,
        local_search_start=0.85,
        local_search_evals=40,
    )

    def __init__(self, problem, config) -> None:
        params = {**self._DEFAULTS, **config.params}
        if params.get("population_size") is None:
            params["population_size"] = max(18 * int(problem.dimension), 20)
            config.params = {**config.params, "population_size": params["population_size"]}
        super().__init__(problem, config)
        if self._n < 4:
            raise ValueError("population_size must be >= 4 for mLSHADE-RL.")
        self._last_operator_contributions = self._blank_operator_contribs()
        self._last_operator_counts = {key: 0 for key in self._last_operator_contributions}

    def _blank_operator_contribs(self) -> dict[str, float]:
        return {
            "mlshade_rl.ms1_current_to_pbest_weight_archive": 0.0,
            "mlshade_rl.ms2_current_to_pbest_no_archive": 0.0,
            "mlshade_rl.ms3_current_to_ordpbest_weight": 0.0,
            "mlshade_rl.crossover": 0.0,
            "mlshade_rl.selection": 0.0,
            "mlshade_rl.strategy_probability_update": 0.0,
            "mlshade_rl.parameter_adaptation": 0.0,
            "mlshade_rl.archive_update": 0.0,
            "mlshade_rl.population_reduction": 0.0,
            "mlshade_rl.restart": 0.0,
            "mlshade_rl.local_search": 0.0,
        }

    @staticmethod
    def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        return float(np.sum(weights * values * values) / (np.sum(weights * values) + 1.0e-30))

    def _sample_f_cauchy(self, center: float) -> float:
        for _ in range(100):
            value = float(np.random.standard_cauchy() * 0.1 + center)
            if value > 0.0:
                return min(value, 1.0)
        return 0.5

    def _progress(self, state, extra_evals: int = 0) -> float:
        if self.config.max_evaluations is not None and self.config.max_evaluations > 0:
            return min(1.0, max(0.0, float(state.evaluations + extra_evals) / float(self.config.max_evaluations)))
        horizon = max(1, int(self.config.max_steps or 100))
        return min(1.0, max(0.0, float(state.step + 1) / float(horizon)))

    def _sample_f(self, M_F: np.ndarray, M_freq: np.ndarray, state, r: int) -> tuple[float, int]:
        progress = self._progress(state)
        if progress < 0.5:
            freq = float(np.clip(M_freq[r] if np.isfinite(M_freq[r]) else self._params.get("sinusoid_frequency", 0.5), 0.05, 1.0))
            if np.random.rand() < 0.5:
                F = 0.5 * (np.sin(np.pi * (2.0 * freq * (state.step + 1) + 1.0)) * (1.0 - progress) + 1.0)
                return float(np.clip(F, 0.05, 1.0)), 0
            freq2 = float(np.clip(np.random.standard_cauchy() * 0.1 + freq, 0.05, 1.0))
            F = 0.5 * (np.sin(np.pi * (2.0 * freq2 * (state.step + 1) + 1.0)) * progress + 1.0)
            return float(np.clip(F, 0.05, 1.0)), 1
        return self._sample_f_cauchy(float(M_F[r])), 2

    def _sample_cr(self, M_CR: np.ndarray, r: int) -> float:
        center = float(M_CR[r])
        if not np.isfinite(center):
            return 0.0
        return float(np.clip(np.random.normal(center, 0.1), 0.0, 1.0))

    def _repair_bound(self, donor: np.ndarray, parent: np.ndarray) -> np.ndarray:
        donor = np.asarray(donor, dtype=float).copy()
        below = donor < self._lo
        above = donor > self._hi
        donor[below] = (self._lo[below] + parent[below]) / 2.0
        donor[above] = (self._hi[above] + parent[above]) / 2.0
        return np.clip(donor, self._lo, self._hi)

    def _weighted_scale(self, F: float, state) -> float:
        progress = self._progress(state)
        if progress <= 0.2:
            return 0.7 * F
        if progress <= 0.4:
            return 0.8 * F
        return 1.2 * F

    def _ord_vectors(self, pop: np.ndarray, order: np.ndarray, pnum: int, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = pop.shape[0]
        pbest = pop[np.random.choice(order[:pnum]), :-1]
        ids = self._rand_indices(n, i, 2)
        candidates = np.vstack((pbest, pop[ids[0], :-1], pop[ids[1], :-1]))
        cand_fit = np.array([
            pop[np.argmin(np.linalg.norm(pop[:, :-1] - candidates[j], axis=1)), -1]
            for j in range(3)
        ])
        rank = self._order(cand_fit)
        return candidates[rank[0]], candidates[rank[1]], candidates[rank[2]]

    def _covariance_crossover(self, target: np.ndarray, donor: np.ndarray, pop_pos: np.ndarray, CR: float) -> np.ndarray:
        n, dim = pop_pos.shape
        frac = float(np.clip(self._params.get("neighborhood_fraction", 0.5), 0.1, 1.0))
        m = max(2, min(n, int(round(frac * n))))
        best = pop_pos[0]
        d = np.linalg.norm(pop_pos - best, axis=1)
        neighbourhood = pop_pos[np.argsort(d)[:m]]
        try:
            cov = np.cov(neighbourhood.T) + np.eye(dim) * 1.0e-12
            _, eigvec = np.linalg.eigh(cov)
            target_t = eigvec.T @ target
            donor_t = eigvec.T @ donor
            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            trial_t = np.where(cross, donor_t, target_t)
            return eigvec @ trial_t
        except Exception:
            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            return np.where(cross, donor, target)

    def _initialize_payload(self, pop):
        h = int(self._params.get("hist_mem_size", 6))
        if h <= 0:
            raise ValueError("hist_mem_size must be positive for mLSHADE-RL.")
        return {
            "M_F": np.full(h, 0.5, dtype=float),
            "M_CR": np.full(h, 0.9, dtype=float),
            "M_freq": np.full(h, float(self._params.get("sinusoid_frequency", 0.5)), dtype=float),
            "k": 0,
            "archive": pop[:, :-1].copy(),
            "initial_n": int(pop.shape[0]),
            "strategy_probs": np.full(3, 1.0 / 3.0, dtype=float),
            "stagnation_count": np.zeros(pop.shape[0], dtype=int),
            "local_search_probability": float(self._params.get("local_search_probability", 0.01)),
        }

    def _restart_volume(self, pop: np.ndarray) -> float:
        pos = pop[:, :-1]
        span_bnd = np.maximum(self._hi - self._lo, 1.0e-30)
        span_pop = np.maximum(np.max(pos, axis=0) - np.min(pos, axis=0), 0.0)
        ratio = np.prod(np.clip(span_pop / span_bnd, 0.0, 1.0))
        return float(np.sqrt(max(ratio, 0.0)))

    def _horizontal_restart(self, pop: np.ndarray, i: int) -> np.ndarray:
        n = pop.shape[0]
        j = self._rand_indices(n, i, 1)[0]
        rd1 = np.random.rand(self.problem.dimension)
        rd2 = 1.0 - rd1
        rnds = np.random.uniform(-1.0, 1.0, self.problem.dimension)
        return rd1 * pop[i, :-1] + rd2 * pop[j, :-1] + rnds * (pop[i, :-1] - pop[j, :-1])

    def _vertical_restart(self, pop: np.ndarray, i: int) -> np.ndarray:
        x = pop[i, :-1].copy()
        dim = self.problem.dimension
        if dim < 2:
            return np.random.uniform(self._lo, self._hi, dim)
        d1, d2 = np.random.choice(dim, 2, replace=False)
        rd1 = float(np.random.rand())
        x[d1] = rd1 * x[d1] + (1.0 - rd1) * x[d2]
        return x

    def _apply_restart(self, pop: np.ndarray, counts: np.ndarray, contrib: dict[str, float], op_counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray, int]:
        if pop.shape[0] == 0:
            return pop, counts, 0
        dim = self.problem.dimension
        volume = self._restart_volume(pop)
        trigger = (counts > 2 * dim) & (volume < 0.001)
        idxs = np.where(trigger)[0]
        if idxs.size == 0:
            return pop, counts, 0
        evals = 0
        for i in idxs:
            if np.random.rand() > 0.5:
                trial = self._horizontal_restart(pop, int(i))
            else:
                trial = self._vertical_restart(pop, int(i))
            trial = self._repair_bound(trial, pop[i, :-1])
            fit = float(self.problem.evaluate(trial))
            evals += 1
            if self._is_better(fit, float(pop[i, -1])):
                gain = abs(float(pop[i, -1]) - fit)
                pop[i, :-1] = trial
                pop[i, -1] = fit
                counts[i] = 0
                contrib["mlshade_rl.restart"] += float(gain)
            else:
                counts[i] = 0
            op_counts["mlshade_rl.restart"] += 1
        return pop, counts, evals

    def _local_search(self, state, pop: np.ndarray, pls: float, contrib: dict[str, float], op_counts: dict[str, int]) -> tuple[np.ndarray, float, int]:
        if self._progress(state) < float(self._params.get("local_search_start", 0.85)):
            return pop, pls, 0
        if np.random.rand() > pls:
            return pop, pls, 0
        best_idx = self._best_index(pop[:, -1])
        x0 = pop[best_idx, :-1].copy()
        old_fit = float(pop[best_idx, -1])
        budget = max(1, int(self._params.get("local_search_evals", 40)))
        evals = 0
        best_x = x0.copy()
        best_f = old_fit

        try:
            from scipy.optimize import minimize  # type: ignore

            def wrapped(x):
                nonlocal evals
                if evals >= budget:
                    return best_f
                evals += 1
                return float(self.problem.evaluate(np.clip(x, self._lo, self._hi)))

            result = minimize(
                wrapped,
                x0,
                method="SLSQP",
                bounds=list(zip(self._lo, self._hi)),
                options={"maxiter": max(1, budget // 2), "ftol": 1.0e-12, "disp": False},
            )
            cand = np.clip(np.asarray(result.x, dtype=float), self._lo, self._hi)
            cand_f = float(self.problem.evaluate(cand))
            evals += 1
            if self._is_better(cand_f, best_f):
                best_x, best_f = cand, cand_f
        except Exception:
            radius = 0.02 * self._span
            for _ in range(budget):
                cand = np.clip(best_x + np.random.normal(0.0, radius, best_x.shape), self._lo, self._hi)
                cand_f = float(self.problem.evaluate(cand))
                evals += 1
                if self._is_better(cand_f, best_f):
                    best_x, best_f = cand, cand_f
                    radius *= 0.8

        op_counts["mlshade_rl.local_search"] += 1
        if self._is_better(best_f, old_fit):
            pop[best_idx, :-1] = best_x
            pop[best_idx, -1] = best_f
            contrib["mlshade_rl.local_search"] += abs(old_fit - best_f)
            pls = float(self._params.get("local_search_success_probability", 0.1))
        else:
            pls = float(self._params.get("local_search_probability", 0.01))
        return pop, pls, evals

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        M_F = np.asarray(state.payload.get("M_F", np.full(6, 0.5)), dtype=float)
        M_CR = np.asarray(state.payload.get("M_CR", np.full(6, 0.9)), dtype=float)
        M_freq = np.asarray(state.payload.get("M_freq", np.full(6, 0.5)), dtype=float)
        archive = np.asarray(state.payload.get("archive", np.empty((0, dim))), dtype=float).reshape(-1, dim)
        k = int(state.payload.get("k", 0)) % len(M_F)
        initial_n = int(state.payload.get("initial_n", n))
        probs = np.asarray(state.payload.get("strategy_probs", np.full(3, 1.0 / 3.0)), dtype=float)
        probs = np.maximum(probs, 0.1)
        probs = probs / np.sum(probs)
        counts = np.asarray(state.payload.get("stagnation_count", np.zeros(n)), dtype=int)
        if counts.size != n:
            counts = np.resize(counts, n)
            counts[counts < 0] = 0
        pls = float(state.payload.get("local_search_probability", self._params.get("local_search_probability", 0.01)))

        order = self._order(pop[:, -1])
        ranked_positions = pop[order, :-1]
        p = float(np.clip(self._params.get("pbest_factor", 0.11), 2.0 / max(2, n), 1.0))
        pnum = max(2, int(round(p * n)))
        union = np.vstack((pop[:, :-1], archive)) if archive.size else pop[:, :-1]
        pc = float(np.clip(self._params.get("crossover_probability", 0.4), 0.0, 1.0))

        trials: list[np.ndarray] = []
        Fs: list[float] = []
        CRs: list[float] = []
        freq_success: list[float] = []
        strategy_used: list[int] = []
        f_modes: list[int] = []
        contrib = self._blank_operator_contribs()
        op_counts = {key: 0 for key in contrib}

        for i in range(n):
            r = int(np.random.randint(len(M_F)))
            F, f_mode = self._sample_f(M_F, M_freq, state, r)
            CR = self._sample_cr(M_CR, r)
            Fw = self._weighted_scale(F, state)
            choice = int(np.random.choice(3, p=probs))
            parent = pop[i, :-1]
            pbest = pop[np.random.choice(order[:pnum]), :-1]

            if choice == 0:
                r1 = pop[self._rand_indices(n, i, 1)[0], :-1]
                r2 = union[np.random.randint(union.shape[0])]
                donor = parent + Fw * (pbest - parent) + F * (r1 - r2)
                op_label = "mlshade_rl.ms1_current_to_pbest_weight_archive"
            elif choice == 1:
                r1, r3 = pop[self._rand_indices(n, i, 2), :-1]
                donor = parent + F * (pbest - parent + r1 - r3)
                op_label = "mlshade_rl.ms2_current_to_pbest_no_archive"
            else:
                ord_best, ord_med, ord_worst = self._ord_vectors(pop, order, pnum, i)
                donor = parent + Fw * (ord_best - parent + ord_med - ord_worst)
                op_label = "mlshade_rl.ms3_current_to_ordpbest_weight"

            donor = self._repair_bound(donor, parent)
            if np.random.rand() < pc:
                trial = self._covariance_crossover(parent, donor, ranked_positions, CR)
            else:
                cross = np.random.rand(dim) < CR
                cross[np.random.randint(dim)] = True
                trial = np.where(cross, donor, parent)
            trial = self._repair_bound(trial, parent)
            trials.append(trial)
            Fs.append(F)
            CRs.append(CR)
            strategy_used.append(choice)
            f_modes.append(f_mode)
            if f_mode == 1:
                freq_success.append(float(M_freq[r]))
            op_counts[op_label] += 1
            op_counts["mlshade_rl.crossover"] += 1

        trial_pop = self._pop_from_positions(np.asarray(trials, dtype=float))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        old_fit = pop[:, -1].copy()
        gains = np.zeros(n, dtype=float)
        if self.problem.objective == "min":
            gains[mask] = old_fit[mask] - trial_pop[mask, -1]
        else:
            gains[mask] = trial_pop[mask, -1] - old_fit[mask]
        gains = np.maximum(gains, 0.0)

        for choice, gain in zip(strategy_used, gains):
            label = [
                "mlshade_rl.ms1_current_to_pbest_weight_archive",
                "mlshade_rl.ms2_current_to_pbest_no_archive",
                "mlshade_rl.ms3_current_to_ordpbest_weight",
            ][int(choice)]
            contrib[label] += float(gain) * 0.5
            contrib["mlshade_rl.crossover"] += float(gain) * 0.25
            contrib["mlshade_rl.selection"] += float(gain) * 0.25
        op_counts["mlshade_rl.selection"] = int(np.count_nonzero(mask))

        counts[mask] = 0
        counts[~mask] += 1
        if np.any(mask):
            archive = np.vstack((archive, pop[mask, :-1])) if archive.size else pop[mask, :-1].copy()
            max_arc = max(1, int(float(self._params.get("extern_arc_rate", 2.6)) * n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            op_counts["mlshade_rl.archive_update"] = int(np.count_nonzero(mask))

            sf = np.asarray(Fs, dtype=float)[mask]
            scr = np.asarray(CRs, dtype=float)[mask]
            df = np.abs(old_fit[mask] - trial_pop[mask, -1])
            weights = df / (float(np.sum(df)) + 1.0e-30)
            M_F[k] = self._weighted_lehmer(sf, weights)
            if np.max(scr) <= 0.0 or not np.isfinite(M_CR[k]):
                M_CR[k] = np.nan
            else:
                M_CR[k] = self._weighted_lehmer(scr, weights)
            successful_freqs = np.asarray([freq_success[j] for j in range(min(len(freq_success), len(weights)))], dtype=float)
            if successful_freqs.size:
                freq_weights = np.full(successful_freqs.size, 1.0 / successful_freqs.size)
                M_freq[k] = self._weighted_lehmer(successful_freqs, freq_weights)
            k = (k + 1) % len(M_F)
            op_counts["mlshade_rl.parameter_adaptation"] = 1
            pop[mask] = trial_pop[mask]

        # Eq. (22)-style strategy probability update using normalized improvement.
        impacts = np.zeros(3, dtype=float)
        for s in range(3):
            idx = np.asarray(strategy_used) == s
            if np.any(idx):
                denom = float(np.sum(np.abs(old_fit[idx]))) + 1.0e-30
                impacts[s] = float(np.sum(gains[idx]) / denom)
        if np.sum(impacts) > 0.0:
            probs = np.clip(impacts / np.sum(impacts), 0.1, 0.9)
            probs = probs / np.sum(probs)
            op_counts["mlshade_rl.strategy_probability_update"] = 1

        restart_evals = 0
        pop, counts, restart_evals = self._apply_restart(pop, counts, contrib, op_counts)

        # Linear population reduction.
        min_n = max(4, int(self._params.get("min_population_size", 4)))
        progress_after = self._progress(state, extra_evals=n + restart_evals)
        target_n = max(min_n, int(round(initial_n + (min_n - initial_n) * progress_after)))
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            removed = pop.shape[0] - target_n
            pop = pop[keep]
            counts = counts[keep]
            max_arc = max(1, int(float(self._params.get("extern_arc_rate", 2.6)) * target_n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            op_counts["mlshade_rl.population_reduction"] = int(removed)

        pop, pls, local_evals = self._local_search(state, pop, pls, contrib, op_counts)

        self._last_operator_contributions = {key: float(value) for key, value in contrib.items()}
        self._last_operator_counts = {key: int(value) for key, value in op_counts.items()}
        return pop, n + restart_evals + local_evals, {
            "M_F": M_F,
            "M_CR": M_CR,
            "M_freq": M_freq,
            "k": k,
            "archive": archive,
            "initial_n": initial_n,
            "strategy_probs": probs,
            "stagnation_count": counts,
            "local_search_probability": pls,
        }

    def observe(self, state):
        obs = super().observe(state)
        obs["operator_contributions"] = dict(self._last_operator_contributions)
        obs["operator_counts"] = dict(self._last_operator_counts)
        obs["evomapx_delta_f"] = "signed"
        obs["evomapx_fidelity"] = "native"
        return obs
