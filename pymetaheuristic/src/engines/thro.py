"""pyMetaheuristic src - Tianji Horse Racing Optimizer Engine.

Paper-faithful THRO port: two horse populations, dynamic race matching,
Levy running factor, greedy replacement, random out-of-bound reinitialization,
and post-race training.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from .protocol import CapabilityProfile, EngineState
from ._ported_common import PortedPopulationEngine, levy_flight


class THROEngine(PortedPopulationEngine):
    """Tianji Horse Racing Optimizer (THRO)."""

    algorithm_id = "thro"
    algorithm_name = "Tianji Horse Racing Optimizer"
    family = "human"
    _REFERENCE = {
        "doi": "10.1007/s10462-025-11269-9",
        "source": "Wang et al. (2025), Tianji's horse racing optimization (THRO).",
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=False,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_checkpoint=True,
        supports_native_constraints=False,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        population_size=30,       # total horses: Tianji + King, matching the released MATLAB driver
        iterations=500,           # used for the p(t) schedule when max_steps is not supplied
        levy_beta=1.5,
        min_horses_per_side=4,    # the paper generalizes the story to n > 3 horses per side
    )

    _OPERATORS = (
        "thro.initialization",
        "thro.competition_scenario_1_slowest_vs_slowest",
        "thro.competition_scenario_2_slowest_vs_fastest",
        "thro.competition_scenario_3_fastest_vs_fastest",
        "thro.competition_scenario_4_slowest_vs_fastest",
        "thro.competition_scenario_5_tie_slowest_vs_fastest",
        "thro.training_random_peer_difference",
        "thro.training_fastest_guidance",
        "thro.greedy_selection",
        "thro.random_bound_repair",
        "thro.candidate_injection",
    )

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        min_side = max(1, int(self._params.get("min_horses_per_side", 4)))
        if "horses_per_side" in self._params:
            n_total = 2 * max(min_side, int(self._params["horses_per_side"]))
        else:
            n_total = max(2 * min_side, int(self._n))
            if n_total % 2:
                n_total += 1
        self._n = int(n_total)
        self._levy_beta = float(self._params.get("levy_beta", 1.5))

    @classmethod
    def _zero_counts(cls) -> dict[str, int]:
        return {label: 0 for label in cls._OPERATORS}

    @classmethod
    def _zero_contrib(cls) -> dict[str, float]:
        return {label: 0.0 for label in cls._OPERATORS}

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        counts = self._zero_counts()
        contrib = self._zero_contrib()
        counts["thro.initialization"] = int(pop.shape[0])
        return {
            "horses_per_side": int(pop.shape[0] // 2),
            "operator_counts": counts.copy(),
            "operator_contributions": contrib.copy(),
            "last_operator_counts": counts,
            "last_operator_contributions": contrib,
            "thro_p": 1.0,
            "native_evomapx_operator_labels": True,
            "evomapx_fidelity": "native",
        }

    def _iterations_for_schedule(self) -> int:
        total = self.config.max_steps
        if total is None:
            total = self._params.get("iterations", 500)
        return max(1, int(total))

    def _progress_weight(self, state: EngineState) -> float:
        return max(0.0, 1.0 - float(state.step + 1) / float(self._iterations_for_schedule()))

    def _levy_scalar(self) -> float:
        return float(levy_flight(1, beta=self._levy_beta, scale=1.0)[0])

    def _binary_running_mask(self, dim: int) -> np.ndarray:
        mask = np.zeros(dim, dtype=float)
        rand_num = int(np.ceil(np.sin(np.pi * 0.5 * np.random.random()) * dim))
        if rand_num > 0:
            mask[np.random.permutation(dim)[:rand_num]] = 1.0
        return mask

    @staticmethod
    def _alpha() -> float:
        return float(1.0 + round(0.5 * (0.5 + np.random.random())) * np.random.randn())

    @staticmethod
    def _beta() -> float:
        return float(round(0.5 * (0.1 + np.random.random())) * np.random.randn())

    def _is_slower(self, a: float, b: float) -> bool:
        return self._is_better(b, a)

    def _space_bound(self, x: np.ndarray, counts: dict[str, int]) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        bad = (y < self._lo) | (y > self._hi) | ~np.isfinite(y)
        n_bad = int(np.count_nonzero(bad))
        if n_bad:
            counts["thro.random_bound_repair"] += n_bad
            random_reset = np.random.uniform(self._lo, self._hi, y.shape)
            y[bad] = random_reset[bad]
        return self.problem.clip_position(y)

    def _improvement(self, old_fit: float, new_fit: float) -> float:
        if self.problem.objective == "min":
            return max(0.0, float(old_fit) - float(new_fit))
        return max(0.0, float(new_fit) - float(old_fit))

    def _competition_trial(
        self,
        current: np.ndarray,
        guide: np.ndarray,
        direction: np.ndarray,
        mean_diff: np.ndarray,
        running_factor: np.ndarray,
        alpha: float,
        beta: float,
        p: float,
    ) -> np.ndarray:
        return ((p * current + (1.0 - p) * guide) + running_factor * (direction + p * mean_diff)) * alpha + beta

    def _try_replace(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        idx: int,
        raw_trial: np.ndarray,
        label: str,
        counts: dict[str, int],
        contrib: dict[str, float],
    ) -> tuple[bool, int]:
        old_fit = float(fitness[idx])
        trial = self._space_bound(raw_trial, counts)
        new_fit = float(self.problem.evaluate(trial))
        if self._is_better(new_fit, old_fit):
            positions[idx, :] = trial
            fitness[idx] = new_fit
            counts["thro.greedy_selection"] += 1
            contrib[label] += self._improvement(old_fit, new_fit)
            contrib["thro.greedy_selection"] += self._improvement(old_fit, new_fit)
            return True, 1
        return False, 1

    def _add_training_contribution(
        self,
        old_fit: float,
        new_fit: float,
        label_counter: Counter,
        contrib: dict[str, float],
    ) -> None:
        gain = self._improvement(old_fit, new_fit)
        total_dims = sum(label_counter.values())
        if gain <= 0.0 or total_dims <= 0:
            return
        for label, n_dims in label_counter.items():
            contrib[label] += gain * float(n_dims) / float(total_dims)
        contrib["thro.greedy_selection"] += gain

    def _training_candidate(
        self,
        positions: np.ndarray,
        idx: int,
        fastest_idx: int,
        counts: dict[str, int],
    ) -> tuple[np.ndarray, Counter]:
        n, dim = positions.shape
        candidate = np.empty(dim, dtype=float)
        labels: Counter = Counter()
        total = self._iterations_for_schedule()
        tau = max(0.0, 1.0 - float(self._current_iteration) / float(total))
        for j in range(dim):
            if np.random.random() > 0.5:
                pair = np.random.permutation(n)[:2]
                lt = 0.2 * self._levy_scalar()
                candidate[j] = positions[idx, j] + lt * (positions[pair[0], j] - positions[pair[1], j])
                label = "thro.training_random_peer_difference"
            else:
                mt = 0.5 * (1.0 + 0.001 * tau * tau * np.sin(np.pi * np.random.random()))
                candidate[j] = positions[fastest_idx, j] + mt * (positions[fastest_idx, j] - positions[idx, j])
                label = "thro.training_fastest_guidance"
            labels[label] += 1
            counts[label] += 1
        return candidate, labels

    def _step_impl(self, state: EngineState, pop: np.ndarray) -> tuple[np.ndarray, int, dict]:
        dim = self.problem.dimension
        total_n = int(pop.shape[0])
        if total_n % 2:
            # Defensive repair for checkpoints created by older odd-population wrappers.
            best_idx = self._best_index(pop[:, -1])
            pop = np.vstack([pop, pop[best_idx].copy()])
            total_n += 1
        n = total_n // 2
        p = self._progress_weight(state)
        self._current_iteration = int(state.step + 1)

        counts = self._zero_counts()
        contrib = self._zero_contrib()
        evals = 0

        perm = np.random.permutation(total_n)
        mixed = pop[perm]
        t_pos = mixed[:n, :-1].copy()
        t_fit = mixed[:n, -1].copy()
        k_pos = mixed[n:, :-1].copy()
        k_fit = mixed[n:, -1].copy()

        t_order = self._order(t_fit)
        k_order = self._order(k_fit)
        t_pos, t_fit = t_pos[t_order], t_fit[t_order]
        k_pos, k_fit = k_pos[k_order], k_fit[k_order]

        t_masks = np.vstack([self._binary_running_mask(dim) for _ in range(n)])
        k_masks = np.vstack([self._binary_running_mask(dim) for _ in range(n)])
        t_slow, t_fast = n - 1, 0
        k_slow, k_fast = n - 1, 0

        for race in range(n):
            t_alpha, t_beta = self._alpha(), self._beta()
            k_alpha, k_beta = self._alpha(), self._beta()
            t_R = self._levy_scalar() * t_masks[race]
            k_R = self._levy_scalar() * k_masks[race]

            # Scenario 1: Tianji's current slowest beats King's current slowest.
            if self._is_better(t_fit[t_slow], k_fit[k_slow]):
                label = "thro.competition_scenario_1_slowest_vs_slowest"
                counts[label] += 2
                t_current = t_pos[t_slow].copy()
                k_current = k_pos[k_slow].copy()
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_t = self._competition_trial(
                    t_current, t_pos[0], t_pos[0] - t_current, mean_diff, t_R, t_alpha, t_beta, p
                )
                _, used = self._try_replace(t_pos, t_fit, t_slow, raw_t, label, counts, contrib)
                evals += used
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_k = self._competition_trial(
                    k_current, t_current, t_current - k_current, mean_diff, k_R, k_alpha, k_beta, p
                )
                _, used = self._try_replace(k_pos, k_fit, k_slow, raw_k, label, counts, contrib)
                evals += used
                t_slow -= 1
                k_slow -= 1

            # Scenario 2: Tianji's current slowest loses to King's current slowest.
            elif self._is_slower(t_fit[t_slow], k_fit[k_slow]):
                label = "thro.competition_scenario_2_slowest_vs_fastest"
                counts[label] += 2
                t_current = t_pos[t_slow].copy()
                k_current = k_pos[k_fast].copy()
                tr1 = int(np.random.randint(n))
                t_rand = t_pos[tr1].copy()
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_t = self._competition_trial(
                    t_current, t_rand, t_rand - t_current, mean_diff, t_R, t_alpha, t_beta, p
                )
                _, used = self._try_replace(t_pos, t_fit, t_slow, raw_t, label, counts, contrib)
                evals += used
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_k = self._competition_trial(
                    k_current, k_pos[0], k_pos[0] - k_current, mean_diff, k_R, k_alpha, k_beta, p
                )
                _, used = self._try_replace(k_pos, k_fit, k_fast, raw_k, label, counts, contrib)
                evals += used
                t_slow -= 1
                k_fast += 1

            # Remaining branches are the equal-slowest cases.
            elif self._is_better(t_fit[t_fast], k_fit[k_fast]):
                label = "thro.competition_scenario_3_fastest_vs_fastest"
                counts[label] += 2
                t_current = t_pos[t_fast].copy()
                k_current = k_pos[k_fast].copy()
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_t = self._competition_trial(
                    t_current, t_pos[0], t_pos[0] - t_current, mean_diff, t_R, t_alpha, t_beta, p
                )
                _, used = self._try_replace(t_pos, t_fit, t_fast, raw_t, label, counts, contrib)
                evals += used
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_k = self._competition_trial(
                    k_current, t_current, t_current - k_current, mean_diff, k_R, k_alpha, k_beta, p
                )
                _, used = self._try_replace(k_pos, k_fit, k_fast, raw_k, label, counts, contrib)
                evals += used
                t_fast += 1
                k_fast += 1

            elif self._is_slower(t_fit[t_fast], k_fit[k_fast]):
                label = "thro.competition_scenario_4_slowest_vs_fastest"
                counts[label] += 2
                t_current = t_pos[t_slow].copy()
                k_current = k_pos[k_fast].copy()
                tr2 = int(np.random.randint(n))
                t_rand = t_pos[tr2].copy()
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_t = self._competition_trial(
                    t_current, t_rand, t_rand - t_current, mean_diff, t_R, t_alpha, t_beta, p
                )
                _, used = self._try_replace(t_pos, t_fit, t_slow, raw_t, label, counts, contrib)
                evals += used
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_k = self._competition_trial(
                    k_current, k_pos[0], k_pos[0] - k_current, mean_diff, k_R, k_alpha, k_beta, p
                )
                _, used = self._try_replace(k_pos, k_fit, k_fast, raw_k, label, counts, contrib)
                evals += used
                t_slow -= 1
                k_fast += 1

            else:
                label = "thro.competition_scenario_5_tie_slowest_vs_fastest"
                counts[label] += 2
                t_current = t_pos[t_slow].copy()
                k_current = k_pos[k_fast].copy()
                tr3 = int(np.random.randint(n))
                t_rand = t_pos[tr3].copy()
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_t = self._competition_trial(
                    t_current, t_rand, t_rand - t_current, mean_diff, t_R, t_alpha, t_beta, p
                )
                _, used = self._try_replace(t_pos, t_fit, t_slow, raw_t, label, counts, contrib)
                evals += used
                mean_diff = np.mean(t_pos, axis=0) - np.mean(k_pos, axis=0)
                raw_k = self._competition_trial(
                    k_current, k_pos[0], k_pos[0] - k_current, mean_diff, k_R, k_alpha, k_beta, p
                )
                _, used = self._try_replace(k_pos, k_fit, k_fast, raw_k, label, counts, contrib)
                evals += used
                t_slow -= 1
                k_fast += 1

        t_fastest = self._best_index(t_fit)
        k_fastest = self._best_index(k_fit)
        for i in range(n):
            old = float(t_fit[i])
            raw_t, labels_t = self._training_candidate(t_pos, i, t_fastest, counts)
            trial_t = self._space_bound(raw_t, counts)
            new_t = float(self.problem.evaluate(trial_t))
            evals += 1
            if self._is_better(new_t, old):
                t_pos[i, :] = trial_t
                t_fit[i] = new_t
                counts["thro.greedy_selection"] += 1
                self._add_training_contribution(old, new_t, labels_t, contrib)

            old = float(k_fit[i])
            raw_k, labels_k = self._training_candidate(k_pos, i, k_fastest, counts)
            trial_k = self._space_bound(raw_k, counts)
            new_k = float(self.problem.evaluate(trial_k))
            evals += 1
            if self._is_better(new_k, old):
                k_pos[i, :] = trial_k
                k_fit[i] = new_k
                counts["thro.greedy_selection"] += 1
                self._add_training_contribution(old, new_k, labels_k, contrib)

        new_pop = np.vstack([np.hstack((t_pos, t_fit[:, None])), np.hstack((k_pos, k_fit[:, None]))])
        totals = dict(state.payload.get("operator_counts", self._zero_counts()))
        total_contrib = dict(state.payload.get("operator_contributions", self._zero_contrib()))
        for label in self._OPERATORS:
            totals[label] = int(totals.get(label, 0)) + int(counts.get(label, 0))
            total_contrib[label] = float(total_contrib.get(label, 0.0)) + float(contrib.get(label, 0.0))

        return new_pop, evals, {
            "horses_per_side": int(n),
            "thro_p": float(p),
            "operator_counts": totals,
            "operator_contributions": total_contrib,
            "last_operator_counts": counts,
            "last_operator_contributions": contrib,
            "native_evomapx_operator_labels": True,
            "evomapx_delta_f": "direct_improvement",
            "evomapx_fidelity": "native",
        }

    def _post_injection_repair(self, state: EngineState, replaced_indices: list[int], candidates) -> None:
        super()._post_injection_repair(state, replaced_indices, candidates)
        counts = dict(state.payload.get("operator_counts", self._zero_counts()))
        last_counts = self._zero_counts()
        k = int(len(replaced_indices))
        counts["thro.candidate_injection"] = int(counts.get("thro.candidate_injection", 0)) + k
        last_counts["thro.candidate_injection"] = k
        state.payload["operator_counts"] = counts
        state.payload["last_operator_counts"] = last_counts
        state.payload["last_operator_contributions"] = self._zero_contrib()

    def observe(self, state: EngineState) -> dict:
        obs = super().observe(state)
        obs.update(
            {
                "horses_per_side": int(state.payload.get("horses_per_side", 0)),
                "thro_p": float(state.payload.get("thro_p", self._progress_weight(state))),
                "operator_counts": dict(state.payload.get("last_operator_counts", state.payload.get("operator_counts", {}))),
                "operator_contributions": dict(
                    state.payload.get("last_operator_contributions", state.payload.get("operator_contributions", {}))
                ),
                "operator_counts_total": dict(state.payload.get("operator_counts", {})),
                "operator_contributions_total": dict(state.payload.get("operator_contributions", {})),
                "evomapx_delta_f": "direct_improvement",
                "evomapx_fidelity": "native",
                "native_evomapx_operator_labels": True,
            }
        )
        return obs
