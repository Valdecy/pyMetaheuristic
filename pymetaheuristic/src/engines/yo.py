"""pyMetaheuristic src — Yukthi Opus Engine.

Yukthi Opus (YO) is implemented as a multi-chain continuous-box optimizer
following the paper's two-phase design:

1. MCMC burn-in with Metropolis acceptance for global exploration.
2. Hybrid optimization combining MCMC proposals, greedy local refinement,
   simulated-annealing acceptance, adaptive reheating, and a spatial blacklist.

The implementation keeps EvoMapX instrumentation passive: every operator
contribution is computed from fitness values already evaluated by the optimizer.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any

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


class YOEngine(BaseEngine):
    """Yukthi Opus — multi-chain hybrid MCMC/greedy/SA optimizer."""

    algorithm_id = "yo"
    algorithm_name = "Yukthi Opus"
    family = "trajectory"
    _REFERENCE = {
        "doi": "10.48550/arXiv.2601.01832",
        "title": "Yukthi Opus: A Multi-Chain Hybrid Metaheuristic for Large-Scale NP-Hard Optimization",
        "authors": "SB Danush Vikraman, Hannah Abagail, Prasanna Kesavraj, Gajanan V. Honnavar",
        "year": 2026,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=False,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )

    _DEFAULTS = dict(
        chains=4,
        burn_in_fraction=0.30,
        initial_temperature=1.0,
        cooling_rate=0.95,
        reheating_factor=1.50,
        reheat_stagnation=10,
        mcmc_step_size=0.10,
        greedy_step_size=0.05,
        greedy_steps=2,
        greedy_neighbors=None,
        min_temperature=1.0e-12,
        blacklist_enabled=True,
        blacklist_cell_size=0.20,
        blacklist_threshold_factor=10.0,
        blacklist_max_size=512,
        blacklist_resample_attempts=5,
        metropolis_clip=700.0,
    )

    _DIRECT_OPERATORS = (
        "yo.mcmc_burn_in",
        "yo.post_burnin_selection",
        "yo.mcmc_proposal",
        "yo.greedy_refinement",
        "yo.simulated_annealing_acceptance",
    )
    _DIAGNOSTIC_OPERATORS = (
        "yo.blacklist_filter",
        "yo.adaptive_reheating",
        "yo.elite_update",
    )
    _ALL_OPERATORS = _DIRECT_OPERATORS + _DIAGNOSTIC_OPERATORS

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        params = {**self._DEFAULTS, **config.params}
        for alias in ("population_size", "pop_size", "n_chains", "num_chains"):
            if alias in config.params and "chains" not in config.params:
                params["chains"] = config.params[alias]
        self._chains = int(params["chains"])
        self._burn_frac = float(params["burn_in_fraction"])
        self._T0 = float(params["initial_temperature"])
        self._cooling = float(params["cooling_rate"])
        self._reheat_factor = float(params["reheating_factor"])
        self._reheat_stagnation = int(params["reheat_stagnation"])
        self._mcmc_step = float(params["mcmc_step_size"])
        self._greedy_step = float(params["greedy_step_size"])
        self._greedy_steps = int(params["greedy_steps"])
        greedy_neighbors = params.get("greedy_neighbors", None)
        self._greedy_neighbors = None if greedy_neighbors is None else int(greedy_neighbors)
        self._min_temperature = float(params["min_temperature"])
        self._blacklist_enabled = bool(params["blacklist_enabled"])
        self._blacklist_cell_size = float(params["blacklist_cell_size"])
        self._blacklist_threshold_factor = float(params["blacklist_threshold_factor"])
        self._blacklist_max_size = int(params["blacklist_max_size"])
        self._blacklist_resample_attempts = int(params["blacklist_resample_attempts"])
        self._metropolis_clip = float(params["metropolis_clip"])
        self._rng = np.random.default_rng(config.seed)
        self._validate_parameters()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @property
    def _lo(self) -> np.ndarray:
        return np.asarray(self.problem.min_values, dtype=float)

    @property
    def _hi(self) -> np.ndarray:
        return np.asarray(self.problem.max_values, dtype=float)

    @property
    def _span(self) -> np.ndarray:
        span = self._hi - self._lo
        return np.where(span == 0.0, 1.0, span)

    def _validate_parameters(self) -> None:
        if self._chains < 1:
            raise ValueError("yo chains must be >= 1.")
        if not (0.0 <= self._burn_frac < 1.0):
            raise ValueError("yo burn_in_fraction must be in [0, 1).")
        if self._T0 <= 0.0:
            raise ValueError("yo initial_temperature must be positive.")
        if not (0.0 < self._cooling <= 1.0):
            raise ValueError("yo cooling_rate must be in (0, 1].")
        if self._reheat_factor < 1.0:
            raise ValueError("yo reheating_factor must be >= 1.")
        if self._reheat_stagnation < 1:
            raise ValueError("yo reheat_stagnation must be >= 1.")
        if self._mcmc_step < 0.0 or self._greedy_step < 0.0:
            raise ValueError("yo step sizes must be non-negative.")
        if self._greedy_steps < 0:
            raise ValueError("yo greedy_steps must be >= 0.")
        if self._greedy_neighbors is not None and self._greedy_neighbors < 1:
            raise ValueError("yo greedy_neighbors must be >= 1 when supplied.")
        if self._min_temperature <= 0.0:
            raise ValueError("yo min_temperature must be positive.")
        if self._blacklist_cell_size <= 0.0:
            raise ValueError("yo blacklist_cell_size must be positive.")
        if self._blacklist_threshold_factor <= 0.0:
            raise ValueError("yo blacklist_threshold_factor must be positive.")
        if self._blacklist_max_size < 0:
            raise ValueError("yo blacklist_max_size must be >= 0.")
        if self._blacklist_resample_attempts < 0:
            raise ValueError("yo blacklist_resample_attempts must be >= 0.")
        if self._metropolis_clip <= 0.0:
            raise ValueError("yo metropolis_clip must be positive.")
        if self.config.max_evaluations is not None and int(self.config.max_evaluations) < 1:
            raise ValueError("yo requires max_evaluations >= 1 when an evaluation budget is supplied.")

    def _effective_chains(self) -> int:
        if self.config.max_evaluations is None:
            return self._chains
        return max(1, min(self._chains, int(self.config.max_evaluations)))

    def _clip(self, position: np.ndarray) -> np.ndarray:
        return self.problem.apply_variable_types(np.asarray(position, dtype=float))

    def _random_position(self) -> np.ndarray:
        return self._clip(self._rng.uniform(self._lo, self._hi, self.problem.dimension))

    def _energy(self, fitness: float) -> float:
        return float(fitness) if self.problem.objective == "min" else -float(fitness)

    def _is_better(self, a: float, b: float) -> bool:
        return self.problem.is_better(float(a), float(b))

    def _positive_improvement(self, before: float, after: float) -> float:
        gain = self._energy(before) - self._energy(after)
        return float(max(0.0, gain if math.isfinite(gain) else 0.0))

    def _remaining_evaluations(self, state: EngineState | None, used: int = 0) -> int | None:
        if self.config.max_evaluations is None:
            return None
        done = int(getattr(state, "evaluations", 0) if state is not None else 0)
        return max(0, int(self.config.max_evaluations) - done - int(used))

    def _budget_allows(self, state: EngineState, used: int = 0) -> bool:
        remaining = self._remaining_evaluations(state, used=used)
        return remaining is None or remaining > 0

    def _burnin_target_evaluations(self, initial_evaluations: int) -> int | None:
        if self.config.max_evaluations is None:
            return None
        return max(int(initial_evaluations), int(round(float(self.config.max_evaluations) * self._burn_frac)))

    def _burnin_target_steps(self) -> int | None:
        if self.config.max_steps is None:
            return None
        return max(0, int(round(float(self.config.max_steps) * self._burn_frac)))

    def _burnin_active(self, state: EngineState) -> bool:
        eval_target = state.payload.get("burnin_target_evaluations")
        if eval_target is not None:
            return int(state.evaluations) < int(eval_target)
        step_target = state.payload.get("burnin_target_steps")
        if step_target is not None:
            return int(state.step) < int(step_target)
        return int(state.step) == 0 and self._burn_frac > 0.0

    def _metropolis_accept(self, candidate_fit: float, current_fit: float, temperature: float) -> bool:
        candidate_energy = self._energy(candidate_fit)
        current_energy = self._energy(current_fit)
        delta = candidate_energy - current_energy
        if delta <= 0.0:
            return True
        denom = max(float(temperature), self._min_temperature)
        exponent = -min(float(delta) / denom, self._metropolis_clip)
        return bool(self._rng.random() < math.exp(exponent))

    def _mcmc_propose(self, current: np.ndarray) -> np.ndarray:
        if self._mcmc_step <= 0.0:
            return self._clip(current.copy())
        step = self._rng.normal(0.0, self._mcmc_step, size=self.problem.dimension) * self._span
        return self._clip(current + step)

    def _population_matrix(self, positions: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        return np.hstack((np.asarray(positions, dtype=float), np.asarray(fitness, dtype=float)[:, np.newaxis]))

    def _best_index(self, fitness: np.ndarray) -> int:
        return int(np.argmin(fitness) if self.problem.objective == "min" else np.argmax(fitness))

    def _blank_operator_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._ALL_OPERATORS}

    def _blank_operator_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._ALL_OPERATORS}

    def _add_contribution(self, contrib: dict[str, float], counts: dict[str, int], label: str, value: float = 0.0, count: int = 1) -> None:
        val = float(value)
        if not math.isfinite(val):
            val = 0.0
        contrib[label] = float(contrib.get(label, 0.0) + max(0.0, val))
        counts[label] = int(counts.get(label, 0) + max(0, int(count)))

    # ------------------------------------------------------------------
    # Blacklist and local refinement
    # ------------------------------------------------------------------

    def _blacklist_key(self, position: np.ndarray) -> tuple[int, ...]:
        norm = (self._clip(position) - self._lo) / self._span
        return tuple(np.floor(norm / self._blacklist_cell_size).astype(int).tolist())

    def _blacklist_contains(self, blacklist: set[tuple[int, ...]], position: np.ndarray) -> bool:
        return bool(self._blacklist_enabled and self._blacklist_max_size > 0 and self._blacklist_key(position) in blacklist)

    def _remember_blacklist(self, blacklist: set[tuple[int, ...]], position: np.ndarray) -> None:
        if not self._blacklist_enabled or self._blacklist_max_size <= 0:
            return
        blacklist.add(self._blacklist_key(position))
        while len(blacklist) > self._blacklist_max_size:
            blacklist.pop()

    def _should_blacklist(self, fitness: float, best_fitness: float) -> bool:
        cand_e = self._energy(fitness)
        best_e = self._energy(best_fitness)
        threshold = best_e + self._blacklist_threshold_factor * max(1.0, abs(best_e))
        return bool(cand_e > threshold)

    def _propose_non_blacklisted(self, current: np.ndarray, blacklist: set[tuple[int, ...]]) -> tuple[np.ndarray, int]:
        skips = 0
        proposal = self._mcmc_propose(current)
        for _ in range(self._blacklist_resample_attempts):
            if not self._blacklist_contains(blacklist, proposal):
                return proposal, skips
            skips += 1
            proposal = self._mcmc_propose(current)
        return proposal, skips

    def _coordinate_neighbors(self, current: np.ndarray, delta: float) -> list[np.ndarray]:
        moves: list[np.ndarray] = []
        if delta <= 0.0:
            return moves
        for j in range(self.problem.dimension):
            step = np.zeros(self.problem.dimension, dtype=float)
            step[j] = delta * self._span[j]
            moves.append(self._clip(current + step))
            moves.append(self._clip(current - step))
        if self._greedy_neighbors is not None:
            moves = moves[: self._greedy_neighbors]
        return moves

    def _greedy_refine(
        self,
        start: np.ndarray,
        start_fit: float,
        state: EngineState,
        used_evaluations: int,
        contrib: dict[str, float],
        counts: dict[str, int],
    ) -> tuple[np.ndarray, float, int]:
        current = self._clip(start)
        current_fit = float(start_fit)
        evals = 0
        delta = self._greedy_step
        for _ in range(self._greedy_steps):
            if delta <= 0.0:
                break
            moves = self._coordinate_neighbors(current, delta)
            if not moves:
                break
            remaining = self._remaining_evaluations(state, used=used_evaluations + evals)
            if remaining is not None:
                moves = moves[:remaining]
            if not moves:
                break
            best_pos = current
            best_fit = current_fit
            for trial in moves:
                trial_fit = float(self.problem.evaluate(trial.copy()))
                evals += 1
                if self._is_better(trial_fit, best_fit):
                    best_pos = trial
                    best_fit = trial_fit
            self._add_contribution(
                contrib,
                counts,
                "yo.greedy_refinement",
                self._positive_improvement(current_fit, best_fit),
                len(moves),
            )
            if self._is_better(best_fit, current_fit):
                current = best_pos.copy()
                current_fit = float(best_fit)
                delta *= 1.05
            else:
                delta *= 0.50
        return current, float(current_fit), int(evals)

    # ------------------------------------------------------------------
    # Mandatory engine API
    # ------------------------------------------------------------------

    def initialize(self) -> EngineState:
        n_chains = self._effective_chains()
        positions = np.vstack([self._random_position() for _ in range(n_chains)])
        fitness = np.asarray([float(self.problem.evaluate(row.copy())) for row in positions], dtype=float)
        best_idx = self._best_index(fitness)
        initial_evaluations = int(n_chains)
        population = self._population_matrix(positions, fitness)
        return EngineState(
            step=0,
            evaluations=initial_evaluations,
            best_position=positions[best_idx].astype(float).tolist(),
            best_fitness=float(fitness[best_idx]),
            initialized=True,
            payload={
                "population": population,
                "current_positions": positions.copy(),
                "current_fitness": fitness.copy(),
                "chain_best_positions": positions.copy(),
                "chain_best_fitness": fitness.copy(),
                "temperatures": np.full(n_chains, self._T0, dtype=float),
                "stagnation": np.zeros(n_chains, dtype=int),
                "phase": "burn_in" if self._burn_frac > 0.0 else "hybrid",
                "burnin_target_evaluations": self._burnin_target_evaluations(initial_evaluations),
                "burnin_target_steps": self._burnin_target_steps(),
                "post_burnin_done": False,
                "blacklist": set(),
                "accepted": 0,
                "rejected": 0,
                "blacklisted_skips": 0,
                "reheats": 0,
                "operator_labels": ["initialization"] * n_chains,
                "last_operator_contributions": self._blank_operator_contribs(),
                "last_operator_counts": self._blank_operator_counts(),
                "last_operator_metadata": {},
            },
        )

    def _post_burnin_select(self, state: EngineState, contrib: dict[str, float], counts: dict[str, int]) -> None:
        payload = state.payload
        if bool(payload.get("post_burnin_done", False)):
            return
        current_fit = np.asarray(payload["current_fitness"], dtype=float)
        current_pos = np.asarray(payload["current_positions"], dtype=float)
        best_fit = np.asarray(payload["chain_best_fitness"], dtype=float)
        best_pos = np.asarray(payload["chain_best_positions"], dtype=float)
        total_gain = 0.0
        for i in range(best_fit.shape[0]):
            total_gain += self._positive_improvement(float(current_fit[i]), float(best_fit[i]))
        payload["current_positions"] = best_pos.copy()
        payload["current_fitness"] = best_fit.copy()
        payload["temperatures"] = np.full(best_fit.shape[0], self._T0, dtype=float)
        payload["stagnation"] = np.zeros(best_fit.shape[0], dtype=int)
        payload["phase"] = "hybrid"
        payload["post_burnin_done"] = True
        payload["operator_labels"] = ["yo.post_burnin_selection"] * best_fit.shape[0]
        self._add_contribution(contrib, counts, "yo.post_burnin_selection", total_gain, best_fit.shape[0])
        _ = current_pos  # retained for readability of the phase transition above

    def _refresh_global_best_from_chains(self, state: EngineState) -> None:
        payload = state.payload
        best_fit = np.asarray(payload["chain_best_fitness"], dtype=float)
        best_pos = np.asarray(payload["chain_best_positions"], dtype=float)
        best_idx = self._best_index(best_fit)
        if state.best_fitness is None or self._is_better(float(best_fit[best_idx]), float(state.best_fitness)):
            state.best_fitness = float(best_fit[best_idx])
            state.best_position = best_pos[best_idx].astype(float).tolist()

    def _burnin_sweep(self, state: EngineState, contrib: dict[str, float], counts: dict[str, int]) -> int:
        payload = state.payload
        positions = np.asarray(payload["current_positions"], dtype=float).copy()
        fitness = np.asarray(payload["current_fitness"], dtype=float).copy()
        best_pos = np.asarray(payload["chain_best_positions"], dtype=float).copy()
        best_fit = np.asarray(payload["chain_best_fitness"], dtype=float).copy()
        labels = ["carryover"] * fitness.shape[0]
        evals = 0
        accepted = rejected = 0
        for i in range(fitness.shape[0]):
            if not self._budget_allows(state, used=evals):
                break
            old_fit = float(fitness[i])
            proposal = self._mcmc_propose(positions[i])
            prop_fit = float(self.problem.evaluate(proposal.copy()))
            evals += 1
            if self._metropolis_accept(prop_fit, old_fit, self._T0):
                positions[i] = proposal
                fitness[i] = prop_fit
                accepted += 1
                labels[i] = "yo.mcmc_burn_in"
                self._add_contribution(contrib, counts, "yo.mcmc_burn_in", self._positive_improvement(old_fit, prop_fit), 1)
                if self._is_better(prop_fit, float(best_fit[i])):
                    best_pos[i] = proposal
                    best_fit[i] = prop_fit
                    self._add_contribution(contrib, counts, "yo.elite_update", 0.0, 1)
            else:
                rejected += 1
                counts["yo.mcmc_burn_in"] = int(counts.get("yo.mcmc_burn_in", 0) + 1)
        payload["current_positions"] = positions
        payload["current_fitness"] = fitness
        payload["chain_best_positions"] = best_pos
        payload["chain_best_fitness"] = best_fit
        payload["population"] = self._population_matrix(best_pos, best_fit)
        payload["operator_labels"] = labels
        payload["accepted"] = accepted
        payload["rejected"] = rejected
        return int(evals)

    def _hybrid_sweep(self, state: EngineState, contrib: dict[str, float], counts: dict[str, int]) -> int:
        payload = state.payload
        positions = np.asarray(payload["current_positions"], dtype=float).copy()
        fitness = np.asarray(payload["current_fitness"], dtype=float).copy()
        best_pos = np.asarray(payload["chain_best_positions"], dtype=float).copy()
        best_fit = np.asarray(payload["chain_best_fitness"], dtype=float).copy()
        temperatures = np.asarray(payload["temperatures"], dtype=float).copy()
        stagnation = np.asarray(payload["stagnation"], dtype=int).copy()
        blacklist = set(payload.get("blacklist", set()) or set())
        labels = ["carryover"] * fitness.shape[0]
        evals = 0
        accepted = rejected = blacklisted_skips = reheats = 0

        for i in range(fitness.shape[0]):
            if not self._budget_allows(state, used=evals):
                break
            old_pos = positions[i].copy()
            old_fit = float(fitness[i])
            proposal, skips = self._propose_non_blacklisted(old_pos, blacklist)
            blacklisted_skips += int(skips)
            if self._blacklist_contains(blacklist, proposal):
                self._add_contribution(contrib, counts, "yo.blacklist_filter", 0.0, max(1, skips))
                labels[i] = "carryover"
                continue

            prop_fit = float(self.problem.evaluate(proposal.copy()))
            evals += 1
            self._add_contribution(contrib, counts, "yo.mcmc_proposal", self._positive_improvement(old_fit, prop_fit), 1)
            refined, refined_fit, greedy_evals = self._greedy_refine(
                proposal,
                prop_fit,
                state,
                evals,
                contrib,
                counts,
            )
            evals += int(greedy_evals)

            if self._metropolis_accept(refined_fit, old_fit, float(temperatures[i])):
                positions[i] = refined
                fitness[i] = refined_fit
                accepted += 1
                labels[i] = "yo.hybrid_mcmc_greedy_sa_update"
                self._add_contribution(
                    contrib,
                    counts,
                    "yo.simulated_annealing_acceptance",
                    self._positive_improvement(old_fit, refined_fit),
                    1,
                )
                if self._is_better(refined_fit, float(best_fit[i])):
                    best_pos[i] = refined
                    best_fit[i] = refined_fit
                    stagnation[i] = 0
                    self._add_contribution(contrib, counts, "yo.elite_update", 0.0, 1)
                else:
                    stagnation[i] += 1
            else:
                rejected += 1
                stagnation[i] += 1
                counts["yo.simulated_annealing_acceptance"] = int(counts.get("yo.simulated_annealing_acceptance", 0) + 1)

            if self._blacklist_enabled and self._should_blacklist(refined_fit, float(best_fit[self._best_index(best_fit)])):
                self._remember_blacklist(blacklist, refined)
                self._add_contribution(contrib, counts, "yo.blacklist_filter", 0.0, 1)

            temperatures[i] = max(self._min_temperature, float(temperatures[i]) * self._cooling)
            if int(stagnation[i]) > self._reheat_stagnation:
                temperatures[i] = max(self._min_temperature, float(temperatures[i]) * self._reheat_factor)
                stagnation[i] = 0
                reheats += 1
                self._add_contribution(contrib, counts, "yo.adaptive_reheating", 0.0, 1)

        payload["current_positions"] = positions
        payload["current_fitness"] = fitness
        payload["chain_best_positions"] = best_pos
        payload["chain_best_fitness"] = best_fit
        payload["population"] = self._population_matrix(best_pos, best_fit)
        payload["temperatures"] = temperatures
        payload["stagnation"] = stagnation
        payload["blacklist"] = blacklist
        payload["operator_labels"] = labels
        payload["accepted"] = accepted
        payload["rejected"] = rejected
        payload["blacklisted_skips"] = blacklisted_skips
        payload["reheats"] = reheats
        return int(evals)

    def step(self, state: EngineState) -> EngineState:
        contrib = self._blank_operator_contribs()
        counts = self._blank_operator_counts()
        evals = 0
        payload = state.payload

        if payload.get("phase") == "burn_in" and not self._burnin_active(state):
            self._post_burnin_select(state, contrib, counts)

        if payload.get("phase") == "burn_in" and self._burnin_active(state):
            evals += self._burnin_sweep(state, contrib, counts)
            if not self._burnin_active(replace(state, evaluations=state.evaluations + evals)):
                self._post_burnin_select(state, contrib, counts)
        else:
            evals += self._hybrid_sweep(state, contrib, counts)

        state.evaluations += int(evals)
        state.step += 1
        self._refresh_global_best_from_chains(state)
        state.payload["last_operator_contributions"] = {k: float(v) for k, v in contrib.items()}
        state.payload["last_operator_counts"] = {k: int(v) for k, v in counts.items()}
        state.payload["last_operator_metadata"] = {
            "direct_operators": list(self._DIRECT_OPERATORS),
            "diagnostic_operators": list(self._DIAGNOSTIC_OPERATORS),
            "blacklist_size": int(len(state.payload.get("blacklist", set()) or set())),
            "mean_temperature": float(np.mean(state.payload.get("temperatures", [self._T0]))),
        }
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        payload = state.payload
        population = np.asarray(payload["population"], dtype=float)
        positions = population[:, :-1]
        fitness = population[:, -1]
        denom = float(np.linalg.norm(self._span)) or 1.0
        centroid = positions.mean(axis=0)
        diversity = float(np.mean(np.linalg.norm(positions - centroid, axis=1)) / denom) if positions.size else 0.0
        accepted = int(payload.get("accepted", 0))
        rejected = int(payload.get("rejected", 0))
        total_decisions = accepted + rejected
        return {
            "step": int(state.step),
            "evaluations": int(state.evaluations),
            "best_fitness": float(state.best_fitness),
            "phase": str(payload.get("phase", "hybrid")),
            "mean_chain_best_fitness": float(np.mean(fitness)),
            "std_chain_best_fitness": float(np.std(fitness)),
            "mean_current_fitness": float(np.mean(np.asarray(payload["current_fitness"], dtype=float))),
            "mean_temperature": float(np.mean(np.asarray(payload["temperatures"], dtype=float))),
            "accepted_ratio": float(accepted / total_decisions) if total_decisions else 0.0,
            "blacklist_size": int(len(payload.get("blacklist", set()) or set())),
            "blacklisted_skips": int(payload.get("blacklisted_skips", 0)),
            "reheats": int(payload.get("reheats", 0)),
            "diversity": diversity,
            "operator_contributions": dict(payload.get("last_operator_contributions", {})),
            "operator_counts": dict(payload.get("last_operator_counts", {})),
            "evomapx_operator_metadata": dict(payload.get("last_operator_metadata", {})),
            "evomapx_delta_f": "objective_consistent_positive",
            "evomapx_fidelity": "native",
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position=list(state.best_position),
            fitness=float(state.best_fitness),
            source_algorithm=self.algorithm_id,
            source_step=int(state.step),
            role="best",
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={
                "algorithm_name": self.algorithm_name,
                "family": self.family,
                "objective": self.problem.objective,
                "elapsed_time": float(state.elapsed_time),
                "chains": int(np.asarray(state.payload.get("current_fitness", [])).shape[0]),
                "phase": str(state.payload.get("phase", "hybrid")),
                "blacklist_size": int(len(state.payload.get("blacklist", set()) or set())),
                "direct_operators": list(self._DIRECT_OPERATORS),
                "diagnostic_operators": list(self._DIAGNOSTIC_OPERATORS),
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"], dtype=float)
        return [
            CandidateRecord(
                position=pop[i, :-1].astype(float).tolist(),
                fitness=float(pop[i, -1]),
                source_algorithm=self.algorithm_id,
                source_step=int(state.step),
                role="chain_best",
            )
            for i in range(pop.shape[0])
        ]

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def _serializable_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(payload)
        if isinstance(out.get("blacklist"), set):
            out["blacklist"] = [tuple(x) for x in out["blacklist"]]
        return out

    def export_state(self, state: EngineState) -> dict[str, Any]:
        cloned = replace(
            state,
            payload=self._serializable_payload(state.payload),
            rng_state=copy.deepcopy(self._rng.bit_generator.state),
        )
        return {
            "algorithm_id": self.algorithm_id,
            "state": cloned,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def import_state(self, payload: dict[str, Any]) -> EngineState:
        state = payload["state"]
        rng_state = payload.get("rng_state", getattr(state, "rng_state", None))
        if rng_state is not None:
            self._rng.bit_generator.state = copy.deepcopy(rng_state)
        state.payload = copy.deepcopy(state.payload)
        if not isinstance(state.payload.get("blacklist"), set):
            state.payload["blacklist"] = {tuple(x) for x in state.payload.get("blacklist", [])}
        return state
