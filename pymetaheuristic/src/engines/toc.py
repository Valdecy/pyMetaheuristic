"""pyMetaheuristic src — Tornado Optimizer with Coriolis Force Engine.

This implementation follows the main population mechanics of Braik et al. (2025):
windstorms, thunderstorms and tornadoes; fitness-proportional windstorm assignment;
Coriolis-force velocity for windstorms that evolve directly into tornadoes; windstorm to
thunderstorm evolution; thunderstorm to tornado evolution; and random windstorm
formation near mature systems.
"""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class TOCEngine(PortedPopulationEngine):
    algorithm_id = "toc"
    algorithm_name = "Tornado Optimizer with Coriolis Force"
    family = "physics"
    _REFERENCE = {"doi": "10.1007/s10462-025-11118-9"}
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        supports_snapshot_fit=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        tornado_count=1,
        thunderstorm_count=None,
        thunderstorm_fraction=0.20,
        acceleration_rate=4.10,
        earth_omega=0.7292115e-4,
        br=100000.0,
        wmin=1.0,
        wmax=4.0,
        a0=2.0,
        random_formation=True,
        max_iterations=1000,
    )

    _DIRECT_OPERATORS = (
        "toc.windstorm_to_tornado_evolution",
        "toc.windstorm_to_thunderstorm_evolution",
        "toc.thunderstorm_to_tornado_evolution",
        "toc.random_windstorm_formation",
    )
    _DIAGNOSTIC_OPERATORS = (
        "toc.fitness_proportional_assignment",
        "toc.coriolis_velocity_update",
        "toc.role_exchange_replacement",
    )
    _ALL_OPERATORS = _DIRECT_OPERATORS + _DIAGNOSTIC_OPERATORS

    def _initialize_payload(self, pop):
        return {
            "V": np.zeros_like(pop[:, :-1], dtype=float),
            "toc_roles": self._role_counts(pop.shape[0]),
            "last_operator_contributions": self._blank_operator_contribs(),
            "last_operator_counts": self._blank_operator_counts(),
            "last_operator_metadata": {},
        }


    def _post_injection_repair(self, state, replaced_indices, candidates) -> None:
        super()._post_injection_repair(state, replaced_indices, candidates)
        V = state.payload.get("V")
        if V is None or not replaced_indices:
            return
        V = np.asarray(V, dtype=float)
        for idx in replaced_indices:
            if 0 <= int(idx) < V.shape[0]:
                V[int(idx), :] = 0.0
        state.payload["V"] = V

    def _blank_operator_contribs(self) -> dict[str, float]:
        return {label: 0.0 for label in self._ALL_OPERATORS}

    def _blank_operator_counts(self) -> dict[str, int]:
        return {label: 0 for label in self._ALL_OPERATORS}

    def _add_contribution(
        self,
        contributions: dict[str, float],
        counts: dict[str, int],
        label: str,
        gain: float = 0.0,
        count: int = 1,
    ) -> None:
        contributions[label] = float(contributions.get(label, 0.0) + max(0.0, float(gain)))
        counts[label] = int(counts.get(label, 0) + int(count))

    def _fitness_gain(self, old_fit: float, new_fit: float) -> float:
        if self.problem.objective == "min":
            return max(0.0, float(old_fit) - float(new_fit))
        return max(0.0, float(new_fit) - float(old_fit))

    def _max_iterations(self) -> int:
        if "max_iterations" in getattr(self.config, "params", {}):
            value = self.config.params["max_iterations"]
        elif self.config.max_steps is not None:
            value = self.config.max_steps
        else:
            value = self._params.get("max_iterations", 1000)
        return max(1, int(value))

    def _role_counts(self, n: int) -> dict[str, int]:
        n = int(n)
        no = max(1, int(self._params.get("tornado_count", 1)))
        no = min(no, max(1, n - 1)) if n > 1 else 1

        requested_nt = self._params.get("thunderstorm_count", None)
        if requested_nt is None:
            requested_nt = int(round(float(self._params.get("thunderstorm_fraction", 0.20)) * n))
        nt = max(0, int(requested_nt))
        nt = min(nt, max(0, n - no - 1))

        # Keep at least one thunderstorm when the population can support the
        # paper's tornado/thunderstorm/windstorm hierarchy.
        if n - no >= 2 and nt == 0:
            nt = 1

        nw = max(0, n - no - nt)
        return {"tornadoes": no, "thunderstorms": nt, "windstorms": nw, "nto": no + nt}

    def _prepare_position(self, x: np.ndarray) -> np.ndarray:
        return self.problem.apply_variable_types(np.clip(np.asarray(x, dtype=float), self._lo, self._hi)).astype(float)

    def _evaluate_position(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        trial = self._prepare_position(x)
        fit = float(self.problem.evaluate(trial))
        # ProblemSpec.evaluate may repair in place. Re-clip defensively.
        trial = self._prepare_position(trial)
        return trial, fit

    def _accept_if_better(self, pop: np.ndarray, idx: int, position: np.ndarray, fitness: float) -> tuple[bool, float]:
        old_fit = float(pop[idx, -1])
        if self._is_better(fitness, old_fit):
            pop[idx, :-1] = position
            pop[idx, -1] = fitness
            return True, self._fitness_gain(old_fit, fitness)
        return False, 0.0

    def _swap_if_better(
        self,
        pop: np.ndarray,
        V: np.ndarray,
        challenger_idx: int,
        controller_idx: int,
        contributions: dict[str, float],
        counts: dict[str, int],
    ) -> bool:
        if challenger_idx == controller_idx:
            return False
        if self._is_better(pop[challenger_idx, -1], pop[controller_idx, -1]):
            pop[[controller_idx, challenger_idx], :] = pop[[challenger_idx, controller_idx], :]
            V[[controller_idx, challenger_idx], :] = V[[challenger_idx, controller_idx], :]
            self._add_contribution(contributions, counts, "toc.role_exchange_replacement", 0.0, 1)
            return True
        return False

    def _assignment_counts(self, controller_fit: np.ndarray, reference_fit: float, nw: int) -> np.ndarray:
        nto = int(controller_fit.size)
        if nto == 0 or nw <= 0:
            return np.zeros(nto, dtype=int)

        # Eq. 28: f_k = fit_k - fit_{nto+1}; Eq. 29 uses absolute/proportional mass.
        weights = np.abs(np.asarray(controller_fit, dtype=float) - float(reference_fit))
        if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 1.0e-300:
            # Tie-safe fallback: ranked controllers receive decreasing mass, so the
            # tornado still attracts at least as many windstorms as weaker systems.
            weights = np.arange(nto, 0, -1, dtype=float)
        raw = weights / (float(np.sum(weights)) + 1.0e-300) * int(nw)
        counts = np.floor(raw).astype(int)
        remainder = int(nw - np.sum(counts))
        if remainder > 0:
            frac_order = np.argsort(-(raw - counts))
            for j in frac_order[:remainder]:
                counts[int(j)] += 1
        elif remainder < 0:
            for j in np.argsort(raw - counts):
                if remainder == 0:
                    break
                if counts[int(j)] > 0:
                    counts[int(j)] -= 1
                    remainder += 1
        return counts

    def _ay(self, t: int, T: int) -> float:
        a0 = float(self._params.get("a0", 2.0))
        return float(np.clip((T - (float(t) ** a0) / T) / T, 0.0, 1.0))

    def _alpha(self, ay: float) -> float:
        return float(abs(2.0 * ay * np.random.rand() - np.random.rand()))

    def _nu(self, t: int, T: int) -> float:
        tau = np.clip(float(t) / max(float(T), 1.0), 0.0, 1.0)
        # Eq. 56, simplified algebraically: 0.1 * exp(-((0.1*(t/T)) / 0.1)^16).
        return float(0.1 * math.exp(-(tau ** 16)))

    def _random_sign(self) -> float:
        return 1.0 if np.random.rand() >= 0.5 else -1.0

    def _random_formation(self, x: np.ndarray, ay: float) -> np.ndarray:
        delta2 = self._random_sign()
        perturb = np.random.rand(self.problem.dimension) * (self._lo - self._hi) - self._lo
        return self._prepare_position(x - (2.0 * ay * perturb) * delta2)

    def _coriolis_velocity(self, current_v: np.ndarray, wind_pos: np.ndarray, tornado_pos: np.ndarray, t: int, T: int) -> np.ndarray:
        chi = float(self._params.get("acceleration_rate", 4.10))
        eta = 2.0 / abs(2.0 - chi - math.sqrt(abs(chi * chi - 4.0 * chi)))
        mu = 0.5 + 0.5 * np.random.rand()

        exp_arg = np.clip((-float(t) + float(T) / 2.0) / 2.0, -60.0, 60.0)
        Rl = 2.0 / (1.0 + math.exp(exp_arg))
        Rr = -Rl
        northern = bool(np.random.rand() >= 0.5)
        R = Rl if northern else Rr

        br = float(self._params.get("br", 100000.0))
        wmin = float(self._params.get("wmin", 1.0))
        wmax = float(self._params.get("wmax", 4.0))
        r1, r2, r3, r4 = np.random.rand(4)
        wr = (2.0 * r1 - (r2 + r3)) / (wmin + r4 * (wmax - wmin) + 1.0e-300)
        c = br * self._random_sign() * wr

        omega = float(self._params.get("earth_omega", 0.7292115e-4))
        f = 2.0 * omega * math.sin(-1.0 + 2.0 * np.random.rand())

        phi = np.asarray(tornado_pos - wind_pos, dtype=float)
        if northern:
            phi = np.where(phi >= 0.0, -phi, phi)
        else:
            phi = np.where(phi <= 0.0, -phi, phi)

        cf = (f * f * R * R) / 4.0 - R * phi
        cf = np.where(cf < 0.0, -cf, cf)
        return eta * (mu * current_v - c * (f * R) / 2.0 + np.sqrt(cf + 1.0e-300))

    def _step_impl(self, state, pop):
        n, d = pop.shape[0], self.problem.dimension
        evals = 0
        T = self._max_iterations()
        t = min(max(int(state.step) + 1, 1), T)
        ay = self._ay(t, T)
        nu = self._nu(t, T)
        V = np.asarray(state.payload.get("V", np.zeros((n, d))), dtype=float)
        if V.shape != (n, d):
            V = np.zeros((n, d), dtype=float)

        contributions = self._blank_operator_contribs()
        counts = self._blank_operator_counts()

        order = self._order(pop[:, -1])
        roles = self._role_counts(n)
        no, nt, nto, nw = roles["tornadoes"], roles["thunderstorms"], roles["nto"], roles["windstorms"]
        if nw <= 0:
            return pop, 0, {
                "V": V,
                "toc_roles": roles,
                "last_operator_contributions": contributions,
                "last_operator_counts": counts,
                "last_operator_metadata": self._operator_metadata(roles, 0, 0, 0, False),
            }

        tornado_idx = order[:no]
        thunder_idx = order[no:nto]
        wind_idx = order[nto:]
        if wind_idx.size == 0:
            return pop, 0, {
                "V": V,
                "toc_roles": roles,
                "last_operator_contributions": contributions,
                "last_operator_counts": counts,
                "last_operator_metadata": self._operator_metadata(roles, 0, 0, 0, False),
            }

        counts_by_controller = self._assignment_counts(pop[order[:nto], -1], pop[order[nto], -1], wind_idx.size)
        self._add_contribution(contributions, counts, "toc.fitness_proportional_assignment", 0.0, 1)
        shuffled_wind = np.array(wind_idx, dtype=int).copy()
        np.random.shuffle(shuffled_wind)

        groups: list[tuple[int, np.ndarray]] = []
        cursor = 0
        for controller_slot, count in enumerate(counts_by_controller):
            count = int(count)
            groups.append((int(order[controller_slot]), shuffled_wind[cursor: cursor + count]))
            cursor += count

        random_formation = bool(self._params.get("random_formation", True))
        accepted = 0
        random_formation_uses = 0

        # Windstorm evolution: directly into tornadoes (Eq. 46 with Eq. 30) or into
        # thunderstorms (Eq. 51). Role exchange is applied immediately when a
        # windstorm becomes better than its controller.
        for controller_slot, (controller, members) in enumerate(groups):
            if members.size == 0:
                continue
            controller_is_tornado = controller_slot < no
            for idx in members:
                idx = int(idx)
                tornado = int(np.random.choice(tornado_idx))
                used_random_formation = False
                if controller_is_tornado:
                    rand_wind = pop[int(np.random.choice(wind_idx)), :-1]
                    V[idx, :] = self._coriolis_velocity(V[idx, :], pop[idx, :-1], pop[tornado, :-1], t, T)
                    self._add_contribution(contributions, counts, "toc.coriolis_velocity_update", 0.0, 1)
                    trial = pop[idx, :-1] + 2.0 * self._alpha(ay) * (pop[controller, :-1] - rand_wind) + V[idx, :]
                    nominal_operator = "toc.windstorm_to_tornado_evolution"
                    if random_formation and np.linalg.norm(pop[idx, :-1] - pop[controller, :-1]) < nu:
                        trial = self._random_formation(pop[idx, :-1], ay)
                        used_random_formation = True
                else:
                    r1 = np.random.rand(d)
                    r2 = np.random.rand(d)
                    trial = (
                        pop[idx, :-1]
                        + 2.0 * r1 * (pop[controller, :-1] - pop[idx, :-1])
                        + 2.0 * r2 * (pop[tornado, :-1] - pop[idx, :-1])
                    )
                    nominal_operator = "toc.windstorm_to_thunderstorm_evolution"
                    if random_formation and np.linalg.norm(pop[idx, :-1] - pop[controller, :-1]) < nu:
                        trial = self._random_formation(pop[idx, :-1], ay)
                        used_random_formation = True

                trial, fit = self._evaluate_position(trial)
                evals += 1
                ok, gain = self._accept_if_better(pop, idx, trial, fit)
                accepted += int(ok)
                if used_random_formation:
                    random_formation_uses += 1
                    self._add_contribution(contributions, counts, "toc.random_windstorm_formation", gain if ok else 0.0, 1)
                else:
                    self._add_contribution(contributions, counts, nominal_operator, gain if ok else 0.0, 1)
                self._swap_if_better(pop, V, idx, controller, contributions, counts)

        # Thunderstorm evolution into tornadoes (Eq. 52), with the same exchange
        # principle when a thunderstorm improves over a tornado.
        if nt > 0 and thunder_idx.size > 0:
            for idx in thunder_idx:
                idx = int(idx)
                tornado = int(np.random.choice(tornado_idx))
                peer = int(np.random.choice(thunder_idx))
                alpha = self._alpha(ay)
                trial = (
                    pop[idx, :-1]
                    + 2.0 * alpha * (pop[idx, :-1] - pop[tornado, :-1])
                    + 2.0 * alpha * (pop[peer, :-1] - pop[idx, :-1])
                )
                used_random_formation = False
                if random_formation and np.linalg.norm(pop[idx, :-1] - pop[tornado, :-1]) < nu:
                    trial = self._random_formation(pop[idx, :-1], ay)
                    used_random_formation = True
                trial, fit = self._evaluate_position(trial)
                evals += 1
                ok, gain = self._accept_if_better(pop, idx, trial, fit)
                accepted += int(ok)
                if used_random_formation:
                    random_formation_uses += 1
                    self._add_contribution(contributions, counts, "toc.random_windstorm_formation", gain if ok else 0.0, 1)
                else:
                    self._add_contribution(contributions, counts, "toc.thunderstorm_to_tornado_evolution", gain if ok else 0.0, 1)
                self._swap_if_better(pop, V, idx, tornado, contributions, counts)

        updates = {
            "V": V,
            "toc_roles": roles,
            "last_operator_contributions": {k: float(v) for k, v in contributions.items()},
            "last_operator_counts": {k: int(v) for k, v in counts.items()},
            "last_operator_metadata": self._operator_metadata(roles, evals, accepted, random_formation_uses, random_formation),
        }
        return pop, evals, updates

    def _operator_metadata(
        self,
        roles: dict[str, int],
        evals: int,
        accepted: int,
        random_formation_uses: int,
        random_formation_enabled: bool,
    ) -> dict:
        return {
            "direct_operators": list(self._DIRECT_OPERATORS),
            "diagnostic_operators": list(self._DIAGNOSTIC_OPERATORS),
            "roles": dict(roles),
            "candidate_evaluations": int(evals),
            "accepted_moves": int(accepted),
            "random_formation_enabled": bool(random_formation_enabled),
            "random_formation_uses": int(random_formation_uses),
        }

    def observe(self, state):
        obs = super().observe(state)
        payload = state.payload
        obs["toc_roles"] = dict(payload.get("toc_roles", {}))
        obs["operator_contributions"] = dict(payload.get("last_operator_contributions", self._blank_operator_contribs()))
        obs["operator_counts"] = dict(payload.get("last_operator_counts", self._blank_operator_counts()))
        obs["evomapx_operator_metadata"] = dict(payload.get("last_operator_metadata", {}))
        obs["evomapx_delta_f"] = "objective_consistent_positive"
        obs["evomapx_fidelity"] = "native"
        return obs
