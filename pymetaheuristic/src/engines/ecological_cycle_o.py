"""pyMetaheuristic src — Ecological Cycle Optimizer Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class EcologicalCycleOEngine(PortedPopulationEngine):
    """Ecological Cycle Optimizer (ECO).

    Uses producer, herbivore, carnivore, omnivore, and decomposer update stages.
    The public ID avoids the existing ``eco`` ID, which is already assigned to
    Educational Competition Optimizer in PMH.
    """

    algorithm_id = "ecological_cycle_o"
    algorithm_name = "Ecological Cycle Optimizer"
    family = "swarm"
    _REFERENCE     = {"doi": "10.48550/arXiv.2508.20458"}
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def _group_counts(self, n: int) -> tuple[int, int, int, int]:
        n_pro = max(1, int(round(0.20 * n)))
        n_her = max(1, int(round(0.30 * n)))
        n_car = max(1, int(round(0.30 * n)))
        n_omn = max(1, n - n_pro - n_her - n_car)
        while n_pro + n_her + n_car + n_omn > n:
            if n_omn > 1:
                n_omn -= 1
            elif n_car > 1:
                n_car -= 1
            elif n_her > 1:
                n_her -= 1
            else:
                n_pro -= 1
        while n_pro + n_her + n_car + n_omn < n:
            n_omn += 1
        return n_pro, n_her, n_car, n_omn

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n_pro, n_her, n_car, n_omn = self._group_counts(pop.shape[0])
        return {"decomposers": pop[:, :-1].copy(), "group_counts": (n_pro, n_her, n_car, n_omn)}

    def _roulette(self, group: np.ndarray, count: int) -> np.ndarray:
        if group.shape[0] == 0:
            return np.empty((0, self.problem.dimension))
        # Lower fitness receives higher probability for minimization; reversed for maximization.
        q = self._quality(group[:, -1]) + 1e-12
        p = q / q.sum()
        idx = np.random.choice(group.shape[0], size=count, replace=True, p=p)
        return group[idx, :-1]

    def _predation_factor(self, t: int, max_iter: int) -> np.ndarray:
        sign = np.where(np.random.randint(1, 3, size=self.problem.dimension) == 1, -1.0, 1.0)
        return 1.0 + 2.0 * np.random.random(self.problem.dimension) * np.exp(-9.0 * (t / max_iter) ** 3) * sign

    def _eval_accept_group(self, old: np.ndarray, new_pos: np.ndarray) -> tuple[np.ndarray, int]:
        new_pos = np.clip(new_pos, self._lo, self._hi)
        fit = self._evaluate_population(new_pos)
        out = old.copy()
        mask = self._better_mask(fit, old[:, -1])
        out[mask, :-1] = new_pos[mask]
        out[mask, -1] = fit[mask]
        return out, new_pos.shape[0]

    def _step_impl(self, state, pop: np.ndarray):
        n, d = pop.shape[0], self.problem.dimension
        t = state.step + 1
        max_iter = max(1, int(self._params.get("max_iterations", self.config.max_steps or 1000)))
        n_pro, n_her, n_car, n_omn = state.payload.get("group_counts", self._group_counts(n))
        evals = 0
        dec_prev = np.asarray(state.payload.get("decomposers", pop[:, :-1]), dtype=float)
        dec_prev_pop = self._pop_from_positions(dec_prev)
        evals += dec_prev_pop.shape[0]

        ordered = pop[self._order(pop[:, -1])]
        pro = ordered[:n_pro].copy()
        her = ordered[n_pro:n_pro + n_her].copy()
        car = ordered[n_pro + n_her:n_pro + n_her + n_car].copy()
        omn = ordered[n_pro + n_her + n_car:n_pro + n_her + n_car + n_omn].copy()

        if state.step > 0:
            nutrients = np.vstack([pro, dec_prev_pop])
            pro = nutrients[self._order(nutrients[:, -1])][:n_pro].copy()

        G = self._predation_factor(t, max_iter)

        if her.shape[0] > 0:
            her_new = her[:, :-1].copy()
            for i in range(her.shape[0]):
                prey = self._roulette(pro, 3)
                if prey.shape[0]:
                    move = sum(np.random.random(d) * (prey_row - her[i, :-1]) for prey_row in prey)
                    her_new[i] = her[i, :-1] + G * move
            her, e = self._eval_accept_group(her, her_new); evals += e

        if car.shape[0] > 0:
            car_new = car[:, :-1].copy()
            for i in range(car.shape[0]):
                prey = self._roulette(her, 3)
                if prey.shape[0]:
                    move = sum(np.random.random(d) * (prey_row - car[i, :-1]) for prey_row in prey)
                    car_new[i] = car[i, :-1] + G * move
            car, e = self._eval_accept_group(car, car_new); evals += e

        if omn.shape[0] > 0:
            omn_new = omn[:, :-1].copy()
            for i in range(omn.shape[0]):
                prey_parts = []
                if pro.shape[0]: prey_parts.append(self._roulette(pro, 1)[0])
                if her.shape[0]: prey_parts.append(self._roulette(her, 1)[0])
                if car.shape[0]: prey_parts.extend(self._roulette(car, 2))
                if prey_parts:
                    move = sum(np.random.random(d) * (prey_row - omn[i, :-1]) for prey_row in prey_parts)
                    omn_new[i] = omn[i, :-1] + G * move
            omn, e = self._eval_accept_group(omn, omn_new); evals += e

        current = np.vstack([pro, her, car, omn])[:n]
        best_idx = self._best_index(current[:, -1])
        best = current[best_idx, :-1].copy()
        span_min = float(np.min(np.abs(self._hi - self._lo))) or 1.0
        H = np.cos(np.random.random() * np.pi) * (1.0 - t / (1.5 * max_iter)) ** (5.0 * t / max_iter)
        decomposers = np.empty((n, d))
        for i in range(n):
            xi = current[i, :-1]
            if np.random.random() < 0.5:
                nei = np.random.random(d) * best
                decomposers[i] = nei + (0.4 * np.random.random(d) - 0.2) * (nei - xi)
            elif np.random.random() < 0.5:
                vrand = 2.0 * np.random.random(d) - 1.0
                norm = np.linalg.norm(vrand) or 1.0
                radius = np.linalg.norm(best - xi)
                decomposers[i] = xi + np.random.random() * radius * (vrand / norm)
            else:
                wrand = (2.0 / 3.0) * np.random.random(d) * H * span_min
                w = np.random.random()
                decomposers[i] = w * xi + (1.0 - w) * wrand

        decomposers = np.clip(decomposers, self._lo, self._hi)
        dec_fit = self._evaluate_population(decomposers); evals += n
        dec_pop = np.hstack([decomposers, dec_fit[:, None]])
        combined = np.vstack([current, dec_pop])
        pop = combined[self._order(combined[:, -1])][:n].copy()
        return pop, evals, {"decomposers": decomposers, "group_counts": (n_pro, n_her, n_car, n_omn)}
