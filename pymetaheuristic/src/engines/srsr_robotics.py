"""pyMetaheuristic src — Swarm Robotics Search and Rescue Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class SRSRRoboticsEngine(PortedPopulationEngine):
    """Swarm Robotics Search And Rescue.

    The registry id is ``srsr_robotics`` to avoid colliding with the existing
    ``srsr`` engine, which implements the Shuffle-based Runner-Root Algorithm.
    """

    algorithm_id = "srsr_robotics"
    algorithm_name = "Swarm Robotics Search And Rescue"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1016/j.asoc.2017.02.028",
        "title": "Swarm robotics search & rescue: A novel artificial intelligence-inspired optimization approach",
        "authors": "M. Bakhshipour, M. J. Ghadi, F. Namdari",
        "year": 2017,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, mu_factor=2.0 / 3.0)

    def __init__(self, problem, config) -> None:
        super().__init__(problem, config)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self._n < 5:
            raise ValueError("srsr_robotics requires population_size >= 5.")
        mu_factor = float(self._params.get("mu_factor", 2.0 / 3.0))
        if not 0.1 <= mu_factor <= 0.9:
            raise ValueError("srsr_robotics mu_factor must be in [0.1, 0.9].")

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        return {"SIF": 2.0, "sigma_temp": np.zeros(pop.shape[0], dtype=float)}

    def _trial_fit(self, pos: np.ndarray) -> tuple[np.ndarray, float]:
        candidate = np.clip(np.asarray(pos, dtype=float), self._lo, self._hi)
        return candidate, float(self.problem.evaluate(candidate))

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        mu_factor = float(self._params.get("mu_factor", 2.0 / 3.0))
        SIF = float(state.payload.get("SIF", 2.0))
        sigma_temp = np.asarray(state.payload.get("sigma_temp", np.zeros(n)), dtype=float).copy()
        if sigma_temp.shape != (n,):
            sigma_temp = np.zeros(n, dtype=float)
        evals = 0

        # The source implementation sorts the population before each step.
        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        master = pop[0, :-1].copy()
        movement_factor = self._span

        # Phase 1: accumulation. Slave robots sample from Gaussian models whose
        # mean is pulled by the master robot.
        sigma_master = np.random.uniform()
        if int(state.step) % 2 == 1:
            mu_master = (1.0 - sigma_master) * master
        else:
            mu_master = (1.0 + (1.0 - mu_factor) * sigma_master) * master
        pop[0, :-1] = np.clip(mu_master, self._lo, self._hi)
        pop[0, -1] = float(self.problem.evaluate(pop[0, :-1]))
        evals += 1
        master = pop[0, :-1].copy()

        if state.step == 0:
            SIF = 6.0
        phase1 = np.empty((n, dim), dtype=float)
        for i in range(n):
            mu_i = mu_factor * master + (1.0 - mu_factor) * pop[i, :-1]
            sigma_temp[i] = SIF * np.random.uniform()
            close_bonus = (np.random.uniform() ** 2) * ((master - pop[i, :-1]) < 0.05)
            sigma_i = sigma_temp[i] * np.abs(master - pop[i, :-1]) + close_bonus
            phase1[i] = np.clip(np.random.normal(mu_i, np.abs(sigma_i) + 1.0e-12, dim), self._lo, self._hi)

        phase1_fit = self._evaluate_population(phase1)
        evals += n
        target_move = pop[:, -1] - phase1_fit if self.problem.objective == "min" else phase1_fit - pop[:, -1]
        improved = self._better_mask(phase1_fit, pop[:, -1])
        pop[improved, :-1] = phase1[improved]
        pop[improved, -1] = phase1_fit[improved]

        fit_id = int(np.argmax(target_move))
        sigma_factor = 1.0 + np.random.uniform() * float(np.max(self._span))
        SIF = sigma_factor * float(sigma_temp[fit_id])
        hi_limit = float(np.max(np.abs(self._hi)))
        if hi_limit > 0.0 and SIF > hi_limit:
            SIF = hi_limit * np.random.uniform()

        # Phase 2: exploration. Robots move toward the current master with a
        # random signed group vector plus a large movement factor.
        order = self._order(pop[:, -1])
        pop = pop[order].copy()
        master = pop[0, :-1].copy()
        phase2 = np.empty((n, dim), dtype=float)
        for i in range(n):
            gb = np.random.uniform(-1.0, 1.0, dim)
            gb = np.where(gb >= 0.0, 1.0, -1.0)
            random_drive = movement_factor * np.random.uniform(self._lo, self._hi)
            candidate = pop[i, :-1] * np.random.uniform() + gb * (master - pop[i, :-1]) + random_drive
            phase2[i] = np.clip(candidate, self._lo, self._hi)

        phase2_fit = self._evaluate_population(phase2)
        evals += n
        improved = self._better_mask(phase2_fit, pop[:, -1])
        pop[improved, :-1] = phase2[improved]
        pop[improved, -1] = phase2_fit[improved]

        # Phase 3: local worker robots around the master, using integer/fraction
        # operators from the source implementation.
        if state.step > 0:
            order = self._order(pop[:, -1])
            pop = pop[order].copy()
            master = pop[0, :-1].copy()
            sign = np.sign(master)
            abs_master = np.abs(master)
            int_part = np.floor(abs_master)
            frac = abs_master - int_part
            workers = []
            p_root = 1.0 / (1.0 + np.random.randint(1, 4))
            p_exp = 1.0 + np.random.randint(1, 4)
            workers.append((int_part + np.power(frac, p_root)) * sign)
            workers.append((int_part + np.power(frac, p_exp)) * sign)
            perm = np.random.permutation(dim)
            split = dim // 2
            worker3 = np.zeros(dim, dtype=float)
            sec1, sec2 = perm[:split], perm[split:]
            worker3[sec1] = (int_part[sec1] + np.power(frac[sec1], p_root)) * sign[sec1]
            worker3[sec2] = (int_part[sec2] + np.power(frac[sec2], p_exp)) * sign[sec2]
            workers.append(worker3)
            workers.append(np.ceil(abs_master) * sign)
            workers.append(np.floor(abs_master) * sign)

            replace_start = max(1, n - len(workers))
            for k, worker in enumerate(workers):
                if k >= n:
                    break
                mask = np.round(np.random.uniform(self._lo, self._hi, dim)).astype(bool)
                candidate = np.asarray(worker, dtype=float).copy()
                candidate[mask] = master[mask]
                candidate, fit = self._trial_fit(candidate)
                evals += 1
                target_idx = replace_start + k
                if target_idx < n and self._is_better(fit, pop[target_idx, -1]):
                    pop[target_idx, :-1] = candidate
                    pop[target_idx, -1] = fit

        return pop, evals, {"SIF": SIF, "sigma_temp": sigma_temp}
