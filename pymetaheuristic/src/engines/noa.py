"""pyMetaheuristic src — Nizar Optimization Algorithm Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class NOAEngine(PortedPopulationEngine):
    """Nizar Optimization Algorithm (NOA).

    Implements NOA's effective mapping idea: transformation maps, mixing maps,
    diversity phase, and conditional overlap phase.
    """

    algorithm_id = "noa"
    algorithm_name = "Nizar Optimization Algorithm"
    family = "math"
    _REFERENCE = {"doi": "10.1007/s11227-023-05579-4"}
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=30)

    def _lambdas(self) -> np.ndarray:
        r = f"{np.random.random():.12f}".split(".", 1)[1]
        digits = [(int(ch) % 2) for ch in r[:8]]
        while len(digits) < 8:
            digits.append(np.random.randint(0, 2))
        return np.asarray(digits, dtype=int)

    @staticmethod
    def _alphas() -> tuple[float, float, float, float]:
        r1, r2, r3, r4 = np.random.random(4)
        return float(r1), float(r2 - 3.0 / 8.0), float(r3 - 1.0 / 4.0), float(r4 - 3.0 / 8.0)

    def _subset(self, dim: int) -> np.ndarray:
        if dim <= 1:
            return np.array([0], dtype=int)
        if np.random.random() < 0.5:
            size = np.random.randint(1, dim + 1)
            return np.random.choice(dim, size=size, replace=False)
        a, b = sorted(np.random.choice(dim, size=2, replace=False))
        return np.arange(a, b + 1, dtype=int)

    def _replace(self, pin: np.ndarray, ptg: np.ndarray) -> np.ndarray:
        out = pin.copy(); idx = self._subset(pin.size); out[idx] = ptg[idx]; return out

    def _scramble(self, pin: np.ndarray, ptg: np.ndarray) -> np.ndarray:
        out = pin.copy(); idx = self._subset(pin.size); vals = ptg[idx].copy(); np.random.shuffle(vals); out[idx] = vals; return out

    def _distribute(self, pin: np.ndarray, ptg: np.ndarray) -> np.ndarray:
        out = pin.copy(); idx = self._subset(pin.size); out[idx] = ptg[np.random.randint(ptg.size)]; return out

    @staticmethod
    def _phi_translate(x: np.ndarray, r, alpha: float) -> np.ndarray:
        if alpha <= 0.5:
            return x.copy()
        return np.round(x + np.asarray(r) ** 2)

    @staticmethod
    def _phi_dilate(x: np.ndarray, r, alpha: float) -> np.ndarray:
        if alpha <= 0.5:
            return x.copy()
        return np.round(x * (np.asarray(r) ** 2))

    @staticmethod
    def _phi_transfer(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        return x.copy() if alpha <= 0.5 else y.copy()

    def _bound_return(self, trial: np.ndarray, original: np.ndarray) -> np.ndarray:
        out = trial.copy()
        mask = (out < self._lo) | (out > self._hi)
        out[mask] = original[mask]
        return np.clip(out, self._lo, self._hi)

    def _make_p1(self, xm: np.ndarray, best: np.ndarray, beta1: float, lam: np.ndarray) -> np.ndarray:
        if lam[2] == 1:
            return xm.copy()
        if lam[5] == 1:
            return 0.5 * (best + xm)
        return xm * beta1 + (1.0 - beta1) * best

    def _make_points(self, xi: np.ndarray, xm: np.ndarray, best: np.ndarray, alpha: tuple[float, float, float, float], lam: np.ndarray, beta1, beta2):
        a1, a2, a3, a4 = alpha
        t3 = best.copy() if lam[7] == 1 else xi.copy()
        p1 = self._make_p1(xm, best, beta1 if np.ndim(beta1) == 0 else np.mean(beta1), lam)
        if lam[6] == 1:
            t2 = self._phi_translate(self._replace(t3, p1), beta2, a4)
        else:
            t2 = self._phi_translate(self._scramble(t3, p1), beta2, a4)
        if lam[3] == 1:
            p2 = xm + beta1 * np.ones_like(xm)
        else:
            p2 = t2
        if lam[4] == 1:
            p3_inner = self._phi_transfer(self._distribute(t3, p1), t2, a3)
        else:
            p3_inner = self._phi_transfer(self._distribute(p1, t3), t2, a3)
        p3 = self._phi_dilate(p3_inner, beta2, a2)
        return p1, p2, p3, t3

    def _step_impl(self, state, pop: np.ndarray):
        n, dim = pop.shape[0], self.problem.dimension
        best = np.asarray(state.best_position, dtype=float)
        alpha = self._alphas()
        lam = self._lambdas()
        trials = np.empty((n, dim))

        for i in range(n):
            ids = self._rand_indices(n, i, 3)
            j, k, m = int(ids[0]), int(ids[1]), int(ids[2])
            xi, xj, xk, xm = pop[i, :-1], pop[j, :-1], pop[k, :-1], pop[m, :-1]
            beta1, beta2 = np.random.random(), np.random.random()
            p1, p2, p3, t3 = self._make_points(xi, xm, best, alpha, lam, beta1, beta2)

            if lam[0] == 1:
                if lam[1] == 1:
                    trial = p1 + beta1 * (xi - xj) + beta2 * (xi - xk)
                else:
                    trial = xi + beta1 * (p1 - xj) - beta2 * (p1 - xk)
            else:
                vj = self._phi_transfer(xj, t3, alpha[0])
                vk = self._phi_transfer(xk, t3, alpha[0])
                dj = beta1 * ((-1.0) ** j)
                dk = beta2 * ((-1.0) ** k)
                if lam[1] == 1:
                    trial = p2 + dj * (p3 - vj) + dk * (p3 - vk)
                else:
                    trial = p3 + dj * (p2 - vj) - dk * (p2 - vk)

            same_trial = np.allclose(trial, xi)
            same_best = np.allclose(xi, best)
            if same_trial or same_best or np.random.random() <= 0.25:
                b1 = np.random.uniform(-1.0, 1.0, dim)
                b2 = np.random.uniform(-1.0, 1.0, dim)
                p1_overlap = self._make_p1(xm, best, float(np.mean(b1)), lam)
                trial = p1_overlap + b1 * (xi - xj) + b2 * (xi - xk)

            trials[i] = self._bound_return(trial, xi)

        fit = self._evaluate_population(trials)
        mask = self._better_mask(fit, pop[:, -1])
        pop[mask, :-1] = trials[mask]
        pop[mask, -1] = fit[mask]
        return pop, n, {}
