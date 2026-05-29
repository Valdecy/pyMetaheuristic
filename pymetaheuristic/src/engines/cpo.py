"""pyMetaheuristic src — Chinese Pangolin Optimizer Engine"""
from __future__ import annotations

import math
import numpy as np

from .protocol import CapabilityProfile, EngineConfig, ProblemSpec
from ._ported_common import PortedPopulationEngine


class CPOEngine(PortedPopulationEngine):
    """Chinese Pangolin Optimizer.

    Implements the paper's aroma-concentration switch between luring and
    predation behaviors, including the attraction/capture, movement/feeding,
    search/localization, rapid-approach, and digging/feeding stages.
    """

    algorithm_id = "cpo"
    algorithm_name = "Chinese Pangolin Optimizer"
    family = "swarm"
    _REFERENCE = {
        "doi": "10.1007/s11227-025-07004-4",
        "title": "Chinese Pangolin Optimizer: a novel bio-inspired metaheuristic for solving optimization problems",
        "year": 2025,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=False,
        supports_restart=False,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(
        population_size=30,
        diffusion_coefficient=0.6,
        levy_beta=1.5,
        levy_scale=0.01,
        aroma_source_strength=100.0,
        aroma_factor_cap=3.0,
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        if self._n < 2:
            raise ValueError("population_size must be at least 2 for Chinese Pangolin Optimizer.")
        if float(self._params.get("diffusion_coefficient", 0.6)) <= 0.0:
            raise ValueError("diffusion_coefficient must be positive.")
        beta = float(self._params.get("levy_beta", 1.5))
        if not (0.0 < beta <= 2.0):
            raise ValueError("levy_beta must be in (0, 2].")
        if float(self._params.get("levy_scale", 0.01)) <= 0.0:
            raise ValueError("levy_scale must be positive.")
        if float(self._params.get("aroma_source_strength", 100.0)) <= 0.0:
            raise ValueError("aroma_source_strength must be positive.")
        if float(self._params.get("aroma_factor_cap", 3.0)) <= 0.0:
            raise ValueError("aroma_factor_cap must be positive.")

    def _max_iter(self) -> int:
        return max(1, int(self._params.get("max_iterations", self.config.max_steps or 1000)))

    def _aroma_raw(self, t: int, max_iter: int, u: float, h: float) -> float:
        # Eqs. (9), (12), and (13), with safeguards near singular values.
        t = max(1, int(t))
        ratio = float(t) / float(max_iter)
        sigma_y = max(1.0e-9, 50.0 - 10.0 * ratio)
        sigma_z = math.sin(math.pi * ratio) + 40.0 * math.exp(-ratio) - 10.0 * math.log(max(math.pi * ratio, 1.0e-12))
        sigma_z = max(1.0e-9, sigma_z)
        q = float(self._params.get("aroma_source_strength", 100.0))
        return float((q / (math.pi * max(u, 1.0e-12) * sigma_y * sigma_z)) * math.exp(-(h * h) / (2.0 * sigma_z * sigma_z)))

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        max_iter = self._max_iter()
        # Eq. (14) requires min(M) and max(M). Precompute the aroma sequence
        # for the planned horizon using Eqs. (10)–(13).
        raw = []
        for t in range(1, max_iter + 1):
            u = 2.0 + np.random.rand()      # Eq. (10)
            h = np.random.rand() / 2.0      # Eq. (11)
            raw.append(self._aroma_raw(t, max_iter, u, h))
        raw_arr = np.asarray(raw, dtype=float)
        denom = float(np.max(raw_arr) - np.min(raw_arr))
        if denom <= 1.0e-30:
            cm = np.zeros_like(raw_arr)
        else:
            cm = (raw_arr - np.min(raw_arr)) / denom
        return {"aroma_concentration": np.clip(cm, 0.0, 1.0)}

    def _levy_scalar(self) -> float:
        # Eqs. (29)–(30), using Mantegna's stable form for numerical robustness.
        beta = float(self._params.get("levy_beta", 1.5))
        scale = float(self._params.get("levy_scale", 0.01))
        sigma = (
            math.gamma(1.0 + beta)
            * math.sin(math.pi * beta / 2.0)
            / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
        ) ** (1.0 / beta)
        u = np.random.normal(0.0, sigma)
        v = np.random.normal(0.0, 1.0)
        return float(scale * u / ((abs(v) ** (1.0 / beta)) + 1.0e-12))

    def _aroma_factor(self, t: int, max_iter: int) -> float:
        # Eqs. (21)–(22). The original model is three-dimensional; for arbitrary
        # d-dimensional optimization we use its scalar norm as a trajectory factor.
        dc = float(self._params.get("diffusion_coefficient", 0.6))
        base = float(t) / float(max_iter)
        step = math.sqrt(max(2.0 * dc * max_iter, 1.0e-12))
        perturb = base + np.random.normal(0.0, 1.0, 3) * step
        factor = float(np.linalg.norm(perturb))
        cap = float(self._params.get("aroma_factor_cap", 3.0))
        return float(np.clip(factor, 1.0e-12, cap))

    def _energy_terms(self, t: int, max_iter: int) -> tuple[float, float, float]:
        # Eqs. (23)–(25).
        fatigue = float(np.log(max((t * np.pi + max_iter) / float(max_iter), 1.0e-12)))
        lam = 0.1 * np.random.rand()
        vo2 = 0.2 * np.random.rand()
        energy = float(np.exp(-lam * vo2 * t * (1.0 + fatigue)))
        a1 = float(2.0 * energy * np.random.rand() - energy)
        return a1, energy, fatigue

    @staticmethod
    def _finite_clip(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        midpoint = 0.5 * (lo + hi)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0e100, neginf=-1.0e100)
        x = np.where(np.isfinite(x), x, midpoint)
        return np.clip(x, lo, hi)

    def _luring_trial(self, x: np.ndarray, best: np.ndarray, second_best: np.ndarray, a: float, a1: float, levy: float, c1: float, t: int, max_iter: int) -> np.ndarray:
        # Attraction/capture — Eqs. (19)–(20).
        da = np.abs(a * second_best - x)
        xa_next = best + second_best - a1 * da

        # Movement/feeding — Eqs. (26)–(27).
        dm = np.abs(c1 * da - x) + levy * (1.0 - float(t) / float(max_iter))
        xm_next = best + x - a1 * dm

        # Luring update — Eq. (31). Trigonometric terms are bounded to avoid
        # tan singularities while preserving the equation's oscillatory role.
        exp_t = math.exp(min(float(t) / float(max_iter), 50.0))
        exp_osc = math.exp(min(4.0 * math.pi * math.pi * float(t) / float(max_iter), 50.0))
        tan_arg = np.clip(xm_next * exp_osc, -1.35, 1.35)
        denom = 4.0 * math.pi * np.tan(tan_arg)
        denom = np.where(np.abs(denom) < 1.0e-12, np.sign(denom) * 1.0e-12 + 1.0e-12, denom)
        perturb = (np.random.rand() ** 3) * np.sin(xa_next * exp_t) / denom
        return self._finite_clip(0.5 * (xm_next + xa_next) + perturb, self._lo, self._hi)

    def _predation_trial(self, x: np.ndarray, best: np.ndarray, cm: float, a: float, a1: float, levy: float, c1: float) -> np.ndarray:
        if cm < 0.3:
            # Search/localization — Eqs. (32)–(33), followed by Eq. (38).
            dm = np.abs(levy * x - best)
            xm_next = np.sin(c1 * best) + a1 * np.abs(x - levy * dm)
        elif cm < 0.6:
            # Rapid approach — Eqs. (34)–(35), followed by Eq. (38).
            dm = np.abs(a * x - best)
            xm_next = best - a1 * np.abs(x - np.exp(-a) * np.sin(np.random.rand() * math.pi) * dm)
        else:
            # Digging/feeding — Eqs. (36)–(37), followed by Eq. (38).
            dm = np.abs(c1 * x - best)
            xm_next = best + a1 * np.abs(x - dm)
        return self._finite_clip(c1 * xm_next, self._lo, self._hi)

    def _step_impl(self, state, pop: np.ndarray):
        n = pop.shape[0]
        t = max(1, state.step + 1)
        max_iter = self._max_iter()
        c1 = max(0.0, 2.0 - 2.0 * float(t) / float(max_iter))  # Eq. (28)
        cm_curve = np.asarray(state.payload.get("aroma_concentration"), dtype=float)
        cm = float(cm_curve[min(t - 1, cm_curve.size - 1)]) if cm_curve.size else 0.0
        a = self._aroma_factor(t, max_iter)
        levy = self._levy_scalar()

        order = self._order(pop[:, -1])
        best = pop[order[0], :-1].copy()
        second_best = pop[order[1 if len(order) > 1 else 0], :-1].copy()

        trials = np.empty_like(pop[:, :-1])
        operator_labels = ["carryover"] * n
        energies = []
        fatigues = []
        for i in range(n):
            a1, energy, fatigue = self._energy_terms(t, max_iter)
            energies.append(energy)
            fatigues.append(fatigue)
            r1 = np.random.rand()
            x = pop[i, :-1]

            if cm >= 0.2 and r1 <= 0.5:
                trial = self._luring_trial(x, best, second_best, a, a1, levy, c1, t, max_iter)
                operator_labels[i] = "cpo.aroma_luring_trial"
            else:
                trial = self._predation_trial(x, best, cm, a, a1, levy, c1)
                operator_labels[i] = "cpo.predation_feeding_trial"
            trials[i] = trial

        fitness = self._evaluate_population(trials)
        pop[:, :-1] = trials
        pop[:, -1] = fitness
        return pop, n, {
            "aroma_cm": cm,
            "aroma_factor": a,
            "c1": c1,
            "levy": levy,
            "mean_energy": float(np.mean(energies)) if energies else None,
            "mean_fatigue": float(np.mean(fatigues)) if fatigues else None,
            "operator_labels": operator_labels,
        }
