"""pyMetaheuristic src — Squirrel Search Algorithm Engine"""
from __future__ import annotations
import math
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

def _levy_sq(dim: int, beta: float = 1.5) -> np.ndarray:
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, dim)
    v = np.abs(np.random.normal(0, 1, dim)) + 1e-30
    return u / v**(1/beta)

class SquirrelSAEngine(PortedPopulationEngine):
    """Squirrel Search Algorithm — acorn-hickory gliding search with predator-induced random walks."""
    algorithm_id   = "squirrel_sa"
    algorithm_name = "Squirrel Search Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.swevo.2018.02.013"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, n_food_sources=4, predator_prob=0.1,
                     gliding_constant=1.9, scaling_factor=18, beta=1.5)

    def _glide_dist(self) -> float:
        sf     = float(self._params.get("scaling_factor", 18))
        hl     = 8.0                   # height loss (m)
        Ld     = 0.675                 # lift coefficient
        Dd     = 0.370                 # drag coefficient
        mass   = 0.05                  # squirrel mass (kg)
        g      = 9.81
        # glide distance
        dist   = (Ld / Dd) * np.log(1.0 + hl * Dd / (Ld)) * (1.0 / g) * mass
        return float(dist / sf) + 1e-6

    def _step_impl(self, state, pop: np.ndarray):
        n, dim       = pop.shape[0], self.problem.dimension
        n_fs         = max(2, int(self._params.get("n_food_sources", 4)))
        n_acorn      = n_fs - 1
        p_pred       = float(self._params.get("predator_prob", 0.1))
        gc           = float(self._params.get("gliding_constant", 1.9))
        beta         = float(self._params.get("beta", 1.5))
        n_normal     = max(0, n - n_fs)

        order    = self._order(pop[:, -1])
        pop      = pop[order]                  # sort: best=hickory, next=acorn, rest=normal
        new_pop  = pop.copy()
        evals    = 0

        # Case 1: Acorn squirrels → hickory
        for idx in range(1, min(n_acorn + 1, n)):
            d_g = self._glide_dist()
            if np.random.random() >= p_pred:
                pos = pop[idx, :-1] + d_g * gc * (pop[0, :-1] - pop[idx, :-1])
            else:
                pos = np.random.uniform(self._lo, self._hi)
            new_pop[idx, :-1] = np.clip(pos, self._lo, self._hi)
            new_pop[idx, -1]  = float(self.problem.evaluate(new_pop[idx, :-1])); evals += 1

        # Case 2 & 3: Normal squirrels → acorn or hickory
        if n_normal > 0:
            idxs = np.arange(n_fs, n)
            np.random.shuffle(idxs)
            n_cut = max(1, np.random.randint(1, max(2, n_normal)))
            for idx in idxs[n_cut:]:            # toward acorn
                jdx = np.random.randint(1, n_acorn + 1) if n_acorn >= 1 else 0
                d_g = self._glide_dist()
                if np.random.random() >= p_pred:
                    pos = pop[idx, :-1] + d_g * gc * (pop[jdx, :-1] - pop[idx, :-1])
                else:
                    pos = np.random.uniform(self._lo, self._hi)
                new_pop[idx, :-1] = np.clip(pos, self._lo, self._hi)
                new_pop[idx, -1]  = float(self.problem.evaluate(new_pop[idx, :-1])); evals += 1
            for idx in idxs[:n_cut]:            # toward hickory
                d_g = self._glide_dist()
                if np.random.random() >= p_pred:
                    pos = pop[idx, :-1] + d_g * gc * (pop[0, :-1] - pop[idx, :-1])
                else:
                    pos = np.random.uniform(self._lo, self._hi)
                new_pop[idx, :-1] = np.clip(pos, self._lo, self._hi)
                new_pop[idx, -1]  = float(self.problem.evaluate(new_pop[idx, :-1])); evals += 1

        mask = self._better_mask(new_pop[:, -1], pop[:, -1])
        pop[mask] = new_pop[mask]
        return pop, evals, {}
