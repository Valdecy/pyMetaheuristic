"""pyMetaheuristic src — Coyote Optimization Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class COAEngine(PortedPopulationEngine):
    """Coyote Optimization Algorithm — pack-based social learning with pup birth and migration."""
    algorithm_id   = "coa"
    algorithm_name = "Coyote Optimization Algorithm"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/CEC.2018.8477769"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, n_coyotes=5)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n         = pop.shape[0]
        nc        = max(2, int(self._params.get("n_coyotes", 5)))
        n_packs   = max(1, n // nc)
        ages      = np.zeros(n, dtype=int)
        return {"ages": ages, "n_packs": n_packs, "nc": nc}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim   = pop.shape[0], self.problem.dimension
        nc       = int(state.payload.get("nc", 5))
        n_packs  = int(state.payload.get("n_packs", max(1, n // nc)))
        ages     = np.asarray(state.payload.get("ages", np.zeros(n, int)), dtype=int)
        ps       = 1.0 / nc          # probability of social condition
        p_leave  = 0.005 * nc ** 2   # leaving probability
        evals    = 0

        # Split into packs
        indices = np.arange(n)
        packs   = [indices[i*nc:(i+1)*nc] for i in range(n_packs)]

        for pack_idx in packs:
            if len(pack_idx) < 2:
                continue
            pack_fit  = pop[pack_idx, -1]
            order     = self._order(pack_fit)
            sorted_pk = pack_idx[order]
            alpha_pos = pop[sorted_pk[0], :-1].copy()      # alpha (best)
            tendency  = pop[pack_idx, :-1].mean(axis=0)    # social tendency

            # Update social condition (Eq. 12)
            new_pos = np.empty((len(pack_idx), dim))
            for li, i in enumerate(pack_idx):
                others = [x for x in range(len(pack_idx)) if x != li]
                rc1, rc2 = np.random.choice(others, 2, replace=False)
                pos = pop[i, :-1] + np.random.random() * (alpha_pos - pop[pack_idx[rc1], :-1]) \
                                  + np.random.random() * (tendency  - pop[pack_idx[rc2], :-1])
                new_pos[li] = np.clip(pos, self._lo, self._hi)

            new_fit = self._evaluate_population(new_pos); evals += len(pack_idx)
            for li, i in enumerate(pack_idx):
                if self._is_better(float(new_fit[li]), float(pop[i, -1])):
                    pop[i, :-1] = new_pos[li]; pop[i, -1] = new_fit[li]

            # Pup birth (Eq. 7)
            id_dad, id_mom = np.random.choice(len(pack_idx), 2, replace=False)
            prob1 = (1.0 - ps) / 2.0
            r     = np.random.random(dim)
            pup   = np.where(r < prob1, pop[pack_idx[id_dad], :-1], pop[pack_idx[id_mom], :-1])
            pup   = np.clip(np.random.normal(0, 1) * pup, self._lo, self._hi)
            pup_fit = float(self.problem.evaluate(pup)); evals += 1
            # Replace oldest if pup is better than worst
            worst_local = sorted_pk[-1]
            if self._is_better(pup_fit, float(pop[worst_local, -1])):
                oldest = pack_idx[np.argmax(ages[pack_idx])]
                pop[oldest, :-1] = pup; pop[oldest, -1] = pup_fit
                ages[oldest] = 0

        # Migration between packs
        if n_packs > 1 and np.random.random() < p_leave:
            p1, p2 = np.random.choice(n_packs, 2, replace=False)
            if len(packs[p1]) and len(packs[p2]):
                i1 = packs[p1][np.random.randint(len(packs[p1]))]
                i2 = packs[p2][np.random.randint(len(packs[p2]))]
                pop[[i1, i2]] = pop[[i2, i1]]

        ages += 1
        return pop, evals, {"ages": ages, "n_packs": n_packs, "nc": nc}
