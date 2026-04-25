"""pyMetaheuristic src — Imperialist Competitive Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class ICAEngine(PortedPopulationEngine):
    """Imperialist Competitive Algorithm — empire assimilation, revolution and competition."""
    algorithm_id   = "ica"
    algorithm_name = "Imperialist Competitive Algorithm"
    family         = "human"
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, empire_count=5, assimilation_coeff=1.5,
                     revolution_prob=0.05, revolution_rate=0.1,
                     revolution_step=0.1, zeta=0.1)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n      = pop.shape[0]
        n_emp  = min(int(self._params.get("empire_count", 5)), n - 1)
        order  = self._order(pop[:, -1])
        # First n_emp are imperialists, rest are colonies
        imps   = order[:n_emp].tolist()
        cols   = order[n_emp:].tolist()
        # Assign colonies to empires by roulette on imperialist fitness
        fit_imp = pop[imps, -1].copy()
        if self.problem.objective == "min":
            inv    = 1.0 / (fit_imp + 1e-30)
            prob   = inv / inv.sum()
        else:
            prob   = fit_imp / (fit_imp.sum() + 1e-30)
        assignment = np.random.choice(n_emp, len(cols), p=prob).tolist()
        return {"imps": imps, "cols": cols, "assignment": assignment}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim         = pop.shape[0], self.problem.dimension
        ac             = float(self._params.get("assimilation_coeff", 1.5))
        rev_prob       = float(self._params.get("revolution_prob", 0.05))
        rev_rate       = float(self._params.get("revolution_rate", 0.1))
        rev_step       = float(self._params.get("revolution_step", 0.1))
        zeta           = float(self._params.get("zeta", 0.1))

        imps       = list(state.payload.get("imps", [0]))
        cols       = list(state.payload.get("cols", list(range(1, n))))
        assignment = list(state.payload.get("assignment", [0]*len(cols)))
        n_emp      = len(imps)
        n_rev_var  = max(1, int(rev_rate * dim))
        evals      = 0

        # 1. Assimilation
        for ci, imp_idx in zip(range(len(cols)), assignment):
            col_row = cols[ci]
            imp_pos = pop[imps[imp_idx], :-1]
            pos     = pop[col_row, :-1] + ac * np.random.random(dim) * (imp_pos - pop[col_row, :-1])
            pos     = np.clip(pos, self._lo, self._hi)
            fit     = float(self.problem.evaluate(pos)); evals += 1
            pop[col_row, :-1] = pos; pop[col_row, -1] = fit

        # 2. Revolution (random-dimension reset)
        for imp_row in imps:
            idx = np.random.choice(dim, n_rev_var, replace=False)
            pos = pop[imp_row, :-1].copy()
            pos[idx] = np.random.uniform(self._lo[idx], self._hi[idx])
            fit = float(self.problem.evaluate(pos)); evals += 1
            pop[imp_row, :-1] = pos; pop[imp_row, -1] = fit
        for ci in range(len(cols)):
            if np.random.random() < rev_prob:
                idx = np.random.choice(dim, n_rev_var, replace=False)
                pos = pop[cols[ci], :-1].copy()
                pos[idx] = np.random.uniform(self._lo[idx], self._hi[idx])
                fit = float(self.problem.evaluate(pos)); evals += 1
                pop[cols[ci], :-1] = pos; pop[cols[ci], -1] = fit

        # 3. Intra-empire competition: colony beats imperialist?
        for ci, imp_idx in zip(range(len(cols)), assignment):
            if self._is_better(float(pop[cols[ci], -1]), float(pop[imps[imp_idx], -1])):
                imps[imp_idx], cols[ci] = cols[ci], imps[imp_idx]

        # 4. Inter-empire competition: steal weakest colony from weakest empire
        emp_costs = []
        for ei, imp_row in enumerate(imps):
            my_cols  = [cols[ci] for ci, a in enumerate(assignment) if a == ei]
            if my_cols:
                mean_col = float(np.mean(pop[my_cols, -1]))
            else:
                mean_col = float(pop[imp_row, -1])
            emp_costs.append(float(pop[imp_row, -1]) + zeta * mean_col)
        emp_costs = np.array(emp_costs)
        worst_emp = int(np.argmax(emp_costs))
        best_emp  = int(np.argmin(emp_costs))
        # Transfer one colony
        worst_cols = [ci for ci, a in enumerate(assignment) if a == worst_emp]
        if worst_cols:
            # Pick the weakest colony
            wc_fit  = [float(pop[cols[ci], -1]) for ci in worst_cols]
            wc_idx  = worst_cols[int(np.argmax(wc_fit))] if self.problem.objective=="min" \
                      else worst_cols[int(np.argmin(wc_fit))]
            assignment[wc_idx] = best_emp

        return pop, evals, {"imps": imps, "cols": cols, "assignment": assignment}
