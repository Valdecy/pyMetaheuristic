"""pyMetaheuristic src — Fruit-Fly Algorithm Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class FFAEngine(PortedPopulationEngine):
    """Fruit-Fly Algorithm — distance-attenuated brightness attraction with random mutation."""
    algorithm_id = "ffa"; algorithm_name = "Fruit-Fly Algorithm"; family = "swarm"
    _REFERENCE     = {"doi": "10.1016/j.knosys.2011.07.001"}
    capabilities = CapabilityProfile(has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True, supports_diversity_metrics=True)
    _DEFAULTS = dict(population_size=50, gamma=0.001, beta_base=2.0, alpha=0.2, alpha_damp=0.99, delta=0.05, exponent=2)

    def _initialize_payload(self, pop):
        return {"dyn_alpha": float(self._params.get("alpha", 0.2))}

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        gamma  = float(self._params.get("gamma", 0.001)); beta_base = float(self._params.get("beta_base", 2.0))
        alpha_damp = float(self._params.get("alpha_damp", 0.99)); delta = float(self._params.get("delta", 0.05))
        exponent   = int(self._params.get("exponent", 2)); dyn_alpha = float(state.payload.get("dyn_alpha", 0.2))
        dmax = np.sqrt(dim)
        order = self._order(pop[:,-1]); evals = 0

        for i in range(n):
            best_child = pop[i].copy()
            for j in range(i+1, n):
                if self._is_better(float(pop[j,-1]), float(best_child[-1])):
                    rij  = np.linalg.norm(best_child[:-1]-pop[j,:-1]) / max(dmax, 1e-30)
                    beta = beta_base*np.exp(-gamma*rij**exponent)
                    mv   = delta*np.random.random(dim)
                    temp = np.dot(pop[j,:-1]-best_child[:-1], np.random.random((dim,dim)))
                    pos  = np.clip(best_child[:-1] + dyn_alpha*mv + beta*temp, self._lo, self._hi)
                    fit  = float(self.problem.evaluate(pos)); evals+=1
                    if self._is_better(fit, float(best_child[-1])):
                        best_child[:-1]=pos; best_child[-1]=fit
            if self._is_better(float(best_child[-1]), float(pop[i,-1])): pop[i]=best_child
        # Append best to maintain diversity (original paper)
        order2 = self._order(pop[:,-1])
        dyn_alpha = alpha_damp * dyn_alpha
        return pop, evals, {"dyn_alpha": dyn_alpha}
