"""pyMetaheuristic src — Heap-Based Optimizer Engine"""
from __future__ import annotations
import numpy as np
from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine

class HBOEngine(PortedPopulationEngine):
    """Heap-Based Optimizer — corporate-rank hierarchy update guided by parent/friend nodes."""
    algorithm_id   = "hbo"
    algorithm_name = "Heap-Based Optimizer"
    family         = "human"
    _REFERENCE     = {"doi": "10.1016/j.eswa.2020.113702"}
    capabilities   = CapabilityProfile(
        has_population=True, supports_candidate_injection=True,
        supports_checkpoint=True, supports_framework_constraints=True,
        supports_diversity_metrics=True,
    )
    _DEFAULTS = dict(population_size=50, degree=2)

    @staticmethod
    def _heapify(pop: np.ndarray, degree: int, is_min: bool) -> np.ndarray:
        """Return heap permutation as array of indices (heap[i] = pop index)."""
        n    = pop.shape[0]
        heap = list(range(n))
        # Simple heapify by fitness
        for i in range(n - 1, 0, -1):
            p = max(0, (i - 1) // degree)
            fi, fp = float(pop[heap[i], -1]), float(pop[heap[p], -1])
            if (fi < fp) == is_min:
                heap[i], heap[p] = heap[p], heap[i]
        return np.array(heap, dtype=int)

    def _initialize_payload(self, pop: np.ndarray) -> dict:
        n    = pop.shape[0]
        deg  = max(2, int(self._params.get("degree", 2)))
        T    = max(1, self.config.max_steps or 500)
        cyc  = max(1, T // 4)
        heap = self._heapify(pop, deg, self.problem.objective == "min")
        # friend_limits[i] = [start, end) of friends in heap
        fl   = np.zeros((n, 2), dtype=int)
        for i in range(n):
            fl[i, 0] = max(0, (i - 1) // deg)
            fl[i, 1] = min(n, i * deg + 1)
        return {"heap": heap, "deg": deg, "it_per_cycle": cyc, "qtr_cycle": max(1, cyc // 4), "fl": fl}

    def _step_impl(self, state, pop: np.ndarray):
        n, dim  = pop.shape[0], self.problem.dimension
        T       = max(1, self.config.max_steps or 500)
        t       = state.step + 1
        heap    = np.asarray(state.payload.get("heap", np.arange(n)), dtype=int)
        deg     = int(state.payload.get("deg", 2))
        ipc     = float(state.payload.get("it_per_cycle", T // 4))
        qc      = float(state.payload.get("qtr_cycle", max(1, T // 16)))
        fl      = np.asarray(state.payload.get("fl", np.zeros((n, 2), int)), dtype=int)
        is_min  = self.problem.objective == "min"
        evals   = 0

        gama_raw = (((t - 1) % max(1, int(ipc))) + 1) / max(1e-9, qc)
        gama     = abs(2.0 - gama_raw)
        p1       = 1.0 - t / T
        p2       = p1 + (1.0 - p1) / 2.0

        new_pop  = pop.copy()
        for ci in range(n - 1, 0, -1):
            par_ci   = max(0, (ci + 1) // deg - 1)
            cur_hi   = int(heap[ci])
            par_hi   = int(heap[par_ci])
            # pick a friend (sibling or nearby)
            f_start, f_end = int(fl[ci, 0]), int(fl[ci, 1])
            candidates = [x for x in range(f_start, f_end) if x != ci]
            fri_ci   = int(np.random.choice(candidates)) if candidates else par_ci
            fri_hi   = int(heap[fri_ci])

            rr = np.random.random(dim)
            rn = 2.0 * np.random.random(dim) - 1.0
            cur_pos = new_pop[cur_hi, :-1].copy()
            for j in range(dim):
                if rr[j] < p1:
                    pass                         # no change
                elif rr[j] < p2:
                    cur_pos[j] = new_pop[par_hi, j] + rn[j] * gama * abs(new_pop[par_hi, j] - cur_pos[j])
                else:
                    fri_better = self._is_better(float(new_pop[fri_hi, -1]), float(new_pop[cur_hi, -1]))
                    ref        = new_pop[fri_hi, j] if fri_better else cur_pos[j]
                    cur_pos[j] = ref + rn[j] * gama * abs(new_pop[fri_hi, j] - cur_pos[j])
            cur_pos  = np.clip(cur_pos, self._lo, self._hi)
            cur_fit  = float(self.problem.evaluate(cur_pos)); evals += 1
            if self._is_better(cur_fit, float(new_pop[cur_hi, -1])):
                new_pop[cur_hi, :-1] = cur_pos; new_pop[cur_hi, -1] = cur_fit

            # re-heapify upward
            idx = ci
            while idx > 0:
                par = max(0, (idx + 1) // deg - 1)
                fi  = float(new_pop[int(heap[idx]), -1])
                fp  = float(new_pop[int(heap[par]), -1])
                if (fi < fp) == is_min:
                    heap[idx], heap[par] = heap[par], heap[idx]
                    idx = par
                else:
                    break

        pop = new_pop
        return pop, evals, {"heap": heap, "deg": deg, "it_per_cycle": ipc,
                            "qtr_cycle": qc, "fl": fl}
