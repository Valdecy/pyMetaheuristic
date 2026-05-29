"""
pyMetaheuristic src — Particle Swarm Optimization Engine
========================================================
Reference implementation of a *population-based* engine.

Native macro-step:  one full swarm update
  position update  →  individual best update  →  global best update  →  velocity update

payload keys
------------
    positions   : np.ndarray  (swarm_size, dim)
    velocities  : np.ndarray  (swarm_size, dim)
    i_best      : np.ndarray  (swarm_size, dim+1)  individual bests with fitness
    g_best      : np.ndarray  (dim+1,)             global best with fitness
    w           : float       current inertia weight
    c1, c2      : float       acceleration coefficients
"""

from __future__ import annotations

import numpy as np

from .protocol import (
    BaseEngine,
    CandidateRecord,
    CapabilityProfile,
    EngineConfig,
    EngineState,
    OptimizationResult,
    ProblemSpec,
)


class PSOEngine(BaseEngine):
    """Particle Swarm Optimization — native engine."""

    algorithm_id   = "pso"
    algorithm_name = "Particle Swarm Optimization"
    family         = "swarm"
    _REFERENCE     = {"doi": "10.1109/ICNN.1995.488968"}
    capabilities   = CapabilityProfile(
        has_population              = True,
        has_archive                 = False,
        supports_candidate_injection = True,
        supports_restart            = False,
        supports_checkpoint         = True,
        supports_diversity_metrics  = True,
    )

    _DEFAULTS = dict(swarm_size=30, w=0.9, c1=2.0, c2=2.0, decay=0)

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._swarm_size = int(p["swarm_size"])
        self._w0         = float(p["w"])
        self._c1         = float(p["c1"])
        self._c2         = float(p["c2"])
        self._decay      = float(p["decay"])
        if config.seed is not None:
            np.random.seed(config.seed)

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    def initialize(self) -> EngineState:
        prob = self.problem
        dim  = prob.dimension
        lo   = np.array(prob.min_values, dtype=float)
        hi   = np.array(prob.max_values, dtype=float)

        positions = np.random.uniform(lo, hi, (self._swarm_size, dim))
        fitness   = self._evaluate_population(positions)
        pop       = np.hstack((positions, fitness[:, np.newaxis]))

        velocities = np.random.uniform(lo, hi, (self._swarm_size, dim))
        i_best     = pop.copy()
        best_idx   = np.argmin(pop[:, -1])
        g_best     = pop[best_idx].copy()

        state = EngineState(
            step         = 0,
            evaluations  = self._swarm_size,
            best_position= g_best[:-1].tolist(),
            best_fitness = float(g_best[-1]),
            initialized  = True,
            payload      = dict(
                positions  = pop,
                velocities = velocities,
                i_best     = i_best,
                g_best     = g_best,
                w          = self._w0,
                c1         = self._c1,
                c2         = self._c2,
            ),
        )
        return state

    def step(self, state: EngineState) -> EngineState:
        pl   = state.payload
        prob = self.problem
        dim  = prob.dimension
        lo   = np.array(prob.min_values, dtype=float)
        hi   = np.array(prob.max_values, dtype=float)

        positions  = pl["positions"]
        velocities = pl["velocities"]
        i_best     = pl["i_best"]
        g_best     = pl["g_best"]
        w, c1, c2  = pl["w"], pl["c1"], pl["c2"]

        # --- velocity & position update ---------------------------------
        r1 = np.random.rand(self._swarm_size, dim)
        r2 = np.random.rand(self._swarm_size, dim)
        inertia_component   = w  * velocities
        cognitive_component = c1 * r1 * (i_best[:, :-1] - positions[:, :-1])
        social_component    = c2 * r2 * (g_best[:-1]    - positions[:, :-1])
        velocities = inertia_component + cognitive_component + social_component

        # Per-particle dominant operator label (used by existing probe logic).
        operator_labels = []
        for _i in range(self._swarm_size):
            norms = {
                "pso.inertia_velocity_update": float(np.linalg.norm(inertia_component[_i])),
                "pso.cognitive_memory_update": float(np.linalg.norm(cognitive_component[_i])),
                "pso.social_global_update":    float(np.linalg.norm(social_component[_i])),
            }
            operator_labels.append(max(norms, key=norms.get))

        prev_fitness = positions[:, -1].copy()
        new_pos  = np.clip(positions[:, :-1] + velocities, lo, hi)
        new_fit  = self._evaluate_population(new_pos)
        positions = np.hstack((new_pos, new_fit[:, np.newaxis]))

        # --- individual best --------------------------------------------
        better = positions[:, -1] < i_best[:, -1]
        i_best[better] = positions[better]

        # --- global best ------------------------------------------------
        best_i = np.argmin(i_best[:, -1])
        if i_best[best_i, -1] < g_best[-1]:
            g_best = i_best[best_i].copy()

        # --- inertia decay ----------------------------------------------
        n = self._decay
        t = state.step + 1
        T = self.config.max_steps or 1
        if n > 0:
            w  = self._w0 * (1 - (t ** n) / (T ** n))
            c1 = (1 - self._c1) * (t / T) + self._c1
            c2 = (1 - self._c2) * (t / T) + self._c2

        # --- EvoMapX lineage: per-particle operator attribution ---------
        # Each particle's signed Δf is split across the three velocity
        # components proportionally to their contribution norms.  This gives
        # the probe real per-operator signed deltas instead of a single
        # macro-step label.
        lineage = []
        for idx in range(self._swarm_size):
            pf = float(prev_fitness[idx])
            cf = float(positions[idx, -1])
            delta = pf - cf  # positive = improvement (minimisation)
            i_norm = float(np.linalg.norm(inertia_component[idx]))
            c_norm = float(np.linalg.norm(cognitive_component[idx]))
            s_norm = float(np.linalg.norm(social_component[idx]))
            total_norm = i_norm + c_norm + s_norm or 1.0
            lineage.append({
                "id": f"pso:inertia:{idx}",
                "index": idx,
                "operator": "pso_inertia_component",
                "parent_ids": [f"parent:{idx}"],
                "parent_index": idx,
                "parent_fitness": pf,
                "child_fitness": cf,
                "lineage_delta": delta * (i_norm / total_norm),
            })
            lineage.append({
                "id": f"pso:cognitive:{idx}",
                "index": idx,
                "operator": "pso_cognitive_component",
                "parent_ids": [f"parent:{idx}"],
                "parent_index": idx,
                "parent_fitness": pf,
                "child_fitness": cf,
                "lineage_delta": delta * (c_norm / total_norm),
            })
            lineage.append({
                "id": f"pso:social:{idx}",
                "index": idx,
                "operator": "pso_social_component",
                "parent_ids": [f"parent:{idx}"],
                "parent_index": idx,
                "parent_fitness": pf,
                "child_fitness": cf,
                "lineage_delta": delta * (s_norm / total_norm),
            })

        # --- write back -------------------------------------------------
        state.payload = dict(
            positions  = positions,
            velocities = velocities,
            i_best     = i_best,
            g_best     = g_best,
            w=w, c1=c1, c2=c2,
            operator_labels=operator_labels,
            lineage=lineage,
        )
        state.step        += 1
        state.evaluations += self._swarm_size
        state.best_position = g_best[:-1].tolist()
        state.best_fitness  = float(g_best[-1])
        return state

    def observe(self, state: EngineState) -> dict:
        pop = state.payload["positions"]
        lineage = state.payload.get("lineage", [])
        contrib: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rec in lineage:
            op = str(rec.get("operator") or "pso_social_component")
            delta = float(rec.get("lineage_delta", 0.0))
            contrib[op] = contrib.get(op, 0.0) + delta
            counts[op] = counts.get(op, 0) + 1
        obs = {
            "step":         state.step,
            "evaluations":  state.evaluations,
            "best_fitness": state.best_fitness,
            "mean_fitness": float(np.mean(pop[:, -1])),
            "std_fitness":  float(np.std(pop[:, -1])),
            "diversity":    self._diversity(state),
            "inertia_w":    state.payload["w"],
        }
        if contrib:
            obs["operator_contributions"] = contrib
            obs["operator_counts"] = counts
            obs["evomapx_delta_f"] = "signed"
            obs["evomapx_fidelity"] = "guessed_from_code_profiles_pso"
        return obs

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(
            position         = list(state.best_position),
            fitness          = state.best_fitness,
            source_algorithm = self.algorithm_id,
            source_step      = state.step,
            role             = "best",
        )

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id       = self.algorithm_id,
            best_position      = list(state.best_position),
            best_fitness       = state.best_fitness,
            steps              = state.step,
            evaluations        = state.evaluations,
            termination_reason = state.termination_reason,
            capabilities       = self.capabilities,
            metadata           = {
                "algorithm_name": self.algorithm_name,
                "swarm_size":     self._swarm_size,
                "elapsed_time":   state.elapsed_time,
            },
        )

    # ------------------------------------------------------------------
    # Population mixin behaviour
    # ------------------------------------------------------------------

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = state.payload["positions"]
        return [
            CandidateRecord(
                position         = pop[i, :-1].tolist(),
                fitness          = float(pop[i, -1]),
                source_algorithm = self.algorithm_id,
                source_step      = state.step,
                role             = "current",
            )
            for i in range(self._swarm_size)
        ]

    def export_candidates(self, state: EngineState, k: int = 1, mode: str = "best") -> list[CandidateRecord]:
        pop = state.payload["positions"]
        if mode == "best":
            idx = np.argsort(pop[:, -1])[:k]
            role = "elite"
        elif mode == "diverse":
            idx = self._diverse_indices(state, k)
            role = "diverse"
        else:
            idx = np.argsort(pop[:, -1])[:k]
            role = "elite"
        return [
            CandidateRecord(
                position         = pop[i, :-1].tolist(),
                fitness          = float(pop[i, -1]),
                source_algorithm = self.algorithm_id,
                source_step      = state.step,
                role             = role,
            )
            for i in idx
        ]

    def inject_candidates(self, state: EngineState, candidates: list[CandidateRecord], policy: str = "native") -> EngineState:
        """Replace the *k* worst particles with migrants."""
        pop = state.payload["positions"]
        worst_idx = np.argsort(pop[:, -1])[::-1]
        prob = self.problem
        for j, (wi, cand) in enumerate(zip(worst_idx, candidates)):
            pos = np.clip(cand.position, prob.min_values, prob.max_values)
            fit = prob.evaluate(pos)
            pop[wi, :-1] = pos
            pop[wi, -1]  = fit
            state.evaluations += 1
            if fit < state.payload["i_best"][wi, -1]:
                state.payload["i_best"][wi, :-1] = pos
                state.payload["i_best"][wi, -1]  = fit
            if fit < state.best_fitness:
                state.best_fitness  = float(fit)
                state.best_position = list(pos)
                state.payload["g_best"][:-1] = pos
                state.payload["g_best"][-1]  = fit
        state.payload["positions"] = pop
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _diversity(self, state: EngineState) -> float:
        pos   = state.payload["positions"][:, :-1]
        lo    = np.array(self.problem.min_values)
        hi    = np.array(self.problem.max_values)
        denom = np.linalg.norm(hi - lo) or 1.0
        n     = len(pos)
        if n < 2:
            return 0.0
        centroid = pos.mean(axis=0)
        return float(np.mean(np.linalg.norm(pos - centroid, axis=1)) / denom)

    def _diverse_indices(self, state: EngineState, k: int) -> list[int]:
        pos = state.payload["positions"][:, :-1]
        n   = len(pos)
        selected = [int(np.argmin(state.payload["positions"][:, -1]))]
        while len(selected) < min(k, n):
            dists = np.min(
                np.linalg.norm(pos[None, :, :] - pos[selected, :][:, None, :], axis=2),
                axis=0,
            )
            selected.append(int(np.argmax(dists)))
        return selected
