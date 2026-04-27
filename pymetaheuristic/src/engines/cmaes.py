"""
pyMetaheuristic src — Covariance Matrix Adaptation Evolution Strategy Engine
=============================================================================
Native macro-step: sample offspring → rank → update mean, step-size, covariance
payload keys: mean (ndarray), ps (path-σ), pc (path-C), C (covariance), sigma (float),
              mu (int), weights (ndarray), cs, ds, cc, c1, cmu, ENN, hth
"""
from __future__ import annotations
import warnings
import numpy as np
from .protocol import (BaseEngine, CandidateRecord, CapabilityProfile,
                        EngineConfig, EngineState, OptimizationResult, ProblemSpec)


class CMAESEngine(BaseEngine):
    algorithm_id   = "cmaes"
    algorithm_name = "Covariance Matrix Adaptation Evolution Strategy"
    family         = "evolutionary"
    capabilities   = CapabilityProfile(has_population=True, supports_candidate_injection=True)
    _DEFAULTS      = dict(population_size=20)
    _REFERENCE     = dict(doi="10.1109/ICEC.1996.542381")

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **config.params}
        self._n = max(4, int(p["population_size"]))
        if config.seed is not None:
            np.random.seed(config.seed)

    def _build_constants(self, D: int, N: int) -> dict:
        mu = N // 2
        w  = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w  = w / w.sum()
        mu_eff = 1.0 / (w ** 2).sum()
        cs  = (mu_eff + 2) / (D + mu_eff + 5)
        ds  = 1 + cs + 2 * max(np.sqrt((mu_eff - 1) / (D + 1)) - 1, 0)
        ENN = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
        cc  = (4 + mu_eff / D) / (4 + D + 2 * mu_eff / D)
        c1  = 2 / ((D + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2) ** 2 + 2 * mu_eff / 2))
        hth = (1.4 + 2 / (D + 1)) * ENN
        return dict(mu=mu, weights=w, mu_eff=mu_eff, cs=cs, ds=ds, ENN=ENN,
                    cc=cc, c1=c1, cmu=cmu, hth=hth)

    def initialize(self) -> EngineState:
        D  = self.problem.dimension
        lo = np.array(self.problem.min_values, dtype=float)
        hi = np.array(self.problem.max_values, dtype=float)
        mean  = np.random.uniform(lo, hi)
        sigma = 0.1 * (hi - lo)
        # evaluate one point to get initial best
        fit = self.problem.evaluate(mean)
        c   = self._build_constants(D, self._n)
        payload = dict(
            mean=mean, ps=np.zeros(D), pc=np.zeros(D),
            C=np.eye(D), sigma=sigma,
            mu=c["mu"], weights=c["weights"], mu_eff=c["mu_eff"],
            cs=c["cs"], ds=c["ds"], ENN=c["ENN"],
            cc=c["cc"], c1=c["c1"], cmu=c["cmu"], hth=c["hth"],
            gen=0,
        )
        return EngineState(step=0, evaluations=1,
                           best_position=mean.tolist(), best_fitness=float(fit),
                           initialized=True, payload=payload)

    def step(self, state: EngineState) -> EngineState:
        p   = state.payload
        D   = self.problem.dimension
        lo  = np.array(self.problem.min_values, dtype=float)
        hi  = np.array(self.problem.max_values, dtype=float)
        N   = self._n
        mu  = p["mu"]
        w   = p["weights"]
        mean, ps, pc = np.array(p["mean"]), np.array(p["ps"]), np.array(p["pc"])
        C, sigma     = np.array(p["C"]), np.array(p["sigma"])
        cs, ds, ENN  = p["cs"], p["ds"], p["ENN"]
        cc, c1, cmu  = p["cc"], p["c1"], p["cmu"]
        hth          = p["hth"]
        gen          = p["gen"]

        # Eigen decomposition for sampling
        eigvals, B = np.linalg.eigh(C)
        eigvals    = np.maximum(eigvals, 0.0)
        D_sqrt     = np.sqrt(eigvals)

        # Sample N offspring
        z     = np.random.randn(N, D)
        steps = (B * D_sqrt) @ z.T  # shape (D, N)
        steps = steps.T             # (N, D)
        dec   = mean + sigma * steps
        dec   = np.clip(dec, lo, hi)

        # Evaluate
        fit = np.array([self.problem.evaluate(dec[i]) for i in range(N)])

        # Rank
        rank   = np.argsort(fit)
        steps  = steps[rank]
        mstep  = w @ steps[:mu]

        # Update mean
        mean = mean + sigma * mstep

        # Step-size control
        # Cholesky of C for ps update
        try:
            L  = np.linalg.cholesky(C)
            Linv_mstep = np.linalg.solve(L, mstep)
        except np.linalg.LinAlgError:
            Linv_mstep = mstep  # fallback
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * p["mu_eff"]) * Linv_mstep
        sigma = sigma * np.exp((cs / ds) * (np.linalg.norm(ps) / ENN - 1)) ** 0.3

        # Covariance update
        gen1  = gen + 1
        hs    = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen1)) < hth
        delta = (1 - hs) * cc * (2 - cc)
        pc    = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * p["mu_eff"]) * mstep
        C     = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta * C)
        for i in range(mu):
            C = C + cmu * w[i] * np.outer(steps[i], steps[i])

        # Enforce PSD
        eigv, V = np.linalg.eigh(C)
        if np.any(eigv < 0):
            C = V @ np.diag(np.maximum(eigv, 0)) @ V.T

        best_idx = int(rank[0])
        bf       = float(fit[best_idx])
        bp       = dec[best_idx].tolist()

        state.payload = dict(mean=mean, ps=ps, pc=pc, C=C, sigma=sigma,
                             mu=mu, weights=w, mu_eff=p["mu_eff"],
                             cs=cs, ds=ds, ENN=ENN, cc=cc, c1=c1, cmu=cmu, hth=hth, gen=gen1)
        state.evaluations += N
        state.step        += 1
        if self.problem.is_better(bf, state.best_fitness):
            state.best_fitness  = bf
            state.best_position = bp
        return state

    def observe(self, state: EngineState) -> dict:
        sigma = state.payload["sigma"]
        return dict(step=state.step, evaluations=state.evaluations,
                    best_fitness=state.best_fitness,
                    sigma=float(np.mean(sigma)) if hasattr(sigma, "__len__") else float(sigma))

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(position=list(state.best_position),
                               fitness=state.best_fitness,
                               source_algorithm=self.algorithm_id,
                               source_step=state.step, role="best")

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(algorithm_id=self.algorithm_id,
                                  best_position=list(state.best_position),
                                  best_fitness=state.best_fitness,
                                  steps=state.step, evaluations=state.evaluations,
                                  termination_reason=state.termination_reason,
                                  capabilities=self.capabilities,
                                  metadata=dict(algorithm_name=self.algorithm_name,
                                                elapsed_time=state.elapsed_time))

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        mean = np.array(state.payload["mean"])
        return [CandidateRecord(position=mean.tolist(), fitness=state.best_fitness,
                                source_algorithm=self.algorithm_id,
                                source_step=state.step, role="mean")]
