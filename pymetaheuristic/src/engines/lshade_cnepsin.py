"""pyMetaheuristic src — LSHADE-cnEpSin Engine"""
from __future__ import annotations

import numpy as np

from .protocol import CapabilityProfile
from ._ported_common import PortedPopulationEngine


class LSHADECnEpSinEngine(PortedPopulationEngine):
    """
    LSHADE-cnEpSin.

    """

    algorithm_id = "lshade_cnepsin"
    algorithm_name = "LSHADE-cnEpSin"
    family = "evolutionary"
    _REFERENCE = {
        "doi": "10.1109/CEC.2016.7744173",
        "title": "LSHADE-cnEpSin: Success-history based differential evolution with covariance matrix learning and ensemble sinusoidal parameter adaptation",
        "authors": "Mohamed W. Awad et al.",
        "year": 2016,
    }
    capabilities = CapabilityProfile(
        has_population=True,
        supports_candidate_injection=True,
        supports_checkpoint=True,
        supports_framework_constraints=True,
        supports_diversity_metrics=True,
        has_archive=True,
    )
    _DEFAULTS = dict(
        population_size=36,
        min_population_size=4,
        hist_mem_size=6,
        extern_arc_rate=2.6,
        pbest_factor=0.11,
        covariance_rate=0.25,
        sinusoid_frequency=0.5,
    )

    def _initialize_payload(self, pop):
        h = int(self._params.get("hist_mem_size", 6))
        return {
            "M_F": np.full(h, 0.5, dtype=float),
            "M_CR": np.full(h, 0.9, dtype=float),
            "k": 0,
            "archive": np.empty((0, self.problem.dimension), dtype=float),
            "initial_n": int(pop.shape[0]),
            "phase": 0.0,
        }

    def _neighbour_covariance_noise(self, positions, center_idx, rate):
        n, dim = positions.shape
        if n < 3 or rate <= 0.0:
            return np.zeros(dim, dtype=float)
        d = np.linalg.norm(positions - positions[center_idx], axis=1)
        nn = np.argsort(d)[1 : min(n, 6)]
        if nn.size < 2:
            return np.zeros(dim, dtype=float)
        cov = np.cov(positions[nn].T)
        cov += np.eye(dim) * 1.0e-12
        try:
            return np.random.multivariate_normal(np.zeros(dim), cov) * float(rate)
        except Exception:
            return np.random.normal(0.0, rate, dim) * np.std(positions[nn], axis=0)

    def _step_impl(self, state, pop):
        n, dim = pop.shape[0], self.problem.dimension
        T = max(1, self.config.max_steps or 100)
        progress = min(1.0, float(state.step + 1) / float(T))
        M_F = np.asarray(state.payload.get("M_F"), dtype=float)
        M_CR = np.asarray(state.payload.get("M_CR"), dtype=float)
        archive = np.asarray(state.payload.get("archive"), dtype=float).reshape(-1, dim)
        k = int(state.payload.get("k", 0)) % len(M_F)
        initial_n = int(state.payload.get("initial_n", n))
        phase = float(state.payload.get("phase", 0.0))

        order = self._order(pop[:, -1])
        pnum = max(2, int(float(self._params.get("pbest_factor", 0.11)) * n))
        union = np.vstack((pop[:, :-1], archive)) if archive.size else pop[:, :-1]

        trials = []
        Fs = []
        CRs = []
        cov_rate = float(self._params.get("covariance_rate", 0.25))
        freq = float(self._params.get("sinusoid_frequency", 0.5))

        for i in range(n):
            r = np.random.randint(len(M_F))
            phase_i = phase + 2.0 * np.pi * freq * progress + 2.0 * np.pi * np.random.rand()
            F_dec = 0.5 * (1.0 + np.sin(phase_i)) * (1.0 - progress)
            F_inc = np.clip(M_F[r] + 0.5 * np.sin(phase_i), 0.05, 1.0)
            if np.random.rand() < 0.5:
                F = np.clip(0.1 + F_dec, 0.05, 1.0)
            else:
                F = np.clip(np.random.standard_cauchy() * 0.1 + F_inc, 0.05, 1.0)
            CR = np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0)
            pbest = pop[np.random.choice(order[:pnum]), :-1]
            r1 = pop[self._rand_indices(n, i, 1)[0], :-1]
            r2 = union[np.random.randint(union.shape[0])]
            donor = pop[i, :-1] + F * (pbest - pop[i, :-1]) + F * (r1 - r2)
            donor = donor + self._neighbour_covariance_noise(pop[:, :-1], i, cov_rate * np.random.rand())
            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, donor, pop[i, :-1])
            trials.append(np.clip(trial, self._lo, self._hi))
            Fs.append(F)
            CRs.append(CR)

        trial_pop = self._pop_from_positions(np.asarray(trials, dtype=float))
        mask = self._better_mask(trial_pop[:, -1], pop[:, -1])
        if np.any(mask):
            archive = np.vstack((archive, pop[mask, :-1])) if archive.size else pop[mask, :-1].copy()
            max_arc = max(1, int(float(self._params.get("extern_arc_rate", 2.6)) * n))
            if archive.shape[0] > max_arc:
                archive = archive[np.random.choice(archive.shape[0], max_arc, replace=False)]
            sf = np.asarray(Fs, dtype=float)[mask]
            scr = np.asarray(CRs, dtype=float)[mask]
            df = np.abs(pop[mask, -1] - trial_pop[mask, -1])
            w = df / (float(np.sum(df)) + 1.0e-30)
            M_F[k] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1.0e-30)
            M_CR[k] = np.sum(w * scr)
            k = (k + 1) % len(M_F)
            pop[mask] = trial_pop[mask]

        target_n = max(int(self._params.get("min_population_size", 4)), int(round(initial_n - (initial_n - int(self._params.get("min_population_size", 4))) * progress)))
        if pop.shape[0] > target_n:
            keep = self._order(pop[:, -1])[:target_n]
            pop = pop[keep]
        return pop, n, {"M_F": M_F, "M_CR": M_CR, "k": k, "archive": archive, "initial_n": initial_n, "phase": phase + np.pi / 12.0}
