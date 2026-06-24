"""
SAPO - Surrogate-assisted Partial Optimization.

Standalone implementation following the released SAPO MATLAB code: LHS archive,
constraint-violation archive bookkeeping, alternating partial-optimization modes
(f,g_m -> g_n and g_m -> f), DE/rand/1 and DE/best/1 trial generation with
binomial crossover, reflection bound repair, cubic RBF models over objective and
constraints, and feasibility-rule selection.
"""
from __future__ import annotations

from typing import Any

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


class _SAPORBF:
    """MATLAB SAPO-style cubic RBF with input/output normalization."""

    def __init__(self, x: np.ndarray, y: np.ndarray, ridge: float = 1e-10) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        keep = np.unique(np.round(x, 14), axis=0, return_index=True)[1]
        keep.sort()
        x = x[keep]
        y = y[keep]
        self.xmin = x.min(axis=0)
        self.xmax = x.max(axis=0)
        self.ymin = y.min(axis=0)
        self.ymax = y.max(axis=0)
        xspan = self.xmax - self.xmin
        yspan = self.ymax - self.ymin
        xspan[xspan == 0.0] = 1.0
        yspan[yspan == 0.0] = 1.0
        self.xspan = xspan
        self.yspan = yspan
        xn = 2.0 * (x - self.xmin) / self.xspan - 1.0
        yn = 2.0 * (y - self.ymin) / self.yspan - 1.0
        n, d = xn.shape
        self.nodes = xn
        if n < d + 2:
            self.mode = "linear"
            A = np.c_[np.ones(n), xn]
            self.beta = np.linalg.pinv(A) @ yn
            self.alpha = np.zeros((n, yn.shape[1]))
            return
        self.mode = "rbf"
        r = np.linalg.norm(xn[:, None, :] - xn[None, :, :], axis=2)
        Phi = r ** 3
        P = np.c_[np.ones(n), xn]
        A = np.block([[Phi + ridge * np.eye(n), P], [P.T, np.zeros((d + 1, d + 1))]])
        b = np.vstack([yn, np.zeros((d + 1, yn.shape[1]))])
        theta = np.linalg.pinv(A) @ b
        self.alpha = theta[:n]
        self.beta = theta[n:]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        xn = 2.0 * (x - self.xmin) / self.xspan - 1.0
        P = np.c_[np.ones(xn.shape[0]), xn]
        if self.mode == "linear":
            yn = P @ self.beta
        else:
            r = np.linalg.norm(xn[:, None, :] - self.nodes[None, :, :], axis=2)
            yn = (r ** 3) @ self.alpha + P @ self.beta
        return (self.yspan / 2.0) * (yn + 1.0) + self.ymin


class SAPOEngine(BaseEngine):
    """Surrogate-assisted partial optimization for expensive constrained search."""

    algorithm_id = "sapo"
    algorithm_name = "Surrogate-Assisted Partial Optimization"
    family = "evolutionary"
    capabilities = CapabilityProfile(
        has_population=True,
        has_archive=True,
        supports_candidate_injection=True,
        supports_restart=False,
        supports_native_constraints=True,
        supports_framework_constraints=True,
    )
    _REFERENCE = {
        "doi": "10.1007/978-3-031-70068-2_24",
        "title": "A Surrogate-assisted Partial Optimization for Expensive Constrained Optimization Problems",
        "authors": "K. Nishihara and M. Nakata",
        "venue": "PPSN 2024, LNCS, pp. 391-407",
    }
    _DEFAULTS = {
        "population_size": 100,
        "F": 0.5,
        "CR": 0.9,
        "mut1": 1,
        "mut2": 3,
        "xov": 1,
        "initsize1": 100,
        "initsize2": 200,
        "change_data_n": 2,
        "ch_d_n_thres": 100,
        "data_n": 100,
        "data_n_2": 200,
        "data_times": 5,
        "kernel": "cubic",
    }
    _EVOMAPX_OPERATORS = (
        "sapo.lhs_initialization",
        "sapo.partial_selection_f_g_to_g",
        "sapo.partial_selection_g_to_f",
        "sapo.de_rand_1_binomial",
        "sapo.de_best_1_binomial",
        "sapo.reflection_bound_repair",
        "sapo.cubic_rbf_fit_predict",
        "sapo.feasibility_rule_selection",
        "sapo.expensive_evaluation_archive_update",
    )

    def __init__(self, problem: ProblemSpec, config: EngineConfig) -> None:
        super().__init__(problem, config)
        p = {**self._DEFAULTS, **(config.params or {})}
        self._n = max(4, int(p["population_size"]))
        self._F = float(p["F"])
        self._CR = float(p["CR"])
        self._mut1 = int(p["mut1"])
        self._mut2 = int(p["mut2"])
        self._xov = int(p["xov"])
        d = problem.dimension
        self._init_size = int(p["initsize1"] if d < 100 else p["initsize2"])
        self._change_data_n = int(p["change_data_n"])
        self._ch_d_n_thres = int(p["ch_d_n_thres"])
        self._data_n = int(p["data_n"])
        self._data_n_2 = int(p["data_n_2"])
        self._data_times = int(p["data_times"])
        self._rng = np.random.default_rng(config.seed)
        self._last_counts = {op: 0 for op in self._EVOMAPX_OPERATORS}
        self._last_contrib = {op: 0.0 for op in self._EVOMAPX_OPERATORS}

    def _lhs(self, n: int) -> np.ndarray:
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        d = self.problem.dimension
        u = np.empty((n, d), dtype=float)
        for j in range(d):
            u[:, j] = (self._rng.permutation(n) + self._rng.random(n)) / n
        return lo + u * (hi - lo)

    def _constraint_vector(self, x: np.ndarray) -> np.ndarray:
        if not self.problem.constraints:
            return np.zeros(1, dtype=float)
        vals: list[float] = []
        for con in self.problem.constraints:
            val = con(np.asarray(x, dtype=float).tolist())
            if isinstance(val, dict):
                arrs = []
                for key in ("ineq", "ineqs", "g"):
                    if key in val:
                        arrs.extend(np.atleast_1d(val[key]).astype(float).tolist())
                for key in ("eq", "eqs", "h"):
                    if key in val:
                        tol = self.problem._equality_tolerance()
                        arrs.extend((np.abs(np.atleast_1d(val[key]).astype(float)) - tol).tolist())
                vals.extend(arrs or [self.problem._normalize_constraint_value(val)])
            elif isinstance(val, (list, tuple, np.ndarray)):
                vals.extend(np.atleast_1d(val).astype(float).tolist())
            else:
                vals.append(float(val))
        return np.asarray(vals if vals else [0.0], dtype=float)

    def _evaluate_raw(self, x: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        x = self.problem.apply_variable_types(x)
        raw = float(self.problem.target_function(x.tolist()))
        c = self._constraint_vector(x)
        cv = float(np.maximum(0.0, c).sum())
        eff = self.problem.score_from_raw(raw, cv)
        return raw, c, cv, eff

    def _evaluate_rows(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y, c, cv, eff = [], [], [], []
        for row in np.atleast_2d(x):
            yi, ci, cvi, efi = self._evaluate_raw(row)
            y.append(yi)
            c.append(ci)
            cv.append(cvi)
            eff.append(efi)
        maxc = max(len(v) for v in c)
        cmat = np.zeros((len(c), maxc))
        for i, ci in enumerate(c):
            cmat[i, : len(ci)] = ci
        return np.asarray(y), cmat, np.asarray(cv), np.asarray(eff)

    def _repair_reflect(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        lo = np.asarray(self.problem.min_values, dtype=float)
        hi = np.asarray(self.problem.max_values, dtype=float)
        lower = x < lo
        upper = x > hi
        x[lower] = np.minimum(hi[np.where(lower)[1] if lower.ndim == 2 else lower], 2 * lo[np.where(lower)[1] if lower.ndim == 2 else lower] - x[lower])
        x[upper] = np.maximum(lo[np.where(upper)[1] if upper.ndim == 2 else upper], 2 * hi[np.where(upper)[1] if upper.ndim == 2 else upper] - x[upper])
        return np.clip(x, lo, hi)

    def _best_feasibility_rule(self, fit: np.ndarray, cv: np.ndarray) -> int:
        feas = np.where(cv <= self.problem._equality_tolerance())[0]
        if len(feas):
            sub = fit[feas]
            return int(feas[np.argmin(sub) if self.problem.objective == "min" else np.argmax(sub)])
        return int(np.argmin(cv))

    def _best_f2g(self, m: int, fit: np.ndarray, con: np.ndarray) -> int:
        feas_m = np.where(con[:, m] <= self.problem._equality_tolerance())[0]
        if len(feas_m):
            sub = fit[feas_m]
            return int(feas_m[np.argmin(sub) if self.problem.objective == "min" else np.argmax(sub)])
        return int(np.argmin(con[:, m]))

    def _train_size(self) -> int:
        d = self.problem.dimension
        if self._change_data_n == 0:
            return self._data_n
        if self._change_data_n == 1:
            return self._data_n_2 if d >= self._ch_d_n_thres else self._data_n
        return max(4, self._data_times * d)

    def _select_g4f(self, db_x: np.ndarray, db_y: np.ndarray, db_c: np.ndarray, db_cv: np.ndarray, db_feasible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ncon = db_c.shape[1]
        N = min(self._n, len(db_x))
        if not self.problem.constraints:
            idx = np.argsort(db_y)[:N]
            train = np.argsort(db_y)[: min(self._train_size(), len(db_y))]
            return db_x[idx], db_y[idx], db_cv[idx], db_x[train], db_y[train], db_c[train]
        bf_idx = self._best_feasibility_rule(db_y, db_cv)
        bf = db_y[bf_idx]
        ind_infea_better = np.where((~db_feasible) & (db_y <= bf))[0]
        ind_worse = np.setdiff1d(np.arange(len(db_x)), ind_infea_better, assume_unique=False)
        selected: list[int] = []
        train_selected: list[int] = []
        for m in range(ncon):
            feas = np.where(db_feasible)[0]
            feas = feas[np.argsort(db_y[feas])] if len(feas) else feas
            selected.extend(feas[: max(1, N // 2)].tolist())
            train_selected.extend(feas.tolist())
            for base, key in ((ind_infea_better, db_c[:, m]), (ind_worse, db_y)):
                part = base[db_c[base, m] <= 0]
                part = part[np.argsort(key[part])] if len(part) else part
                selected.extend(part.tolist())
                train_selected.extend(part.tolist())
            for base, key in ((ind_infea_better, db_c[:, m]), (ind_worse, db_y)):
                part = base[db_c[base, m] > 0]
                part = part[np.argsort(key[part])] if len(part) else part
                selected.extend(part.tolist())
                train_selected.extend(part.tolist())
        idx = np.array(list(dict.fromkeys(selected))[:N] or np.argsort(db_cv)[:N].tolist(), dtype=int)
        tidx = np.array(list(dict.fromkeys(train_selected))[: min(self._train_size(), len(db_x))] or idx.tolist(), dtype=int)
        return db_x[idx], db_y[idx], db_cv[idx], db_x[tidx], db_y[tidx], db_c[tidx]

    def _select_fg4g(self, m: int, db_x: np.ndarray, db_y: np.ndarray, db_c: np.ndarray, db_cv: np.ndarray, db_feasible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ncon = db_c.shape[1]
        N = min(self._n, len(db_x))
        if not self.problem.constraints:
            idx = np.argsort(db_y)[:N]
            train = np.argsort(db_y)[: min(self._train_size(), len(db_y))]
            return db_x[idx], db_y[idx], db_c[idx], db_x[train], db_y[train], db_c[train]
        bf_idx = self._best_feasibility_rule(db_y, db_cv)
        bf = db_y[bf_idx]
        ind_infea_better = np.where((~db_feasible) & (db_y <= bf))[0]
        ind_worse = np.setdiff1d(np.arange(len(db_x)), ind_infea_better, assume_unique=False)
        selected: list[int] = []
        p = ind_infea_better[db_c[ind_infea_better, m] > 0]
        selected.extend(p[np.argsort(db_c[p, m])].tolist() if len(p) else [])
        p = ind_worse[db_c[ind_worse, m] > 0]
        selected.extend(p[np.argsort(db_y[p])].tolist() if len(p) else [])
        for g in range(ncon):
            if g == m:
                continue
            p = ind_infea_better[(db_c[ind_infea_better, g] <= 0) & (db_c[ind_infea_better, m] > 0)]
            selected.extend(p[np.argsort(db_c[p, m])].tolist() if len(p) else [])
            p = ind_worse[(db_c[ind_worse, g] <= 0) & (db_c[ind_worse, m] > 0)]
            selected.extend(p[np.argsort(db_y[p])].tolist() if len(p) else [])
            p = np.where((db_c[:, g] > 0) & (db_c[:, m] > 0))[0]
            selected.extend(p[np.argsort(db_c[p, g])].tolist() if len(p) else [])
        idx = np.array(list(dict.fromkeys(selected))[:N] or np.argsort(db_cv)[:N].tolist(), dtype=int)
        tidx = np.array(list(dict.fromkeys(selected))[: min(self._train_size(), len(db_x))] or idx.tolist(), dtype=int)
        return db_x[idx], db_y[idx], db_c[idx], db_x[tidx], db_y[tidx], db_c[tidx]

    def _de_trial(self, pop: np.ndarray, fit: np.ndarray | None, cv: np.ndarray | None, mut: int, bid: int | None) -> np.ndarray:
        pop = np.asarray(pop, dtype=float)
        N, D = pop.shape
        if N < 4:
            return pop.copy()
        if bid is None:
            bid = self._best_feasibility_rule(fit if fit is not None else np.zeros(N), cv if cv is not None else np.zeros(N))
        bx = pop[int(bid)]
        v = np.empty_like(pop)
        for i in range(N):
            pool = [j for j in range(N) if j != i and (mut not in {3, 4, 6, 8} or j != bid)]
            if mut == 1 or len(pool) < 2:
                r = self._rng.choice([j for j in range(N) if j != i], 3, replace=False)
                v[i] = pop[r[0]] + self._F * (pop[r[1]] - pop[r[2]])
            elif mut == 3:
                r = self._rng.choice(pool, 2, replace=False)
                v[i] = bx + self._F * (pop[r[0]] - pop[r[1]])
            elif mut == 6:
                r = self._rng.choice(pool, 2, replace=False)
                v[i] = pop[i] + self._F * (bx - pop[i]) + self._F * (pop[r[0]] - pop[r[1]])
            else:
                r = self._rng.choice([j for j in range(N) if j != i], 3, replace=False)
                v[i] = pop[r[0]] + self._F * (pop[r[1]] - pop[r[2]])
        if self._xov == 1:
            mask = self._rng.random((N, D)) <= self._CR
            mask[np.arange(N), self._rng.integers(0, D, size=N)] = True
            return np.where(mask, v, pop)
        return v

    def _blank_counts(self) -> dict[str, int]:
        return {op: 0 for op in self._EVOMAPX_OPERATORS}

    def _blank_contrib(self) -> dict[str, float]:
        return {op: 0.0 for op in self._EVOMAPX_OPERATORS}

    def initialize(self) -> EngineState:
        x = self._lhs(self._init_size)
        y, c, cv, eff = self._evaluate_rows(x)
        db = np.c_[x, eff]
        bi = self._best_feasibility_rule(y, cv)
        counts = self._blank_counts()
        counts["sapo.lhs_initialization"] = self._init_size
        contrib = self._blank_contrib()
        self._last_counts = counts
        self._last_contrib = contrib
        return EngineState(
            step=0,
            evaluations=self._init_size,
            best_position=x[bi].tolist(),
            best_fitness=float(eff[bi]),
            initialized=True,
            payload={
                "population": db.copy(),
                "db_x": x,
                "db_y": y,
                "db_c": c,
                "db_cv": cv,
                "db_eff": eff,
                "gen": 1,
                "operator_counts": counts,
                "operator_contributions": contrib,
                "last_operator_counts": counts,
                "last_operator_contributions": contrib,
            },
        )

    def step(self, state: EngineState) -> EngineState:
        db_x = np.asarray(state.payload["db_x"], dtype=float).copy()
        db_y = np.asarray(state.payload["db_y"], dtype=float).copy()
        db_c = np.asarray(state.payload["db_c"], dtype=float).copy()
        db_cv = np.asarray(state.payload["db_cv"], dtype=float).copy()
        db_eff = np.asarray(state.payload["db_eff"], dtype=float).copy()
        gen = int(state.payload.get("gen", 1))
        counts = self._blank_counts()
        contrib = self._blank_contrib()
        db_feasible = db_cv <= self.problem._equality_tolerance()
        ncon = db_c.shape[1]
        old_best = float(state.best_fitness)

        for m in range(ncon):
            if gen % 2 == 1:
                pop, fit, con, train_x, train_y, train_c = self._select_fg4g(m, db_x, db_y, db_c, db_cv, db_feasible)
                counts["sapo.partial_selection_f_g_to_g"] += 1
                if len(pop) < 4:
                    continue
                bid = self._best_f2g(m, fit, con)
                cand1 = self._de_trial(pop, None, None, self._mut1, bid)
                cand2 = self._de_trial(pop, None, None, self._mut2, bid)
                counts["sapo.de_rand_1_binomial"] += len(cand1) if self._mut1 == 1 else 0
                counts["sapo.de_best_1_binomial"] += len(cand2) if self._mut2 == 3 else 0
                cand = self._repair_reflect(np.vstack([cand1, cand2]))
                counts["sapo.reflection_bound_repair"] += len(cand)
                model = _SAPORBF(train_x, np.c_[train_y, train_c])
                pred = model.predict(cand)
                counts["sapo.cubic_rbf_fit_predict"] += 1
                chosen = self._best_f2g(min(m, pred.shape[1] - 2), pred[:, 0], pred[:, 1:])
            else:
                pop, fit, cv, train_x, train_y, train_c = self._select_g4f(db_x, db_y, db_c, db_cv, db_feasible)
                counts["sapo.partial_selection_g_to_f"] += 1
                if len(pop) < 4:
                    continue
                bid = self._best_feasibility_rule(fit, cv)
                cand1 = self._de_trial(pop, fit, cv, self._mut1, bid)
                cand2 = self._de_trial(pop, fit, cv, self._mut2, bid)
                counts["sapo.de_rand_1_binomial"] += len(cand1) if self._mut1 == 1 else 0
                counts["sapo.de_best_1_binomial"] += len(cand2) if self._mut2 == 3 else 0
                cand = self._repair_reflect(np.vstack([cand1, cand2]))
                counts["sapo.reflection_bound_repair"] += len(cand)
                model = _SAPORBF(train_x, np.c_[train_y, train_c])
                pred = model.predict(cand)
                counts["sapo.cubic_rbf_fit_predict"] += 1
                pc = pred[:, 1:]
                pcv = np.maximum(0.0, pc).sum(axis=1)
                chosen = self._best_feasibility_rule(pred[:, 0], pcv)
            counts["sapo.feasibility_rule_selection"] += 1
            xnew = cand[int(chosen)].copy()
            ynew, cnew, cvnew, effnew = self._evaluate_rows(xnew.reshape(1, -1))
            state.evaluations += 1
            db_x = np.vstack([db_x, xnew])
            db_y = np.r_[db_y, ynew[0]]
            if cnew.shape[1] != db_c.shape[1]:
                width = max(cnew.shape[1], db_c.shape[1])
                tmp = np.zeros((db_c.shape[0], width)); tmp[:, :db_c.shape[1]] = db_c; db_c = tmp
                tmpn = np.zeros((cnew.shape[0], width)); tmpn[:, :cnew.shape[1]] = cnew; cnew = tmpn
            db_c = np.vstack([db_c, cnew[0]])
            db_cv = np.r_[db_cv, cvnew[0]]
            db_eff = np.r_[db_eff, effnew[0]]
            counts["sapo.expensive_evaluation_archive_update"] += 1
            if self.problem.is_better(float(effnew[0]), state.best_fitness):
                contrib["sapo.expensive_evaluation_archive_update"] += abs(float(state.best_fitness) - float(effnew[0]))
                state.best_fitness = float(effnew[0])
                state.best_position = xnew.tolist()

        order = np.argsort(db_eff)[: min(self._n, len(db_eff))]
        population = np.c_[db_x[order], db_eff[order]]
        prev_counts = state.payload.get("operator_counts", {})
        prev_contrib = state.payload.get("operator_contributions", {})
        state.payload = {
            "population": population,
            "db_x": db_x,
            "db_y": db_y,
            "db_c": db_c,
            "db_cv": db_cv,
            "db_eff": db_eff,
            "gen": gen + 1,
            "operator_counts": {k: int(prev_counts.get(k, 0)) + int(v) for k, v in counts.items()},
            "operator_contributions": {k: float(prev_contrib.get(k, 0.0)) + float(v) for k, v in contrib.items()},
            "last_operator_counts": counts,
            "last_operator_contributions": contrib,
        }
        self._last_counts = counts
        self._last_contrib = contrib
        if self.problem.is_better(state.best_fitness, old_best):
            pass
        state.step += 1
        return state

    def observe(self, state: EngineState) -> dict[str, Any]:
        return {
            "step": state.step,
            "evaluations": state.evaluations,
            "best_fitness": state.best_fitness,
            "archive_size": int(len(state.payload.get("db_eff", []))),
            "operator_counts": dict(state.payload.get("last_operator_counts", self._last_counts)),
            "operator_contributions": dict(state.payload.get("last_operator_contributions", self._last_contrib)),
            "operator_contributions_total": dict(state.payload.get("operator_contributions", {})),
            "evomapx_delta_f": "direct_improvement",
            "evomapx_fidelity": "native_matlab_reference",
            "native_evomapx_operator_labels": True,
        }

    def get_best_candidate(self, state: EngineState) -> CandidateRecord:
        return CandidateRecord(list(state.best_position), float(state.best_fitness), self.algorithm_id, state.step, "best")

    def finalize(self, state: EngineState) -> OptimizationResult:
        return OptimizationResult(
            algorithm_id=self.algorithm_id,
            best_position=list(state.best_position),
            best_fitness=float(state.best_fitness),
            steps=int(state.step),
            evaluations=int(state.evaluations),
            termination_reason=state.termination_reason,
            capabilities=self.capabilities,
            metadata={
                "algorithm_name": self.algorithm_name,
                "elapsed_time": state.elapsed_time,
                "reference": dict(self._REFERENCE),
                "defaults": dict(self._DEFAULTS),
                "operator_counts": dict(state.payload.get("operator_counts", {})),
                "operator_contributions": dict(state.payload.get("operator_contributions", {})),
            },
        )

    def get_population(self, state: EngineState) -> list[CandidateRecord]:
        pop = np.asarray(state.payload["population"])
        return [
            CandidateRecord(pop[i, :-1].tolist(), float(pop[i, -1]), self.algorithm_id, state.step, "current")
            for i in range(pop.shape[0])
        ]


__all__ = ["SAPOEngine"]
