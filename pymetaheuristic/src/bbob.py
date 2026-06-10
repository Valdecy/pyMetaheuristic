"""COCO/BBOB noiseless single-objective benchmark functions.

This module provides the 24 noiseless BBOB functions as pure NumPy callables.
The implementation follows the public COCO/BBOB definitions closely enough for
algorithm benchmarking without requiring the external COCO C extension or input
files.  Each function is minimised on the usual BBOB region of interest
``[-5, 5]^D`` and supports deterministic COCO-style instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray

BBOB_DEFAULT_INSTANCE = 1
BBOB_DEFAULT_DIMENSION = 2
BBOB_BOUNDS = (-5.0, 5.0)
BBOB_FUNCTION_IDS = tuple(range(1, 25))

BBOB_NAMES: Dict[int, str] = {
    1: "Sphere",
    2: "Ellipsoidal",
    3: "Rastrigin",
    4: "Bueche-Rastrigin",
    5: "Linear Slope",
    6: "Attractive Sector",
    7: "Step Ellipsoidal",
    8: "Rosenbrock",
    9: "Rosenbrock Rotated",
    10: "Ellipsoidal Rotated",
    11: "Discus",
    12: "Bent Cigar",
    13: "Sharp Ridge",
    14: "Different Powers",
    15: "Rastrigin Rotated",
    16: "Weierstrass",
    17: "Schaffers F7, condition 10",
    18: "Schaffers F7, condition 1000",
    19: "Griewank-Rosenbrock",
    20: "Schwefel",
    21: "Gallagher 101 Peaks",
    22: "Gallagher 21 Peaks",
    23: "Katsuura",
    24: "Lunacek bi-Rastrigin",
}

BBOB_TYPES: Dict[int, str] = {
    1: "separable",
    2: "separable ill-conditioned",
    3: "separable multi-modal",
    4: "separable multi-modal",
    5: "separable linear",
    6: "moderate-conditioning",
    7: "moderate-conditioning step",
    8: "moderate-conditioning valley",
    9: "moderate-conditioning rotated valley",
    10: "ill-conditioned",
    11: "ill-conditioned",
    12: "ill-conditioned",
    13: "ill-conditioned ridge",
    14: "ill-conditioned smooth",
    15: "multi-modal rotated",
    16: "multi-modal rugged",
    17: "multi-modal weak global structure",
    18: "multi-modal weak global structure",
    19: "multi-modal weak global structure",
    20: "weak global structure",
    21: "weak global structure",
    22: "weak global structure",
    23: "weak global structure",
    24: "weak global structure",
}


def _asarray(x: ArrayLike | None, default_dimension: int = BBOB_DEFAULT_DIMENSION) -> np.ndarray:
    if x is None:
        x = np.zeros(int(default_dimension), dtype=float)
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError("BBOB functions require dimension D >= 2.")
    return arr


def _round_half_up(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5)


def _unif(n: int, seed: int) -> np.ndarray:
    """Legacy BBOB uniform generator used for instances."""
    n = int(n)
    seed = int(seed)
    if seed < 0:
        seed = -seed
    if seed < 1:
        seed = 1
    aktseed = seed
    rgrand = [0] * 32
    for i in range(39, -1, -1):
        tmp = int(np.floor(aktseed / 127773.0))
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp
        if aktseed < 0:
            aktseed += 2147483647
        if i < 32:
            rgrand[i] = aktseed
    aktrand = rgrand[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        tmp = int(np.floor(aktseed / 127773.0))
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp
        if aktseed < 0:
            aktseed += 2147483647
        tmp = int(np.floor(aktrand / 67108865.0))
        aktrand = rgrand[tmp]
        rgrand[tmp] = aktseed
        out[i] = aktrand / 2.147483647e9
        if out[i] == 0.0:
            out[i] = 1e-99
    return out


def _gauss(n: int, seed: int) -> np.ndarray:
    u = _unif(2 * int(n), int(seed))
    g = np.sqrt(-2.0 * np.log(u[:n])) * np.cos(2.0 * np.pi * u[n:])
    g[g == 0.0] = 1e-99
    return g


def _compute_rotation(seed: int, dimension: int) -> np.ndarray:
    d = int(dimension)
    gvect = _gauss(d * d, int(seed))
    b = np.empty((d, d), dtype=float)
    for i in range(d):
        for j in range(d):
            b[i, j] = gvect[j * d + i]
    for i in range(d):
        for j in range(i):
            prod = float(np.dot(b[:, i], b[:, j]))
            b[:, i] -= prod * b[:, j]
        norm = float(np.linalg.norm(b[:, i]))
        if norm == 0.0:
            b[:, i] = 0.0
            b[i, i] = 1.0
        else:
            b[:, i] /= norm
    return b


def _compute_xopt(seed: int, dimension: int) -> np.ndarray:
    xopt = _unif(int(dimension), int(seed))
    xopt = 8.0 * np.floor(1e4 * xopt) / 1e4 - 4.0
    xopt[xopt == 0.0] = -1e-5
    return xopt.astype(float)


def _compute_fopt(function_id: int, instance: int) -> float:
    function_id = int(function_id)
    instance = int(instance)
    if function_id == 4:
        rseed = 3
    elif function_id == 18:
        rseed = 17
    else:
        rseed = function_id
    rrseed = rseed + 10000 * instance
    gval = float(_gauss(1, rrseed)[0])
    gval2 = float(_gauss(1, rrseed + 1)[0])
    return float(min(1000.0, max(-1000.0, np.floor(100.0 * 100.0 * gval / gval2 + 0.5) / 100.0)))


def _conditioning(x: np.ndarray, alpha: float) -> np.ndarray:
    d = x.size
    exponents = 0.5 * np.arange(d, dtype=float) / float(d - 1)
    return np.power(float(alpha), exponents) * x


def _asymmetric(x: np.ndarray, beta: float) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    d = y.size
    pos = y > 0.0
    if np.any(pos):
        idx = np.arange(d, dtype=float)
        exponents = 1.0 + (float(beta) * idx / float(d - 1)) * np.sqrt(np.maximum(y, 0.0))
        y[pos] = np.power(y[pos], exponents[pos])
    return y


def _oscillate(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    pos = x > 0.0
    neg = x < 0.0
    if np.any(pos):
        tmp = np.log(x[pos]) / 0.1
        base = np.exp(tmp + 0.49 * (np.sin(tmp) + np.sin(0.79 * tmp)))
        y[pos] = np.power(base, 0.1)
    if np.any(neg):
        tmp = np.log(-x[neg]) / 0.1
        base = np.exp(tmp + 0.49 * (np.sin(0.55 * tmp) + np.sin(0.31 * tmp)))
        y[neg] = -np.power(base, 0.1)
    return y


def _brs(x: np.ndarray) -> np.ndarray:
    d = x.size
    factors = np.power(np.sqrt(10.0), np.arange(d, dtype=float) / float(d - 1))
    y = factors * x
    even = (np.arange(d) % 2 == 0) & (x > 0.0)
    y[even] *= 10.0
    return y


def _affine(x: np.ndarray, m: np.ndarray, b: np.ndarray | float = 0.0) -> np.ndarray:
    return np.asarray(m, dtype=float).dot(x) + b


def _boundary_penalty(x: np.ndarray, factor: float = 1.0, bound: float = 5.0) -> float:
    excess = np.maximum(np.abs(np.asarray(x, dtype=float)) - float(bound), 0.0)
    return float(factor) * float(np.sum(excess * excess))


def _obj_oscillate(y: float) -> float:
    y = float(y)
    if y > 0.0:
        log_y = np.log(y) / 0.1
        return float(np.power(np.exp(log_y + 0.49 * (np.sin(log_y) + np.sin(0.79 * log_y))), 0.1))
    if y < 0.0:
        log_y = np.log(-y) / 0.1
        return float(-np.power(np.exp(log_y + 0.49 * (np.sin(0.55 * log_y) + np.sin(0.31 * log_y))), 0.1))
    return 0.0


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def _ellipsoid(x: np.ndarray, conditioning: float = 1.0e6) -> float:
    d = x.size
    weights = np.power(float(conditioning), np.arange(d, dtype=float) / float(d - 1))
    return float(np.sum(weights * x * x))


def _rastrigin(x: np.ndarray) -> float:
    return float(10.0 * (x.size - np.sum(np.cos(2.0 * np.pi * x))) + np.sum(x * x))


def _linear_slope(x: np.ndarray, best_parameter: np.ndarray) -> float:
    d = x.size
    result = 0.0
    for i in range(d):
        si = np.power(np.sqrt(100.0), i / float(d - 1))
        if best_parameter[i] < 0.0:
            si = -si
        if x[i] * best_parameter[i] < 25.0:
            result += 5.0 * abs(si) - si * x[i]
        else:
            result += 5.0 * abs(si) - si * best_parameter[i]
    return float(result)


def _attractive_sector(x: np.ndarray, xopt: np.ndarray) -> float:
    weights = np.where(xopt * x > 0.0, 10000.0, 1.0)
    return float(np.sum(weights * x * x))


def _step_ellipsoid(x: np.ndarray, xopt: np.ndarray, rot1: np.ndarray, rot2: np.ndarray, penalty_scale: float = 1.0) -> float:
    d = x.size
    penalty = _boundary_penalty(x, penalty_scale)
    c1 = np.sqrt(np.power(100.0 / 10.0, np.arange(d, dtype=float) / float(d - 1)))
    z = c1 * rot2.dot(x - xopt)
    z1 = float(z[0])
    z = np.where(np.abs(z) > 0.5, _round_half_up(z), _round_half_up(10.0 * z) / 10.0)
    zz = rot1.dot(z)
    weights = np.power(100.0, np.arange(d, dtype=float) / float(d - 1))
    result = float(np.sum(weights * zz * zz))
    return float(0.1 * max(abs(z1) * 1.0e-4, result) + penalty)


def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[:-1] * x[:-1] - x[1:]) ** 2 + (x[:-1] - 1.0) ** 2))


def _discus(x: np.ndarray) -> float:
    return float(1.0e6 * x[0] * x[0] + np.sum(x[1:] * x[1:]))


def _bent_cigar(x: np.ndarray) -> float:
    return float(x[0] * x[0] + 1.0e6 * np.sum(x[1:] * x[1:]))


def _sharp_ridge(x: np.ndarray) -> float:
    return float(x[0] * x[0] + 100.0 * np.sqrt(np.sum(x[1:] * x[1:])))


def _different_powers(x: np.ndarray) -> float:
    d = x.size
    exponents = 2.0 + 4.0 * np.arange(d, dtype=float) / float(d - 1)
    return float(np.sqrt(np.sum(np.power(np.abs(x), exponents))))


def _weierstrass(x: np.ndarray) -> float:
    k = np.arange(12, dtype=float)
    ak = np.power(0.5, k)
    bk = np.power(3.0, k)
    f0 = float(np.sum(ak * np.cos(np.pi * bk)))
    total = float(np.sum(np.cos(2.0 * np.pi * (x[:, None] + 0.5) * bk[None, :]) * ak[None, :]))
    return float(10.0 * np.power(total / x.size - f0, 3.0))


def _schaffers_f7(x: np.ndarray) -> float:
    tmp = x[:-1] * x[:-1] + x[1:] * x[1:]
    terms = np.power(tmp, 0.25) * (1.0 + np.sin(50.0 * np.power(tmp, 0.1)) ** 2)
    return float(np.power(np.sum(terms) / float(x.size - 1), 2.0))


def _griewank_rosenbrock(x: np.ndarray, facftrue: float = 10.0) -> float:
    c1 = x[:-1] * x[:-1] - x[1:]
    c2 = 1.0 - x[:-1]
    tmp = 100.0 * c1 * c1 + c2 * c2
    return float(facftrue + facftrue * np.sum(tmp / 4000.0 - np.cos(tmp)) / float(x.size - 1))


def _schwefel_raw(x: np.ndarray) -> float:
    penalty = np.sum(np.maximum(np.abs(x) - 500.0, 0.0) ** 2)
    return float(0.01 * (penalty + 418.9828872724339 - np.sum(x * np.sin(np.sqrt(np.abs(x)))) / x.size))


def _katsuura(x: np.ndarray) -> float:
    d = x.size
    result = 1.0
    for i in range(d):
        js = np.arange(1, 33, dtype=float)
        p = np.power(2.0, js)
        tmp = np.sum(np.abs(p * x[i] - _round_half_up(p * x[i])) / p)
        result *= np.power(1.0 + (i + 1.0) * tmp, 10.0 / np.power(float(d), 1.2))
    return float(10.0 / (d * d) * (-1.0 + result))


def _gallagher(x: np.ndarray, rotation: np.ndarray, x_local: np.ndarray, arr_scales: np.ndarray, peak_values: np.ndarray, penalty_scale: float) -> float:
    d = x.size
    f_pen = _boundary_penalty(x, penalty_scale)
    tmx = rotation.dot(x)
    fac = -0.5 / float(d)
    f = 0.0
    for i in range(peak_values.size):
        diff = tmx - x_local[:, i]
        tmp2 = float(np.sum(arr_scales[i, :] * diff * diff))
        f = max(f, float(peak_values[i] * np.exp(fac * tmp2)))
    y = 10.0 - f
    f_true = _obj_oscillate(y)
    return float(f_true * f_true + f_pen)


def _lunacek_raw(x: np.ndarray, xopt: np.ndarray, rot1: np.ndarray, rot2: np.ndarray) -> float:
    d = x.size
    mu0 = 2.5
    delta = 1.0
    s = 1.0 - 0.5 / (np.sqrt(float(d + 20)) - 4.1)
    mu1 = -np.sqrt((mu0 * mu0 - delta) / s)
    penalty = _boundary_penalty(x, 1.0)
    x_hat = 2.0 * x.copy()
    x_hat[xopt < 0.0] *= -1.0
    c = np.power(np.sqrt(100.0), np.arange(d, dtype=float) / float(d - 1))
    tmpvect = c * rot2.dot(x_hat - mu0)
    z = rot1.dot(tmpvect)
    sum1 = float(np.sum((x_hat - mu0) ** 2))
    sum2 = float(np.sum((x_hat - mu1) ** 2))
    sum3 = float(np.sum(np.cos(2.0 * np.pi * z)))
    return float(min(sum1, delta * d + s * sum2) + 10.0 * (d - sum3) + 1.0e4 * penalty)


def _matrix_two_rotations(rot1: np.ndarray, rot2: np.ndarray, condition: float) -> np.ndarray:
    d = rot1.shape[0]
    exponents = np.arange(d, dtype=float) / float(d - 1)
    diag = np.power(np.sqrt(float(condition)), exponents)
    return rot1.dot(np.diag(diag)).dot(rot2)


@dataclass(frozen=True)
class BBOBFunction:
    """Callable deterministic BBOB problem instance."""

    function_id: int
    dimension: int = BBOB_DEFAULT_DIMENSION
    instance: int = BBOB_DEFAULT_INSTANCE

    def __post_init__(self) -> None:
        if int(self.function_id) not in BBOB_FUNCTION_IDS:
            raise ValueError("function_id must be in {1, ..., 24}.")
        if int(self.dimension) < 2:
            raise ValueError("BBOB functions require dimension D >= 2.")
        object.__setattr__(self, "function_id", int(self.function_id))
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "instance", int(self.instance))

    @property
    def name(self) -> str:
        return f"bbob_f{self.function_id:02d}"

    @property
    def display_name(self) -> str:
        return BBOB_NAMES[self.function_id]

    @property
    def lower(self) -> list[float]:
        return [BBOB_BOUNDS[0]] * self.dimension

    @property
    def upper(self) -> list[float]:
        return [BBOB_BOUNDS[1]] * self.dimension

    @property
    def rseed(self) -> int:
        if self.function_id == 4:
            return 3 + 10000 * self.instance
        if self.function_id == 18:
            return 17 + 10000 * self.instance
        return self.function_id + 10000 * self.instance

    @property
    def fopt(self) -> float:
        return _compute_fopt(self.function_id, self.instance)

    @property
    def xopt(self) -> np.ndarray:
        fid = self.function_id
        d = self.dimension
        rs = self.rseed
        if fid == 5:
            raw = _compute_xopt(rs, d)
            return np.where(raw < 0.0, -5.0, 5.0)
        if fid == 8:
            return 0.75 * _compute_xopt(rs, d)
        if fid == 9:
            rot = _compute_rotation(rs, d)
            factor = max(1.0, np.sqrt(float(d)) / 8.0)
            return np.sum(rot, axis=0) / (2.0 * factor)
        if fid == 19:
            rot = _compute_rotation(rs, d)
            scale = max(1.0, np.sqrt(float(d)) / 8.0)
            rot_scaled = scale * rot
            return np.sum(rot_scaled, axis=0) / (2.0 * scale)
        if fid == 20:
            signs = np.where(_unif(d, rs) < 0.5, -1.0, 1.0)
            return signs * 0.5 * 4.2096874637
        if fid == 21:
            data = self._gallagher_data(101)
            return data[4].copy()
        if fid == 22:
            data = self._gallagher_data(21)
            return data[4].copy()
        if fid == 24:
            g = _gauss(d, rs)
            return np.where(g < 0.0, -0.5 * 2.5, 0.5 * 2.5)
        return _compute_xopt(rs, d)

    def _rot1(self) -> np.ndarray:
        return _compute_rotation(self.rseed + 1000000, self.dimension)

    def _rot2(self) -> np.ndarray:
        return _compute_rotation(self.rseed, self.dimension)

    def _gallagher_data(self, number_of_peaks: int):
        return _gallagher_data(self.rseed, self.dimension, int(number_of_peaks))

    def __call__(self, variables_values: ArrayLike | None = None) -> float:
        x = _asarray(variables_values, self.dimension)
        if x.size != self.dimension:
            raise ValueError(f"{self.name} expects D={self.dimension}. Got D={x.size}.")
        return float(_evaluate_bbob_prepared(x, self.function_id, self.dimension, self.instance, self.rseed, self.fopt))


@lru_cache(maxsize=None)
def _gallagher_data(rseed: int, dimension: int, number_of_peaks: int):
    d = int(dimension)
    p = int(number_of_peaks)
    if p == 101:
        maxcondition1 = np.sqrt(1000.0)
        b, c = 10.0, 5.0
    elif p == 21:
        maxcondition1 = 1000.0
        b, c = 9.8, 4.9
    else:
        raise ValueError("Gallagher BBOB functions only support 21 or 101 peaks.")
    rotation = _compute_rotation(int(rseed), d)
    random_numbers = _unif(p - 1, int(rseed))
    order = np.argsort(random_numbers, kind="mergesort")
    arr_condition = np.empty(p, dtype=float)
    peak_values = np.empty(p, dtype=float)
    arr_condition[0] = maxcondition1
    peak_values[0] = 10.0
    for i in range(1, p):
        idx = order[i - 1]
        arr_condition[i] = np.power(1000.0, idx / float(p - 2))
        peak_values[i] = (i - 1) / float(p - 2) * (9.1 - 1.1) + 1.1
    arr_scales = np.empty((p, d), dtype=float)
    for i in range(p):
        rn = _unif(d, int(rseed) + 1000 * i)
        perm = np.argsort(rn, kind="mergesort")
        inv = np.empty(d, dtype=int)
        inv[perm] = np.arange(d)
        for j in range(d):
            arr_scales[i, j] = np.power(arr_condition[i], inv[j] / float(d - 1) - 0.5)
    rnd = _unif(d * p, int(rseed))
    x_local = np.empty((d, p), dtype=float)
    best_parameter = 0.8 * (b * rnd[:d] - c)
    for i in range(d):
        for j in range(p):
            value = 0.0
            for k in range(d):
                value += rotation[i, k] * (b * rnd[j * d + k] - c)
            if j == 0:
                value *= 0.8
            x_local[i, j] = value
    return rotation, x_local, arr_scales, peak_values, best_parameter


@lru_cache(maxsize=None)
def _prepared(function_id: int, dimension: int, instance: int):
    problem = BBOBFunction(function_id=function_id, dimension=dimension, instance=instance)
    fid, d, rs = problem.function_id, problem.dimension, problem.rseed
    rot1 = _compute_rotation(rs + 1000000, d)
    rot2 = _compute_rotation(rs, d)
    xopt = problem.xopt
    return problem, xopt, rot1, rot2


def _evaluate_bbob_prepared(x: np.ndarray, fid: int, d: int, instance: int, rseed: int, fopt: float) -> float:
    problem, xopt, rot1, rot2 = _prepared(int(fid), int(d), int(instance))

    if fid == 1:
        return _sphere(x - xopt) + fopt
    if fid == 2:
        z = _oscillate(x - xopt)
        return _ellipsoid(z, 1.0e6) + fopt
    if fid == 3:
        z = _conditioning(_asymmetric(_oscillate(x - xopt), 0.2), 10.0)
        return _rastrigin(z) + fopt
    if fid == 4:
        z = _brs(_oscillate(x - xopt))
        return _rastrigin(z) + fopt
    if fid == 5:
        best = xopt
        return _linear_slope(x, best) + fopt
    if fid == 6:
        m = _matrix_two_rotations(rot1, rot2, 10.0)
        z = m.dot(x - xopt)
        y = _attractive_sector(z, xopt)
        return np.power(_obj_oscillate(y), 0.9) + fopt
    if fid == 7:
        return _step_ellipsoid(x, xopt, rot1, rot2, 1.0) + fopt
    if fid == 8:
        factor = max(1.0, np.sqrt(float(d)) / 8.0)
        z = factor * (x - xopt) + 1.0
        return _rosenbrock(z) + fopt
    if fid == 9:
        factor = max(1.0, np.sqrt(float(d)) / 8.0)
        z = factor * rot2.dot(x) + 0.5
        return _rosenbrock(z) + fopt
    if fid == 10:
        z = _oscillate(rot1.dot(x - xopt))
        return _ellipsoid(z, 1.0e6) + fopt
    if fid == 11:
        z = _oscillate(rot1.dot(x - xopt))
        return _discus(z) + fopt
    if fid == 12:
        z = _asymmetric(rot1.dot(x - xopt), 0.5)
        m = _matrix_two_rotations(rot1, rot2, 100.0)
        z = m.dot(z)
        return _bent_cigar(z) + fopt
    if fid == 13:
        m = _matrix_two_rotations(rot1, rot2, 10.0)
        z = m.dot(x - xopt)
        return _sharp_ridge(z) + fopt
    if fid == 14:
        z = rot1.dot(x - xopt)
        return _different_powers(z) + fopt
    if fid == 15:
        m = _matrix_two_rotations(rot1, rot2, 10.0)
        z = rot1.dot(x - xopt)
        z = _oscillate(z)
        z = _asymmetric(z, 0.2)
        z = m.dot(z)
        return _rastrigin(z) + fopt
    if fid == 16:
        m = _matrix_two_rotations(rot1, rot2, 1.0 / 100.0)
        z = rot1.dot(x - xopt)
        z = _oscillate(z)
        z = m.dot(z)
        return _weierstrass(z) + fopt + _boundary_penalty(x, 10.0 / d)
    if fid == 17:
        m = np.diag(np.power(np.sqrt(10.0), np.arange(d, dtype=float) / float(d - 1))).dot(rot2)
        z = rot1.dot(x - xopt)
        z = _asymmetric(z, 0.5)
        z = m.dot(z)
        return _schaffers_f7(z) + fopt + _boundary_penalty(x, 10.0)
    if fid == 18:
        m = np.diag(np.power(np.sqrt(1000.0), np.arange(d, dtype=float) / float(d - 1))).dot(rot2)
        z = rot1.dot(x - xopt)
        z = _asymmetric(z, 0.5)
        z = m.dot(z)
        return _schaffers_f7(z) + fopt + _boundary_penalty(x, 10.0)
    if fid == 19:
        scale = max(1.0, np.sqrt(float(d)) / 8.0)
        z = scale * rot2.dot(x) + 0.5
        return _griewank_rosenbrock(z, 10.0) + fopt
    if fid == 20:
        signs = np.where(_unif(d, rseed) < 0.5, -1.0, 1.0)
        xopt20 = signs * 0.5 * 4.2096874637
        x_hat = np.where(_unif(d, rseed) < 0.5, -x, x)
        z = 2.0 * x_hat
        zhat = z.copy()
        for i in range(1, d):
            zhat[i] = z[i] + 0.25 * (z[i - 1] - 2.0 * abs(xopt20[i - 1]))
        z = zhat - 2.0 * np.abs(xopt20)
        z = _conditioning(z, 10.0)
        z = z - (-2.0 * np.abs(xopt20))
        z = 100.0 * z
        return _schwefel_raw(z) + fopt
    if fid == 21:
        rotation, x_local, arr_scales, peak_values, _best = problem._gallagher_data(101)
        return _gallagher(x, rotation, x_local, arr_scales, peak_values, 1.0) + fopt
    if fid == 22:
        rotation, x_local, arr_scales, peak_values, _best = problem._gallagher_data(21)
        return _gallagher(x, rotation, x_local, arr_scales, peak_values, 1.0) + fopt
    if fid == 23:
        m = _matrix_two_rotations(rot1, rot2, 100.0)
        z = m.dot(x - xopt)
        return _katsuura(z) + fopt + _boundary_penalty(x, 1.0)
    if fid == 24:
        # COCO's standard implementation stores signs through xopt and applies the raw function directly.
        rot1_l = rot1
        rot2_l = rot2
        xopt_l = xopt
        value = _lunacek_raw(x, xopt_l, rot1_l, rot2_l)
        # raw value at the generated best parameter is not exactly zero; normalize to expose fopt as f*.
        best_raw = _lunacek_raw(xopt_l, xopt_l, rot1_l, rot2_l)
        return value - best_raw + fopt
    raise ValueError("function_id must be in {1, ..., 24}.")


def evaluate_bbob(variables_values: ArrayLike | None, function_id: int, instance: int = BBOB_DEFAULT_INSTANCE) -> float:
    """Evaluate one BBOB function, inferring dimension from ``variables_values``."""
    x = _asarray(variables_values)
    problem = BBOBFunction(function_id=function_id, dimension=x.size, instance=instance)
    return problem(x)


def get_bbob_function(function_id: int, dimension: int = BBOB_DEFAULT_DIMENSION, instance: int = BBOB_DEFAULT_INSTANCE) -> BBOBFunction:
    """Return a callable deterministic BBOB function instance."""
    return BBOBFunction(function_id=function_id, dimension=dimension, instance=instance)


def get_bbob_optimum(function_id: int, dimension: int = BBOB_DEFAULT_DIMENSION, instance: int = BBOB_DEFAULT_INSTANCE) -> tuple[np.ndarray, float]:
    """Return the deterministic optimizer and objective value for a BBOB problem."""
    problem = get_bbob_function(function_id, dimension=dimension, instance=instance)
    return problem.xopt.copy(), problem.fopt


def list_bbob_functions() -> list[str]:
    """Return registered BBOB function IDs as ``bbob_fXX`` names."""
    return [f"bbob_f{i:02d}" for i in BBOB_FUNCTION_IDS]


def bbob_suite(dimensions: Iterable[int] = (2, 3, 5, 10, 20, 40), instances: Iterable[int] = (1,), functions: Iterable[int] = BBOB_FUNCTION_IDS) -> list[BBOBFunction]:
    """Build a list of callable BBOB problems for dimensions/instances/functions."""
    return [BBOBFunction(fid, int(dim), int(inst)) for fid in functions for dim in dimensions for inst in instances]


def _make_bbob_wrapper(function_id: int) -> Callable[[ArrayLike | None], float]:
    def wrapper(variables_values: ArrayLike | None = None, instance: int = BBOB_DEFAULT_INSTANCE) -> float:
        return evaluate_bbob(variables_values, function_id=function_id, instance=instance)
    wrapper.__name__ = f"bbob_f{function_id:02d}"
    wrapper.__doc__ = f"BBOB F{function_id:02d} — {BBOB_NAMES[function_id]} (default instance 1)."
    return wrapper


for _fid in BBOB_FUNCTION_IDS:
    globals()[f"bbob_f{_fid:02d}"] = _make_bbob_wrapper(_fid)

BBOB_FUNCTIONS: Dict[str, Callable[..., float]] = {f"bbob_f{i:02d}": globals()[f"bbob_f{i:02d}"] for i in BBOB_FUNCTION_IDS}
BBOB_METADATA: Dict[str, Dict[str, str]] = {
    f"bbob_f{i:02d}": {
        "name": f"BBOB F{i:02d}: {BBOB_NAMES[i]}",
        "domain": "[-5, 5]^D, D >= 2; deterministic COCO/BBOB instances via instance=...",
        "optimum": "Use get_bbob_optimum(function_id, dimension, instance) for the shifted optimizer and f*.",
        "type": BBOB_TYPES[i],
    }
    for i in BBOB_FUNCTION_IDS
}

__all__ = [
    "BBOBFunction",
    "BBOB_FUNCTION_IDS",
    "BBOB_FUNCTIONS",
    "BBOB_METADATA",
    "BBOB_NAMES",
    "bbob_suite",
    "evaluate_bbob",
    "get_bbob_function",
    "get_bbob_optimum",
    "list_bbob_functions",
] + [f"bbob_f{i:02d}" for i in BBOB_FUNCTION_IDS]
