###############################################################################
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# Test Functions
#
# Cleaned/corrected version.
#
# Convention:
#   - All functions are written for minimization.
#   - The returned value is a Python float.
#   - Classical continuous benchmark functions accept an array-like vector.
#   - CEC 2022 functions require the official input data files.
###############################################################################

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _asarray(variables_values: ArrayLike | None, default: Sequence[float]) -> np.ndarray:
    """Convert input to a non-empty 1-D float array."""
    if variables_values is None:
        variables_values = default
    x = np.asarray(variables_values, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("variables_values must contain at least one value.")
    return x


def _require_dim(x: np.ndarray, dim: int, name: str) -> None:
    if x.size != dim:
        raise ValueError(f"{name} is defined for D={dim}. Got D={x.size}.")


###############################################################################
# Classical benchmark functions
###############################################################################

def ackley(variables_values: ArrayLike | None = None) -> float:
    """Ackley function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    d = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20.0 + np.e)


def alpine_1(variables_values: ArrayLike | None = None) -> float:
    """Alpine N.1. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def alpine_2(variables_values: ArrayLike | None = None) -> float:
    """
    Alpine N.2, minimization form.

    Classical Alpine N.2 is often written as prod(sqrt(x_i) sin(x_i)) and maximized
    on [0, 10]^D. Here the sign is flipped so the benchmark is a minimization
    problem: f* = -(2.808131180007...)^D at x_i ~= 7.917052698245946.
    """
    x = _asarray(variables_values, [7.917052698245946, 7.917052698245946])
    if np.any(x < 0.0):
        raise ValueError("alpine_2 is defined on x_i >= 0, usually [0, 10]^D.")
    return float(-np.prod(np.sqrt(x) * np.sin(x)))


def axis_parallel_hyper_ellipsoid(variables_values: ArrayLike | None = None) -> float:
    """Axis-parallel hyper-ellipsoid. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(i * x**2))


def beale(variables_values: ArrayLike | None = None) -> float:
    """Beale function. Global minimum: f(3,0.5)=0."""
    x = _asarray(variables_values, [3.0, 0.5])
    _require_dim(x, 2, "beale")
    x1, x2 = x
    return float((1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x2**3)**2)


def bent_cigar(variables_values: ArrayLike | None = None) -> float:
    """Bent Cigar function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(x[0]**2 + 1_000_000.0 * np.sum(x[1:]**2))


def bohachevsky_1(variables_values: ArrayLike | None = None) -> float:
    """Bohachevsky F1. Global minimum: f(0,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "bohachevsky_1")
    x1, x2 = x
    return float(x1**2 + 2.0*x2**2 - 0.3*np.cos(3.0*np.pi*x1) - 0.4*np.cos(4.0*np.pi*x2) + 0.7)


def bohachevsky_2(variables_values: ArrayLike | None = None) -> float:
    """Bohachevsky F2. Global minimum: f(0,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "bohachevsky_2")
    x1, x2 = x
    return float(x1**2 + 2.0*x2**2 - 0.3*np.cos(3.0*np.pi*x1)*np.cos(4.0*np.pi*x2) + 0.3)


def bohachevsky_3(variables_values: ArrayLike | None = None) -> float:
    """Bohachevsky F3. Global minimum: f(0,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "bohachevsky_3")
    x1, x2 = x
    return float(x1**2 + 2.0*x2**2 - 0.3*np.cos(3.0*np.pi*x1 + 4.0*np.pi*x2) + 0.3)


def booth(variables_values: ArrayLike | None = None) -> float:
    """Booth function. Global minimum: f(1,3)=0."""
    x = _asarray(variables_values, [1.0, 3.0])
    _require_dim(x, 2, "booth")
    x1, x2 = x
    return float((x1 + 2.0*x2 - 7.0)**2 + (2.0*x1 + x2 - 5.0)**2)


def branin_rcos(variables_values: ArrayLike | None = None) -> float:
    """Branin RCOS. Three global minima with f*=0.39788735772973816."""
    x = _asarray(variables_values, [-np.pi, 12.275])
    _require_dim(x, 2, "branin_rcos")
    x1, x2 = x
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return float(a * (x2 - b*x1**2 + c*x1 - r)**2 + s * (1.0 - t) * np.cos(x1) + s)


def bukin_6(variables_values: ArrayLike | None = None) -> float:
    """Bukin F6. Global minimum: f(-10,1)=0."""
    x = _asarray(variables_values, [-10.0, 1.0])
    _require_dim(x, 2, "bukin_6")
    x1, x2 = x
    return float(100.0 * np.sqrt(abs(x2 - 0.01*x1**2)) + 0.01 * abs(x1 + 10.0))


def chung_reynolds(variables_values: ArrayLike | None = None) -> float:
    """Chung-Reynolds function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(x**2)**2)


def cosine_mixture(variables_values: ArrayLike | None = None) -> float:
    """
    Cosine Mixture, minimization form.

    This implementation uses sum(x_i^2) - 0.1 sum(cos(5 pi x_i)), which has
    f* = -0.1D at x=0 on the usual bounded domain [-1,1]^D.
    """
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(x**2) - 0.1 * np.sum(np.cos(5.0 * np.pi * x)))


def cross_in_tray(variables_values: ArrayLike | None = None) -> float:
    """Cross-in-Tray. Four global minima with f*=-2.062611870822739."""
    x = _asarray(variables_values, [1.349406608602084, 1.349406608602084])
    _require_dim(x, 2, "cross_in_tray")
    x1, x2 = x
    a = np.sin(x1) * np.sin(x2)
    b = np.exp(abs(100.0 - np.sqrt(x1**2 + x2**2) / np.pi))
    return float(-0.0001 * (abs(a * b) + 1.0)**0.1)


def csendes(variables_values: ArrayLike | None = None) -> float:
    """Csendes function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    out = np.zeros_like(x, dtype=float)
    mask = x != 0.0
    out[mask] = (x[mask]**6) * (2.0 + np.sin(1.0 / x[mask]))
    return float(np.sum(out))


def de_jong_1(variables_values: ArrayLike | None = None) -> float:
    """De Jong F1 / Sphere. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(x**2))


def discus(variables_values: ArrayLike | None = None) -> float:
    """Discus function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(1_000_000.0 * x[0]**2 + np.sum(x[1:]**2))


def dixon_price(variables_values: ArrayLike | None = None) -> float:
    """Dixon-Price function. Global minimum: f(x*)=0."""
    x = _asarray(variables_values, [1.0, 1.0 / np.sqrt(2.0)])
    if x.size == 1:
        return float((x[0] - 1.0)**2)
    i = np.arange(2, x.size + 1, dtype=float)
    return float((x[0] - 1.0)**2 + np.sum(i * (2.0*x[1:]**2 - x[:-1])**2))


def drop_wave(variables_values: ArrayLike | None = None) -> float:
    """Drop-Wave function. Global minimum: f(0,0)=-1."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "drop_wave")
    x1, x2 = x
    r = np.sqrt(x1**2 + x2**2)
    return float(-(1.0 + np.cos(12.0*r)) / (0.5*r**2 + 2.0))


def easom(variables_values: ArrayLike | None = None) -> float:
    """Easom function. Global minimum: f(pi,pi)=-1."""
    x = _asarray(variables_values, [np.pi, np.pi])
    _require_dim(x, 2, "easom")
    x1, x2 = x
    return float(-np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2))


def eggholder(variables_values: ArrayLike | None = None) -> float:
    """Eggholder function. Global minimum near f(512,404.2319)=-959.6407."""
    x = _asarray(variables_values, [512.0, 404.2319])
    _require_dim(x, 2, "eggholder")
    x1, x2 = x
    return float(-(x2 + 47.0) * np.sin(np.sqrt(abs(x1/2.0 + x2 + 47.0))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47.0)))))


def elliptic(variables_values: ArrayLike | None = None) -> float:
    """High-conditioned elliptic function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    if x.size == 1:
        return float(x[0]**2)
    i = np.arange(x.size, dtype=float)
    weights = 1_000_000.0 ** (i / (x.size - 1.0))
    return float(np.sum(weights * x**2))


def expanded_griewank_plus_rosenbrock(variables_values: ArrayLike | None = None) -> float:
    """Expanded Griewank plus Rosenbrock. Global minimum: f(1,...,1)=0."""
    x = _asarray(variables_values, [1.0, 1.0])
    if x.size < 2:
        raise ValueError("expanded_griewank_plus_rosenbrock requires D >= 2.")
    x_next = np.roll(x, -1)
    g = 100.0 * (x**2 - x_next)**2 + (x - 1.0)**2
    return float(np.sum(g**2 / 4000.0 - np.cos(g) + 1.0))


def goldstein_price(variables_values: ArrayLike | None = None) -> float:
    """Goldstein-Price function. Global minimum: f(0,-1)=3."""
    x = _asarray(variables_values, [0.0, -1.0])
    _require_dim(x, 2, "goldstein_price")
    x1, x2 = x
    a = 1.0 + (x1 + x2 + 1.0)**2 * (19.0 - 14.0*x1 + 3.0*x1**2 - 14.0*x2 + 6.0*x1*x2 + 3.0*x2**2)
    b = 30.0 + (2.0*x1 - 3.0*x2)**2 * (18.0 - 32.0*x1 + 12.0*x1**2 + 48.0*x2 - 36.0*x1*x2 + 27.0*x2**2)
    return float(a * b)


def griewangk_8(variables_values: ArrayLike | None = None) -> float:
    """Griewank function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def happy_cat(variables_values: ArrayLike | None = None, alpha: float = 1.0/8.0) -> float:
    """Happy Cat function. Global minimum: f(-1,...,-1)=0."""
    x = _asarray(variables_values, [-1.0, -1.0])
    d = x.size
    r2 = np.sum(x**2)
    sx = np.sum(x)
    return float(abs(r2 - d)**(2.0 * alpha) + (0.5*r2 + sx) / d + 0.5)


def hgbat(variables_values: ArrayLike | None = None, alpha: float = 1.0/4.0) -> float:
    """HGBat function. Global minimum: f(-1,...,-1)=0."""
    x = _asarray(variables_values, [-1.0, -1.0])
    d = x.size
    r2 = np.sum(x**2)
    sx = np.sum(x)
    return float(abs(r2**2 - sx**2)**(2.0 * alpha) + (0.5*r2 + sx) / d + 0.5)


def himmelblau(variables_values: ArrayLike | None = None) -> float:
    """Himmelblau function. Four global minima with f*=0."""
    x = _asarray(variables_values, [3.0, 2.0])
    _require_dim(x, 2, "himmelblau")
    x1, x2 = x
    return float((x1**2 + x2 - 11.0)**2 + (x1 + x2**2 - 7.0)**2)


def holder_table(variables_values: ArrayLike | None = None) -> float:
    """Hölder Table function. Four global minima with f*=-19.20850256788675."""
    x = _asarray(variables_values, [8.055023472141116, 9.664590028909654])
    _require_dim(x, 2, "holder_table")
    x1, x2 = x
    return float(-abs(np.sin(x1) * np.cos(x2) * np.exp(abs(1.0 - np.sqrt(x1**2 + x2**2) / np.pi))))


def katsuura(variables_values: ArrayLike | None = None) -> float:
    """Katsuura function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    d = x.size
    j = np.arange(1, 33, dtype=float)[:, None]
    i = np.arange(1, d + 1, dtype=float)
    inner = np.sum(np.abs((2.0**j) * x - np.round((2.0**j) * x)) / (2.0**j), axis=0)
    return float(np.prod((1.0 + i * inner) ** (10.0 / (d ** 1.2))) - 1.0)


def levy(variables_values: ArrayLike | None = None) -> float:
    """Levy function. Global minimum: f(1,...,1)=0."""
    x = _asarray(variables_values, [1.0, 1.0])
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi*w[:-1] + 1.0)**2))
    term3 = (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0*np.pi*w[-1])**2)
    return float(term1 + term2 + term3)


def levi_13(variables_values: ArrayLike | None = None) -> float:
    """Levi N.13 function. Global minimum: f(1,1)=0."""
    x = _asarray(variables_values, [1.0, 1.0])
    _require_dim(x, 2, "levi_13")
    x1, x2 = x
    return float(np.sin(3.0*np.pi*x1)**2 + (x1 - 1.0)**2 * (1.0 + np.sin(3.0*np.pi*x2)**2) + (x2 - 1.0)**2 * (1.0 + np.sin(2.0*np.pi*x2)**2))


def matyas(variables_values: ArrayLike | None = None) -> float:
    """Matyas function. Global minimum: f(0,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "matyas")
    x1, x2 = x
    return float(0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2)


def mccormick(variables_values: ArrayLike | None = None) -> float:
    """McCormick function. Global minimum near f(-0.54719,-1.54719)=-1.91322."""
    x = _asarray(variables_values, [-0.5471975602214493, -1.5471975602214493])
    _require_dim(x, 2, "mccormick")
    x1, x2 = x
    return float(np.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1.0)


def michalewicz(variables_values: ArrayLike | None = None, m: int = 10) -> float:
    """Michalewicz function. Known global minima depend on D and m."""
    x = _asarray(variables_values, [2.20290552014618, 1.5707963267948966])
    i = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2.0 * m)))


def modified_schwefel(variables_values: ArrayLike | None = None) -> float:
    """
    Shifted Schwefel variant used in CEC-style benchmarks.
    Global minimum: f(0,...,0)=0.
    """
    x = _asarray(variables_values, [0.0, 0.0])
    d = x.size
    z = x + 420.9687462275036
    f = 0.0
    for zi in z:
        if zi > 500.0:
            r = np.fmod(zi, 500.0)
            f -= (500.0 - r) * np.sin(np.sqrt(abs(500.0 - r)))
            f += ((zi - 500.0) / 100.0)**2 / d
        elif zi < -500.0:
            r = np.fmod(abs(zi), 500.0)
            f -= (-500.0 + r) * np.sin(np.sqrt(abs(500.0 - r)))
            f += ((zi + 500.0) / 100.0)**2 / d
        else:
            f -= zi * np.sin(np.sqrt(abs(zi)))
    f += 418.9828872724338 * d
    return float(f)


def perm(variables_values: ArrayLike | None = None, beta: float = 0.5) -> float:
    """Perm 0,d,beta function. Global minimum: f(1,1/2,...,1/D)=0."""
    x = _asarray(variables_values, [1.0, 0.5])
    d = x.size
    i = np.arange(1, d + 1, dtype=float)[:, None]
    j = np.arange(1, d + 1, dtype=float)[None, :]
    xj = x[None, :]
    inner = np.sum((j + beta) * (xj**i - (1.0 / j)**i), axis=1)
    return float(np.sum(inner**2))


def pinter(variables_values: ArrayLike | None = None) -> float:
    """Pinter function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    sub = np.roll(x, 1)
    add = np.roll(x, -1)
    i = np.arange(1, x.size + 1, dtype=float)
    a = sub * np.sin(x) + np.sin(add)
    b = sub**2 - 2.0*x + 3.0*add - np.cos(x) + 1.0
    return float(np.sum(i*x**2) + np.sum(20.0*i*np.sin(a)**2) + np.sum(i*np.log10(1.0 + i*b**2)))


def powell(variables_values: ArrayLike | None = None) -> float:
    """Powell singular function. Requires D multiple of 4. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0, 0.0, 0.0])
    if x.size % 4 != 0:
        raise ValueError("powell requires dimension D to be a multiple of 4.")
    x1, x2, x3, x4 = x[0::4], x[1::4], x[2::4], x[3::4]
    return float(np.sum((x1 + 10.0*x2)**2 + 5.0*(x3 - x4)**2 + (x2 - 2.0*x3)**4 + 10.0*(x1 - x4)**4))


def qing(variables_values: ArrayLike | None = None) -> float:
    """Qing function. Global minima: x_i=+-sqrt(i), i=1,...,D; f*=0."""
    x = _asarray(variables_values, [1.0, np.sqrt(2.0)])
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum((x**2 - i)**2))


def quintic(variables_values: ArrayLike | None = None) -> float:
    """Quintic function. Global minima: each x_i in {-1, 2}; f*=0."""
    x = _asarray(variables_values, [-1.0, -1.0])
    return float(np.sum(np.abs(x**5 - 3.0*x**4 + 4.0*x**3 + 2.0*x**2 - 10.0*x - 4.0)))


def rastrigin(variables_values: ArrayLike | None = None) -> float:
    """Rastrigin function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(10.0*x.size + np.sum(x**2 - 10.0*np.cos(2.0*np.pi*x)))


def ridge(variables_values: ArrayLike | None = None) -> float:
    """Cumulative ridge function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(np.cumsum(x)**2))


def rosenbrocks_valley(variables_values: ArrayLike | None = None) -> float:
    """Rosenbrock valley. Global minimum: f(1,...,1)=0."""
    x = _asarray(variables_values, [1.0, 1.0])
    if x.size < 2:
        raise ValueError("rosenbrocks_valley requires D >= 2.")
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))


def salomon(variables_values: ArrayLike | None = None) -> float:
    """Salomon function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    r = np.sqrt(np.sum(x**2))
    return float(1.0 - np.cos(2.0*np.pi*r) + 0.1*r)


def schaffer_2(variables_values: ArrayLike | None = None) -> float:
    """Schaffer F2. Global minimum: f(0,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "schaffer_2")
    x1, x2 = x
    return float(0.5 + (np.sin(x1**2 - x2**2)**2 - 0.5) / (1.0 + 0.001*(x1**2 + x2**2))**2)


def schaffer_4(variables_values: ArrayLike | None = None) -> float:
    """Schaffer F4. Four global minima with f*=0.29257863204552975."""
    x = _asarray(variables_values, [0.0, 1.25313])
    _require_dim(x, 2, "schaffer_4")
    x1, x2 = x
    return float(0.5 + (np.cos(np.sin(abs(x1**2 - x2**2)))**2 - 0.5) / (1.0 + 0.001*(x1**2 + x2**2))**2)


def schaffer_6(variables_values: ArrayLike | None = None) -> float:
    """Schaffer F6. Global minimum: f(0,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "schaffer_6")
    x1, x2 = x
    r2 = x1**2 + x2**2
    return float(0.5 + (np.sin(np.sqrt(r2))**2 - 0.5) / (1.0 + 0.001*r2)**2)


def schumer_steiglitz(variables_values: ArrayLike | None = None) -> float:
    """Schumer-Steiglitz function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(x**4))


def schwefel(variables_values: ArrayLike | None = None) -> float:
    """Schwefel 2.26. Global minimum: x_i=420.9687462275036, f*=0."""
    x = _asarray(variables_values, [420.9687462275036, 420.9687462275036])
    return float(418.9828872724338 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def schwefel_221(variables_values: ArrayLike | None = None) -> float:
    """Schwefel 2.21. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.max(np.abs(x)))


def schwefel_222(variables_values: ArrayLike | None = None) -> float:
    """Schwefel 2.22. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def six_hump_camel_back(variables_values: ArrayLike | None = None) -> float:
    """Six-Hump Camel Back. Two global minima with f*=-1.031628453489877."""
    x = _asarray(variables_values, [0.08984201368301331, -0.7126564032704135])
    _require_dim(x, 2, "six_hump_camel_back")
    x1, x2 = x
    return float(4.0*x1**2 - 2.1*x1**4 + x1**6/3.0 + x1*x2 - 4.0*x2**2 + 4.0*x2**4)


def sphere_2(variables_values: ArrayLike | None = None) -> float:
    """Sum of Different Powers. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    exponents = np.arange(2, x.size + 2, dtype=float)
    return float(np.sum(np.abs(x)**exponents))


def sphere_3(variables_values: ArrayLike | None = None) -> float:
    """Rotated hyper-ellipsoid. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(np.cumsum(x**2)))


def step(variables_values: ArrayLike | None = None) -> float:
    """Step function. Global minimum: f*=0 for |x_i|<1."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(np.floor(np.abs(x))))


def step_2(variables_values: ArrayLike | None = None) -> float:
    """Step 2 function. Global minimum: f*=0 for -0.5 <= x_i < 0.5."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(np.floor(x + 0.5)**2))


def step_3(variables_values: ArrayLike | None = None) -> float:
    """Step 3 function. Global minimum: f*=0 for |x_i|<1."""
    x = _asarray(variables_values, [0.0, 0.0])
    return float(np.sum(np.floor(x**2)))


def stepint(variables_values: ArrayLike | None = None) -> float:
    """
    Stepint function: 25 + sum floor(x_i).

    On the usual domain [-5.12, 5.12]^D, f* = 25 - 6D for x_i in [-5.12, -5).
    Without explicit bounds this function is unbounded below.
    """
    x = _asarray(variables_values, [-5.1, -5.1])
    return float(25.0 + np.sum(np.floor(x)))


def styblinski_tang(variables_values: ArrayLike | None = None) -> float:
    """Styblinski-Tang. Global minimum: x_i=-2.903534027771177, f*=-39.16616570377141D."""
    x = _asarray(variables_values, [-2.903534027771177, -2.903534027771177])
    return float(0.5 * np.sum(x**4 - 16.0*x**2 + 5.0*x))


def three_hump_camel_back(variables_values: ArrayLike | None = None) -> float:
    """Three-Hump Camel Back. Global minimum: f(0,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    _require_dim(x, 2, "three_hump_camel_back")
    x1, x2 = x
    return float(2.0*x1**2 - 1.05*x1**4 + x1**6/6.0 + x1*x2 + x2**2)


def trid(variables_values: ArrayLike | None = None) -> float:
    """Trid function. Global minimum: x_i=i(D+1-i), f*=-D(D+4)(D-1)/6."""
    x = _asarray(variables_values, [2.0, 2.0])
    return float(np.sum((x - 1.0)**2) - np.sum(x[1:] * x[:-1]))


def weierstrass(variables_values: ArrayLike | None = None, a: float = 0.5, b: float = 3.0, k_max: int = 20) -> float:
    """Weierstrass function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    k = np.arange(k_max + 1, dtype=float)
    term1 = np.sum([(a**ki) * np.cos(2.0*np.pi*(b**ki)*(x + 0.5)) for ki in k])
    term2 = x.size * np.sum([(a**ki) * np.cos(np.pi*(b**ki)) for ki in k])
    return float(term1 - term2)


def whitley(variables_values: ArrayLike | None = None) -> float:
    """Whitley function. Global minimum: f(1,...,1)=0."""
    x = _asarray(variables_values, [1.0, 1.0])
    xi = x[:, None]
    xj = x[None, :]
    tmp = 100.0 * (xi**2 - xj)**2 + (1.0 - xj)**2
    return float(np.sum(tmp**2 / 4000.0 - np.cos(tmp) + 1.0))


def zakharov(variables_values: ArrayLike | None = None) -> float:
    """Zakharov function. Global minimum: f(0,...,0)=0."""
    x = _asarray(variables_values, [0.0, 0.0])
    i = np.arange(1, x.size + 1, dtype=float)
    s1 = np.sum(x**2)
    s2 = np.sum(0.5 * i * x)
    return float(s1 + s2**2 + s2**4)



###############################################################################
# Constrained engineering design benchmark functions
###############################################################################

def tension_spring(variables_values: ArrayLike | None = None) -> float:
    """Tension/compression spring design. Variables: d, D, N."""
    x = _asarray(variables_values, [0.05169, 0.35675, 11.2871])
    _require_dim(x, 3, "tension_spring")
    d, D, N = x
    return float((N + 2.0) * D * d**2)


def tension_spring_constraints(variables_values: ArrayLike | None = None) -> List[float]:
    """Tension/compression spring constraints in g(x) <= 0 form."""
    x = _asarray(variables_values, [0.05169, 0.35675, 11.2871])
    _require_dim(x, 3, "tension_spring_constraints")
    d, D, N = x
    return [
        float(1.0 - (D**3 * N) / (71785.0 * d**4)),
        float((4.0*D**2 - d*D) / (12566.0 * (D*d**3 - d**4)) + 1.0/(5108.0*d**2) - 1.0),
        float(1.0 - 140.45*d / (D**2 * N)),
        float((d + D) / 1.5 - 1.0),
    ]


def welded_beam(variables_values: ArrayLike | None = None) -> float:
    """Welded beam design. Variables: h, l, t, b."""
    x = _asarray(variables_values, [0.20572964, 3.47048867, 9.03662391, 0.20572964])
    _require_dim(x, 4, "welded_beam")
    h, l, t, b = x
    return float(1.10471*h**2*l + 0.04811*t*b*(14.0 + l))


def welded_beam_constraints(variables_values: ArrayLike | None = None) -> List[float]:
    """Welded beam constraints in g(x) <= 0 form."""
    x = _asarray(variables_values, [0.20572964, 3.47048867, 9.03662391, 0.20572964])
    _require_dim(x, 4, "welded_beam_constraints")
    h, l, t, b = x

    P = 6000.0
    L = 14.0
    E = 30.0e6
    G = 12.0e6
    tau_max = 13600.0
    sigma_max = 30000.0

    M = P * (L + l / 2.0)
    R = np.sqrt(l**2 / 4.0 + ((h + t) / 2.0)**2)
    J = 2.0 * (np.sqrt(2.0) * h * l) * (l**2 / 12.0 + ((h + t) / 2.0)**2)
    tau_prime = P / (np.sqrt(2.0) * h * l)
    tau_double = M * R / J
    tau = np.sqrt(tau_prime**2 + 2.0*tau_prime*tau_double*l/(2.0*R) + tau_double**2)
    sigma = 6.0 * P * L / (b * t**2)
    delta = 4.0 * P * L**3 / (E * t**3 * b)
    pc = (4.013 * E * np.sqrt(t**2 * b**6 / 36.0) / L**2) * (1.0 - t/(2.0*L) * np.sqrt(E/(4.0*G)))

    return [
        float(tau - tau_max),
        float(sigma - sigma_max),
        float(h - b),
        float(0.10471*h**2 + 0.04811*t*b*(14.0 + l) - 5.0),
        float(0.125 - h),
        float(delta - 0.25),
        float(P - pc),
    ]


def pressure_vessel(variables_values: ArrayLike | None = None) -> float:
    """Pressure vessel design, continuous relaxation. Variables: Ts, Th, R, L."""
    x = _asarray(variables_values, [0.7275909294, 0.3596485734, 37.699011884, 240.0])
    _require_dim(x, 4, "pressure_vessel")
    Ts, Th, R, L = x
    return float(0.6224*Ts*R*L + 1.7781*Th*R**2 + 3.1661*Ts**2*L + 19.84*Ts**2*R)


def pressure_vessel_constraints(variables_values: ArrayLike | None = None) -> List[float]:
    """Pressure vessel continuous-relaxation constraints in g(x) <= 0 form."""
    x = _asarray(variables_values, [0.7275909294, 0.3596485734, 37.699011884, 240.0])
    _require_dim(x, 4, "pressure_vessel_constraints")
    Ts, Th, R, L = x
    return [
        float(0.0193*R - Ts),
        float(0.00954*R - Th),
        float(1296000.0 - np.pi*R**2*L - (4.0/3.0)*np.pi*R**3),
        float(L - 240.0),
    ]


def _pressure_vessel_round_thicknesses(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    y[0] = np.ceil(y[0] / 0.0625) * 0.0625
    y[1] = np.ceil(y[1] / 0.0625) * 0.0625
    return y


def pressure_vessel_discrete(variables_values: ArrayLike | None = None) -> float:
    """Pressure vessel design with Ts and Th rounded upward to multiples of 1/16."""
    x = _asarray(variables_values, [0.8125, 0.4375, 42.098445596, 176.636595842])
    _require_dim(x, 4, "pressure_vessel_discrete")
    return pressure_vessel(_pressure_vessel_round_thicknesses(x))


def pressure_vessel_discrete_constraints(variables_values: ArrayLike | None = None) -> List[float]:
    """Discrete-thickness pressure vessel constraints in g(x) <= 0 form."""
    x = _asarray(variables_values, [0.8125, 0.4375, 42.098445596, 176.636595842])
    _require_dim(x, 4, "pressure_vessel_discrete_constraints")
    return pressure_vessel_constraints(_pressure_vessel_round_thicknesses(x))


def speed_reducer(variables_values: ArrayLike | None = None) -> float:
    """Speed reducer design benchmark."""
    x = _asarray(variables_values, [3.5, 0.7, 17.0, 7.3, 7.71531991, 3.35021467, 5.28665446])
    _require_dim(x, 7, "speed_reducer")
    x1, x2, x3, x4, x5, x6, x7 = x
    return float(
        0.7854*x1*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934)
        - 1.508*x1*(x6**2 + x7**2)
        + 7.4777*(x6**3 + x7**3)
        + 0.7854*(x4*x6**2 + x5*x7**2)
    )


def speed_reducer_constraints(variables_values: ArrayLike | None = None) -> List[float]:
    """Speed reducer constraints in g(x) <= 0 form."""
    x = _asarray(variables_values, [3.5, 0.7, 17.0, 7.3, 7.71531991, 3.35021467, 5.28665446])
    _require_dim(x, 7, "speed_reducer_constraints")
    x1, x2, x3, x4, x5, x6, x7 = x
    return [
        float(27.0/(x1*x2**2*x3) - 1.0),
        float(397.5/(x1*x2**2*x3**2) - 1.0),
        float(1.93*x4**3/(x2*x3*x6**4) - 1.0),
        float(1.93*x5**3/(x2*x3*x7**4) - 1.0),
        float(np.sqrt((745.0*x4/(x2*x3))**2 + 1.69e7)/(110.0*x6**3) - 1.0),
        float(np.sqrt((745.0*x5/(x2*x3))**2 + 1.575e8)/(85.0*x7**3) - 1.0),
        float(x2*x3/40.0 - 1.0),
        float(5.0*x2/x1 - 1.0),
        float(x1/(12.0*x2) - 1.0),
        float((1.5*x6 + 1.9)/x4 - 1.0),
        float((1.1*x7 + 1.9)/x5 - 1.0),
    ]


def three_bar_truss(variables_values: ArrayLike | None = None) -> float:
    """Three-bar truss design benchmark."""
    x = _asarray(variables_values, [0.7886751346, 0.4082482905])
    _require_dim(x, 2, "three_bar_truss")
    A1, A2 = x
    return float(100.0 * (2.0*np.sqrt(2.0)*A1 + A2))


def three_bar_truss_constraints(variables_values: ArrayLike | None = None, P: float = 2.0, sigma: float = 2.0) -> List[float]:
    """Three-bar truss constraints in g(x) <= 0 form."""
    x = _asarray(variables_values, [0.7886751346, 0.4082482905])
    _require_dim(x, 2, "three_bar_truss_constraints")
    A1, A2 = x
    den = np.sqrt(2.0)*A1**2 + 2.0*A1*A2
    return [
        float((np.sqrt(2.0)*A1 + A2)*P/den - sigma),
        float(A2*P/den - sigma),
        float(P/(A1 + np.sqrt(2.0)*A2) - sigma),
    ]


def cantilever_beam(variables_values: ArrayLike | None = None) -> float:
    """Cantilever beam design benchmark."""
    x = _asarray(variables_values, [6.01601575, 5.30917339, 4.49432962, 3.50147536, 2.15266507])
    _require_dim(x, 5, "cantilever_beam")
    return float(0.0624 * np.sum(x))


def cantilever_beam_constraints(variables_values: ArrayLike | None = None) -> List[float]:
    """Cantilever beam constraint in g(x) <= 0 form."""
    x = _asarray(variables_values, [6.01601575, 5.30917339, 4.49432962, 3.50147536, 2.15266507])
    _require_dim(x, 5, "cantilever_beam_constraints")
    coeff = np.array([61.0, 37.0, 19.0, 7.0, 1.0])
    return [float(np.sum(coeff / (x**3)) - 1.0)]


def gear_train(variables_values: ArrayLike | None = None, round_variables: bool = True) -> float:
    """Gear train design. Classical variables are integer tooth counts in [12, 60]."""
    x = _asarray(variables_values, [16.0, 19.0, 43.0, 49.0])
    _require_dim(x, 4, "gear_train")
    if round_variables:
        x = np.rint(x)
    x1, x2, x3, x4 = x
    return float((1.0/6.931 - (x1*x2)/(x3*x4))**2)


def gear_train_constraints(variables_values: ArrayLike | None = None) -> List[float]:
    """Gear train has no inequality constraints beyond integer box bounds."""
    _ = _asarray(variables_values, [16.0, 19.0, 43.0, 49.0])
    return []


def _constraint_list(constraint_function: Callable[[ArrayLike | None], List[float]], n_constraints: int) -> List[Callable[[ArrayLike], float]]:
    """Convert a vector-valued constraint function into pymetaheuristic-compatible callables."""
    return [lambda x, idx=i: float(constraint_function(x)[idx]) for i in range(n_constraints)]


ENGINEERING_BENCHMARKS: Dict[str, Dict[str, object]] = {
    "tension_spring": {
        "name": "Tension/compression spring design",
        "objective": tension_spring,
        "constraints": _constraint_list(tension_spring_constraints, 4),
        "constraint_function": tension_spring_constraints,
        "min_values": (0.05, 0.25, 2.0),
        "max_values": (2.00, 1.30, 15.0),
        "best_known_position": (0.05169, 0.35675, 11.2871),
        "best_known_fitness": 0.012665,
        "notes": "Continuous constrained spring benchmark; constraints use g(x) <= 0.",
    },
    "welded_beam": {
        "name": "Welded beam design",
        "objective": welded_beam,
        "constraints": _constraint_list(welded_beam_constraints, 7),
        "constraint_function": welded_beam_constraints,
        "min_values": (0.1, 0.1, 0.1, 0.1),
        "max_values": (2.0, 10.0, 10.0, 2.0),
        "best_known_position": (0.20572964, 3.47048867, 9.03662391, 0.20572964),
        "best_known_fitness": 1.724852,
        "notes": "Common seven-constraint welded-beam formulation.",
    },
    "pressure_vessel": {
        "name": "Pressure vessel design, continuous relaxation",
        "objective": pressure_vessel,
        "constraints": _constraint_list(pressure_vessel_constraints, 4),
        "constraint_function": pressure_vessel_constraints,
        "min_values": (0.0, 0.0, 10.0, 10.0),
        "max_values": (99.0, 99.0, 200.0, 240.0),
        "best_known_position": (0.7275909294, 0.3596485734, 37.699011884, 240.0),
        "best_known_fitness": 5804.376217,
        "notes": "Continuous relaxation. For the classical discrete-thickness variant, use pressure_vessel_discrete.",
    },
    "pressure_vessel_discrete": {
        "name": "Pressure vessel design, discrete thickness",
        "objective": pressure_vessel_discrete,
        "constraints": _constraint_list(pressure_vessel_discrete_constraints, 4),
        "constraint_function": pressure_vessel_discrete_constraints,
        "min_values": (0.0625, 0.0625, 10.0, 10.0),
        "max_values": (6.1875, 6.1875, 200.0, 240.0),
        "best_known_position": (0.8125, 0.4375, 42.098445596, 176.636595842),
        "best_known_fitness": 6059.714335,
        "notes": "Rounds Ts and Th upward to multiples of 1/16 before objective/constraint evaluation.",
    },
    "speed_reducer": {
        "name": "Speed reducer design",
        "objective": speed_reducer,
        "constraints": _constraint_list(speed_reducer_constraints, 11),
        "constraint_function": speed_reducer_constraints,
        "min_values": (2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0),
        "max_values": (3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5),
        "best_known_position": (3.5, 0.7, 17.0, 7.3, 7.71531991, 3.35021467, 5.28665446),
        "best_known_fitness": 2994.471066,
        "notes": "Seven-variable mechanical design benchmark with eleven constraints.",
    },
    "three_bar_truss": {
        "name": "Three-bar truss design",
        "objective": three_bar_truss,
        "constraints": _constraint_list(three_bar_truss_constraints, 3),
        "constraint_function": three_bar_truss_constraints,
        "min_values": (0.0, 0.0),
        "max_values": (1.0, 1.0),
        "best_known_position": (0.7886751346, 0.4082482905),
        "best_known_fitness": 263.895843,
        "notes": "Two-variable structural design benchmark.",
    },
    "cantilever_beam": {
        "name": "Cantilever beam design",
        "objective": cantilever_beam,
        "constraints": _constraint_list(cantilever_beam_constraints, 1),
        "constraint_function": cantilever_beam_constraints,
        "min_values": (0.01, 0.01, 0.01, 0.01, 0.01),
        "max_values": (100.0, 100.0, 100.0, 100.0, 100.0),
        "best_known_position": (6.01601575, 5.30917339, 4.49432962, 3.50147536, 2.15266507),
        "best_known_fitness": 1.339956,
        "notes": "Five-variable constrained beam design benchmark.",
    },
    "gear_train": {
        "name": "Gear train design",
        "objective": gear_train,
        "constraints": [],
        "constraint_function": gear_train_constraints,
        "min_values": (12.0, 12.0, 12.0, 12.0),
        "max_values": (60.0, 60.0, 60.0, 60.0),
        "best_known_position": (16.0, 19.0, 43.0, 49.0),
        "best_known_fitness": 2.7008571488865134e-12,
        "notes": "Integer tooth-count benchmark; objective rounds variables by default.",
    },
}


def list_engineering_benchmarks() -> List[str]:
    """Return available constrained/discrete engineering benchmark IDs."""
    return sorted(ENGINEERING_BENCHMARKS.keys())


def get_engineering_benchmark(name: str) -> Dict[str, object]:
    """
    Return objective, constraints, bounds, and metadata for an engineering benchmark.

    The returned ``constraints`` entry is directly compatible with
    ``pymetaheuristic.optimize(..., constraints=..., constraint_handler='deb')``.
    """
    key = str(name).strip().lower()
    if key not in ENGINEERING_BENCHMARKS:
        raise KeyError(f"Unknown engineering benchmark: {name}. Available: {', '.join(list_engineering_benchmarks())}")
    return dict(ENGINEERING_BENCHMARKS[key])


def validate_engineering_benchmarks(tol: float = 1e-4, feasibility_tol: float = 1e-4) -> Dict[str, Dict[str, float]]:
    """Smoke-test engineering benchmarks at their stored best-known designs."""
    report: Dict[str, Dict[str, float]] = {}
    for name, info in ENGINEERING_BENCHMARKS.items():
        objective = info["objective"]
        constraint_function = info["constraint_function"]
        x_star = info["best_known_position"]
        f_star = float(info["best_known_fitness"])
        value = float(objective(x_star))
        constraints = list(constraint_function(x_star))
        max_violation = max(constraints) if constraints else 0.0
        err = abs(value - f_star)
        report[name] = {"objective_error": err, "max_violation": float(max_violation)}
        if err > tol:
            raise AssertionError(f"{name}: expected objective near {f_star}, got {value}, error={err}")
        if max_violation > feasibility_tol:
            raise AssertionError(f"{name}: best-known design violates constraints by {max_violation}")
    return report

###############################################################################
# CEC 2022 benchmark functions
###############################################################################

_CEC2022_DIMENSIONS = (2, 10, 20)
_CEC2022_NO_D2 = {6, 7, 8}


def _resolve_cec2022_data_source() -> Path:
    """Return either a directory or a .zip file containing the CEC 2022 data."""
    env_dir = os.environ.get("PYMETAHEURISTIC_CEC2022_DATA")
    candidates: List[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))

    here = Path(__file__).resolve().parent
    candidates.extend(
        [
            here / "cec2022_input_data",
            here / "input_data",
            here / "cec2022_input_data.zip",
            here / "input_data.zip",
            here / "Python-CEC2022.zip",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and (candidate.is_dir() or candidate.suffix.lower() == ".zip"):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "CEC 2022 input data not found. Place the official input-data folder next "
        "to this file as 'cec2022_input_data'/'input_data', place the official zip "
        "there, or set PYMETAHEURISTIC_CEC2022_DATA to the folder/zip path. "
        f"Searched: {searched}"
    )


def _loadtxt_cec2022(source: Path, filename: str) -> np.ndarray:
    """Load a CEC 2022 text data file from a directory or zip archive."""
    source = Path(source)

    if source.is_dir():
        path = source / filename
        if not path.exists():
            raise FileNotFoundError(f"Required CEC 2022 data file not found: {path}")
        return np.loadtxt(path)

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = archive.namelist()
            candidates = [
                filename,
                f"cec2022_input_data/{filename}",
                f"input_data/{filename}",
            ]

            member = next((name for name in candidates if name in members), None)
            if member is None:
                suffix = "/" + filename
                member = next((name for name in members if name.endswith(suffix)), None)

            if member is None:
                raise FileNotFoundError(
                    f"Required CEC 2022 data file '{filename}' not found inside {source}."
                )

            with archive.open(member) as fh:
                return np.loadtxt(io.TextIOWrapper(fh, encoding="utf-8"))

    raise FileNotFoundError(f"Invalid CEC 2022 data source: {source}")


def _shiftfunc_cec2022(x: np.ndarray, nx: int, Os: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)[:nx] - np.asarray(Os, dtype=float)[:nx]


def _rotatefunc_cec2022(x: np.ndarray, nx: int, Mr: np.ndarray) -> np.ndarray:
    Mr = np.asarray(Mr, dtype=float)
    if Mr.ndim == 1:
        Mr = Mr.reshape((nx, nx))
    return Mr[:nx, :nx].dot(np.asarray(x, dtype=float)[:nx])


def _sr_func_cec2022(x: np.ndarray, nx: int, Os: np.ndarray, Mr: np.ndarray, sh_rate: float, s_flag: int, r_flag: int) -> np.ndarray:
    y = _shiftfunc_cec2022(x, nx, Os) * sh_rate if s_flag == 1 else np.asarray(x, dtype=float)[:nx] * sh_rate
    return _rotatefunc_cec2022(y, nx, Mr) if r_flag == 1 else y


def _ellips_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    if nx == 1:
        return float(z[0]**2)
    return float(sum((10.0 ** (6.0 * i / (nx - 1))) * z[i] * z[i] for i in range(nx)))


def _bent_cigar_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    return float(z[0]**2 + sum(1.0e6 * z[i]**2 for i in range(1, nx)))


def _discus_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    return float(1.0e6 * z[0]**2 + sum(z[i]**2 for i in range(1, nx)))


def _rosenbrock_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 2.048 / 100.0, s_flag, r_flag) + 1.0
    return float(np.sum(100.0 * (z[:-1]**2 - z[1:])**2 + (z[:-1] - 1.0)**2))


def _ackley_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    return float(np.e - 20.0*np.exp(-0.2*np.sqrt(np.sum(z[:nx]**2) / nx)) - np.exp(np.sum(np.cos(2.0*np.pi*z[:nx])) / nx) + 20.0)


def _griewank_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 600.0 / 100.0, s_flag, r_flag)
    i = np.arange(1, nx + 1, dtype=float)
    return float(1.0 + np.sum(z[:nx]**2) / 4000.0 - np.prod(np.cos(z[:nx] / np.sqrt(i))))


def _rastrigin_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag)
    return float(np.sum(z[:nx]**2 - 10.0*np.cos(2.0*np.pi*z[:nx]) + 10.0))


def _schwefel_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1000.0 / 100.0, s_flag, r_flag)
    f = 0.0
    for zi0 in z[:nx]:
        zi = zi0 + 420.9687462275036
        if zi > 500.0:
            r = np.fmod(zi, 500.0)
            f -= (500.0 - r) * np.sin(np.sqrt(abs(500.0 - r)))
            f += ((zi - 500.0) / 100.0)**2 / nx
        elif zi < -500.0:
            r = np.fmod(abs(zi), 500.0)
            f -= (-500.0 + r) * np.sin(np.sqrt(abs(500.0 - r)))
            f += ((zi + 500.0) / 100.0)**2 / nx
        else:
            f -= zi * np.sin(np.sqrt(abs(zi)))
    return float(f + 418.9828872724338 * nx)


def _grie_rosen_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag) + 1.0
    z_next = np.roll(z, -1)
    g = 100.0 * (z**2 - z_next)**2 + (z - 1.0)**2
    return float(np.sum(g**2 / 4000.0 - np.cos(g) + 1.0))


def _escaffer6_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    z_next = np.roll(z, -1)
    r2 = z**2 + z_next**2
    return float(np.sum(0.5 + (np.sin(np.sqrt(r2))**2 - 0.5) / (1.0 + 0.001*r2)**2))


def _happycat_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag) - 1.0
    r2 = np.sum(z[:nx]**2)
    sx = np.sum(z[:nx])
    return float(abs(r2 - nx)**0.25 + (0.5*r2 + sx) / nx + 0.5)


def _hgbat_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag) - 1.0
    r2 = np.sum(z[:nx]**2)
    sx = np.sum(z[:nx])
    return float(abs(r2**2 - sx**2)**0.5 + (0.5*r2 + sx) / nx + 0.5)


def _schaffer_F7_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    s = 0.0
    for i in range(nx - 1):
        zi = np.sqrt(z[i]**2 + z[i+1]**2)
        s += zi**0.5 * (1.0 + np.sin(50.0 * zi**0.2)**2)
    return float((s / (nx - 1))**2)


def _step_rastrigin_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    y = np.asarray(x, dtype=float).copy()
    Os = np.asarray(Os, dtype=float)
    for i in range(nx):
        if abs(y[i] - Os[i]) > 0.5:
            y[i] = Os[i] + np.floor(2.0*(y[i] - Os[i]) + 0.5) / 2.0
    return _rastrigin_func_cec2022(y, nx, Os, Mr, s_flag, r_flag)


def _levy_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    w = 1.0 + z[:nx] / 4.0
    return float(
        np.sin(np.pi*w[0])**2
        + np.sum((w[:-1] - 1.0)**2 * (1.0 + 10.0*np.sin(np.pi*w[:-1] + 1.0)**2))
        + (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0*np.pi*w[-1])**2)
    )


def _zakharov_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    i = np.arange(1, nx + 1, dtype=float)
    s1 = np.sum(z[:nx]**2)
    s2 = np.sum(0.5 * i * z[:nx])
    return float(s1 + s2**2 + s2**4)


def _katsuura_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    f = 1.0
    tmp3 = nx**1.2
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            temp += abs((2.0**j) * z[i] - np.floor((2.0**j) * z[i] + 0.5)) / (2.0**j)
        f *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
    return float(10.0 / (nx * nx) * f - 10.0 / (nx * nx))


def _hybrid_slices(nx: int, proportions: Sequence[float]) -> List[slice]:
    sizes = [int(np.ceil(p * nx)) for p in proportions[:-1]]
    sizes.append(nx - sum(sizes))
    starts = np.cumsum([0] + sizes[:-1])
    return [slice(int(s), int(s + m)) for s, m in zip(starts, sizes)]


def _hf02_cec2022(x, nx, Os, Mr, S, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    idx = np.asarray(S, dtype=int).reshape(-1)[:nx] - 1
    y = z[idx]
    sl = _hybrid_slices(nx, [0.4, 0.4, 0.2])
    return float(
        _bent_cigar_func_cec2022(y[sl[0]], sl[0].stop - sl[0].start, Os, Mr, 0, 0)
        + _hgbat_func_cec2022(y[sl[1]], sl[1].stop - sl[1].start, Os, Mr, 0, 0)
        + _rastrigin_func_cec2022(y[sl[2]], sl[2].stop - sl[2].start, Os, Mr, 0, 0)
    )


def _hf10_cec2022(x, nx, Os, Mr, S, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    idx = np.asarray(S, dtype=int).reshape(-1)[:nx] - 1
    y = z[idx]
    sl = _hybrid_slices(nx, [0.1, 0.2, 0.2, 0.2, 0.1, 0.2])
    funcs = [_hgbat_func_cec2022, _katsuura_func_cec2022, _ackley_func_cec2022, _rastrigin_func_cec2022, _schwefel_func_cec2022, _schaffer_F7_func_cec2022]
    return float(sum(fn(y[s], s.stop - s.start, Os, Mr, 0, 0) for fn, s in zip(funcs, sl)))


def _hf06_cec2022(x, nx, Os, Mr, S, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    idx = np.asarray(S, dtype=int).reshape(-1)[:nx] - 1
    y = z[idx]
    sl = _hybrid_slices(nx, [0.3, 0.2, 0.2, 0.1, 0.2])
    funcs = [_katsuura_func_cec2022, _happycat_func_cec2022, _grie_rosen_func_cec2022, _schwefel_func_cec2022, _ackley_func_cec2022]
    return float(sum(fn(y[s], s.stop - s.start, Os, Mr, 0, 0) for fn, s in zip(funcs, sl)))


def _cf_cal_cec2022(x, nx, Os, delta, bias, fit, cf_num):
    x = np.asarray(x, dtype=float)
    Os = np.asarray(Os, dtype=float).reshape(cf_num, nx)
    fit = np.asarray(fit, dtype=float) + np.asarray(bias, dtype=float)
    delta = np.asarray(delta, dtype=float)

    dist2 = np.sum((Os[:, :nx] - x[:nx])**2, axis=1)
    exact = dist2 == 0.0
    if np.any(exact):
        return float(np.mean(fit[exact]))

    w = (1.0 / np.sqrt(dist2)) * np.exp(-dist2 / (2.0 * nx * delta**2))
    if np.all(w == 0.0):
        w = np.ones(cf_num)
    return float(np.sum((w / np.sum(w)) * fit))


def _cf01_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    fit = [
        10000.0 * _rosenbrock_func_cec2022(x, nx, Os[0*nx:1*nx], Mr[0*nx:1*nx, 0:nx], 1, r_flag) / 1e4,
        10000.0 * _ellips_func_cec2022(x, nx, Os[1*nx:2*nx], Mr[1*nx:2*nx, 0:nx], 1, r_flag) / 1e10,
        10000.0 * _bent_cigar_func_cec2022(x, nx, Os[2*nx:3*nx], Mr[2*nx:3*nx, 0:nx], 1, r_flag) / 1e30,
        10000.0 * _discus_func_cec2022(x, nx, Os[3*nx:4*nx], Mr[3*nx:4*nx, 0:nx], 1, r_flag) / 1e10,
        10000.0 * _ellips_func_cec2022(x, nx, Os[4*nx:5*nx], Mr[4*nx:5*nx, 0:nx], 1, 0) / 1e10,
    ]
    return _cf_cal_cec2022(x, nx, Os, [10, 20, 30, 40, 50], [0, 200, 300, 100, 400], fit, 5)


def _cf02_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    fit = [
        _schwefel_func_cec2022(x, nx, Os[0*nx:1*nx], Mr[0*nx:1*nx, 0:nx], 1, 0),
        _rastrigin_func_cec2022(x, nx, Os[1*nx:2*nx], Mr[1*nx:2*nx, 0:nx], 1, r_flag),
        _hgbat_func_cec2022(x, nx, Os[2*nx:3*nx], Mr[2*nx:3*nx, 0:nx], 1, r_flag),
    ]
    return _cf_cal_cec2022(x, nx, Os, [20, 10, 10], [0, 200, 100], fit, 3)


def _cf06_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    fit = [
        10000.0 * _escaffer6_func_cec2022(x, nx, Os[0*nx:1*nx], Mr[0*nx:1*nx, 0:nx], 1, r_flag) / 2e7,
        _schwefel_func_cec2022(x, nx, Os[1*nx:2*nx], Mr[1*nx:2*nx, 0:nx], 1, r_flag),
        1000.0 * _griewank_func_cec2022(x, nx, Os[2*nx:3*nx], Mr[2*nx:3*nx, 0:nx], 1, r_flag) / 100.0,
        _rosenbrock_func_cec2022(x, nx, Os[3*nx:4*nx], Mr[3*nx:4*nx, 0:nx], 1, r_flag),
        10000.0 * _rastrigin_func_cec2022(x, nx, Os[4*nx:5*nx], Mr[4*nx:5*nx, 0:nx], 1, r_flag) / 1e3,
    ]
    return _cf_cal_cec2022(x, nx, Os, [20, 20, 30, 30, 20], [0, 200, 300, 400, 200], fit, 5)


def _cf07_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    fit = [
        10000.0 * _hgbat_func_cec2022(x, nx, Os[0*nx:1*nx], Mr[0*nx:1*nx, 0:nx], 1, r_flag) / 1000.0,
        10000.0 * _rastrigin_func_cec2022(x, nx, Os[1*nx:2*nx], Mr[1*nx:2*nx, 0:nx], 1, r_flag) / 1e3,
        10000.0 * _schwefel_func_cec2022(x, nx, Os[2*nx:3*nx], Mr[2*nx:3*nx, 0:nx], 1, r_flag) / 4e3,
        10000.0 * _bent_cigar_func_cec2022(x, nx, Os[3*nx:4*nx], Mr[3*nx:4*nx, 0:nx], 1, r_flag) / 1e30,
        10000.0 * _ellips_func_cec2022(x, nx, Os[4*nx:5*nx], Mr[4*nx:5*nx, 0:nx], 1, r_flag) / 1e10,
        10000.0 * _escaffer6_func_cec2022(x, nx, Os[5*nx:6*nx], Mr[5*nx:6*nx, 0:nx], 1, r_flag) / 2e7,
    ]
    return _cf_cal_cec2022(x, nx, Os, [10, 20, 30, 40, 50, 60], [0, 300, 500, 100, 400, 200], fit, 6)


def _cec2022_eval(variables_values: ArrayLike, func_num: int) -> float:
    x = np.asarray(variables_values, dtype=float).reshape(-1)
    nx = int(x.size)

    if func_num < 1 or func_num > 12:
        raise ValueError(f"CEC 2022 function {func_num} is not defined. Valid IDs are 1..12.")
    if nx not in _CEC2022_DIMENSIONS:
        raise ValueError(f"CEC 2022 functions are only defined for D in {_CEC2022_DIMENSIONS}. Got D={nx}.")
    if nx == 2 and func_num in _CEC2022_NO_D2:
        raise ValueError("CEC 2022 functions 6, 7, and 8 are not defined for D=2.")

    data_source = _resolve_cec2022_data_source()
    M = _loadtxt_cec2022(data_source, f"M_{func_num}_D{nx}.txt")
    OShift_temp = _loadtxt_cec2022(data_source, f"shift_data_{func_num}.txt")

    # CEC 2022 shift-data files store either one 100-dimensional vector
    # (basic/hybrid functions) or a 10 x 100 matrix (composition functions).
    # For a requested dimension nx, use only the first nx coordinates.  For
    # composition functions, keep the component-wise shifts as a flat vector
    # [component_1, component_2, ...] because the official cf_cal routine
    # indexes Os[i * nx + j].
    OShift_raw = np.asarray(OShift_temp, dtype=float)
    if func_num < 9:
        OShift = OShift_raw.reshape(-1)[:nx]
    else:
        cf_num = {9: 5, 10: 3, 11: 5, 12: 6}[func_num]
        if OShift_raw.ndim == 1:
            OShift_raw = OShift_raw.reshape(10, -1)
        OShift = OShift_raw[:cf_num, :nx].reshape(-1)

    SS = _loadtxt_cec2022(data_source, f"shuffle_data_{func_num}_D{nx}.txt") if func_num in (6, 7, 8) else None

    if M.ndim == 1:
        if func_num < 9:
            M = M.reshape((nx, nx))
        else:
            cf_num = {9: 5, 10: 3, 11: 5, 12: 6}[func_num]
            M = M.reshape((cf_num * nx, nx))

    if func_num == 1:
        return float(_zakharov_func_cec2022(x, nx, OShift, M, 1, 1) + 300.0)
    if func_num == 2:
        return float(_rosenbrock_func_cec2022(x, nx, OShift, M, 1, 1) + 400.0)
    if func_num == 3:
        return float(_schaffer_F7_func_cec2022(x, nx, OShift, M, 1, 1) + 600.0)
    if func_num == 4:
        return float(_step_rastrigin_func_cec2022(x, nx, OShift, M, 1, 1) + 800.0)
    if func_num == 5:
        return float(_levy_func_cec2022(x, nx, OShift, M, 1, 1) + 900.0)
    if func_num == 6:
        return float(_hf02_cec2022(x, nx, OShift, M, SS, 1, 1) + 1800.0)
    if func_num == 7:
        return float(_hf10_cec2022(x, nx, OShift, M, SS, 1, 1) + 2000.0)
    if func_num == 8:
        return float(_hf06_cec2022(x, nx, OShift, M, SS, 1, 1) + 2200.0)
    if func_num == 9:
        return float(_cf01_cec2022(x, nx, OShift, M, 1, 1) + 2300.0)
    if func_num == 10:
        return float(_cf02_cec2022(x, nx, OShift, M, 1, 1) + 2400.0)
    if func_num == 11:
        return float(_cf06_cec2022(x, nx, OShift, M, 1, 1) + 2600.0)
    return float(_cf07_cec2022(x, nx, OShift, M, 1, 1) + 2700.0)


def cec_2022_f01(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 1)


def cec_2022_f02(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 2)


def cec_2022_f03(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 3)


def cec_2022_f04(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 4)


def cec_2022_f05(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 5)


def cec_2022_f06(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0] * 10), 6)


def cec_2022_f07(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0] * 10), 7)


def cec_2022_f08(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0] * 10), 8)


def cec_2022_f09(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 9)


def cec_2022_f10(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 10)


def cec_2022_f11(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 11)


def cec_2022_f12(variables_values: ArrayLike | None = None) -> float:
    return _cec2022_eval(_asarray(variables_values, [0.0, 0.0]), 12)


###############################################################################
# Registry and metadata
###############################################################################

FUNCTIONS: Dict[str, Callable[..., float]] = {
    "ackley": ackley,
    "alpine_1": alpine_1,
    "alpine_2": alpine_2,
    "axis_parallel_hyper_ellipsoid": axis_parallel_hyper_ellipsoid,
    "beale": beale,
    "bent_cigar": bent_cigar,
    "bohachevsky_1": bohachevsky_1,
    "bohachevsky_2": bohachevsky_2,
    "bohachevsky_3": bohachevsky_3,
    "booth": booth,
    "branin_rcos": branin_rcos,
    "bukin_6": bukin_6,
    "chung_reynolds": chung_reynolds,
    "cosine_mixture": cosine_mixture,
    "cross_in_tray": cross_in_tray,
    "csendes": csendes,
    "de_jong_1": de_jong_1,
    "discus": discus,
    "dixon_price": dixon_price,
    "drop_wave": drop_wave,
    "easom": easom,
    "eggholder": eggholder,
    "elliptic": elliptic,
    "expanded_griewank_plus_rosenbrock": expanded_griewank_plus_rosenbrock,
    "goldstein_price": goldstein_price,
    "griewangk_8": griewangk_8,
    "happy_cat": happy_cat,
    "hgbat": hgbat,
    "himmelblau": himmelblau,
    "holder_table": holder_table,
    "katsuura": katsuura,
    "levy": levy,
    "levi_13": levi_13,
    "matyas": matyas,
    "mccormick": mccormick,
    "michalewicz": michalewicz,
    "modified_schwefel": modified_schwefel,
    "perm": perm,
    "pinter": pinter,
    "powell": powell,
    "qing": qing,
    "quintic": quintic,
    "rastrigin": rastrigin,
    "ridge": ridge,
    "rosenbrocks_valley": rosenbrocks_valley,
    "salomon": salomon,
    "schaffer_2": schaffer_2,
    "schaffer_4": schaffer_4,
    "schaffer_6": schaffer_6,
    "schumer_steiglitz": schumer_steiglitz,
    "schwefel": schwefel,
    "schwefel_221": schwefel_221,
    "schwefel_222": schwefel_222,
    "six_hump_camel_back": six_hump_camel_back,
    "sphere_2": sphere_2,
    "sphere_3": sphere_3,
    "step": step,
    "step_2": step_2,
    "step_3": step_3,
    "stepint": stepint,
    "styblinski_tang": styblinski_tang,
    "three_hump_camel_back": three_hump_camel_back,
    "trid": trid,
    "weierstrass": weierstrass,
    "whitley": whitley,
    "zakharov": zakharov,
    "tension_spring": tension_spring,
    "welded_beam": welded_beam,
    "pressure_vessel": pressure_vessel,
    "pressure_vessel_discrete": pressure_vessel_discrete,
    "speed_reducer": speed_reducer,
    "three_bar_truss": three_bar_truss,
    "cantilever_beam": cantilever_beam,
    "gear_train": gear_train,
    "cec_2022_f01": cec_2022_f01,
    "cec_2022_f02": cec_2022_f02,
    "cec_2022_f03": cec_2022_f03,
    "cec_2022_f04": cec_2022_f04,
    "cec_2022_f05": cec_2022_f05,
    "cec_2022_f06": cec_2022_f06,
    "cec_2022_f07": cec_2022_f07,
    "cec_2022_f08": cec_2022_f08,
    "cec_2022_f09": cec_2022_f09,
    "cec_2022_f10": cec_2022_f10,
    "cec_2022_f11": cec_2022_f11,
    "cec_2022_f12": cec_2022_f12,
}


TEST_FUNCTIONS: Dict[str, Dict[str, str]] = {
    "ackley": {"name": "Ackley", "domain": "usually [-32.768, 32.768]^D", "optimum": "x*=0_D; f*=0"},
    "alpine_1": {"name": "Alpine 1", "domain": "usually [-10, 10]^D", "optimum": "x*=0_D; f*=0"},
    "alpine_2": {"name": "Alpine 2", "domain": "[0, 10]^D", "optimum": "x_i*≈7.917052698; f*≈-(2.808131180)^D in this minimization implementation"},
    "axis_parallel_hyper_ellipsoid": {"name": "Axis Parallel Hyper-Ellipsoid", "domain": "usually [-5.12, 5.12]^D", "optimum": "x*=0_D; f*=0"},
    "beale": {"name": "Beale", "domain": "[-4.5, 4.5]^2", "optimum": "x*=(3, 0.5); f*=0"},
    "bent_cigar": {"name": "Bent Cigar", "domain": "usually [-100, 100]^D", "optimum": "x*=0_D; f*=0"},
    "bohachevsky_1": {"name": "Bohachevsky F1", "domain": "[-100, 100]^2", "optimum": "x*=(0,0); f*=0"},
    "bohachevsky_2": {"name": "Bohachevsky F2", "domain": "[-100, 100]^2", "optimum": "x*=(0,0); f*=0"},
    "bohachevsky_3": {"name": "Bohachevsky F3", "domain": "[-100, 100]^2", "optimum": "x*=(0,0); f*=0"},
    "booth": {"name": "Booth", "domain": "[-10, 10]^2", "optimum": "x*=(1,3); f*=0"},
    "branin_rcos": {"name": "Branin RCOS", "domain": "x1∈[-5,10], x2∈[0,15]", "optimum": "f*=0.3978873577 at (-π,12.275), (π,2.275), (3π,2.475)"},
    "bukin_6": {"name": "Bukin F6", "domain": "x1∈[-15,-5], x2∈[-3,3]", "optimum": "x*=(-10,1); f*=0"},
    "chung_reynolds": {"name": "Chung-Reynolds", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "cosine_mixture": {"name": "Cosine Mixture", "domain": "usually [-1,1]^D", "optimum": "x*=0_D; f*=-0.1D for the minimization sign used here"},
    "cross_in_tray": {"name": "Cross in Tray", "domain": "[-10,10]^2", "optimum": "f*=-2.0626118708 at (±1.349406609, ±1.349406609)"},
    "csendes": {"name": "Csendes", "domain": "usually [-1,1]^D", "optimum": "x*=0_D; f*=0"},
    "de_jong_1": {"name": "De Jong F1 / Sphere", "domain": "usually [-5.12,5.12]^D", "optimum": "x*=0_D; f*=0"},
    "discus": {"name": "Discus", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "dixon_price": {"name": "Dixon-Price", "domain": "[-10,10]^D", "optimum": "x_i*=2^{-((2^i-2)/2^i)}, i=1..D; f*=0"},
    "drop_wave": {"name": "Drop Wave", "domain": "[-5.12,5.12]^2", "optimum": "x*=(0,0); f*=-1"},
    "easom": {"name": "Easom", "domain": "[-100,100]^2", "optimum": "x*=(π,π); f*=-1"},
    "eggholder": {"name": "Eggholder", "domain": "[-512,512]^2", "optimum": "x*≈(512,404.2319); f*≈-959.6407"},
    "elliptic": {"name": "Elliptic", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "expanded_griewank_plus_rosenbrock": {"name": "Expanded Griewank plus Rosenbrock", "domain": "usually [-5,5]^D", "optimum": "x*=1_D; f*=0"},
    "goldstein_price": {"name": "Goldstein-Price", "domain": "[-2,2]^2", "optimum": "x*=(0,-1); f*=3"},
    "griewangk_8": {"name": "Griewank", "domain": "[-600,600]^D", "optimum": "x*=0_D; f*=0"},
    "happy_cat": {"name": "Happy Cat", "domain": "usually [-100,100]^D", "optimum": "x*=-1_D; f*=0"},
    "hgbat": {"name": "HGBat", "domain": "usually [-100,100]^D", "optimum": "x*=-1_D; f*=0"},
    "himmelblau": {"name": "Himmelblau", "domain": "[-5,5]^2", "optimum": "f*=0 at (3,2), (-2.805118,3.131312), (-3.779310,-3.283186), (3.584428,-1.848126)"},
    "holder_table": {"name": "Hölder Table", "domain": "[-10,10]^2", "optimum": "f*≈-19.208502568 at (±8.055023472, ±9.664590029)"},
    "katsuura": {"name": "Katsuura", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "levy": {"name": "Levy", "domain": "[-10,10]^D", "optimum": "x*=1_D; f*=0"},
    "levi_13": {"name": "Levi F13", "domain": "[-10,10]^2", "optimum": "x*=(1,1); f*=0"},
    "matyas": {"name": "Matyas", "domain": "[-10,10]^2", "optimum": "x*=(0,0); f*=0"},
    "mccormick": {"name": "McCormick", "domain": "x1∈[-1.5,4], x2∈[-3,4]", "optimum": "x*≈(-0.54719756,-1.54719756); f*≈-1.913222955"},
    "michalewicz": {"name": "Michalewicz", "domain": "[0,π]^D", "optimum": "dimension- and m-dependent; for m=10: D=2 f*≈-1.8013, D=5 f*≈-4.6877, D=10 f*≈-9.6602"},
    "modified_schwefel": {"name": "Modified Schwefel", "domain": "usually [-100,100]^D for shifted CEC-style input", "optimum": "x*=0_D; f*=0"},
    "perm": {"name": "Perm 0,d,beta", "domain": "[-D,D]^D", "optimum": "x_i*=1/i; f*=0"},
    "pinter": {"name": "Pinter", "domain": "usually [-10,10]^D", "optimum": "x*=0_D; f*=0"},
    "powell": {"name": "Powell", "domain": "usually [-4,5]^D", "optimum": "D must be multiple of 4; x*=0_D; f*=0"},
    "qing": {"name": "Qing", "domain": "usually [-500,500]^D", "optimum": "x_i*=±sqrt(i); f*=0"},
    "quintic": {"name": "Quintic", "domain": "usually [-10,10]^D", "optimum": "each x_i∈{-1,2}; f*=0"},
    "rastrigin": {"name": "Rastrigin", "domain": "[-5.12,5.12]^D", "optimum": "x*=0_D; f*=0"},
    "ridge": {"name": "Ridge", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "rosenbrocks_valley": {"name": "Rosenbrock Valley", "domain": "usually [-5,10]^D", "optimum": "x*=1_D; f*=0"},
    "salomon": {"name": "Salomon", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "schaffer_2": {"name": "Schaffer F2", "domain": "[-100,100]^2", "optimum": "x*=(0,0); f*=0"},
    "schaffer_4": {"name": "Schaffer F4", "domain": "[-100,100]^2", "optimum": "f*≈0.292578632 at (0,±1.25313), (±1.25313,0)"},
    "schaffer_6": {"name": "Schaffer F6", "domain": "[-100,100]^2", "optimum": "x*=(0,0); f*=0"},
    "schumer_steiglitz": {"name": "Schumer-Steiglitz", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "schwefel": {"name": "Schwefel", "domain": "[-500,500]^D", "optimum": "x_i*=420.968746228; f*=0"},
    "schwefel_221": {"name": "Schwefel 2.21", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "schwefel_222": {"name": "Schwefel 2.22", "domain": "usually [-100,100]^D", "optimum": "x*=0_D; f*=0"},
    "six_hump_camel_back": {"name": "Six-Hump Camel Back", "domain": "x1∈[-3,3], x2∈[-2,2]", "optimum": "f*≈-1.031628453 at (0.089842,-0.712656), (-0.089842,0.712656)"},
    "sphere_2": {"name": "Sphere 2 / Sum of Different Powers", "domain": "usually [-1,1]^D", "optimum": "x*=0_D; f*=0"},
    "sphere_3": {"name": "Sphere 3 / Rotated Hyper-Ellipsoid", "domain": "usually [-65.536,65.536]^D", "optimum": "x*=0_D; f*=0"},
    "step": {"name": "Step", "domain": "usually [-100,100]^D", "optimum": "|x_i|<1; f*=0"},
    "step_2": {"name": "Step 2", "domain": "usually [-100,100]^D", "optimum": "-0.5≤x_i<0.5; f*=0"},
    "step_3": {"name": "Step 3", "domain": "usually [-100,100]^D", "optimum": "|x_i|<1; f*=0"},
    "stepint": {"name": "Stepint", "domain": "usually [-5.12,5.12]^D", "optimum": "bounded optimum f*=25-6D for x_i∈[-5.12,-5); unbounded below without bounds"},
    "styblinski_tang": {"name": "Styblinski-Tang", "domain": "[-5,5]^D", "optimum": "x_i*≈-2.903534028; f*≈-39.166165704D"},
    "three_hump_camel_back": {"name": "Three-Hump Camel Back", "domain": "[-5,5]^2", "optimum": "x*=(0,0); f*=0"},
    "trid": {"name": "Trid", "domain": "usually [-D^2,D^2]^D", "optimum": "x_i*=i(D+1-i); f*=-D(D+4)(D-1)/6"},
    "weierstrass": {"name": "Weierstrass", "domain": "usually [-0.5,0.5]^D", "optimum": "x*=0_D; f*=0"},
    "whitley": {"name": "Whitley", "domain": "usually [-10.24,10.24]^D", "optimum": "x*=1_D; f*=0"},
    "zakharov": {"name": "Zakharov", "domain": "[-5,10]^D", "optimum": "x*=0_D; f*=0"},
    "tension_spring": {"name": "Tension/compression spring design", "domain": "d∈[0.05,2], D∈[0.25,1.30], N∈[2,15]", "optimum": "x*≈(0.05169,0.35675,11.2871); f*≈0.012665; constrained"},
    "welded_beam": {"name": "Welded beam design", "domain": "h,l,t,b with bounds (0.1,0.1,0.1,0.1) to (2,10,10,2)", "optimum": "x*≈(0.20573,3.47049,9.03662,0.20573); f*≈1.724852; constrained"},
    "pressure_vessel": {"name": "Pressure vessel design, continuous relaxation", "domain": "Ts,Th∈[0,99], R∈[10,200], L∈[10,240]", "optimum": "x*≈(0.727591,0.359649,37.699012,240); f*≈5804.376217; constrained"},
    "pressure_vessel_discrete": {"name": "Pressure vessel design, discrete thickness", "domain": "Ts,Th multiples of 1/16; R∈[10,200], L∈[10,240]", "optimum": "x*≈(0.8125,0.4375,42.098446,176.636596); f*≈6059.714335; constrained/discrete"},
    "speed_reducer": {"name": "Speed reducer design", "domain": "7-variable bounded constrained engineering domain", "optimum": "x*≈(3.5,0.7,17,7.3,7.71532,3.35021,5.28665); f*≈2994.471066; constrained"},
    "three_bar_truss": {"name": "Three-bar truss design", "domain": "A1,A2∈[0,1]", "optimum": "x*≈(0.788675,0.408248); f*≈263.895843; constrained"},
    "cantilever_beam": {"name": "Cantilever beam design", "domain": "x_i∈[0.01,100], i=1..5", "optimum": "x*≈(6.016016,5.309173,4.494330,3.501475,2.152665); f*≈1.339956; constrained"},
    "gear_train": {"name": "Gear train design", "domain": "integer x_i∈[12,60], i=1..4", "optimum": "x*=(16,19,43,49); f*≈2.700857e-12; discrete"},
    "cec_2022_f01": {"name": "CEC 2022 F1", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=300 at official shifted optimum"},
    "cec_2022_f02": {"name": "CEC 2022 F2", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=400 at official shifted optimum"},
    "cec_2022_f03": {"name": "CEC 2022 F3", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=600 at official shifted optimum"},
    "cec_2022_f04": {"name": "CEC 2022 F4", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=800 at official shifted optimum"},
    "cec_2022_f05": {"name": "CEC 2022 F5", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=900 at official shifted optimum"},
    "cec_2022_f06": {"name": "CEC 2022 F6", "domain": "D∈{10,20}; official bounds/data", "optimum": "f*=1800 at official shifted optimum"},
    "cec_2022_f07": {"name": "CEC 2022 F7", "domain": "D∈{10,20}; official bounds/data", "optimum": "f*=2000 at official shifted optimum"},
    "cec_2022_f08": {"name": "CEC 2022 F8", "domain": "D∈{10,20}; official bounds/data", "optimum": "f*=2200 at official shifted optimum"},
    "cec_2022_f09": {"name": "CEC 2022 F9", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=2300 at official shifted optimum"},
    "cec_2022_f10": {"name": "CEC 2022 F10", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=2400 at official shifted optimum"},
    "cec_2022_f11": {"name": "CEC 2022 F11", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=2600 at official shifted optimum"},
    "cec_2022_f12": {"name": "CEC 2022 F12", "domain": "D∈{2,10,20}; official bounds/data", "optimum": "f*=2700 at official shifted optimum"},
}


def list_test_functions(include_cec: bool = True, include_engineering: bool = True) -> List[str]:
    names = sorted(FUNCTIONS.keys())
    if not include_cec:
        names = [name for name in names if not name.startswith("cec_2022_")]
    if not include_engineering:
        engineering_ids = set(ENGINEERING_BENCHMARKS.keys())
        names = [name for name in names if name not in engineering_ids]
    return names


def get_test_function(name: str) -> Callable[..., float]:
    key = str(name).strip().lower()
    if key not in FUNCTIONS:
        raise KeyError(f"Unknown test function: {name}. Available: {', '.join(list_test_functions())}")
    return FUNCTIONS[key]


def get_test_function_info(name: str) -> Dict[str, str]:
    key = str(name).strip().lower()
    if key not in TEST_FUNCTIONS:
        raise KeyError(f"Unknown test function metadata: {name}. Available: {', '.join(sorted(TEST_FUNCTIONS))}")
    return dict(TEST_FUNCTIONS[key])



_CEC2022_BIASES: Mapping[int, float] = {
    1: 300.0,
    2: 400.0,
    3: 600.0,
    4: 800.0,
    5: 900.0,
    6: 1800.0,
    7: 2000.0,
    8: 2200.0,
    9: 2300.0,
    10: 2400.0,
    11: 2600.0,
    12: 2700.0,
}


def get_cec2022_optimum(func_num: int, dimension: int = 10) -> tuple[np.ndarray, float]:
    """
    Return the official shifted optimizer and biased optimum value for CEC 2022.

    Parameters
    ----------
    func_num
        CEC 2022 function number in {1, ..., 12}.
    dimension
        Supported dimension. F1-F5 and F9-F12 support 2, 10, and 20.
        F6-F8 support 10 and 20.

    Returns
    -------
    tuple
        ``(x_star, f_star)`` where ``x_star`` is obtained from the official
        shift-data file and ``f_star`` is the official bias/optimum value.
    """
    if func_num not in _CEC2022_BIASES:
        raise ValueError("func_num must be in {1, ..., 12}.")
    if dimension not in _CEC2022_DIMENSIONS:
        raise ValueError(f"CEC 2022 dimensions must be one of {_CEC2022_DIMENSIONS}. Got {dimension}.")
    if dimension == 2 and func_num in _CEC2022_NO_D2:
        raise ValueError("CEC 2022 F6, F7, and F8 are not defined for D=2.")

    data_source = _resolve_cec2022_data_source()
    raw = np.asarray(_loadtxt_cec2022(data_source, f"shift_data_{func_num}.txt"), dtype=float)

    if func_num < 9:
        # Basic and hybrid functions use one shifted optimum vector.  Some
        # official files are stored as 10 x 100 matrices; the first row is the
        # optimizer used by the benchmark definition.
        if raw.ndim == 1:
            x_star = raw.reshape(-1)[:dimension]
        else:
            x_star = raw[0, :dimension]
    else:
        # For composition functions, the global optimum is the first component
        # shift vector because the first component has zero internal bias.
        if raw.ndim == 1:
            raw = raw.reshape(10, -1)
        x_star = raw[0, :dimension]

    return np.asarray(x_star, dtype=float), float(_CEC2022_BIASES[func_num])


def validate_cec2022_optima(tol: float = 1e-7, dimensions: Iterable[int] = _CEC2022_DIMENSIONS) -> Dict[str, float]:
    """
    Validate all supported CEC 2022 functions at their official shifted optima.

    Returns
    -------
    dict
        Mapping ``cec_2022_fXX_DD`` to absolute error.
    """
    errors: Dict[str, float] = {}

    for dimension in dimensions:
        if dimension not in _CEC2022_DIMENSIONS:
            raise ValueError(f"Unsupported CEC 2022 dimension: {dimension}.")

        for func_num in range(1, 13):
            if dimension == 2 and func_num in _CEC2022_NO_D2:
                continue

            x_star, f_star = get_cec2022_optimum(func_num, dimension)
            value = _cec2022_eval(x_star, func_num)
            key = f"cec_2022_f{func_num:02d}_D{dimension}"
            err = abs(float(value) - float(f_star))
            errors[key] = err

            if err > tol:
                raise AssertionError(f"{key}: expected {f_star}, got {value}, error={err}")

    return errors


def validate_known_optima(tol: float = 1e-7, include_cec: bool = False) -> Dict[str, float]:
    """
    Evaluate a representative known global optimizer for each supported function.

    Returns
    -------
    dict
        Mapping function ID to absolute error against the documented optimum.

    Notes
    -----
    CEC functions are skipped by default because they require the official external
    shift/rotation/shuffle data files. If ``include_cec=True``, the CEC functions
    are checked at their official shifted optima.
    """
    tests = {
        "ackley": ([0, 0, 0], 0.0),
        "alpine_1": ([0, 0, 0], 0.0),
        "alpine_2": ([7.917052698245946, 7.917052698245946], -(2.8081311800070052**2)),
        "axis_parallel_hyper_ellipsoid": ([0, 0, 0], 0.0),
        "beale": ([3, 0.5], 0.0),
        "bent_cigar": ([0, 0, 0], 0.0),
        "bohachevsky_1": ([0, 0], 0.0),
        "bohachevsky_2": ([0, 0], 0.0),
        "bohachevsky_3": ([0, 0], 0.0),
        "booth": ([1, 3], 0.0),
        "branin_rcos": ([-np.pi, 12.275], 0.39788735772973816),
        "bukin_6": ([-10, 1], 0.0),
        "chung_reynolds": ([0, 0, 0], 0.0),
        "cosine_mixture": ([0, 0, 0], -0.3),
        "cross_in_tray": ([1.349406608602084, 1.349406608602084], -2.062611870822739),
        "csendes": ([0, 0, 0], 0.0),
        "de_jong_1": ([0, 0, 0], 0.0),
        "discus": ([0, 0, 0], 0.0),
        "dixon_price": ([1.0, 1.0 / np.sqrt(2.0), 2.0**(-6.0/8.0)], 0.0),
        "drop_wave": ([0, 0], -1.0),
        "easom": ([np.pi, np.pi], -1.0),
        "eggholder": ([512, 404.2319], -959.6406627106155),
        "elliptic": ([0, 0, 0], 0.0),
        "expanded_griewank_plus_rosenbrock": ([1, 1, 1], 0.0),
        "goldstein_price": ([0, -1], 3.0),
        "griewangk_8": ([0, 0, 0], 0.0),
        "happy_cat": ([-1, -1, -1], 0.0),
        "hgbat": ([-1, -1, -1], 0.0),
        "himmelblau": ([3, 2], 0.0),
        "holder_table": ([8.055023472141116, 9.664590028909654], -19.20850256788675),
        "katsuura": ([0, 0, 0], 0.0),
        "levy": ([1, 1, 1], 0.0),
        "levi_13": ([1, 1], 0.0),
        "matyas": ([0, 0], 0.0),
        "mccormick": ([-0.5471975602214493, -1.5471975602214493], -1.9132229549810367),
        "modified_schwefel": ([0, 0, 0], 0.0),
        "perm": ([1.0, 0.5, 1.0/3.0], 0.0),
        "pinter": ([0, 0, 0], 0.0),
        "powell": ([0, 0, 0, 0], 0.0),
        "qing": ([1.0, np.sqrt(2.0), np.sqrt(3.0)], 0.0),
        "quintic": ([-1, 2, -1], 0.0),
        "rastrigin": ([0, 0, 0], 0.0),
        "ridge": ([0, 0, 0], 0.0),
        "rosenbrocks_valley": ([1, 1, 1], 0.0),
        "salomon": ([0, 0, 0], 0.0),
        "schaffer_2": ([0, 0], 0.0),
        "schaffer_4": ([0, 1.25313], 0.29257863204552975),
        "schaffer_6": ([0, 0], 0.0),
        "schumer_steiglitz": ([0, 0, 0], 0.0),
        "schwefel": ([420.9687462275036, 420.9687462275036], 0.0),
        "schwefel_221": ([0, 0, 0], 0.0),
        "schwefel_222": ([0, 0, 0], 0.0),
        "six_hump_camel_back": ([0.08984201368301331, -0.7126564032704135], -1.031628453489877),
        "sphere_2": ([0, 0, 0], 0.0),
        "sphere_3": ([0, 0, 0], 0.0),
        "step": ([0, 0, 0], 0.0),
        "step_2": ([0, 0, 0], 0.0),
        "step_3": ([0, 0, 0], 0.0),
        "stepint": ([-5.1, -5.1], 13.0),
        "styblinski_tang": ([-2.903534027771177, -2.903534027771177], -78.33233140754282),
        "three_hump_camel_back": ([0, 0], 0.0),
        "trid": ([3, 4, 3], -7.0),
        "weierstrass": ([0, 0, 0], 0.0),
        "whitley": ([1, 1, 1], 0.0),
        "zakharov": ([0, 0, 0], 0.0),
    }

    cec_errors: Dict[str, float] = {}
    if include_cec:
        cec_errors = validate_cec2022_optima(tol=tol)

    errors: Dict[str, float] = {}
    for name, (x, expected) in tests.items():
        value = FUNCTIONS[name](x)
        err = abs(float(value) - float(expected))
        errors[name] = err
        if err > tol:
            raise AssertionError(f"{name}: expected {expected}, got {value}, error={err}")

    errors.update(cec_errors)
    return errors


if __name__ == "__main__":
    errs = validate_known_optima()
    eng = validate_engineering_benchmarks()
    print(f"Validated {len(errs)} analytic non-CEC functions. Max absolute error = {max(errs.values()):.3e}")
    print(
        f"Validated {len(eng)} engineering benchmarks. "
        f"Max objective error = {max(v['objective_error'] for v in eng.values()):.3e}; "
        f"max violation = {max(v['max_violation'] for v in eng.values()):.3e}"
    )
