from typing import Iterable, Callable
# from collections import namedtuple

import numpy as np

# (func, dimension, lower boundary, upper boundary)
Target = tuple[Callable, int, np.ndarray, np.ndarray]
# Target = namedtuple(
#     "Target",
#     ["func", "dim", "lb", "ub"]
# )


def optimize(optimizer: Iterable, *, verbose: bool = False) -> tuple[np.array, float]:
    opt = None
    val = None
    for i, (x, y) in enumerate(optimizer):
        opt = x
        val = y
        if verbose:
            print(f"Iteration = {i}, optimum = {opt}, value = {val}")
    return opt, val


def conditional_rand(condition: Callable[[float], bool], *args):
    """returns random value satisfy the condition"""
    while True:
        r = np.random.rand(*args)
        if np.all(np.vectorize(condition)(r)):
            break
    return r
