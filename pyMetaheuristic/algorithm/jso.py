"""
Artificial Jellyfish Search Optimizer
ref: Chou, J. S. , and D. N. Truong . "A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean."
    Applied Mathematics and Computation 389(2021):125535.
"""

import numpy as np
from numpy.random import rand

from ..utils import Target, optimize


def initialize(target: Target, n_pop: int, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """use logistic chaos map to generate initial population"""
    func, _, lb, ub = target
    # generate logistic chaos values
    while (x := rand()) in {0.0, 0.25, 0.5, 0.75, 1.0}:
        continue
    chaos = []
    for i in range(n_pop):
        chaos.append(x)
        x = eta * x * (1 - x)
    chaos = np.array(chaos)

    pop = np.array(
        [chaos[i] * (ub - lb) + lb for i in range(n_pop)]
    )
    cost = np.array(
        [func(idt) for idt in pop]
    )
    return pop, cost


def bounded(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """boundary conditions"""
    x[x > ub] = ((x - ub) + lb)[x > ub]
    x[x < lb] = ((x - lb) + ub)[x < lb]
    return x


def ocean_current(x: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """let individual x move by trend"""
    return x + rand() * trend


def passive_move(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, gamma: float) -> np.ndarray:
    """individual x do passive motion"""
    return x + gamma * rand() * (ub - lb)


def active_move(xi: np.ndarray, fi: float, xj: np.ndarray, fj: float) -> np.ndarray:
    """individual xi do active motion towards or backwards xj"""
    if fi >= fj:
        direction = xj - xi
    else:
        direction = xi - xj
    step = rand() * direction
    return xi + step


def jso(
        target: Target,
        *,
        n_pop: int,
        iter_max: int,
        gamma: float = 0.1,
        beta: float = 3,
        c_0: float = 0.5,
        eta: float = 4
):
    func, _, lb, ub = target
    pop, cost = initialize(target, n_pop, eta)  # population and cost
    i = np.argmin(cost)
    opt = np.copy(pop[i])  # historical optimum
    opt_cost = cost[i]
    yield opt, opt_cost

    t = 1  # time / round
    while t < iter_max:
        interest = np.arange(n_pop)
        np.random.shuffle(interest)
        for i in range(n_pop):
            # time control
            c = np.abs(
                (1 - t / iter_max) * (2 * rand() - 1)
            )
            if c >= c_0:  # jellyfish follows ocean current
                e_c = beta * rand()
                mu = np.mean(pop, axis=0)
                trend = opt - e_c * mu
                pop[i] = ocean_current(pop[i], trend)
            else:  # jellyfish moves inside swarm
                if rand() > (1 - c):  # do passive move
                    pop[i] = passive_move(pop[i], lb, ub, gamma)
                else:  # do active move
                    j = interest[i]
                    pop[i] = active_move(pop[i], cost[i], pop[j], cost[j])
            pop[i] = bounded(pop[i], lb, ub)
            cost[i] = func(pop[i])
            if cost[i] < opt_cost:
                opt[:] = pop[i]  # copy instead of view
                opt_cost = cost[i]

        yield opt, opt_cost  # yield opt for this round
        t += 1


def artificial_jellyfish_search_optimizer(*args, **kwargs):
    """conventional entrance for jso"""
    return optimize(jso(*args, **kwargs), verbose=True)
