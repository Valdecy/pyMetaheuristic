from numpy import cos, exp, pi


def easom(variables_values: tuple[float, float]) -> float:
    x1, x2 = variables_values
    return -cos(x1) * cos(x2) * exp(-(x1 - pi) ** 2 - (x2 - pi) ** 2)
