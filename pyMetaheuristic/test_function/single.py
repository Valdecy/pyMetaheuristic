import numpy as np

#
def easom(variables_values: tuple[float, float]) -> float:
    x1, x2 = variables_values
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - pi) ** 2)
