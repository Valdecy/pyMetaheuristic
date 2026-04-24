from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "full_array",
    "Problem",
    "FunctionalProblem",
    "SphereProblem",
    "RastriginProblem",
    "AckleyProblem",
    "RosenbrockProblem",
    "ZakharovProblem",
    "TEST_PROBLEM_REGISTRY",
    "get_test_problem",
]


def full_array(value, dimension: int) -> list[float]:
    if np.isscalar(value):
        return [float(value)] * int(dimension)
    arr = list(value)
    if len(arr) != int(dimension):
        raise ValueError(f"Expected {dimension} bounds, received {len(arr)}")
    return [float(v) for v in arr]


@dataclass
class Problem(ABC):
    dimension: int
    lower: list[float] | float
    upper: list[float] | float
    name: str = "problem"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        self.lower = full_array(self.lower, self.dimension)
        self.upper = full_array(self.upper, self.dimension)

    def __call__(self, x) -> float:
        values = np.asarray(x, dtype=float).tolist()
        return float(self.evaluate(values))

    @abstractmethod
    def evaluate(self, x) -> float:
        raise NotImplementedError

    @staticmethod
    def latex_code() -> str:
        return r"f(x)"

    def to_problem_spec(self, objective: str = "min", constraints=None, constraint_handler=None, variable_types=None, metadata=None):
        from ..engines.protocol import ProblemSpec
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return ProblemSpec(
            target_function=self,
            min_values=list(self.lower),
            max_values=list(self.upper),
            objective=objective,
            constraints=constraints,
            constraint_handler=constraint_handler,
            variable_types=variable_types,
            metadata=merged,
        )


@dataclass
class FunctionalProblem(Problem):
    function: Callable[[list[float]], float] = lambda x: 0.0
    latex: str = r"f(x)"

    def evaluate(self, x) -> float:
        return float(self.function(list(x)))

    def latex_expression(self) -> str:
        return str(self.latex)


@dataclass
class SphereProblem(Problem):
    name: str = "sphere"

    def __init__(self, dimension: int = 2, lower=-5.12, upper=5.12, name: str = "sphere", metadata=None):
        super().__init__(dimension=dimension, lower=lower, upper=upper, name=name, metadata=metadata or {})

    def evaluate(self, x) -> float:
        return float(np.sum(np.square(x)))

    @staticmethod
    def latex_code() -> str:
        return r"f(x)=\sum_{i=1}^{n} x_i^2"


@dataclass
class RastriginProblem(Problem):
    name: str = "rastrigin"

    def __init__(self, dimension: int = 2, lower=-5.12, upper=5.12, name: str = "rastrigin", metadata=None):
        super().__init__(dimension=dimension, lower=lower, upper=upper, name=name, metadata=metadata or {})

    def evaluate(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    @staticmethod
    def latex_code() -> str:
        return r"f(x)=10n + \sum_{i=1}^{n} \left(x_i^2 - 10\cos(2\pi x_i)\right)"


@dataclass
class AckleyProblem(Problem):
    name: str = "ackley"

    def __init__(self, dimension: int = 2, lower=-5.0, upper=5.0, name: str = "ackley", metadata=None):
        super().__init__(dimension=dimension, lower=lower, upper=upper, name=name, metadata=metadata or {})

    def evaluate(self, x) -> float:
        x = np.asarray(x, dtype=float)
        n = len(x)
        a = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        b = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        return float(a + b + np.e + 20.0)

    @staticmethod
    def latex_code() -> str:
        return r"f(x)=-20\exp\!\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right)-\exp\!\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right)+e+20"


@dataclass
class RosenbrockProblem(Problem):
    name: str = "rosenbrock"

    def __init__(self, dimension: int = 2, lower=-5.0, upper=10.0, name: str = "rosenbrock", metadata=None):
        super().__init__(dimension=dimension, lower=lower, upper=upper, name=name, metadata=metadata or {})

    def evaluate(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    @staticmethod
    def latex_code() -> str:
        return r"f(x)=\sum_{i=1}^{n-1}\left[100\left(x_{i+1}-x_i^2\right)^2 + (1-x_i)^2\right]"


@dataclass
class ZakharovProblem(Problem):
    name: str = "zakharov"

    def __init__(self, dimension: int = 2, lower=-5.0, upper=10.0, name: str = "zakharov", metadata=None):
        super().__init__(dimension=dimension, lower=lower, upper=upper, name=name, metadata=metadata or {})

    def evaluate(self, x) -> float:
        x = np.asarray(x, dtype=float)
        i = np.arange(1, len(x) + 1)
        s = np.sum(0.5 * i * x)
        return float(np.sum(x**2) + s**2 + s**4)

    @staticmethod
    def latex_code() -> str:
        return r"f(x)=\sum_{i=1}^{n}x_i^2 + \left(\sum_{i=1}^{n}\frac{i x_i}{2}\right)^2 + \left(\sum_{i=1}^{n}\frac{i x_i}{2}\right)^4"


TEST_PROBLEM_REGISTRY = {
    "sphere": SphereProblem,
    "de_jong_1": SphereProblem,
    "ackley": AckleyProblem,
    "rastrigin": RastriginProblem,
    "rosenbrock": RosenbrockProblem,
    "rosenbrocks_valley": RosenbrockProblem,
    "zakharov": ZakharovProblem,
}


TEST_FUNCTION_PROBLEM_SPECS = {
    "alpine_1": {"lower": -10.0, "upper": 10.0, "latex": r"f(x)=\sum_{i=1}^{n}|\sin(x_i)+0.1x_i|", "origin": "https://arxiv.org/abs/1308.4008"},
    "alpine_2": {"lower": 0.0, "upper": 10.0, "latex": r"f(x)=\prod_{i=1}^{n}\sqrt{x_i}\sin(x_i)", "origin": "https://arxiv.org/abs/1308.4008"},
    "bent_cigar": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=x_1^2+10^6\sum_{i=2}^{n}x_i^2", "origin": "http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf"},
    "chung_reynolds": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=(\sum_{i=1}^{n}x_i^2)^2", "origin": "https://arxiv.org/abs/1308.4008"},
    "cosine_mixture": {"lower": -1.0, "upper": 1.0, "latex": r"f(x)=-0.1\sum_{i=1}^{n}\cos(5\pi x_i)-\sum_{i=1}^{n}x_i^2", "origin": "http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture"},
    "csendes": {"lower": -1.0, "upper": 1.0, "latex": r"f(x)=\sum_{i=1}^{n}x_i^6(2+\sin(1/x_i))", "origin": "https://arxiv.org/abs/1308.4008"},
    "discus": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=10^6x_1^2+\sum_{i=2}^{n}x_i^2", "origin": "http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf"},
    "dixon_price": {"lower": -10.0, "upper": 10.0, "latex": r"f(x)=(x_1-1)^2+\sum_{i=2}^{n} i(2x_i^2-x_{i-1})^2", "origin": "https://www.sfu.ca/~ssurjano/dixonpr.html"},
    "elliptic": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum_{i=1}^{n}10^{6(i-1)/(n-1)}x_i^2", "origin": "http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf"},
    "expanded_griewank_plus_rosenbrock": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum (g_i^2/4000-\cos(g_i)+1)", "origin": "http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf"},
    "happy_cat": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=|\|x\|^2-n|^{1/4}+(0.5\|x\|^2+\sum x_i)/n+0.5", "origin": "http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf"},
    "hgbat": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sqrt{|(\sum x_i^2)^2-(\sum x_i)^2|}+(0.5\sum x_i^2+\sum x_i)/n+0.5", "origin": "http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf"},
    "katsuura": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\prod_{i=1}^{n}(1+i\sum_{k=1}^{32}|2^k x_i-\lfloor 2^k x_i+0.5\rfloor|/2^k)^{10/n^{1.2}}-10/n^2", "origin": "https://www.geocities.ws/eadorio/mvf.pdf"},
    "levy": {"lower": -10.0, "upper": 10.0, "latex": r"f(x)=\sin^2(\pi w_1)+\sum_{i=1}^{n-1}(w_i-1)^2(1+10\sin^2(\pi w_i+1))+(w_n-1)^2(1+\sin^2(2\pi w_n))", "origin": "https://www.sfu.ca/~ssurjano/levy.html"},
    "michalewicz": {"lower": 0.0, "upper": float(np.pi), "latex": r"f(x)=-\sum_{i=1}^{n}\sin(x_i)\sin^{2m}(ix_i^2/\pi)", "origin": "https://www.sfu.ca/~ssurjano/michal.html"},
    "perm": {"lower": lambda d: -int(d), "upper": lambda d: int(d), "latex": r"f(x)=\sum_{j=1}^{n}[\sum_{i=1}^{n}(i+\beta)(x_i^j-(1/i)^j)]^2", "origin": "https://www.sfu.ca/~ssurjano/perm0db.html"},
    "pinter": {"lower": -10.0, "upper": 10.0, "latex": r"f(x)=\sum ix_i^2+20i\sin^2(A_i)+i\log_{10}(1+ib_i^2)", "origin": "https://arxiv.org/abs/1308.4008"},
    "powell": {"lower": -4.0, "upper": 5.0, "latex": r"f(x)=\sum[(x_{4i-3}+10x_{4i-2})^2+5(x_{4i-1}-x_{4i})^2+(x_{4i-2}-2x_{4i-1})^4+10(x_{4i-3}-x_{4i})^4]", "origin": "https://www.sfu.ca/~ssurjano/powell.html"},
    "qing": {"lower": -500.0, "upper": 500.0, "latex": r"f(x)=\sum_{i=1}^{n}(x_i^2-i)^2", "origin": "https://arxiv.org/abs/1308.4008"},
    "quintic": {"lower": -10.0, "upper": 10.0, "latex": r"f(x)=\sum |x_i^5-3x_i^4+4x_i^3+2x_i^2-10x_i-4|", "origin": "https://arxiv.org/abs/1308.4008"},
    "ridge": {"lower": -64.0, "upper": 64.0, "latex": r"f(x)=\sum_{i=1}^{n}(\sum_{j=1}^{i}x_j)^2", "origin": "http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html"},
    "salomon": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=1-\cos(2\pi\|x\|)+0.1\|x\|", "origin": "https://arxiv.org/abs/1308.4008"},
    "schumer_steiglitz": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum_{i=1}^{n}x_i^4", "origin": "https://arxiv.org/abs/1308.4008"},
    "schwefel_221": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\max_i |x_i|", "origin": "https://arxiv.org/abs/1308.4008"},
    "schwefel_222": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum |x_i|+\prod |x_i|", "origin": "https://arxiv.org/abs/1308.4008"},
    "modified_schwefel": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=418.9829n-\sum z_i\sin(\sqrt{|z_i|})", "origin": "http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf"},
    "sphere_2": {"lower": -1.0, "upper": 1.0, "latex": r"f(x)=\sum_{i=1}^{n}|x_i|^{i+1}", "origin": "https://www.sfu.ca/~ssurjano/sumpow.html"},
    "sphere_3": {"lower": -65.536, "upper": 65.536, "latex": r"f(x)=\sum_{i=1}^{n}\sum_{j=1}^{i}x_j^2", "origin": "https://www.sfu.ca/~ssurjano/rothyp.html"},
    "step": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum \lfloor|x_i|\rfloor", "origin": "https://arxiv.org/abs/1308.4008"},
    "step_2": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum \lfloor x_i+0.5\rfloor^2", "origin": "https://arxiv.org/abs/1308.4008"},
    "step_3": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum \lfloor x_i^2\rfloor", "origin": "https://arxiv.org/abs/1308.4008"},
    "stepint": {"lower": -5.12, "upper": 5.12, "latex": r"f(x)=25+\sum \lfloor x_i\rfloor", "origin": "https://arxiv.org/abs/1308.4008"},
    "trid": {"lower": lambda d: -(int(d) ** 2), "upper": lambda d: int(d) ** 2, "latex": r"f(x)=\sum(x_i-1)^2-\sum x_ix_{i-1}", "origin": "https://www.sfu.ca/~ssurjano/trid.html"},
    "weierstrass": {"lower": -100.0, "upper": 100.0, "latex": r"f(x)=\sum_{i=1}^{n}\sum_{k=0}^{k_{max}}a^k\cos(2\pi b^k(x_i+0.5))-n\sum_{k=0}^{k_{max}}a^k\cos(\pi b^k)", "origin": "http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf"},
    "whitley": {"lower": -10.24, "upper": 10.24, "latex": r"f(x)=\sum(q_{ij}^2/4000-\cos q_{ij}+1)", "origin": "https://arxiv.org/abs/1308.4008"},
}



def _build_functional_problem(key: str, dimension: int, lower=None, upper=None):
    from ..test_functions import get_test_function
    spec = TEST_FUNCTION_PROBLEM_SPECS[key]
    return FunctionalProblem(
        dimension=int(dimension),
        lower=(spec["lower"](dimension) if callable(spec["lower"]) else spec["lower"]) if lower is None else lower,
        upper=(spec["upper"](dimension) if callable(spec["upper"]) else spec["upper"]) if upper is None else upper,
        name=key,
        function=get_test_function(key),
        latex=spec["latex"],
        metadata={"origin": spec.get("origin"), "source": "Derived wrapper"},
    )


def list_test_problems():
    return sorted(set(TEST_PROBLEM_REGISTRY) | set(TEST_FUNCTION_PROBLEM_SPECS))


def get_test_problem(name: str, dimension: int = 2, lower=None, upper=None):
    key = str(name).strip().lower()
    if key in TEST_PROBLEM_REGISTRY:
        cls = TEST_PROBLEM_REGISTRY[key]
        kwargs = {"dimension": int(dimension)}
        if lower is not None:
            kwargs["lower"] = lower
        if upper is not None:
            kwargs["upper"] = upper
        return cls(**kwargs)
    if key in TEST_FUNCTION_PROBLEM_SPECS:
        return _build_functional_problem(key, dimension=dimension, lower=lower, upper=upper)
    raise KeyError(f"Unknown problem class: {name}. Available: {', '.join(list_test_problems())}")
