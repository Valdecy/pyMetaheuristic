from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Callback:
    """Base callback with four hook points and cooperative stop support."""

    def __init__(self) -> None:
        self.engine = None

    def set_engine(self, engine) -> None:
        self.engine = engine

    # compatibility alias
    set_algorithm = set_engine

    def before_run(self, **kwargs) -> None:
        pass

    def after_run(self, **kwargs) -> None:
        pass

    def before_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        pass

    def after_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        pass

    def stop(self, reason: str = "callback_stop") -> None:
        if self.engine is not None:
            self.engine.request_stop(reason)


class CallbackList(Callback):
    """Container that chains multiple callbacks transparently."""

    def __init__(self, callbacks=None) -> None:
        super().__init__()
        self.callbacks = []
        for callback in callbacks or []:
            self.append(callback)

    @classmethod
    def from_any(cls, callbacks):
        if callbacks is None:
            return cls([])
        if isinstance(callbacks, cls):
            return callbacks
        if isinstance(callbacks, Callback):
            return cls([callbacks])
        return cls(list(callbacks))

    def append(self, callback) -> None:
        if not isinstance(callback, Callback):
            raise TypeError("All callbacks must inherit from Callback")
        self.callbacks.append(callback)
        if self.engine is not None:
            callback.set_engine(self.engine)

    def set_engine(self, engine) -> None:
        self.engine = engine
        for callback in self.callbacks:
            callback.set_engine(engine)

    set_algorithm = set_engine

    def before_run(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.before_run(**kwargs)

    def after_run(self, **kwargs) -> None:
        for callback in self.callbacks:
            callback.after_run(**kwargs)

    def before_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        for callback in self.callbacks:
            callback.before_iteration(population, fitness, best_x, best_fitness, **kwargs)

    def after_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        for callback in self.callbacks:
            callback.after_iteration(population, fitness, best_x, best_fitness, **kwargs)


@dataclass
class HistoryRecorder(Callback):
    """Simple callback that stores shallow copies of observations."""

    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        Callback.__init__(self)

    def after_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        observation = dict(kwargs.get("observation") or {})
        if observation:
            self.records.append(observation)


@dataclass
class ProgressPrinter(Callback):
    every: int = 1
    prefix: str = ""

    def __post_init__(self) -> None:
        Callback.__init__(self)
        self.every = max(1, int(self.every))

    def after_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        state = kwargs.get("state")
        if state is None or state.step % self.every != 0:
            return
        obs = dict(kwargs.get("observation") or {})
        label = self.prefix or f"[{getattr(self.engine, 'algorithm_id', 'run')}]"
        print(f"{label} step={state.step:5d} evals={state.evaluations:7d} best={best_fitness:.6g}")
        if obs.get("diversity") is not None:
            print(f"{label} diversity={obs['diversity']:.6g}")


@dataclass
class EarlyStopping(Callback):
    """Callback-driven early stopping based on patience/min_delta."""

    patience: int = 20
    min_delta: float = 1e-12
    reason: str = "callback_early_stopping"

    def __post_init__(self) -> None:
        Callback.__init__(self)
        self._best = None
        self._bad_steps = 0

    def after_iteration(self, population, fitness, best_x, best_fitness, **kwargs) -> None:
        objective = getattr(getattr(self.engine, 'problem', None), 'objective', 'min')
        if self._best is None:
            self._best = float(best_fitness)
            self._bad_steps = 0
            return
        improved = (best_fitness < self._best - self.min_delta) if objective == 'min' else (best_fitness > self._best + self.min_delta)
        if improved:
            self._best = float(best_fitness)
            self._bad_steps = 0
            return
        self._bad_steps += 1
        if self._bad_steps >= max(1, int(self.patience)):
            self.stop(self.reason)
