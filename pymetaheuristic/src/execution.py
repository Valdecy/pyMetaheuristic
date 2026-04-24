from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import concurrent.futures as cf
import multiprocessing as mp


@dataclass
class ChunkExecutionResult:
    label: str
    state: Any
    observations: list[dict[str, Any]]
    steps_taken: int
    stopped: bool


def _evolve_engine_chunk(label: str, engine, state, n_steps: int) -> ChunkExecutionResult:
    observations: list[dict[str, Any]] = []
    current_state = state
    steps_taken = 0
    target_steps = max(0, int(n_steps))
    for _ in range(target_steps):
        if engine.should_stop(current_state):
            break
        current_state = engine.step(current_state)
        observations.append(dict(engine.observe(current_state)))
        steps_taken += 1
    return ChunkExecutionResult(
        label=label,
        state=current_state,
        observations=observations,
        steps_taken=steps_taken,
        stopped=engine.should_stop(current_state),
    )


def run_engine_chunks(
    chunk_items: list[tuple[str, Any, Any, int]],
    execution_backend: str = "serial",
    n_jobs: int | None = None,
    fallback_to_serial: bool = True,
) -> tuple[list[ChunkExecutionResult], str, str | None]:
    if not chunk_items:
        return [], "serial", None

    backend = (execution_backend or "serial").lower()
    if backend not in {"serial", "process"}:
        raise ValueError(f"Unsupported execution_backend: {execution_backend}. Use 'serial' or 'process'.")

    if backend == "serial" or len(chunk_items) <= 1:
        return ([_evolve_engine_chunk(*item) for item in chunk_items], "serial", None)

    max_workers = max(1, min(int(n_jobs or len(chunk_items)), len(chunk_items)))
    try:
        ctx = mp.get_context("spawn")
        ordered: dict[str, ChunkExecutionResult] = {}
        with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_evolve_engine_chunk, *item) for item in chunk_items]
            for fut in futures:
                result = fut.result()
                ordered[result.label] = result
        results = [ordered[item[0]] for item in chunk_items]
        return results, "process", None
    except Exception as exc:
        if not fallback_to_serial:
            raise
        warning = f"Parallel execution failed and fell back to serial mode: {exc}"
        return ([_evolve_engine_chunk(*item) for item in chunk_items], "serial", warning)
