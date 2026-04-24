from __future__ import annotations

from dataclasses import asdict
from typing import Any
import math
import random

import numpy as np

from .engines import CandidateRecord
from .schemas import ActionSpec, ActionOutcome, DecisionPlan


def select_candidates(engine, state, mode: str = "best", k: int = 1) -> list[CandidateRecord]:
    k = max(1, int(k))
    return list(engine.export_candidates(state, k=k, mode=mode))



def perturb_candidates(
    candidates: list[CandidateRecord],
    sigma: float,
    min_values: list[float],
    max_values: list[float],
    seed: int | None = None,
) -> list[CandidateRecord]:
    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    lo = np.asarray(min_values, dtype=float)
    hi = np.asarray(max_values, dtype=float)
    out: list[CandidateRecord] = []
    for cand in candidates:
        pos = np.asarray(cand.position, dtype=float)
        eps = rng.normal(0.0, float(sigma), size=pos.shape)
        new_pos = np.clip(pos + eps, lo, hi)
        out.append(CandidateRecord(
            position=new_pos.tolist(),
            fitness=float(cand.fitness),
            source_algorithm=cand.source_algorithm,
            source_step=cand.source_step,
            role=cand.role,
            metadata=dict(cand.metadata or {}),
        ))
    return out



def _random_candidate(problem, rng: random.Random) -> CandidateRecord:
    pos = [rng.uniform(lo, hi) for lo, hi in zip(problem.min_values, problem.max_values)]
    fit = float(problem.evaluate(pos))
    return CandidateRecord(position=pos, fitness=fit, role="restart")



def execute_decision_plan(
    plan: DecisionPlan,
    engines: dict[str, Any],
    states: dict[str, Any],
    objective: str,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[ActionOutcome]]:
    outcomes: list[ActionOutcome] = []
    rng = random.Random(seed)

    for action in plan.actions:
        try:
            if action.type == "wait":
                outcomes.append(ActionOutcome(action=action, executed=True, status="applied", message="No action taken."))
                continue

            if action.type == "set_checkpoint":
                outcomes.append(ActionOutcome(action=action, executed=True, status="applied", message="Checkpoint interval updated by runner."))
                continue

            target_engine = engines.get(action.target_label) if action.target_label else None
            target_state = states.get(action.target_label) if action.target_label else None
            source_engine = engines.get(action.source_label) if action.source_label else None
            source_state = states.get(action.source_label) if action.source_label else None

            if action.type in {"inject", "inject_perturbed", "restart_agent"} and (source_engine is None and not action.params.get("random_seed_receiver")):
                outcomes.append(ActionOutcome(action=action, executed=False, status="skipped", message="Missing source engine."))
                continue
            if action.type in {"inject", "inject_perturbed", "rebalance", "reconfigure", "restart_agent"} and target_engine is None:
                outcomes.append(ActionOutcome(action=action, executed=False, status="skipped", message="Missing target engine."))
                continue

            if action.type in {"inject", "inject_perturbed"}:
                mode = action.source_mode or "best"
                k = int(action.k or 1)
                migrants = select_candidates(source_engine, source_state, mode=mode, k=k)
                source_fit = migrants[0].fitness if migrants else None
                if action.type == "inject_perturbed":
                    sigma = float((action.perturbation or {}).get("sigma", 0.05))
                    migrants = perturb_candidates(migrants, sigma=sigma, min_values=target_engine.problem.min_values, max_values=target_engine.problem.max_values, seed=seed)
                before = target_state.best_fitness
                if not target_engine.capabilities.supports_candidate_injection:
                    outcomes.append(ActionOutcome(action=action, executed=False, status="skipped", message="Target engine does not support candidate injection.", source_fitness=source_fit, target_fitness_before=before, target_fitness_after=before))
                    continue
                states[action.target_label] = target_engine.inject_candidates(target_state, migrants, policy=action.replace_policy or "native")
                after = states[action.target_label].best_fitness
                outcomes.append(ActionOutcome(action=action, executed=True, status="applied", message="Candidates injected.", source_fitness=source_fit, target_fitness_before=before, target_fitness_after=after))
                continue

            if action.type == "rebalance":
                before = target_state.best_fitness
                if not target_engine.capabilities.supports_candidate_injection or not target_engine.capabilities.has_population:
                    outcomes.append(ActionOutcome(action=action, executed=False, status="skipped", message="Target engine cannot rebalance population.", target_fitness_before=before, target_fitness_after=before))
                    continue
                fraction = float(action.params.get("fraction", 0.2))
                try:
                    pop = target_engine.get_population(target_state)
                    n = max(1, int(math.ceil(len(pop) * fraction)))
                except Exception:
                    n = max(1, int(action.k or 1))
                migrants = [_random_candidate(target_engine.problem, rng) for _ in range(n)]
                states[action.target_label] = target_engine.inject_candidates(target_state, migrants, policy=action.replace_policy or "native")
                after = states[action.target_label].best_fitness
                outcomes.append(ActionOutcome(action=action, executed=True, status="applied", message=f"Population rebalanced with {n} random candidates.", target_fitness_before=before, target_fitness_after=after))
                continue

            if action.type == "reconfigure":
                before = target_state.best_fitness
                states[action.target_label] = target_engine.reconfigure(target_state, dict(action.params or {}))
                after = states[action.target_label].best_fitness
                outcomes.append(ActionOutcome(action=action, executed=True, status="applied", message="Engine reconfigured.", target_fitness_before=before, target_fitness_after=after))
                continue

            if action.type == "restart_agent":
                before = target_state.best_fitness
                seeds = []
                if action.params.get("random_seed_receiver"):
                    seeds = [_random_candidate(target_engine.problem, rng)]
                else:
                    seeds = select_candidates(source_engine, source_state, mode=action.source_mode or "best", k=int(action.k or 1))
                if not target_engine.capabilities.supports_restart:
                    msg = "Target engine does not support restart; downgraded to injection." if target_engine.capabilities.supports_candidate_injection else "Target engine does not support restart."
                    if target_engine.capabilities.supports_candidate_injection:
                        states[action.target_label] = target_engine.inject_candidates(target_state, seeds, policy=action.replace_policy or "native")
                        after = states[action.target_label].best_fitness
                        outcomes.append(ActionOutcome(action=action, executed=True, status="downgraded", message=msg, target_fitness_before=before, target_fitness_after=after))
                    else:
                        outcomes.append(ActionOutcome(action=action, executed=False, status="skipped", message=msg, target_fitness_before=before, target_fitness_after=before))
                    continue
                states[action.target_label] = target_engine.restart(target_state, seeds=seeds, preserve_best=bool(action.params.get("preserve_best", True)))
                after = states[action.target_label].best_fitness
                outcomes.append(ActionOutcome(action=action, executed=True, status="applied", message="Engine restarted.", target_fitness_before=before, target_fitness_after=after))
                continue

            outcomes.append(ActionOutcome(action=action, executed=False, status="skipped", message=f"Unsupported action: {action.type}"))
        except Exception as exc:
            outcomes.append(ActionOutcome(action=action, executed=False, status="failed", message=str(exc)))
    return states, outcomes



def outcome_to_dict(outcome: ActionOutcome) -> dict[str, Any]:
    data = asdict(outcome)
    data["action"] = asdict(outcome.action)
    return data


# ---------------------------------------------------------------------------
# Action-cost estimation
# ---------------------------------------------------------------------------
# Default evaluation-equivalent action cost table. The spec requires the
# ordering:
#     wait < inject ≤ inject_perturbed < rebalance ≤ reconfigure < restart_agent.
# These are dimensionless units that sum with the 1/H_t weighting used by the
# cost heuristics; they are intentionally small and monotonic, not tuned.
_ACTION_COST_BASE: dict[str, float] = {
    "wait": 0.0,
    "set_checkpoint": 0.0,
    "inject": 1.0,
    "inject_perturbed": 1.0,
    "rebalance": 5.0,
    "reconfigure": 5.0,
    "restart_agent": 20.0,
}


def estimate_action_cost(
    action: ActionSpec,
    snapshot: Any,
    states: dict[str, Any] | None,
    engines: dict[str, Any] | None,
) -> float:
    """Return an evaluation-equivalent cost estimate for ``action``.

    Deterministic and conservative: the returned value is a pure function of
    the ActionSpec's type and of the target engine's population size when
    relevant. It is used by orchestration diagnostics and for comparing intervention intensity.

    The ordering invariant required by the spec is preserved by the base
    table ``_ACTION_COST_BASE``.
    """
    if action is None or action.type is None:
        return 0.0
    base = float(_ACTION_COST_BASE.get(action.type, 1.0))

    # Scale rebalance / restart by the target population size when we have it
    # so larger populations cost more. We cap the multiplier so a 200-strong
    # population does not destroy the ordering.
    if action.type in {"rebalance", "restart_agent"}:
        mult = 1.0
        try:
            target_label = getattr(action, "target_label", None)
            if target_label and engines and states and target_label in engines:
                eng = engines[target_label]
                st = states[target_label]
                if eng.capabilities.has_population:
                    try:
                        pop = eng.get_population(st)
                        n = max(1, int(len(pop)))
                        # Cost grows with sqrt(n) and is capped at 4x for
                        # very large populations.
                        mult = min(4.0, max(1.0, math.sqrt(n / 20.0)))
                    except Exception:
                        pass
        except Exception:
            pass
        base *= float(mult)

    if action.type in {"inject", "inject_perturbed"}:
        k = int(action.k or 1)
        base *= max(1.0, float(k) / 2.0)

    return float(base)
