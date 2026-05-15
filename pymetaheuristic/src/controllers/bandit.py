from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Any

from ..schemas import ActionSpec, BanditConfig, DecisionPlan
from .rules import RuleBasedController


_ACTION_COST_BASE: dict[str, float] = {
    "wait": 0.0,
    "set_checkpoint": 0.0,
    "inject": 1.0,
    "inject_perturbed": 1.25,
    "rebalance": 4.0,
    "reconfigure": 4.0,
    "restart_agent": 8.0,
}


class BanditController:
    """Adaptive action selector based on a lightweight multi-armed bandit.

    The controller deliberately does *not* invent unsafe actions.  It first asks
    the rule-based controller for feasible candidate interventions, then chooses
    among those candidates using rewards observed at previous checkpoints.
    """

    name = "bandit_ucb"

    def __init__(self, config, rules_config, bandit_config: BanditConfig | None = None):
        self.config = config
        self.rules = rules_config
        self.bandit = bandit_config or BanditConfig()
        self.rule_controller = RuleBasedController(config, rules_config)
        self._rewards: dict[tuple[Any, ...], deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, int(self.bandit.reward_window)))
        )
        self._counts: dict[tuple[Any, ...], int] = defaultdict(int)
        self._total_updates = 0
        self._processed_action_signatures: set[tuple[Any, ...]] = set()
        self._rng = random.Random(0)
        self.name = f"bandit_{self.bandit.policy}"

    def initialize(self, snapshot) -> None:
        self.rule_controller.initialize(snapshot)
        return None

    @staticmethod
    def _action_key(action: ActionSpec | dict[str, Any]) -> tuple[Any, ...]:
        if isinstance(action, dict):
            return (
                action.get("type"),
                action.get("source_label"),
                action.get("target_label"),
                action.get("source_mode"),
                action.get("replace_policy"),
                action.get("k"),
            )
        return (
            action.type,
            action.source_label,
            action.target_label,
            action.source_mode,
            action.replace_policy,
            action.k,
        )

    @staticmethod
    def _status_penalty(status: str | None) -> float:
        if status == "failed":
            return -1.0
        if status == "skipped":
            return -0.5
        if status == "downgraded":
            return -0.1
        return 0.0

    def _outcome_reward(self, outcome: dict[str, Any], objective: str) -> float:
        action = outcome.get("action") or {}
        before = outcome.get("target_fitness_before")
        after = outcome.get("target_fitness_after")
        reward = self._status_penalty(outcome.get("status"))
        if before is not None and after is not None:
            try:
                before_f = float(before)
                after_f = float(after)
                improvement = (before_f - after_f) if objective == "min" else (after_f - before_f)
                scale = 1.0 + abs(before_f)
                reward += improvement / scale
            except Exception:
                pass
        reward -= float(self.bandit.action_cost_penalty) * _ACTION_COST_BASE.get(str(action.get("type")), 1.0)
        return float(reward)

    def _update_from_recent_actions(self, snapshot) -> None:
        for idx, outcome in enumerate(getattr(snapshot, "recent_actions", []) or []):
            if not isinstance(outcome, dict):
                continue
            action = outcome.get("action") or {}
            signature = (
                getattr(snapshot, "checkpoint_id", None),
                idx,
                tuple(sorted((k, repr(v)) for k, v in action.items())),
                outcome.get("status"),
                repr(outcome.get("target_fitness_before")),
                repr(outcome.get("target_fitness_after")),
            )
            if signature in self._processed_action_signatures:
                continue
            self._processed_action_signatures.add(signature)
            key = self._action_key(action)
            reward = self._outcome_reward(outcome, getattr(snapshot, "objective", "min"))
            self._rewards[key].append(reward)
            self._counts[key] += 1
            self._total_updates += 1

    def _mean_reward(self, key: tuple[Any, ...]) -> float:
        values = self._rewards.get(key)
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _score(self, action: ActionSpec) -> float:
        key = self._action_key(action)
        n = self._counts.get(key, 0)
        if n <= 0 and action.type != "wait":
            return float("inf")
        mean = self._mean_reward(key)
        policy = (self.bandit.policy or "ucb").lower()
        if policy == "ucb":
            total = max(1, self._total_updates)
            bonus = float(self.bandit.exploration) * math.sqrt(math.log(total + 1.0) / max(1, n))
            return mean + bonus
        return mean

    def _choose_actions(self, candidates: list[ActionSpec]) -> list[ActionSpec]:
        if not candidates:
            return [ActionSpec(type="wait", rationale="No feasible bandit candidate actions.", confidence="high")]
        wait_actions = [a for a in candidates if a.type == "wait"]
        non_wait = [a for a in candidates if a.type != "wait"]
        if not non_wait:
            return wait_actions[:1] or [ActionSpec(type="wait", rationale="Only wait arm is feasible.", confidence="high")]

        policy = (self.bandit.policy or "ucb").lower()
        if policy == "epsilon_greedy" and self._rng.random() < float(self.bandit.epsilon):
            self._rng.shuffle(non_wait)
            ordered = non_wait
        elif policy == "greedy":
            ordered = sorted(non_wait, key=lambda a: self._mean_reward(self._action_key(a)), reverse=True)
        else:
            ordered = sorted(non_wait, key=self._score, reverse=True)

        chosen = ordered[: max(1, int(self.config.max_actions_per_checkpoint))]
        # If every known intervention looks worse than doing nothing, wait.
        known_scores = [self._score(a) for a in chosen if self._counts.get(self._action_key(a), 0) > 0]
        if known_scores and max(known_scores) < float(self.bandit.min_reward_to_act):
            return wait_actions[:1] or [ActionSpec(type="wait", rationale="Bandit expected reward is below the action threshold.", confidence="medium")]
        return chosen

    def decide(self, snapshot) -> DecisionPlan:
        self._update_from_recent_actions(snapshot)
        if getattr(snapshot, "checkpoint_id", 0) < self.config.warmup_checkpoints:
            return DecisionPlan(
                controller_mode="bandit",
                controller_name=self.name,
                reasoning="Warm-up phase: collecting initial island behavior before adaptive bandit decisions.",
                confidence="high",
                actions=[ActionSpec(type="wait", rationale="Warm-up checkpoint.")],
                diagnostics={"bandit_updates": self._total_updates},
            )

        candidates = self.rule_controller.generate_candidate_actions(snapshot)
        if not self.bandit.include_wait_arm:
            candidates = [a for a in candidates if a.type != "wait"]
        chosen = self._choose_actions(candidates)
        diagnostics = {
            "policy": self.bandit.policy,
            "candidate_count": len(candidates),
            "bandit_updates": self._total_updates,
            "chosen_scores": [self._score(a) for a in chosen],
            "chosen_keys": [self._action_key(a) for a in chosen],
            "arm_counts": {str(k): int(v) for k, v in self._counts.items()},
            "arm_mean_rewards": {str(k): self._mean_reward(k) for k in self._counts},
        }
        reasoning = " | ".join(a.rationale or f"Bandit selected {a.type}." for a in chosen)
        return DecisionPlan(
            controller_mode="bandit",
            controller_name=self.name,
            reasoning=reasoning or "Bandit selected the highest-scoring feasible orchestration actions.",
            confidence="medium",
            actions=chosen,
            next_checkpoint_interval=self.config.checkpoint_interval,
            diagnostics=diagnostics,
        )
