from __future__ import annotations

from ..schemas import ActionSpec, DecisionPlan


class RuleBasedController:
    name = "rule_based"

    def __init__(self, config, rules_config):
        self.config = config
        self.rules = rules_config

    def initialize(self, snapshot) -> None:
        return None

    def _best_agent(self, snapshot):
        if not snapshot.agents:
            return None
        if snapshot.objective == "max":
            return max(snapshot.agents, key=lambda a: float("-inf") if a.best_fitness is None else a.best_fitness)
        return min(snapshot.agents, key=lambda a: float("inf") if a.best_fitness is None else a.best_fitness)

    def _best_diverse_donor(self, snapshot, exclude_label=None):
        pops = [a for a in snapshot.agents if a.has_population and a.supports_injection and a.label != exclude_label]
        if not pops:
            return None
        return max(
            pops,
            key=lambda a: (
                -1.0 if a.diversity is None else a.diversity,
                0.0 if a.delta_best is None else abs(a.delta_best),
            ),
        )

    def _population_receivers(self, snapshot):
        out = []
        for a in snapshot.agents:
            if not a.has_population or not a.supports_injection:
                continue
            low_div = a.diversity is not None and a.diversity <= self.rules.low_diversity_threshold
            stagnant = a.stagnation_steps is not None and a.stagnation_steps >= self.rules.stagnation_threshold
            if low_div and stagnant:
                out.append(a)
        return out

    def _single_receivers(self, snapshot):
        out = []
        for a in snapshot.agents:
            if a.has_population or not a.supports_injection:
                continue
            stagnant = a.stagnation_steps is not None and a.stagnation_steps >= self.rules.stagnation_threshold
            unhealthy = a.health in {"poor", "frozen", "stagnating"}
            if stagnant or unhealthy:
                out.append(a)
        return out

    def _restart_candidates(self, snapshot):
        out = []
        severe = max(int(self.rules.stagnation_threshold) * 2, int(self.rules.stagnation_threshold) + 2)
        for a in snapshot.agents:
            if not getattr(a, "supports_restart", False):
                continue
            stagnant = a.stagnation_steps is not None and a.stagnation_steps >= severe
            collapsed = a.health in {"poor", "frozen"}
            if stagnant or collapsed:
                out.append(a)
        return out

    def _wait_candidate(self, snapshot, best=None):
        best = best or self._best_agent(snapshot)
        rationale = "All islands are either healthy or lack a clear intervention target."
        expected = "Preserve autonomy and reassess at the next checkpoint."
        if best is not None:
            rationale = f"No strong intervention signal. Global best currently held by {best.label}."
        return ActionSpec(
            type="wait",
            rationale=rationale,
            expected_effect=expected,
            confidence="medium",
        )

    def generate_candidate_actions(self, snapshot) -> list[ActionSpec]:
        best = self._best_agent(snapshot)
        candidates: list[ActionSpec] = [self._wait_candidate(snapshot, best=best)]
        seen: set[tuple] = set()

        if best is None:
            return candidates

        for receiver in self._single_receivers(snapshot):
            if receiver.label == best.label:
                continue
            action = ActionSpec(
                type="inject_perturbed",
                source_label=best.label,
                target_label=receiver.label,
                source_mode="best",
                k=1,
                replace_policy="native",
                perturbation={"sigma": self.rules.perturbation_sigma},
                rationale=f"Reseed trajectory-style receiver {receiver.label} around the basin found by {best.label}.",
                expected_effect="Improve local refinement while preserving some diversity.",
                confidence="medium",
            )
            key = (action.type, action.source_label, action.target_label)
            if key not in seen:
                candidates.append(action)
                seen.add(key)
            if self.rules.reheat_temperature is not None:
                reheat = ActionSpec(
                    type="reconfigure",
                    target_label=receiver.label,
                    params={"temperature": self.rules.reheat_temperature},
                    rationale=f"Reheat {receiver.label} after reseeding.",
                    expected_effect="Allow controlled exploration after restart near a strong basin.",
                    confidence="medium",
                )
                key = (reheat.type, reheat.source_label, reheat.target_label, tuple(sorted(reheat.params.items())))
                if key not in seen:
                    candidates.append(reheat)
                    seen.add(key)

        for receiver in self._population_receivers(snapshot):
            donor = self._best_diverse_donor(snapshot, exclude_label=receiver.label)
            if donor is None:
                continue
            source_mode = "diverse" if donor.diversity is not None and donor.diversity >= self.rules.high_diversity_threshold else "best"
            inject = ActionSpec(
                type="inject",
                source_label=donor.label,
                target_label=receiver.label,
                source_mode=source_mode,
                k=1,
                replace_policy="native",
                rationale=f"Inject a strong/diverse candidate from {donor.label} into collapsed population {receiver.label}.",
                expected_effect="Increase diversity and reduce stagnation.",
                confidence="medium",
            )
            key = (inject.type, inject.source_label, inject.target_label, inject.source_mode)
            if key not in seen:
                candidates.append(inject)
                seen.add(key)
            rebalance = ActionSpec(
                type="rebalance",
                target_label=receiver.label,
                replace_policy="native",
                params={"fraction": self.rules.rebalance_fraction},
                rationale=f"Rebalance {receiver.label} because diversity is low and progress stalled.",
                expected_effect="Refresh part of the population with exploratory material.",
                confidence="medium",
            )
            key = (rebalance.type, rebalance.source_label, rebalance.target_label, tuple(sorted(rebalance.params.items())))
            if key not in seen:
                candidates.append(rebalance)
                seen.add(key)

        for receiver in self._restart_candidates(snapshot):
            donor = best if best is not None and best.label != receiver.label else None
            restart = ActionSpec(
                type="restart_agent",
                source_label=donor.label if donor is not None else None,
                target_label=receiver.label,
                source_mode="best",
                k=1,
                replace_policy="native",
                params={} if donor is not None else {"random_seed_receiver": True},
                rationale=f"Restart {receiver.label} because stagnation or collapse appears severe.",
                expected_effect="Escape an unproductive regime and resume search from a healthier state.",
                confidence="medium",
            )
            key = (restart.type, restart.source_label, restart.target_label, tuple(sorted(restart.params.items())))
            if key not in seen:
                candidates.append(restart)
                seen.add(key)

        return candidates

    def candidate_records(self, snapshot) -> list[dict]:
        records = []
        for idx, action in enumerate(self.generate_candidate_actions(snapshot), start=1):
            tags = []
            if action.type == "wait":
                tags.append("safe")
            if action.type in {"inject", "inject_perturbed"}:
                tags.extend(["donor_receiver", "transfer"])
            if action.type == "rebalance":
                tags.extend(["diversity", "population"])
            if action.type == "restart_agent":
                tags.extend(["severe", "reset"])
            if action.type == "reconfigure":
                tags.extend(["parameter_change"])
            if action.type != "wait":
                tags.append("intervention")
            records.append({
                "candidate_id": f"cand_{idx}",
                "action": {
                    "type": action.type,
                    "source_label": action.source_label,
                    "target_label": action.target_label,
                    "source_mode": action.source_mode,
                    "k": action.k,
                    "replace_policy": action.replace_policy,
                    "perturbation": dict(action.perturbation or {}) or None,
                    "params": dict(action.params or {}),
                    "rationale": action.rationale,
                    "expected_effect": action.expected_effect,
                    "confidence": action.confidence,
                },
                "tags": tags,
                "choose_when": action.expected_effect or action.rationale,
            })
        return records

    def decide(self, snapshot) -> DecisionPlan:
        best = self._best_agent(snapshot)
        if best is None:
            return DecisionPlan(
                controller_mode="rules",
                controller_name=self.name,
                reasoning="No agents available.",
                confidence="high",
                actions=[ActionSpec(type="wait", rationale="No agents available.")],
            )

        if snapshot.checkpoint_id < self.config.warmup_checkpoints:
            return DecisionPlan(
                controller_mode="rules",
                controller_name=self.name,
                reasoning="Warm-up phase: allowing islands to develop their native search dynamics before adaptive cooperation.",
                confidence="high",
                actions=[ActionSpec(type="wait", rationale="Warm-up checkpoint.")],
                diagnostics={"global_best_label": best.label, "global_best_fitness": best.best_fitness},
            )

        candidates = self.generate_candidate_actions(snapshot)
        non_wait = [a for a in candidates if a.type != "wait"]
        if not non_wait:
            reasoning = f"No strong intervention signal. Global best currently held by {best.label}. Preserve autonomy and reassess later."
            return DecisionPlan(
                controller_mode="rules",
                controller_name=self.name,
                reasoning=reasoning,
                confidence="high",
                actions=[self._wait_candidate(snapshot, best=best)],
                diagnostics={
                    "global_best_label": best.label,
                    "global_best_fitness": best.best_fitness,
                    "candidate_count": len(candidates),
                },
            )

        chosen = non_wait[: self.config.max_actions_per_checkpoint]
        next_interval = self.config.checkpoint_interval
        reasoning = " | ".join(a.rationale for a in chosen)
        return DecisionPlan(
            controller_mode="rules",
            controller_name=self.name,
            reasoning=reasoning,
            confidence="medium",
            actions=chosen,
            next_checkpoint_interval=next_interval,
            diagnostics={
                "global_best_label": best.label,
                "global_best_fitness": best.best_fitness,
                "candidate_count": len(candidates),
                "candidate_action_types": [a.type for a in candidates],
            },
        )
