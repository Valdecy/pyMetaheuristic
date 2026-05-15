from __future__ import annotations

from ..schemas import ActionSpec, DecisionPlan, PortfolioConfig
from .rules import RuleBasedController


class PortfolioAdaptiveController:
    """Budget-aware portfolio controller for heterogeneous islands.

    The controller uses the island portfolio as an ecosystem.  Early in the run
    it favors diversity and exploration; in the middle it reuses the safe
    rule-based candidate generator; late in the run it pushes the global-best
    basin toward receivers that can exploit injected candidates.
    """

    name = "portfolio_adaptive"

    def __init__(self, config, rules_config, portfolio_config: PortfolioConfig | None = None):
        self.config = config
        self.rules = rules_config
        self.portfolio = portfolio_config or PortfolioConfig()
        self.rule_controller = RuleBasedController(config, rules_config)

    def initialize(self, snapshot) -> None:
        self.rule_controller.initialize(snapshot)
        return None

    def _best_agent(self, snapshot):
        return self.rule_controller._best_agent(snapshot)

    def _phase(self, snapshot) -> str:
        ratio = getattr(snapshot, "budget_used_ratio", None)
        if ratio is None:
            # Fall back to checkpoint progress when no evaluation budget exists.
            warm = max(1, int(self.config.warmup_checkpoints))
            if snapshot.checkpoint_id <= warm:
                return "early"
            return "middle"
        if ratio < float(self.portfolio.early_budget_ratio):
            return "early"
        if ratio >= float(self.portfolio.late_budget_ratio):
            return "late"
        return "middle"

    @staticmethod
    def _family(agent) -> str:
        return str(getattr(agent, "family", "") or getattr(agent, "algorithm", ""))

    def _diverse_donors(self, snapshot, exclude_label: str | None = None):
        donors = [
            a for a in snapshot.agents
            if a.label != exclude_label and a.best_fitness is not None and a.supports_injection
        ]
        if not donors:
            donors = [a for a in snapshot.agents if a.label != exclude_label and a.best_fitness is not None]
        return sorted(
            donors,
            key=lambda a: (
                -1.0 if a.diversity is None else float(a.diversity),
                0.0 if a.delta_best is None else abs(float(a.delta_best)),
            ),
            reverse=True,
        )

    def _needs_help(self, agent) -> bool:
        stagnant = (agent.stagnation_steps is not None and int(agent.stagnation_steps) >= int(self.portfolio.stagnation_threshold))
        low_div = (agent.diversity is not None and float(agent.diversity) <= float(self.portfolio.diversity_threshold))
        unhealthy = agent.health in {"poor", "frozen", "stagnating"}
        return bool(stagnant or low_div or unhealthy)

    def _wait(self, message: str) -> ActionSpec:
        return ActionSpec(type="wait", rationale=message, expected_effect="Preserve island autonomy until a clearer signal appears.", confidence="medium")

    def _early_actions(self, snapshot) -> list[ActionSpec]:
        actions: list[ActionSpec] = []
        receivers = [a for a in snapshot.agents if a.supports_injection and self._needs_help(a)]
        for receiver in receivers:
            donors = self._diverse_donors(snapshot, exclude_label=receiver.label)
            if self.portfolio.prefer_heterogeneous_donors:
                hetero = [d for d in donors if self._family(d) != self._family(receiver)]
                donors = hetero or donors
            if donors:
                donor = donors[0]
                actions.append(ActionSpec(
                    type="inject",
                    source_label=donor.label,
                    target_label=receiver.label,
                    source_mode="diverse" if donor.has_population else "best",
                    k=1,
                    replace_policy="native",
                    rationale=f"Early portfolio phase: transfer diverse material from {donor.label} to {receiver.label}.",
                    expected_effect="Increase exploration and prevent premature island collapse.",
                    confidence="medium",
                ))
            if receiver.has_population:
                actions.append(ActionSpec(
                    type="rebalance",
                    target_label=receiver.label,
                    replace_policy="native",
                    params={"fraction": self.portfolio.rebalance_fraction},
                    rationale=f"Early portfolio phase: rebalance low-diversity population {receiver.label}.",
                    expected_effect="Restore population diversity.",
                    confidence="medium",
                ))
        return actions

    def _late_actions(self, snapshot) -> list[ActionSpec]:
        best = self._best_agent(snapshot)
        if best is None:
            return []
        actions: list[ActionSpec] = []
        receivers = [a for a in snapshot.agents if a.label != best.label and a.supports_injection]
        receivers = [a for a in receivers if self._needs_help(a)] or receivers
        for receiver in receivers:
            action_type = "inject_perturbed" if not receiver.has_population else "inject"
            actions.append(ActionSpec(
                type=action_type,
                source_label=best.label,
                target_label=receiver.label,
                source_mode="best",
                k=1,
                replace_policy="native",
                perturbation={"sigma": self.portfolio.perturbation_sigma} if action_type == "inject_perturbed" else None,
                rationale=f"Late portfolio phase: exploit the best basin from {best.label} through {receiver.label}.",
                expected_effect="Intensify search around the current global-best basin.",
                confidence="medium",
            ))
        if self.portfolio.allow_restarts:
            for receiver in receivers:
                severe = receiver.stagnation_steps is not None and int(receiver.stagnation_steps) >= 2 * int(self.portfolio.stagnation_threshold)
                if severe and receiver.supports_restart:
                    actions.append(ActionSpec(
                        type="restart_agent",
                        source_label=best.label,
                        target_label=receiver.label,
                        source_mode="best",
                        k=1,
                        replace_policy="native",
                        params={"preserve_best": True},
                        rationale=f"Late portfolio phase: restart severely stagnant island {receiver.label} around the best known basin.",
                        expected_effect="Recover useful exploitation capacity without discarding the global best.",
                        confidence="medium",
                    ))
        return actions

    def decide(self, snapshot) -> DecisionPlan:
        if getattr(snapshot, "checkpoint_id", 0) < self.config.warmup_checkpoints:
            return DecisionPlan(
                controller_mode="portfolio_adaptive",
                controller_name=self.name,
                reasoning="Warm-up phase: portfolio roles are not yet reliable.",
                confidence="high",
                actions=[self._wait("Warm-up checkpoint.")],
                diagnostics={"phase": "warmup"},
            )
        phase = self._phase(snapshot)
        if phase == "early":
            candidates = self._early_actions(snapshot)
        elif phase == "late":
            candidates = self._late_actions(snapshot)
        else:
            candidates = [a for a in self.rule_controller.generate_candidate_actions(snapshot) if a.type != "wait"]

        if not candidates:
            candidates = [self._wait(f"Portfolio-adaptive {phase} phase found no feasible intervention.")]
        chosen = candidates[: max(1, int(self.config.max_actions_per_checkpoint))]
        reasoning = " | ".join(a.rationale for a in chosen)
        return DecisionPlan(
            controller_mode="portfolio_adaptive",
            controller_name=self.name,
            reasoning=reasoning,
            confidence="medium",
            actions=chosen,
            next_checkpoint_interval=self.config.checkpoint_interval,
            diagnostics={
                "phase": phase,
                "budget_used_ratio": snapshot.budget_used_ratio,
                "candidate_count": len(candidates),
                "global_best_label": snapshot.global_best_label,
                "global_best_fitness": snapshot.global_best_fitness,
            },
        )
