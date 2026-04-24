from __future__ import annotations

from ..schemas import ActionSpec, DecisionPlan


class FixedMigrationController:
    """Placeholder fixed controller.

    The legacy cooperative path remains available through cooperative_optimize().
    This controller is primarily used as a schema-compatible no-op fallback inside
    the orchestration stack when the caller wants explicit checkpoint objects but
    no adaptive behaviour.
    """

    name = "fixed_migration"

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, snapshot) -> None:
        return None

    def decide(self, snapshot) -> DecisionPlan:
        return DecisionPlan(
            controller_mode="fixed",
            controller_name=self.name,
            reasoning="Fixed mode delegates cooperation to the legacy runner or preserves current behaviour without adaptive actions.",
            confidence="high",
            actions=[ActionSpec(type="wait", rationale="No adaptive action in fixed mode.")],
            next_checkpoint_interval=None,
            diagnostics={"mode": "fixed"},
        )
