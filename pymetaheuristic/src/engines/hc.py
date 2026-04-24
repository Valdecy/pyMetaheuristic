"""pyMetaheuristic src — Hill Climb Engine"""
from __future__ import annotations

from .protocol import CapabilityProfile
from ._ported_common import PortedTrajectoryEngine


class HCEngine(PortedTrajectoryEngine):
    """Hill Climb Algorithm — best-improving random neighbourhood search."""
    algorithm_id = "hc"
    algorithm_name = "Hill Climb Algorithm"
    family = "trajectory"
    capabilities = CapabilityProfile(has_population=False, supports_candidate_injection=False, supports_checkpoint=True, supports_framework_constraints=True)
    _DEFAULTS = dict(delta=0.5, neighborhood_size=20, expand=1.0, contract=0.95)
