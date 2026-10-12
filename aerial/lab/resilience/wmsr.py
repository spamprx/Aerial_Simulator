"""W-MSR (weighted mean-subsequence-reduced) resilient consensus."""
from __future__ import annotations

from typing import Optional

from ..coordination.consensus import SecondOrderConsensus
from ..registry import controller_registry


@controller_registry.register("wmsr_consensus")
class WMSRConsensus(SecondOrderConsensus):
    """``consensus`` with ``filter="wmsr"``, kept under this name so that old configs still work."""

    def __init__(self, k_p: float = 1.0, k_v: float = 1.5, f: Optional[int] = None, damping: Optional[str] = None):
        super().__init__(k_p, k_v, filter="wmsr", f=f, damping=damping)

    @property
    def accepted(self):
        return self.log.accepted

    @property
    def rejected(self):
        return self.log.rejected
