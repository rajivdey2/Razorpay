"""Bounded LLM arbitration: proposer only, never approver, never above the gate."""

from __future__ import annotations

from .arbiter import (
    ARBITRATION_SCHEMA,
    MAX_CONFIDENCE_DELTA,
    TIER3_FLOOR,
    Arbitration,
    ClaudeArbiter,
    NullArbiter,
    OfflineArbiter,
    build_payload,
    default_arbiter,
)
from .tier3 import (
    Tier3Report,
    apply_arbitration,
    repolicy,
    run_tier3,
    select_for_arbitration,
    substitute,
)

__all__ = [
    "Arbitration", "ClaudeArbiter", "NullArbiter", "OfflineArbiter",
    "default_arbiter", "build_payload", "ARBITRATION_SCHEMA",
    "MAX_CONFIDENCE_DELTA", "TIER3_FLOOR",
    "Tier3Report", "run_tier3", "select_for_arbitration", "apply_arbitration",
    "repolicy", "substitute",
]
