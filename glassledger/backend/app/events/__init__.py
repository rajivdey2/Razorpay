"""Event-sourced audit core: append-only, hash-chained, idempotent."""

from __future__ import annotations

from .recorder import (
    ACC_BANK,
    ACC_FEES,
    ACC_GATEWAY_RECEIVABLE,
    ACC_GST_INPUT,
    ACC_SUSPENSE,
    RecordingSummary,
    journal_legs_for,
    post_journal_entries,
    record_decision,
    record_exception,
    record_ingestion,
    record_run,
    record_snapshot,
    record_tier3,
)
from .store import (
    EVENT_TYPES,
    GENESIS_HASH,
    ChainVerification,
    Event,
    EventStore,
)

__all__ = [
    "EventStore", "Event", "ChainVerification", "EVENT_TYPES", "GENESIS_HASH",
    "record_run", "record_ingestion", "record_decision", "record_exception",
    "record_snapshot", "record_tier3", "post_journal_entries", "journal_legs_for",
    "RecordingSummary",
    "ACC_BANK", "ACC_GATEWAY_RECEIVABLE", "ACC_FEES", "ACC_GST_INPUT", "ACC_SUSPENSE",
]
