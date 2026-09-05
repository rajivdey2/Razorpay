"""CQRS query side: read models rebuilt from the event stream, never written directly.

The rule is one line and everything else follows from it: **a projection is a pure
function of the event stream.** ``rebuild(store)`` starts from empty state and folds
every event forward. Nothing in this module writes to the event store, and nothing
outside it writes to a projection.

Why that constraint is worth the inconvenience
----------------------------------------------
It makes the audit trail *checkable*. ``ProjectionSet.fingerprint()`` hashes the
projected state, so:

    p1 = rebuild(store)
    p2 = rebuild(store)          # from scratch, again
    assert p1.fingerprint() == p2.fingerprint()

If those ever differ, a projection has hidden state -- a wall-clock read, a dict
iteration order, an accumulator that was not reset -- and any answer it gives about
history is unreliable. The test suite asserts it, and ``glctl audit`` prints it.

The practical version of the same property: a bug in a projection is fixed by
correcting the fold and replaying, with no migration and no data loss, because the
events were never the thing that was wrong. Compare with a mutable state table,
where a bug that mis-updated a balance three months ago leaves you with no way to
recover what the balance should have been.

What is projected
-----------------
``matches``     every hypothesis and its outcome, with features and evidence
``exceptions``  the open queue, ranked, with human resolutions applied
``ledger``      account balances from posted journal entries
``metrics``     counters for the dashboard
``txn_index``   transaction id -> the events that touched it, for the audit view
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.core.schema import canonical_json, sha256_hex

from app.events.store import Event, EventStore


@dataclass
class MatchView:
    match_key: str
    leg: str
    tier: int
    algorithm: str
    cardinality: str
    left_ids: list[str]
    right_ids: list[str]
    confidence: float
    features: dict[str, float]
    evidence: dict[str, Any]
    left_amount_paise: int
    right_amount_paise: int
    residual_paise: int
    runners_up: list[dict] = field(default_factory=list)
    status: str = "proposed"          # proposed | confirmed | rejected
    confirmed_by: str | None = None
    rationale: str | None = None
    gate: str | None = None
    #: Sequence numbers of every event about this match, oldest first.
    history: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "match_key": self.match_key, "leg": self.leg, "tier": self.tier,
            "algorithm": self.algorithm, "cardinality": self.cardinality,
            "left_ids": self.left_ids, "right_ids": self.right_ids,
            "confidence": self.confidence, "features": self.features,
            "evidence": self.evidence,
            "left_amount_paise": self.left_amount_paise,
            "right_amount_paise": self.right_amount_paise,
            "residual_paise": self.residual_paise,
            "runners_up": self.runners_up, "status": self.status,
            "confirmed_by": self.confirmed_by, "rationale": self.rationale,
            "gate": self.gate, "history": self.history,
        }


@dataclass
class ExceptionView:
    exception_id: str
    leg: str
    txn_ids: list[str]
    category: str
    max_confidence: float
    suggested_action: str
    amount_paise: int
    age_days: int
    priority: float
    evidence: dict[str, Any]
    status: str = "open"              # open | resolved
    resolution: str | None = None
    resolved_by: str | None = None
    resolution_note: str | None = None
    history: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "exception_id": self.exception_id, "leg": self.leg,
            "txn_ids": self.txn_ids, "category": self.category,
            "max_confidence": self.max_confidence,
            "suggested_action": self.suggested_action,
            "amount_paise": self.amount_paise, "age_days": self.age_days,
            "priority": self.priority, "evidence": self.evidence,
            "status": self.status, "resolution": self.resolution,
            "resolved_by": self.resolved_by, "resolution_note": self.resolution_note,
            "history": self.history,
        }


@dataclass
class TxnView:
    txn_id: str
    source: str
    external_id: str
    amount_paise: int
    currency: str
    value_date: str
    utr: str | None
    narration: str | None
    fees_paise: int = 0
    tax_paise: int = 0
    ref_candidates: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ProjectionSet:
    matches: dict[str, MatchView] = field(default_factory=dict)
    exceptions: dict[str, ExceptionView] = field(default_factory=dict)
    txns: dict[str, TxnView] = field(default_factory=dict)
    #: account -> net paise (debits positive)
    ledger: dict[str, int] = field(default_factory=dict)
    journal_entries: list[dict] = field(default_factory=list)
    snapshots: list[dict] = field(default_factory=list)
    arbitrations: list[dict] = field(default_factory=list)
    write_offs: list[dict] = field(default_factory=list)
    #: txn_id -> event sequence numbers, for the audit-trail view
    txn_index: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    events_applied: int = 0
    last_seq: int = 0

    # -- derived views ----------------------------------------------------

    def open_exceptions(self) -> list[ExceptionView]:
        return sorted(
            (e for e in self.exceptions.values() if e.status == "open"),
            key=lambda e: (-e.priority, e.exception_id),
        )

    def confirmed_matches(self) -> list[MatchView]:
        return [m for m in self.matches.values() if m.status == "confirmed"]

    def pending_matches(self) -> list[MatchView]:
        return sorted(
            (m for m in self.matches.values() if m.status == "proposed"),
            key=lambda m: (-max(abs(m.left_amount_paise), abs(m.right_amount_paise)),
                           m.match_key),
        )

    def ledger_balanced(self) -> tuple[bool, int]:
        """Debits must equal credits across every account, always.

        The ledger dict holds signed nets, so a balanced book sums to exactly zero.
        Any non-zero total means an unbalanced entry got through, which should be
        impossible -- ``recorder.post_journal_entries`` refuses to write one -- so a
        failure here points at the projection fold rather than at the data.
        """
        total = sum(self.ledger.values())
        return total == 0, total

    def metrics(self) -> dict:
        confirmed = self.confirmed_matches()
        pending = self.pending_matches()
        open_exc = self.open_exceptions()
        by_tier: dict[int, int] = {}
        by_algo: dict[str, int] = {}
        for m in confirmed:
            by_tier[m.tier] = by_tier.get(m.tier, 0) + 1
            by_algo[m.algorithm] = by_algo.get(m.algorithm, 0) + 1
        by_cat: dict[str, int] = {}
        for e in open_exc:
            by_cat[e.category] = by_cat.get(e.category, 0) + 1

        reconciled = sum(
            max(abs(m.left_amount_paise), abs(m.right_amount_paise))
            for m in confirmed if m.leg == "gateway_bank"
        )
        pending_paise = sum(
            max(abs(m.left_amount_paise), abs(m.right_amount_paise)) for m in pending
        )
        balanced, drift = self.ledger_balanced()
        return {
            "events_applied": self.events_applied,
            "transactions": len(self.txns),
            "matches_total": len(self.matches),
            "matches_confirmed": len(confirmed),
            "matches_pending_human": len(pending),
            "matches_rejected": sum(
                1 for m in self.matches.values() if m.status == "rejected"
            ),
            "exceptions_open": len(open_exc),
            "exceptions_resolved": sum(
                1 for e in self.exceptions.values() if e.status == "resolved"
            ),
            "confirmed_by_tier": {str(k): v for k, v in sorted(by_tier.items())},
            "confirmed_by_algorithm": dict(
                sorted(by_algo.items(), key=lambda kv: -kv[1])
            ),
            "open_exceptions_by_category": dict(
                sorted(by_cat.items(), key=lambda kv: -kv[1])
            ),
            "bank_leg_reconciled_paise": reconciled,
            "pending_human_paise": pending_paise,
            "exception_paise": sum(abs(e.amount_paise) for e in open_exc),
            "journal_entries": len(self.journal_entries),
            "ledger": dict(sorted(self.ledger.items())),
            "ledger_balanced": balanced,
            "ledger_drift_paise": drift,
            "arbitrations": len(self.arbitrations),
            "write_offs": len(self.write_offs),
        }

    def fingerprint(self) -> str:
        """Content hash of the projected state.

        Two rebuilds of the same stream must produce the same fingerprint. That is
        what makes the replay claim verifiable rather than aspirational -- and it is
        why nothing in the fold may read the clock or depend on iteration order.
        """
        payload = {
            "matches": {k: v.to_json() for k, v in sorted(self.matches.items())},
            "exceptions": {k: v.to_json() for k, v in sorted(self.exceptions.items())},
            "ledger": dict(sorted(self.ledger.items())),
            "txns": sorted(self.txns),
            "journal": sorted(
                (j["entry_id"], j["total_debit_paise"]) for j in self.journal_entries
            ),
        }
        return sha256_hex(canonical_json(payload))


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------

def apply_event(p: ProjectionSet, ev: Event) -> None:
    """Fold one event into the projection set. Pure with respect to the event.

    Deliberately a plain if-chain rather than a dispatch dict. Fifteen lines of
    ``elif`` is easier to read and to diff than a registry, and an unknown event type
    hitting the final ``else`` raises -- so adding an event type without teaching the
    projections about it is a loud failure rather than a projection that quietly
    lags reality.
    """
    t = ev.event_type
    pl = ev.payload

    if t == "TransactionIngested":
        p.txns[ev.aggregate_id] = TxnView(
            txn_id=ev.aggregate_id,
            source=pl["source"], external_id=pl["external_id"],
            amount_paise=pl["amount_paise"], currency=pl["currency"],
            value_date=pl["value_date"], utr=pl.get("utr"),
            narration=pl.get("narration"),
            fees_paise=pl.get("fees_paise", 0), tax_paise=pl.get("tax_paise", 0),
            ref_candidates=pl.get("ref_candidates", []),
            provenance=pl.get("provenance", {}),
        )
        p.txn_index[ev.aggregate_id].append(ev.seq)

    elif t == "MatchCandidateProposed":
        p.matches[ev.aggregate_id] = MatchView(
            match_key=ev.aggregate_id, leg=pl["leg"], tier=pl["tier"],
            algorithm=pl["algorithm"], cardinality=pl["cardinality"],
            left_ids=pl["left_ids"], right_ids=pl["right_ids"],
            confidence=pl["confidence"], features=pl.get("features", {}),
            evidence=pl.get("evidence", {}),
            left_amount_paise=pl["left_amount_paise"],
            right_amount_paise=pl["right_amount_paise"],
            residual_paise=pl["residual_paise"],
            runners_up=pl.get("runners_up", []),
            gate=pl.get("decision_gate"),
            rationale=pl.get("decision_reason"),
            history=[ev.seq],
        )
        for tid in (*pl["left_ids"], *pl["right_ids"]):
            p.txn_index[tid].append(ev.seq)

    elif t == "MatchConfirmed":
        m = p.matches.get(ev.aggregate_id)
        if m is None:
            # A confirmation with no proposal. Only the wash-pair rule does this,
            # and it is recorded rather than dropped so the replay stays complete.
            m = MatchView(
                match_key=ev.aggregate_id, leg=pl.get("leg", "unknown"),
                tier=pl.get("tier", 1), algorithm=pl.get("algorithm", "unknown"),
                cardinality="1:1", left_ids=[], right_ids=[],
                confidence=pl.get("confidence", 1.0), features={}, evidence={},
                left_amount_paise=0, right_amount_paise=0, residual_paise=0,
            )
            p.matches[ev.aggregate_id] = m
        m.status = "confirmed"
        m.confirmed_by = pl.get("confirmed_by")
        m.rationale = pl.get("rationale")
        m.gate = pl.get("gate")
        m.history.append(ev.seq)

    elif t == "MatchRejected":
        m = p.matches.get(ev.aggregate_id)
        if m is not None:
            m.status = "rejected"
            m.rationale = pl.get("reason")
            m.gate = pl.get("gate")
            m.history.append(ev.seq)

    elif t == "ExceptionRaised":
        p.exceptions[ev.aggregate_id] = ExceptionView(
            exception_id=ev.aggregate_id, leg=pl["leg"], txn_ids=pl["txn_ids"],
            category=pl["category"], max_confidence=pl["max_confidence"],
            suggested_action=pl["suggested_action"], amount_paise=pl["amount_paise"],
            age_days=pl["age_days"], priority=pl.get("priority", 0.0),
            evidence=pl.get("evidence", {}), history=[ev.seq],
        )
        for tid in pl["txn_ids"]:
            p.txn_index[tid].append(ev.seq)

    elif t == "ExceptionResolved":
        e = p.exceptions.get(ev.aggregate_id)
        if e is not None:
            e.status = "resolved"
            e.resolution = pl.get("resolution")
            e.resolved_by = pl.get("resolved_by")
            e.resolution_note = pl.get("note")
            e.history.append(ev.seq)

    elif t == "JournalEntryPosted":
        for leg in pl["legs"]:
            delta = leg["debit_paise"] - leg["credit_paise"]
            p.ledger[leg["account"]] = p.ledger.get(leg["account"], 0) + delta
        p.journal_entries.append(
            {
                "entry_id": ev.aggregate_id,
                "source_match_key": pl["source_match_key"],
                "leg": pl["leg"], "legs": pl["legs"],
                "total_debit_paise": pl["total_debit_paise"],
                "total_credit_paise": pl["total_credit_paise"],
                "seq": ev.seq,
            }
        )

    elif t == "SuspenseWriteOff":
        p.ledger[pl["account"]] = p.ledger.get(pl["account"], 0)
        p.write_offs.append({**pl, "match_key": ev.aggregate_id, "seq": ev.seq})

    elif t == "ReconciliationSnapshot":
        p.snapshots.append({**pl, "batch_id": ev.aggregate_id, "seq": ev.seq})

    elif t in ("Tier3ArbitrationRequested", "Tier3ArbitrationReturned"):
        p.arbitrations.append(
            {"kind": t, "match_key": ev.aggregate_id, "seq": ev.seq, **pl}
        )

    else:
        raise ValueError(
            f"projection has no rule for event type {ev.event_type!r} at seq {ev.seq}. "
            "Adding an event type requires teaching the fold about it, or every "
            "read model silently lags reality."
        )

    p.events_applied += 1
    p.last_seq = ev.seq


def rebuild(store: EventStore, *, after_seq: int = 0) -> ProjectionSet:
    """Fold the whole stream from scratch. The only way a projection is built."""
    p = ProjectionSet()
    for ev in store.stream(after_seq=after_seq):
        apply_event(p, ev)
    return p


def catch_up(p: ProjectionSet, store: EventStore) -> ProjectionSet:
    """Apply only events newer than the projection's high-water mark.

    An optimisation over ``rebuild``, and provably equivalent because the fold is
    associative in sequence order. The test suite asserts the equivalence rather
    than assuming it, since an incremental path that diverges from the full rebuild
    is the classic CQRS bug and it is invisible until someone compares.
    """
    for ev in store.stream(after_seq=p.last_seq):
        apply_event(p, ev)
    return p


def audit_trail(store: EventStore, txn_id: str) -> list[dict]:
    """Everything that ever happened to one transaction, in order.

    This is the answer to "why did the agent match these two on March 5th": the
    proposal with its features and its rejected alternatives, the confirmation with
    its rationale, the journal entry, and any human override -- reconstructed from
    the log rather than from a summary somebody wrote at the time.
    """
    return [ev.to_json() for ev in store.touching(txn_id)]
