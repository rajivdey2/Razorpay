"""Recording a reconciliation run as events, and the double-entry postings it implies.

The engine produces decisions. This module turns them into the append-only history
that makes the decisions auditable: one event per fact, in causal order, with the
evidence attached.

Journal entries
---------------
A confirmed gateway->bank match posts a real double-entry pair:

    Dr  Bank                       (money arrived)
      Cr  Gateway receivable       (the claim is settled)
    Dr  Payment processing fees    (what the gateway kept)
    Dr  GST input credit           (recoverable tax on that fee)

The fee and tax legs come from the settlement entity's own ``fees`` and ``tax``, so
the entry balances to the paise without a plug. That matters: a reconciliation
system that posts a balancing "difference" account has not reconciled anything, it
has relabelled the problem. If an entry does not balance, ``post_journal_entries``
raises rather than posting -- with the residual in the message.

Idempotency keys
----------------
Every event carries one, and every key is derived from content rather than from a
counter or a timestamp:

    ingest    sha256(source, external_id, payload)   -- the payload hash
    match     match_key + decision action
    exception exception_id
    journal   match_key + entry role

So re-running the entire pipeline over the same inputs appends nothing the second
time. That is the property that makes "just run it again" a safe response to a
partial failure, which in turn is the difference between an operable system and one
that needs a human to reason about what got half-written.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.core.config import MATERIALITY_PAISE
from app.core.schema import canonical_json, sha256_hex
from app.ingestion import IngestedBatch
from app.matching.types import Candidate, Decision, Exception_

from .store import Event, EventStore

# Chart of accounts. Deliberately tiny and explicit -- a reconciliation engine that
# invents account codes is worse than one that posts to five well-named accounts.
ACC_BANK = "1010:Bank"
ACC_GATEWAY_RECEIVABLE = "1210:Gateway receivable"
ACC_TRADE_RECEIVABLE = "1200:Trade receivable"
ACC_FEES = "6100:Payment processing fees"
ACC_GST_INPUT = "1360:GST input credit"
ACC_SUSPENSE = "2990:Reconciliation suspense"


@dataclass
class RecordingSummary:
    events_written: int = 0
    journal_entries: int = 0
    journal_paise: int = 0
    unbalanced_rejected: int = 0
    by_type: dict[str, int] = None

    def to_json(self) -> dict:
        return {
            "events_written": self.events_written,
            "journal_entries": self.journal_entries,
            "journal_paise": self.journal_paise,
            "unbalanced_rejected": self.unbalanced_rejected,
            "by_type": self.by_type or {},
        }


def record_ingestion(store: EventStore, batch: IngestedBatch) -> int:
    """One ``TransactionIngested`` per line, keyed by payload hash.

    The key is the hash of the *source bytes*, so a byte-identical re-import is a
    no-op while a genuinely changed line (a bank re-exporting with a corrected
    narration) is a new fact with its own event. Both behaviours are correct and
    neither is achievable with a key based on the external id alone.
    """
    n = 0
    for t in batch.all_txns:
        ev = store.append(
            "TransactionIngested",
            t.txn_id,
            {
                "source": t.source,
                "external_id": t.external_id,
                "amount_paise": t.amount_paise,
                "currency": t.currency,
                "value_date": t.value_date.isoformat(),
                "utr": t.utr,
                "narration": t.narration,
                "fees_paise": t.fees_paise,
                "tax_paise": t.tax_paise,
                "ref_candidates": list(t.ref_candidates or ()),
                "payload_hash": t.raw_payload_hash,
                "provenance": t.provenance,
            },
            idempotency_key=f"ingest:{t.raw_payload_hash}",
        )
        if ev.seq:
            n += 1
    return n


def record_decision(store: EventStore, d: Decision, *, leg: str) -> list[Event]:
    """Proposal then outcome, always both.

    The proposal event carries the full feature vector and the runners-up; the
    outcome carries the action and the gate that produced it. Writing only the
    outcome would make the history smaller and useless -- "why did it pick this
    one" is answered by the alternatives it rejected, not by the winner.
    """
    c = d.candidate
    out: list[Event] = []
    out.append(
        store.append(
            "MatchCandidateProposed",
            c.match_key,
            {
                "leg": leg,
                "tier": c.tier,
                "algorithm": c.algorithm,
                "cardinality": c.cardinality,
                "left_ids": list(c.left_ids),
                "right_ids": list(c.right_ids),
                "confidence": round(c.score, 6),
                "features": {k: round(v, 6) for k, v in sorted(c.features.items())},
                "evidence": c.evidence,
                "left_amount_paise": c.left_amount_paise,
                "right_amount_paise": c.right_amount_paise,
                "residual_paise": c.residual_paise,
                # The policy outcome is recorded on the proposal itself, not only
                # on the follow-up event. A match that is *proposed* gets no
                # second event -- there is nothing to confirm or reject yet -- so
                # without this the reviewer sees "awaiting approval" with no
                # record of which rule put it there.
                "decision_action": d.action,
                "decision_gate": d.gate,
                "decision_reason": d.reason,
                "runners_up": [r.to_json() for r in d.runners_up[:3]],
            },
            idempotency_key=f"propose:{c.match_key}",
        )
    )
    if d.action == "auto_confirm":
        out.append(
            store.append(
                "MatchConfirmed",
                c.match_key,
                {
                    "leg": leg,
                    "confirmed_by": "agent",
                    "confidence": round(c.score, 6),
                    "tier": c.tier,
                    "algorithm": c.algorithm,
                    "rationale": d.reason,
                    "gate": d.gate,
                    # The ids are repeated here even though the proposal already
                    # carries them. ``EventStore.touching`` finds events by
                    # transaction id, and without this the confirmation -- the
                    # single most important event about a match -- is absent from
                    # that transaction's audit trail. An event that cannot be
                    # found from the thing it decided is not an audit record.
                    "left_ids": list(c.left_ids),
                    "right_ids": list(c.right_ids),
                    "left_amount_paise": c.left_amount_paise,
                    "right_amount_paise": c.right_amount_paise,
                },
                idempotency_key=f"confirm:{c.match_key}",
            )
        )
    elif d.action == "reject":
        out.append(
            store.append(
                "MatchRejected",
                c.match_key,
                {
                    "leg": leg, "rejected_by": "agent", "reason": d.reason,
                    "gate": d.gate,
                    "left_ids": list(c.left_ids), "right_ids": list(c.right_ids),
                },
                idempotency_key=f"reject:{c.match_key}",
            )
        )
    return out


def record_exception(store: EventStore, e: Exception_) -> Event:
    return store.append(
        "ExceptionRaised",
        e.exception_id,
        {
            "leg": e.leg,
            "txn_ids": list(e.txn_ids),
            "category": e.category,
            "max_confidence": round(e.max_confidence, 6),
            "suggested_action": e.suggested_action,
            "amount_paise": e.amount_paise,
            "age_days": e.age_days,
            "priority": round(e.priority, 4),
            "evidence": e.evidence,
        },
        idempotency_key=f"exception:{e.exception_id}",
    )


def record_tier3(store: EventStore, report, *, leg: str) -> list[Event]:
    """The question and the answer, as two events per arbitration.

    Both are written, always, including when the answer was
    ``insufficient_evidence`` or an error. A tier that only records the times it
    had something to say would make its own hit rate unmeasurable, and "the arbiter
    was asked 23 times and declined 19 of them" is the single most useful fact about
    it -- more useful than any individual rationale.

    The **request** event carries every candidate that was offered, with its score.
    An arbitration recorded without its alternatives has the same defect
    POSTMORTEM #4 was about, one level up: the answer "candidate 2" is unreadable
    six months later unless what candidates 1 and 3 were is also on the record.

    The **return** event repeats the transaction ids for the same reason
    ``MatchConfirmed`` does -- ``EventStore.touching`` finds events by scanning
    payloads for an id, so an arbitration that does not name the transactions it
    concerned is absent from the audit trail of the thing it decided.
    """
    out: list[Event] = []
    for group, a in zip(report.groups, report.arbitrations):
        anchor = group[0].match_key
        offered = [
            {
                "index": i,
                "match_key": c.match_key,
                "left_ids": list(c.left_ids),
                "right_ids": list(c.right_ids),
                "tier": c.tier,
                "algorithm": c.algorithm,
                "score": round(c.score, 6),
                "residual_paise": c.residual_paise,
                "exposure_paise": c.exposure_paise,
            }
            for i, c in enumerate(group, start=1)
        ]
        req = {
            "leg": leg,
            "arbiter": a.arbiter,
            "band": report.stats.get("band"),
            "candidates_offered": offered,
            "candidate_count": len(offered),
            "materiality_paise": MATERIALITY_PAISE,
            "max_confidence_delta": report.stats.get("max_confidence_delta"),
        }
        out.append(
            store.append(
                "Tier3ArbitrationRequested", anchor, req,
                idempotency_key=(
                    f"tier3req:{anchor}:{sha256_hex(canonical_json(req))[:16]}"
                ),
            )
        )

        chosen = group[a.chosen_index - 1] if 1 <= a.chosen_index <= len(group) else None
        ret = {
            "leg": leg,
            **a.to_json(),
            "chosen_match_key": chosen.match_key if chosen else None,
            "score_before": round(chosen.score, 6) if chosen else None,
            "score_after": (
                round(max(0.0, min(1.0, chosen.score + a.confidence_delta)), 6)
                if chosen else None
            ),
            # Named so ``touching`` can find this event from any transaction the
            # arbitration concerned, not only from the one it chose.
            "left_ids": list(chosen.left_ids) if chosen else [],
            "right_ids": list(chosen.right_ids) if chosen else [],
            "group_txn_ids": sorted({i for c in group for i in c.all_ids()}),
        }
        out.append(
            store.append(
                "Tier3ArbitrationReturned", anchor, ret,
                idempotency_key=(
                    f"tier3ret:{anchor}:{sha256_hex(canonical_json(ret))[:16]}"
                ),
            )
        )
    return out


def journal_legs_for(c: Candidate, leg: str) -> list[dict[str, Any]] | None:
    """The double-entry postings a confirmed match implies, or None if it posts nothing.

    Only the gateway->bank leg moves money in the ledger: it is the event where cash
    arrives and a receivable clears. The books leg is an *attribution* -- which
    invoices this payout covered -- and posting it as well would double-count the
    same rupees, which is the single easiest way to make a reconciliation system
    produce a balance sheet that does not balance.

    A match whose two sides do not agree to the paise posts **nothing**. The
    receivable was booked at gross; clearing it requires knowing that exactly this
    payout arrived. If the bank credited a different amount, the correct outcome is
    an exception, not an entry with a plug -- and "post the bank amount and let the
    difference land in the receivable" is precisely the plug, just spelled in a way
    that balances.
    """
    if leg != "gateway_bank":
        return None
    if c.residual_paise != 0:
        return None
    net = c.right_amount_paise
    if net <= 0:
        return None
    ev = c.evidence or {}
    fees = int(ev.get("settlement_fees_paise") or 0)
    tax = int(ev.get("settlement_tax_paise") or 0)
    gross = net + fees + tax
    legs = [
        {"account": ACC_BANK, "debit_paise": net, "credit_paise": 0, "role": "cash"},
        {"account": ACC_GATEWAY_RECEIVABLE, "debit_paise": 0, "credit_paise": gross,
         "role": "clear_receivable"},
    ]
    if fees:
        legs.append({"account": ACC_FEES, "debit_paise": fees, "credit_paise": 0,
                     "role": "fee_expense"})
    if tax:
        legs.append({"account": ACC_GST_INPUT, "debit_paise": tax, "credit_paise": 0,
                     "role": "gst_input"})
    return legs


def post_journal_entries(
    store: EventStore, decisions: list[Decision], *, leg: str
) -> tuple[int, int, int]:
    """Post one balanced entry per confirmed match. Returns (entries, paise, rejected).

    An entry that does not balance is *not* posted. The alternative -- forcing it to
    balance with a suspense plug -- turns an arithmetic bug into a permanent
    line item that someone writes off next quarter, and the bug survives. Raising
    here means an accounting error is a build failure rather than a discovery.
    """
    entries = 0
    total = 0
    rejected = 0
    for d in decisions:
        if d.action != "auto_confirm":
            continue
        legs = journal_legs_for(d.candidate, leg)
        if not legs:
            continue
        debit = sum(l["debit_paise"] for l in legs)
        credit = sum(l["credit_paise"] for l in legs)
        if debit != credit:
            rejected += 1
            raise AssertionError(
                f"refusing to post an unbalanced entry for {d.candidate.match_key}: "
                f"debits {debit} != credits {credit} (residual {debit - credit} paise). "
                "A reconciliation engine that plugs the difference has not reconciled "
                "anything."
            )
        entry_id = "je-" + sha256_hex(d.candidate.match_key)[:16]
        store.append(
            "JournalEntryPosted",
            entry_id,
            {
                "source_match_key": d.candidate.match_key,
                "leg": leg,
                "left_ids": list(d.candidate.left_ids),
                "right_ids": list(d.candidate.right_ids),
                "legs": legs,
                "total_debit_paise": debit,
                "total_credit_paise": credit,
                "confidence": round(d.candidate.score, 6),
                "posted_by": "agent",
            },
            idempotency_key=f"journal:{entry_id}",
        )
        entries += 1
        total += debit
    return entries, total, rejected


def record_snapshot(
    store: EventStore, batch_id: str, metrics: dict, *, as_of: date | None = None
) -> Event:
    return store.append(
        "ReconciliationSnapshot",
        batch_id,
        {"as_of": as_of.isoformat() if as_of else None, **metrics},
        idempotency_key=f"snapshot:{batch_id}:{sha256_hex(canonical_json(metrics))[:16]}",
    )


def record_run(store: EventStore, batch: IngestedBatch, run) -> RecordingSummary:
    """Write a whole engine run to the store, in causal order.

    Order matters and is not cosmetic: ingestion before matching before journal
    postings before the snapshot. A replay walks the stream forward and rebuilds
    state, so an event that arrives before its cause produces a projection that is
    briefly, silently wrong -- and "briefly" is forever if nothing replays again.
    """
    s = RecordingSummary(by_type={})
    before = store.count()

    record_ingestion(store, batch)

    # Enrich gateway->bank candidates with the settlement's own fee and tax, so the
    # journal entry can balance without looking anything up at posting time.
    gw = {t.txn_id: t for t in batch.gateway.settlements}
    for leg_name, leg_run in run.legs.items():
        # Tier 3 first, where it ran. The ordering is causal, not cosmetic: the
        # arbiter adjusts scores and *then* the policy produces the decisions the
        # proposal events below record, so an arbitration written after them would
        # appear in the trail as a consequence of the decision it actually caused.
        if leg_run.tier3 is not None:
            record_tier3(store, leg_run.tier3, leg=leg_name)

        decisions = [
            *leg_run.decisions.auto_confirmed,
            *leg_run.decisions.proposed,
            *leg_run.decisions.rejected,
        ]
        enriched: list[Decision] = []
        for d in decisions:
            c = d.candidate
            sid = next((i for i in c.left_ids if i in gw), None)
            if sid and leg_name == "gateway_bank":
                from dataclasses import replace

                c = replace(
                    c,
                    evidence={
                        **c.evidence,
                        "settlement_fees_paise": gw[sid].fees_paise,
                        "settlement_tax_paise": gw[sid].tax_paise,
                    },
                )
                d = replace(d, candidate=c)
            enriched.append(d)
            record_decision(store, d, leg=leg_name)

        for c in leg_run.wash_pairs:
            store.append(
                "MatchConfirmed",
                c.match_key,
                {
                    "leg": "bank_internal", "confirmed_by": "agent", "confidence": 1.0,
                    "tier": 1, "algorithm": c.algorithm,
                    "rationale": "credit and debit of equal magnitude sharing a "
                                 "reference; net effect on cash is zero",
                    "gate": "deterministic_rule",
                    "left_ids": list(c.left_ids), "right_ids": list(c.right_ids),
                },
                idempotency_key=f"confirm:{c.match_key}",
            )

        for e in leg_run.exceptions.exceptions:
            record_exception(store, e)

        for w in leg_run.exceptions.written_off:
            store.append(
                "SuspenseWriteOff",
                w["txn_id"],
                {
                    "leg": leg_name,
                    "amount_paise": w["amount_paise"],
                    "account": ACC_SUSPENSE,
                    "reason": w["reason"],
                    "category": w["category"],
                    "value_date": w["value_date"],
                    "narration": w["narration"],
                },
                idempotency_key=f"writeoff:txn:{w['txn_id']}",
            )

        for d in leg_run.decisions.exempted_immaterial:
            store.append(
                "SuspenseWriteOff",
                d.candidate.match_key,
                {
                    "leg": leg_name,
                    "amount_paise": d.candidate.exposure_paise,
                    "account": ACC_SUSPENSE,
                    "reason": d.reason,
                },
                idempotency_key=f"writeoff:{d.candidate.match_key}",
            )

        n, paise, rej = post_journal_entries(store, enriched, leg=leg_name)
        s.journal_entries += n
        s.journal_paise += paise
        s.unbalanced_rejected += rej

    s.events_written = store.count() - before
    s.by_type = store.type_counts()
    return s
