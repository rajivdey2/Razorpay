"""Exceptions: what the engine looked at, could not resolve, and how sure it is.

Every transaction-side fact that survives tier 1 and tier 2 without a resolution
becomes an exception. The ordering matters more than the listing: this is the
queue a human actually works, so it is sorted by ``priority`` -- amount at risk
amplified by age -- rather than by transaction date, which would bury a
Rs 400,000 stuck payout behind a week of Rs 300 rounding breaks.

What the suggestion classifier knows
-----------------------------------
The category name is *derived from the failure signature*, not invented:
* a settlement with a valid UTR whose amount is wholly unmatched inside its
  window -> ``missing_settlement`` (the gateway says it paid, the bank disagrees).
* a candidate whose best hypothesis just barely failed the threshold ->
  ``low_confidence`` (wants a human eye, not a rule change).
* a settlement and a bank credit whose amounts sit within a few paise but which
  no reference supports -> ``amount_proximity`` (very likely right, but proves
  nothing to a reviewer).
* a bank credit entirely without candidates -> ``orphan_bank_credit``.
* a bank *debit* that tier 1's wash-pair rule did not net against anything ->
  ``unmatched_debit`` (a return or a fee the merchant needs to see).

The exception itself carries ``near_misses``: the strongest hypotheses the engine
entertained for it, with their features and scores. That is what lets a human
review in one screen, and it is what lets the evaluation report "of everything the
system did not resolve, humans agreed with fraction X" -- a number the workbench
computes from actual accept/reject decisions rather than from the engine's own
opinion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from app.core.config import LEGS, IMMATERIAL_PAISE
from app.core.schema import NormalizedTxn

from .blocking import BlockingGraph
from .policy import Policy
from .types import Candidate, Decision, Exception_


@dataclass
class ExceptionReport:
    exceptions: list[Exception_] = field(default_factory=list)
    #: Unmatched lines below the attention floor, written off to suspense.
    #:
    #: These used to be skipped with a bare ``continue``, which meant a Rs 9.99
    #: withholding accrual could be neither matched nor flagged nor recorded --
    #: it just disappeared. ``ReconciliationRun.assert_complete`` caught it on the
    #: 2x-break-rate stress run, which is the whole reason that assertion exists.
    #: An item too small to be worth a human's time still has to be *accounted
    #: for*; "not worth reviewing" and "not recorded" are different states.
    written_off: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def ranked(self) -> list[Exception_]:
        return sorted(self.exceptions, key=lambda e: (-e.priority, e.exception_id))

    def to_json(self) -> dict:
        return {
            "stats": self.stats,
            "exceptions": [e.to_json() for e in self.ranked()],
        }

    def summary(self) -> dict:
        total = sum(abs(e.amount_paise) for e in self.exceptions)
        n_high_priority = sum(1 for e in self.exceptions if e.priority > 8.0)
        return {
            "count": len(self.exceptions),
            "amount_paise_outstanding": total,
            "high_priority_count": n_high_priority,
            "by_category": _by_category(self.exceptions),
        }


def _by_category(excs: list[Exception_]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in excs:
        out[e.category] = out.get(e.category, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _cat_suggest(c: Candidate, leg: str) -> tuple[str, str]:
    """Classify a decision that did not get a resolution.

    Returns (category, suggested_action) for the exception.
    """
    if not c:
        return ("unmatched", "review this line manually")
    if c.features.get("utr_exact", 0.0) == 1.0 or c.features.get("amount_exact", 0.0) == 1.0:
        return (
            "low_confidence_on_exact_signals",
            "amounts and references agree but the engine could not reach the "
            "confirmation threshold; this is likely correct and worth one human click",
        )
    if c.features.get("amount_rel_delta", 1.0) < 0.05:
        return (
            "amount_proximity",
            "amounts are within 5% but no reference supports the pairing; "
            "likely true, needs one confirmation",
        )
    return (
        "no_strong_hypothesis",
        "evidence is too thin to automate; check the gateway lifecycle and the "
        "bank feed for this amount",
    )


def _suggest_txn(t: NormalizedTxn, leg: str) -> tuple[str, str]:
    if t.source == "gateway":
        return (
            "missing_settlement",
            "gateway marks this processed but no bank credit is inside the window; "
            "escalate to the sponsor bank or wait for the NEFT window to close",
        )
    if t.source == "bank" and t.amount_paise < 0:
        return (
            "unmatched_debit",
            "a debit with no corresponding wash credit; check for a return, a "
            "chargeback, or a fee the bank applied",
        )
    if t.source == "bank":
        return (
            "orphan_bank_credit",
            "money in the bank with no settlement counterpart in the window; "
            "check interest, another PSP, or a direct customer transfer",
        )
    if t.source == "books":
        return (
            "awaiting_settlement",
            "booked as receivable but its order is not linked to any payout; "
            "check whether this is still in the settlement queue",
        )
    return ("unmatched", "no hypothesis, review")


def build_exceptions(
    leg: str,
    graph: BlockingGraph,
    decisions: list[Decision],
    *,
    policy: Policy,
    resolved_left: set[str],
    resolved_right: set[str],
    unresolved: set[str],
    as_of: date,
    immaterial_floor: int = IMMATERIAL_PAISE,
) -> ExceptionReport:
    """One exception per residual fact, with near-miss hypotheses attached.

    ``resolved_left``/``resolved_right`` are the ids claimed by anything that
    auto-confirmed or that a human is reviewing; ``unresolved`` is the ids on the
    *policy* side that were not even proposed. The two sets are what makes the
    report complete: an unresolved id that is also not in the decision list is a
    fact with no candidate at all, which is a different category from one whose
    candidates all failed.
    """
    rep = ExceptionReport()

    best_by_id: dict[str, Candidate] = {}
    for d in decisions:
        for cid in d.candidate.all_ids():
            if cid not in best_by_id or d.candidate.score > best_by_id[cid].score:
                best_by_id[cid] = d.candidate

    all_ids = set(graph.left_by_id) | set(graph.right_by_id)
    seen_exception_for: set[str] = set()

    def add(txns: list[NormalizedTxn], tag: str, category: str, action: str) -> None:
        ids = tuple(sorted(t.txn_id for t in txns))
        key = tag + "|" + "|".join(ids)
        if key in seen_exception_for:
            return
        seen_exception_for.add(key)
        best = best_by_id.get(ids[0])
        amount = sum(abs(t.amount_paise) for t in txns)
        age = max(0, (as_of - max(t.value_date for t in txns)).days)
        rep.exceptions.append(
            Exception_(
                exception_id=tag + "-" + "".join(i.split(":")[-1][:8] for i in ids),
                leg=leg,
                txn_ids=ids,
                category=category,
                max_confidence=best.score if best else 0.0,
                suggested_action=action,
                amount_paise=amount,
                age_days=age,
                evidence={
                    "basis": "id appears in neither the resolved nor the proposed set",
                    "txn_details": [
                        {
                            "txn_id": t.txn_id,
                            "amount_paise": t.amount_paise,
                            "value_date": t.value_date.isoformat(),
                            "narration": (t.narration or "")[:100],
                            "ref_candidates": list(t.ref_candidates or ()),
                            "source_file": (t.provenance or {}).get("source_file"),
                        }
                        for t in txns
                    ],
                },
                near_misses=tuple(
                    c
                    for c in best_by_id.values()
                    if c.match_key and c.match_key != (best.match_key if best else None)
                )
                if best
                else (),
            )
        )

    # Side that is not resolved. On the gateway->bank leg this is the gateway side.
    for lid in sorted(graph.left_by_id):
        if lid in resolved_left or lid in unresolved:
            continue
        t = graph.left_by_id[lid]
        cand = best_by_id.get(lid)
        if cand and cand.score >= policy.auto_confirm_threshold:
            continue
        cat, action = _cat_suggest(cand, leg) if cand else _suggest_txn(t, leg)
        if t.amount_paise and abs(t.amount_paise) < immaterial_floor:
            rep.written_off.append(_write_off(t, cat, immaterial_floor))
            continue
        add([t], lid.split(":")[0], cat, action)

    # Right side that is not resolved.
    for rid in sorted(graph.right_by_id):
        if rid in resolved_right:
            continue
        t = graph.right_by_id[rid]
        cand = best_by_id.get(rid)
        if cand and cand.score >= policy.auto_confirm_threshold:
            continue
        cat, action = _cat_suggest(cand, leg) if cand else _suggest_txn(t, leg)
        if t.amount_paise and abs(t.amount_paise) < immaterial_floor:
            rep.written_off.append(_write_off(t, cat, immaterial_floor))
            continue
        add([t], rid.split(":")[0], cat, action)

    rep.stats = {
        "exceptions": len(rep.exceptions),
        "by_category": _by_category(rep.exceptions),
        "amount_outstanding_paise": sum(abs(e.amount_paise) for e in rep.exceptions),
        "written_off": len(rep.written_off),
        "written_off_paise": sum(abs(w["amount_paise"]) for w in rep.written_off),
    }
    return rep


def _write_off(t: NormalizedTxn, category: str, floor: int) -> dict:
    return {
        "txn_id": t.txn_id,
        "amount_paise": t.amount_paise,
        "value_date": t.value_date.isoformat(),
        "narration": (t.narration or "")[:100],
        "category": category,
        "reason": f"unmatched and below the {floor} paise attention floor; "
                  "written off to suspense with an event rather than reviewed",
    }
