"""Tier 1: deterministic matching. Cheap, exact, and where most of the volume goes.

Three rules, all identity-based rather than probabilistic:

1. **Wash-pair detection.** A credit and a debit of exactly equal magnitude
   sharing a reference, within a few days, cancel out. Neither is a settlement
   match; together they are a self-resolving non-event.

2. **Exact reference match** (gateway -> bank). The settlement's UTR appears among
   a bank line's extracted reference candidates, amounts agree to the paise, and
   the date sits in the cycle window -- and crucially, *no other* bank line makes
   the same claim.

3. **Order-key join** (books -> gateway). The merchant's ``order_ref`` resolves
   through the settlement recon report to a settlement id. A real key, so a real
   join.

Why the wash-pair rule earns its place
--------------------------------------
Break pattern #6 puts three lines in the feed for one payout: a credit, its
reversal, and the re-settlement. Rule 2 alone sees two credits with the correct
UTR and the correct amount and has to refuse to decide, pushing a case with an
unambiguous right answer into the probabilistic tier where it will sometimes be
resolved wrongly. Netting the reversal against the stale credit first leaves
exactly one credit standing, and rule 2 then resolves it deterministically.

That is worth stating plainly because it is the general lesson: reaching for a
model on a case that domain structure already determines does not make the system
smarter, it makes it *less* reliable. Tier 1 should be pushed as far as the domain
actually justifies, and no further.

Why rule 2 refuses when it is ambiguous
---------------------------------------
Multiple exact-amount, exact-UTR candidates is not "pick the closest". A rule that
returns a match here is wrong roughly half the time and reports full confidence
while doing it -- and a rule at 100% claimed confidence and 50% real accuracy is
worse than no rule at all, because it poisons the calibration of everything
downstream. Ambiguity is passed to tier 2, whose ambiguity features exist to price
it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.core.config import LegConfig
from app.core.schema import NormalizedTxn

from .types import Candidate

#: How far apart a credit and its reversal may sit and still be netted. Returns
#: are usually next-day; a week is generous without being so wide that two
#: unrelated equal-and-opposite movements get paired by coincidence.
WASH_WINDOW_DAYS = 8


@dataclass
class Tier1Result:
    matches: list[Candidate] = field(default_factory=list)
    wash_pairs: list[Candidate] = field(default_factory=list)
    #: Ids tier 1 has claimed, so later tiers do not reconsider them.
    consumed_left: set[str] = field(default_factory=set)
    consumed_right: set[str] = field(default_factory=set)
    #: Cases tier 1 identified but refused to decide, with the reason.
    deferred: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rule 1: wash pairs
# ---------------------------------------------------------------------------

def detect_wash_pairs(bank: list[NormalizedTxn]) -> tuple[list[Candidate], set[str]]:
    """Net equal-and-opposite bank movements that share a reference.

    Requires *both* an exact magnitude match and a shared reference candidate.
    Magnitude alone would pair a genuine Rs 4,998.50 settlement credit against an
    unrelated Rs 4,998.50 vendor payment and delete two real lines from the
    reconciliation -- and because both then vanish from the exception list, the
    error is invisible. The reference requirement makes a coincidental pairing
    essentially impossible.
    """
    by_ref: dict[str, list[NormalizedTxn]] = defaultdict(list)
    for t in bank:
        for ref in (t.ref_candidates or ()):
            if ref and len(ref) >= 10:
                by_ref[ref].append(t)

    pairs: list[Candidate] = []
    consumed: set[str] = set()
    for ref, group in sorted(by_ref.items()):
        credits = sorted(
            (t for t in group if t.amount_paise > 0 and t.txn_id not in consumed),
            key=lambda t: (t.value_date, t.txn_id),
        )
        debits = sorted(
            (t for t in group if t.amount_paise < 0 and t.txn_id not in consumed),
            key=lambda t: (t.value_date, t.txn_id),
        )
        for d in debits:
            if d.txn_id in consumed:
                continue
            for c in credits:
                if c.txn_id in consumed:
                    continue
                if c.amount_paise != -d.amount_paise:
                    continue
                if abs((d.value_date - c.value_date).days) > WASH_WINDOW_DAYS:
                    continue
                consumed.add(c.txn_id)
                consumed.add(d.txn_id)
                pairs.append(
                    Candidate(
                        leg="bank_internal",
                        left_ids=(c.txn_id,),
                        right_ids=(d.txn_id,),
                        tier=1,
                        algorithm="wash_pair_netting",
                        score=1.0,
                        left_amount_paise=c.amount_paise,
                        right_amount_paise=d.amount_paise,
                        evidence={
                            "rule": "credit and debit of equal magnitude sharing "
                                    f"reference {ref}, {abs((d.value_date - c.value_date).days)} "
                                    "day(s) apart",
                            "shared_reference": ref,
                            "credit": c.txn_id,
                            "debit": d.txn_id,
                            "net_effect_paise": 0,
                        },
                    )
                )
                break
    return pairs, consumed


# ---------------------------------------------------------------------------
# Rule 2: exact reference + amount (gateway -> bank)
# ---------------------------------------------------------------------------

def match_exact_reference(
    gateway: list[NormalizedTxn],
    bank: list[NormalizedTxn],
    leg: LegConfig,
    *,
    exclude_bank: frozenset[str] = frozenset(),
) -> Tier1Result:
    res = Tier1Result()

    by_ref: dict[str, list[NormalizedTxn]] = defaultdict(list)
    for t in bank:
        if t.txn_id in exclude_bank or t.amount_paise <= 0:
            continue
        for ref in (t.ref_candidates or ()):
            if ref:
                by_ref[ref].append(t)

    n_split = 0
    for s in gateway:
        if not s.utr:
            continue
        pool = [
            t for t in by_ref.get(s.utr, ())
            if t.txn_id not in res.consumed_right
            and -leg.date_window_before <= (t.value_date - s.value_date).days <= leg.date_window_after
        ]
        if not pool:
            continue

        exact = [t for t in pool if t.amount_paise == s.amount_paise]

        if len(exact) == 1 and len(pool) == 1:
            b = exact[0]
            res.matches.append(_pair(s, b, "exact_utr_and_amount", leg))
            res.consumed_left.add(s.txn_id)
            res.consumed_right.add(b.txn_id)
            continue

        # A split settlement: several credits carrying the UTR whose total is the
        # payout exactly. Unambiguous, so it belongs in tier 1 -- there is no
        # other reading of "these three lines sum to the amount and all three
        # quote the reference".
        if len(exact) == 0 and len(pool) > 1 and sum(t.amount_paise for t in pool) == s.amount_paise:
            res.matches.append(
                Candidate(
                    leg=leg.name,
                    left_ids=(s.txn_id,),
                    right_ids=tuple(sorted(t.txn_id for t in pool)),
                    tier=1,
                    algorithm="exact_utr_split_sum",
                    score=1.0,
                    left_amount_paise=s.amount_paise,
                    right_amount_paise=sum(t.amount_paise for t in pool),
                    features={},
                    evidence={
                        "rule": f"{len(pool)} bank credits quote UTR {s.utr} and sum "
                                "exactly to the settlement net",
                        "parts": [
                            {"txn_id": t.txn_id, "amount_paise": t.amount_paise,
                             "value_date": t.value_date.isoformat()}
                            for t in sorted(pool, key=lambda x: x.value_date)
                        ],
                        "residual_paise": 0,
                    },
                )
            )
            res.consumed_left.add(s.txn_id)
            res.consumed_right.update(t.txn_id for t in pool)
            n_split += 1
            continue

        if len(exact) > 1:
            res.deferred.append(
                {
                    "left_id": s.txn_id,
                    "reason": "ambiguous_exact_match",
                    "detail": f"{len(exact)} bank credits share UTR {s.utr} AND the exact "
                              "amount; identity is not determined",
                    "candidates": [t.txn_id for t in exact],
                }
            )
        elif exact:
            res.deferred.append(
                {
                    "left_id": s.txn_id,
                    "reason": "reference_matches_amount_does_not_sum",
                    "detail": f"UTR {s.utr} appears on {len(pool)} credits but neither a "
                              "single line nor their total equals the settlement net",
                    "candidates": [t.txn_id for t in pool],
                }
            )

    res.stats = {
        "exact_pairs": len(res.matches) - n_split,
        "exact_splits": n_split,
        "deferred": len(res.deferred),
    }
    return res


def _pair(left: NormalizedTxn, right: NormalizedTxn, algorithm: str, leg: LegConfig) -> Candidate:
    return Candidate(
        leg=leg.name,
        left_ids=(left.txn_id,),
        right_ids=(right.txn_id,),
        tier=1,
        algorithm=algorithm,
        score=1.0,
        left_amount_paise=left.amount_paise,
        right_amount_paise=right.amount_paise,
        evidence={
            "rule": f"reference {left.utr} matched exactly; amounts identical to the paise; "
                    f"{(right.value_date - left.value_date).days} day settlement lag",
            "matched_reference": left.utr,
            "residual_paise": right.amount_paise - left.amount_paise,
            "date_delta_days": (right.value_date - left.value_date).days,
        },
    )


# ---------------------------------------------------------------------------
# Rule 3: order-key join (books -> gateway)
# ---------------------------------------------------------------------------

def match_order_keys(
    books: list[NormalizedTxn],
    gateway: list[NormalizedTxn],
    order_to_settlement: dict[str, str],
    leg: LegConfig,
    *,
    unsettled_orders: frozenset[str] = frozenset(),
) -> Tier1Result:
    """Join book entries to settlements through the gateway's order id.

    Note what this does *not* claim. The join establishes that this invoice was
    part of this payout; it says nothing about whether the amounts agree, and they
    usually do not (fee drift, withholding, FX). So the residual is recorded on the
    candidate as evidence rather than being treated as a discrepancy to resolve --
    the group's arithmetic is checked once, at the group level, in
    ``residuals.attribute``.

    An order key that resolves to a *captured but unsettled* payment produces no
    match but is recorded as ``awaiting_settlement``. Same open item either way,
    completely different message to whoever works the queue: "we cannot explain
    this" versus "this settles on Thursday".
    """
    res = Tier1Result()
    gw = {t.txn_id: t for t in gateway}

    grouped: dict[str, list[NormalizedTxn]] = defaultdict(list)
    n_awaiting = 0
    for e in books:
        order_ref = (e.provenance or {}).get("order_ref") or ""
        if not order_ref:
            continue
        sid = order_to_settlement.get(order_ref)
        if sid is None:
            if order_ref in unsettled_orders:
                n_awaiting += 1
                res.deferred.append(
                    {
                        "left_id": e.txn_id,
                        "reason": "awaiting_settlement",
                        "detail": f"order {order_ref} is captured at the gateway but not yet "
                                  "included in any payout",
                        "candidates": [],
                    }
                )
            continue
        key = f"gateway:{sid}"
        if key in gw:
            grouped[key].append(e)

    for sid, entries in sorted(grouped.items()):
        s = gw[sid]
        entries = sorted(entries, key=lambda t: t.txn_id)
        total = sum(t.amount_paise for t in entries)
        for e in entries:
            res.matches.append(
                Candidate(
                    leg=leg.name,
                    left_ids=(e.txn_id,),
                    right_ids=(sid,),
                    tier=1,
                    algorithm="order_key_join",
                    score=1.0,
                    left_amount_paise=e.amount_paise,
                    right_amount_paise=s.amount_paise,
                    evidence={
                        "rule": "book entry carries a gateway order id that the settlement "
                                "recon report attributes to this payout",
                        "order_ref": (e.provenance or {}).get("order_ref"),
                        "group_size": len(entries),
                        "group_total_paise": total,
                        "note": "amount agreement is checked at group level, not per entry",
                    },
                )
            )
            res.consumed_left.add(e.txn_id)

    res.stats = {
        "joined_entries": len(res.matches),
        "settlements_touched": len(grouped),
        "awaiting_settlement": n_awaiting,
    }
    return res
