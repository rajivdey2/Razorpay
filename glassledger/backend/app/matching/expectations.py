"""Deriving what the books *should* say, from the gateway's own data.

This module exists because of a measurement. The first working version of the
books leg set each settlement's subset target to ``settlement_net - sum(keyed
entries)`` and searched an 18-day window. Result: pools averaging 75 entries,
3.4 million search nodes, 211 of 288 pools hitting the solution cap, 827 set-packing
conflicts, and 5.3 seconds for a 900-payment month. The search was not failing to
find the right subset -- it was finding *hundreds* of subsets that summed to within
tolerance, and picking among them with no real evidence.

The problem was never the solver. It was that the target and the window were both
far looser than the available information justified. Three constraints were sitting
unused in data already ingested:

**1. A settlement batch covers one capture day.** Tier 1 has already joined the
keyed entries, so their ``booked_on`` dates are known -- and the unkeyed entries in
the same batch share them. That replaces an 18-day window with a 1-3 day one.

**2. The gateway reports which payments are in the payout.** The recon report lists
every payment with its ``order_id``. Subtract the ones whose invoice tier 1 already
claimed, and what remains is *exactly* the set of payments whose invoices are
unkeyed. Their gross amounts are known, so the target can be built from
first principles instead of as a leftover.

**3. The merchant's own fee assumption is recoverable.** Every keyed entry pairs a
book amount with a gateway gross, and the ratio between them is the blended fee
rate the merchant's books apply. Estimating it from the *deterministically matched*
population and applying it to the unkeyed one is self-calibrating: it adapts to a
merchant who books receivables gross, or net of 2%, or net of a negotiated rate,
without being told which.

The result is a target with a ~150bp tolerance instead of a ~2.5% one, on a window
a fifth the size. But the bigger realisation came next, and it changed the solver:
if the recon report gives each unclaimed payment's gross *individually*, then the
expected book amount for each one is individually known too. The books leg is not
a subset-sum problem at all for the majority of its volume -- it is a per-component
*assignment* problem, and subset-sum was the wrong tool applied with great
efficiency.

``expected_components`` builds one synthetic expectation per unclaimed payment and
refund, which the engine then matches 1:1 against unkeyed book entries with a ~2%
amount tolerance. Subset-sum keeps only the work it is genuinely right for: the
leftover accrual and adjustment lines that have no component counterpart at all.

Worth stating plainly, because it is the general lesson of the whole project: on
this kind of problem, information you already have beats a better search almost
every time -- and picking the solver that matches the structure beats tuning the
one that does not.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

from app.core.schema import NormalizedTxn
from app.ingestion import BooksIngest, GatewayIngest

#: Tolerance around the derived target, in basis points of gross. Has to cover
#: what remains genuinely unknowable to the engine:
#:   ~100bp  a withholding accrual the gateway never sees
#:   ~130bp  FX movement between invoice date and settlement date
#:   ~10bp   per-payment rounding of fee and GST
#: Set above their sum rather than at it, because they compound in the same
#: direction on an FX invoice that also carries withholding.
DERIVED_TOLERANCE_BPS = 165

#: Floor for small batches, where a basis-point figure is a few paise.
DERIVED_TOLERANCE_FLOOR_PAISE = 30 * 100


@dataclass
class BooksExpectation:
    settlement_id: str
    #: What the still-unclaimed book entries for this settlement should sum to.
    target_paise: int
    tolerance_paise: int
    #: Dates the unclaimed entries are expected to carry.
    date_hints: set[date] = field(default_factory=set)
    #: Diagnostics for the evidence panel.
    detail: dict = field(default_factory=dict)


def estimate_books_fee_bps(
    keyed_matches: list,          # list[Candidate] from tier1.match_order_keys
    gateway: GatewayIngest,
    books: BooksIngest,
) -> tuple[int, dict]:
    """Recover the merchant's blended fee assumption from the deterministic matches.

    For every book entry that tier 1 joined by order key, the gateway's recon
    report gives the payment's gross. ``1 - entry/gross`` is the fee fraction the
    merchant's books applied to that sale. The *median* over all such pairs is the
    blended assumption.

    Median rather than mean, deliberately: an FX invoice's ratio is contaminated by
    the rate movement, and a credit note's is meaningless. Both are outliers, and a
    mean would let a handful of them shift the estimate enough to make every
    derived target slightly wrong -- which is worse than useless, because a
    systematically biased target looks like a systematically unexplained residual.
    """
    entry_by_id = {t.txn_id: t for t in books.entries}
    gross_by_order: dict[str, int] = {}
    for sid, comps in gateway.components.items():
        for c in comps:
            if c.kind == "payment" and c.order_ref:
                gross_by_order[c.order_ref] = c.amount_paise

    ratios: list[float] = []
    for cand in keyed_matches:
        entry = entry_by_id.get(cand.left_ids[0])
        if entry is None:
            continue
        order_ref = (entry.provenance or {}).get("order_ref")
        gross = gross_by_order.get(order_ref or "")
        if not gross or gross <= 0 or entry.amount_paise <= 0:
            continue
        # Only domestic, positive receivables carry a clean signal.
        if (entry.provenance or {}).get("kind") != "receivable":
            continue
        ratios.append(1.0 - entry.amount_paise / gross)

    if len(ratios) < 20:
        # Not enough deterministic matches to learn from. Falling back to a
        # stated default is fine; silently using a bad estimate is not.
        return 236, {
            "method": "default",
            "reason": f"only {len(ratios)} keyed pairs; need 20 to estimate",
            "bps": 236,
        }

    bps = int(round(statistics.median(ratios) * 10_000))
    return bps, {
        "method": "median_of_tier1_keyed_pairs",
        "samples": len(ratios),
        "bps": bps,
        "p10_bps": int(round(sorted(ratios)[len(ratios) // 10] * 10_000)),
        "p90_bps": int(round(sorted(ratios)[len(ratios) * 9 // 10] * 10_000)),
    }


def claimed_component_ids(
    gateway: GatewayIngest,
    books: BooksIngest,
    claimed_entry_ids: set[str],
) -> set[str]:
    """Which payments/refunds already have their book entry accounted for.

    Resolves through ``order_ref``, so it only sees what tier 1's key join claimed.
    Tier 2a's component matches are added by the caller, because tier 2a knows the
    component id directly and does not need to go through the key.
    """
    out: set[str] = set()
    for comps in gateway.components.values():
        for c in comps:
            entry_ids = books.order_index.get(c.order_ref or "", [])
            if entry_ids and any(f"books:{e}" in claimed_entry_ids for e in entry_ids):
                out.add(c.component_id)
    return out


def build_expectations(
    gateway: GatewayIngest,
    books: BooksIngest,
    *,
    claimed_component_ids: set[str],
    claimed_entries_by_settlement: dict[str, list[NormalizedTxn]],
    fee_bps: int,
) -> dict[str, BooksExpectation]:
    """One expectation per settlement: target, tolerance, and expected dates.

    ``claimed_component_ids`` is what makes this callable twice. It is invoked once
    after tier 1 (to build the per-component predictions) and again after tier 2a
    (to set the residual target for tier 2b). Recomputing rather than reusing the
    first result matters: after tier 2a has claimed 200 more entries, a target still
    sized for those entries would send tier 2b hunting for money that is already
    accounted for -- and it would find something, because a large enough target
    inside a loose enough tolerance always does.
    """
    out: dict[str, BooksExpectation] = {}
    for s in gateway.settlements:
        sid = s.txn_id
        comps = gateway.components.get(s.external_id, [])
        gross = sum(c.amount_paise for c in comps) or (
            s.amount_paise + s.fees_paise + s.tax_paise
        )

        unclaimed_gross = 0
        n_unclaimed = 0
        refund_total = 0
        for c in comps:
            if c.component_id in claimed_component_ids:
                continue
            if c.kind == "refund":
                refund_total += c.amount_paise
            else:
                unclaimed_gross += c.amount_paise
                n_unclaimed += 1

        # Refunds are booked as credit notes at their full amount, not net of a
        # fee -- a credit note reverses the invoice, and the gateway's fee on the
        # original sale is not refunded.
        expected_receivables = unclaimed_gross - (unclaimed_gross * fee_bps) // 10_000
        target = expected_receivables + refund_total

        tol = max(
            DERIVED_TOLERANCE_FLOOR_PAISE,
            (abs(gross) * DERIVED_TOLERANCE_BPS) // 10_000,
        )

        claimed = claimed_entries_by_settlement.get(sid, [])
        hints = {e.value_date for e in claimed}
        hints.add(s.value_date)   # accruals and credit notes book on payout day

        out[sid] = BooksExpectation(
            settlement_id=sid,
            target_paise=target,
            tolerance_paise=tol,
            date_hints=hints if claimed else set(),
            detail={
                "derivation": "sum(unclaimed payment gross) * (1 - merchant_fee) "
                              "+ refunds booked at full value",
                "merchant_fee_bps": fee_bps,
                "unclaimed_payments": n_unclaimed,
                "unclaimed_gross_paise": unclaimed_gross,
                "refunds_paise": refund_total,
                "tolerance_bps": DERIVED_TOLERANCE_BPS,
                "date_hints_from_matched_siblings": sorted(d.isoformat() for d in hints)
                if claimed else [],
            },
        )
    return out


# ---------------------------------------------------------------------------
# Per-component expectations -- the assignment formulation
# ---------------------------------------------------------------------------

#: Amount tolerance for matching one expected component against one book entry.
#: Has to absorb FX drift on an international invoice (up to ~130bp) and per-payment
#: fee rounding, and nothing else -- withholding is a batch-level accrual and never
#: touches an individual invoice line.
COMPONENT_TOLERANCE_BPS = 220
COMPONENT_TOLERANCE_FLOOR_PAISE = 20 * 100


def expected_components(
    gateway: GatewayIngest,
    books: BooksIngest,
    *,
    claimed_component_ids: set[str],
    fee_bps: int,
) -> list[NormalizedTxn]:
    """One synthetic transaction per unclaimed payment/refund in every settlement.

    These are not real transactions -- they are *predictions* of what the merchant's
    books ought to contain, derived entirely from the gateway's own recon report and
    the fee ratio recovered from tier 1. Expressing them as ``NormalizedTxn`` lets
    the existing blocking, feature extraction and assignment code run over them
    unchanged, which is worth more than the mild dishonesty of the type name.

    ``external_id`` is the component id, and ``provenance["settlement_txn_id"]``
    carries the settlement the match should ultimately be attributed to. The engine
    re-targets each resulting candidate from the component to the settlement, so
    ground truth is compared against the pairing that actually matters.

    ``value_date`` is set to the settlement's own value date rather than the
    payment's capture date. The gateway's recon report does carry a capture
    timestamp, but the leg's date prior is defined as
    ``settlement_date - booked_on`` and this keeps every date feature on that one
    scale. Substituting the capture date here would put the same quantity on two
    different scales in one leg -- a bug that costs nothing today and two hours of
    confusion the first time a prior gets retuned.
    """
    out: list[NormalizedTxn] = []
    for s in gateway.settlements:
        comps = gateway.components.get(s.external_id, [])
        gross_total = sum(c.amount_paise for c in comps) or s.amount_paise
        for c in comps:
            entry_ids = books.order_index.get(c.order_ref or "", [])
            if c.component_id in claimed_component_ids:
                continue
            if c.kind == "refund":
                # A credit note reverses the invoice at full value: the gateway's
                # fee on the original sale is not refunded, so no fee ratio applies.
                expected = c.amount_paise
                memo = f"expected credit note for refund {c.component_id}"
            else:
                expected = c.amount_paise - (c.amount_paise * fee_bps) // 10_000
                memo = (
                    f"expected receivable for payment {c.component_id} "
                    f"(gross {c.amount_paise}, merchant fee {fee_bps}bp)"
                )
            if expected == 0:
                continue
            out.append(
                NormalizedTxn(
                    source="gateway",
                    external_id=c.component_id,
                    amount_paise=expected,
                    currency="INR",
                    utr=None,
                    narration=memo,
                    value_date=s.value_date,
                    ref_candidates=(c.order_ref,) if c.order_ref else (),
                    raw_payload_hash=f"expectation:{c.component_id}:{expected}",
                    provenance={
                        "synthetic": True,
                        "settlement_txn_id": s.txn_id,
                        "component_kind": c.kind,
                        "component_id": c.component_id,
                        "gross_paise": c.amount_paise,
                        "settlement_gross_paise": gross_total,
                        "invoice_currency": c.invoice_currency,
                        "fee_bps_applied": fee_bps if c.kind == "payment" else 0,
                    },
                )
            )
    return out


def component_tolerance(expected_paise: int) -> int:
    return max(
        COMPONENT_TOLERANCE_FLOOR_PAISE,
        (abs(expected_paise) * COMPONENT_TOLERANCE_BPS) // 10_000,
    )
