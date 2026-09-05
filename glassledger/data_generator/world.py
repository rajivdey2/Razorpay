"""The simulated merchant's world, before it gets written out to files.

The generator does not emit ``NormalizedTxn`` objects. It builds a mutable world
of payments, settlements, bank lines and book entries, lets each break pattern
mutate that world, and only then serialises to *raw source formats* -- Razorpay
settlement JSON, three different bank statement dialects, a books CSV. The
ingestion layer then has to parse those files back.

That round trip is deliberate. A generator that hands the matcher pre-normalised
objects tests the matcher and nothing else; one that goes through the real file
formats also tests the parsers, and parsers are where reconciliation systems
actually break in production (a bank changes a column header, a date flips from
DD-MM to MM-DD, and every downstream number is quietly wrong).

Ground truth is *derived from the final world state*, never accumulated as the
patterns run. If a later pattern moves a bank line, the answer key moves with
it, because the key is a function of the world rather than a log of intentions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from app.core.money import apply_bps

# Bank-line roles. Only the first two are matchable; the rest are the honest
# exception population and carry a reason string straight into the answer key.
BankRole = Literal[
    "settlement_credit",   # the credit that corresponds to a settlement
    "split_part",          # one of N credits that together make a settlement
    "orphan",              # money with no gateway counterpart
    "reversal_debit",      # the debit half of a failed-then-re-sent settlement
    "stale_credit",        # the credit that was later reversed
]


@dataclass
class Payment:
    payment_id: str
    gross_paise: int            # INR paise the customer actually paid
    captured_on: date
    instrument: str             # upi / card / netbanking / intl_card
    invoice_currency: str = "INR"
    invoice_amount_minor: int = 0   # in invoice currency's minor unit
    booking_fx_micros: int = 1_000_000   # invoice ccy -> INR, x1e6, at invoice date
    settlement_fx_micros: int = 1_000_000  # ...at settlement date
    settlement_id: str | None = None


@dataclass
class Refund:
    refund_id: str
    payment_id: str
    amount_paise: int           # positive magnitude; netted out of a settlement
    created_on: date
    settlement_id: str | None = None


@dataclass
class Settlement:
    settlement_id: str
    utr: str
    created_on: date
    payment_ids: list[str] = field(default_factory=list)
    refund_ids: list[str] = field(default_factory=list)
    fees_paise: int = 0         # what the gateway ACTUALLY deducted
    tax_paise: int = 0          # GST on that fee
    status: str = "processed"
    patterns: set[str] = field(default_factory=set)

    def gross_paise(self, world: "World") -> int:
        pay = sum(world.payments[p].gross_paise for p in self.payment_ids)
        ref = sum(world.refunds[r].amount_paise for r in self.refund_ids)
        return pay - ref

    def net_paise(self, world: "World") -> int:
        """What actually hits the bank account.

        Razorpay's settlement entity reports ``amount`` as the net credited and
        ``fees``/``tax`` as what was withheld. Modelling it the other way round
        -- amount as gross -- is the classic mistake, and it makes every
        gateway<->bank comparison off by exactly the MDR, which then looks like
        a systemic timing problem rather than a units problem.
        """
        return self.gross_paise(world) - self.fees_paise - self.tax_paise


@dataclass
class BankLine:
    bank_ref: str
    value_date: date
    posted_date: date
    amount_paise: int           # signed: credit +, debit -
    narration: str
    utr_field: str | None
    role: BankRole
    settlement_id: str | None = None
    unmatchable_reason: str | None = None
    bank_format: str = "hdfc"


@dataclass
class BookEntry:
    entry_id: str
    amount_paise: int           # signed expected receivable
    booked_on: date
    memo: str
    payment_id: str | None = None
    settlement_id: str | None = None
    unmatchable_reason: str | None = None
    kind: str = "receivable"    # receivable | refund | tds | fx_adjustment


@dataclass
class World:
    seed: int
    start_date: date
    payments: dict[str, Payment] = field(default_factory=dict)
    refunds: dict[str, Refund] = field(default_factory=dict)
    settlements: dict[str, Settlement] = field(default_factory=dict)
    bank_lines: list[BankLine] = field(default_factory=list)
    book_entries: list[BookEntry] = field(default_factory=list)
    injection_log: list[dict] = field(default_factory=list)

    # -- helpers used by the break-pattern modules ---------------------------

    def note(self, pattern: str, **details) -> None:
        """Record that a pattern fired. Diagnostics only.

        The answer key is derived from world state, not from this log, so a
        pattern that forgets to call ``note`` produces a less readable report
        but never a wrong metric.
        """
        self.injection_log.append({"pattern": pattern, **details})

    def credits_for(self, settlement_id: str) -> list[BankLine]:
        return [
            b for b in self.bank_lines
            if b.settlement_id == settlement_id
            and b.role in ("settlement_credit", "split_part")
        ]

    def entries_for(self, settlement_id: str) -> list[BookEntry]:
        return [b for b in self.book_entries if b.settlement_id == settlement_id]

    def settlements_in_order(self) -> list[Settlement]:
        return sorted(self.settlements.values(), key=lambda s: (s.created_on, s.settlement_id))


# ---------------------------------------------------------------------------
# Fee arithmetic used by both the gateway side and the merchant's books
# ---------------------------------------------------------------------------

#: What the gateway actually charges, per instrument, in basis points. Real
#: pricing is negotiated and per-instrument; the merchant's books collapse this
#: to one blended guess, and the gap is break pattern #4.
ACTUAL_MDR_BPS = {
    "upi": 0,
    "netbanking": 190,
    "card": 215,
    "intl_card": 340,
}

GST_ON_FEE_BPS = 1800  # 18% of the fee


def actual_fee(payment: Payment) -> tuple[int, int]:
    """(fee, gst_on_fee) in paise for one payment, as the gateway computes it.

    Per-payment rounding, then summation -- not sum-then-round. This ordering is
    the source of the one-to-few-paise drift that makes a naive
    ``sum(gross) * 0.02`` comparison fail on large batches, and reproducing it
    faithfully is the difference between a generator that produces a realistic
    long tail and one that produces a clean dataset with noise sprinkled on top.
    """
    bps = ACTUAL_MDR_BPS.get(payment.instrument, 200)
    fee = apply_bps(payment.gross_paise, bps)
    return fee, apply_bps(fee, GST_ON_FEE_BPS)


def recompute_settlement_fees(world: World, s: Settlement) -> None:
    fee = tax = 0
    for pid in s.payment_ids:
        f, t = actual_fee(world.payments[pid])
        fee += f
        tax += t
    s.fees_paise = fee
    s.tax_paise = tax


def business_days_after(d: date, n: int, holidays: frozenset[date]) -> date:
    """T+n where n counts settlement-eligible days.

    Weekends and the holiday set push the credit later, which is exactly how
    timing drift (#9) shows up in a real bank feed: the gateway's own
    ``created_at`` is T+2 like clockwork, and the bank credit is not.
    """
    cur = d
    left = n
    while left > 0:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in holidays:
            left -= 1
    return cur
