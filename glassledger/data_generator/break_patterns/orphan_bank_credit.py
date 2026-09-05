"""#7 Orphan bank credit: money in the bank with no gateway record.

Real merchant accounts are not fed only by one PSP. Interest postings, a second
gateway, a customer paying by direct transfer, a GST refund, a director's loan --
all land in the same statement the reconciliation reads.

These are deliberately generated at amounts drawn from the *same distribution* as
real settlements, and some are given plausible payment-ish narrations. An orphan
of Rs 47.13 labelled "SB INT" is trivial to exclude; an orphan of Rs 41,208.00
labelled "PAYU SETTLEMENT" is exactly the line that tempts a matcher into filling
a gap with the nearest available amount.

They are the population that measures whether the exception list is honest. Every
one of them belongs there, and each one a matcher "resolves" is a fabricated
match -- the failure mode that matters most in a finance context, because it does
not look like an error, it looks like a better match rate.
"""

from __future__ import annotations

from datetime import timedelta

from ..world import BankLine, World

PATTERN = "orphan_bank_credit"
PHASE = "post_bank"

SOURCES = [
    ("SB INT CREDIT {q}", (1_000, 40_000), "bank_interest"),
    ("PAYU SETTLEMENT {ref}", (500_000, 9_000_000), "other_psp"),
    ("CCAVENUE PAYOUT {ref}", (300_000, 4_500_000), "other_psp"),
    ("NEFT CR-{ref}-DIRECT CUSTOMER", (200_000, 3_000_000), "direct_customer"),
    ("GST REFUND ORDER {ref}", (1_000_000, 12_000_000), "statutory_refund"),
    ("RTGS CR {ref} DIRECTORS LOAN", (5_000_000, 40_000_000), "capital_injection"),
    ("CASH DEPOSIT BR-{q}", (50_000, 800_000), "cash_deposit"),
]


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    settlement_count = max(1, len(world.settlements))
    n = int(round(settlement_count * ctx.rate))
    if n == 0 and ctx.rate > 0:
        n = 1

    days = sorted({b.value_date for b in world.bank_lines}) or [world.start_date]
    fmts = sorted({b.bank_format for b in world.bank_lines}) or ["hdfc"]

    for i in range(n):
        template, (lo, hi), kind = rng.choice(SOURCES)
        amount = rng.randrange(lo, hi)
        # Round to a plausible granularity: humans and other systems transfer
        # round-ish numbers, and that roundness is itself a weak signal a matcher
        # could learn -- so a third of them are left un-rounded.
        if rng.random() < 0.65:
            amount = (amount // 100) * 100
        day = rng.choice(days) + timedelta(days=rng.choice([-1, 0, 0, 1]))
        ref = "".join(rng.choices("0123456789", k=12))
        world.bank_lines.append(
            BankLine(
                bank_ref=f"ORP{i:05d}",
                value_date=day,
                posted_date=day,
                amount_paise=amount,
                narration=template.format(ref=ref, q=f"Q{rng.randrange(1, 5)}"),
                utr_field=None,
                role="orphan",
                settlement_id=None,
                unmatchable_reason=f"orphan_{kind}",
                bank_format=rng.choice(fmts),
            )
        )

    world.note(PATTERN, orphan_credits=n)
