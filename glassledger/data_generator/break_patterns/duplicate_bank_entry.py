"""#6 Duplicate bank entry: reversal + re-settlement shows twice in the feed.

The realistic shape, not a naive copy-paste of one row:

    T+2  credit  Rs X   original settlement attempt
    T+3  debit  -Rs X   the bank returns it (beneficiary mismatch, IMPS timeout)
    T+4  credit  Rs X   re-settlement, same UTR, "RE-SETL" in the narration

Three lines, one real payout. The correct answer is unambiguous -- the settlement
matches the *third* line -- and that is what makes it a good test. A copy of one
row would be genuinely ambiguous, so grading a matcher on it would be unfair; here
there is a right answer and a naive matcher reliably gets it wrong, because the
stale credit sits closer to the expected T+2 date and wins on the date feature.

The stale credit and its reversal go into the answer key as a wash pair that
*should* end up on the exception list. A system that matches one of them and
calls the day done has silently absorbed a duplicate payout into its books.
"""

from __future__ import annotations

from datetime import timedelta

from ..world import BankLine, World

PATTERN = "duplicate_bank_entry"
PHASE = "post_bank"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    n = 0
    for s in world.settlements_in_order():
        credits = world.credits_for(s.settlement_id)
        if len(credits) != 1 or rng.random() >= ctx.rate:
            continue
        real = credits[0]
        amt = real.amount_paise
        d0 = real.value_date

        world.bank_lines.append(
            BankLine(
                bank_ref=f"{real.bank_ref}-STALE",
                value_date=d0,
                posted_date=d0,
                amount_paise=amt,
                narration=f"RAZORPAY SETTLEMENT {s.utr}",
                utr_field=s.utr,
                role="stale_credit",
                settlement_id=None,
                unmatchable_reason="reversal_wash_pair",
                bank_format=real.bank_format,
            )
        )
        world.bank_lines.append(
            BankLine(
                bank_ref=f"{real.bank_ref}-REV",
                value_date=d0 + timedelta(days=1),
                posted_date=d0 + timedelta(days=1),
                amount_paise=-amt,
                narration=f"REV/RETURN NEFT {s.utr} BENEFICIARY",
                utr_field=s.utr,
                role="reversal_debit",
                settlement_id=None,
                unmatchable_reason="reversal_wash_pair",
                bank_format=real.bank_format,
            )
        )
        # The payout that actually stuck lands two days late.
        real.value_date = d0 + timedelta(days=2)
        real.posted_date = real.value_date
        real.narration = f"RE-SETL RAZORPAY {s.utr}"
        real.bank_ref = f"{real.bank_ref}-RE"

        s.patterns.add(PATTERN)
        n += 1

    world.note(PATTERN, settlements_affected=n, extra_bank_lines=n * 2)
