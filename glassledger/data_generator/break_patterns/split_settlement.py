"""#2 Split settlement: 1 payout -> N bank credits.

Partial holds, risk review releases, and phased payouts mean one settlement
arrives as two or three separate credits, days apart, that only make sense once
you add them together.

Two details make this hard rather than tedious:

* The parts share the settlement's UTR (the bank appends a sequence suffix), so
  a UTR-equality matcher finds *three* candidates for one settlement and has to
  decide they are all correct rather than picking one.
* The split is uneven. Even splits are easy to spot; a 70/20/10 split looks like
  one legitimate credit plus two unrelated small ones.

The parts sum to the settlement net exactly, with the remainder pushed onto the
last part -- no silent rounding loss, which the money-conservation property test
in ``tests/property`` checks across the whole dataset.
"""

from __future__ import annotations

from datetime import timedelta

from ..world import BankLine, World

PATTERN = "split_settlement"
PHASE = "post_bank"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    n = 0
    for s in world.settlements_in_order():
        credits = world.credits_for(s.settlement_id)
        if len(credits) != 1 or rng.random() >= ctx.rate:
            continue
        original = credits[0]
        if original.amount_paise < 1000:  # not worth splitting Rs 10
            continue

        weights = rng.choice([(70, 30), (55, 45), (60, 25, 15), (80, 12, 8), (40, 35, 25)])
        total = original.amount_paise
        parts: list[int] = []
        for w in weights[:-1]:
            parts.append(total * w // 100)
        parts.append(total - sum(parts))  # remainder absorbs rounding, exactly
        assert sum(parts) == total

        world.bank_lines.remove(original)
        day = original.value_date
        for idx, amt in enumerate(parts):
            day = day + timedelta(days=rng.choice([0, 1, 1, 2]))
            world.bank_lines.append(
                BankLine(
                    bank_ref=f"{original.bank_ref}-P{idx + 1}",
                    value_date=day,
                    posted_date=day,
                    amount_paise=amt,
                    narration=f"RAZORPAY SETTLEMENT PART {idx + 1}/{len(parts)} {s.utr}/{idx + 1}",
                    utr_field=f"{s.utr}/{idx + 1}",
                    role="split_part",
                    settlement_id=s.settlement_id,
                    bank_format=original.bank_format,
                )
            )
        s.patterns.add(PATTERN)
        n += 1

    world.note(PATTERN, settlements_split=n)
