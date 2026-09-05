"""#5 TDS-style mismatch: a statutory deduction the gateway never sees.

The merchant's books expect a withholding the gateway has no knowledge of, so the
books' expected receivable is systematically *lower* than the settlement -- by a
small percentage of gross that scales with the batch.

A deliberate scoping note, because getting compliance details wrong in front of a
fintech panel is worse than not claiming them: the rate here is a **configurable
generator parameter** (``--tds-bps``, default 100 = 1.00%) chosen because it
produces residuals in a realistic range. It is not an assertion about any
specific section of the Income Tax Act, and nothing in the matching engine
encodes a statutory rate. The engine's job is to notice that the residual is
*consistent with a percentage-of-gross deduction* rather than random noise, which
is a structural test and stays correct whatever the rate turns out to be.

That distinction is the interesting part. Fee drift and a TDS deduction both make
the books sum come up short. What separates them is the *shape* of the shortfall:
fee drift is bounded by a few tens of basis points and lands on the gateway's own
fee line, while a withholding is a clean percentage of gross with no gateway
counterpart. The features expose both so the model can learn the difference
instead of being told it.
"""

from __future__ import annotations

from app.core.money import apply_bps

from ..world import BookEntry, World

PATTERN = "tds_mismatch"
PHASE = "books"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    bps = int(ctx.settings.get("tds_bps", 100))
    n = 0
    for s in world.settlements_in_order():
        if rng.random() >= ctx.rate:
            continue
        gross = s.gross_paise(world)
        if gross <= 0:
            continue
        amount = apply_bps(gross, bps)
        if amount == 0:
            continue
        world.book_entries.append(
            BookEntry(
                entry_id=f"TDS-{s.settlement_id[-8:].upper()}",
                amount_paise=-amount,   # a deduction the books expect
                booked_on=s.created_on,
                memo=f"Withholding accrual @ {bps / 100:.2f}% on batch {s.settlement_id[-6:]}",
                settlement_id=s.settlement_id,
                kind="tds",
            )
        )
        s.patterns.add(PATTERN)
        n += 1

    world.note(PATTERN, entries_injected=n, rate_bps=bps)
