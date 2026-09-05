"""#9 Timing drift: T+2 lands on a bank holiday; statement date != value date.

Two separate effects, both injected here because they are the same underlying
cause -- calendars.

* The credit slips further than the settlement-cycle prior expects (+3 to +6
  days), typically because a holiday or a weekend intervened, or the NEFT sat in
  a queue.
* ``posted_date`` and ``value_date`` diverge. Statements are exported by posting
  date; the money is available on the value date. Picking the wrong one shifts
  every date feature by a day or two, which does not break a matcher outright --
  it just quietly degrades its confidence everywhere, which is worse, because it
  looks like the model is bad rather than the input being misread.

This is the pattern that punishes a fixed +/-2-day blocking window. The window
has to be generous on the late side and the *prior* has to do the discriminating,
which is why ``LegConfig.cycle_prior`` is a distribution rather than a bound.
"""

from __future__ import annotations

from datetime import timedelta

from ..world import World

PATTERN = "timing_drift"
PHASE = "post_bank"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    drifted = 0
    split_dates = 0
    for b in world.bank_lines:
        if b.role not in ("settlement_credit", "split_part"):
            continue
        if rng.random() < ctx.rate:
            b.value_date = b.value_date + timedelta(days=rng.choice([3, 3, 4, 4, 5, 6]))
            b.posted_date = b.value_date
            if b.settlement_id:
                world.settlements[b.settlement_id].patterns.add(PATTERN)
            drifted += 1
        # Independently, and more often: the statement's posting date is a day
        # or two off the value date.
        if rng.random() < ctx.rate * 2:
            b.posted_date = b.value_date + timedelta(days=rng.choice([1, 1, 2]))
            split_dates += 1

    world.note(PATTERN, credits_drifted=drifted, posted_vs_value_divergence=split_dates)
