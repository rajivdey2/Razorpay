"""#8 Missing settlement: gateway says ``processed``, the bank has not credited.

The gateway record exists with a UTR and a status of ``processed``; there is no
bank line at all. Either the NEFT window has not closed yet, or the payout is
genuinely stuck at the sponsor bank.

This is the mirror image of the orphan credit and it tests the same honesty
property from the other direction: the correct behaviour is an exception saying
"gateway claims this paid out, bank disagrees, escalate", and the tempting wrong
behaviour is to match it to whichever unclaimed bank credit is nearest.

It also carries an operational subtlety the workbench surfaces: a missing
settlement two days old is *probably fine* (the window has not closed), while one
nine days old is a support ticket. Same break pattern, entirely different
urgency -- which is why the exception list is ranked by amount-weighted age
rather than just listed.
"""

from __future__ import annotations

from ..world import World

PATTERN = "missing_settlement"
PHASE = "post_bank"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    removed = 0
    for s in world.settlements_in_order():
        credits = world.credits_for(s.settlement_id)
        if not credits or rng.random() >= ctx.rate:
            continue
        # Only drop whole settlements. Dropping one leg of a split would create a
        # partially-credited payout, which is a different (and genuinely
        # ambiguous) situation -- worth building, but it should not be smuggled
        # in under this pattern's label where the metrics would attribute it
        # wrongly.
        if len(credits) > 1:
            continue
        for c in credits:
            world.bank_lines.remove(c)
        s.patterns.add(PATTERN)
        s.status = "processed"  # the gateway still insists it paid
        removed += 1

    world.note(PATTERN, settlements_without_credit=removed)
