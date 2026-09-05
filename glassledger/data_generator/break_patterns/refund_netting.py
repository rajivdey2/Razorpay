"""#3 Refund netting: a settlement batch nets out refunds issued since the last cycle.

Modelled as negative amounts rather than a special case. Because
``NormalizedTxn.amount_paise`` is signed, a refund inside a batch is just a
negative term in the subset that has to sum to the settlement -- the subset-sum
search needs no refund-specific branch, which is one fewer place for a sign error
to hide.

The nasty part for a matcher is that a netted batch's payout can be far smaller
than the sum of its invoices, so an amount-band blocking window that is tight
enough to be useful on clean data will *exclude the correct match here*. That
tension is real and it is why the gateway<->books leg runs a wider band than the
gateway<->bank leg.
"""

from __future__ import annotations

from ..world import Refund, World

PATTERN = "refund_netting"
PHASE = "pre_bank"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    n = 0
    for s in world.settlements_in_order():
        if rng.random() >= ctx.rate:
            continue
        # Refund an earlier payment, not one in this batch: that is the realistic
        # case (a customer returns something bought last week) and it means the
        # refund's own invoice is not in the same window, which is what makes the
        # books-side subset genuinely hard.
        earlier = [
            p for p in world.payments.values()
            if p.settlement_id and p.settlement_id != s.settlement_id
            and p.captured_on < s.created_on
        ]
        if not earlier:
            continue
        for _ in range(rng.choice([1, 1, 2])):
            victim = rng.choice(earlier)
            # Partial refunds are common; full refunds happen. Never refund more
            # than was captured.
            frac = rng.choice([25, 50, 100, 100])
            amount = max(1, victim.gross_paise * frac // 100)
            rid = "rfnd_" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=14))
            world.refunds[rid] = Refund(
                refund_id=rid,
                payment_id=victim.payment_id,
                amount_paise=amount,
                created_on=s.created_on,
                settlement_id=s.settlement_id,
            )
            s.refund_ids.append(rid)
            n += 1
        s.patterns.add(PATTERN)

    world.note(PATTERN, refunds_injected=n)
