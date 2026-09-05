"""#1 Batch settlement: N payments -> 1 payout.

Not an edge case -- it is how Razorpay settles. Every settlement in this dataset
bundles between 1 and ~12 payments, and ``rate`` controls what share are
multi-payment. This module owns the batching itself rather than merely perturbing
it, because the batching *is* the pattern: on the gateway<->books leg every
multi-payment settlement is an N:1 subset-sum problem by construction, so a
dataset without it would never exercise the part of the engine that matters.
"""

from __future__ import annotations

from ..world import Settlement, World, business_days_after, recompute_settlement_fees

PATTERN = "batch_settlement"
PHASE = "structural"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    unassigned = sorted(world.payments.values(), key=lambda p: (p.captured_on, p.payment_id))

    # Payments settle by capture date: one settlement cycle per business day.
    by_day: dict = {}
    for p in unassigned:
        by_day.setdefault(p.captured_on, []).append(p)

    n_multi = 0
    for day in sorted(by_day):
        pool = by_day[day]
        rng.shuffle(pool)
        i = 0
        while i < len(pool):
            if rng.random() < ctx.rate:
                # Heavy-tailed batch sizes: most batches are small, a few are
                # large. A uniform size distribution makes subset-sum look
                # easier than it is, because the hard instances are precisely
                # the big windows.
                size = min(len(pool) - i, rng.choice([2, 2, 3, 3, 4, 5, 6, 8, 11]))
            else:
                size = 1
            batch = pool[i : i + size]
            i += size

            sid = "setl_" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=14))
            utr = _utr(rng, day)
            created = business_days_after(day, 2, ctx.holidays)
            s = Settlement(settlement_id=sid, utr=utr, created_on=created)
            for p in batch:
                p.settlement_id = sid
                s.payment_ids.append(p.payment_id)
            if size > 1:
                s.patterns.add(PATTERN)
                n_multi += 1
            recompute_settlement_fees(world, s)
            world.settlements[sid] = s

    world.note(
        PATTERN,
        settlements=len(world.settlements),
        multi_payment=n_multi,
        payments=len(world.payments),
    )


def _utr(rng, d) -> str:
    """A 16-character UTR in the shape Indian bank feeds carry.

    Format is deliberately realistic-but-synthetic: a four-letter bank prefix,
    the date, and a sequence. Realistic length matters because the truncation in
    break pattern #10 chops a fixed number of characters, and the Levenshtein
    feature's discriminating power depends on how much of a 16-char string
    survives.
    """
    bank = rng.choice(["HDFC", "ICIC", "UTIB", "SBIN", "KKBK"])
    return f"{bank}{d.strftime('%y%m%d')}{rng.randrange(10**6):06d}"
