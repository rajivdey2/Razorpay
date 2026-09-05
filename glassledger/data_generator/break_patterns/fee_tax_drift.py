"""#4 Fee/tax drift: MDR + GST-on-fee rounding differs from the merchant's assumption.

The books apply one blended MDR to every payment. The gateway applies a
per-instrument rate, rounds each payment's fee independently, and occasionally
bills something the merchant's model has no field for -- a payment-page
surcharge, a promotional rebate, an international assessment fee.

This module injects that extra term. It deliberately does *not* touch the bank
side: the bank credit always equals the gateway's net, so fee drift is invisible
on the gateway<->bank leg and only shows up gateway<->books. A matcher that
treats both legs with the same tolerance therefore either misses these or accepts
garbage on the bank leg, which is the argument for per-leg configuration rather
than one global tolerance.
"""

from __future__ import annotations

from app.core.money import apply_bps

from ..world import GST_ON_FEE_BPS, World

PATTERN = "fee_tax_drift"
PHASE = "pre_bank"


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    n = 0
    for s in world.settlements_in_order():
        if rng.random() >= ctx.rate:
            continue
        gross = s.gross_paise(world)
        if gross <= 0:
            continue
        # Between -15 and +45 bps of unmodelled cost. Asymmetric because
        # surprises in payments pricing are far more often charges than credits.
        drift_bps = rng.choice([-15, -8, 12, 18, 25, 33, 45])
        delta = apply_bps(gross, abs(drift_bps))
        delta = -delta if drift_bps < 0 else delta
        s.fees_paise += delta
        # GST rides on the fee, so it moves too -- and it is recomputed from the
        # new fee rather than adjusted proportionally, because that is what the
        # gateway's invoice does and the 1-2 paise difference between those two
        # arithmetics is exactly the kind of thing that makes an exact-match
        # reconciliation report a break.
        s.tax_paise = apply_bps(s.fees_paise, GST_ON_FEE_BPS)
        s.patterns.add(PATTERN)
        n += 1

    world.note(PATTERN, settlements_affected=n)
