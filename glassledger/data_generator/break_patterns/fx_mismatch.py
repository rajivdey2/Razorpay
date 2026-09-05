"""#11 FX mismatch: invoiced in USD, settled against a different rate snapshot.

The merchant raises an invoice in USD and books the receivable at the invoice-day
rate. The gateway settles in INR at the rate that applied when the payment was
captured or converted. Two different snapshots of the same rate, days apart, on
the same underlying sale.

The residual is proportional to the invoice size, which is what makes it
dangerous: on a small invoice it looks like fee rounding, and on a large one it
looks like a missing line item. It is also the pattern most likely to interact
with the materiality gate, since international invoices skew large.
"""

from __future__ import annotations

from ..world import World, recompute_settlement_fees

PATTERN = "fx_mismatch"
PHASE = "pre_bank"

#: USD/INR, x1e6. Held as an int so no float rate ever multiplies a money value
#: (the multiply-then-integer-divide below keeps the whole computation exact).
BASE_USDINR_MICROS = 83_250_000


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    n = 0
    for p in sorted(world.payments.values(), key=lambda x: x.payment_id):
        if p.settlement_id is None or rng.random() >= ctx.rate:
            continue
        s = world.settlements[p.settlement_id]

        # Re-express this payment as a USD invoice. Round to whole dollars: a
        # human raising an international invoice types 1200, not 1187.43, and
        # that rounding is itself a source of residual.
        booking = BASE_USDINR_MICROS + rng.randrange(-900_000, 900_000)
        usd_cents = max(100, (p.gross_paise * 1_000_000) // booking)
        usd_cents = round(usd_cents / 100) * 100  # whole dollars

        # The rate moved between invoice and settlement: +/- ~1.2%.
        drift = rng.choice([-1_050_000, -620_000, -310_000, 380_000, 700_000, 1_100_000])
        settling = booking + drift

        p.invoice_currency = "USD"
        p.invoice_amount_minor = usd_cents
        p.booking_fx_micros = booking
        p.settlement_fx_micros = settling
        p.instrument = "intl_card"
        # What the gateway actually converts and settles.
        p.gross_paise = (usd_cents * settling) // 1_000_000

        recompute_settlement_fees(world, s)
        s.patterns.add(PATTERN)
        n += 1

    world.note(PATTERN, payments_affected=n)
