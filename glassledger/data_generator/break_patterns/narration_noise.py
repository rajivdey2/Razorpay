"""#10 Narration noise: the bank truncates or merges the UTR into free text.

When a bank feed carries a clean UTR field, reconciliation is a dictionary
lookup. This pattern removes that luxury, which is the normal case for a lot of
statement exports:

* the dedicated reference column is empty
* the UTR is embedded in a longer narration string, sometimes truncated
* OCR-style / keying confusions appear: ``O`` <-> ``0``, ``I`` <-> ``1``,
  ``S`` <-> ``5``, ``B`` <-> ``8``
* the string gets split by a slash, or padded with the bank's own routing noise

The point is to force the matcher to earn the UTR signal through fuzzy string
similarity instead of equality -- and, more importantly, to make it *distrust*
that signal appropriately. A 12-of-16-character match on a truncated UTR is
strong evidence; a 6-of-16 match is nearly none, and a system that scores them
the same will confidently mispost.
"""

from __future__ import annotations

from ..world import World

PATTERN = "narration_noise"
PHASE = "post_bank"

CONFUSIONS = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S", "B": "8", "8": "B"}

TEMPLATES = [
    "NEFT-{utr}-RAZORPAY SOFTWARE PVT",
    "MB/{utr}/RAZORPAYSOFT/PAYOUT",
    "IMPS/P2A/{utr}/RAZORPAY/SETTL",
    "TRF FROM RAZORPAY REF {utr} CR",
    "ACH C- RAZORPAY-SETTLEMENT-{utr}",
    "BY TRANSFER-NEFT*{utr}*RZP",
]


def apply(world: World, ctx) -> None:
    rng = ctx.rng
    n_stripped = 0
    n_garbled = 0
    for b in world.bank_lines:
        if b.role not in ("settlement_credit", "split_part") or not b.utr_field:
            continue
        if rng.random() >= ctx.rate:
            continue

        utr = b.utr_field
        # Truncation: banks cut the narration field at 30-40 chars, and the UTR
        # is rarely at the front.
        if rng.random() < 0.55:
            keep = rng.choice([9, 10, 11, 12, 13])
            utr = utr[:keep]
        # Keying confusion on one character.
        if rng.random() < 0.40:
            idx = rng.randrange(len(utr))
            ch = utr[idx]
            if ch in CONFUSIONS:
                utr = utr[:idx] + CONFUSIONS[ch] + utr[idx + 1 :]
                n_garbled += 1

        b.narration = rng.choice(TEMPLATES).format(utr=utr)
        b.utr_field = None  # the structured column is empty; earn it from text
        n_stripped += 1
        if b.settlement_id:
            world.settlements[b.settlement_id].patterns.add(PATTERN)

    world.note(PATTERN, utr_field_stripped=n_stripped, characters_garbled=n_garbled)
