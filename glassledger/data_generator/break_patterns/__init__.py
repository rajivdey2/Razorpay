"""Break-pattern registry.

One module per pattern from the problem statement's table. Each exposes:

    PATTERN : str        the label that flows into ground truth and the metrics
    PHASE   : str        when it may run (see below)
    apply(world, ctx)    mutates the world, calls world.note(...)

Phases exist because these mutations are not commutative. Refund netting changes
a settlement's *net* amount, so it has to happen before the bank credit that
mirrors that net is created; splitting a credit into three has to happen after.
Running them in registration order without phases produced a dataset where 4% of
"clean" settlements silently disagreed with their own bank line -- a generator
bug that would have shown up as a matcher accuracy ceiling nobody could explain.

    structural  -> group payments into settlement batches
    pre_bank    -> change what the gateway will pay out
    post_bank   -> change what the bank feed looks like
    books       -> change what the merchant's books expect
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from . import (
    batch_settlement,
    duplicate_bank_entry,
    fee_tax_drift,
    fx_mismatch,
    missing_settlement,
    narration_noise,
    orphan_bank_credit,
    refund_netting,
    split_settlement,
    tds_mismatch,
    timing_drift,
)


@dataclass
class Ctx:
    """Everything a pattern is allowed to depend on.

    Passing the rng explicitly (rather than using module-level ``random``) is
    what makes a dataset reproducible from its seed alone. A single call to the
    global rng anywhere in this package would make every published metric
    unreproducible, so patterns receive the generator and never import
    ``random`` themselves.
    """

    rng: random.Random
    holidays: frozenset[date]
    rate: float
    settings: dict


#: Registration order within a phase is the execution order. Kept explicit
#: rather than discovered by directory scan, because "which pattern ran first"
#: is a reproducibility-relevant fact and should be diffable.
PHASES: dict[str, list] = {
    "structural": [batch_settlement],
    "pre_bank": [refund_netting, fee_tax_drift, fx_mismatch],
    "post_bank": [
        split_settlement,
        duplicate_bank_entry,
        timing_drift,
        narration_noise,
        orphan_bank_credit,
        missing_settlement,
    ],
    "books": [tds_mismatch],
}

MODULES = [m for mods in PHASES.values() for m in mods]

__all__ = ["Ctx", "PHASES", "MODULES"]
