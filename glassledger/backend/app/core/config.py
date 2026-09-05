"""Every tunable number in the system, in one file, with its justification.

If a threshold is buried in the matcher, nobody can audit it. If it lives here,
the diff that loosens a tolerance is visible in review and the number that
produced a published metric is recoverable from git history.

Two knobs are load-bearing and worth reading before anything else:

``AUTO_CONFIRM_THRESHOLD``
    Not hand-picked. ``matching.threshold.choose_threshold`` selects the lowest
    confidence whose *calibrated* precision on the training split clears
    ``TARGET_PRECISION``, and this constant is only the fallback for when no
    calibrated model is loaded.

``MATERIALITY_PAISE``
    A hard rule that sits outside every model. Above it, nothing auto-confirms
    regardless of what any tier reports. See ``matching.rules``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RUPEE = 100


@dataclass(frozen=True)
class FeeModel:
    """The merchant's *assumption* about gateway pricing.

    Real MDR is per-instrument and negotiated; the merchant's books encode one
    blended guess. The gap between this assumption and what actually got
    deducted is break pattern #4, so this object exists to be slightly wrong on
    purpose -- it is the merchant's view, not the gateway's truth.
    """

    mdr_bps: int = 200          # 2.00% blended assumption
    gst_on_fee_bps: int = 1800  # 18% GST charged on the fee, not on the txn
    tolerance_bps: int = 40     # how far off the assumption may be before it is a break


@dataclass(frozen=True)
class LegConfig:
    """Blocking + tolerance parameters for one reconciliation leg.

    Sign convention, stated once so it is not re-derived per feature:
    ``date_delta = right_representative_date - left_representative_date``.
    Left is the earlier-in-life side of the leg (the settlement for
    gateway->bank, the invoice for books->gateway), so a healthy delta is
    positive and a negative one means money arrived before it was released --
    nearly always a data error rather than a real event.

    Blocking is the only reason this is tractable: without a window, matching N
    settlements against M bank lines is N*M edges and the subset search is
    2**N. With a few-day window and an amount band, both collapse to something
    linear-ish in practice. The cost of blocking is recall on genuinely late
    settlements, which is why ``date_window_after`` is generous while
    ``date_window_before`` stays tight.
    """

    name: str
    # Candidate window on ``date_delta``, in days: [-before, +after].
    date_window_before: int = 1
    date_window_after: int = 8
    # Amount band: an edge is only a candidate if the relative delta is inside
    # this, or the absolute delta is under the floor (protects small amounts,
    # where a 5% band is a handful of paise).
    amount_rel_tolerance: float = 0.05
    amount_abs_tolerance_paise: int = 50 * RUPEE
    # Subset search bounds for N:1 / 1:N.
    max_subset_size: int = 8
    max_window_candidates: int = 40
    subset_tolerance_paise: int = 5 * RUPEE
    # Relative slack on a subset target, added to the absolute figure above.
    # Needed on the books leg, where the explainable gap (fees, withholding, FX)
    # scales with the batch: a flat rupee tolerance that works for a Rs 5,000
    # batch rejects every correct group at Rs 500,000.
    subset_rel_tolerance: float = 0.0
    # Settlement-cycle prior: P(date_delta == d) for a healthy relationship.
    cycle_prior: dict[int, float] = field(
        default_factory=lambda: {
            0: 0.02, 1: 0.10, 2: 0.55, 3: 0.18, 4: 0.07,
            5: 0.03, 6: 0.02, 7: 0.02, 8: 0.01,
        }
    )
    #: Basis points of the larger amount that a residual may reach and still be
    #: considered "explainable by fees/tax/withholding" rather than a break.
    explainable_bps: int = 20
    #: Absolute floor on that band, for amounts small enough that a basis-point
    #: figure rounds to nothing.
    #:
    #: This started as ``amount_abs_tolerance_paise // 10`` inside the feature
    #: extractor, and that shared default caused the one silent-wrong match in the
    #: eval: on the gateway->bank leg it worked out to Rs 10, so a 95-paise residual
    #: between a *missing* settlement and half of a different settlement's split
    #: scored ``within_fee_band = 1``, and the model confirmed it. Bank credits do
    #: not have an explainable band -- they either equal the payout or they are a
    #: different payout -- so this leg gets a floor of zero and the books leg keeps
    #: a real one. A tolerance shared between legs with opposite requirements is a
    #: tolerance that is wrong on one of them.
    explainable_floor_paise: int = 0


GATEWAY_BANK = LegConfig(
    name="gateway_bank",
    # T+2 is the default Razorpay cycle; a bank holiday pushes it to T+4, and a
    # stuck NEFT can land at T+6. Eight days of slack costs candidate edges but
    # buys the timing-drift pattern.
    date_window_before=1,
    date_window_after=9,
    # Gateway net -> bank credit should be *exact*. The tolerance here exists for
    # split settlements and partial credits, not for fees.
    amount_rel_tolerance=0.03,
    amount_abs_tolerance_paise=100 * RUPEE,
    max_subset_size=6,
    max_window_candidates=48,
    subset_tolerance_paise=1,  # 1 paise: bank credits do not "approximately" match
    subset_rel_tolerance=0.0,
    explainable_bps=0,
    explainable_floor_paise=0,   # exact, or it is not this payout
)

GATEWAY_BOOKS = LegConfig(
    name="gateway_books",
    # Left is the invoice, right is the settlement. Receivables are booked on the
    # capture date and settle T+2 business days later; credit notes and accruals
    # are booked on the payout date itself, at delta 0. So the true distribution
    # is bimodal and narrow -- 0, and 2 to 6. The first version of this used
    # [-10, +4] on a guess and produced pools of 75 entries; see expectations.py
    # for what that cost and how the window is now derived from tier 1 instead.
    date_window_before=1,
    date_window_after=8,
    # Fees, GST-on-fee, withholding and FX all live in this gap, so the band is
    # wide and the feature vector -- not the window -- does the discriminating.
    amount_rel_tolerance=0.14,
    amount_abs_tolerance_paise=500 * RUPEE,
    # Batches run to ~11 payments, of which the unkeyed share is typically under
    # half. 8 covers the realistic tail; raising it multiplies search cost for
    # subsets nobody would accept anyway.
    max_subset_size=8,
    max_window_candidates=40,
    subset_tolerance_paise=30 * RUPEE,
    subset_rel_tolerance=0.0165,
    explainable_floor_paise=25 * RUPEE,
    cycle_prior={
        -2: 0.01, -1: 0.02, 0: 0.22, 1: 0.06, 2: 0.19, 3: 0.16, 4: 0.13,
        5: 0.09, 6: 0.06, 7: 0.03, 8: 0.02,
    },
    explainable_bps=180,
)

LEGS: dict[str, LegConfig] = {
    "gateway_bank": GATEWAY_BANK,
    "gateway_books": GATEWAY_BOOKS,
}

#: Sub-configuration for the books leg's per-component assignment step. Same leg
#: name (so candidates are attributed to the right leg) but a far tighter amount
#: band, because here the engine is comparing one expected invoice amount against
#: one book entry rather than a group sum against a batch total. The only variance
#: it has to absorb is FX drift on an international invoice and per-payment fee
#: rounding -- a 14% band would match almost anything.
GATEWAY_BOOKS_COMPONENT = LegConfig(
    name="gateway_books",
    date_window_before=1,
    date_window_after=8,
    amount_rel_tolerance=0.022,
    amount_abs_tolerance_paise=20 * RUPEE,
    max_subset_size=4,
    max_window_candidates=40,
    subset_tolerance_paise=1,
    cycle_prior=GATEWAY_BOOKS.cycle_prior,
    explainable_bps=220,
    explainable_floor_paise=20 * RUPEE,
)


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------

#: Precision we are willing to promise on auto-confirmed matches. 0.995 means we
#: accept roughly one mispost per 200 auto-confirmations, and everything else
#: goes to a human. Chosen because a mispost costs a finance team an hour of
#: investigation while an exception costs them thirty seconds of clicking.
TARGET_PRECISION = 0.995

#: Fallback only. The real value is derived from the calibration curve.
AUTO_CONFIRM_THRESHOLD = 0.90

#: The band where an LLM is allowed to have an opinion. Below it the evidence is
#: too thin for anyone; above it the deterministic tiers already agree and
#: paying for a model call would add latency and a hallucination surface for
#: nothing.
TIER3_BAND = (0.40, 0.75)

#: Hard rule, outside every model: above this amount a match is *proposed*,
#: never *confirmed*, no matter what confidence any tier reports.
#: Rs 50,000.
MATERIALITY_PAISE = 50_000 * RUPEE

#: Anything below this is not worth a human's attention even when ambiguous;
#: it is written off to a suspense account with an event recorded. Rs 10.
IMMATERIAL_PAISE = 10 * RUPEE


# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Small on purpose.

    A few hundred trees on a thousand rows overfits, and the calibration curve
    then lies in exactly the direction that matters (over-confident on the hard
    tail). Depth 3 with a real isotonic calibration split beats depth 12 with an
    optimistic in-sample estimate for this job.
    """

    max_depth: int = 3
    max_iter: int = 120
    learning_rate: float = 0.08
    l2_regularization: float = 1.0
    min_samples_leaf: int = 15
    calibration_method: str = "isotonic"
    #: Slice sizes for the three-way split. The threshold slice is separate from
    #: the calibration slice on purpose: choosing a decision threshold on data the
    #: calibrator already fitted biases it optimistically, and it biases it in the
    #: high-confidence region the auto-confirm rule lives in.
    calibration_fraction: float = 0.25
    threshold_fraction: float = 0.20
    random_state: int = 17


MODEL = ModelConfig()

#: Feature order is frozen and versioned. A model artefact records this list; if
#: the code's list ever drifts from the artefact's, loading raises instead of
#: silently scoring garbage because column 7 changed meaning.
FEATURE_NAMES: tuple[str, ...] = (
    "amount_rel_delta",
    "amount_abs_delta_log",
    "amount_exact",
    "within_fee_band",
    "date_delta_days",
    "date_delta_abs",
    "cycle_prior_logit",
    "utr_exact",
    "utr_similarity",
    "utr_in_narration",
    "narration_cosine",
    "currency_match",
    "subset_size",
    "is_subset",
    "left_ambiguity",
    "right_ambiguity",
    "amount_uniqueness",
    "left_amount_log",
)

FEATURE_SCHEMA_VERSION = "fs-1"
