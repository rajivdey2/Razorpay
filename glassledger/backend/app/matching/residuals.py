"""Explaining why two sides of a candidate do not sum to the same number.

Every hypothetical group carries a residual -- the difference between what the
right side sums to and what the left side sums to. On the gateway->books leg the
residual is *expected* (fees, GST-on-fee, withholding, FX) and its size is the
only practical evidence of whether a group is right. This module turns that
residual into a claim about its cause, so the confidence model sees a structural
signal ("the gap is 2.1% of gross, which fees explain") rather than a bare
magnitude ("the gap is Rs 1,847").

Why it has to run before scoring rather than after
--------------------------------------------------
The residual-attribution test is a genuine feature of the *group*, and features
have to be in the vector when the model scores. Running it afterwards would be
using the answer to justify the answer: "this group is a good match" endorsed by
"its gap looks like fees", where the gap was computed from the group the model
already accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import LegConfig
from app.core.money import apply_bps
from app.core.schema import NormalizedTxn

#: GST is 18% of the fee. Regulatory detail -- kept as the system's single point
#: of knowledge about it, because nothing else in the engine should hard-code a
#: statutory rate (the generator parameterises its own and the eval compares
#: flows, not rates).
GST_ON_FEE_BPS = 1800


@dataclass
class ResidualExplanation:
    #: one of: exact | within_fee_band | consistent | unexplained
    verdict: str
    #: structured diagnostics for evidence and for human review
    detail: dict = field(default_factory=dict)


@dataclass
class ResidualAttribution:
    gst_on_fee_bps: int = GST_ON_FEE_BPS

    def explain(
        self,
        left: list[NormalizedTxn],
        right: list[NormalizedTxn],
        leg: LegConfig,
        gross_paise: int,
    ) -> ResidualExplanation:
        la = sum(t.amount_paise for t in left)
        ra = sum(t.amount_paise for t in right)
        residual = ra - la
        if residual == 0:
            return ResidualExplanation(
                verdict="exact",
                detail={"residual_paise": 0, "exact": True, "explainable": True},
            )

        # The relevant scale for fee-band judgement is gross, not net: fees are a
        # percentage of what the customer paid, so a Rs 25,000 gross carrying a
        # Rs 600 fee is normal even though 600 is 2.4% of the *net*.
        basis = gross_paise or max(abs(la), abs(ra))
        band_paise = max(
            leg.explainable_floor_paise,
            (basis * leg.explainable_bps) // 10_000,
        )
        # The band is directional. The gateway can withhold more than the books
        # expected, and the books can also expect more than the gateway paid (an
        # unmodelled rebate). A *negative* residual of the same magnitude is a
        # different question.
        if abs(residual) <= band_paise:
            return ResidualExplanation(
                verdict="within_fee_band",
                detail={
                    "residual_paise": residual,
                    "residual_pct_of_gross": round(residual / basis, 6),
                    "band_paise": band_paise,
                    "band_bps": leg.explainable_bps,
                    "explainable": True,
                    "note": "residual is in the range that fee, tax, withholding and "
                            "FX movements occupy",
                },
            )

        # A *specific* structural test, and the one that separates transactional
        # noise from real breaks: is the residual approximately a percentage of
        # gross? The TDS pattern injects a clean ~1% of gross; a mis-parse or a
        # genuinely split batch produces a residual with no such shape.
        pct_bps = round(residual / basis * 10_000)
        consistent = (
            250 <= abs(pct_bps) <= 1850
            and abs(residual * 10_000 - basis * pct_bps) <= max(basis // 1000, 100)
        )
        if consistent:
            verdict = "consistent_with_withholding"
            note = (
                f"residual is {pct_bps}bp (+/- rounding) of gross, consistent with a "
                "withholding accrual"
            )
        else:
            verdict = "unexplained"
            note = "residual does not match a fee-band or a percentage-of-gross shape"

        return ResidualExplanation(
            verdict=verdict,
            detail={
                "residual_paise": residual,
                "residual_pct_of_gross": round(pct_bps / 100, 4),
                "band_paise": band_paise,
                "band_bps": leg.explainable_bps,
                "explainable": verdict in ("within_fee_band", "consistent_with_withholding"),
                "note": note,
            },
        )
