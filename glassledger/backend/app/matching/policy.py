"""Decision policy: what gets auto-confirmed, what goes to a human, what dies.

Four rules, evaluated in order on every candidate:

1. **Immateriality.** Below Rs 10, a candidate that is not provable is written off
   to suspense with an event -- it is not worth a human's attention, and marking
   it as an exception would engineer a queue full of Rs 2 items. Trivial, but it
   exists to keep the exception list meaningful at scale.

2. **Materiality gate.** Above Rs 50,000, no *inferred* match auto-confirms.
   Period. The rule sits outside every model and is not a prompt instruction,
   which is the whole point: a model can be miscalibrated, retrained, or have a
   bad day and the gate still holds. Prompts get rewritten; this does not.

   The gate binds on tier >= 2 -- the tiers whose conclusions rest on a
   probability. A tier 1 match does not, and the distinction is not a loophole,
   it is the difference between two kinds of claim. "These two lines carry the
   same 16-character UTR and identical amounts to the paise" is a statement about
   identity that is either true or false and can be re-checked by anyone in one
   second. "The model scores this 0.93" is a statement about a distribution. It is
   the second kind that needs a human above a material amount, because it is the
   second kind that can be quietly wrong at scale.

   The exemption is *reported*, not assumed: ``PolicyResult.stats`` counts
   tier 1 auto-confirmations above the gate, and the eval prints the figure and the
   money it represents, so the size of what this rule waves through is a number
   on the page rather than a footnote. If a reviewer disagrees with the
   reasoning, ``MATERIALITY_APPLIES_TO_TIER1 = True`` restores the strict reading
   in one line, and the eval numbers move accordingly.

3. **Uniqueness.** If a candidate's partner is claimed by a *better* candidate,
   this one cannot be right either -- no decision is made about a transaction
   that already has a better-supported explanation. This is how overlapping
   hypotheses from the assignment and the subset solver resolve: only the
   mutually-consistent best set survives, and the losers are visible as
   ``runners_up``.

4. **Threshold.** The derived auto-confirm threshold divides the rest into
   ``auto_confirm`` and ``propose``.

The layer is deliberately small and deliberately outside the model. It owns no
probability, no resemblance logic, and no judgement. It owns four rules. The first
two are legal constraints in any merchant's policy, the third is arithmetic, and
the fourth is the model's trained threshold -- and each one being separately
reproducible is what makes an audit trail an audit trail rather than a story.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from app.core.config import (
    AUTO_CONFIRM_THRESHOLD,
    IMMATERIAL_PAISE,
    MATERIALITY_PAISE,
    TIER3_BAND,
)
from .types import Candidate, Decision

#: Tier 1 rules are identity-based and declare confidence 1.0; they skip the
#: calibration model entirely, which also means a Tier-1 rule can *never* be
#: vetoed by a model quirk.
TIER1_ALGORITHMS = frozenset(
    {"exact_utr_and_amount", "exact_utr_split_sum", "order_key_join", "wash_pair_netting"}
)

#: Flip to True for the strict reading of the materiality rule: every match above
#: the threshold needs a human, including deterministic identity matches. Left
#: False because a UTR-and-amount identity match is re-verifiable by anyone in a
#: second and does not rest on a probability -- but the switch is here so the
#: choice is a configuration decision with a measurable cost rather than an
#: assumption buried in a branch.
MATERIALITY_APPLIES_TO_TIER1 = False


@dataclass
class PolicyResult:
    auto_confirmed: list[Decision] = field(default_factory=list)
    proposed: list[Decision] = field(default_factory=list)
    rejected: list[Decision] = field(default_factory=list)
    exempted_immaterial: list[Decision] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _gate_applies(c: Candidate) -> bool:
    """Does the materiality gate bind on this candidate?"""
    if c.exposure_paise < MATERIALITY_PAISE:
        return False
    if c.tier == 1 and not MATERIALITY_APPLIES_TO_TIER1:
        return False
    return True


def _is_above_materiality(c: Candidate) -> bool:
    return c.exposure_paise >= MATERIALITY_PAISE


@dataclass
class Policy:
    """One policy per leg, carrying only the leg's threshold and the global gate."""

    auto_confirm_threshold: float = AUTO_CONFIRM_THRESHOLD

    def decide_many(
        self,
        candidates: Iterable[Candidate],
        *,
        claimed_right: set[str] | None = None,
        claimed_left: set[str] | None = None,
    ) -> PolicyResult:
        claimed_right = claimed_right or set()
        claimed_left = claimed_left or set()
        out = PolicyResult()

        ranked = sorted(candidates, key=lambda c: (-c.score, len(c.left_ids), c.match_key))
        # Pass 1: find the strongest claim on each partner id, so pass 2 can see
        # whether a candidate is even the best explanation for the money it
        # touches.
        best_for_right: dict[str, float] = {}
        best_for_left: dict[str, float] = {}
        for c in ranked:
            for rid in c.right_ids:
                best_for_right[rid] = max(best_for_right.get(rid, -1.0), c.score)
            for lid in c.left_ids:
                best_for_left[lid] = max(best_for_left.get(lid, -1.0), c.score)

        for c in ranked:
            # Rule 1: immaterial write-off.
            if c.exposure_paise < IMMATERIAL_PAISE and c.tier > 1:
                d = Decision(
                    candidate=c,
                    action="reject",
                    reason="immaterial: below the Rs 10 attention floor; "
                           "no human review is worth this much",
                    gate="immateriality",
                )
                out.exempted_immaterial.append(d)
                continue

            # Rule 3: uniqueness. A candidate that is not the best-supported
            # explanation for one of its own partners cannot be confirmed.
            if any(best_for_right.get(rid, 0.0) > c.score + 1e-9 for rid in c.right_ids):
                out.rejected.append(
                    Decision(
                        candidate=c,
                        action="reject",
                        reason="a better-supported candidate claims one of its partners",
                        gate="uniqueness",
                    )
                )
                continue

            # Rule 2: the gate that outranks everything else, including the
            # model. Keyed on *exposure* -- the money the decision moves -- not on
            # the residual, because agreement between two numbers is not authority
            # over them.
            if _gate_applies(c):
                out.proposed.append(
                    Decision(
                        candidate=c,
                        action="propose",
                        reason=(
                            f"exposure {c.exposure_paise} paise is above the Rs {MATERIALITY_PAISE / 100:,.0f} "
                            "materiality gate; human approval required regardless of confidence"
                        ),
                        gate="materiality",
                        runners_up=tuple(
                            other
                            for other in ranked
                            if other.match_key != c.match_key
                            and (set(other.right_ids) & set(c.right_ids))
                        ),
                    )
                )
                continue

            # Rule 4: derived threshold.
            if c.score >= self.auto_confirm_threshold:
                out.auto_confirmed.append(
                    Decision(
                        candidate=c,
                        action="auto_confirm",
                        reason=(
                            f"score {c.score:.4f} clears threshold "
                            f"{self.auto_confirm_threshold:.4f}"
                        ),
                        gate="threshold",
                    )
                )
                claimed_right.update(c.right_ids)
                claimed_left.update(c.left_ids)
            else:
                out.proposed.append(
                    Decision(
                        candidate=c,
                        action="propose",
                        reason=(
                            f"score {c.score:.4f} below threshold "
                            f"{self.auto_confirm_threshold:.4f} but inside the Tier 3 band"
                            if (TIER3_BAND[0] <= c.score <= TIER3_BAND[1])
                            else f"score {c.score:.4f} below threshold "
                                 f"{self.auto_confirm_threshold:.4f}; evidence too thin "
                                 "for automation"
                        ),
                        gate="threshold",
                    )
                )

        out.stats = {
            "considered": len(ranked),
            "auto_confirmed": len(out.auto_confirmed),
            "proposed": len(out.proposed),
            "rejected": len(out.rejected),
            "immaterial": len(out.exempted_immaterial),
            # Transparency on the tier 1 exemption: how much money went through
            # above the materiality line without a human, and on how many matches.
            "tier1_above_materiality": sum(
                1 for d in out.auto_confirmed
                if d.candidate.tier == 1 and _is_above_materiality(d.candidate)
            ),
            "tier1_above_materiality_paise": sum(
                d.candidate.exposure_paise for d in out.auto_confirmed
                if d.candidate.tier == 1 and _is_above_materiality(d.candidate)
            ),
            "gated_by_materiality": sum(
                1 for d in out.proposed if d.gate == "materiality"
            ),
        }
        return out


def is_auto_confirmable(c: Candidate, *, threshold: float, tier: int = 4) -> bool:
    """Pure predicate, used by the exception builder and by Tier 3's guard.

    This is the function Tier 3 must call before *proposing anything*: if a
    candidate would be auto-confirmable absent the gate and is above the
    materiality line, Tier 3 must not confirm it either. The gate being callable
    by everyone is how it stays outside any single component.
    """
    if _is_above_materiality(c):
        return False
    if c.tier == 1:
        return True
    return c.score >= threshold
