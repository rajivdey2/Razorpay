"""Types the matching engine produces. Deliberately plain and serialisable.

A ``Candidate`` is a *hypothesis*: this group of left transactions corresponds to
this group of right transactions, produced by this algorithm, with this feature
vector and this calibrated probability. A ``Decision`` is what the policy layer
did with that hypothesis. An ``Exception`` is what happens when no hypothesis
survived.

The split matters for the audit trail. Every hypothesis the engine seriously
entertained is recorded, not just the winner, so six months later "why did it pick
this one" has an answer that includes the runners-up and their scores. A system
that only logs its conclusions cannot explain them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """One hypothesised correspondence between two groups of transactions."""

    leg: str
    left_ids: tuple[str, ...]
    right_ids: tuple[str, ...]
    tier: int
    algorithm: str
    features: dict[str, float] = field(default_factory=dict)
    #: Calibrated P(correct). Tier 1 rules assert 1.0 because they are
    #: identity-based, not probabilistic -- and the eval measures whether that
    #: assertion holds rather than taking it on faith.
    score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    left_amount_paise: int = 0
    right_amount_paise: int = 0

    @property
    def cardinality(self) -> str:
        n, m = len(self.left_ids), len(self.right_ids)
        if n == 1 and m == 1:
            return "1:1"
        if m == 1:
            return "N:1"
        if n == 1:
            return "1:N"
        return "N:M"

    @property
    def match_key(self) -> str:
        left = ",".join(sorted(self.left_ids))
        right = ",".join(sorted(self.right_ids))
        return f"{self.leg}|{left}|{right}"

    @property
    def residual_paise(self) -> int:
        return self.right_amount_paise - self.left_amount_paise

    @property
    def exposure_paise(self) -> int:
        """The money this decision moves -- the larger side, not the difference.

        The materiality gate keys on this. Using the residual instead would let a
        Rs 8,00,000 match through the gate whenever the two sides happened to
        agree closely, which is exactly backwards: agreement is not authority.
        """
        return max(abs(self.left_amount_paise), abs(self.right_amount_paise))

    def all_ids(self) -> tuple[str, ...]:
        return (*self.left_ids, *self.right_ids)

    def pair_keys(self) -> set[tuple[str, str]]:
        return {(l, r) for l in self.left_ids for r in self.right_ids}

    def to_json(self) -> dict:
        return {
            "leg": self.leg,
            "left_ids": list(self.left_ids),
            "right_ids": list(self.right_ids),
            "tier": self.tier,
            "algorithm": self.algorithm,
            "cardinality": self.cardinality,
            "score": round(self.score, 6),
            "features": {k: round(v, 6) for k, v in sorted(self.features.items())},
            "evidence": self.evidence,
            "left_amount_paise": self.left_amount_paise,
            "right_amount_paise": self.right_amount_paise,
            "residual_paise": self.residual_paise,
        }


@dataclass(frozen=True)
class Decision:
    """What the policy layer did with a candidate, and why."""

    candidate: Candidate
    action: str            # auto_confirm | propose | reject
    reason: str
    gate: str | None = None          # the hard rule that intervened, if any
    runners_up: tuple[Candidate, ...] = ()

    def to_json(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "gate": self.gate,
            "candidate": self.candidate.to_json(),
            "runners_up": [c.to_json() for c in self.runners_up],
        }


@dataclass(frozen=True)
class Exception_:
    """An unresolved item, with enough context for a human to act in one screen."""

    exception_id: str
    leg: str
    txn_ids: tuple[str, ...]
    category: str
    max_confidence: float
    suggested_action: str
    amount_paise: int
    age_days: int
    evidence: dict[str, Any] = field(default_factory=dict)
    near_misses: tuple[Candidate, ...] = ()

    @property
    def priority(self) -> float:
        """Ranking score: money at risk, amplified by how long it has sat.

        Amount-weighted rather than FIFO because a Rs 4,00,000 orphan credit that
        appeared this morning matters more than a Rs 300 rounding break from last
        week, and a queue sorted by age buries it. The log keeps a single very
        large item from monopolising the top of the list.
        """
        import math

        return math.log1p(abs(self.amount_paise)) * (1.0 + min(self.age_days, 30) / 10.0)

    def to_json(self) -> dict:
        return {
            "exception_id": self.exception_id,
            "leg": self.leg,
            "txn_ids": list(self.txn_ids),
            "category": self.category,
            "max_confidence": round(self.max_confidence, 6),
            "suggested_action": self.suggested_action,
            "amount_paise": self.amount_paise,
            "age_days": self.age_days,
            "priority": round(self.priority, 4),
            "evidence": self.evidence,
            "near_misses": [c.to_json() for c in self.near_misses],
        }
