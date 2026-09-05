"""Metrics. Defined here, once, before any of them were computed.

Pair-level, not group-level
---------------------------
A candidate that groups four invoices into a settlement and gets three of them
right is not 75% correct -- it is one wrong journal entry. So every match, however
it was produced, is flattened to the set of ``(left_id, right_id)`` pairs it
asserts, and precision/recall run over pairs. A 4-invoice group that gets one
member wrong scores 3 true positives, 1 false positive, and 1 false negative
against the member it should have included. That is the strict reading and it is
the one that corresponds to what a reviewer would actually find wrong.

Money-weighted as well as count-weighted
----------------------------------------
A CFO does not care that 96% of *lines* reconciled; they care that 96% of *rupees*
did. The two diverge exactly when the failures cluster in large transactions,
which is the case that matters, so both are reported and the money figure leads.

Exception honesty
-----------------
Two numbers, because "did it flag the right things" has two failure modes:

* **exception precision** -- of everything flagged, what fraction was genuinely
  unresolvable per ground truth. Low precision means crying wolf: a queue full of
  items a human closes with no action, which trains them to close everything.
* **exception recall** -- of everything genuinely unresolvable, what fraction got
  flagged. Low recall is the dangerous one: an unresolvable item that was *not*
  flagged has been silently absorbed, and the merchant's books are wrong with no
  indication.

Silent-wrong rate
-----------------
The number the whole system exists to keep at zero: auto-confirmed matches that
ground truth says are wrong. These are the ones a human never sees, so an error
here goes straight into the books. Reported separately from precision because
precision averages it away against the proposed-for-review population, where a
mistake is caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.schema import Dataset, GroundTruthLink


@dataclass
class PairMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_json(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class MoneyMetrics:
    reconciled_paise: int = 0
    total_paise: int = 0
    misreconciled_paise: int = 0

    @property
    def fraction(self) -> float:
        return self.reconciled_paise / self.total_paise if self.total_paise else 0.0

    def to_json(self) -> dict:
        return {
            "reconciled_paise": self.reconciled_paise,
            "total_paise": self.total_paise,
            "fraction_reconciled": round(self.fraction, 4),
            "misreconciled_paise": self.misreconciled_paise,
        }


@dataclass
class LegScore:
    leg: str
    all_matches: PairMetrics = field(default_factory=PairMetrics)
    auto_confirmed_only: PairMetrics = field(default_factory=PairMetrics)
    money: MoneyMetrics = field(default_factory=MoneyMetrics)
    by_pattern: dict[str, PairMetrics] = field(default_factory=dict)
    silent_wrong: int = 0
    silent_wrong_paise: int = 0

    def to_json(self) -> dict:
        return {
            "leg": self.leg,
            "all_proposed_or_confirmed": self.all_matches.to_json(),
            "auto_confirmed_only": self.auto_confirmed_only.to_json(),
            "money": self.money.to_json(),
            "silent_wrong_matches": self.silent_wrong,
            "silent_wrong_paise": self.silent_wrong_paise,
            "by_break_pattern": {
                k: v.to_json() for k, v in sorted(self.by_pattern.items())
            },
        }


def truth_pairs(ds: Dataset, leg: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for link in ds.links_for(leg):
        out |= link.pair_keys()
    return out


def truth_pattern_by_pair(ds: Dataset, leg: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for link in ds.links_for(leg):
        for pair in link.pair_keys():
            out[pair] = link.pattern
    return out


def score_leg(
    leg: str,
    ds: Dataset,
    confirmed_pairs: set[tuple[str, str]],
    proposed_pairs: set[tuple[str, str]],
    amount_by_id: dict[str, int],
) -> LegScore:
    """Grade one leg's output against the answer key."""
    truth = truth_pairs(ds, leg)
    pattern = truth_pattern_by_pair(ds, leg)
    score = LegScore(leg=leg)

    predicted = confirmed_pairs | proposed_pairs

    score.all_matches.tp = len(predicted & truth)
    score.all_matches.fp = len(predicted - truth)
    score.all_matches.fn = len(truth - predicted)

    score.auto_confirmed_only.tp = len(confirmed_pairs & truth)
    score.auto_confirmed_only.fp = len(confirmed_pairs - truth)
    score.auto_confirmed_only.fn = len(truth - confirmed_pairs)

    score.silent_wrong = len(confirmed_pairs - truth)
    score.silent_wrong_paise = sum(
        max(amount_by_id.get(l, 0), amount_by_id.get(r, 0))
        for l, r in (confirmed_pairs - truth)
    )

    # Money weighting keys on the *left* side of the leg, so each rupee is counted
    # once. On gateway->bank that is the settlement; on books->gateway it is the
    # invoice. Summing both sides would double-count every correct match and make
    # the denominator meaningless.
    left_ids_truth = {l for l, _ in truth}
    score.money.total_paise = sum(abs(amount_by_id.get(l, 0)) for l in left_ids_truth)
    correct_left = {l for (l, r) in (predicted & truth)}
    score.money.reconciled_paise = sum(abs(amount_by_id.get(l, 0)) for l in correct_left)
    wrong_left = {l for (l, r) in (predicted - truth)}
    score.money.misreconciled_paise = sum(abs(amount_by_id.get(l, 0)) for l in wrong_left)

    # Per-break-pattern slice. False positives are attributed to the pattern of the
    # *true* link that the mispredicted left id belongs to, so a wrong match on a
    # duplicate-bank-entry case is charged to that pattern rather than to "unknown".
    left_to_pattern: dict[str, str] = {}
    for (l, r), pat in pattern.items():
        left_to_pattern.setdefault(l, pat)

    for pair in truth:
        pat = pattern[pair]
        m = score.by_pattern.setdefault(pat, PairMetrics())
        if pair in predicted:
            m.tp += 1
        else:
            m.fn += 1
    for pair in predicted - truth:
        pat = left_to_pattern.get(pair[0], "not_in_truth")
        score.by_pattern.setdefault(pat, PairMetrics()).fp += 1

    return score


def exception_honesty(
    ds: Dataset,
    flagged_ids: set[str],
    *,
    leg: str | None = None,
    ignore_categories: frozenset[str] = frozenset(),
) -> dict:
    """Precision and recall of the exception list itself.

    ``should_be_flagged`` is the answer key's ``unmatchable`` population -- orphan
    credits, reversal wash pairs, missing settlements, invoices awaiting settlement.
    Anything in there that the engine resolved instead of flagging is a *fabricated
    match*: the failure mode that looks like a better score.
    """
    should = {
        u.txn_id for u in ds.unmatchable
        if (leg is None or u.leg == leg) and u.reason not in ignore_categories
    }
    tp = len(flagged_ids & should)
    fp = len(flagged_ids - should)
    fn = len(should - flagged_ids)
    return {
        "flagged": len(flagged_ids),
        "should_be_flagged": len(should),
        "correctly_flagged": tp,
        "over_flagged": fp,
        "missed": fn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 1.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 1.0,
        "missed_ids": sorted(should - flagged_ids)[:12],
        "reasons_missed": _reason_counts(ds, should - flagged_ids),
    }


def _reason_counts(ds: Dataset, ids: set[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for u in ds.unmatchable:
        if u.txn_id in ids:
            out[u.reason] = out.get(u.reason, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
