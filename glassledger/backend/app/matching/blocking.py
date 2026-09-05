"""Blocking: turning an O(N*M) matching problem into something that finishes.

Matching 288 settlements against 352 bank lines is 101,376 pairs -- fine. Doing it
for a merchant with 40,000 settlements a month against 45,000 bank lines is 1.8
billion, and the subset search on top of that is 2**N. Blocking is the step that
makes the difference between an algorithm and a demo.

Two different windows, because there are two different questions:

``pair_edges``
    "Could this one left line be this one right line?" Requires the amounts to be
    *close*: within a relative band or an absolute floor.

``subset_pool``
    "Could this right line be the sum of some of these left lines?" Requires each
    left amount to be *no larger than* the target plus tolerance -- a completely
    different predicate. A pool built with the pair predicate would exclude every
    genuine batch member, which is a subtle and expensive way to get 0% recall on
    break pattern #1.

Every reduction is measured and reported (``BlockingGraph.stats``). A blocking
step that silently drops candidates is indistinguishable from a matcher that
cannot find them, and the honest thing is to know which one you have. When a pool
hits ``max_window_candidates`` the truncation is counted and surfaced in the run
report rather than being absorbed quietly -- a capped pool means reduced coverage,
and reduced coverage the operator does not know about is a lie by omission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.core.config import LegConfig
from app.core.schema import NormalizedTxn


@dataclass
class BlockingGraph:
    leg: LegConfig
    left: list[NormalizedTxn]
    right: list[NormalizedTxn]
    left_by_id: dict[str, NormalizedTxn] = field(default_factory=dict)
    right_by_id: dict[str, NormalizedTxn] = field(default_factory=dict)
    #: left_id -> candidate right_ids (the pair window)
    edges: dict[str, list[str]] = field(default_factory=dict)
    #: right_id -> candidate left_ids
    reverse: dict[str, list[str]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def degree_left(self, left_id: str) -> int:
        return len(self.edges.get(left_id, ()))

    def degree_right(self, right_id: str) -> int:
        return len(self.reverse.get(right_id, ()))

    def amount_peers(self, left_id: str, right_id: str) -> int:
        """How many of this left's candidates are amount-indistinguishable from this one.

        The core ambiguity signal. Two bank credits of exactly Rs 4,998.50 in the
        same window are not two pieces of evidence, they are one piece of evidence
        and a coin flip -- and the confidence score has to reflect that or the
        engine will confidently pick one at random.
        """
        target = self.right_by_id[right_id].amount_paise
        tol = max(100, abs(target) // 1000)  # 1 paise floor of Rs 1, else 0.1%
        return sum(
            1
            for rid in self.edges.get(left_id, ())
            if abs(self.right_by_id[rid].amount_paise - target) <= tol
        )


def _amount_admissible(left_amt: int, right_amt: int, leg: LegConfig) -> bool:
    delta = abs(right_amt - left_amt)
    if delta <= leg.amount_abs_tolerance_paise:
        return True
    scale = max(abs(left_amt), abs(right_amt), 1)
    return delta / scale <= leg.amount_rel_tolerance


def build_graph(
    left: list[NormalizedTxn],
    right: list[NormalizedTxn],
    leg: LegConfig,
    *,
    right_sign: int = 1,
) -> BlockingGraph:
    """Pair-window candidate graph.

    ``right_sign`` restricts the right side by sign. On the gateway->bank leg the
    matcher is looking for *credits*: a debit can never be the counterpart of a
    positive settlement, and admitting them doubles the candidate count while
    adding only ways to be wrong. Reversal debits still get handled -- by the
    wash-pair rule in tier 1, which is the right tool for them.
    """
    g = BlockingGraph(leg=leg, left=left, right=right)
    g.left_by_id = {t.txn_id: t for t in left}
    g.right_by_id = {t.txn_id: t for t in right}

    by_date: dict[date, list[NormalizedTxn]] = defaultdict(list)
    for t in right:
        if right_sign and (t.amount_paise > 0) != (right_sign > 0):
            continue
        by_date[t.value_date].append(t)

    considered = 0
    for lt in left:
        cands: list[str] = []
        for offset in range(-leg.date_window_before, leg.date_window_after + 1):
            for rt in by_date.get(lt.value_date + timedelta(days=offset), ()):
                considered += 1
                if lt.currency != rt.currency:
                    continue
                if _amount_admissible(lt.amount_paise, rt.amount_paise, leg):
                    cands.append(rt.txn_id)
        g.edges[lt.txn_id] = cands
        for rid in cands:
            g.reverse.setdefault(rid, []).append(lt.txn_id)

    n_edges = sum(len(v) for v in g.edges.values())
    full = len(left) * len(right)
    degrees = [len(v) for v in g.edges.values()] or [0]
    g.stats = {
        "left": len(left),
        "right": len(right),
        "full_cross_product": full,
        "pairs_considered": considered,
        "edges_kept": n_edges,
        "reduction_vs_full": round(1.0 - (n_edges / full), 6) if full else 0.0,
        "max_left_degree": max(degrees),
        "mean_left_degree": round(sum(degrees) / len(degrees), 3),
        "left_with_no_candidate": sum(1 for v in g.edges.values() if not v),
    }
    return g


@dataclass
class SubsetPool:
    """The lines that could sum to one anchor target."""

    anchor_id: str
    target_paise: int
    members: list[NormalizedTxn]
    truncated: bool = False
    considered: int = 0
    slack_paise: int = 0


def build_subset_pools(
    members: list[NormalizedTxn],
    anchors: list[NormalizedTxn],
    leg: LegConfig,
    *,
    exclude_members: frozenset[str] = frozenset(),
    exclude_anchors: frozenset[str] = frozenset(),
    residual_targets: dict[str, int] | None = None,
    slack_lookup: dict[str, int] | None = None,
    date_hints: dict[str, set[date]] | None = None,
    anchor_is_left: bool = False,
) -> tuple[list[SubsetPool], dict]:
    """One pool per anchor, for the subset search.

    ``anchor_is_left`` selects the direction of the date window, and it is not
    cosmetic. On gateway->bank the anchor is the settlement and its constituent
    credits arrive *after* it; on books->gateway the anchor is the settlement and
    its constituent invoices were booked *before* it. Using one direction for both
    silently empties every pool on one of the two legs.

    ``residual_targets`` is the interesting parameter. On the books leg, tier 1 has
    already claimed the entries that carry an order key, so the amount the
    remaining unkeyed entries must sum to is not the settlement's net -- it is what
    is *left over* after the keyed ones. Passing the reduced target shrinks the
    search space and sharpens the answer at the same time: a Rs 4,000 residual has
    far fewer plausible subsets than a Rs 90,000 total, so the search is both faster
    and less ambiguous. Tier 1 is not merely a fast path here, it is what makes
    tier 2 accurate.

    ``slack_lookup`` gives the per-anchor tolerance. It has to be per-anchor
    because the explainable gap (fees, withholding, FX) is a percentage of *gross*,
    and a flat rupee tolerance either rejects every large batch or accepts garbage
    on every small one.

    ``date_hints`` overrides the window entirely for anchors that have one. A
    settlement whose keyed entries tier 1 already joined has *told us* which
    capture day its batch covers, so guessing with a window afterwards is strictly
    worse information. See ``expectations.py`` for how much this matters -- it is
    the single largest factor in this leg's search cost.
    """
    pools: list[SubsetPool] = []
    truncated = 0
    considered_total = 0
    hint_constrained = 0

    by_date: dict[date, list[NormalizedTxn]] = defaultdict(list)
    for t in members:
        if t.txn_id in exclude_members:
            continue
        by_date[t.value_date].append(t)

    if anchor_is_left:
        offsets = range(-leg.date_window_before, leg.date_window_after + 1)
    else:
        offsets = range(-leg.date_window_after, leg.date_window_before + 1)

    for at in anchors:
        if at.txn_id in exclude_anchors:
            continue
        target = (residual_targets or {}).get(at.txn_id, at.amount_paise)
        slack = (slack_lookup or {}).get(
            at.txn_id,
            leg.subset_tolerance_paise + int(abs(target) * leg.subset_rel_tolerance),
        )
        ceiling = abs(target) + slack

        hints = (date_hints or {}).get(at.txn_id)
        if hints:
            search_dates = sorted(hints)
            hint_constrained += 1
        else:
            search_dates = [at.value_date + timedelta(days=o) for o in offsets]

        pool_members: list[NormalizedTxn] = []
        considered = 0
        for d in search_dates:
            for lt in by_date.get(d, ()):
                considered += 1
                if lt.currency != at.currency:
                    continue
                # A single member larger than the target (plus slack) cannot be
                # part of a subset that sums to it -- unless there are negative
                # members to offset it, which is why refunds are admitted
                # regardless of magnitude.
                if lt.amount_paise < 0 or abs(lt.amount_paise) <= ceiling:
                    pool_members.append(lt)

        considered_total += considered
        # Keep the members most likely to participate: large positives first
        # (they constrain the sum hardest) then refunds. Ordering matters because
        # this is also the order the depth-first search will explore, and a
        # descending order makes the suffix-sum bound bite early.
        pool_members.sort(key=lambda t: (-t.amount_paise, t.txn_id))
        was_truncated = len(pool_members) > leg.max_window_candidates
        if was_truncated:
            truncated += 1
            pool_members = pool_members[: leg.max_window_candidates]

        pools.append(
            SubsetPool(
                anchor_id=at.txn_id,
                target_paise=target,
                members=pool_members,
                truncated=was_truncated,
                considered=considered,
                slack_paise=slack,
            )
        )

    sizes = [len(p.members) for p in pools] or [0]
    stats = {
        "pools": len(pools),
        "pairs_considered": considered_total,
        "pools_truncated": truncated,
        "pools_date_hinted": hint_constrained,
        "max_pool_size": max(sizes),
        "mean_pool_size": round(sum(sizes) / len(sizes), 3),
        "empty_pools": sum(1 for p in pools if not p.members),
    }
    return pools, stats
