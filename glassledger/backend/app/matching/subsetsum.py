"""Bounded subset-sum: which of these lines add up to that one?

The N:1 and 1:N cases (batch settlements, split payouts, refund-netted batches)
reduce to subset-sum, which is NP-complete in general. What makes it tractable here
is that blocking has already reduced each instance to a handful of lines inside a
few-day window, so the search is exhaustive-with-pruning over 10-90 items rather
than over the whole month.

Three properties this implementation guarantees, in decreasing order of how much
they matter:

**Termination.** A node budget bounds the search absolutely. If the budget is
exhausted the result says so (``exhausted=False``) and the caller reports reduced
coverage rather than presenting a partial search as a complete one. A
reconciliation run that hangs on one pathological window is an outage; one that
says "I could not fully search this window" is an exception with a clear cause.

**Admissible pruning.** The bounds only ever cut branches that provably cannot
reach the target, so within the node budget the enumeration is complete -- no
heuristic silently drops a valid subset. This is where signed amounts complicate
things: with refunds present, the running sum is not monotone, so a single
suffix-sum bound is *not* admissible. The fix is two bounds, one over the positive
tail and one over the negative tail, giving the true reachable interval.

**Ambiguity is output, not hidden.** The search returns up to ``max_solutions``
distinct subsets, and the count matters as much as the winner. One solution is
evidence. Six solutions means the true answer is unknowable from amounts alone, and
that is what should reach the confidence model -- not the first subset the DFS
happened to find.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubsetSolution:
    indices: tuple[int, ...]
    total_paise: int
    residual_paise: int

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass
class SubsetSearchResult:
    solutions: list[SubsetSolution] = field(default_factory=list)
    nodes_visited: int = 0
    #: False when the node budget or the solution cap stopped the search early.
    #: The caller must surface this; a truncated search reported as complete is a
    #: silent coverage gap.
    exhausted: bool = True
    hit_solution_cap: bool = False

    @property
    def ambiguity(self) -> int:
        return len(self.solutions)


def find_subsets(
    amounts: list[int],
    target: int,
    tolerance: int,
    *,
    max_size: int = 8,
    max_solutions: int = 24,
    node_budget: int = 120_000,
) -> SubsetSearchResult:
    """Every subset of *amounts* summing to within *tolerance* of *target*.

    Items are assumed pre-sorted descending by the caller (``blocking``
    guarantees it). Descending order makes the positive-tail bound bite in the
    first few levels instead of the last few, which is worth roughly an order of
    magnitude of nodes on the pools that matter.
    """
    n = len(amounts)
    res = SubsetSearchResult()
    if n == 0:
        return res

    # Reachable interval from index i onward: adding every positive item gives the
    # maximum, adding every negative item gives the minimum. Any running sum whose
    # interval misses the target band cannot be completed into a solution.
    suffix_pos = [0] * (n + 1)
    suffix_neg = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        a = amounts[i]
        suffix_pos[i] = suffix_pos[i + 1] + (a if a > 0 else 0)
        suffix_neg[i] = suffix_neg[i + 1] + (a if a < 0 else 0)

    lo, hi = target - tolerance, target + tolerance
    seen: set[tuple[int, ...]] = set()
    nodes = 0
    stopped = False

    def dfs(i: int, current: int, chosen: tuple[int, ...]) -> None:
        nonlocal nodes, stopped
        if stopped:
            return
        nodes += 1
        if nodes > node_budget:
            stopped = True
            res.exhausted = False
            return

        if chosen and lo <= current <= hi and chosen not in seen:
            seen.add(chosen)
            res.solutions.append(
                SubsetSolution(
                    indices=chosen, total_paise=current, residual_paise=target - current
                )
            )
            if len(res.solutions) >= max_solutions:
                res.hit_solution_cap = True
                res.exhausted = False
                stopped = True
                return

        if i >= n or len(chosen) >= max_size:
            return
        # Admissible bounds -- these are the only two cuts, and both are exact.
        if current + suffix_pos[i] < lo:
            return
        if current + suffix_neg[i] > hi:
            return

        dfs(i + 1, current + amounts[i], chosen + (i,))
        dfs(i + 1, current, chosen)

    dfs(0, 0, ())
    res.nodes_visited = nodes

    # Prefer the smallest explanation, then the closest. Occam is the right prior
    # here: a settlement is far more likely to be four invoices than nine invoices
    # that happen to hit the same total, and a reviewer can verify four.
    res.solutions.sort(key=lambda s: (s.size, abs(s.residual_paise), s.indices))
    return res


def find_subsets_exact_first(
    amounts: list[int],
    target: int,
    tolerance: int,
    **kwargs,
) -> SubsetSearchResult:
    """Try a zero-tolerance search first, widening only if it finds nothing.

    A cheap but genuinely valuable ordering. On the gateway->bank leg most batches
    sum *exactly*, and an exact search has a far tighter feasible region -- both
    faster and unambiguous. Widening to the full tolerance only when the exact
    search comes up empty means the loose tolerance never gets the chance to
    manufacture a plausible-looking alternative to an exact answer that was
    already sitting there.
    """
    if tolerance > 0:
        strict = find_subsets(amounts, target, 0, **kwargs)
        if strict.solutions:
            return strict
    return find_subsets(amounts, target, tolerance, **kwargs)
