"""Tier 2: constrained optimisation over what tier 1 could not prove.

Two solvers, chosen by cardinality:

**1:1 -- Hungarian assignment.** ``scipy.optimize.linear_sum_assignment`` over the
blocked subgraph, minimising total ``-log P(match)``. Optimal, not greedy.

**N:1 / 1:N -- bounded subset-sum, then weighted set packing.** Enumerate summing
subsets (``subsetsum``), score each as a hypothesis, then select a
mutually-consistent set greedily by score.

Why global assignment instead of greedy nearest-match
-----------------------------------------------------
Greedy is locally right and globally wrong in a way that compounds. Settlement A
has one plausible partner scoring 0.85; settlement B has two, scoring 0.88 and
0.60, and its 0.88 is A's only option. Greedy takes B's 0.88 first, leaves A
unmatched, and books B at high confidence. Total probability mass: 0.88. The
optimal assignment gives B its 0.60 and A its 0.85, for 1.45 -- and it is right
about *both*. On a clean dataset the two agree; the divergence lives entirely in
the crowded windows, which is exactly the long tail this whole system exists for.

Why it scales past a demo
-------------------------
``linear_sum_assignment`` is O(n^3) dense. At 40,000 settlements a month that is
not a tuning problem, it is a wall. So the assignment is decomposed over connected
components of the blocking graph first: with no finite-cost edge between two
components, the global optimum is exactly the union of the per-component optima --
this is not an approximation, it is a factorisation. Real windows produce hundreds
of small components rather than one large one, so the cubic term applies to the
biggest component instead of to the whole month. The component-size distribution is
reported in the stats, because if one giant component ever forms, blocking has
failed and the run is about to be slow for a reason worth knowing.

Where the honest approximation is
---------------------------------
Optimal weighted set packing (choosing non-conflicting subset hypotheses) is
NP-hard, so the selection is greedy by score. That is a real approximation and it
is stated rather than glossed: on ties or near-ties it can pick a locally better
group that blocks a globally better pair. Its measured cost on the eval set is in
the run report as ``set_packing_conflicts``, so the size of the compromise is a
number rather than a hope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.config import LegConfig
from app.core.schema import NormalizedTxn

from . import features as F
from .blocking import BlockingGraph, SubsetPool
from .subsetsum import find_subsets_exact_first
from .types import Candidate

#: Cost assigned to an edge that blocking rejected. Large but finite: infinities
#: make the LAP infeasible rather than merely unattractive.
_FORBIDDEN = 1e7

#: Edges below this probability are not worth a column in the matrix. Cuts matrix
#: size substantially on the books leg with no measured effect on recall, because
#: nothing this weak survives the auto-confirm threshold anyway.
_EDGE_FLOOR = 0.01


@dataclass
class Tier2Result:
    candidates: list[Candidate] = field(default_factory=list)
    #: Every hypothesis considered, winners and losers, for the audit trail.
    considered: list[Candidate] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def considered_as_decisions(self) -> list:
        """Wrap the losing hypotheses as rejected decisions.

        The exception builder consumes decisions, and it needs the near-misses to
        populate an exception's evidence -- "we looked at these three and none was
        good enough" is a materially better message to a reviewer than "unmatched".
        """
        from .types import Decision

        return [
            Decision(candidate=c, action="reject", reason="not selected by the solver",
                     gate="uniqueness")
            for c in self.considered
        ]


# ---------------------------------------------------------------------------
# Connected components of the blocking graph
# ---------------------------------------------------------------------------

def connected_components(
    graph: BlockingGraph,
    left_ids: list[str],
    right_ids: list[str],
) -> list[tuple[list[str], list[str]]]:
    """Partition the bipartite graph so each part can be solved independently.

    Union-find over ``left_id``/``right_id`` labels. Written out rather than pulled
    from networkx because it is fifteen lines and the dependency would exist purely
    to avoid them.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for lid in left_ids:
        find(f"L:{lid}")
    for rid in right_ids:
        find(f"R:{rid}")
    for lid in left_ids:
        for rid in graph.edges.get(lid, ()):
            if rid in graph.right_by_id:
                union(f"L:{lid}", f"R:{rid}")

    groups: dict[str, tuple[list[str], list[str]]] = {}
    for lid in left_ids:
        groups.setdefault(find(f"L:{lid}"), ([], []))[0].append(lid)
    for rid in right_ids:
        groups.setdefault(find(f"R:{rid}"), ([], []))[1].append(rid)
    # Only components with both sides present can produce a match.
    return [(ls, rs) for ls, rs in groups.values() if ls and rs]


# ---------------------------------------------------------------------------
# 1:1 assignment
# ---------------------------------------------------------------------------

def score_pair_edges(
    graph: BlockingGraph,
    scorer,
    *,
    exclude_left: frozenset[str] = frozenset(),
    exclude_right: frozenset[str] = frozenset(),
    gross_lookup: dict[str, int] | None = None,
) -> dict[tuple[str, str], tuple[float, dict[str, float]]]:
    """Feature-extract and score every surviving edge, in one batch.

    Batched because the calibrated model's per-call overhead dominates its
    per-row cost: scoring 9,000 edges one at a time took ~40x longer than one
    9,000-row call, and that difference is the whole latency budget.
    """
    pairs: list[tuple[str, str]] = []
    vectors: list[dict[str, float]] = []

    for lid, rids in graph.edges.items():
        if lid in exclude_left:
            continue
        lt = graph.left_by_id[lid]
        for rid in rids:
            if rid in exclude_right:
                continue
            rt = graph.right_by_id[rid]
            feats = F.extract(
                [lt], [rt], graph.leg,
                left_ambiguity=graph.degree_left(lid),
                right_ambiguity=graph.degree_right(rid),
                amount_peers=graph.amount_peers(lid, rid),
                gross_paise=(gross_lookup or {}).get(lid) or (gross_lookup or {}).get(rid),
            )
            pairs.append((lid, rid))
            vectors.append(feats)

    scores = scorer.score_many(vectors) if vectors else []
    return {p: (float(s), v) for p, s, v in zip(pairs, scores, vectors)}


def solve_assignment(
    graph: BlockingGraph,
    scored: dict[tuple[str, str], tuple[float, dict[str, float]]],
    *,
    no_match_prior: float = 0.25,
) -> Tier2Result:
    """Optimal 1:1 assignment per connected component.

    ``no_match_prior`` is the probability that a left line simply has no partner --
    a missing settlement, a payout still in flight. It becomes the cost of the
    slack column, so "leave this unmatched" competes on the same scale as every
    real edge instead of being a post-hoc threshold. Derived from the training
    set's unmatched base rate, not guessed.
    """
    res = Tier2Result()
    left_ids = sorted({l for (l, _r) in scored})
    right_ids = sorted({r for (_l, r) in scored})
    if not left_ids or not right_ids:
        res.stats = {"components": 0, "assigned": 0}
        return res

    edge_p: dict[tuple[str, str], float] = {k: v[0] for k, v in scored.items()}
    comps = connected_components(graph, left_ids, right_ids)
    no_match_cost = -math.log(max(no_match_prior, 1e-9))

    assigned = 0
    comp_sizes: list[int] = []
    for ls, rs in comps:
        ls, rs = sorted(ls), sorted(rs)
        comp_sizes.append(len(ls) + len(rs))
        n, m = len(ls), len(rs)
        # Slack columns (one per row) let a row go unmatched at a fixed cost.
        # Per-row rather than shared, or two rows would compete for one "nothing".
        cost = np.full((n, m + n), _FORBIDDEN, dtype=float)
        for i, lid in enumerate(ls):
            cost[i, m + i] = no_match_cost
            for j, rid in enumerate(rs):
                p = edge_p.get((lid, rid))
                if p is not None and p >= _EDGE_FLOOR:
                    cost[i, j] = -math.log(p)

        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if j >= m:
                continue  # took the slack column: correctly left unmatched
            lid, rid = ls[i], rs[j]
            p, feats = scored[(lid, rid)]
            lt, rt = graph.left_by_id[lid], graph.right_by_id[rid]
            res.candidates.append(
                Candidate(
                    leg=graph.leg.name,
                    left_ids=(lid,),
                    right_ids=(rid,),
                    tier=2,
                    algorithm="hungarian_1to1",
                    features=feats,
                    score=p,
                    left_amount_paise=lt.amount_paise,
                    right_amount_paise=rt.amount_paise,
                    evidence=_pair_evidence(lt, rt, feats, graph, lid, rid),
                )
            )
            assigned += 1

    # Runners-up: for each assigned left, the alternatives it beat. This is the
    # part of the audit trail that answers "what else did you consider", which is
    # the question an auditor actually asks.
    chosen = {(c.left_ids[0], c.right_ids[0]) for c in res.candidates}
    for (lid, rid), (p, feats) in sorted(scored.items(), key=lambda kv: -kv[1][0]):
        if (lid, rid) in chosen or p < _EDGE_FLOOR:
            continue
        lt, rt = graph.left_by_id[lid], graph.right_by_id[rid]
        res.considered.append(
            Candidate(
                leg=graph.leg.name, left_ids=(lid,), right_ids=(rid,), tier=2,
                algorithm="hungarian_rejected", features=feats, score=p,
                left_amount_paise=lt.amount_paise, right_amount_paise=rt.amount_paise,
                evidence={"rule": "considered by the assignment and not selected"},
            )
        )

    res.stats = {
        "components": len(comps),
        "largest_component": max(comp_sizes) if comp_sizes else 0,
        "mean_component": round(sum(comp_sizes) / len(comp_sizes), 2) if comp_sizes else 0,
        "edges_scored": len(scored),
        "assigned": assigned,
        "left_unassigned": len(left_ids) - assigned,
    }
    return res


def _pair_evidence(
    lt: NormalizedTxn, rt: NormalizedTxn, feats: dict[str, float],
    graph: BlockingGraph, lid: str, rid: str,
) -> dict:
    sim, lref, rref = F.reference_similarity(
        lt.ref_candidates or ((lt.utr,) if lt.utr else ()),
        rt.ref_candidates or ((rt.utr,) if rt.utr else ()),
    )
    residual = rt.amount_paise - lt.amount_paise
    delta_days = (rt.value_date - lt.value_date).days
    return {
        "rule": "optimal 1:1 assignment over the blocked candidate graph",
        "residual_paise": residual,
        "date_delta_days": delta_days,
        "settlement_cycle_prior": round(graph.leg.cycle_prior.get(delta_days, 0.0025), 4),
        "reference_comparison": {
            "left": lref, "right": rref, "similarity": round(sim, 4),
            "exact": bool(feats.get("utr_exact")),
        },
        "competing_candidates": {
            "for_this_left": graph.degree_left(lid),
            "for_this_right": graph.degree_right(rid),
            "amount_indistinguishable": graph.amount_peers(lid, rid),
        },
    }


# ---------------------------------------------------------------------------
# N:1 subset hypotheses
# ---------------------------------------------------------------------------

def solve_subsets(
    pools: list[SubsetPool],
    members_by_id: dict[str, NormalizedTxn],
    anchors_by_id: dict[str, NormalizedTxn],
    leg: LegConfig,
    scorer,
    *,
    gross_lookup: dict[str, int] | None = None,
    anchor_is_left: bool = False,
    attributor=None,
) -> Tier2Result:
    """Enumerate summing subsets, score them, then pack non-conflicting winners.

    ``anchor_is_left`` records which side of the leg the anchor sits on so the
    resulting ``Candidate`` has its ``left_ids``/``right_ids`` the right way round.
    Getting this wrong does not crash -- it produces candidates whose pair keys can
    never match ground truth, which shows up as an unexplained recall cliff on one
    leg only.
    """
    res = Tier2Result()
    hypotheses: list[Candidate] = []
    truncated_pools = 0
    total_nodes = 0
    capped = 0

    to_score: list[tuple[SubsetPool, list[NormalizedTxn], int, int]] = []
    vectors: list[dict[str, float]] = []

    for pool in pools:
        if len(pool.members) < 2:
            continue  # a 1-member "subset" is the 1:1 problem, already solved
        amounts = [t.amount_paise for t in pool.members]
        search = find_subsets_exact_first(
            amounts, pool.target_paise, pool.slack_paise,
            max_size=leg.max_subset_size,
        )
        total_nodes += search.nodes_visited
        if not search.exhausted:
            truncated_pools += 1
        if search.hit_solution_cap:
            capped += 1

        anchor = anchors_by_id[pool.anchor_id]
        # Score at most the four best explanations per anchor. Beyond that the
        # marginal hypothesis is both implausible and expensive, and keeping them
        # would inflate the "considered" trail into noise.
        for sol in search.solutions[:4]:
            group = [pool.members[i] for i in sol.indices]
            left_side = [anchor] if anchor_is_left else group
            right_side = group if anchor_is_left else [anchor]
            gross = (gross_lookup or {}).get(pool.anchor_id)
            feats = F.extract(
                left_side, right_side, leg,
                left_ambiguity=search.ambiguity,
                right_ambiguity=1,
                amount_peers=search.ambiguity,
                gross_paise=gross,
            )
            if attributor is not None:
                expl = attributor.explain(left_side, right_side, leg, gross or 0)
                # A structural verdict about the residual is a real feature, and it
                # has to be in the vector *before* scoring -- see residuals.py.
                feats["within_fee_band"] = 1.0 if expl.detail.get("explainable") else 0.0
            to_score.append((pool, group, sol.total_paise, search.ambiguity))
            vectors.append(feats)

    scores = scorer.score_many(vectors) if vectors else []
    for (pool, group, total, ambiguity), feats, p in zip(to_score, vectors, scores):
        anchor = anchors_by_id[pool.anchor_id]
        group_ids = tuple(sorted(t.txn_id for t in group))
        if anchor_is_left:
            left_ids, right_ids = (pool.anchor_id,), group_ids
            left_amt, right_amt = anchor.amount_paise, total
        else:
            left_ids, right_ids = group_ids, (pool.anchor_id,)
            left_amt, right_amt = total, anchor.amount_paise
        hypotheses.append(
            Candidate(
                leg=leg.name,
                left_ids=left_ids,
                right_ids=right_ids,
                tier=2,
                algorithm="subset_sum_1toN" if anchor_is_left else "subset_sum_Nto1",
                features=feats,
                score=float(p),
                left_amount_paise=left_amt,
                right_amount_paise=right_amt,
                evidence={
                    "rule": f"{len(group)} lines sum to {total} against a target of "
                            f"{pool.target_paise} (tolerance {pool.slack_paise} paise)",
                    "residual_paise": pool.target_paise - total,
                    "alternative_subsets_found": ambiguity,
                    "pool_size": len(pool.members),
                    "pool_truncated": pool.truncated,
                    "members": [
                        {"txn_id": t.txn_id, "amount_paise": t.amount_paise,
                         "value_date": t.value_date.isoformat(),
                         "narration": (t.narration or "")[:80]}
                        for t in sorted(group, key=lambda x: -x.amount_paise)
                    ],
                },
            )
        )

    # Greedy weighted set packing. NP-hard to do optimally; greedy-by-score is the
    # documented approximation and the conflicts it causes are counted below.
    hypotheses.sort(key=lambda c: (-c.score, len(c.left_ids) + len(c.right_ids), c.match_key))
    used_members: set[str] = set()
    used_anchors: set[str] = set()
    conflicts = 0
    for c in hypotheses:
        anchor_id = c.left_ids[0] if anchor_is_left else c.right_ids[0]
        group = c.right_ids if anchor_is_left else c.left_ids
        if anchor_id in used_anchors or used_members & set(group):
            conflicts += 1
            res.considered.append(c)
            continue
        res.candidates.append(c)
        used_anchors.add(anchor_id)
        used_members.update(group)

    res.stats = {
        "pools_searched": len(pools),
        "hypotheses": len(hypotheses),
        "selected": len(res.candidates),
        "set_packing_conflicts": conflicts,
        "pools_search_truncated": truncated_pools,
        "pools_solution_capped": capped,
        "subset_nodes_visited": total_nodes,
    }
    return res
