"""Tier 3 orchestration: selecting what to arbitrate, and what to do with the answer.

The arbiter itself (``llm.arbiter``) knows how to ask a question. This module decides
*which* questions are legitimate to ask, which is where the safety properties live.

Selection, in order:

1. Take the candidates the policy layer left open -- the ones it *proposed* rather
   than auto-confirmed -- together with the candidates it rejected under the
   ``uniqueness`` gate, which are the rival explanations for the same money.
   Without the rivals every group has one member and there is nothing to arbitrate;
   see ``select_for_arbitration`` for why that is a property of the policy rather
   than an accident.
2. Drop anything at or above the materiality gate. Not "instruct the model to be
   careful" -- the item never enters the payload.
3. Drop anything outside ``[TIER3_FLOOR, threshold)``. Above the threshold there is
   nothing to decide; below the floor there is nothing to decide it with.
4. Group by the contested transaction, so the arbiter sees a *choice* rather than
   one candidate in isolation. A model shown a single option will find reasons to
   like it; a model shown three has to discriminate, which is the task.
5. Cap at ``max_arbitrations``. Cost and latency are bounded before the first call,
   not discovered afterwards -- and the cap is reported, so a run that hit it says
   so instead of looking like a run that had nothing left to arbitrate.

Applying the answer: the delta adjusts the candidate's calibrated score, and the
policy layer then runs again **over the leg's whole candidate set** with the
arbitrated candidates substituted in. The arbiter never writes a decision. Every
gate that applied before arbitration applies after it, unchanged -- including the
materiality gate, which the adjusted candidate is re-checked against even
though it was filtered out at selection.

Cost is measured, not estimated. ``Tier3Report`` carries token usage per call, so
the eval reports what arbitration actually costs per hundred transactions rather
than asserting it is cheap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

from app.core.config import MATERIALITY_PAISE
from app.matching.policy import Policy, PolicyResult
from app.matching.types import Candidate, Decision

from .arbiter import (
    MAX_CONFIDENCE_DELTA,
    TIER3_FLOOR,
    Arbitration,
    Arbiter,
    default_arbiter,
)


@dataclass
class Tier3Report:
    arbitrations: list[Arbitration] = field(default_factory=list)
    eligible: int = 0
    skipped_above_materiality: int = 0
    skipped_outside_band: int = 0
    capped: int = 0
    adjusted: list[Candidate] = field(default_factory=list)
    #: The groups as offered, parallel to ``arbitrations``. Kept so the recorder can
    #: write down what choice the arbiter was actually given -- an arbitration
    #: recorded without its alternatives is the same omission POSTMORTEM #4 was
    #: about, one level up.
    groups: list[list[Candidate]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        proposed = sum(1 for a in self.arbitrations if a.action == "propose_match")
        tokens_in = sum((a.usage or {}).get("input_tokens") or 0 for a in self.arbitrations)
        tokens_out = sum((a.usage or {}).get("output_tokens") or 0 for a in self.arbitrations)
        return {
            "arbiter": self.stats.get("arbiter"),
            "band": self.stats.get("band"),
            "eligible": self.eligible,
            "calls_made": len(self.arbitrations),
            "proposed_match": proposed,
            "insufficient_evidence": len(self.arbitrations) - proposed,
            "clamped": sum(1 for a in self.arbitrations if a.clamped),
            "errors": sum(1 for a in self.arbitrations if a.error),
            "error_kinds": _count_by(a.error for a in self.arbitrations if a.error),
            "skipped_above_materiality": self.skipped_above_materiality,
            "skipped_outside_band": self.skipped_outside_band,
            "capped_not_arbitrated": self.capped,
            "singleton_groups": self.stats.get("singleton_groups"),
            "chose_unselected_rival": self.stats.get("chose_unselected_rival"),
            "wall_ms": self.stats.get("wall_ms"),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "decisions": [a.to_json() for a in self.arbitrations],
        }


def _count_by(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        # Only the error *kind* is aggregated. Exception messages carry ids and
        # addresses, and a histogram keyed on the full string is one row per call.
        key = str(v).split(":", 1)[0]
        out[key] = out.get(key, 0) + 1
    return out


def select_for_arbitration(
    decisions: list[Decision],
    *,
    threshold: float,
    max_arbitrations: int = 25,
) -> tuple[list[list[Candidate]], dict]:
    """Group contested candidates into arbitration units, applying every gate first.

    ``decisions`` must contain the *losers as well as the survivors*: the policy's
    ``proposed`` list, plus the rival hypotheses wrapped as ``uniqueness`` rejects.
    Getting this wrong makes the tier useless rather than merely worse, and the
    reason is structural.

    ``Policy``'s uniqueness rule rejects any candidate that is not the best-supported
    claim on each of its right-hand ids. So every candidate in ``proposed`` has
    already won its subject outright, and two of them can share a right id only on a
    near-exact score tie. Group ``proposed`` alone and the groups come out of size
    one -- the arbiter is then shown a single option and asked to discriminate, which
    is the one thing this module exists to prevent.

    Where the rivals actually come from took a measurement to establish. The obvious
    source, ``PolicyResult.rejected`` under the ``uniqueness`` gate, is **empty on
    both legs**: tier 2's assignment and set-packing steps already enforce mutual
    exclusivity, so the policy's uniqueness rule is a backstop that almost never
    fires. The real alternatives are the solvers' scored-but-not-selected hypotheses
    (``Tier2Result.considered``), which the engine wraps as uniqueness rejects before
    calling in. Those carry full feature vectors and are genuinely the competing
    explanations for the same money.

    A group is kept only if it contains at least one *proposed* candidate. A cluster
    made entirely of also-rans is money the policy already settled somewhere else,
    and re-opening it would be arbitrating a decided question.
    """
    above_materiality = 0
    outside_band = 0
    pool: list[Candidate] = []
    #: match_keys the policy left open. Used to drop groups that are pure
    #: also-rans, and to sort by how contested the open item actually is.
    open_keys: set[str] = set()

    for d in decisions:
        c = d.candidate
        if c.exposure_paise >= MATERIALITY_PAISE:
            above_materiality += 1
            continue
        if not (TIER3_FLOOR <= c.score < threshold):
            outside_band += 1
            continue
        pool.append(c)
        if d.action == "propose":
            open_keys.add(c.match_key)

    # Group by contested right-hand id: the candidates competing to explain the
    # same money are the ones an arbiter can usefully compare.
    by_subject: dict[str, list[Candidate]] = {}
    for c in pool:
        by_subject.setdefault(c.right_ids[0], []).append(c)

    groups = [
        sorted(g, key=lambda x: -x.score)
        for g in by_subject.values()
        if any(c.match_key in open_keys for c in g)
    ]
    singletons = sum(1 for g in groups if len(g) == 1)
    # Hardest first: the ones where the top two candidates are closest together are
    # where a tie-break is worth most. Sorting by raw score instead would spend the
    # budget on cases that were nearly decided anyway.
    groups.sort(key=lambda g: (g[0].score - g[1].score) if len(g) > 1 else 1.0)

    capped = max(0, len(groups) - max_arbitrations)
    return groups[:max_arbitrations], {
        "above_materiality": above_materiality,
        "outside_band": outside_band,
        "capped": capped,
        "eligible": len(groups),
        # Reported because a singleton is a degenerate arbitration: there is
        # nothing to discriminate against. A run that is mostly singletons is a
        # run where this tier is not doing its job, and that should be visible as
        # a number rather than inferred from a disappointing match rate.
        "singleton_groups": singletons,
        "pool": len(pool),
    }


def apply_arbitration(group: list[Candidate], a: Arbitration) -> list[Candidate]:
    """Return the group with the arbiter's delta applied. Never confirms anything.

    The adjusted score is clamped to [0, 1] as well as to the delta bound, so an
    arbiter cannot push a candidate past 1.0 and out of the range every downstream
    consumer assumes.
    """
    if a.action != "propose_match" or not (1 <= a.chosen_index <= len(group)):
        return group

    out: list[Candidate] = []
    for i, c in enumerate(group, start=1):
        if i != a.chosen_index:
            out.append(c)
            continue
        new_score = max(0.0, min(1.0, c.score + a.confidence_delta))
        out.append(
            replace(
                c,
                score=new_score,
                tier=3,
                algorithm=f"{c.algorithm}+arbitrated",
                evidence={
                    **c.evidence,
                    "tier3_arbiter": a.arbiter,
                    "tier3_rationale": a.rationale,
                    "tier3_evidence_cited": a.evidence_cited,
                    "tier3_confidence_delta": round(a.confidence_delta, 6),
                    "tier3_score_before": round(c.score, 6),
                    "tier3_score_after": round(new_score, 6),
                    "tier3_clamped": a.clamped,
                },
            )
        )
    return out


def run_tier3(
    decisions: list[Decision],
    *,
    threshold: float,
    arbiter: Arbiter | None = None,
    max_arbitrations: int = 25,
    context_builder=None,
) -> Tier3Report:
    """Arbitrate the ambiguous band and return the adjusted candidates.

    The caller re-runs the policy layer over ``report.adjusted``. Tier 3 producing
    candidates rather than decisions is the structural expression of "proposer, never
    approver" -- there is no code path from here to a confirmation that does not go
    through the same policy the deterministic tiers go through.
    """
def run_tier3(
    decisions: list[Decision],
    *,
    threshold: float,
    arbiter: Arbiter | None = None,
    max_arbitrations: int = 25,
    context_builder=None,
    selectable: set[str] | None = None,
) -> Tier3Report:
    """Arbitrate the ambiguous band and return the adjusted candidates.

    The caller re-runs the policy layer over ``report.adjusted``. Tier 3 producing
    candidates rather than decisions is the structural expression of "proposer, never
    approver" -- there is no code path from here to a confirmation that does not go
    through the same policy the deterministic tiers go through.

    ``selectable`` is the set of match_keys that may actually receive a delta: the
    hypotheses the solvers *selected*. Rival hypotheses are shown to the arbiter
    because a choice needs alternatives, but a rival cannot be promoted by
    arbitration -- it was never in the mutually-consistent set the packing step
    produced, so confirming it could conflict with a match already confirmed
    elsewhere. When the arbiter picks a rival, the delta is **not** applied and the
    disagreement is counted as ``chose_unselected_rival``. That number is reported
    rather than swallowed: "the arbiter preferred a hypothesis the solver had
    discarded, 4 times" is a fact about the solver worth surfacing, and silently
    dropping the answer would make the tier look like it had no opinion.
    """
    arbiter = arbiter or default_arbiter()
    rep = Tier3Report()
    groups, sel = select_for_arbitration(
        decisions, threshold=threshold, max_arbitrations=max_arbitrations
    )
    rep.eligible = sel["eligible"]
    rep.skipped_above_materiality = sel["above_materiality"]
    rep.skipped_outside_band = sel["outside_band"]
    rep.capped = sel["capped"]
    rep.stats["arbiter"] = getattr(arbiter, "name", type(arbiter).__name__)
    rep.stats["singleton_groups"] = sel["singleton_groups"]
    rep.stats["pool"] = sel["pool"]
    rep.stats["threshold"] = round(threshold, 6)
    rep.stats["band"] = [TIER3_FLOOR, round(threshold, 6)]
    #: Carried on the report so the recorder can write the bound into the event
    #: without the audit core having to import the LLM module. The events layer
    #: knowing nothing about arbiters is deliberate: it records facts, and a
    #: dependency in that direction would make the audit core unbuildable without
    #: the tier it is supposed to be able to audit the absence of.
    rep.stats["max_confidence_delta"] = MAX_CONFIDENCE_DELTA

    rivals_chosen = 0
    t0 = time.perf_counter()
    for group in groups:
        context = context_builder(group) if context_builder else {}
        a = arbiter.arbitrate(group, context)

        if (
            selectable is not None
            and a.action == "propose_match"
            and 1 <= a.chosen_index <= len(group)
            and group[a.chosen_index - 1].match_key not in selectable
        ):
            rivals_chosen += 1
            a = replace(
                a,
                action="insufficient_evidence",
                chosen_index=0,
                confidence_delta=0.0,
                reason=(
                    "arbiter preferred a hypothesis the solver did not select; "
                    "recorded as a disagreement, not applied -- an unselected rival "
                    "is not part of the mutually-consistent match set"
                ),
                error="chose_unselected_rival",
            )

        rep.arbitrations.append(a)
        rep.adjusted.extend(apply_arbitration(group, a))
        rep.groups.append(group)
    rep.stats["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    rep.stats["chose_unselected_rival"] = rivals_chosen

    return rep


def substitute(candidates: list[Candidate], report: Tier3Report) -> list[Candidate]:
    """The full candidate list with arbitrated candidates swapped in by ``match_key``.

    This exists because re-running the policy over ``report.adjusted`` alone is
    wrong, and wrong in the direction that manufactures matches. ``Policy``'s
    uniqueness rule is a statement about the *whole* candidate set -- "no candidate
    survives if a better-supported one claims its partner" -- so evaluating it over
    the arbitrated subset asks a different question and gets a different answer. A
    candidate the full set had ruled out can be the best thing in a three-element
    slice, and it would come back confirmed on the strength of an arbitration that
    never addressed the rival that beat it.

    ``match_key`` is derived from the leg and the two id tuples, and
    ``apply_arbitration`` only ever replaces ``score``, ``tier``, ``algorithm`` and
    ``evidence`` -- so the key is stable across arbitration and identifies the same
    hypothesis before and after.
    """
    adjusted_by_key = {c.match_key: c for c in report.adjusted}
    return [adjusted_by_key.get(c.match_key, c) for c in candidates]


def repolicy(
    report: Tier3Report,
    policy: Policy,
    candidates: list[Candidate],
    *,
    repair=None,
) -> PolicyResult:
    """Re-run the policy layer over the full candidate set, arbitration applied.

    Every gate applies again from scratch. In particular the materiality gate is
    re-evaluated: a candidate excluded from arbitration for being above the line
    would still be caught here if it somehow arrived, which means the gate holds
    even if the selection filter is one day changed by someone who did not read
    this file.

    ``candidates`` is the leg's whole candidate list, not the arbitrated subset --
    see ``substitute`` for why that distinction is load-bearing. ``repair`` is the
    leg's own uniqueness fix-up (the books leg has one, because a settlement
    legitimately has many book entries); it has to run again after re-policying or
    arbitrated book entries are rejected by a rule that does not apply to them.
    """
    result = policy.decide_many(substitute(candidates, report))
    return repair(result, policy) if repair is not None else result
