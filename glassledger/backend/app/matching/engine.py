"""The reconciliation engine: tiers, in order, once per leg.

    engine = ReconciliationEngine(scorer)
    run = engine.run(batch)

Per-leg pipeline, and the *order* is the design:

    gateway -> bank                          books -> gateway
    ---------------------------------        ---------------------------------
    1. wash-pair netting (tier 1)            1. order-key join (tier 1)
    2. exact reference match (tier 1)        2. residual targets per settlement
    3. blocked graph + Hungarian (tier 2)    3. subset-sum on unkeyed entries
    4. subset-sum for splits (tier 2)        4. residual attribution
    5. policy: gate, threshold, uniqueness   5. policy
    6. exceptions for the remainder          6. exceptions

The two legs use *different solvers*, on purpose. Gateway->bank is dominated by 1:1
correspondences with a hard exactness requirement, which is precisely what an
optimal assignment is for. Books->gateway has no 1:1 structure at all -- a payout
is inherently a group of invoices, and the amounts are never expected to agree
exactly -- so assignment is the wrong shape and subset-sum with residual
attribution is the right one. Running one generic solver over both would mean
choosing which leg to do badly.

Everything the engine touches ends up in exactly one of three states: matched
(auto-confirmed or awaiting human approval), exception (with a category, an
evidence bundle, and near-miss hypotheses), or explicitly written off as
immaterial. There is no fourth state, and no silent drop -- ``run.assert_complete``
verifies that every ingested transaction id is accounted for, and raises if any
is not. A reconciliation that loses track of a line is worse than one that flags
it, because the exception list is the only thing standing between the merchant
and a wrong number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.core.config import LEGS, TIER3_BAND, GATEWAY_BOOKS_COMPONENT, LegConfig
from app.core.schema import NormalizedTxn
from app.ingestion import IngestedBatch

from . import tier1, tier2
from .blocking import BlockingGraph, build_graph, build_subset_pools
from .confidence import DummyScorer
from .exceptions import ExceptionReport, build_exceptions
from .expectations import (
    build_expectations,
    claimed_component_ids,
    estimate_books_fee_bps,
    expected_components,
)
from .policy import Policy, PolicyResult
from .residuals import ResidualAttribution
from .types import Candidate, Decision, Exception_


@dataclass
class LegRun:
    leg: str
    graph: BlockingGraph
    decisions: PolicyResult
    exceptions: ExceptionReport
    all_candidates: list[Candidate] = field(default_factory=list)
    #: Hypotheses the solvers scored and did *not* select. Kept because they are
    #: the negative class for training: a model fitted only on selected candidates
    #: has never seen a bad one, since the solver filtered them out first. That is
    #: train/serve skew of the worst kind -- the model looks calibrated on a
    #: population it will never encounter.
    considered: list[Candidate] = field(default_factory=list)
    wash_pairs: list[Candidate] = field(default_factory=list)
    deferred: list[dict] = field(default_factory=list)
    #: The tier-3 report, when an arbiter was supplied. ``None`` means the tier did
    #: not run at all, which is a different fact from "it ran and proposed nothing"
    #: and is reported as such everywhere downstream.
    tier3: Any = None
    stats: dict = field(default_factory=dict)

    def confirmed(self) -> list[Decision]:
        return self.decisions.auto_confirmed

    def proposed(self) -> list[Decision]:
        return self.decisions.proposed


@dataclass
class ReconciliationRun:
    legs: dict[str, LegRun] = field(default_factory=dict)
    batch_summary: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    scorer_name: str = ""
    threshold: float = 0.0
    as_of: date | None = None

    def all_confirmed(self) -> list[Decision]:
        return [d for lr in self.legs.values() for d in lr.decisions.auto_confirmed]

    def all_proposed(self) -> list[Decision]:
        return [d for lr in self.legs.values() for d in lr.decisions.proposed]

    def all_exceptions(self) -> list[Exception_]:
        return [e for lr in self.legs.values() for e in lr.exceptions.exceptions]

    def all_wash_pairs(self) -> list[Candidate]:
        return [c for lr in self.legs.values() for c in lr.wash_pairs]

    def tier3_ran(self) -> bool:
        return any(lr.tier3 is not None for lr in self.legs.values())

    def tier3_totals(self) -> dict:
        """Roll the per-leg tier-3 reports into one set of figures.

        Returns an empty dict when the tier did not run, so a caller cannot print
        zeroes for a tier that was never asked a question -- "0 arbitrations" and
        "no arbitration tier" are different claims.
        """
        reports = [lr.tier3 for lr in self.legs.values() if lr.tier3 is not None]
        if not reports:
            return {}
        js = [r.to_json() for r in reports]
        total = {
            k: sum(j.get(k) or 0 for j in js)
            for k in (
                "eligible", "calls_made", "proposed_match", "insufficient_evidence",
                "clamped", "errors", "skipped_above_materiality",
                "skipped_outside_band", "capped_not_arbitrated", "singleton_groups",
                "chose_unselected_rival", "tokens_in", "tokens_out",
            )
        }
        total["arbiter"] = js[0].get("arbiter")
        total["wall_ms"] = round(sum(j.get("wall_ms") or 0.0 for j in js), 1)
        error_kinds: dict[str, int] = {}
        for j in js:
            for k, n in (j.get("error_kinds") or {}).items():
                error_kinds[k] = error_kinds.get(k, 0) + n
        total["error_kinds"] = error_kinds
        return total

    def assert_complete(self, batch: IngestedBatch) -> dict:
        """Every ingested id must be matched, excepted, washed, or written off.

        This is the guard that makes the honesty claim checkable rather than
        rhetorical. A transaction the engine neither resolved nor flagged has
        silently vanished, and the merchant would have no way to know. Raising
        here converts that from an invisible data-loss bug into a failed run.
        """
        accounted: set[str] = set()
        for lr in self.legs.values():
            for d in (*lr.decisions.auto_confirmed, *lr.decisions.proposed):
                accounted.update(d.candidate.all_ids())
            for d in lr.decisions.exempted_immaterial:
                accounted.update(d.candidate.all_ids())
            for c in lr.wash_pairs:
                accounted.update(c.all_ids())
            for e in lr.exceptions.exceptions:
                accounted.update(e.txn_ids)
            for w in lr.exceptions.written_off:
                accounted.add(w["txn_id"])

        # Book entries reachable only through a settlement that itself has no bank
        # side still count as accounted: they appear on the books leg.
        all_ids = {t.txn_id for t in batch.all_txns}
        missing = sorted(all_ids - accounted)
        if missing:
            raise AssertionError(
                f"{len(missing)} transactions were neither resolved nor flagged: "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}. "
                "Every ingested line must land in exactly one bucket."
            )
        return {
            "ingested": len(all_ids),
            "accounted_for": len(accounted & all_ids),
            "complete": True,
        }

    def summary(self) -> dict:
        return {
            "scorer": self.scorer_name,
            "auto_confirm_threshold": round(self.threshold, 4),
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "timings_ms": self.timings_ms,
            "tier3": self.tier3_totals() or None,
            "legs": {
                name: {
                    "auto_confirmed": len(lr.decisions.auto_confirmed),
                    "proposed_for_human": len(lr.decisions.proposed),
                    "rejected": len(lr.decisions.rejected),
                    "immaterial": len(lr.decisions.exempted_immaterial),
                    "wash_pairs": len(lr.wash_pairs),
                    "exceptions": len(lr.exceptions.exceptions),
                    "written_off_immaterial": len(lr.exceptions.written_off),
                    **lr.stats,
                }
                for name, lr in self.legs.items()
            },
        }


class ReconciliationEngine:
    """Runs the tiers, in order, once per leg.

    ``max_tier`` caps which tiers run. It exists for the ablation table, and it
    caps *tiers* rather than raising the threshold to 1.01, which was the first
    attempt: a threshold above 1.0 also blocks tier 1's own score of exactly 1.0
    from confirming, so "tier 1 only" reported a 0% match rate for a tier that
    actually resolves most of the volume. An ablation that disables more than it
    claims to is worse than no ablation, because the number it produces looks
    like a finding.

    ``arbiter`` is what makes tier 3 run. It defaults to ``None``, meaning the tier
    does not run *at all* -- no API calls, no cost, and a result byte-identical to
    one produced without the tier existing. That default is deliberate: an
    arbitration tier that switches itself on because a credential happens to be
    present in the environment makes the same command mean two different things on
    two machines, with nothing in the output to say which one you got. It has to be
    asked for.
    """

    def __init__(
        self,
        scorer=None,
        *,
        threshold: float | None = None,
        max_tier: int = 3,
        arbiter=None,
        max_arbitrations: int = 25,
    ):
        self.scorer = scorer or DummyScorer()
        self.threshold = (
            threshold
            if threshold is not None
            else float(getattr(self.scorer, "threshold", 0.90))
        )
        self.max_tier = max_tier
        self.arbiter = arbiter
        self.max_arbitrations = max_arbitrations
        self.attributor = ResidualAttribution()

    # -- tier 3 ------------------------------------------------------------

    def _run_tier3(
        self,
        decisions: PolicyResult,
        candidates: list[Candidate],
        policy: Policy,
        *,
        rivals: list[Candidate] = (),
        repair=None,
    ) -> tuple[PolicyResult, Any]:
        """Arbitrate the ambiguous band, then re-run the policy. Returns both.

        ``rivals`` is where the alternatives come from, and finding the right source
        for them took a measurement. The obvious candidate -- ``decisions.rejected``
        under the ``uniqueness`` gate -- is **empty on both legs**, because tier 2's
        assignment and set-packing steps already enforce mutual exclusivity before
        the policy ever sees a candidate; the policy's uniqueness rule is a backstop
        that almost never fires. Offering the arbiter only what the policy left open
        therefore produces groups of exactly one, and a group of one asks the model
        to discriminate against nothing.

        The real alternatives are the solvers' own scored-but-not-selected
        hypotheses (``Tier2Result.considered``, the ``hungarian_rejected``
        candidates), which already carry full feature vectors. Those are the rivals.

        Nothing here decides anything. ``repolicy`` re-runs the same ``Policy``
        object over the same candidate list with adjusted scores substituted in, so
        the materiality gate, the uniqueness rule and the threshold all apply again
        exactly as they did the first time. There is no code path from an
        arbitration to a confirmation that skips them.
        """
        if self.max_tier < 3 or self.arbiter is None:
            return decisions, None

        from app.llm.tier3 import repolicy, run_tier3

        offered = [
            *decisions.proposed,
            *(d for d in decisions.rejected if d.gate == "uniqueness"),
            *(
                Decision(
                    candidate=c, action="reject",
                    reason="scored by the solver and not selected",
                    gate="uniqueness",
                )
                for c in rivals
                if c is not None
            ),
        ]
        report = run_tier3(
            offered,
            threshold=self.threshold,
            arbiter=self.arbiter,
            max_arbitrations=self.max_arbitrations,
            # Only hypotheses the solver actually selected may receive a delta.
            selectable={c.match_key for c in candidates},
        )
        if not report.arbitrations:
            return decisions, report
        return repolicy(report, policy, candidates, repair=repair), report

    # -- gateway -> bank ---------------------------------------------------

    def _run_gateway_bank(self, batch: IngestedBatch, as_of: date) -> LegRun:
        leg: LegConfig = LEGS["gateway_bank"]
        gateway = batch.gateway.settlements
        bank = batch.bank
        gross_lookup = {
            t.txn_id: (t.provenance or {}).get("gross_paise") or t.amount_paise + t.fees_paise + t.tax_paise
            for t in gateway
        }
        stats: dict = {}

        # Tier 1a: net out reversal/re-settlement pairs so tier 1b sees one credit.
        wash_pairs, washed = tier1.detect_wash_pairs(bank)
        stats["wash_pairs"] = len(wash_pairs)

        # Tier 1b: exact reference + amount, refusing when ambiguous.
        t1 = tier1.match_exact_reference(gateway, bank, leg, exclude_bank=frozenset(washed))
        stats["tier1"] = t1.stats

        # Tier 2a: blocked graph over the remainder, optimal 1:1 assignment.
        remaining_gateway = [t for t in gateway if t.txn_id not in t1.consumed_left]
        remaining_bank = [
            t for t in bank if t.txn_id not in t1.consumed_right and t.txn_id not in washed
        ]
        graph = build_graph(remaining_gateway, remaining_bank, leg, right_sign=1)
        stats["blocking"] = graph.stats

        if self.max_tier < 2:
            t2_pairs = tier2.Tier2Result(stats={"skipped": "max_tier=1"})
            t2_sub = tier2.Tier2Result(stats={"skipped": "max_tier=1"})
            stats["hungarian"] = t2_pairs.stats
            stats["subset_sum"] = t2_sub.stats
        else:
            scored = tier2.score_pair_edges(graph, self.scorer, gross_lookup=gross_lookup)
            t2_pairs = tier2.solve_assignment(graph, scored)
            stats["hungarian"] = t2_pairs.stats

            # Tier 2b: split settlements -- one payout arriving as several credits.
            claimed = {i for c in t2_pairs.candidates for i in c.all_ids()}
            pools, pool_stats = build_subset_pools(
                remaining_bank, remaining_gateway, leg,
                exclude_members=frozenset(claimed | washed),
                exclude_anchors=frozenset(claimed),
                slack_lookup={t.txn_id: 1 for t in remaining_gateway},  # exact credits
                anchor_is_left=True,
            )
            stats["subset_pools"] = pool_stats
            t2_sub = tier2.solve_subsets(
                pools,
                {t.txn_id: t for t in remaining_bank},
                {t.txn_id: t for t in remaining_gateway},
                leg, self.scorer,
                gross_lookup=gross_lookup,
                anchor_is_left=True,
            )
            stats["subset_sum"] = t2_sub.stats

        candidates = [*t1.matches, *t2_pairs.candidates, *t2_sub.candidates]
        policy = Policy(auto_confirm_threshold=self.threshold)
        decisions = policy.decide_many(candidates)

        # Tier 3, before anything downstream reads the decisions. Placing it here
        # rather than after the exception build is what makes the tier real: the
        # exception list, the completeness assertion and the journal postings all
        # derive from ``decisions``, so arbitrating afterwards would adjust scores
        # nothing ever looks at again.
        decisions, tier3 = self._run_tier3(
            decisions, candidates, policy,
            rivals=[*t2_pairs.considered, *t2_sub.considered],
        )
        if tier3 is not None:
            stats["tier3"] = tier3.to_json()

        resolved_left = {i for d in (*decisions.auto_confirmed, *decisions.proposed)
                         for i in d.candidate.left_ids}
        resolved_right = {i for d in (*decisions.auto_confirmed, *decisions.proposed)
                          for i in d.candidate.right_ids} | washed

        # The exception builder needs a graph covering *everything*, not just what
        # tier 2 saw, or lines resolved by tier 1 would look unaccounted-for.
        full_graph = BlockingGraph(leg=leg, left=gateway, right=bank)
        full_graph.left_by_id = {t.txn_id: t for t in gateway}
        full_graph.right_by_id = {t.txn_id: t for t in bank}
        full_graph.edges = graph.edges
        full_graph.reverse = graph.reverse
        full_graph.stats = graph.stats

        exceptions = build_exceptions(
            "gateway_bank", full_graph,
            [*decisions.auto_confirmed, *decisions.proposed, *decisions.rejected,
             *t2_pairs.considered_as_decisions()],
            policy=policy,
            resolved_left=resolved_left,
            resolved_right=resolved_right,
            unresolved=set(),
            as_of=as_of,
        )
        # Wash pairs are flagged, but as a self-resolving item with no work
        # attached. They belong on the list -- a duplicate payout the merchant
        # never sees is exactly the thing reconciliation exists to surface -- but
        # ranking them alongside real breaks would be noise.
        for c in wash_pairs:
            exceptions.exceptions.append(
                Exception_(
                    exception_id="wash-" + c.left_ids[0].split(":")[-1][:10],
                    leg="gateway_bank",
                    txn_ids=(*c.left_ids, *c.right_ids),
                    category="reversal_wash_pair",
                    max_confidence=1.0,
                    suggested_action="no action required: a credit and its reversal "
                                     "cancel exactly; retained on the list so the "
                                     "duplicate payout attempt is visible",
                    amount_paise=abs(c.left_amount_paise),
                    age_days=0,
                    evidence=c.evidence,
                )
            )
        exceptions.stats["by_category"] = {
            **exceptions.stats.get("by_category", {}),
            "reversal_wash_pair": len(wash_pairs),
        }

        return LegRun(
            leg="gateway_bank", graph=full_graph, decisions=decisions,
            exceptions=exceptions, all_candidates=candidates,
            considered=[*t2_pairs.considered, *t2_sub.considered],
            wash_pairs=wash_pairs, deferred=t1.deferred, tier3=tier3, stats=stats,
        )

    # -- books -> gateway --------------------------------------------------

    def _run_gateway_books(self, batch: IngestedBatch, as_of: date) -> LegRun:
        leg: LegConfig = LEGS["gateway_books"]
        books = batch.books.entries
        gateway = batch.gateway.settlements
        gross_lookup = {
            t.txn_id: (t.provenance or {}).get("gross_paise") or t.amount_paise + t.fees_paise + t.tax_paise
            for t in gateway
        }
        stats: dict = {}

        # Tier 1: the order-key join.
        t1 = tier1.match_order_keys(
            books, gateway, batch.gateway.order_to_settlement, leg,
            unsettled_orders=frozenset(batch.gateway.unsettled_orders),
        )
        stats["tier1"] = t1.stats

        # What tier 1 taught us: the merchant's own blended fee assumption,
        # recovered from the pairs it proved. See expectations.py.
        fee_bps, fee_detail = estimate_books_fee_bps(t1.matches, batch.gateway, batch.books)
        stats["merchant_fee_estimate"] = fee_detail

        # Tier 2a: per-component assignment. Each unclaimed payment/refund becomes a
        # predicted book amount; those predictions are matched 1:1 against unkeyed
        # entries with a tight band. This is where most of the leg's volume resolves.
        claimed_components = claimed_component_ids(
            batch.gateway, batch.books, set(t1.consumed_left)
        )
        predictions = expected_components(
            batch.gateway, batch.books,
            claimed_component_ids=claimed_components, fee_bps=fee_bps,
        )
        unkeyed = [t for t in books if t.txn_id not in t1.consumed_left]
        comp_graph = build_graph(
            unkeyed, predictions, GATEWAY_BOOKS_COMPONENT, right_sign=0
        )
        stats["component_blocking"] = comp_graph.stats

        if self.max_tier < 2:
            # Tier 1 only: the order-key join and nothing else. Everything it did
            # not claim becomes an exception, which is exactly the point of the
            # ablation -- it shows how much of the long tail a deterministic join
            # leaves on the table.
            stats["component_assignment"] = {"skipped": "max_tier=1"}
            stats["subset_sum"] = {"skipped": "max_tier=1"}
            policy_1 = Policy(auto_confirm_threshold=self.threshold)
            decisions_1 = _repair_books_uniqueness(
                policy_1.decide_many(list(t1.matches)), policy_1
            )
            graph_1 = BlockingGraph(leg=leg, left=books, right=gateway)
            graph_1.left_by_id = {t.txn_id: t for t in books}
            graph_1.right_by_id = {t.txn_id: t for t in gateway}
            resolved_1 = {
                i for d in (*decisions_1.auto_confirmed, *decisions_1.proposed)
                for i in d.candidate.left_ids
            }
            exceptions_1 = build_exceptions(
                "gateway_books", graph_1,
                [*decisions_1.auto_confirmed, *decisions_1.proposed,
                 *decisions_1.rejected],
                policy=policy_1, resolved_left=resolved_1,
                resolved_right=set(graph_1.right_by_id), unresolved=set(), as_of=as_of,
            )
            return LegRun(
                leg="gateway_books", graph=graph_1, decisions=decisions_1,
                exceptions=exceptions_1, all_candidates=list(t1.matches),
                deferred=t1.deferred, stats=stats,
            )

        comp_scored = tier2.score_pair_edges(
            comp_graph, self.scorer,
            gross_lookup={
                t.txn_id: (t.provenance or {}).get("gross_paise", t.amount_paise)
                for t in predictions
            },
        )
        t2_comp = tier2.solve_assignment(comp_graph, comp_scored, no_match_prior=0.12)
        # Re-target: a match against a *predicted component* is really a match
        # against the settlement that predicted it.
        comp_candidates = [
            _retarget_to_settlement(c, predictions) for c in t2_comp.candidates
        ]
        comp_candidates = [c for c in comp_candidates if c is not None]
        stats["component_assignment"] = t2_comp.stats

        # Tier 2b: subset-sum for the residue -- accrual and adjustment lines with no
        # component counterpart at all (a withholding accrual, an FX revaluation).
        # Small pools by construction, because tiers 1 and 2a have taken everything
        # that had a counterpart. Expectations are rebuilt here rather than reused:
        # a target still sized for what tier 2a already claimed would send this
        # search hunting for money that is accounted for, and a loose enough
        # tolerance always finds something.
        entry_by_id = {t.txn_id: t for t in books}
        claimed_entries_by_settlement: dict[str, list[NormalizedTxn]] = {}
        for c in (*t1.matches, *comp_candidates):
            entry = entry_by_id.get(c.left_ids[0])
            if entry is not None:
                claimed_entries_by_settlement.setdefault(c.right_ids[0], []).append(entry)
            cid = (c.evidence or {}).get("component_id")
            if cid:
                claimed_components.add(cid)

        expectations = build_expectations(
            batch.gateway, batch.books,
            claimed_component_ids=claimed_components,
            claimed_entries_by_settlement=claimed_entries_by_settlement,
            fee_bps=fee_bps,
        )
        stats["expectations"] = {
            "settlements": len(expectations),
            "with_date_hints": sum(1 for e in expectations.values() if e.date_hints),
        }
        claimed_after_2a = set(t1.consumed_left) | {
            i for c in comp_candidates for i in c.left_ids
        }
        residual_targets = {sid: e.target_paise for sid, e in expectations.items()}
        slack_lookup = {sid: e.tolerance_paise for sid, e in expectations.items()}
        date_hints = {sid: e.date_hints for sid, e in expectations.items() if e.date_hints}
        stats["residual_after_2a"] = {
            "settlements_with_nonzero_target": sum(1 for v in residual_targets.values() if v),
            "entries_still_unclaimed": len(books) - len(claimed_after_2a),
        }
        pools, pool_stats = build_subset_pools(
            books, gateway, leg,
            exclude_members=frozenset(claimed_after_2a),
            residual_targets=residual_targets,
            slack_lookup=slack_lookup,
            date_hints=date_hints,
            anchor_is_left=False,
        )
        stats["subset_pools"] = pool_stats

        t2_sub = tier2.solve_subsets(
            pools,
            {t.txn_id: t for t in books},
            {t.txn_id: t for t in gateway},
            leg, self.scorer,
            gross_lookup=gross_lookup,
            anchor_is_left=False,
            attributor=self.attributor,
        )
        stats["subset_sum"] = t2_sub.stats

        candidates = [*t1.matches, *comp_candidates, *t2_sub.candidates]
        policy = Policy(auto_confirm_threshold=self.threshold)
        # Uniqueness on the books leg is per *entry*, not per settlement: a
        # settlement legitimately has many book entries, so the constraint runs on
        # the left side only.
        decisions = policy.decide_many(candidates)
        # The policy's right-side uniqueness rule would reject every keyed entry
        # after the first for a given settlement, so it is re-applied here with the
        # correct notion of exclusivity for this leg.
        decisions = _repair_books_uniqueness(decisions, policy)

        # Tier 3. The repair has to run again after re-policying, or an arbitrated
        # book entry is thrown out by the same right-side uniqueness rule that does
        # not apply to this leg -- which would look like the arbiter's proposal
        # being rejected on the evidence when it was rejected on a technicality.
        decisions, tier3 = self._run_tier3(
            decisions, candidates, policy,
            rivals=[
                *(_retarget_or_none(c, predictions) for c in t2_comp.considered),
                *t2_sub.considered,
            ],
            repair=_repair_books_uniqueness,
        )
        if tier3 is not None:
            stats["tier3"] = tier3.to_json()

        resolved_left = {i for d in (*decisions.auto_confirmed, *decisions.proposed)
                         for i in d.candidate.left_ids}
        resolved_right = {i for d in (*decisions.auto_confirmed, *decisions.proposed)
                          for i in d.candidate.right_ids}

        graph = BlockingGraph(leg=leg, left=books, right=gateway)
        graph.left_by_id = {t.txn_id: t for t in books}
        graph.right_by_id = {t.txn_id: t for t in gateway}
        graph.stats = pool_stats

        exceptions = build_exceptions(
            "gateway_books", graph,
            [*decisions.auto_confirmed, *decisions.proposed, *decisions.rejected],
            policy=policy,
            resolved_left=resolved_left,
            # The settlement side of this leg is not an independent fact to flag:
            # it is already evaluated on the gateway->bank leg, and reporting it
            # twice would double-count the same money in the exception list.
            resolved_right=set(graph.right_by_id),
            unresolved=set(),
            as_of=as_of,
        )
        return LegRun(
            leg="gateway_books", graph=graph, decisions=decisions,
            exceptions=exceptions, all_candidates=candidates,
            considered=[
                *(_retarget_or_none(c, predictions) for c in t2_comp.considered),
                *t2_sub.considered,
            ],
            deferred=t1.deferred, tier3=tier3, stats=stats,
        )

    # -- entry point -------------------------------------------------------

    def run(self, batch: IngestedBatch, *, as_of: date | None = None) -> ReconciliationRun:
        as_of = as_of or max(
            (t.value_date for t in batch.all_txns), default=date.today()
        )
        timings: dict = {}

        t0 = time.perf_counter()
        bank_leg = self._run_gateway_bank(batch, as_of)
        timings["gateway_bank"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        books_leg = self._run_gateway_books(batch, as_of)
        timings["gateway_books"] = round((time.perf_counter() - t0) * 1000, 1)

        return ReconciliationRun(
            legs={"gateway_bank": bank_leg, "gateway_books": books_leg},
            batch_summary=batch.summary(),
            timings_ms=timings,
            scorer_name=getattr(self.scorer, "name", type(self.scorer).__name__),
            threshold=self.threshold,
            as_of=as_of,
        )


def _retarget_or_none(c, predictions: list[NormalizedTxn]):
    """``_retarget_to_settlement`` but tolerant, for the training-only pool."""
    out = _retarget_to_settlement(c, predictions)
    return out if out is not None else c


def _retarget_to_settlement(c, predictions: list[NormalizedTxn]):
    """Rewrite a candidate matched against a predicted component onto its settlement.

    Tier 2a matches a book entry against a *prediction* -- a synthetic transaction
    standing in for one payment inside a payout. The fact worth recording is that
    the entry belongs to the settlement, so the component id moves into the
    evidence and the settlement id becomes the candidate's right-hand side.

    Doing this at the boundary rather than inside the solver keeps the synthetic
    transactions from leaking into the audit trail, where a ``pay_...`` id on a
    books-leg match would be genuinely confusing to a reviewer six months later.
    """
    by_id = {t.txn_id: t for t in predictions}
    pred = by_id.get(c.right_ids[0])
    if pred is None:
        return None
    sid = (pred.provenance or {}).get("settlement_txn_id")
    if not sid:
        return None
    from dataclasses import replace

    return replace(
        c,
        right_ids=(sid,),
        algorithm="component_assignment",
        right_amount_paise=pred.amount_paise,
        evidence={
            **c.evidence,
            "rule": "book entry matched the amount predicted for one payment inside "
                    "this payout, derived from the gateway recon report and the fee "
                    "ratio recovered from tier 1",
            "component_id": (pred.provenance or {}).get("component_id"),
            "component_kind": (pred.provenance or {}).get("component_kind"),
            "component_gross_paise": (pred.provenance or {}).get("gross_paise"),
            "predicted_book_amount_paise": pred.amount_paise,
            "merchant_fee_bps_applied": (pred.provenance or {}).get("fee_bps_applied"),
        },
    )


def _repair_books_uniqueness(decisions: PolicyResult, policy: Policy) -> PolicyResult:
    """Re-admit candidates the generic uniqueness rule wrongly rejected on this leg.

    ``Policy`` enforces that no two candidates claim the same right-hand id, which
    is correct for gateway->bank (one settlement, one bank credit) and wrong for
    books->gateway (one settlement, many invoices). Rather than parameterise the
    policy with a per-leg exclusivity flag -- which would put leg-specific
    knowledge inside the rule engine -- the leg fixes it up here, where the domain
    fact that "a settlement has many book entries" actually lives.
    """
    out = PolicyResult(
        auto_confirmed=list(decisions.auto_confirmed),
        proposed=list(decisions.proposed),
        rejected=[],
        exempted_immaterial=list(decisions.exempted_immaterial),
    )
    claimed_left = {
        i for d in (*out.auto_confirmed, *out.proposed) for i in d.candidate.left_ids
    }
    for d in sorted(decisions.rejected, key=lambda x: -x.candidate.score):
        if d.gate != "uniqueness":
            out.rejected.append(d)
            continue
        if claimed_left & set(d.candidate.left_ids):
            out.rejected.append(d)
            continue
        claimed_left.update(d.candidate.left_ids)
        if d.candidate.tier == 1 or d.candidate.score >= policy.auto_confirm_threshold:
            from .policy import _gate_applies

            if _gate_applies(d.candidate):
                out.proposed.append(
                    Decision(
                        candidate=d.candidate, action="propose",
                        reason="above the materiality gate; human approval required",
                        gate="materiality",
                    )
                )
            else:
                out.auto_confirmed.append(
                    Decision(
                        candidate=d.candidate, action="auto_confirm",
                        reason="left-side exclusive on the books leg; score clears threshold",
                        gate="threshold",
                    )
                )
        else:
            out.proposed.append(
                Decision(
                    candidate=d.candidate, action="propose",
                    reason=f"score {d.candidate.score:.4f} below threshold",
                    gate="threshold",
                )
            )
    out.stats = {
        "considered": decisions.stats.get("considered", 0),
        "auto_confirmed": len(out.auto_confirmed),
        "proposed": len(out.proposed),
        "rejected": len(out.rejected),
        "immaterial": len(out.exempted_immaterial),
    }
    return out
