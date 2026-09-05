"""Tier 3: the bounds, and that they are code rather than requests.

The tests are split by what they protect:

* **Selection** -- what the arbiter is allowed to see. An item above the materiality
  gate must never reach a payload, and the group it does see must contain a real
  choice.
* **Clamping and validation** -- what the arbiter is allowed to return. A rogue
  answer must be bounded by arithmetic, not by the prompt.
* **The gate, end to end** -- that no arbitration, however confident, can move a
  material item into auto-confirmed. This is the one that matters; the others are
  how it is achieved.

Nothing here touches the network. The live-API test at the bottom skips unless a key
is present, so the suite is fully deterministic by default.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.config import MATERIALITY_PAISE
from app.llm.arbiter import (
    ARBITRATION_SCHEMA,
    MAX_CONFIDENCE_DELTA,
    MODEL,
    TIER3_FLOOR,
    Arbitration,
    ClaudeArbiter,
    NullArbiter,
    OfflineArbiter,
    _clamp,
    build_payload,
    default_arbiter,
)
from app.llm.tier3 import (
    apply_arbitration,
    repolicy,
    run_tier3,
    select_for_arbitration,
    substitute,
)
from app.matching.policy import Policy
from app.matching.types import Candidate, Decision

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures: hand-built candidates, so a test's premise is visible in the test
# ---------------------------------------------------------------------------

def cand(
    *,
    right: str = "gw:setl_1",
    left: str = "bank:BR1",
    score: float = 0.5,
    amount: int = 10_000_00,
    tier: int = 2,
    algorithm: str = "hungarian_assignment",
) -> Candidate:
    return Candidate(
        leg="gateway_bank", left_ids=(left,), right_ids=(right,), tier=tier,
        algorithm=algorithm, score=score,
        features={"utr_exact": 0.0, "utr_similarity": 0.6, "amount_exact": 1.0},
        left_amount_paise=amount, right_amount_paise=amount,
        evidence={"rule": "test"},
    )


def proposed(c: Candidate) -> Decision:
    return Decision(candidate=c, action="propose", reason="test", gate="threshold")


def rival(c: Candidate) -> Decision:
    return Decision(candidate=c, action="reject", reason="test", gate="uniqueness")


class FixedArbiter:
    """Returns whatever it was constructed with. For testing the bounds."""

    name = "fixed_test_arbiter"
    available = True

    def __init__(self, **kw):
        self.kw = kw

    def arbitrate(self, group, context):
        raw = self.kw.get("confidence_delta", 0.0)
        delta, clamped = _clamp(raw)
        return Arbitration(
            match_key=group[0].match_key, arbiter=self.name,
            action=self.kw.get("action", "propose_match"),
            chosen_index=self.kw.get("chosen_index", 1),
            confidence_delta=delta, rationale="fixed", raw_delta=raw, clamped=clamped,
        )


# ---------------------------------------------------------------------------
# Selection: what the arbiter is allowed to see
# ---------------------------------------------------------------------------

def test_material_items_never_reach_the_payload():
    """The gate is a filter on selection, not an instruction in the prompt."""
    big = cand(right="gw:big", amount=MATERIALITY_PAISE, score=0.5)
    small = cand(right="gw:small", amount=1_000_00, score=0.5)

    groups, stats = select_for_arbitration(
        [proposed(big), proposed(small)], threshold=0.9
    )

    assert stats["above_materiality"] == 1
    offered = {c.right_ids[0] for g in groups for c in g}
    assert "gw:big" not in offered
    assert "gw:small" in offered


def test_exactly_at_the_materiality_gate_is_excluded():
    """The comparison is >=, so an item exactly on the line is held, not offered."""
    on_the_line = cand(amount=MATERIALITY_PAISE, score=0.5)
    groups, stats = select_for_arbitration([proposed(on_the_line)], threshold=0.9)
    assert stats["above_materiality"] == 1
    assert groups == []


@pytest.mark.parametrize(
    "score,expected_in_band",
    [
        (TIER3_FLOOR - 0.01, False),   # below the floor: nothing to reason with
        (TIER3_FLOOR, True),           # the floor itself is inclusive
        (0.5, True),
        (0.899, True),
        (0.9, False),                  # at the threshold: policy already confirms it
        (0.95, False),
    ],
)
def test_band_is_floor_inclusive_threshold_exclusive(score, expected_in_band):
    groups, stats = select_for_arbitration(
        [proposed(cand(score=score, amount=1_000_00))], threshold=0.9
    )
    assert bool(groups) is expected_in_band
    assert stats["outside_band"] == (0 if expected_in_band else 1)


def test_rivals_make_the_group_a_choice():
    """The regression test for the defect this wiring existed to fix.

    Candidates the policy left open have already won their subject outright, so
    grouping them alone yields groups of one -- and an arbiter shown one option is
    not discriminating, it is rationalising. The solver's discarded hypotheses are
    what make the group a choice.
    """
    winner = cand(left="bank:A", score=0.52, amount=1_000_00)
    loser = cand(left="bank:B", score=0.48, amount=1_000_00)  # same right id

    alone, stats_alone = select_for_arbitration([proposed(winner)], threshold=0.9)
    assert [len(g) for g in alone] == [1]
    assert stats_alone["singleton_groups"] == 1

    withrival, stats = select_for_arbitration(
        [proposed(winner), rival(loser)], threshold=0.9
    )
    assert [len(g) for g in withrival] == [2]
    assert stats["singleton_groups"] == 0
    # Highest score first, so ``chosen_index`` 1 means "the solver's pick".
    assert withrival[0][0].score > withrival[0][1].score


def test_a_group_of_pure_also_rans_is_not_arbitrated():
    """Money the policy settled elsewhere is not re-opened."""
    only_losers = [rival(cand(left="bank:B", score=0.4, amount=1_000_00))]
    groups, _ = select_for_arbitration(only_losers, threshold=0.9)
    assert groups == []


def test_the_cap_is_reported_not_silent():
    ds = [
        proposed(cand(right=f"gw:{i}", score=0.5, amount=1_000_00))
        for i in range(10)
    ]
    groups, stats = select_for_arbitration(ds, threshold=0.9, max_arbitrations=4)
    assert len(groups) == 4
    assert stats["capped"] == 6
    assert stats["eligible"] == 10


def test_hardest_groups_are_arbitrated_first():
    """The budget goes where a tie-break is worth most, not to the highest score."""
    near_tie = [
        proposed(cand(right="gw:tie", left="bank:A", score=0.50, amount=1_000_00)),
        rival(cand(right="gw:tie", left="bank:B", score=0.49, amount=1_000_00)),
    ]
    clear = [
        proposed(cand(right="gw:clear", left="bank:C", score=0.80, amount=1_000_00)),
        rival(cand(right="gw:clear", left="bank:D", score=0.20, amount=1_000_00)),
    ]
    groups, _ = select_for_arbitration(clear + near_tie, threshold=0.9, max_arbitrations=1)
    assert groups[0][0].right_ids[0] == "gw:tie"


def test_payload_excludes_material_items_and_names_its_own_bounds():
    group = [cand(score=0.5, amount=1_000_00)]
    payload = build_payload(group, {})
    assert payload["candidates"][0]["calibrated_confidence"] == 0.5
    assert str(MAX_CONFIDENCE_DELTA) in payload["policy_note"]
    # The payload is curated, not a dump: only whitelisted features travel.
    assert set(payload["candidates"][0]["features"]) <= {
        "utr_exact", "utr_similarity", "utr_in_narration", "amount_exact",
        "amount_rel_delta", "within_fee_band", "date_delta_abs", "left_ambiguity",
        "right_ambiguity", "amount_uniqueness", "narration_cosine",
    }


# ---------------------------------------------------------------------------
# Clamping and validation: what the arbiter is allowed to return
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected,clamped",
    [
        (0.0, 0.0, False),
        (0.1, 0.1, False),
        (MAX_CONFIDENCE_DELTA, MAX_CONFIDENCE_DELTA, False),
        (0.9, MAX_CONFIDENCE_DELTA, True),
        (5.0, MAX_CONFIDENCE_DELTA, True),
        (-0.9, -MAX_CONFIDENCE_DELTA, True),
        (1e9, MAX_CONFIDENCE_DELTA, True),
    ],
)
def test_clamp_bounds_any_answer(raw, expected, clamped):
    got, was_clamped = _clamp(raw)
    assert got == pytest.approx(expected)
    assert was_clamped is clamped


def test_arbitration_cannot_push_a_score_out_of_range():
    group = [cand(score=0.95, amount=1_000_00)]
    a = Arbitration(
        match_key=group[0].match_key, arbiter="t", action="propose_match",
        chosen_index=1, confidence_delta=MAX_CONFIDENCE_DELTA, rationale="",
    )
    out = apply_arbitration(group, a)
    assert out[0].score <= 1.0
    assert out[0].tier == 3
    assert out[0].evidence["tier3_score_before"] == 0.95


def test_arbitration_records_what_it_changed():
    group = [cand(score=0.40, amount=1_000_00)]
    a = Arbitration(
        match_key=group[0].match_key, arbiter="t", action="propose_match",
        chosen_index=1, confidence_delta=0.20, rationale="because",
        evidence_cited=["utr_similarity"], raw_delta=0.20,
    )
    out = apply_arbitration(group, a)
    ev = out[0].evidence
    assert ev["tier3_score_before"] == 0.40
    assert ev["tier3_score_after"] == pytest.approx(0.60)
    assert ev["tier3_rationale"] == "because"
    assert ev["tier3_evidence_cited"] == ["utr_similarity"]


def test_insufficient_evidence_changes_nothing():
    group = [cand(score=0.5, amount=1_000_00)]
    a = Arbitration(
        match_key=group[0].match_key, arbiter="t",
        action="insufficient_evidence", chosen_index=0, confidence_delta=0.0,
        rationale="",
    )
    out = apply_arbitration(group, a)
    assert out[0] is group[0]


@pytest.mark.parametrize("bad_index", [0, 2, -1, 99])
def test_out_of_range_candidate_index_is_refused(bad_index):
    """The schema guarantees an integer. It does not guarantee a valid one."""
    group = [cand(score=0.5, amount=1_000_00)]
    a = Arbitration(
        match_key=group[0].match_key, arbiter="t", action="propose_match",
        chosen_index=bad_index, confidence_delta=0.25, rationale="",
    )
    out = apply_arbitration(group, a)
    assert out[0] is group[0], "an out-of-range index must move nothing"


def test_choosing_a_rival_is_recorded_and_not_applied():
    """An unselected hypothesis cannot be promoted by arbitration.

    It was never part of the mutually-consistent set the packing step produced, so
    confirming it could conflict with a match confirmed elsewhere. The disagreement
    is a reported number rather than a silent no-op.
    """
    winner = cand(left="bank:A", score=0.52, amount=1_000_00)
    loser = cand(left="bank:B", score=0.48, amount=1_000_00)

    report = run_tier3(
        [proposed(winner), rival(loser)],
        threshold=0.9,
        arbiter=FixedArbiter(chosen_index=2, confidence_delta=0.25),
        selectable={winner.match_key},
    )
    a = report.arbitrations[0]
    assert a.action == "insufficient_evidence"
    assert a.error == "chose_unselected_rival"
    assert report.stats["chose_unselected_rival"] == 1
    assert all(c.tier != 3 for c in report.adjusted)


# ---------------------------------------------------------------------------
# The gate, end to end: the bound that actually matters
# ---------------------------------------------------------------------------

def test_no_arbitration_can_auto_confirm_a_material_item():
    """The whole safety claim, in one test.

    A material candidate is fed straight past selection into the re-policy step with
    the maximum possible delta applied, i.e. the arbiter is assumed to have been
    fully compromised. It must still not auto-confirm.
    """
    big = cand(amount=MATERIALITY_PAISE * 2, score=0.66)
    policy = Policy(auto_confirm_threshold=0.6675)

    arbitrated = replace(big, score=1.0, tier=3)
    result = policy.decide_many([arbitrated])

    assert result.auto_confirmed == []
    assert len(result.proposed) == 1
    assert result.proposed[0].gate == "materiality"


def test_repolicy_reapplies_the_gate_to_arbitrated_candidates():
    big = cand(right="gw:big", amount=MATERIALITY_PAISE * 2, score=0.5)
    small = cand(right="gw:small", amount=1_000_00, score=0.5)
    candidates = [big, small]
    policy = Policy(auto_confirm_threshold=0.6675)

    report = run_tier3(
        [proposed(big), proposed(small)],
        threshold=0.6675,
        arbiter=FixedArbiter(confidence_delta=MAX_CONFIDENCE_DELTA),
        selectable={c.match_key for c in candidates},
    )
    out = repolicy(report, policy, candidates)

    # The material one was never offered, so it is untouched and still held.
    assert report.skipped_above_materiality == 1
    confirmed_ids = {i for d in out.auto_confirmed for i in d.candidate.right_ids}
    assert "gw:big" not in confirmed_ids
    gates = {d.gate for d in out.proposed if "gw:big" in d.candidate.right_ids}
    assert gates == {"materiality"}


def test_substitute_preserves_the_full_candidate_set():
    """Re-policy must see every candidate, not just the arbitrated ones.

    ``Policy``'s uniqueness rule is a statement about the whole set. Evaluating it
    over a slice asks a different question, and a candidate the full set ruled out
    can be the best thing in a three-element slice.
    """
    a = cand(right="gw:1", left="bank:A", score=0.5, amount=1_000_00)
    b = cand(right="gw:2", left="bank:B", score=0.5, amount=1_000_00)
    c = cand(right="gw:3", left="bank:C", score=0.5, amount=1_000_00)

    report = run_tier3(
        [proposed(a)], threshold=0.9,
        arbiter=FixedArbiter(confidence_delta=0.2),
        selectable={a.match_key, b.match_key, c.match_key},
    )
    merged = substitute([a, b, c], report)

    assert len(merged) == 3, "no candidate may be dropped by substitution"
    by_right = {m.right_ids[0]: m for m in merged}
    assert by_right["gw:1"].score == pytest.approx(0.7)
    assert by_right["gw:1"].tier == 3
    assert by_right["gw:2"] is b, "an unarbitrated candidate must pass through as-is"
    assert by_right["gw:3"] is c


# ---------------------------------------------------------------------------
# The arbiters themselves
# ---------------------------------------------------------------------------

def test_null_arbiter_degrades_to_a_human_never_a_guess():
    group = [cand(score=0.5)]
    a = NullArbiter().arbitrate(group, {})
    assert a.action == "insufficient_evidence"
    assert a.confidence_delta == 0.0
    assert "ANTHROPIC_API_KEY" in a.reason


def test_offline_arbiter_is_tagged_so_it_cannot_pass_for_a_model():
    """A stub that silently stands in for a model makes every number a fiction."""
    a = OfflineArbiter().arbitrate([cand(score=0.5)], {})
    assert a.arbiter == "offline_rule_arbiter"
    assert "claude" not in a.arbiter.lower()


def test_offline_arbiter_calls_a_near_tie_a_tie():
    group = [
        cand(left="bank:A", score=0.51),
        cand(left="bank:B", score=0.50),
    ]
    a = OfflineArbiter().arbitrate(group, {})
    assert a.action == "insufficient_evidence"
    assert "tie" in a.reason


def test_auto_mode_does_not_fall_back_to_the_rule_arbiter(monkeypatch):
    """Same command, same meaning. A missing key must not silently swap engines."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(default_arbiter("auto"), NullArbiter)
    assert isinstance(default_arbiter("offline"), OfflineArbiter)


def test_default_arbiter_modes_are_distinct():
    assert default_arbiter("null").name == "null_arbiter"
    assert default_arbiter("offline").name == "offline_rule_arbiter"
    assert default_arbiter("claude").name == "claude_arbiter"


# ---------------------------------------------------------------------------
# The API contract, without the network
# ---------------------------------------------------------------------------

class _StubMessages:
    def __init__(self, outer, payload, stop_reason="end_turn"):
        self.outer = outer
        self.payload = payload
        self.stop_reason = stop_reason

    def create(self, **kw):
        self.outer.calls.append(kw)

        class _Block:
            type = "text"
            text = json.dumps(self.payload)

        class _Usage:
            input_tokens = 100
            output_tokens = 20

        class _Resp:
            content = [_Block()]
            stop_reason = self.stop_reason
            usage = _Usage()

        return _Resp()


class StubClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.calls: list[dict] = []
        self.messages = _StubMessages(self, payload, stop_reason)


def _claude_with(stub) -> ClaudeArbiter:
    arb = ClaudeArbiter()
    arb._client = stub
    return arb


def test_claude_arbiter_sends_structured_output_and_the_current_model():
    """A guard against silent API drift.

    If the model id or the structured-output shape ever regresses to something the
    API no longer accepts, every arbitration degrades to ``insufficient_evidence``
    and the tier looks like it simply had no opinions. This asserts the request.
    """
    stub = StubClient({
        "action": "insufficient_evidence", "candidate_index": 0,
        "confidence_delta": 0.0, "rationale": "", "evidence_cited": [], "reason": "tie",
    })
    _claude_with(stub).arbitrate([cand(score=0.5)], {})

    sent = stub.calls[0]
    assert sent["model"] == MODEL == "claude-opus-5"
    fmt = sent["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] is ARBITRATION_SCHEMA
    assert fmt["schema"]["additionalProperties"] is False
    # Structured output, not a forced tool call: the response cannot be prose.
    assert "tools" not in sent and "tool_choice" not in sent


def test_claude_arbiter_clamps_a_rogue_delta():
    stub = StubClient({
        "action": "propose_match", "candidate_index": 1, "confidence_delta": 9.9,
        "rationale": "very sure", "evidence_cited": ["utr_exact"], "reason": "",
    })
    a = _claude_with(stub).arbitrate([cand(score=0.5)], {})
    assert a.confidence_delta == MAX_CONFIDENCE_DELTA
    assert a.raw_delta == 9.9
    assert a.clamped is True


def test_claude_arbiter_refuses_an_out_of_range_index():
    stub = StubClient({
        "action": "propose_match", "candidate_index": 7, "confidence_delta": 0.2,
        "rationale": "", "evidence_cited": [], "reason": "",
    })
    a = _claude_with(stub).arbitrate([cand(score=0.5)], {})
    assert a.action == "insufficient_evidence"
    assert a.error == "index_out_of_range"


def test_truncation_is_named_not_reported_as_a_parse_failure():
    """A budget set too low would apply to every call in the run.

    Both paths end in ``insufficient_evidence``, but "the arbiter never finished
    answering" is a configuration bug and "the arbiter said something unreadable" is
    a one-off. Conflating them hides the first behind the second.
    """
    stub = StubClient({"action": "propose_match"}, stop_reason="max_tokens")
    a = _claude_with(stub).arbitrate([cand(score=0.5)], {})
    assert a.action == "insufficient_evidence"
    assert a.error == "max_tokens_truncated"
    assert a.usage == {"input_tokens": 100, "output_tokens": 20}, (
        "a truncated call still costs money and must still report its usage"
    )


def test_refusal_is_named():
    stub = StubClient({}, stop_reason="refusal")
    a = _claude_with(stub).arbitrate([cand(score=0.5)], {})
    assert a.action == "insufficient_evidence"
    assert a.error == "refusal"


def test_unreadable_response_is_named():
    class BadStub(StubClient):
        def __init__(self):
            super().__init__({})
            self.messages.create = self._create

        def _create(self, **kw):
            class _Block:
                type = "text"
                text = "{not json"

            class _Resp:
                content = [_Block()]
                stop_reason = "end_turn"
                usage = None

            return _Resp()

    a = _claude_with(BadStub()).arbitrate([cand(score=0.5)], {})
    assert a.action == "insufficient_evidence"
    assert a.error and "JSONDecodeError" in a.error


def test_api_failure_routes_to_a_human():
    class BoomStub:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise RuntimeError("connection reset")

    a = _claude_with(BoomStub()).arbitrate([cand(score=0.5)], {})
    assert a.action == "insufficient_evidence"
    assert "RuntimeError" in a.error
    assert a.confidence_delta == 0.0


# ---------------------------------------------------------------------------
# End to end through the engine and the audit core
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (ROOT / "data" / "eval" / "ground_truth.json").exists(),
    reason="needs a generated dataset: python glctl.py generate",
)
def test_engine_with_tier3_stays_complete_and_balanced(tmp_path):
    """The invariants the whole system rests on, with tier 3 in the loop.

    Uses ``OfflineArbiter`` so the test is deterministic and offline. What is being
    checked is the *plumbing* -- that arbitration flows through re-policy, exception
    building, event recording and the ledger without breaking completeness or the
    hash chain -- which is exactly the claim the README makes for this arbiter.
    """
    from app.events import EventStore, record_run
    from app.ingestion import load_batch
    from app.matching import CalibratedScorer, DummyScorer, ReconciliationEngine
    from app.projections import rebuild

    batch = load_batch(ROOT / "data" / "eval")
    model = ROOT / "data" / "model.pkl"
    scorer = CalibratedScorer.load(model) if model.exists() else DummyScorer()
    threshold = float(getattr(scorer, "threshold", 0.85))

    engine = ReconciliationEngine(
        scorer, threshold=threshold, arbiter=OfflineArbiter()
    )
    run = engine.run(batch)

    assert run.assert_complete(batch)["complete"] is True
    assert run.tier3_ran() is True

    store = EventStore(tmp_path / "t3.db")
    record_run(store, batch, run)

    proj = rebuild(store)
    balanced, drift = proj.ledger_balanced()
    assert balanced, f"ledger out by {drift} paise with tier 3 enabled"
    assert store.verify_chain().ok

    # The arbitrations are in the log, and findable from their own transactions.
    assert proj.arbitrations, "tier 3 ran but wrote no events"
    kinds = {a["kind"] for a in proj.arbitrations}
    assert kinds == {"Tier3ArbitrationRequested", "Tier3ArbitrationReturned"}

    returned = next(
        a for a in proj.arbitrations if a["kind"] == "Tier3ArbitrationReturned"
    )
    for txn_id in returned["group_txn_ids"]:
        assert store.touching(txn_id), (
            f"arbitration is not findable from {txn_id}; an event that cannot be "
            "reached from the thing it decided is not an audit record"
        )
    store.close()


@pytest.mark.skipif(
    not (ROOT / "data" / "eval" / "ground_truth.json").exists(),
    reason="needs a generated dataset",
)
def test_tier3_off_is_byte_identical_to_no_tier3():
    """The published numbers must not move because the tier exists."""
    from app.ingestion import load_batch
    from app.matching import DummyScorer, ReconciliationEngine

    batch = load_batch(ROOT / "data" / "eval")
    a = ReconciliationEngine(DummyScorer(), threshold=0.85).run(batch)
    b = ReconciliationEngine(DummyScorer(), threshold=0.85, arbiter=None).run(batch)

    assert a.tier3_ran() is False
    assert a.tier3_totals() == {} == b.tier3_totals()
    assert len(a.all_confirmed()) == len(b.all_confirmed())
    assert len(a.all_proposed()) == len(b.all_proposed())
    assert len(a.all_exceptions()) == len(b.all_exceptions())


# ---------------------------------------------------------------------------
# Live API. Skipped without a key, so the suite stays deterministic.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no ANTHROPIC_API_KEY; tier 3's live path is not exercised",
)
def test_live_claude_arbiter_returns_a_schema_conforming_answer():
    group = [
        cand(left="bank:A", score=0.52, amount=2_499_00),
        cand(left="bank:B", score=0.48, amount=2_499_95),
    ]
    a = ClaudeArbiter().arbitrate(group, {})

    assert a.error is None, f"live arbitration failed: {a.error} / {a.reason}"
    assert a.action in ("propose_match", "insufficient_evidence")
    assert abs(a.confidence_delta) <= MAX_CONFIDENCE_DELTA
    if a.action == "propose_match":
        assert 1 <= a.chosen_index <= len(group)
    assert a.usage and a.usage["input_tokens"] > 0
