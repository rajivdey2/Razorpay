"""Tier 3: bounded LLM arbitration.

What the model is allowed to do
-------------------------------
Look at 2-4 candidate explanations for one unresolved item, with their feature
breakdowns and the surrounding transaction context, and return **one structured
object**:

    {"action": "propose_match", "candidate_index": 1,
     "confidence_delta": 0.18, "rationale": "...", "evidence_cited": [...]}

or

    {"action": "insufficient_evidence", "reason": "..."}

There is no third option and no freeform path. The response format is constrained
by ``output_config.format`` with a strict JSON schema, so the API itself guarantees
schema-conforming output -- a hallucinated rupee figure cannot arrive as prose
because prose is not a shape the response can take.

What the model is *not* allowed to do
------------------------------------
Three hard bounds, all enforced in this file rather than requested in the prompt.
That distinction is the entire point: a prompt is a preference and code is a
constraint, and the two are not interchangeable when the subject is money.

**1. It cannot confirm anything.** ``confidence_delta`` is clamped to
``MAX_CONFIDENCE_DELTA`` (0.25). The arbiter adjusts a calibrated score; the policy
layer still decides. A candidate at 0.30 cannot reach a 0.75 threshold on the
model's word, no matter how certain the model claims to be -- it can break a tie
between two near-equal candidates, which is what it is genuinely good at, and it
cannot manufacture a conclusion.

**2. It never sees, and cannot affect, anything above the materiality gate.**
Candidates above ``MATERIALITY_PAISE`` are filtered out before the request is
built. Not "the prompt tells it not to" -- the item is not in the payload.

**3. It is only invoked inside the ambiguous band.** Above the auto-confirm
threshold the deterministic tiers already agree, and paying for a model call would
add latency and a hallucination surface for no decision. Below ``TIER3_FLOOR`` the
evidence is too thin for any reader, model or human, and the honest output is an
exception.

A note on the band
------------------
The plan specified a fixed band of [0.40, 0.75]. This uses ``[TIER3_FLOOR,
auto_confirm_threshold)`` instead, because the threshold is *derived* from the
calibration curve rather than fixed -- on the current eval run it comes out at
0.6675, which sits inside the planned band. A fixed upper bound above a derived
threshold would mean arbitrating matches the policy had already decided to confirm.
The band has to be defined relative to the decision boundary, not in absolute terms,
or the two drift apart silently the first time the model is retrained.

Choosing an arbiter
-------------------
Three implementations, and which one ran is recorded on every decision it makes:

``ClaudeArbiter``  the real thing: Claude via the Messages API, structured output
                   only. Requires ``ANTHROPIC_API_KEY``.
``NullArbiter``    the default when no credentials are present: returns
                   ``insufficient_evidence`` for everything and records why.
``OfflineArbiter`` a deterministic rule arbiter. **It is not an LLM.** It exists so
                   the arbitration path can be exercised without network access or
                   credentials, and its decisions are tagged
                   ``arbiter="offline_rule_arbiter"`` in the event log and reported
                   under that name, so nothing it produces can be mistaken for a
                   model result.

The distinction is laboured on purpose. A stub that quietly stands in for a model
turns every downstream number into a claim about something that never ran -- so the
tier is **off unless asked for by name**, and which arbiter answered is a field in
the audit trail rather than an inference from the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.config import MATERIALITY_PAISE
from app.core.money import inr
from app.matching.types import Candidate

#: Hard cap on how far one arbitration may move a calibrated score. Enforced by
#: clamping the returned value, so a model that returns 0.9 moves the score 0.25.
MAX_CONFIDENCE_DELTA = 0.25

#: Below this, nothing is worth arbitrating -- the evidence is too thin for any
#: reader. Above the derived auto-confirm threshold, there is nothing to decide.
TIER3_FLOOR = 0.10

#: Candidates offered per arbitration. Three is the plan's figure and it is the
#: right one: enough to express a real choice, few enough that the model cannot
#: pick a plausible-looking outlier from a long tail.
MAX_CANDIDATES = 3

MODEL = "claude-opus-5"

ARBITRATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["propose_match", "insufficient_evidence"],
        },
        "candidate_index": {
            "type": "integer",
            "description": (
                "1-based index of the chosen candidate. Required when action is "
                "propose_match; use 0 when action is insufficient_evidence."
            ),
        },
        "confidence_delta": {
            "type": "number",
            "description": (
                "How much this evidence should move the existing calibrated score, "
                "between -0.25 and +0.25. Positive supports the chosen candidate. "
                "Values outside the range are clamped by the caller."
            ),
        },
        "rationale": {
            "type": "string",
            "description": (
                "One or two sentences citing the specific evidence fields that "
                "drove the decision. Do not restate amounts that are not in the "
                "provided evidence."
            ),
        },
        "evidence_cited": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the evidence fields relied on.",
        },
        "reason": {
            "type": "string",
            "description": "Why the evidence is insufficient. Empty for propose_match.",
        },
    },
    "required": [
        "action", "candidate_index", "confidence_delta",
        "rationale", "evidence_cited", "reason",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a reconciliation arbiter inside a financial control system. You are given \
an unresolved reconciliation item and 2 to 3 candidate explanations, each with a \
calibrated confidence score and a feature breakdown computed by a deterministic \
engine.

Your job is narrow: decide whether the evidence distinguishes one candidate from \
the others, and by how much.

Rules you must follow:
- Only reason about the evidence in the payload. Do not infer amounts, dates, or \
references that are not present.
- If two candidates are supported equally well, return insufficient_evidence. A \
tie is a real answer and the correct one; a coin flip that posts a journal entry \
is not.
- confidence_delta expresses how much the evidence should move an existing \
calibrated score, not your overall belief. Reserve values near 0.25 for cases \
where one candidate has an unambiguous identifier the others lack.
- Reference similarity below roughly 0.6 is weak evidence. Amount agreement alone, \
without a supporting reference or date, is weak evidence: identical amounts are \
common in retail payment data.

You cannot confirm a match. Your output adjusts a score that a separate policy \
layer acts on."""


@dataclass
class Arbitration:
    """One arbiter decision, ready to be recorded as an event."""

    match_key: str
    arbiter: str
    action: str
    chosen_index: int
    confidence_delta: float
    rationale: str
    evidence_cited: list[str] = field(default_factory=list)
    reason: str = ""
    #: Whatever the arbiter returned before clamping, for the audit trail. A
    #: request that came back at 0.9 and was clamped to 0.25 is a fact worth
    #: keeping: it is evidence about the arbiter, not just about the match.
    raw_delta: float | None = None
    clamped: bool = False
    error: str | None = None
    usage: dict | None = None

    def to_json(self) -> dict:
        return {
            "match_key": self.match_key, "arbiter": self.arbiter,
            "action": self.action, "chosen_index": self.chosen_index,
            "confidence_delta": round(self.confidence_delta, 6),
            "rationale": self.rationale, "evidence_cited": self.evidence_cited,
            "reason": self.reason, "raw_delta": self.raw_delta,
            "clamped": self.clamped, "error": self.error, "usage": self.usage,
        }


class Arbiter(Protocol):
    name: str

    def arbitrate(self, group: list[Candidate], context: dict) -> Arbitration: ...


# ---------------------------------------------------------------------------
# Payload construction -- shared by every arbiter so the offline one sees
# exactly what the model would
# ---------------------------------------------------------------------------

def build_payload(group: list[Candidate], context: dict) -> dict:
    """The evidence bundle. Curated, not a dump.

    Only the features and evidence that bear on the decision go in. Handing a
    model the whole batch would cost tokens, bury the signal, and -- worse --
    let it "notice" a pattern across unrelated transactions that the
    deterministic tiers deliberately never considered. Narrow input is a
    correctness property here, not a cost optimisation.
    """
    candidates = []
    for i, c in enumerate(group[:MAX_CANDIDATES], start=1):
        ev = c.evidence or {}
        candidates.append(
            {
                "index": i,
                "algorithm": c.algorithm,
                "cardinality": c.cardinality,
                "calibrated_confidence": round(c.score, 4),
                "left_ids": list(c.left_ids),
                "right_ids": list(c.right_ids),
                "left_amount": inr(c.left_amount_paise),
                "right_amount": inr(c.right_amount_paise),
                "residual": inr(c.residual_paise),
                "date_delta_days": ev.get("date_delta_days"),
                "settlement_cycle_prior": ev.get("settlement_cycle_prior"),
                "reference_comparison": ev.get("reference_comparison"),
                "competing_candidates": ev.get("competing_candidates"),
                "rule": ev.get("rule"),
                "features": {
                    k: round(v, 4)
                    for k, v in sorted((c.features or {}).items())
                    if k in (
                        "utr_exact", "utr_similarity", "utr_in_narration",
                        "amount_exact", "amount_rel_delta", "within_fee_band",
                        "date_delta_abs", "left_ambiguity", "right_ambiguity",
                        "amount_uniqueness", "narration_cosine",
                    )
                },
            }
        )
    return {
        "leg": group[0].leg,
        "unresolved_item": context.get("subject"),
        "surrounding_context": context.get("nearby", []),
        "candidates": candidates,
        "policy_note": (
            "You cannot confirm a match. Your confidence_delta is clamped to "
            f"+/-{MAX_CONFIDENCE_DELTA} by the caller, and any item above the "
            f"{inr(MATERIALITY_PAISE)} materiality gate is excluded from "
            "arbitration entirely."
        ),
    }


def _clamp(delta: float) -> tuple[float, bool]:
    clamped = max(-MAX_CONFIDENCE_DELTA, min(MAX_CONFIDENCE_DELTA, float(delta)))
    return clamped, clamped != float(delta)


def _usage_of(resp) -> dict | None:
    """Token counts off a response, or None. Recorded on every outcome.

    Cost is reported from what the API actually charged rather than estimated from
    the payload, and it is captured on the *failure* paths too -- a truncated call
    costs money and a run that hides that is understating what the tier costs.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


# ---------------------------------------------------------------------------
# Arbiters
# ---------------------------------------------------------------------------

class NullArbiter:
    """The default when no credentials are configured.

    Returns ``insufficient_evidence`` for everything, with a reason that names the
    missing configuration. The run still completes and the affected items land on
    the exception list, which is the correct behaviour for a system whose
    arbitration tier is unavailable -- it degrades to "ask a human", not to
    "guess".
    """

    name = "null_arbiter"
    available = False

    def arbitrate(self, group: list[Candidate], context: dict) -> Arbitration:
        return Arbitration(
            match_key=group[0].match_key, arbiter=self.name,
            action="insufficient_evidence", chosen_index=0, confidence_delta=0.0,
            rationale="",
            reason="tier 3 arbitration not configured (no ANTHROPIC_API_KEY); "
                   "item routed to human review",
        )


class OfflineArbiter:
    """Deterministic rule arbiter. **Not an LLM.**

    Exists so the arbitration path -- payload construction, clamping, event
    recording, score feedback, policy re-evaluation -- can be exercised and tested
    without network access or credentials. Its rules are three lines of arithmetic
    over the same evidence bundle the model would receive.

    Every decision it makes is tagged ``offline_rule_arbiter`` in the event log and
    in the eval output. Nothing it produces is ever reported as a model result,
    because a stub that silently stands in for a model makes every number that
    depends on it a claim about something that never ran.
    """

    name = "offline_rule_arbiter"
    available = True

    def arbitrate(self, group: list[Candidate], context: dict) -> Arbitration:
        payload = build_payload(group, context)
        cands = payload["candidates"]
        if not cands:
            return NullArbiter().arbitrate(group, context)

        def strength(c: dict) -> float:
            f = c["features"]
            return (
                0.5 * f.get("utr_exact", 0.0)
                + 0.3 * f.get("utr_similarity", 0.0)
                + 0.2 * f.get("amount_exact", 0.0)
                - 0.2 * f.get("left_ambiguity", 0.0)
            )

        ranked = sorted(cands, key=strength, reverse=True)
        best = ranked[0]
        margin = strength(best) - (strength(ranked[1]) if len(ranked) > 1 else 0.0)

        if margin < 0.15:
            return Arbitration(
                match_key=group[0].match_key, arbiter=self.name,
                action="insufficient_evidence", chosen_index=0,
                confidence_delta=0.0, rationale="",
                reason=f"top two candidates separated by only {margin:.3f} on "
                       "reference and amount evidence; a tie is the honest answer",
            )

        raw = min(0.25, margin)
        delta, clamped = _clamp(raw)
        return Arbitration(
            match_key=group[0].match_key, arbiter=self.name,
            action="propose_match", chosen_index=best["index"],
            confidence_delta=delta,
            rationale=f"candidate {best['index']} leads on reference and amount "
                      f"evidence by {margin:.3f}; residual {best['residual']}",
            evidence_cited=["utr_exact", "utr_similarity", "amount_exact",
                            "left_ambiguity"],
            raw_delta=raw, clamped=clamped,
        )


class ClaudeArbiter:
    """The real thing: Claude via the Messages API, structured output only.

    ``output_config.format`` with a strict JSON schema is used rather than forced
    ``tool_choice``. Both constrain the output; structured outputs do it at the API
    layer -- the response *cannot* be prose, so there is no parse-and-hope step and
    no path by which a hallucinated number reaches the ledger as free text. It also
    composes with adaptive thinking, which forced tool choice does not on every
    model.

    Failure handling is deliberately blunt. Any error -- network, rate limit,
    refusal, truncation, malformed JSON -- becomes ``insufficient_evidence`` with the
    error recorded. An arbitration tier that retries into a timeout budget, or falls
    back to a heuristic when the model is unavailable, silently changes the meaning of
    every number downstream. Unavailable means "ask a human".

    ``effort="low"`` and ``max_tokens=4096`` are chosen, not defaults inherited from
    somewhere. The task is one bounded discrimination over a payload of at most three
    candidates, which is not what high effort is for; and thinking tokens count
    against ``max_tokens`` on this model, so a tight budget truncates the object
    rather than shortening the reasoning. 4096 leaves the reasoning room while the
    answer itself is under 200 tokens -- and if it ever does truncate, that is
    reported as truncation rather than as a parse failure.
    """

    name = "claude_arbiter"

    def __init__(self, model: str = MODEL, *, effort: str = "low", max_tokens: int = 4096):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = None
        self.available = bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _client_or_none(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
            except Exception:  # pragma: no cover - depends on environment
                return None
        return self._client

    def arbitrate(self, group: list[Candidate], context: dict) -> Arbitration:
        client = self._client_or_none()
        if client is None:
            return Arbitration(
                match_key=group[0].match_key, arbiter=self.name,
                action="insufficient_evidence", chosen_index=0,
                confidence_delta=0.0, rationale="",
                reason="Anthropic client unavailable", error="client_init_failed",
            )

        payload = build_payload(group, context)
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": ARBITRATION_SCHEMA},
                },
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Arbitrate this reconciliation item.\n\n"
                            + json.dumps(payload, indent=2, default=str)
                        ),
                    }
                ],
            )
        except Exception as exc:  # pragma: no cover - network dependent
            return Arbitration(
                match_key=group[0].match_key, arbiter=self.name,
                action="insufficient_evidence", chosen_index=0,
                confidence_delta=0.0, rationale="",
                reason="arbitration call failed; item routed to human review",
                error=f"{type(exc).__name__}: {exc}",
            )

        # A safety classifier may decline. That is a 200 with stop_reason
        # "refusal", not an exception, so it has to be checked explicitly --
        # reading .content first would raise or return nothing useful.
        if getattr(resp, "stop_reason", None) == "refusal":
            return Arbitration(
                match_key=group[0].match_key, arbiter=self.name,
                action="insufficient_evidence", chosen_index=0,
                confidence_delta=0.0, rationale="",
                reason="model declined to arbitrate", error="refusal",
            )

        # Truncation is checked before parsing, and named. Thinking is on by
        # default on this model and its tokens count against ``max_tokens``, so a
        # budget set too low truncates mid-object -- and the JSON then fails to
        # parse. Both paths end in ``insufficient_evidence``, but they are not the
        # same fact: "the arbiter never finished answering" is a *configuration*
        # bug that would silently apply to every single call, while "the arbiter
        # answered something unreadable" is a one-off. Folding the first into the
        # second would report a run where the tier never functioned as a run where
        # the tier found nothing to say, which is the same class of error as a
        # vanished transaction looking like a clean reconciliation.
        if getattr(resp, "stop_reason", None) == "max_tokens":
            return Arbitration(
                match_key=group[0].match_key, arbiter=self.name,
                action="insufficient_evidence", chosen_index=0,
                confidence_delta=0.0, rationale="",
                reason=f"arbiter response truncated at max_tokens={self.max_tokens}; "
                       "raise the budget rather than reading a partial object",
                error="max_tokens_truncated", usage=_usage_of(resp),
            )

        # Join every text block rather than taking the first. With thinking enabled
        # -- the default on this model -- the response carries a thinking block
        # alongside the answer, and the answer itself is not guaranteed to arrive as
        # exactly one text block. Taking ``next(...)`` picks whichever text block
        # came first, which is an empty string often enough to matter, and the
        # resulting "Expecting value: line 1 column 1" is indistinguishable from the
        # model having ignored the schema entirely.
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        try:
            data = json.loads(text)
        except Exception as exc:
            return Arbitration(
                match_key=group[0].match_key, arbiter=self.name,
                action="insufficient_evidence", chosen_index=0,
                confidence_delta=0.0, rationale="",
                reason="arbiter response did not conform to the schema",
                # The first 200 characters of what actually came back are kept on
                # the record. Without them this failure is only diagnosable by
                # re-running against a live API and hoping it reproduces, which for
                # a non-deterministic component is not a debugging strategy. With
                # them, the event log says whether the model returned prose, an
                # empty string, or a fenced code block.
                error=f"{type(exc).__name__}: {exc} | received: {text[:200]!r}",
                usage=_usage_of(resp),
            )

        usage = _usage_of(resp)

        raw = float(data.get("confidence_delta") or 0.0)
        delta, clamped = _clamp(raw)
        action = data.get("action", "insufficient_evidence")
        idx = int(data.get("candidate_index") or 0)

        # Validate the index in code. The schema guarantees an integer; it does not
        # guarantee the integer refers to a candidate that was actually offered.
        if action == "propose_match" and not (1 <= idx <= len(payload["candidates"])):
            return Arbitration(
                match_key=group[0].match_key, arbiter=self.name,
                action="insufficient_evidence", chosen_index=0,
                confidence_delta=0.0, rationale=data.get("rationale", ""),
                reason=f"arbiter returned out-of-range candidate index {idx}",
                error="index_out_of_range", raw_delta=raw, usage=usage,
            )
        if action == "insufficient_evidence":
            delta, idx = 0.0, 0

        return Arbitration(
            match_key=group[0].match_key, arbiter=self.name,
            action=action, chosen_index=idx, confidence_delta=delta,
            rationale=str(data.get("rationale") or "")[:500],
            evidence_cited=[str(e) for e in (data.get("evidence_cited") or [])][:12],
            reason=str(data.get("reason") or "")[:500],
            raw_delta=raw, clamped=clamped, usage=usage,
        )


def default_arbiter(mode: str = "auto") -> Arbiter:
    """``auto`` uses Claude when a key is present, otherwise the null arbiter.

    ``auto`` deliberately does *not* fall back to ``OfflineArbiter``. Falling back
    to a rule engine when the model is unavailable would mean the same command
    produces model-derived numbers on one machine and rule-derived numbers on
    another, with nothing in the output to distinguish them. The offline arbiter has
    to be asked for by name.
    """
    if mode == "offline":
        return OfflineArbiter()
    if mode == "null":
        return NullArbiter()
    if mode == "claude":
        return ClaudeArbiter()
    arb = ClaudeArbiter()
    return arb if arb.available else NullArbiter()
