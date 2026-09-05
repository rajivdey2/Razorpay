# GlassLedger

**Multi-source reconciliation that never guesses with your money.** It matches what
it can prove, holds what it can't, and shows its work on both.

Razorpay AI Buildathon 2026 · Track 04 (AI Finance Controller) · direction:
multi-source reconciliation.

---

## The numbers

Measured on a held-out synthetic batch (seed 42, fingerprint `372a49a344090701`)
using a model trained on a **different** batch (seed 7). Every row runs the same
dataset through the same ingestion layer in the same process.

| Method | Match rate | Precision | Recall | Auto precision | ₹ reconciled | **Silently wrong** |
|---|---|---|---|---|---|---|
| Naive rules baseline | 60.0% | 68.0% | 60.0% | 68.0% | 73.5% | **350 matches · ₹26,22,806** |
| GlassLedger tier 1 only | 59.6% | 100.0% | 59.6% | 100.0% | 67.4% | **0** |
| GlassLedger tier 1+2 (heuristic) | 61.6% | 95.5% | 96.1% | 100.0% | 98.9% | **0** |
| **GlassLedger tier 1+2 (calibrated)** | **97.1%** | **99.5%** | **99.3%** | **100.0%** | **99.8%** | **0 · ₹0.00** |

Under a **2× break-rate stress test** (every failure mode twice as frequent):
96.0% match, 99.4% precision, 100% auto precision, still **0 silently wrong**.

**The last column is the whole pitch.** An auto-confirmed match is one no human ever
looks at, so a wrong one goes straight into the books. The baseline makes 350 of
them. GlassLedger makes none, and flags 94 items it could not prove — of which 85
genuinely were unprovable (90.4% exception precision, 94.4% recall).

Everything else, honestly stated:

| | |
|---|---|
| Throughput | **4,254 txn/s** end-to-end (1,593 transactions in 374 ms), single-threaded Python |
| Calibration | Brier **0.0067**, ECE **0.0148** on 479 held-out scored candidates |
| Auto-confirm threshold | **0.6675** — *derived*, not chosen (see below) |
| Blocking reduction | 98.97% of candidate pairs eliminated before scoring |
| Completeness | 1,593 / 1,593 transactions accounted for. Asserted, not assumed |
| Audit core | 4,293 events, hash chain verified, replay deterministic, ledger balances to **0 paise** |
| Tests | 93 passing, incl. 18 property-based |

Reproduce with `python eval/run_eval.py`. Numbers regenerate from scratch.

---

## What actually makes this hard

Three systems describe the same money and none of them agree:

```
Razorpay settlement          bank statement                   merchant's books
  net ₹4,157.26      ←→        credit ₹4,157.26        ←→       expected ₹4,244.90
  UTR KKBK260303306222         "…KK8K26030330…"                 (their own 2% fee guess)
  created T+0                  value date T+2                   booked on capture date
```

A `WHERE amount = amount AND date = date` join gets the clean cases. The rest —
batch settlements, splits, refund netting, fee drift, withholding, duplicate
credits, orphans, missing payouts, timing drift, mangled narrations, FX — is where
finance teams actually spend their week, and it is a **constrained combinatorial
optimisation problem**, not a lookup.

All eleven break patterns are generated, injected at configurable rates, and scored
individually. Per-pattern precision/recall is in the eval output — no aggregate
hiding a subgroup failure.

---

## Run it

```bash
pip install -r requirements.txt

python glctl.py generate --seed 42 --payments 900   # dataset + answer key
python glctl.py reconcile --explain                 # run + record, show evidence
python glctl.py exceptions --top 10                 # the ranked queue
python glctl.py audit --txn gateway:setl_…          # replay one transaction
python glctl.py verify                              # chain, replay, balance, append-only
python glctl.py tamper --seq 5                      # prove tampering is detected
python glctl.py serve                               # workbench at :8000

python eval/run_eval.py                             # the full table, from scratch
python -m pytest backend/tests -q                   # 93 tests
```

No database server, no API key, no build step. SQLite and vanilla JS on purpose —
for a system whose central claim is "you can verify this yourself", setup friction
is a real cost.

---

## How it works

```
  3 bank dialects + settlements API + books CSV
              ↓  strict per-format parsers, running-balance verification
        canonical NormalizedTxn  (int paise, always)
              ↓
  ┌─────────────────── TIER 1 · deterministic ────────────────────┐
  │ wash-pair netting · exact reference+amount · order-key join   │  60% of pairs
  │ refuses when ambiguous → passes down, never guesses           │  100% precision
  └───────────────────────────────────────────────────────────────┘
              ↓
  ┌─────────────────── TIER 2 · optimisation ─────────────────────┐
  │ blocked graph → Hungarian assignment (optimal, not greedy)    │  +37% of pairs
  │ bounded subset-sum with admissible pruning (N:1, 1:N)         │
  │ per-component assignment from the gateway's own recon report  │
  │ → calibrated confidence (GBM + isotonic), validated           │
  └───────────────────────────────────────────────────────────────┘
              ↓
  ┌─────────────────── TIER 3 · bounded arbitration ──────────────┐
  │ only inside [0.10, threshold) · never above materiality       │  built, not run
  │ structured output only · Δconfidence clamped to ±0.25         │  (no API key here)
  │ proposer, never approver                                      │
  └───────────────────────────────────────────────────────────────┘
              ↓
  POLICY: immateriality → materiality gate → uniqueness → threshold
              ↓
  append-only hash-chained event store → CQRS projections → workbench
```

### Six decisions worth defending

**1. Money is `int` paise. Enforced, not remembered.**
`check_paise` rejects floats *and* `bool` (an `int` subclass worth 1 paise). The
statement parser is string-based because `int(float("861.55") * 100)` is `86154`,
not `86155` — off by one paise in the direction that accumulates. A property test
searches for its own witnesses rather than hard-coding one.

**2. The threshold is derived from the calibration curve, never tuned.**
`choose_threshold` sweeps the held-out slice and takes the *lowest* probability
whose precision clears the 0.995 target — most permissive we can justify, so recall
is maximised while precision holds. It came out at **0.6675 with precision 1.0**.
If no threshold can meet the target, it reports `target_met: false` and the
shortfall rather than silently falling back — and that report is what surfaced the
worst bug in the project (POSTMORTEM #1).

Three-way split: fit / calibrate / **choose-threshold**. The third slice is the one
people skip, and skipping it biases the threshold optimistically in exactly the
high-confidence region auto-confirm depends on.

**3. The materiality gate is code, outside every model.**
Above ₹50,000 no *inferred* match auto-confirms — 8 held for a human on the eval
batch. Not a prompt instruction: models get retrained, prompts get rewritten, this
does not. Tier 1 identity matches are exempt by documented policy
(`MATERIALITY_APPLIES_TO_TIER1`), because "these two lines carry the same 16-char
UTR and identical amounts" is re-checkable by anyone in one second, while "the model
scored 0.93" is a statement about a distribution. The exemption is **reported**, not
assumed: 26 matches, ₹23,43,655 — the eval prints it, and one flag restores the
strict reading.

**4. Optimal assignment, not greedy.**
Greedy is locally right and globally wrong in a way that compounds: it takes B's
0.88, leaves A unmatched, books B confidently. The Hungarian assignment gives B its
0.60 and A its 0.85 — and is right about *both*. Decomposed over connected
components first, so the O(n³) applies to the largest component (max 4 here) rather
than the whole month. That is a factorisation, not an approximation.

**5. Ambiguity is an output, not a thing to resolve.**
Tier 1 *refuses* when two bank lines share a UTR and an amount — that case is 50/50,
and a rule claiming 100% confidence at 50% accuracy poisons everything downstream.
Subset-sum returns the **count** of valid subsets, and that count is a feature. Two
identical candidates in a crowded window are one piece of evidence and a coin flip,
and the model learns to price that.

**6. Nothing is silently dropped. Asserted.**
Every transaction lands in exactly one of three buckets — matched, excepted, or
explicitly written off with an event. `assert_complete` raises otherwise. It caught
a ₹9.99 accrual vanishing on the stress run (POSTMORTEM #3), which is the bug class
that looks like a *better* match rate.

---

## The audit core

Append-only, enforced by the database (SQLite triggers refuse `UPDATE`/`DELETE`),
and hash-chained: `hash = sha256(prev_hash ‖ canonical_json(event))`. The chain
covers content **and position**, so reordering is as detectable as editing.

`python glctl.py tamper` proves it — drops the triggers (the file-access attack),
edits an event, and shows the chain catching it at the exact sequence number:

```
  before tampering : OK (4293 events)
  tampered with seq 5: inflated the amount by ₹100
  after tampering  : BROKEN
    seq 5: stored hash 40be0605d22d… but content hashes to 60382e44ea75…;
           this event's payload was edited after it was written
```

Projections are pure folds over that stream — `rebuild()` twice yields an identical
fingerprint, asserted in tests and printed by `glctl verify`. A projection bug is
fixed by correcting the fold and replaying: no migration, no data loss, because the
events were never what was wrong.

Every event carries a content-derived idempotency key, so re-running the whole
pipeline appends nothing the second time. That is what makes "just run it again" a
safe response to a partial failure — and at-least-once webhook delivery is the norm,
not an edge case.

Confirmed gateway→bank matches post real double-entry:

```
Dr  1010:Bank                     ₹50,17,680.43
  Cr  1210:Gateway receivable    -₹50,95,493.48
Dr  6100:Payment processing fees     ₹65,943.04
Dr  1360:GST input credit            ₹11,870.01
                                 ─────────────
                          net              0     ← asserted every run
```

Entries balance without a plug. An unbalanced entry **raises rather than posting** —
a reconciliation system that posts a balancing "difference" account has relabelled
the problem, not solved it. A match whose sides differ by even one paise posts
nothing and becomes an exception.

---

## Tier 3: built, bounded, and honestly not exercised

There is no `ANTHROPIC_API_KEY` in this environment, so **every published number is
tiers 1+2 only**. The tier is implemented and its plumbing is exercised end-to-end
by `OfflineArbiter` — a **deterministic rule arbiter that is not an LLM**, tagged
`offline_rule_arbiter` everywhere it appears so nothing it produces can be mistaken
for a model result.

The bounds are code, not prompt text:

| Bound | Enforcement |
|---|---|
| Cannot confirm anything | `confidence_delta` clamped to ±0.25; policy still decides |
| Never sees material items | Filtered out **before the payload is built** — 23 of 27 on the eval batch |
| Only the ambiguous band | `[0.10, threshold)`; outside it there is nothing to decide |
| Cannot emit prose | `output_config.format` with a strict JSON schema, at the API layer |
| Cannot pick a candidate it wasn't offered | Index range-checked in code |
| Unavailable ⇒ human, never a guess | Any error → `insufficient_evidence`, recorded |

`auto` mode deliberately does **not** fall back to the offline arbiter — that would
mean the same command yields model-derived numbers on one machine and rule-derived
numbers on another with nothing to distinguish them.

The band is `[floor, derived_threshold)` rather than the planned fixed `[0.40,
0.75]`, because the threshold is derived per training run. A fixed upper bound above
a derived threshold would mean arbitrating matches the policy had already confirmed.

---

## The workbench

`python glctl.py serve` → four views, reading from projections rebuilt off the event
stream. Human overrides append events, so they are exactly as auditable as agent
decisions and land in the same trail.

- **Dashboard** — tiles, tier-1-vs-tier-2 split, exception categories, money flow,
  live ledger, integrity chips
- **Exception queue** — ranked by money at risk × age, with evidence and one-click
  resolution
- **Awaiting approval** — feature breakdowns, runners-up, and the materiality gate
  explaining itself on 21 held matches
- **Audit trail** — replay any transaction: ingestion → every hypothesis with its
  features → the alternatives rejected → confirmation → journal entry

Charts are hand-rolled inline SVG. Categorical colours are capped at three
validated slots (all-pairs CVD ΔE 9.4, normal-vision 20.9 against this surface);
status colours are reserved and always paired with a glyph and a word, never hue
alone.

---

## What broke

[`POSTMORTEM.md`](POSTMORTEM.md) — ten entries, kept as the work happened. The ones
worth reading:

- **A shared tolerance across two opposite legs** produced the single silent-wrong
  match. The fix took it to zero *and* let the threshold search meet its precision
  target for the first time — the bad feature had been poisoning the calibration
  curve, and the honest-reporting code had been telling me so for two hours before I
  understood it.
- **The books leg blew up combinatorially**, and the fix was not a better search:
  it was noticing the gateway's recon report made most of it an *assignment* problem,
  not a subset-sum problem. 780× fewer nodes, 15× faster, +44 points of recall.
- **`assert_complete` caught a ₹9.99 accrual disappearing** on the stress run — the
  bug class that looks like a better match rate.
- **A property test caught a sign-flipped refund** at −1 paise, from a currency
  symbol hiding a minus sign.

Also still open, listed there: greedy set packing is a real approximation (442
conflicts, reported); the model is slightly *under*-confident in the 0.6–0.8 band;
`touching()` is a `LIKE` scan that would need an index table at a million events.

---

## Layout

```
glassledger/
├── glctl.py                    generate · reconcile · exceptions · audit · verify · tamper · serve
├── backend/app/
│   ├── core/                   money (int paise, enforced) · schema · config (every threshold, justified)
│   ├── ingestion/              3 bank dialects · settlements API · books · reference extraction
│   ├── matching/               blocking · subsetsum · tier1 · tier2 · confidence · expectations
│   │                           residuals · policy · exceptions · engine
│   ├── events/                 append-only hash-chained store · recorder + double-entry
│   ├── projections/            CQRS read models (pure folds)
│   ├── llm/                    bounded arbiter + tier-3 orchestration
│   └── api/                    FastAPI: reads from projections, writes events
├── data_generator/             world · 11 break patterns (one module each) · 3 emitters
├── eval/                       run_eval · metrics · baseline
├── frontend/                   workbench (vanilla JS, inline SVG)
└── backend/tests/              93 tests incl. property-based
```

---

## Why this complements RazorpayX rather than duplicating it

The Bookkeeping Agent posts entries from predefined rules. Rules handle the 60% that
is deterministically identifiable — and GlassLedger's own tier 1 measures exactly
that: **59.6% of pairs at 100% precision**. That number is not a criticism of rules;
it is the honest ceiling of what rules can do.

The other 40% is the batch settlements, splits, netted refunds, fee drift and
timing noise that a rules engine either misses or guesses at. The baseline in the
table *does* guess: it reaches 60% match by making 350 wrong matches worth ₹26 lakh,
and reports none of them.

GlassLedger takes that 40% to 97% — with a calibrated confidence, a hard gate above
₹50,000, an evidence-backed exception list for the remainder, and a replayable
reason for every decision. **The honesty is the feature.** A system that says "I am
not sure about these 94 items, here is exactly why" is worth more to a finance team
than one claiming 100%, because the second one is lying and they will find out at
close.
