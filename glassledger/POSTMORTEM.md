# POSTMORTEM

What broke, what it cost, and what the fix taught. Written as it happened.

The entries are ordered by how much they changed the system, not chronologically.
Every one of them is a bug I shipped and then found — several of them found *by the
system's own guards*, which is the part I care about most, because a guard that has
never caught anything is a guard you cannot trust.

---

## 1. The silent-wrong match: one shared tolerance across two opposite legs

**Symptom.** The eval reported exactly one auto-confirmed match that ground truth
said was wrong. One in 1,202. It would have been easy to call that noise.

**What it actually was.** A settlement of ₹2,499.00 that the bank had *never
credited* — a `missing_settlement`, which should have become an exception — was
matched to `bank:BR6816834194-P2`, the second half of a **different** settlement's
split. 95 paise apart, same value date, different UTRs (similarity 0.56). The model
scored it **1.0000**.

The reason a 95-paise gap looked acceptable was one line in `features.py`:

```python
band = max(leg.amount_abs_tolerance_paise // 10, (basis * leg.explainable_bps) // 10_000)
```

`amount_abs_tolerance_paise` is a *blocking* parameter — how far apart two amounts
can be and still be worth **considering**. I reused it as the floor for
`within_fee_band`, which is a *scoring* feature meaning "this gap is explainable by
fees". On the gateway→bank leg that floor worked out to ₹10, so any residual under
₹10 set `within_fee_band = 1`, and the model — correctly, given the feature —
treated that as strong evidence.

**The deeper error.** Those two legs have opposite requirements. A bank credit
either equals the payout to the paise or it is a different payout; there is no
explainable band at all. A books entry is *expected* to differ by fees, GST,
withholding and FX. I had one parameter serving both, and a parameter shared between
two requirements that contradict each other is wrong on at least one of them.

**Fix.** `explainable_floor_paise` as its own per-leg config: `0` for
gateway→bank, ₹25 for books, ₹20 for the component sub-config.

**What it cost, and the thing I nearly missed.** Silent-wrong went 1 → **0** and
auto-confirm precision 99.9% → **100%**. But the more interesting effect was on the
threshold: before the fix, `choose_threshold` could not find *any* threshold meeting
the 0.995 precision target and reported `target_met=False`, settling for 0.4167.
After the fix it found 0.6675 with precision 1.0. The bad feature had been poisoning
the calibration curve, and the "cannot reach the target" message was the *symptom* I
had already surfaced without understanding. The lesson: the honest-reporting code
told me something was wrong two hours before I worked out what.

---

## 2. The books leg blew up combinatorially — and the fix was not a better search

**Symptom.** 5.3 seconds for a 900-payment month, against 26ms for the other leg.
Pools averaging 75 candidate entries, 3.4M search nodes, **211 of 288 pools hitting
the solution cap**, 827 set-packing conflicts.

**First instinct, which was wrong.** Tune the search: raise the node budget, tighten
the tolerance, cap subset size. That treats a symptom. The search was not failing to
find the right subset — it was finding *hundreds* of subsets that summed to within
tolerance, and choosing among them with no real evidence. Ambiguity, not slowness.

**What was actually wrong.** I was using subset-sum on a problem that had stopped
being a subset-sum problem. Three constraints were sitting unused in data I had
already ingested:

1. **Tier 1 knew which capture day each batch covered.** It had already joined the
   keyed entries; their dates were right there. An 18-day window became 1–3 days.
2. **The gateway's recon report lists every payment in the payout, individually.**
   So the *expected book amount per payment* was computable — meaning the majority
   of this leg is a per-component **assignment** problem, not a subset-sum problem
   at all.
3. **The merchant's own fee assumption was recoverable.** Every tier-1 keyed pair
   gives a (book amount, gateway gross) ratio. The median over ~500 of them is the
   blended rate the books apply — self-calibrating, no configuration.

**Fix.** `expectations.py`, and a three-step books pipeline: order-key join →
per-component assignment (Hungarian, tight band) → subset-sum for the genuine
residue (accruals, FX adjustments — lines with no component counterpart).

**Cost.** 5,285ms → **339ms** (15×). Search nodes 3.4M → 4,386 (**780×**). Pools
hitting the cap 211 → 11. Books-leg auto-confirm recall 53.7% → **97.7%**.

**The lesson, and it is the one I would lead with.** I spent an hour tuning a solver
that was the wrong solver. The 780× came from *using information I already had* and
from *matching the algorithm to the problem's actual shape* — not from a better
search. On this class of problem that is almost always where the wins are.

---

## 3. `assert_complete` caught a transaction disappearing — under stress, not before

**Symptom.** The 2× break-rate stress run crashed:

```
AssertionError: 1 transactions were neither resolved nor flagged:
['books:TDS-38K0HF36']. Every ingested line must land in exactly one bucket.
```

**Cause.** A ₹9.99 withholding accrual. The exception builder skipped unmatched
lines below the ₹10 attention floor with a bare `continue` — so it was not matched,
not flagged, and not recorded. It simply vanished from the reconciliation.

**Why this is the entry I am most glad about.** The bug is small; the *class* of bug
is the worst one this system can have, because a vanished line looks exactly like a
line that reconciled cleanly. It did not appear at 1× rates. It appeared only when
the stress run generated a sub-₹10 accrual that nothing else claimed. Without the
completeness assertion, the run would have printed a slightly better match rate and
I would have shipped it.

**Fix.** `ExceptionReport.written_off` — an explicit third bucket with a
`SuspenseWriteOff` event per line. "Not worth a human's attention" and "not recorded"
are different states, and conflating them is a data-loss bug wearing a policy
costume.

---

## 4. The audit trail was missing its own conclusion

**Symptom.** Opened the workbench audit view on a confirmed settlement. Fourteen
events: one `TransactionIngested`, thirteen `MatchCandidateProposed`. No
`MatchConfirmed`. The single most important event about the match was absent from
the trail of the transaction it decided.

**Cause.** `EventStore.touching(txn_id)` finds events by scanning payloads for the
id. `MatchCandidateProposed` carries `left_ids`/`right_ids`; `MatchConfirmed` carried
only the match key, the confidence and the rationale. Correct as a record of the
decision, invisible from the thing decided.

**Fix.** Repeat the ids on `MatchConfirmed`, `MatchRejected` and
`JournalEntryPosted` — denormalisation, on purpose, with a comment saying why. Plus
`test_touching_finds_the_confirmation_too` so it cannot regress.

**Lesson.** The demo *was* the test. I had unit tests asserting the event was
written; none asserted it was **findable from the transaction**, which is the only
thing an auditor will ever do with it.

---

## 5. The generator's break patterns were not commutative

**Symptom.** Roughly 4% of settlements labelled `clean` had a bank line that
disagreed with their own settlement amount. In a generator, that means the *answer
key* is wrong — and every metric computed against it.

**Cause.** Refund netting changes a settlement's net. Split settlement divides the
bank credit that mirrors that net. Running them in registration order meant some
settlements had their credit created, then their net changed underneath it.

**Fix.** Four explicit phases — `structural → pre_bank → post_bank → books` — with
bank credits materialised between `pre_bank` and `post_bank`, and ground truth
*derived from final world state* rather than accumulated as patterns run. A pattern
that moves a line cannot desynchronise the key, because the key is a pure function of
the world at the end.

**Lesson.** The generator needs the same design rigour as the engine. A subtly wrong
answer key produces confidently wrong metrics, and nothing downstream can detect it.

---

## 6. The ablation disabled more than it claimed

**Symptom.** "GlassLedger tier 1 only" reported a **0.0% match rate**. Tier 1
resolves most of the volume; 0% is not a plausible number.

**Cause.** I disabled tier 2 by setting `threshold=1.01`. Tier 1 asserts confidence
exactly 1.0, so `1.0 >= 1.01` is false — the threshold disabled tier 1 as well.

**Fix.** `max_tier` on the engine, which caps tiers rather than abusing the
threshold. Tier 1 only now reads **59.6% match at 100% precision**, which is the
actual, meaningful finding: deterministic identity matching gets you 60% of pairs
and *never lies*; the remaining 40% is the long tail that needs everything else.

**Lesson.** A broken ablation is worse than no ablation, because its output looks
like a result. I nearly published "tier 1 gets 0%" as though it meant something.

---

## 7. Float money, caught by a property test I almost did not write

**Symptom.** `test_format_parse_round_trip` failed on `p = -1`:
`from_rupee_string(inr(-1))` raised on `'₹-0.01'`.

**Cause.** Two bugs stacked. `inr()` concatenated the rupee sign in front of an
already-signed string, producing `₹-0.01` instead of the conventional `-₹0.01`.
Then `from_rupee_string` tested `startswith('-')` *before* stripping the currency
mark — so the `₹` hid the minus, the sign test failed, and the later `replace`
removed the symbol without reconsidering. **A one-paise refund parsed as positive.**

**Why it matters more than the amount suggests.** A silently sign-flipped refund is
close to the worst single-character bug available in this domain. It does not crash.
It does not look wrong. It just moves money the other way.

**Fix.** Strip currency marks first, then determine sign. `inr()` emits `-₹0.01`.

**Lesson.** No example-based test I would have thought to write covers `-1` paise.
The Hypothesis property found it in seconds, and it is the reason the money module
is property-tested rather than example-tested.

---

## 8. `sqrt(a) * sqrt(b)` is not `sqrt(a*b)`

**Symptom.** `char_ngram_cosine('0001', '0001')` returned `0.9999999999999998`.

**Cause.** Normalising with two separate square roots. For identical vectors the
sums of squares are equal, so `sqrt(sa*sb)` is exactly `sa` and the ratio is exactly
1.0 — but `sqrt(sa) * sqrt(sb)` lands a few ulps off.

**Fix.** One `sqrt` of the product, plus a clamp.

**Why it is in here at all.** Two ulps of a similarity score changes nothing about
any decision. But every consumer treats this as a bounded `[0,1]` feature, and a
value that is *almost* 1.0 for an exact match is a genuinely miserable thing to
debug six months later inside a serialised model artefact. Cheap to fix now,
expensive to diagnose later.

---

## 9. The benchmark was measuring sklearn's import

**Symptom.** `gateway_bank` reported 2,488ms against `gateway_books` at 339ms — the
opposite of the true ratio, since the bank leg does far less work.

**Cause.** sklearn's first `predict_proba` pays a one-off ~2.4s JIT/import cost. It
landed inside the timed window, on whichever leg ran first.

**Fix.** Warm the scorer before starting the clock.

**Cost.** Reported throughput went from 570 to **4,254 txn/s** — a 7.5× correction
that was entirely measurement error. **Lesson:** a benchmark that measures library
warm-up and calls it throughput would have sent me optimising the wrong code, and
the published number would have been wrong in my own favour's *opposite* direction,
which is the only reason I bothered to look.

---

## 10. The dataset stopped describing the merchant it claimed to

**Symptom.** Mean settlement ₹85,000 — above the ₹50,000 materiality gate. So most
matches were gated, and every policy number in the eval was really measuring "what
happens when almost everything needs human approval".

**Cause.** 10% of payments drawn uniformly from ₹5,000–₹500,000 as a "B2B tail".
That tail dominated the mean. The dataset had quietly become a B2B merchant.

**Fix.** 4% at ₹8,000–₹120,000. Mean settlement ₹21,000, ~9% of matches above the
gate — a D2C merchant with real large-transaction exposure.

**Lesson.** A generator parameter that seems like flavour can silently change what
every downstream metric *means*. The number that gave it away was not the mean — it
was noticing that the materiality gate was firing on most matches, which made me go
back and look at the distribution.

---

## Still open, honestly

- **Tier 3 has never run against the real API.** No `ANTHROPIC_API_KEY` in this
  environment. The plumbing — payload construction, schema, clamping, gate
  filtering, event recording, policy re-evaluation — is exercised end-to-end by
  `OfflineArbiter`, a **deterministic rule arbiter that is not an LLM** and is tagged
  as such everywhere it appears. Every published number is tiers 1+2 only. I would
  rather ship an unexercised tier that says so than quietly let a stub stand in for a
  model.

- **Set packing is greedy, and greedy is not optimal.** Optimal weighted set packing
  is NP-hard. The eval reports `set_packing_conflicts` (442 on the books leg) so the
  size of the compromise is a number rather than a hope, but it is a real
  approximation.

- **The model is slightly *under*-confident in the 0.6–0.8 band** — predicts 0.74,
  is right 100% of the time (n=22). Conservative in the safe direction, so it costs
  recall rather than precision, but it is genuine miscalibration and isotonic
  regression on ~800 threshold rows is the likely cause. More training data or a
  different calibration method would probably fix it.

- **`EventStore.touching` is a `LIKE` scan over JSON payloads.** Fine at 4,300
  events, wrong at a million, where the ids would need their own index table. The
  cost is documented in the docstring rather than hidden.

- **11 pools still hit the subset-search solution cap** on the books leg. They are
  reported, not swallowed — but they represent windows where coverage is genuinely
  incomplete.
