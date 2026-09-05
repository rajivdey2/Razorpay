# GlassLedger — Multi-Source Reconciliation Agent
### Razorpay AI Buildathon 2026 — Track 04: AI Finance Controller
**Builder:** Max, NIST Berhampur (CSE, 2024–2028) · **Repo target:** `rajivdey2/glassledger`

---

## 0. Why this track, and why this doc exists

All four tracks Razorpay listed map to products they've *already shipped* — Agentic Payments and Agent Studio (Track 1), the Dispute Auto-Responder and RTO Shielder (Track 2), the Subscription Recovery Agent and Intelligent Retry Engine (Track 3), and the Bookkeeping/Reporting/Insights Agents inside Agentic Business Banking (Track 4). The judging panel builds these for a living. A hackathon clone of any of them reads as "I watched the demo video." What reads as hire-me signal is picking the one track where you can go **one layer deeper than the marketing page**, into the part of the problem that's genuinely unsolved.

Razorpay said it themselves in the Track 4 brief: *verification capacity, not generation speed, is the bottleneck* — reconciliation, settlement and forecasting are still done by hand, and the bar is *throughput plus measured accuracy plus an honest exception list*, because *one cherry-picked match proves nothing.*

That sentence is the whole project. It's not asking for a chatbot that reads a CSV. It's asking: **can you build something that knows what it doesn't know, at scale, and prove it with numbers instead of a demo gif?**

**Verdict: build Track 04, direction "Multi-source reconciliation."** It's the hardest of the four example directions to fake, it's a straight extension of the CQRS + event-sourcing ledger you already shipped (so you're not starting from zero on the one component — the audit core — that's genuinely hard to get right), and it lets you demonstrate exactly the muscle Razorpay's own Bookkeeping Agent doesn't: their agent posts entries "based on predefined rules." Yours handles the cases rules can't — the fuzzy, partial, many-to-one, timing-drifted 15% that every finance team still does by hand in a spreadsheet at 11pm on closing day — and it never lies about its confidence while doing it.

---

## 1. The problem, precisely

A mid-size D2C merchant on Razorpay has (at minimum) three independent records of "the same" money, and they never agree perfectly:

1. **Gateway settlement records** — Razorpay's `settlement` entity: `id`, `amount`, `fees`, `tax`, `utr`, `status`, `created_at`, amounts in paise. One settlement batch usually bundles many individual payments.
2. **Bank statement lines** — CSV/MT940-style exports from the merchant's current account, referencing the UTR (if you're lucky), a truncated/garbled narration (if you're not), a credit amount, and a value date.
3. **Internal books** — the merchant's own ledger/ERP: sales invoices, expected receivables, what *should* have landed.

Reconciliation is the act of proving, line by line, that these three views describe the same underlying reality — and flagging, with evidence, every place they don't. The reasons they don't agree are not exotic edge cases; they are the **majority of real transaction volume** at any merchant with meaningful GMV:

| # | Break pattern | Why it happens |
|---|---|---|
| 1 | Batch settlement (N payments → 1 payout) | Razorpay settles in batches, not per-transaction |
| 2 | Split settlement (1 payment → N payouts) | Partial holds, risk reviews, phased release |
| 3 | Refund netting | A settlement batch nets out refunds issued since the last cycle |
| 4 | Fee/tax drift | MDR% + GST-on-fee rounding differs from the merchant's own fee assumption |
| 5 | TDS mismatch | Statutory deductions the gateway doesn't see but the merchant's books expect |
| 6 | Duplicate bank entry | Reversal + re-settlement shows twice in the bank feed |
| 7 | Orphan bank credit | Money in the bank with no matching gateway record (manual adjustment, interest, another PSP) |
| 8 | Missing settlement | Gateway marks `processed`, bank hasn't credited yet — NEFT/RTGS window, or genuinely stuck |
| 9 | Timing drift | T+2 lands on a bank holiday, statement date ≠ value date |
| 10 | Narration noise | Bank truncates or merges UTR with other text in the description field |
| 11 | FX mismatch | International payment settles in a different currency snapshot than invoiced |

Naive reconciliation (`WHERE amount = amount AND date = date`) catches maybe 70–85% of volume — the clean 1:1 cases. The remaining 15–30% is where finance teams actually spend their week, and it's exactly the part that determines whether an "AI finance controller" is a toy or a tool. **Solving that long tail — correctly, provably, and without ever silently misposting money — is the hard part**, and it's an assignment/matching problem, not a lookup problem: matching N settlement lines against M bank lines against K book entries under uncertainty is a constrained combinatorial optimization, not a `JOIN`.

---

## 2. Product framing

**GlassLedger** is an agent that closes the settlement-reconciliation loop across gateway, bank, and books — and instead of trying to resolve 100% of transactions (which would mean lying about the ones it's unsure of), it resolves everything it can prove, and produces a ranked, evidence-backed exception list for everything it can't. Every decision — matched or unmatched — is an immutable, replayable event, so a human (or an auditor) can ask "why did the system do that" six months later and get a real answer, not a shrug.

**One-line pitch:** *"An AI finance controller that never guesses with your money — it matches what it can prove, and shows its work on what it can't."*

---

## 3. System architecture

```
                         ┌─────────────────────────────────────────┐
                         │              INGESTION LAYER              │
                         │  Razorpay Settlements API (test mode)     │
                         │  Bank statement CSV parser (3 bank formats)│
                         │  Internal books CSV/API                   │
                         └───────────────────┬───────────────────────┘
                                              │  normalize → canonical txn schema
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │           MATCHING ENGINE (tiered)        │
                         │                                           │
                         │  Tier 1 — Deterministic                   │
                         │    exact UTR / exact amount+date          │
                         │                                           │
                         │  Tier 2 — Constrained optimization        │
                         │    windowed bipartite graph +             │
                         │    Hungarian algorithm (1:1)              │
                         │    bounded subset-sum search (N:1 / 1:N)  │
                         │    → calibrated confidence score          │
                         │                                           │
                         │  Tier 3 — LLM arbitration (bounded)       │
                         │    only for confidence ∈ [0.40, 0.75]     │
                         │    structured tool-call output only       │
                         │    hard rule: > ₹50,000 always needs      │
                         │    human sign-off regardless of score     │
                         └───────────────────┬───────────────────────┘
                                              │  every decision → event
                                              ▼
                         ┌─────────────────────────────────────────┐
                         │        EVENT-SOURCED AUDIT CORE (CQRS)    │
                         │  Postgres append-only event store         │
                         │  Command side: ingestion + match writes   │
                         │  Query side: projected read models        │
                         └──────┬───────────────────────┬────────────┘
                                │                         │
                                ▼                         ▼
                 ┌───────────────────────┐   ┌───────────────────────────┐
                 │  Exception Workbench   │   │  Metrics & Eval Dashboard  │
                 │  (React/TS)            │   │  match rate, precision,   │
                 │  accept/reject/reassign│   │  recall, ₹ reconciled,    │
                 │  agent's rationale shown│   │  exception aging          │
                 └───────────────────────┘   └───────────────────────────┘
```

---

## 4. The matching engine — where the actual difficulty lives

### 4.1 Canonical transaction schema

```python
class NormalizedTxn(BaseModel):
    source: Literal["gateway", "bank", "books"]
    external_id: str            # settlement id / bank ref / invoice id
    amount_paise: int           # always smallest unit — no float money, ever
    currency: str
    utr: str | None
    narration: str | None
    value_date: date
    fees_paise: int = 0
    tax_paise: int = 0
    raw_payload_hash: str       # sha256 — idempotency key for re-ingestion
```

Money is an integer in paise everywhere in the system. No floats touch a currency field — this alone is worth stating explicitly in the pitch, because it's the kind of decision a judge who's shipped payments infra actually notices.

### 4.2 Tier 1 — deterministic exact match

Cheap, fast, explainable. Exact UTR match, or exact `(amount, value_date ± 0)` match. This clears the easy 70–85% of volume in milliseconds and should never touch the LLM tier — running an LLM over transactions that a hash lookup can resolve is both slower and less trustworthy than the alternative.

### 4.3 Tier 2 — constrained optimization for the hard cases

Model unresolved transactions as a **weighted bipartite graph**: gateway settlements on one side, bank lines on the other, edges only between candidates inside a *blocking window* (± settlement-cycle days, amount within a tolerance band derived from expected fees) — blocking is what keeps this tractable, since full N×M matching is not something you want to run unconstrained.

Edge cost combines:
- `|amount_delta|` normalized against expected fee/tax range
- `|date_delta|` in days, weighted by acquirer-specific settlement cycle priors
- UTR similarity (Levenshtein on partial/truncated UTRs)
- narration text similarity (TF‑IDF cosine)

For clean 1:1 candidates: solve with the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) over the windowed subgraph — optimal assignment, not greedy nearest-match, which matters once volumes get large enough that greedy starts compounding errors.

For batch (N:1) and split (1:N) settlements: bounded **subset-sum search** within each window (small N per window after blocking, so exhaustive-with-pruning or DP is fine) to find combinations of bank/gateway lines that sum to within a cent-level tolerance of a candidate settlement.

Each proposed match gets a **calibrated confidence score** — not a raw model output, a probability that's been checked against reality. Train a small gradient-boosted classifier (XGBoost/LightGBM) on the synthetic labeled set (Section 6) using the same features as the cost function, then calibrate with isotonic regression or Platt scaling, and validate with a **reliability diagram + Brier score** — i.e., prove that when the system says "82% confident," it's actually right about 82% of the time. This is the single most important piece of ML maturity to show, because raw LLM/model confidence is well known to be miscalibrated, and a finance agent that doesn't know this about itself is a liability, not a product.

### 4.4 Tier 3 — bounded LLM arbitration

Only invoked for matches landing in an ambiguous confidence band (e.g., 0.40–0.75) after Tier 2 — the genuinely hard remainder, not the whole dataset. The LLM (Claude, via the Anthropic API, using strict tool-calling / structured output) receives:

- the top-3 candidate matches with their feature breakdown (not raw dumps of unrelated data)
- surrounding transaction history for context
- the business rules it must obey

It is constrained to one of exactly two outputs: `{action: "propose_match", candidate_id, rationale}` or `{action: "insufficient_evidence", reason}` — never freeform prose that could bury a hallucinated number. And a **hard-coded business rule sits outside the model entirely**: any match above a materiality threshold (e.g., ₹50,000) requires human approval no matter what confidence anyone reports. This is the "bounded and gated" property that separates a demo from something a finance team could actually trust — the LLM is a *proposer*, never an *approver*, above the line that matters.

---

## 5. Event-sourced audit core (your existing edge)

This is the component where prior ledger/CQRS work pays off directly — reuse the pattern, not just the idea.

**Event types (append-only, immutable):**
```
TransactionIngested       {source, external_id, payload_hash, ingested_at}
MatchCandidateProposed    {match_id, tier, algorithm, candidates[], confidence, features}
MatchConfirmed            {match_id, confirmed_by: "agent" | "human", rationale, confidence}
MatchRejected             {match_id, reason, rejected_by}
ExceptionRaised           {exception_id, txn_ids[], category, max_confidence, suggested_action}
ExceptionResolved         {exception_id, resolution, resolved_by}
JournalEntryPosted        {entry_id, debit_account, credit_account, amount_paise, source_match_id}
ReconciliationSnapshot    {batch_id, match_rate, precision, recall, exceptions_count, ts}
```

**Idempotency:** every ingested transaction is keyed by `sha256(source, external_id, payload)` — replaying a webhook or re-running a batch never double-posts. This is a real production concern, not hackathon theater — it's the difference between a demo and something you'd trust with a merchant's books.

**CQRS split:** command side (ingestion + matching) writes events only; query side serves the dashboard and exception workbench from denormalized projections rebuilt from the event stream. Full replay means any journal entry can be traced back to the exact evidence and rationale that produced it — an auditor's question ("why did the agent match these two on March 5th?") gets answered by replaying events, not by guessing.

**Reversibility:** because nothing is a direct mutation, an incorrect auto-match is undone by appending `MatchRejected` + a corrective entry — the history stays intact, which matters both for compliance and for debugging your own agent's mistakes.

---

## 6. Synthetic data generator + eval harness

This section exists because "the bar" explicitly disqualifies a cherry-picked demo. Build the generator *before* the matcher, and treat it as a first-class deliverable, not test scaffolding.

**Generator design:**
- Produces a ground-truth batch of 200–500 "clean" triplets (gateway settlement ↔ bank line ↔ book entry), each with a known correct mapping.
- Deliberately injects the 11 break patterns from Section 1 at configurable rates (e.g., 8% batch settlements, 5% duplicates, 4% orphan credits, 3% missing settlements, etc.) — every injection is logged with the ground truth, so you always know the right answer even when the system doesn't.
- Three distinct bank statement CSV formats (mimicking HDFC/ICICI/generic MT940-style exports) to force the ingestion layer to actually generalize, not memorize one schema.

**Metrics reported (not chosen after the fact — decided up front, in the repo README, before the first run):**
- **Match rate** — % of gateway transactions resolved with confidence above the auto-confirm threshold
- **Precision / recall / F1** on the held-out labeled set, computed against known ground truth
- **₹ reconciled vs ₹ total** — money-weighted accuracy, which matters more than transaction-count accuracy to a CFO
- **Exception list honesty** — of everything the system *didn't* auto-resolve, what fraction did a human reviewer agree was genuinely ambiguous (vs. the system just being wrong)
- **Baseline comparison** — naive exact-match-only reconciliation vs. GlassLedger's tiered approach, on the identical dataset, so the lift is a number, not a claim

Publish a small table like this in the repo (numbers are illustrative — replace with your actual run):

| Method | Match rate | Precision | Recall | ₹ reconciled |
|---|---|---|---|---|
| Naive exact-match | 74% | 99% | 74% | 71% |
| GlassLedger (Tier 1+2) | 91% | 97% | 91% | 89% |
| GlassLedger (full, incl. Tier 3) | 96% | 95% | 96% | 95% |
| **Honest exceptions remaining** | 4% flagged, 0% silently wrong | | | |

That last row is the entire pitch in one line.

---

## 7. Stretch goals (only after the core loop is solid)

Don't start these until the reconciliation loop has real numbers. Pick at most one if time allows:

- **Settlement Q&A agent** — RAG over the event store; answers "why wasn't payment `pay_xxx` settled?" by replaying its own audit trail. Cheap to build once the event core exists, and it's a great demo moment because the answer is provably grounded, not generated.
- **Forward cash forecaster** — predicts T+2/T+3 settlement cash position from historical cycle patterns + pending payment pipeline; report a prediction interval, not a point estimate (and back-test it).
- **Tax-line matcher** — reconciles GST/TDS line items between the gateway's tax breakdown and the merchant's books; keep claims about specific statutory rates general in the pitch unless you've verified them against current law, since getting compliance details wrong in front of a fintech panel is worse than not mentioning them.

---

## 8. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI (async), PostgreSQL, SQLAlchemy async, Alembic | Same stack as your ledger project — no new learning curve on infra |
| Matching engine | Python, `scipy.optimize.linear_sum_assignment`, `networkx`, XGBoost/scikit-learn | Standard, explainable, fast enough for the batch sizes in scope |
| LLM tier | Claude via Anthropic API, strict tool-calling / structured outputs | Constrained output prevents hallucinated numbers from reaching the ledger |
| Frontend | React + TypeScript + Vite, Tailwind, Recharts | Matches your existing frontend stack |
| Testing | pytest, Hypothesis (property-based tests on the matcher) | Matching engines have nasty edge cases; property tests catch them cheaper than manual test-writing |
| Deployment | Backend on Render, frontend on Vercel | Reuse what you already know how to debug |

---

## 9. Build plan (10–14 day version, compressible to 72 hours)

| Phase | Days | Deliverable |
|---|---|---|
| 1. Foundations | 1–2 | Canonical schema, event store skeleton, ingestion parsers for all 3 sources |
| 2. Synthetic generator | 2–3 | Ground-truth batch generator with all 11 break patterns, seeded and reproducible |
| 3. Tier 1 + Tier 2 matcher | 3–4 | Deterministic + Hungarian/subset-sum matching, confidence scoring, calibration |
| 4. Audit core + projections | 2 | Full event sourcing, CQRS read models, replay capability |
| 5. Tier 3 LLM arbitration | 1–2 | Bounded, structured-output arbitration for the ambiguous band only |
| 6. Exception workbench UI | 1–2 | Accept/reject/reassign, rationale display, metrics dashboard |
| 7. Eval + polish | 1–2 | Final metrics run, baseline comparison table, README, architecture diagram |
| 8. Submission | 1 | 5-minute pitch video, "what broke" writeup, repo cleanup |

**If compressing to 72 hours:** cut to Tier 1 + Tier 2 only (skip Tier 3 LLM arbitration and all stretch goals), but keep the synthetic generator and honest metrics — those two things are what the judging bar actually asks for, and a solid Tier 1+2 system with real numbers beats a flashy but unverified LLM demo.

---

## 10. What actually breaks (write this down as you go — it's graded)

The submission explicitly asks you to explain what broke and how you recovered. Keep a running `POSTMORTEM.md` from day one — don't reconstruct it at the end. Likely candidates worth documenting honestly if they happen:

- Blocking window too tight/loose in Tier 2 → either missed valid matches or combinatorial blowup on subset-sum search
- Confidence calibration looking good in aggregate but badly miscalibrated on one break-pattern subclass (very likely — check per-category reliability, not just overall)
- LLM tier occasionally proposing a syntactically valid but financially wrong match when two candidates are near-identical — this is exactly why the materiality gate exists; if it happens, it's a good story, not a hidden bug
- Async SQLAlchemy + Alembic migration issues (you've hit these before on the ledger project — document the recurrence, it shows the lesson stuck)

A judge reading a postmortem that says "the LLM tier initially proposed a wrong high-confidence match on near-duplicate transactions, so we moved the materiality gate to be a hard rule outside the model instead of a soft prompt instruction" is reading exactly the kind of engineering judgment this program is filtering for.

---

## 11. 5-minute pitch video — suggested structure

1. **0:00–0:30** — The problem in one sentence, using their own framing: reconciliation is still done by hand because rules can't cover the long tail and LLMs alone can't be trusted with money.
2. **0:30–1:30** — Live demo: run the batch, show the dashboard, show the match rate/precision/recall table computed live, not pre-baked.
3. **1:30–2:30** — Open one exception. Show the agent's evidence and rationale. Show the materiality gate refusing to auto-confirm a large ambiguous match. This is the moment that proves it's not a toy.
4. **2:30–3:30** — Open the audit trail for one matched transaction. Replay it. "Here's exactly why the system did what it did, six months from now, for an auditor."
5. **3:30–4:15** — The numbers: baseline vs. GlassLedger, honestly, including the exception rate.
6. **4:15–5:00** — What broke, what you'd build next (Q&A agent / forecaster), and one sentence on why this complements rather than duplicates the Bookkeeping Agent already in RazorpayX.

---

## 12. Repo structure

```
glassledger/
├── README.md                 # architecture, setup, eval numbers up top
├── POSTMORTEM.md              # running log of what broke
├── backend/
│   ├── app/
│   │   ├── ingestion/         # gateway/bank/books parsers
│   │   ├── matching/          # tier1/tier2/tier3 engines
│   │   ├── events/            # event store, event types
│   │   ├── projections/       # CQRS read models
│   │   ├── api/                # FastAPI routers
│   │   └── llm/                # bounded Claude arbitration client
│   ├── tests/
│   │   ├── test_matching.py
│   │   ├── test_idempotency.py
│   │   └── property/          # Hypothesis-based edge case tests
│   └── alembic/
├── data-generator/
│   ├── generate_synthetic.py
│   └── break_patterns/        # one module per injected break type
├── frontend/
│   ├── src/
│   │   ├── ExceptionWorkbench/
│   │   └── MetricsDashboard/
│   └── vite.config.ts
└── eval/
    ├── run_eval.py             # baseline vs GlassLedger, reproducible
    └── results/
```

---

## 13. Why this should genuinely land

- **It answers the exact bar Razorpay wrote, not a reinterpretation of it** — throughput, measured accuracy, honest exceptions, on a real batch size.
- **It's provably not a wrapper.** The Hungarian-algorithm assignment, the calibrated confidence, and the event-sourced audit trail are all things a thin GPT-call submission won't have — and a panel that ships this stuff daily will notice the difference in about ten seconds of looking at the repo.
- **It shows you know what already exists.** The pitch explicitly distinguishes itself from the Bookkeeping Agent (rules-based) instead of pretending to have invented reconciliation — that's the difference between a student pitch and someone who's read the company's own docs.
- **The honesty is the feature.** A system that says "I'm not sure about these 4%, here's exactly why" is more valuable in a finance context than one that claims 100% — and demonstrating that you understand *why* is worth more to a hiring panel than any single metric.

Good luck — build the generator first, keep the postmortem honest, and don't let Tier 3 touch anything above the materiality line.
