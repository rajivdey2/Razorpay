"""The baseline: a competent rules-based reconciler.

This is the number GlassLedger's lift is measured against, so it matters that it
is not a straw man. Comparing a tiered engine against ``WHERE amount = amount``
produces an impressive-looking table that proves nothing -- any reader who has
built a reconciliation script knows the real baseline is better than that, and
publishing the weak comparison is worse than publishing no comparison, because it
signals either ignorance or salesmanship.

So this baseline does everything a good engineer would do in an afternoon with SQL
and a CSV reader:

* exact UTR match, extracted from bank narration with the same parser the engine
  uses -- the baseline gets the good ingestion layer for free
* exact amount + exact date fallback when no reference is available
* the deterministic ``order_id`` join on the books leg
* greedy nearest-amount matching inside a +/- 2-day window as a last resort

What it deliberately does not have is the four things that are the actual thesis:
no calibrated confidence, no optimal assignment (greedy only), no subset-sum for
batches and splits, and no notion of "I am not sure". It matches what it can and
silently takes its best guess on the rest -- which is exactly how a rules-based
reconciler behaves, and exactly why the interesting metric is not its match rate
but its *silent-wrong* rate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from app.ingestion import IngestedBatch


@dataclass
class BaselineResult:
    #: Every pair the baseline asserts. It has no confidence, so there is no
    #: distinction between "confirmed" and "proposed" -- everything is confirmed,
    #: which is the point.
    pairs: dict[str, set[tuple[str, str]]] = field(default_factory=dict)
    flagged: set[str] = field(default_factory=set)
    stats: dict = field(default_factory=dict)


#: Greedy fallback window. Two days either side of the settlement date, which is
#: what a hand-written script typically uses for a T+2 cycle.
GREEDY_WINDOW_DAYS = 2


def run_baseline(batch: IngestedBatch) -> BaselineResult:
    res = BaselineResult(pairs={"gateway_bank": set(), "gateway_books": set()})
    stats: dict = {}

    # ---- gateway -> bank -------------------------------------------------
    gateway = batch.gateway.settlements
    bank = [t for t in batch.bank if t.amount_paise > 0]

    by_ref: dict[str, list] = defaultdict(list)
    for t in bank:
        for ref in (t.ref_candidates or ()):
            by_ref[ref].append(t)

    used_bank: set[str] = set()
    n_utr = n_amount_date = n_greedy = 0

    for s in gateway:
        if not s.utr:
            continue
        for cand in by_ref.get(s.utr, ()):
            if cand.txn_id in used_bank:
                continue
            if cand.amount_paise == s.amount_paise:
                res.pairs["gateway_bank"].add((s.txn_id, cand.txn_id))
                used_bank.add(cand.txn_id)
                n_utr += 1
                break

    matched_gateway = {l for l, _ in res.pairs["gateway_bank"]}

    # Exact amount + exact date, for settlements whose reference did not resolve.
    by_amount_date: dict[tuple[int, object], list] = defaultdict(list)
    for t in bank:
        by_amount_date[(t.amount_paise, t.value_date)].append(t)
    for s in gateway:
        if s.txn_id in matched_gateway:
            continue
        for cand in by_amount_date.get((s.amount_paise, s.value_date), ()):
            if cand.txn_id in used_bank:
                continue
            res.pairs["gateway_bank"].add((s.txn_id, cand.txn_id))
            used_bank.add(cand.txn_id)
            matched_gateway.add(s.txn_id)
            n_amount_date += 1
            break

    # Greedy nearest amount inside a small window -- the last resort, and the
    # source of most of the baseline's silent errors.
    for s in gateway:
        if s.txn_id in matched_gateway:
            continue
        best = None
        for t in bank:
            if t.txn_id in used_bank:
                continue
            if abs((t.value_date - s.value_date).days) > GREEDY_WINDOW_DAYS:
                continue
            delta = abs(t.amount_paise - s.amount_paise)
            if delta > max(100, abs(s.amount_paise) // 20):  # within 5%
                continue
            if best is None or delta < best[0]:
                best = (delta, t)
        if best is not None:
            res.pairs["gateway_bank"].add((s.txn_id, best[1].txn_id))
            used_bank.add(best[1].txn_id)
            matched_gateway.add(s.txn_id)
            n_greedy += 1

    stats["gateway_bank"] = {
        "exact_utr": n_utr,
        "exact_amount_date": n_amount_date,
        "greedy_nearest": n_greedy,
        "settlements_unmatched": len(gateway) - len(matched_gateway),
    }
    res.flagged |= {s.txn_id for s in gateway if s.txn_id not in matched_gateway}
    res.flagged |= {t.txn_id for t in batch.bank if t.txn_id not in used_bank}

    # ---- books -> gateway ------------------------------------------------
    n_join = 0
    matched_entries: set[str] = set()
    for e in batch.books.entries:
        order_ref = (e.provenance or {}).get("order_ref") or ""
        sid = batch.gateway.order_to_settlement.get(order_ref)
        if sid:
            res.pairs["gateway_books"].add((e.txn_id, f"gateway:{sid}"))
            matched_entries.add(e.txn_id)
            n_join += 1

    # Amount-and-date heuristic for the unkeyed remainder: attach an entry to a
    # settlement whose *net* is within 12% and whose date is within the window. A
    # crude rule, and a common one, because without a key there is nothing else
    # a SQL-shaped approach can do.
    n_heur = 0
    for e in batch.books.entries:
        if e.txn_id in matched_entries or e.amount_paise <= 0:
            continue
        for s in gateway:
            delta_days = (s.value_date - e.value_date).days
            if not (0 <= delta_days <= 6):
                continue
            if abs(s.amount_paise - e.amount_paise) <= abs(s.amount_paise) * 12 // 100:
                res.pairs["gateway_books"].add((e.txn_id, s.txn_id))
                matched_entries.add(e.txn_id)
                n_heur += 1
                break

    stats["gateway_books"] = {
        "order_key_join": n_join,
        "amount_date_heuristic": n_heur,
        "entries_unmatched": len(batch.books.entries) - len(matched_entries),
    }
    res.flagged |= {
        e.txn_id for e in batch.books.entries if e.txn_id not in matched_entries
    }
    res.stats = stats
    return res
