"""``glctl`` -- the command line for GlassLedger.

    python glctl.py generate --seed 42 --payments 900
    python glctl.py reconcile --data data/eval --explain
    python glctl.py exceptions --top 10
    python glctl.py audit --txn bank:BR1234567890
    python glctl.py verify
    python glctl.py serve

Every command reads from or writes to the event store, so the CLI and the API are
two views of the same history rather than two implementations of the same logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core import console  # noqa: E402
from app.core.config import IMMATERIAL_PAISE, MATERIALITY_PAISE  # noqa: E402
from app.events import EventStore, record_run, record_snapshot  # noqa: E402
from app.ingestion import load_batch  # noqa: E402
from app.llm import MAX_CONFIDENCE_DELTA  # noqa: E402
from app.matching import CalibratedScorer, DummyScorer, ReconciliationEngine  # noqa: E402
from app.projections import audit_trail, rebuild  # noqa: E402

DEFAULT_DB = ROOT / "data" / "glassledger.db"
DEFAULT_MODEL = ROOT / "data" / "model.pkl"
DEFAULT_DATA = ROOT / "data" / "eval"


def _money(paise: int) -> str:
    return console.money(paise)


def _scorer(model_path: Path):
    if model_path.exists():
        try:
            return CalibratedScorer.load(model_path)
        except ValueError as exc:
            print(f"  ! model artefact rejected: {exc}")
            print("  ! falling back to the heuristic scorer")
    return DummyScorer()


def _arbiter(mode: str):
    """Build the tier-3 arbiter, or ``None`` for "do not run the tier at all".

    ``off`` is the default and is not the same thing as ``null``. ``off`` means no
    tier-3 phase runs, so the output contains no tier-3 section and the result is
    identical to a build without the tier. ``null`` means the tier ran and every
    item came back "ask a human". Reporting the first as the second would claim a
    measurement that was never taken.
    """
    if mode == "off":
        return None
    from app.llm import default_arbiter

    return default_arbiter(mode)


def _rule(title: str = "", width: int = 92) -> None:
    if title:
        print(f"\n{title}")
    print("-" * width)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def cmd_generate(args) -> int:
    from data_generator.generate_synthetic import main as gen_main

    argv = [
        "--seed", str(args.seed), "--payments", str(args.payments),
        "--days", str(args.days), "--out", str(args.out),
        "--rate-multiplier", str(args.rate_multiplier),
    ]
    return gen_main(argv)


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

def cmd_reconcile(args) -> int:
    data = Path(args.data)
    print(f"GlassLedger reconcile  --  {data}")
    _rule()

    t0 = time.perf_counter()
    batch = load_batch(data)
    ingest_ms = (time.perf_counter() - t0) * 1000

    bs = batch.summary()
    print(f"  ingested {len(batch.all_txns)} transactions in {ingest_ms:.0f} ms")
    for s in bs["statements"]:
        flag = "verified" if s["balance_checked"] else "NOT VERIFIED"
        print(f"    {s['file']:<24} {s['format']:<7} {s['lines']:>4} lines  "
              f"running balance {flag} ({s['balance_verified_rows']} rows)")
    print(f"    {'gateway settlements':<24} {'json':<7} {bs['gateway_settlements']:>4}")
    print(f"    {'book entries':<24} {'csv':<7} {bs['book_entries']:>4}  "
          f"({bs['book_entries_with_order_ref']} carry a gateway order id)")

    scorer = _scorer(Path(args.model))
    arbiter = _arbiter(args.arbiter)
    engine = ReconciliationEngine(
        scorer, threshold=args.threshold,
        arbiter=arbiter, max_arbitrations=args.max_arbitrations,
    )
    if arbiter is not None:
        print(f"\n  tier 3: {getattr(arbiter, 'name', type(arbiter).__name__)} "
              f"(max {args.max_arbitrations} arbitrations)")
        if not getattr(arbiter, "available", True):
            print("    ! this arbiter is unavailable; every item will be routed to "
                  "a human and recorded as such")

    # Warm the scorer before the clock starts, and report the warm-up separately.
    # sklearn's first predict_proba pays a one-off ~2.5s import/JIT cost. With it
    # inside the measured window this command reported ~480 txn/s while the eval,
    # which warms first, reported ~4,500 for identical work on the same data.
    # Publishing two numbers for one thing is worse than either alone, so both are
    # measured and both are labelled.
    from app.core.config import FEATURE_NAMES

    t0 = time.perf_counter()
    scorer.score_many([{k: 0.5 for k in FEATURE_NAMES}])
    warm_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    run = engine.run(batch)
    match_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  matched with {run.scorer_name} at threshold {run.threshold:.4f} "
          f"in {match_ms:.0f} ms ({len(batch.all_txns) / (match_ms / 1000):.0f} txn/s)")
    print(f"    per-leg {run.timings_ms}"
          + (f"  ·  +{warm_ms:.0f} ms one-off scorer warm-up, excluded"
             if warm_ms > 50 else ""))
    completeness = run.assert_complete(batch)
    print(f"  completeness: {completeness['accounted_for']}/{completeness['ingested']} "
          f"transactions accounted for -- no silent drops")

    _rule("DECISIONS")
    for leg, lr in run.legs.items():
        d = lr.decisions
        conf_paise = sum(x.candidate.exposure_paise for x in d.auto_confirmed)
        prop_paise = sum(x.candidate.exposure_paise for x in d.proposed)
        print(f"  {leg}")
        print(f"    auto-confirmed      {len(d.auto_confirmed):>5}   {_money(conf_paise)}")
        print(f"    proposed for human  {len(d.proposed):>5}   {_money(prop_paise)}")
        gated = d.stats.get("gated_by_materiality", 0)
        if gated:
            print(f"      of which held by the {_money(MATERIALITY_PAISE)} materiality "
                  f"gate: {gated}")
        t1a = d.stats.get("tier1_above_materiality", 0)
        if t1a:
            print(f"      tier-1 identity matches above the gate: {t1a} "
                  f"({_money(d.stats.get('tier1_above_materiality_paise', 0))})")
        if d.exempted_immaterial:
            print(f"    written off (< {_money(IMMATERIAL_PAISE)})  "
                  f"{len(d.exempted_immaterial):>5}")
        print(f"    exceptions          {len(lr.exceptions.exceptions):>5}   "
              f"{_money(lr.exceptions.stats.get('amount_outstanding_paise', 0))}")
        if lr.wash_pairs:
            print(f"    wash pairs netted   {len(lr.wash_pairs):>5}   "
                  "(reversal + re-settlement)")

    if run.tier3_ran():
        _print_tier3(run, explain=args.explain)

    if args.explain:
        _rule("EVIDENCE  --  three confirmed matches, one per algorithm")
        seen: set[str] = set()
        for lr in run.legs.values():
            for d in lr.decisions.auto_confirmed:
                algo = d.candidate.algorithm
                if algo in seen:
                    continue
                seen.add(algo)
                _print_match(d)
                if len(seen) >= 4:
                    break
            if len(seen) >= 4:
                break

    # -- record --------------------------------------------------------
    db = Path(args.db)
    if args.fresh and db.exists():
        db.unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(str(db) + suffix)
            if p.exists():
                p.unlink()
    store = EventStore(db)
    t0 = time.perf_counter()
    summary = record_run(store, batch, run)
    record_ms = (time.perf_counter() - t0) * 1000

    proj = rebuild(store)
    metrics = proj.metrics()
    record_snapshot(store, f"batch-{data.name}", {
        "matches_confirmed": metrics["matches_confirmed"],
        "matches_pending_human": metrics["matches_pending_human"],
        "exceptions_open": metrics["exceptions_open"],
        "bank_leg_reconciled_paise": metrics["bank_leg_reconciled_paise"],
    })

    _rule("AUDIT CORE")
    print(f"  {summary.events_written} events written in {record_ms:.0f} ms "
          f"({summary.events_written / (record_ms / 1000):.0f} events/s)")
    for t, n in summary.by_type.items():
        print(f"    {t:<28}{n:>6}")
    v = store.verify_chain()
    print(f"  hash chain: {'OK' if v.ok else 'BROKEN'} -- {v.detail}")
    balanced, drift = proj.ledger_balanced()
    print(f"  ledger: {len(proj.journal_entries)} journal entries, "
          f"{'balanced' if balanced else f'OUT BY {drift} paise'}")
    for acct, bal in sorted(proj.ledger.items()):
        print(f"    {acct:<32}{_money(bal):>18}")
    print(f"\n  event store: {db}")
    store.close()
    return 0


def _print_match(d) -> None:
    c = d.candidate
    ev = c.evidence or {}
    print(f"\n  [{c.algorithm}]  tier {c.tier}  confidence {c.score:.4f}  {c.cardinality}")
    print(f"    {', '.join(c.left_ids)}")
    print(f"      -> {', '.join(c.right_ids)}")
    print(f"    {_money(c.left_amount_paise)} vs {_money(c.right_amount_paise)}  "
          f"residual {_money(c.residual_paise)}")
    print(f"    rule: {ev.get('rule', 'n/a')}")
    if ev.get("reference_comparison"):
        rc = ev["reference_comparison"]
        print(f"    reference: {rc.get('left')} vs {rc.get('right')}  "
              f"similarity {rc.get('similarity')}  exact={rc.get('exact')}")
    if ev.get("competing_candidates"):
        cc = ev["competing_candidates"]
        print(f"    competition: {cc.get('for_this_left')} candidates for this line, "
              f"{cc.get('amount_indistinguishable')} indistinguishable by amount")
    if ev.get("members"):
        for m in ev["members"][:4]:
            print(f"      + {m['txn_id']:<28}{_money(m['amount_paise']):>16}  {m['value_date']}")
    print(f"    decision: {d.action} ({d.reason})")


def _print_tier3(run, *, explain: bool = False) -> None:
    """What the arbitration tier was asked, what it answered, and what it cost.

    Printed as counts rather than a verdict. Every line here is a number that can be
    checked against the event log, including the ones that make the tier look bad --
    the singleton count, the disagreements, the errors. A tier that reports only its
    successes cannot be evaluated, and this one is the only tier in the system whose
    reasoning is not reproducible from the inputs.
    """
    t = run.tier3_totals()
    _rule(f"TIER 3  --  bounded arbitration by {t['arbiter']}")
    band = None
    for lr in run.legs.values():
        if lr.tier3 is not None:
            band = lr.tier3.stats.get("band")
            break
    if band:
        print(f"  band                  [{band[0]:.2f}, {band[1]:.4f})   "
              f"-- floor to the derived threshold, not a fixed window")
    print(f"  groups eligible       {t['eligible']:>5}")
    print(f"    of which singleton  {t['singleton_groups']:>5}   "
          "(one candidate, nothing to discriminate against)")
    print(f"  calls made            {t['calls_made']:>5}   "
          f"in {t['wall_ms']:.0f} ms")
    print(f"    proposed a match    {t['proposed_match']:>5}")
    print(f"    insufficient        {t['insufficient_evidence']:>5}   "
          "(a tie is a real answer)")
    print(f"    delta clamped       {t['clamped']:>5}   "
          f"(hard cap +/-{MAX_CONFIDENCE_DELTA})")
    if t.get("chose_unselected_rival"):
        print(f"    disagreed w/ solver {t['chose_unselected_rival']:>5}   "
              "(preferred a hypothesis the solver discarded; recorded, not applied)")
    if t["errors"]:
        print(f"    errors              {t['errors']:>5}   {t['error_kinds']}")
    print(f"  never offered         {t['skipped_above_materiality']:>5}   "
          f"above the {_money(MATERIALITY_PAISE)} materiality gate -- "
          "filtered before the payload was built")
    print(f"                        {t['skipped_outside_band']:>5}   outside the band")
    if t["capped_not_arbitrated"]:
        print(f"                        {t['capped_not_arbitrated']:>5}   "
              "over the per-run cap")
    if t["tokens_in"] or t["tokens_out"]:
        total = t["tokens_in"] + t["tokens_out"]
        print(f"  tokens                {t['tokens_in']} in / {t['tokens_out']} out "
              f"= {total} total, measured from the API's own usage figures")

    if explain and t["calls_made"]:
        print()
        for lr in run.legs.values():
            if lr.tier3 is None:
                continue
            for g, a in zip(lr.tier3.groups, lr.tier3.arbitrations):
                print(f"  [{a.arbiter}] {lr.leg}  group of {len(g)}")
                for i, c in enumerate(g, start=1):
                    mark = "->" if i == a.chosen_index else "  "
                    print(f"    {mark} {i}. {c.algorithm:<26} {c.score:.4f}  "
                          f"{', '.join(c.right_ids)}")
                if a.action == "propose_match":
                    print(f"       {a.action}: delta {a.confidence_delta:+.4f}"
                          + (f" (clamped from {a.raw_delta:+.4f})" if a.clamped else ""))
                    print(f"       {a.rationale}")
                    if a.evidence_cited:
                        print(f"       cited: {', '.join(a.evidence_cited)}")
                else:
                    print(f"       {a.action}: {a.reason}")
                print()


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------

def cmd_exceptions(args) -> int:
    store = EventStore(Path(args.db))
    proj = rebuild(store)
    excs = proj.open_exceptions()
    print(f"Open exceptions: {len(excs)}  "
          f"({_money(sum(abs(e.amount_paise) for e in excs))} outstanding)")
    _rule()
    by_cat: dict[str, int] = {}
    for e in excs:
        by_cat[e.category] = by_cat.get(e.category, 0) + 1
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<38}{n:>5}")

    _rule(f"TOP {args.top} BY PRIORITY (amount at risk, weighted by age)")
    for e in excs[: args.top]:
        print(f"\n  [{e.priority:6.2f}]  {e.category}   {_money(e.amount_paise)}   "
              f"{e.age_days}d old")
        print(f"    {', '.join(e.txn_ids)}")
        print(f"    -> {e.suggested_action}")
        for det in (e.evidence.get("txn_details") or [])[:2]:
            print(f"       {det['txn_id']}  {_money(det['amount_paise'])}  "
                  f"{det['value_date']}  {(det.get('narration') or '')[:52]}")
    store.close()
    return 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def cmd_audit(args) -> int:
    store = EventStore(Path(args.db))
    trail = audit_trail(store, args.txn)
    if not trail:
        print(f"No events touch {args.txn}.")
        store.close()
        return 1
    print(f"Audit trail for {args.txn}  --  {len(trail)} events")
    _rule()
    for ev in trail:
        print(f"\n  seq {ev['seq']:<6} {ev['event_type']:<26} {ev['occurred_at']}")
        print(f"    aggregate: {ev['aggregate_id']}")
        pl = ev["payload"]
        if ev["event_type"] == "MatchCandidateProposed":
            print(f"    {pl['algorithm']} (tier {pl['tier']}), confidence "
                  f"{pl['confidence']:.4f}, {pl['cardinality']}")
            print(f"    residual {_money(pl['residual_paise'])}")
            print(f"    rule: {(pl.get('evidence') or {}).get('rule', 'n/a')}")
            top = sorted(pl.get("features", {}).items(), key=lambda kv: -abs(kv[1]))[:6]
            print(f"    features: {', '.join(f'{k}={v:.3f}' for k, v in top)}")
            for r in pl.get("runners_up", [])[:2]:
                print(f"    runner-up: {r['right_ids']} at {r['score']:.4f} "
                      f"({r['algorithm']})")
        elif ev["event_type"] == "MatchConfirmed":
            print(f"    confirmed by {pl['confirmed_by']} via gate '{pl.get('gate')}'")
            print(f"    rationale: {pl.get('rationale')}")
        elif ev["event_type"] == "JournalEntryPosted":
            for leg in pl["legs"]:
                side = "Dr" if leg["debit_paise"] else "Cr"
                amt = leg["debit_paise"] or leg["credit_paise"]
                print(f"    {side} {leg['account']:<34}{_money(amt):>16}")
        elif ev["event_type"] == "ExceptionRaised":
            print(f"    {pl['category']}: {pl['suggested_action'][:80]}")
        elif ev["event_type"] == "Tier3ArbitrationRequested":
            print(f"    asked {pl['arbiter']} to choose between "
                  f"{pl['candidate_count']} candidates")
            for c in pl.get("candidates_offered", []):
                print(f"      {c['index']}. {c['algorithm']:<26} {c['score']:.4f}  "
                      f"{', '.join(c['right_ids'])}")
            print(f"    band {pl.get('band')}, delta capped at "
                  f"+/-{pl.get('max_confidence_delta')}")
        elif ev["event_type"] == "Tier3ArbitrationReturned":
            print(f"    {pl['arbiter']} -> {pl['action']}")
            if pl.get("action") == "propose_match":
                print(f"    chose candidate {pl['chosen_index']}, confidence "
                      f"{pl.get('score_before')} -> {pl.get('score_after')} "
                      f"(delta {pl['confidence_delta']:+})"
                      + ("  [CLAMPED]" if pl.get("clamped") else ""))
                print(f"    rationale: {pl.get('rationale')}")
                if pl.get("evidence_cited"):
                    print(f"    cited: {', '.join(pl['evidence_cited'])}")
            else:
                print(f"    reason: {pl.get('reason')}")
            if pl.get("error"):
                print(f"    error: {pl['error']}")
            print("    the policy layer decided what happened next; this event "
                  "changed a score, not a state")
        elif ev["event_type"] == "TransactionIngested":
            print(f"    {pl['source']} {_money(pl['amount_paise'])} {pl['value_date']} "
                  f"utr={pl.get('utr')}")
            prov = pl.get("provenance") or {}
            if prov.get("source_file"):
                print(f"    from {prov['source_file']} row {prov.get('row')}")
    print(f"\n  Every line above was reconstructed from the append-only log, not "
          f"from a summary.")
    store.close()
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(args) -> int:
    store = EventStore(Path(args.db))
    print("Integrity checks")
    _rule()

    v = store.verify_chain()
    print(f"  hash chain             {'PASS' if v.ok else 'FAIL'}  {v.detail}")

    p1 = rebuild(store)
    p2 = rebuild(store)
    same = p1.fingerprint() == p2.fingerprint()
    print(f"  replay determinism     {'PASS' if same else 'FAIL'}  "
          f"fingerprint {p1.fingerprint()[:24]}...")

    balanced, drift = p1.ledger_balanced()
    print(f"  double-entry balance   {'PASS' if balanced else 'FAIL'}  "
          f"drift {drift} paise across {len(p1.ledger)} accounts")

    # Append-only is enforced by the database, not by convention -- prove it.
    try:
        store._conn.execute("UPDATE events SET payload_json='{}' WHERE seq=1")
        store._conn.commit()
        mutable = True
    except Exception as exc:
        mutable = False
        detail = str(exc)
    print(f"  append-only enforced   {'PASS' if not mutable else 'FAIL'}  "
          f"{'UPDATE refused: ' + detail if not mutable else 'UPDATE SUCCEEDED'}")

    try:
        store._conn.execute("DELETE FROM events WHERE seq=1")
        store._conn.commit()
        deletable = True
    except Exception as exc:
        deletable = False
        detail = str(exc)
    print(f"  delete refused         {'PASS' if not deletable else 'FAIL'}  {detail}")

    ok = v.ok and same and balanced and not mutable and not deletable
    print(f"\n  {'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'}")
    store.close()
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# tamper -- demonstrate that the chain actually detects an edit
# ---------------------------------------------------------------------------

def cmd_tamper(args) -> int:
    """Edit history behind the application's back and show the chain catching it.

    Uses a throwaway copy of the store. The triggers block the application from
    mutating events; this bypasses them by dropping the triggers first, which is
    exactly what someone with file access would do. The point is that the hash chain
    still catches it -- integrity does not depend on the triggers surviving.
    """
    import shutil

    src = Path(args.db)
    if not src.exists():
        print(f"No event store at {src}; run `reconcile` first.")
        return 1
    tmp = src.with_name("tamper_demo.db")
    shutil.copy(src, tmp)

    store = EventStore(tmp)
    before = store.verify_chain()
    print(f"  before tampering : {'OK' if before.ok else 'BROKEN'} "
          f"({before.events_checked} events)")

    target = args.seq
    row = store._conn.execute(
        "SELECT payload_json FROM events WHERE seq=?", (target,)
    ).fetchone()
    if row is None:
        print(f"  no event at seq {target}")
        store.close()
        return 1
    payload = json.loads(row["payload_json"])
    original = json.dumps(payload)[:110]

    # Drop the triggers, edit the row, put them back -- the file-access attack.
    store._conn.executescript(
        "DROP TRIGGER IF EXISTS events_no_update;"
        "DROP TRIGGER IF EXISTS events_no_delete;"
    )
    if "amount_paise" in payload:
        payload["amount_paise"] = int(payload["amount_paise"]) + 100_00
        what = "inflated the amount by Rs 100"
    else:
        payload["tampered"] = True
        what = "added a field"
    store._conn.execute(
        "UPDATE events SET payload_json=? WHERE seq=?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), target),
    )
    store._conn.commit()
    store._conn.executescript(SCHEMA_TRIGGERS)

    print(f"  tampered with seq {target}: {what}")
    print(f"    was: {original}...")
    after = store.verify_chain()
    print(f"  after tampering  : {'OK' if after.ok else 'BROKEN'}")
    print(f"    {after.detail}")
    print(f"\n  The edit was made with the triggers dropped -- i.e. by someone with "
          f"direct\n  file access. The hash chain caught it anyway, at the exact "
          f"event.")
    store.close()
    tmp.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(tmp) + suffix).unlink(missing_ok=True)
    return 0


SCHEMA_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only: UPDATE is forbidden'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only: DELETE is forbidden'); END;
"""


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

def cmd_serve(args) -> int:
    import uvicorn

    from app.api.main import create_app

    app = create_app(db_path=Path(args.db))
    print(f"GlassLedger API on http://{args.host}:{args.port}  (store: {args.db})")
    print(f"  workbench: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    console.init()
    ap = argparse.ArgumentParser(
        prog="glctl", description="GlassLedger reconciliation control"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a synthetic dataset + answer key")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--payments", type=int, default=900)
    g.add_argument("--days", type=int, default=30)
    g.add_argument("--out", type=Path, default=DEFAULT_DATA)
    g.add_argument("--rate-multiplier", type=float, default=1.0)
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("reconcile", help="run the engine and record the events")
    r.add_argument("--data", type=Path, default=DEFAULT_DATA)
    r.add_argument("--db", type=Path, default=DEFAULT_DB)
    r.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    r.add_argument("--threshold", type=float, default=None)
    r.add_argument("--explain", action="store_true", help="print evidence for sample matches")
    r.add_argument("--fresh", action="store_true", help="start a new event store")
    r.add_argument(
        "--arbiter", default="off",
        choices=("off", "auto", "claude", "offline", "null"),
        help="tier-3 arbiter. off (default) = the tier does not run at all; "
             "claude = the real model (needs ANTHROPIC_API_KEY); "
             "offline = a deterministic rule arbiter that is NOT an LLM; "
             "null = ran but declined everything; auto = claude if a key is set",
    )
    r.add_argument(
        "--max-arbitrations", type=int, default=25,
        help="hard cap on tier-3 calls per run (default 25)",
    )
    r.set_defaults(fn=cmd_reconcile)

    e = sub.add_parser("exceptions", help="the ranked queue")
    e.add_argument("--db", type=Path, default=DEFAULT_DB)
    e.add_argument("--top", type=int, default=10)
    e.set_defaults(fn=cmd_exceptions)

    a = sub.add_parser("audit", help="replay everything that touched one transaction")
    a.add_argument("--txn", required=True)
    a.add_argument("--db", type=Path, default=DEFAULT_DB)
    a.set_defaults(fn=cmd_audit)

    v = sub.add_parser("verify", help="chain, replay, balance, append-only")
    v.add_argument("--db", type=Path, default=DEFAULT_DB)
    v.set_defaults(fn=cmd_verify)

    t = sub.add_parser("tamper", help="prove the hash chain detects an edit")
    t.add_argument("--db", type=Path, default=DEFAULT_DB)
    t.add_argument("--seq", type=int, default=3)
    t.set_defaults(fn=cmd_tamper)

    s = sub.add_parser("serve", help="run the API + workbench")
    s.add_argument("--db", type=Path, default=DEFAULT_DB)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    if getattr(args, "threshold", None) is None and args.cmd == "reconcile":
        sc = _scorer(Path(args.model))
        args.threshold = float(getattr(sc, "threshold", 0.90))
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
