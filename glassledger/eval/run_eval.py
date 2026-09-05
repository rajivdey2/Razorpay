"""Reproducible evaluation: train on seed 7, measure on seed 42, print the table.

    python eval/run_eval.py                       # full run, all methods
    python eval/run_eval.py --skip-train          # reuse the model artefact
    python eval/run_eval.py --rate-multiplier 2   # stress test: double break rates

Everything published about this system comes out of this one script, and it is
written so that a reader can re-run it and get the same numbers. That means:

* the training dataset (seed 7) and the eval dataset (seed 42) are separate, and
  the model never sees the eval set
* the auto-confirm threshold is *derived* from the training run's calibration
  curve, never tuned against the eval numbers
* a dataset fingerprint is printed alongside every table, so a number can be tied
  to the exact bytes it was computed on
* the ablations (baseline / tier 1 only / +tier 2 heuristic / +calibrated) all run
  on the identical dataset in the same process

The order of operations matters and is enforced by the script's structure: the
threshold is fixed before the eval set is loaded. There is no point in the run
where an eval number could influence a choice that affects an eval number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "eval")]

from app.core import console  # noqa: E402
from app.core.config import (  # noqa: E402
    FEATURE_NAMES,
    MATERIALITY_PAISE,
    TARGET_PRECISION,
    TIER3_BAND,
)
from app.core.money import inr  # noqa: E402
from app.core.schema import Dataset  # noqa: E402
from app.ingestion import load_batch  # noqa: E402
from app.matching import (  # noqa: E402
    CalibratedScorer,
    DummyScorer,
    ReconciliationEngine,
    brier_score,
    build_training_data,
    calibration_by_slice,
    expected_calibration_error,
    reliability_diagram_points,
)

from baseline import run_baseline  # noqa: E402
from metrics import exception_honesty, score_leg, truth_pairs  # noqa: E402

from data_generator.generate_synthetic import fingerprint, generate  # noqa: E402


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate_json((path / "ground_truth.json").read_text(encoding="utf-8"))


def amounts_by_id(ds: Dataset) -> dict[str, int]:
    return {t.txn_id: t.amount_paise for t in (*ds.gateway, *ds.bank, *ds.books)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def collect_training_rows(train_dir: Path, train_ds: Dataset) -> tuple[list[dict], list[int], list[str]]:
    """Run the engine over the *training* set to harvest labelled candidates.

    Candidates have to come from the engine itself, not from an independent
    enumeration. The model's job at inference time is to score exactly the
    hypotheses this pipeline produces -- including its blocking decisions, its
    ambiguity counts, and its per-component predictions. Training on a different
    candidate distribution is the textbook train/serve skew, and here it would
    show up as a model that is well calibrated on paper and over-confident in the
    crowded windows that actually matter.
    """
    batch = load_batch(train_dir)
    # Heuristic scorer for candidate *generation* only. Which hypotheses get
    # enumerated does not depend on the score except through the assignment step,
    # and using a bootstrap scorer here is what breaks the circular dependency
    # (model needs candidates, assignment needs model).
    engine = ReconciliationEngine(DummyScorer(), threshold=0.0)
    run = engine.run(batch)

    X: list[dict] = []
    y: list[int] = []
    slices: list[str] = []

    for leg_name, leg_run in run.legs.items():
        truth = truth_pairs(train_ds, leg_name)
        pattern_by_pair = {}
        for link in train_ds.links_for(leg_name):
            for pair in link.pair_keys():
                pattern_by_pair[pair] = link.pattern

        # Winners and losers both. A model trained only on selected candidates
        # never learns what a bad candidate looks like, because the solver already
        # filtered them out -- it would be trained on a population it will never
        # see at inference time.
        pool = [
            *(d.candidate for d in leg_run.decisions.auto_confirmed),
            *(d.candidate for d in leg_run.decisions.proposed),
            *(d.candidate for d in leg_run.decisions.rejected),
            *leg_run.considered,
        ]
        for c in pool:
            if not c.features:
                continue
            label = 1 if c.pair_keys().issubset(truth) else 0
            X.append(c.features)
            y.append(label)
            pat = next(
                (pattern_by_pair[p] for p in c.pair_keys() if p in pattern_by_pair),
                "not_in_truth",
            )
            slices.append(f"{leg_name}/{pat}")

    return X, y, slices


# ---------------------------------------------------------------------------
# Evaluation of one configured engine
# ---------------------------------------------------------------------------

def evaluate_engine(label: str, engine: ReconciliationEngine, batch, ds: Dataset) -> dict:
    # Warm the scorer before timing. sklearn's first predict_proba pays a one-off
    # ~2.4s import/JIT cost, and leaving it inside the measured window put it on
    # whichever leg happened to run first -- making the gateway->bank leg look 8x
    # slower than the books leg when it is in fact the faster of the two. A
    # benchmark that measures library warm-up and calls it throughput is just
    # wrong, and wrong in a way that would send someone optimising the wrong code.
    from app.core.config import FEATURE_NAMES

    engine.scorer.score_many([{k: 0.5 for k in FEATURE_NAMES}])

    t0 = time.perf_counter()
    run = engine.run(batch)
    elapsed = (time.perf_counter() - t0) * 1000
    completeness = run.assert_complete(batch)

    amt = amounts_by_id(ds)
    out = {
        "label": label,
        "scorer": run.scorer_name,
        "threshold": round(run.threshold, 4),
        "wall_ms": round(elapsed, 1),
        "throughput_txn_per_s": round(len(batch.all_txns) / (elapsed / 1000), 1),
        "completeness": completeness,
        "legs": {},
    }

    all_flagged: set[str] = set()
    for leg_name, leg_run in run.legs.items():
        confirmed = set()
        proposed = set()
        for d in leg_run.decisions.auto_confirmed:
            confirmed |= d.candidate.pair_keys()
        for d in leg_run.decisions.proposed:
            proposed |= d.candidate.pair_keys()
        score = score_leg(leg_name, ds, confirmed, proposed, amt)
        flagged = {i for e in leg_run.exceptions.exceptions for i in e.txn_ids}
        all_flagged |= flagged
        out["legs"][leg_name] = {
            **score.to_json(),
            "exception_honesty": exception_honesty(ds, flagged, leg=leg_name),
            "exceptions": leg_run.exceptions.summary(),
            "policy": leg_run.decisions.stats,
        }

    out["exception_honesty_overall"] = exception_honesty(ds, all_flagged)
    out["tier3"] = run.tier3_totals() or None
    out["_run"] = run
    return out


def evaluate_baseline(batch, ds: Dataset) -> dict:
    t0 = time.perf_counter()
    res = run_baseline(batch)
    elapsed = (time.perf_counter() - t0) * 1000
    amt = amounts_by_id(ds)
    out = {
        "label": "Naive rules baseline",
        "scorer": "rules_only",
        "threshold": None,
        "wall_ms": round(elapsed, 1),
        "throughput_txn_per_s": round(len(batch.all_txns) / (elapsed / 1000), 1),
        "completeness": {"complete": None},
        "legs": {},
        "baseline_stats": res.stats,
    }
    for leg_name in ("gateway_bank", "gateway_books"):
        # A rules engine has no confidence, so everything it asserts is
        # auto-confirmed: passing the same set as both arguments is not a
        # shortcut, it is the accurate model of how such a system behaves.
        pairs = res.pairs[leg_name]
        score = score_leg(leg_name, ds, pairs, pairs, amt)
        out["legs"][leg_name] = {
            **score.to_json(),
            "exception_honesty": exception_honesty(ds, res.flagged, leg=leg_name),
        }
    out["exception_honesty_overall"] = exception_honesty(ds, res.flagged)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def combined(results: dict) -> dict:
    """Roll the two legs into one headline row."""
    tp = fp = fn = 0
    ctp = cfp = cfn = 0
    rec = tot = mis = 0
    sw = swp = 0
    for leg in results["legs"].values():
        a = leg["all_proposed_or_confirmed"]
        c = leg["auto_confirmed_only"]
        tp += a["tp"]; fp += a["fp"]; fn += a["fn"]
        ctp += c["tp"]; cfp += c["fp"]; cfn += c["fn"]
        rec += leg["money"]["reconciled_paise"]
        tot += leg["money"]["total_paise"]
        mis += leg["money"]["misreconciled_paise"]
        sw += leg.get("silent_wrong_matches", 0)
        swp += leg.get("silent_wrong_paise", 0)

    def pr(t, f):
        return t / (t + f) if (t + f) else 1.0

    return {
        "match_rate": ctp / (ctp + cfn) if (ctp + cfn) else 0.0,
        "precision": pr(tp, fp),
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "auto_precision": pr(ctp, cfp),
        "money_fraction": rec / tot if tot else 0.0,
        "misreconciled_paise": mis,
        "silent_wrong": sw,
        "silent_wrong_paise": swp,
    }


def print_headline_table(all_results: list[dict]) -> None:
    print()
    print("=" * 108)
    print("HEADLINE  --  identical dataset, identical ingestion, four methods")
    print("=" * 108)
    hdr = (
        f"{'Method':<34}{'Match':>8}{'Prec':>8}{'Recall':>8}{'AutoPrec':>10}"
        f"{'Money':>8}{'Silent':>8}{'Wrong money':>16}"
    )
    print(hdr)
    print("-" * 108)
    for r in all_results:
        c = combined(r)
        print(
            f"{r['label']:<34}"
            f"{c['match_rate'] * 100:>7.1f}%"
            f"{c['precision'] * 100:>7.1f}%"
            f"{c['recall'] * 100:>7.1f}%"
            f"{c['auto_precision'] * 100:>9.1f}%"
            f"{c['money_fraction'] * 100:>7.1f}%"
            f"{c['silent_wrong']:>8}"
            f"{inr(c['silent_wrong_paise']):>16}"
        )
    print("-" * 108)
    print(
        "  Match    = share of true pairs the system auto-confirmed (no human in the loop)\n"
        "  Prec/Rec = pair-level, counting everything matched or proposed for review\n"
        "  AutoPrec = precision restricted to auto-confirmed matches -- the ones nobody checks\n"
        "  Money    = share of rupees on the left side of each leg correctly accounted for\n"
        "  Silent   = auto-confirmed matches that are wrong. This is the number that matters."
    )


def print_leg_detail(r: dict) -> None:
    print()
    print("=" * 108)
    print(f"PER-LEG DETAIL  --  {r['label']}")
    print("=" * 108)
    for leg_name, leg in r["legs"].items():
        a, c, m = leg["all_proposed_or_confirmed"], leg["auto_confirmed_only"], leg["money"]
        print(f"\n  {leg_name}")
        print(
            f"    all matches      tp={a['tp']:<5} fp={a['fp']:<5} fn={a['fn']:<5} "
            f"P={a['precision']:.4f} R={a['recall']:.4f} F1={a['f1']:.4f}"
        )
        print(
            f"    auto-confirmed   tp={c['tp']:<5} fp={c['fp']:<5} fn={c['fn']:<5} "
            f"P={c['precision']:.4f} R={c['recall']:.4f}"
        )
        print(
            f"    money            {inr(m['reconciled_paise'])} of {inr(m['total_paise'])} "
            f"= {m['fraction_reconciled'] * 100:.2f}%   "
            f"misreconciled {inr(m['misreconciled_paise'])}"
        )
        h = leg["exception_honesty"]
        print(
            f"    exceptions       flagged={h['flagged']:<5} should={h['should_be_flagged']:<5} "
            f"precision={h['precision']:.4f} recall={h['recall']:.4f} missed={h['missed']}"
        )
        if h["reasons_missed"]:
            print(f"      missed by reason: {h['reasons_missed']}")
        pol = leg.get("policy") or {}
        if pol.get("tier1_above_materiality"):
            print(
                f"    materiality      {pol['tier1_above_materiality']} tier-1 identity matches "
                f"auto-confirmed above the gate, {inr(pol['tier1_above_materiality_paise'])} total; "
                f"{pol.get('gated_by_materiality', 0)} inferred matches held for a human"
            )
        print("    by break pattern:")
        for pat, pm in leg["by_break_pattern"].items():
            print(
                f"      {pat:<26} tp={pm['tp']:<5} fp={pm['fp']:<4} fn={pm['fn']:<5} "
                f"P={pm['precision']:.3f} R={pm['recall']:.3f}"
            )


def print_calibration(cal: dict) -> None:
    print()
    print("=" * 108)
    print("CALIBRATION  --  does 0.82 actually mean 82%?")
    print("=" * 108)
    print(f"\n  Brier score : {cal['brier']:.4f}   (0 = perfect, 0.25 = a coin flip)")
    print(f"  ECE         : {cal['ece']:.4f}   (weighted mean |confidence - accuracy|)")
    print(f"  n           : {cal['n']} scored candidates on the held-out eval set")
    print()
    print("  Reliability diagram")
    print(f"  {'confidence bin':<18}{'predicted':>11}{'actual':>9}{'n':>7}   {'':<32}")
    print("  " + "-" * 74)
    for p in cal["reliability"]:
        pred, act, n = p["mean_predicted"], p["fraction_positive"], p["count"]
        # Bar shows actual vs predicted: '#' is where reality landed, '|' the claim.
        width = 30
        a_col = int(round(act * width))
        p_col = int(round(pred * width))
        bar = ["."] * (width + 1)
        bar[a_col] = "#"
        if bar[p_col] == "#":
            bar[p_col] = "X"
        else:
            bar[p_col] = "|"
        print(
            f"  {p['bin_center'] - 0.05:.2f}-{p['bin_center'] + 0.05:.2f}      "
            f"{pred:>10.4f}{act:>9.4f}{n:>7}   {''.join(bar)}"
        )
    print("  " + "-" * 74)
    print("   '#' = actual accuracy, '|' = mean predicted, 'X' = they coincide")
    print()
    print("  Per-slice calibration (aggregate calibration hides subgroup failure):")
    print(f"  {'slice':<44}{'n':>6}{'conf':>8}{'acc':>8}{'ECE':>8}")
    print("  " + "-" * 74)
    worst = sorted(cal["by_slice"].items(), key=lambda kv: -kv[1]["ece"])
    for name, s in worst:
        flag = "  <-- worst" if name == worst[0][0] and s["ece"] > 0.10 else ""
        print(
            f"  {name:<44}{s['n']:>6}{s['mean_confidence']:>8.3f}"
            f"{s['accuracy']:>8.3f}{s['ece']:>8.3f}{flag}"
        )


def print_tier3(r: dict) -> None:
    """What tier 3 was asked, what it answered, what it cost, what it changed.

    The last one is the column that matters and the easiest to omit. An arbitration
    tier can look busy -- calls made, rationales written, tokens spent -- while
    moving no decision at all, because the policy layer still has to agree. So the
    delta against the tiers-1+2 row is printed alongside the activity, and if it is
    zero that is stated rather than left to be inferred from two tables.
    """
    t = r["tier3"]
    print()
    print("=" * 108)
    print(f"TIER 3  --  {r['label']}")
    print("=" * 108)
    print(f"\n  arbiter               {t['arbiter']}")
    print(f"  band                  {t.get('band') or '(see per-leg)'}")
    print(f"  groups eligible       {t['eligible']}  "
          f"({t['singleton_groups']} of them singletons -- nothing to compare against)")
    print(f"  calls made            {t['calls_made']} in {t['wall_ms']:.0f} ms")
    print(f"    proposed a match    {t['proposed_match']}")
    print(f"    insufficient        {t['insufficient_evidence']}")
    print(f"    delta clamped       {t['clamped']}")
    if t.get("chose_unselected_rival"):
        print(f"    disagreed w/ solver {t['chose_unselected_rival']}  "
              "(preferred a discarded hypothesis; recorded, not applied)")
    if t["errors"]:
        print(f"    errors              {t['errors']}  {t['error_kinds']}")
    print(f"  never offered         {t['skipped_above_materiality']} above the "
          f"{inr(MATERIALITY_PAISE)} materiality gate, "
          f"{t['skipped_outside_band']} outside the band"
          + (f", {t['capped_not_arbitrated']} over the cap"
             if t["capped_not_arbitrated"] else ""))
    if t["tokens_in"] or t["tokens_out"]:
        print(f"  tokens                {t['tokens_in']} in / {t['tokens_out']} out")


def print_tier3_effect(tier3_row: dict, base_row: dict) -> None:
    """The tier-3 row against the tiers-1+2 row it was built from."""
    a, b = combined(base_row), combined(tier3_row)
    print()
    print(f"  effect against '{base_row['label']}':")
    rows = [
        ("match rate", a["match_rate"] * 100, b["match_rate"] * 100, "%"),
        ("precision", a["precision"] * 100, b["precision"] * 100, "%"),
        ("recall", a["recall"] * 100, b["recall"] * 100, "%"),
        ("auto precision", a["auto_precision"] * 100, b["auto_precision"] * 100, "%"),
        ("money reconciled", a["money_fraction"] * 100, b["money_fraction"] * 100, "%"),
        ("silently wrong", a["silent_wrong"], b["silent_wrong"], ""),
    ]
    for name, before, after, unit in rows:
        delta = after - before
        arrow = "  (no change)" if abs(delta) < 1e-9 else f"  {delta:+.2f}{unit}"
        print(f"    {name:<20}{before:>9.2f}{unit} -> {after:>8.2f}{unit}{arrow}")
    if b["silent_wrong"] > a["silent_wrong"]:
        print("    !! tier 3 introduced silently-wrong matches. That is a regression "
              "in the one number this system exists to keep at zero.")


def main(argv: list[str] | None = None) -> int:
    console.init()
    ap = argparse.ArgumentParser(description="GlassLedger evaluation")
    ap.add_argument("--train-seed", type=int, default=7)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--train-payments", type=int, default=1100)
    ap.add_argument("--eval-payments", type=int, default=900)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rate-multiplier", type=float, default=1.0)
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--model", type=Path, default=ROOT / "data" / "model.pkl")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument(
        "--arbiter", default="off",
        choices=("off", "auto", "claude", "offline", "null"),
        help="when not 'off', run a fifth method with tier 3 enabled. The four "
             "published rows never use it, so the headline table stays reproducible "
             "without an API key",
    )
    ap.add_argument("--max-arbitrations", type=int, default=25)
    args = ap.parse_args(argv)

    train_dir = ROOT / "data" / "train"
    eval_dir = ROOT / "data" / "eval"

    print("GlassLedger evaluation")
    print("-" * 108)
    if not args.skip_generate:
        train_ds = generate(args.train_seed, args.train_payments, args.days, train_dir,
                            args.rate_multiplier)
        eval_ds = generate(args.eval_seed, args.eval_payments, args.days, eval_dir,
                           args.rate_multiplier)
    else:
        train_ds = load_dataset(train_dir)
        eval_ds = load_dataset(eval_dir)

    print(f"  train : seed={train_ds.seed} fingerprint={fingerprint(train_ds)} "
          f"({len(train_ds.gateway)} settlements, {len(train_ds.bank)} bank lines, "
          f"{len(train_ds.books)} book entries)")
    print(f"  eval  : seed={eval_ds.seed} fingerprint={fingerprint(eval_ds)} "
          f"({len(eval_ds.gateway)} settlements, {len(eval_ds.bank)} bank lines, "
          f"{len(eval_ds.books)} book entries)")
    print(f"  break-rate multiplier : {args.rate_multiplier}x")
    print(f"  target precision      : {TARGET_PRECISION}  "
          f"(threshold is derived from this, not tuned)")
    print(f"  materiality gate      : {inr(MATERIALITY_PAISE)}")

    # ---- train (never touches the eval set) ------------------------------
    if args.skip_train and args.model.exists():
        scorer = CalibratedScorer.load(args.model)
        print(f"\n  loaded model from {args.model} (threshold {scorer.threshold:.4f})")
    else:
        print("\n  harvesting labelled candidates from the TRAINING set...")
        X, y, slices = collect_training_rows(train_dir, train_ds)
        pos = sum(y)
        print(f"    {len(X)} candidates, {pos} positive ({pos / max(1, len(X)) * 100:.1f}%), "
              f"{len(FEATURE_NAMES)} features")
        scorer = CalibratedScorer().fit(X, y)
        scorer.save(args.model)
        tr = scorer.threshold_report
        print(f"    fitted + isotonic-calibrated on "
              f"{tr.get('train_rows')}/{tr.get('calibration_rows')}/{tr.get('threshold_rows')} "
              f"(fit/calibrate/threshold) rows")
        print(f"    derived threshold = {scorer.threshold:.4f}  "
              f"[{tr.get('method')}]  target_met={tr.get('target_met')}")
        if tr.get("precision_at_chosen") is not None:
            print(f"      at that threshold: precision={tr['precision_at_chosen']} "
                  f"recall={tr['recall_at_chosen']} on {tr['confirmed_at_chosen']} confirmations")
        if tr.get("target_met") is False:
            print(f"      !! precision target {tr.get('target_precision')} NOT reachable: "
                  f"{tr.get('note') or tr.get('reason')}")
        print(f"    saved to {args.model}")

    # ---- evaluate --------------------------------------------------------
    batch = load_batch(eval_dir)
    print(f"\n  ingested eval batch: {json.dumps(batch.summary()['statements'])}")

    results = [
        evaluate_baseline(batch, eval_ds),
        evaluate_engine(
            "GlassLedger tier 1 only",
            # max_tier=1, not an impossible threshold: raising the threshold above
            # 1.0 would also block tier 1's own score of exactly 1.0, and the
            # ablation would report 0% for a tier that resolves most of the volume.
            ReconciliationEngine(DummyScorer(), threshold=0.85, max_tier=1),
            batch, eval_ds,
        ),
        evaluate_engine(
            "GlassLedger tier 1+2 (heuristic)",
            ReconciliationEngine(DummyScorer(), threshold=0.85),
            batch, eval_ds,
        ),
        evaluate_engine(
            "GlassLedger tier 1+2 (calibrated)",
            ReconciliationEngine(scorer, threshold=scorer.threshold),
            batch, eval_ds,
        ),
    ]
    #: The tiers-1+2 calibrated run. Calibration is always measured on *this* run,
    #: never on the tier-3 row: an arbitrated score is the model's output plus a
    #: delta from somewhere else, so folding it into the Brier score and the ECE
    #: would attribute the arbiter's judgement to the calibration model.
    calibrated = results[-1]

    # The fifth row, and only on request. Tier 3 is not part of the published four
    # for a reason that is about reproducibility rather than modesty: rows 1-4 run
    # from the repository with no credentials and no network, so anyone can check
    # them. A row that calls a hosted model cannot make that promise, and mixing the
    # two in one table would quietly remove the promise from all of them.
    if args.arbiter != "off":
        from app.llm import default_arbiter

        arbiter = default_arbiter(args.arbiter)
        name = getattr(arbiter, "name", type(arbiter).__name__)
        print(f"\n  tier 3 enabled: {name} (max {args.max_arbitrations} arbitrations)")
        if not getattr(arbiter, "available", True):
            print("    ! arbiter unavailable -- every item will be routed to a human")
        results.append(
            evaluate_engine(
                f"GlassLedger tier 1+2+3 ({name})",
                ReconciliationEngine(
                    scorer, threshold=scorer.threshold,
                    arbiter=arbiter, max_arbitrations=args.max_arbitrations,
                ),
                batch, eval_ds,
            )
        )

    print_headline_table(results)
    print_leg_detail(results[-1])
    if results[-1].get("tier3"):
        print_tier3(results[-1])
        print_tier3_effect(results[-1], calibrated)

    # ---- calibration on the eval set -------------------------------------
    final_run = calibrated["_run"]
    probs: list[float] = []
    labels: list[int] = []
    slices: list[str] = []
    for leg_name, leg_run in final_run.legs.items():
        truth = truth_pairs(eval_ds, leg_name)
        pat_by_pair = {}
        for link in eval_ds.links_for(leg_name):
            for p in link.pair_keys():
                pat_by_pair[p] = link.pattern
        for d in (*leg_run.decisions.auto_confirmed, *leg_run.decisions.proposed,
                  *leg_run.decisions.rejected):
            c = d.candidate
            if c.tier == 1 or not c.features:
                continue  # tier 1 asserts 1.0 by construction; not a model prediction
            if c.tier == 3:
                # Arbitrated: the score is the model's output plus an arbiter's
                # delta, so it is not a claim the calibration model made and
                # scoring it here would credit or blame the wrong component.
                continue
            probs.append(c.score)
            labels.append(1 if c.pair_keys().issubset(truth) else 0)
            slices.append(
                f"{leg_name}/"
                + next((pat_by_pair[p] for p in c.pair_keys() if p in pat_by_pair),
                       "not_in_truth")
            )
    cal = {
        "n": len(probs),
        "brier": brier_score(probs, labels),
        "ece": expected_calibration_error(probs, labels),
        "reliability": reliability_diagram_points(probs, labels),
        "by_slice": calibration_by_slice(probs, labels, slices),
    }
    print_calibration(cal)

    # ---- performance -----------------------------------------------------
    print()
    print("=" * 108)
    print("THROUGHPUT")
    print("=" * 108)
    for r in results:
        print(
            f"  {r['label']:<34}{r['wall_ms']:>9.1f} ms   "
            f"{r['throughput_txn_per_s']:>9.1f} txn/s   "
            f"complete={r['completeness'].get('complete')}"
        )
    fr = calibrated["_run"]
    print(f"\n  per-leg (calibrated): {fr.timings_ms}")
    for leg, lr in fr.legs.items():
        b = lr.stats.get("blocking") or lr.stats.get("component_blocking") or {}
        if b:
            print(
                f"    {leg:<16} blocking kept {b.get('edges_kept')} of "
                f"{b.get('full_cross_product')} possible pairs "
                f"({b.get('reduction_vs_full', 0) * 100:.2f}% reduction)"
            )
        ss = lr.stats.get("subset_sum") or {}
        if ss:
            print(
                f"    {leg:<16} subset search {ss.get('subset_nodes_visited')} nodes, "
                f"{ss.get('pools_search_truncated')} pools truncated, "
                f"{ss.get('set_packing_conflicts')} packing conflicts"
            )

    # ---- persist ---------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "train": {"seed": train_ds.seed, "fingerprint": fingerprint(train_ds)},
        "eval": {"seed": eval_ds.seed, "fingerprint": fingerprint(eval_ds)},
        "rate_multiplier": args.rate_multiplier,
        "target_precision": TARGET_PRECISION,
        "derived_threshold": scorer.threshold,
        "threshold_report": scorer.threshold_report,
        "materiality_paise": MATERIALITY_PAISE,
        "tier3_band": list(TIER3_BAND),
        "tier3_arbiter": args.arbiter,
        "tier3": results[-1].get("tier3") if args.arbiter != "off" else None,
        "calibration": cal,
        "methods": [
            {k: v for k, v in r.items() if k != "_run"} for r in results
        ],
        "headline": {r["label"]: combined(r) for r in results},
    }
    out_path = args.out / "latest.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
