"""GlassLedger synthetic data generator + answer key.

    python data_generator/generate_synthetic.py --seed 42 --payments 900 --out data/eval

Built before the matcher, on purpose. Without a ground-truth generator there is no
way to distinguish "the engine resolved 96% of volume" from "the engine asserted
96% of volume", and those are opposite claims. Everything the engine later reports
is measured against the answer key this file writes.

Reproducibility contract
------------------------
Same ``--seed`` and same flags produce byte-identical outputs. Every random draw
goes through one seeded ``random.Random``; nothing calls the module-level
``random``, nothing reads the clock into the data, and dict iteration is sorted
wherever it feeds a draw. The run prints a dataset fingerprint (sha256 over the
normalised dataset) so a published metric can be tied to the exact bytes it came
from.

The two-dataset discipline that matters
---------------------------------------
The confidence model trains on ``--seed 7`` and every published number is measured
on ``--seed 42``. Different seeds mean different amounts, different UTRs,
different break placements -- so a model that memorised the training batch scores
no better than chance on the eval batch. Training and evaluating on one dataset
would produce a beautiful reliability diagram that means nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core import console  # noqa: E402
from app.core.schema import (  # noqa: E402
    Dataset,
    GroundTruthLink,
    NormalizedTxn,
    UnmatchableTxn,
    canonical_json,
    payload_hash,
    sha256_hex,
)

from data_generator import break_patterns as bp  # noqa: E402
from data_generator.emitters import (  # noqa: E402
    build_books,
    write_bank_hdfc,
    write_bank_icici,
    write_bank_mt940,
    write_books,
    write_gateway,
)
from data_generator.world import BankLine, World, business_days_after  # noqa: E402

# Real e-commerce price points. This matters more than it looks: drawing amounts
# from a continuous distribution makes every settlement amount unique, which
# hands the matcher a free identifier and inflates every accuracy number. Real
# catalogues cluster on round prices, so identical settlement amounts collide
# inside the same window constantly -- and that ambiguity is precisely what the
# confidence model has to learn to be uncertain about.
PRICE_POINTS = [
    49900, 79900, 99900, 129900, 149900, 199900, 249900, 299900,
    349900, 449900, 499900, 599900, 799900, 999900,
]
#: A D2C merchant's occasional wholesale or corporate invoice. Rare (4%) and
#: capped, because an earlier version drew 10% of payments from a
#: Rs 5,000-Rs 500,000 range and that tail alone set the *mean* settlement at
#: Rs 85,000 -- above the materiality gate. The dataset stopped describing a D2C
#: merchant and started describing a B2B one, which quietly changed what every
#: policy number in the eval meant.
B2B_RATE = 0.04
B2B_RANGE = (8_000_00, 120_000_00)

DEFAULT_RATES = {
    # Break-pattern injection rates. Deliberately in the range the problem
    # statement quotes for a real merchant (15-30% of volume touched by
    # something), not tuned to make the engine look good. Raising these is the
    # honest stress test; see eval/run_eval.py --rate-multiplier.
    "batch_settlement": 0.55,      # share of settlement slots that bundle >1 payment
    "refund_netting": 0.10,
    "fee_tax_drift": 0.22,
    "fx_mismatch": 0.045,          # share of payments invoiced in USD
    "split_settlement": 0.07,
    "duplicate_bank_entry": 0.045,
    "timing_drift": 0.09,
    "narration_noise": 0.30,       # share of credits whose UTR column is empty
    "orphan_bank_credit": 0.05,    # count relative to settlement count
    "missing_settlement": 0.035,
    "tds_mismatch": 0.12,
}


def make_holidays(year: int) -> frozenset[date]:
    """A plausible Indian bank-holiday set.

    Fixed rather than looked up: the generator must not depend on the network or
    on a calendar library's opinion, or a dataset regenerated next year would
    silently differ from the one a published metric was computed on.
    """
    return frozenset(
        {
            date(year, 1, 26), date(year, 3, 4), date(year, 3, 25),
            date(year, 4, 14), date(year, 5, 1), date(year, 8, 15),
            date(year, 10, 2), date(year, 11, 8), date(year, 12, 25),
        }
    )


# ---------------------------------------------------------------------------
# Base world
# ---------------------------------------------------------------------------

def build_payments(world: World, rng: random.Random, n: int, days: int) -> None:
    start = world.start_date
    for i in range(n):
        # Volume is weekday-heavy, which is what makes settlement batches uneven
        # and gives the date features something real to key on.
        for _ in range(12):
            day = start + timedelta(days=rng.randrange(days))
            if day.weekday() < 5 or rng.random() < 0.45:
                break
        if rng.random() >= B2B_RATE:
            amount = rng.choice(PRICE_POINTS)
            if rng.random() < 0.35:  # multi-item basket
                amount += rng.choice(PRICE_POINTS) // rng.choice([1, 2])
        else:
            amount = rng.randrange(*B2B_RANGE)
        instrument = rng.choices(
            ["upi", "card", "netbanking", "intl_card"], weights=[52, 33, 13, 2]
        )[0]
        pid = "pay_" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=14))
        from data_generator.world import Payment

        world.payments[pid] = Payment(
            payment_id=pid,
            gross_paise=int(amount),
            captured_on=day,
            instrument=instrument,
        )


def create_bank_credits(world: World, rng: random.Random, holidays: frozenset[date]) -> None:
    """One credit per settlement, at the settlement's net, T+0..T+1 after release.

    Runs between the ``pre_bank`` and ``post_bank`` phases: every mutation that
    changes what the gateway will pay out has landed, and every mutation that
    perturbs the bank feed has not started yet. Getting that ordering wrong is
    how you produce a "clean" settlement whose own bank line disagrees with it.
    """
    # A merchant typically has more than one account. Route each settlement to
    # one of three, so the ingestion layer has to merge three dialects into one
    # canonical stream and the matcher never sees which file a line came from.
    formats = ["hdfc", "icici", "mt940"]
    weights = [0.55, 0.30, 0.15]
    for s in world.settlements_in_order():
        net = s.net_paise(world)
        if net <= 0:
            # A batch whose refunds exceeded its captures produces a debit, not a
            # credit. Real, but out of scope for a settlement-credit matcher, so
            # it is excluded here and shows up as an exception rather than being
            # quietly forced into a shape the engine expects.
            s.status = "reversed"
            continue
        fmt = rng.choices(formats, weights=weights)[0]
        value_date = business_days_after(s.created_on, rng.choice([0, 0, 1]), holidays)
        world.bank_lines.append(
            BankLine(
                bank_ref="BR" + "".join(rng.choices("0123456789", k=10)),
                value_date=value_date,
                posted_date=value_date,
                amount_paise=net,
                narration=f"RAZORPAY SETTLEMENT {s.utr}",
                utr_field=s.utr,
                role="settlement_credit",
                settlement_id=s.settlement_id,
                bank_format=fmt,
            )
        )


def unsettle_tail(world: World, window_end: date) -> int:
    """Drop settlements that would be released after the statement window closes.

    Their payments stay in the books as expected receivables with no settlement
    to match against -- an "awaiting settlement" exception, which is the single
    most common legitimate open item on a real reconciliation and needs to be in
    the dataset or the exception-honesty metric is measured against an
    unrealistically clean population.
    """
    dropped = 0
    for s in list(world.settlements.values()):
        if s.created_on > window_end:
            for pid in s.payment_ids:
                world.payments[pid].settlement_id = None
            for rid in s.refund_ids:
                world.refunds[rid].settlement_id = None
            del world.settlements[s.settlement_id]
            dropped += 1
    return dropped


# ---------------------------------------------------------------------------
# Normalisation of the world into the answer key
# ---------------------------------------------------------------------------

def to_normalized(world: World) -> tuple[list, list, list]:
    """Build the canonical view the answer key is expressed in.

    Note this is *not* the path the engine takes. The engine reads the emitted
    files through ``app.ingestion``. This function exists only so ground truth can
    be stated in the same vocabulary the engine will eventually produce, and the
    ingestion round-trip test asserts the two agree.
    """
    gateway, bank, books = [], [], []

    for s in world.settlements_in_order():
        raw = {
            "id": s.settlement_id, "amount": s.net_paise(world), "fees": s.fees_paise,
            "tax": s.tax_paise, "utr": s.utr, "status": s.status,
            "created_at": s.created_on.isoformat(),
        }
        gateway.append(
            NormalizedTxn(
                source="gateway", external_id=s.settlement_id,
                amount_paise=s.net_paise(world), currency="INR", utr=s.utr,
                narration=f"settlement {s.settlement_id} status={s.status}",
                value_date=s.created_on, fees_paise=s.fees_paise, tax_paise=s.tax_paise,
                raw_payload_hash=payload_hash("gateway", s.settlement_id, raw),
                provenance={"n_payments": len(s.payment_ids), "n_refunds": len(s.refund_ids)},
            )
        )

    for b in sorted(world.bank_lines, key=lambda x: (x.value_date, x.bank_ref)):
        raw = {
            "ref": b.bank_ref, "amount": b.amount_paise, "narration": b.narration,
            "value_date": b.value_date.isoformat(), "posted": b.posted_date.isoformat(),
        }
        bank.append(
            NormalizedTxn(
                source="bank", external_id=b.bank_ref, amount_paise=b.amount_paise,
                currency="INR", utr=b.utr_field, narration=b.narration,
                value_date=b.value_date,
                raw_payload_hash=payload_hash("bank", b.bank_ref, raw),
                provenance={"format": b.bank_format, "posted_date": b.posted_date.isoformat()},
            )
        )

    for e in sorted(world.book_entries, key=lambda x: (x.booked_on, x.entry_id)):
        raw = {
            "entry": e.entry_id, "amount": e.amount_paise, "memo": e.memo,
            "booked_on": e.booked_on.isoformat(), "kind": e.kind,
        }
        books.append(
            NormalizedTxn(
                source="books", external_id=e.entry_id, amount_paise=e.amount_paise,
                currency="INR", utr=None, narration=e.memo, value_date=e.booked_on,
                raw_payload_hash=payload_hash("books", e.entry_id, raw),
                provenance={"kind": e.kind, "order_ref": getattr(e, "order_ref", None) or ""},
            )
        )

    return gateway, bank, books


def derive_ground_truth(world: World) -> tuple[list[GroundTruthLink], list[UnmatchableTxn]]:
    """The answer key, computed from final world state.

    Derived rather than logged. A pattern that moves a bank line after another
    pattern claimed it cannot desynchronise the key from the data, because the
    key is a pure function of the world at the end.
    """
    links: list[GroundTruthLink] = []
    unmatchable: list[UnmatchableTxn] = []

    for s in world.settlements_in_order():
        sid = f"gateway:{s.settlement_id}"
        credits = sorted(world.credits_for(s.settlement_id), key=lambda b: b.bank_ref)
        pattern = _headline_pattern(s.patterns)

        if credits:
            links.append(
                GroundTruthLink(
                    leg="gateway_bank",
                    left_ids=(sid,),
                    right_ids=tuple(f"bank:{c.bank_ref}" for c in credits),
                    pattern=pattern,
                )
            )
        else:
            unmatchable.append(
                UnmatchableTxn(
                    txn_id=sid,
                    reason="missing_settlement" if s.status == "processed" else f"settlement_{s.status}",
                    leg="gateway_bank",
                )
            )

        entries = sorted(world.entries_for(s.settlement_id), key=lambda e: e.entry_id)
        if entries:
            links.append(
                GroundTruthLink(
                    leg="gateway_books",
                    left_ids=tuple(f"books:{e.entry_id}" for e in entries),
                    right_ids=(sid,),
                    pattern=pattern,
                )
            )

    for b in world.bank_lines:
        if b.role in ("orphan", "stale_credit", "reversal_debit"):
            unmatchable.append(
                UnmatchableTxn(
                    txn_id=f"bank:{b.bank_ref}",
                    reason=b.unmatchable_reason or b.role,
                    leg="gateway_bank",
                )
            )

    for e in world.book_entries:
        if e.settlement_id is None:
            unmatchable.append(
                UnmatchableTxn(
                    txn_id=f"books:{e.entry_id}",
                    reason=e.unmatchable_reason or "awaiting_settlement",
                    leg="gateway_books",
                )
            )

    return links, unmatchable


#: When several patterns hit one settlement, the metrics need a single label.
#: Ordered by how much each one actually changes the matching problem, hardest
#: first, so a settlement that is both batched and missing is reported as
#: missing -- the harder fact about it.
PATTERN_PRIORITY = [
    "missing_settlement", "duplicate_bank_entry", "split_settlement",
    "fx_mismatch", "refund_netting", "narration_noise", "timing_drift",
    "tds_mismatch", "fee_tax_drift", "batch_settlement",
]


def _headline_pattern(patterns: set[str]) -> str:
    for p in PATTERN_PRIORITY:
        if p in patterns:
            return p
    return "clean"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def generate(
    seed: int, n_payments: int, days: int, out: Path,
    rate_multiplier: float = 1.0, tds_bps: int = 100,
) -> Dataset:
    rng = random.Random(seed)
    start = date(2026, 3, 2)
    holidays = make_holidays(2026) | make_holidays(2025)
    world = World(seed=seed, start_date=start)

    build_payments(world, rng, n_payments, days)

    rates = {k: min(1.0, v * rate_multiplier) for k, v in DEFAULT_RATES.items()}
    settings = {"tds_bps": tds_bps}

    def run_phase(name: str) -> None:
        for mod in bp.PHASES[name]:
            ctx = bp.Ctx(
                rng=rng, holidays=holidays,
                rate=rates.get(mod.PATTERN, 0.0), settings=settings,
            )
            mod.apply(world, ctx)

    run_phase("structural")
    dropped = unsettle_tail(world, start + timedelta(days=days))
    run_phase("pre_bank")
    create_bank_credits(world, rng, holidays)
    run_phase("post_bank")
    build_books(world, rng)
    run_phase("books")

    world.note("tail_unsettled", settlements_dropped=dropped)

    # Bank references are the only stable identity a statement line has, so a
    # collision would silently merge two lines during ingestion and corrupt the
    # answer key. Cheap to check, impossible to debug later.
    refs = [b.bank_ref for b in world.bank_lines]
    if len(refs) != len(set(refs)):
        dupes = sorted({r for r in refs if refs.count(r) > 1})
        raise AssertionError(f"duplicate bank references generated: {dupes[:5]}")

    out.mkdir(parents=True, exist_ok=True)
    write_gateway(world, out)
    write_bank_hdfc(world, out)
    write_bank_icici(world, out)
    write_bank_mt940(world, out)
    write_books(world, out)

    gateway, bank, books = to_normalized(world)
    links, unmatchable = derive_ground_truth(world)
    ds = Dataset(
        seed=seed,
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        gateway=gateway, bank=bank, books=books,
        links=links, unmatchable=unmatchable,
        injection_log=world.injection_log,
    )
    (out / "ground_truth.json").write_text(
        ds.model_dump_json(indent=2), encoding="utf-8"
    )
    return ds


def fingerprint(ds: Dataset) -> str:
    """Content hash over the data only -- never the wall clock.

    ``generated_at`` is excluded so regenerating the same seed twice yields the
    same fingerprint. A fingerprint that changed every run would be decoration;
    this one is a claim you can check.
    """
    payload = {
        "gateway": [t.model_dump(mode="json") for t in ds.gateway],
        "bank": [t.model_dump(mode="json") for t in ds.bank],
        "books": [t.model_dump(mode="json") for t in ds.books],
        "links": [l.model_dump(mode="json") for l in ds.links],
        "unmatchable": [u.model_dump(mode="json") for u in ds.unmatchable],
    }
    return sha256_hex(canonical_json(payload))[:16]


def main(argv: list[str] | None = None) -> int:
    console.init()
    ap = argparse.ArgumentParser(description="GlassLedger synthetic reconciliation dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--payments", type=int, default=900)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval")
    ap.add_argument(
        "--rate-multiplier", type=float, default=1.0,
        help="scale every break-pattern rate; >1 is the stress test",
    )
    ap.add_argument("--tds-bps", type=int, default=100)
    args = ap.parse_args(argv)

    ds = generate(
        args.seed, args.payments, args.days, args.out,
        args.rate_multiplier, args.tds_bps,
    )
    fp = fingerprint(ds)
    (args.out / "FINGERPRINT").write_text(fp + "\n", encoding="utf-8")

    n_pairs_a = sum(len(l.pair_keys()) for l in ds.links_for("gateway_bank"))
    n_pairs_b = sum(len(l.pair_keys()) for l in ds.links_for("gateway_books"))
    total = sum(t.amount_paise for t in ds.gateway)

    print(f"seed={args.seed}  fingerprint={fp}")
    print(f"  gateway settlements : {len(ds.gateway):5d}   ({console.money(total)} net)")
    print(f"  bank lines          : {len(ds.bank):5d}   across 3 formats")
    print(f"  book entries        : {len(ds.books):5d}")
    print(f"  truth links         : {len(ds.links):5d}  "
          f"({n_pairs_a} bank pairs, {n_pairs_b} books pairs)")
    print(f"  must-be-exceptions  : {len(ds.unmatchable):5d}")
    print()
    card: dict[str, int] = {}
    for l in ds.links:
        card[f"{l.leg}/{l.cardinality}"] = card.get(f"{l.leg}/{l.cardinality}", 0) + 1
    for k in sorted(card):
        print(f"    {k:28s} {card[k]:5d}")
    print()
    for rec in ds.injection_log:
        extras = " ".join(f"{k}={v}" for k, v in rec.items() if k != "pattern")
        print(f"    {rec['pattern']:24s} {extras}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
