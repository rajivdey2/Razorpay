"""Serialise the world into the raw formats a real merchant would actually hand you.

Five outputs:

    gateway_settlements.json   Razorpay ``settlement`` entities, paise, epoch ts
    gateway_payments.json      the settlement recon view: payment -> settlement_id
    bank_hdfc.csv              HDFC-style export: DD/MM/YY, separate Dr/Cr columns
    bank_icici.csv             ICICI-style export: DD-MM-YYYY, S.No, different headers
    bank_generic.mt940         SWIFT MT940: :61:/:86: tags, comma decimal separator
    books_receivables.csv      the merchant's expected receivables

Three bank dialects is not padding. Each one breaks a different assumption a
lazily-written parser makes:

* HDFC puts five preamble lines above the header and pads every field with
  spaces, so a parser that assumes row 0 is the header reads garbage.
* ICICI writes ``DD-MM-YYYY`` where HDFC writes ``DD/MM/YY``. Both parse fine as
  dates under a permissive parser, and one of them will be wrong by a decade or
  by a month/day transposition. Ambiguous dates are silently wrong, which is the
  worst failure class in a money system, so the parsers here are strict and
  per-format rather than clever and shared.
* MT940 is not CSV at all, uses ``,`` as the decimal separator, encodes sign as a
  ``C``/``D`` mark, and splits one transaction across two tag lines.

The balance column is computed as a real running balance. It is not used for
matching, but it lets the ingestion layer assert internal consistency
(``closing - opening == sum(movements)``), which is a genuinely cheap way to
catch a misparsed row before it reaches the matcher.

Line identity, and why the UTR is not a column
----------------------------------------------
The reference column carries the *bank's own* reference (``BR…``), which is
unique per line, and the UTR is left embedded in the narration text. That is how
these exports actually look, and it has a consequence worth being explicit about:
there is no clean UTR field to join on. Extracting the UTR is an ingestion
problem solved by pattern-matching over free text, and it succeeds fully on
roughly 70% of lines, partially on the ones break pattern #10 has mangled, and
not at all on the rest. A generator that emitted a tidy ``utr`` column would hand
the matcher a primary key and quietly turn the whole exercise into a SQL join.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from app.core.money import apply_bps, indian_grouping

from .world import BankLine, World

# The merchant's blended fee assumption, applied uniformly in their books. The
# gateway's real per-instrument rates live in world.ACTUAL_MDR_BPS; the gap is
# break pattern #4.
BOOKS_ASSUMED_MDR_BPS = 200
BOOKS_ASSUMED_GST_BPS = 1800


def _rupees_2dp(paise: int) -> str:
    """Lakh-grouped, like an actual Indian bank statement.

    Not cosmetic: the parsers strip commas, so emitting ``1,23,456.78`` rather
    than ``123,456.78`` is what proves they strip *Indian* grouping and not just
    the Western kind. A parser tested only against thousands separators passes
    right up until it meets a real statement.
    """
    sign = "-" if paise < 0 else ""
    whole, frac = divmod(abs(paise), 100)
    return f"{sign}{indian_grouping(whole)}.{frac:02d}"


def _mt940_amount(paise: int) -> str:
    whole, frac = divmod(abs(paise), 100)
    return f"{whole},{frac:02d}"


def _sorted_lines(world: World, fmt: str) -> list[BankLine]:
    return sorted(
        (b for b in world.bank_lines if b.bank_format == fmt),
        key=lambda b: (b.posted_date, b.value_date, b.bank_ref),
    )


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

def write_gateway(world: World, out: Path) -> None:
    """Razorpay ``settlement`` entities.

    Field semantics mirror the real entity: ``amount`` is the **net** credited to
    the bank account and ``fees``/``tax`` are what was withheld, all in paise,
    with ``created_at`` as a Unix timestamp. Reproducing the real shape means the
    ingestion parser is the one that would run against the live API, not a
    bespoke reader for a made-up format.
    """
    settlements = []
    for s in world.settlements_in_order():
        settlements.append(
            {
                "id": s.settlement_id,
                "entity": "settlement",
                "amount": s.net_paise(world),
                "status": s.status,
                "fees": s.fees_paise,
                "tax": s.tax_paise,
                "utr": s.utr,
                "created_at": int(
                    __import__("calendar").timegm(s.created_on.timetuple())
                ),
            }
        )
    payload = {
        "entity": "collection",
        "count": len(settlements),
        "items": settlements,
    }
    (out / "gateway_settlements.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # The settlement recon view: which payments and refunds rolled into which
    # payout. Razorpay exposes this; a merchant reconciling without it is working
    # blind, so withholding it here would make the benchmark unrealistically hard
    # in a way that flatters the matcher's Tier 2.
    recon = []
    for p in sorted(world.payments.values(), key=lambda x: x.payment_id):
        recon.append(
            {
                "entity": "payment",
                "id": p.payment_id,
                "order_id": _order_ref(p.payment_id),
                "amount": p.gross_paise,
                "currency": "INR",
                "invoice_currency": p.invoice_currency,
                "invoice_amount": p.invoice_amount_minor or None,
                "method": p.instrument,
                "captured_at": int(__import__("calendar").timegm(p.captured_on.timetuple())),
                "settlement_id": p.settlement_id,
            }
        )
    for r in sorted(world.refunds.values(), key=lambda x: x.refund_id):
        recon.append(
            {
                "entity": "refund",
                "id": r.refund_id,
                "payment_id": r.payment_id,
                "amount": -r.amount_paise,   # signed at the boundary, once
                "currency": "INR",
                "created_at": int(__import__("calendar").timegm(r.created_on.timetuple())),
                "settlement_id": r.settlement_id,
            }
        )
    (out / "gateway_payments.json").write_text(
        json.dumps({"entity": "collection", "count": len(recon), "items": recon}, indent=2),
        encoding="utf-8",
    )


def _order_ref(payment_id: str) -> str:
    return "order_" + payment_id[4:]


# ---------------------------------------------------------------------------
# Bank dialects
# ---------------------------------------------------------------------------

def write_bank_hdfc(world: World, out: Path, opening_paise: int = 1_250_000_00) -> None:
    lines = _sorted_lines(world, "hdfc")
    balance = opening_paise
    with (out / "bank_hdfc.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        # Real HDFC exports carry an account preamble. Included on purpose: the
        # parser has to find its header row rather than assume row 0.
        w.writerow(["HDFC BANK LIMITED - STATEMENT OF ACCOUNT"])
        w.writerow(["Account No :", "50200012345678"])
        w.writerow(["Account Branch :", "BERHAMPUR ODISHA"])
        w.writerow(["Statement From :", lines[0].posted_date.strftime("%d/%m/%y") if lines else ""])
        w.writerow([])
        w.writerow(
            ["Date", "Narration", "Chq./Ref.No.", "Value Dt",
             "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"]
        )
        for b in lines:
            balance += b.amount_paise
            wd = _rupees_2dp(-b.amount_paise) if b.amount_paise < 0 else ""
            dp = _rupees_2dp(b.amount_paise) if b.amount_paise > 0 else ""
            w.writerow(
                [
                    b.posted_date.strftime("%d/%m/%y"),
                    # HDFC space-pads narrations to a fixed width.
                    f"{b.narration:<60}",
                    b.bank_ref,
                    b.value_date.strftime("%d/%m/%y"),
                    wd,
                    dp,
                    _rupees_2dp(balance),
                ]
            )


def write_bank_icici(world: World, out: Path, opening_paise: int = 880_000_00) -> None:
    lines = _sorted_lines(world, "icici")
    balance = opening_paise
    with (out / "bank_icici.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["S No.", "Value Date", "Transaction Date", "Cheque Number",
             "Transaction Remarks", "Withdrawal Amount (INR )",
             "Deposit Amount (INR )", "Balance (INR )"]
        )
        for i, b in enumerate(lines, start=1):
            balance += b.amount_paise
            wd = _rupees_2dp(-b.amount_paise) if b.amount_paise < 0 else "0.00"
            dp = _rupees_2dp(b.amount_paise) if b.amount_paise > 0 else "0.00"
            w.writerow(
                [
                    i,
                    b.value_date.strftime("%d-%m-%Y"),
                    b.posted_date.strftime("%d-%m-%Y"),
                    b.bank_ref,
                    b.narration,
                    wd,
                    dp,
                    _rupees_2dp(balance),
                ]
            )


def write_bank_mt940(world: World, out: Path, opening_paise: int = 410_000_00) -> None:
    lines = _sorted_lines(world, "mt940")
    balance = opening_paise
    first = lines[0].value_date if lines else date(2026, 1, 1)
    last = lines[-1].value_date if lines else first
    body: list[str] = [
        ":20:GLSTMT0001",
        ":25:IN91GENB0000012345678901",
        ":28C:00042/001",
        f":60F:C{first.strftime('%y%m%d')}INR{_mt940_amount(opening_paise)}",
    ]
    for b in lines:
        balance += b.amount_paise
        mark = "C" if b.amount_paise > 0 else "D"
        body.append(
            f":61:{b.value_date.strftime('%y%m%d')}{b.posted_date.strftime('%m%d')}"
            f"{mark}{_mt940_amount(b.amount_paise)}NTRF{b.bank_ref[:16]}"
            f"//{b.bank_ref}"
        )
        # MT940 :86: is capped at 6 x 65 chars; wrap rather than truncate so no
        # information is lost that the generator's answer key still assumes.
        text = b.narration
        for chunk in [text[i : i + 65] for i in range(0, len(text), 65)] or [""]:
            body.append(f":86:{chunk}")
    body.append(f":62F:C{last.strftime('%y%m%d')}INR{_mt940_amount(balance)}")
    body.append("-")
    (out / "bank_generic.mt940").write_text("\n".join(body) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------

def build_books(world: World, rng, keyed_fraction: float = 0.60) -> None:
    """Create the merchant's expected-receivable ledger.

    One entry per settled payment, at the amount the merchant *expects* to
    receive: gross minus their own blended fee assumption. Two things are
    deliberately partial:

    * ``order_ref`` is present on only ~60% of entries. A merchant whose checkout
      writes the gateway order id back into the ERP gets a deterministic join for
      those; legacy, manual and offline invoices have nothing. That mix is the
      realistic one, and it means Tier 1 has real work on this leg while Tier 2
      still has a real population to earn its keep on.
    * Refunds, withholding accruals and FX adjustments have no payment
      counterpart at all, so they can only ever be resolved as part of a subset.

    Invoices for payments captured *after* the statement window closes are booked
    too, with no settlement attached. They are the single most common legitimate
    open item on a real reconciliation -- "invoiced, money not here yet" -- and
    leaving them out would make the exception-honesty metric a measurement against
    a population that has no honest exceptions in it.
    """
    for p in sorted(world.payments.values(), key=lambda x: x.payment_id):
        if p.invoice_currency == "USD":
            # Booked at the invoice-day rate -- the settlement used a later one.
            gross_booked = (p.invoice_amount_minor * p.booking_fx_micros) // 1_000_000
            memo = (
                f"INV {_invoice_no(p.payment_id)} USD {p.invoice_amount_minor / 100:.2f} "
                f"@ {p.booking_fx_micros / 1_000_000:.4f}"
            )
        else:
            gross_booked = p.gross_paise
            memo = f"INV {_invoice_no(p.payment_id)} domestic sale"

        fee = apply_bps(gross_booked, BOOKS_ASSUMED_MDR_BPS)
        gst = apply_bps(fee, BOOKS_ASSUMED_GST_BPS)
        expected = gross_booked - fee - gst

        world.book_entries.append(
            _BookEntry(
                entry_id=_invoice_no(p.payment_id),
                amount_paise=expected,
                booked_on=p.captured_on,
                memo=memo,
                payment_id=p.payment_id,
                settlement_id=p.settlement_id,
                unmatchable_reason=None if p.settlement_id else "awaiting_settlement",
                kind="receivable",
                order_ref=_order_ref(p.payment_id) if rng.random() < keyed_fraction else None,
            )
        )

    for r in sorted(world.refunds.values(), key=lambda x: x.refund_id):
        world.book_entries.append(
            _BookEntry(
                entry_id=f"CRN-{r.refund_id[-8:].upper()}",
                amount_paise=-r.amount_paise,
                booked_on=r.created_on,
                memo=f"Credit note against {_invoice_no(r.payment_id)}",
                payment_id=r.payment_id,
                settlement_id=r.settlement_id,
                unmatchable_reason=None if r.settlement_id else "awaiting_settlement",
                kind="refund",
                order_ref=None,   # credit notes never carry the gateway order id
            )
        )


def _invoice_no(payment_id: str) -> str:
    return "INV-" + payment_id[4:12].upper()


def write_books(world: World, out: Path) -> None:
    rows = sorted(world.book_entries, key=lambda b: (b.booked_on, b.entry_id))
    with (out / "books_receivables.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["entry_id", "booked_on", "kind", "order_ref", "expected_amount_inr", "memo"]
        )
        for b in rows:
            w.writerow(
                [
                    b.entry_id,
                    b.booked_on.isoformat(),
                    b.kind,
                    getattr(b, "order_ref", None) or "",
                    _rupees_2dp(b.amount_paise),
                    b.memo,
                ]
            )


# BookEntry gains one generator-only field (``order_ref``) that the world model
# does not need to know about. Subclassing here keeps the emitter's concern out of
# the shared dataclass instead of widening it for one writer's benefit.
from dataclasses import dataclass  # noqa: E402

from .world import BookEntry  # noqa: E402


@dataclass
class _BookEntry(BookEntry):
    order_ref: str | None = None
