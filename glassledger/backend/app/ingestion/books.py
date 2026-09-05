"""Books ingestion: the merchant's expected-receivable ledger.

Simplest of the three sources and the one most likely to be wrong in practice,
because it is maintained by hand. Two fields carry all the signal:

``order_ref``
    The gateway order id, present when the checkout wrote it back into the ERP.
    Blank on legacy invoices, manual entries, credit notes and accrual lines.
    This is the deterministic bridge to a settlement -- when it exists.

``expected_amount_inr``
    What the merchant thinks will land: gross minus their own blended fee
    assumption. Systematically *not* equal to what the gateway actually pays,
    which is the entire point of this leg.

Amounts go through ``from_rupee_string``, not ``float``. ``"10,015.55"`` parsed as
a float and multiplied by 100 lands on ``1001554.9999999999``, and ``int()`` of
that is 1001554 -- one paise short, on an entry that was perfectly correct. Do
that a few hundred times and the books leg develops a systematic downward bias
that looks like an unexplained fee.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from app.core.money import from_rupee_string
from app.core.schema import NormalizedTxn, payload_hash


@dataclass
class BooksIngest:
    entries: list[NormalizedTxn] = field(default_factory=list)
    #: order_ref -> books external_id, for the deterministic tier
    order_index: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


REQUIRED = ("entry_id", "booked_on", "expected_amount_inr")


def parse_books(path: Path) -> BooksIngest:
    out = BooksIngest()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: books export missing columns {missing}")

        for n, row in enumerate(reader, start=2):
            entry_id = (row["entry_id"] or "").strip()
            if not entry_id:
                out.warnings.append(f"{path.name} row {n}: blank entry_id, skipped")
                continue
            amount = from_rupee_string(row["expected_amount_inr"])
            booked_on = _parse_iso(row["booked_on"], path.name, n)
            order_ref = (row.get("order_ref") or "").strip() or None
            memo = (row.get("memo") or "").strip()
            kind = (row.get("kind") or "receivable").strip()

            raw = {
                "entry": entry_id, "amount": amount, "memo": memo,
                "booked_on": booked_on.isoformat(), "kind": kind,
            }
            out.entries.append(
                NormalizedTxn(
                    source="books",
                    external_id=entry_id,
                    amount_paise=amount,
                    currency="INR",
                    utr=None,
                    narration=memo,
                    value_date=booked_on,
                    ref_candidates=(order_ref,) if order_ref else (),
                    raw_payload_hash=payload_hash("books", entry_id, raw),
                    provenance={
                        "kind": kind,
                        "order_ref": order_ref or "",
                        "source_file": path.name,
                        "row": n,
                    },
                )
            )
            if order_ref:
                out.order_index.setdefault(order_ref, []).append(entry_id)

    out.entries.sort(key=lambda t: (t.value_date, t.external_id))
    return out


def _parse_iso(text: str, filename: str, row: int):
    from datetime import datetime

    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{filename} row {row}: bad date {text!r} (want YYYY-MM-DD)") from exc
