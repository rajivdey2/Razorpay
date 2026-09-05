"""Bank statement parsers: HDFC-style CSV, ICICI-style CSV, SWIFT MT940.

One parser per dialect, each strict about its own format, sharing nothing but the
canonical output type. That is a deliberate rejection of the obvious alternative
-- one clever parser with permissive date handling and fuzzy column detection.

The reason is that in a money system, a *loud* parse failure is cheap and a
*silent* misparse is unbounded. ``05/03/26`` is the 5th of March in an HDFC export
and the 3rd of May in a US-formatted one; a permissive ``dateutil``-style parser
accepts both and picks one, and if it picks wrong every date feature downstream is
off by two months. Nothing crashes, the match rate just drops and the exception
list fills with things that look like timing drift. Per-format ``strptime`` with an
exact pattern turns that class of bug into an exception on row 1.

Each parser also verifies the statement's own running balance. If
``closing[i] - closing[i-1] != amount[i]``, a row was misread -- wrong column,
lost minus sign, thousands separator eaten -- and the parser says so with the row
number instead of handing the matcher a plausible-looking wrong number.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

from app.core.money import from_rupee_string
from app.core.schema import NormalizedTxn, payload_hash

from .refs import extract_references


class ParseError(Exception):
    """A statement row could not be read with certainty.

    Deliberately fatal rather than skip-and-continue. A reconciliation run over
    a statement with silently-dropped rows produces a clean-looking report about
    a subset of reality, which is worse than no report.
    """


@dataclass
class StatementParse:
    """Parsed lines plus the integrity evidence that they were read correctly."""

    txns: list[NormalizedTxn] = field(default_factory=list)
    source_file: str = ""
    bank_format: str = ""
    balance_checked: bool = False
    balance_rows_verified: int = 0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_txn(
    *,
    external_id: str,
    amount_paise: int,
    value_date: date,
    narration: str,
    bank_format: str,
    row_no: int,
    source_file: str,
    posted_date: date | None = None,
) -> NormalizedTxn:
    refs = extract_references(narration)
    raw = {
        "ref": external_id,
        "amount": amount_paise,
        "narration": narration,
        "value_date": value_date.isoformat(),
        "posted": (posted_date or value_date).isoformat(),
    }
    return NormalizedTxn(
        source="bank",
        external_id=external_id,
        amount_paise=amount_paise,
        currency="INR",
        utr=refs[0] if refs else None,
        narration=narration,
        value_date=value_date,
        ref_candidates=refs,
        raw_payload_hash=payload_hash("bank", external_id, raw),
        provenance={
            "format": bank_format,
            "posted_date": (posted_date or value_date).isoformat(),
            "source_file": source_file,
            "row": row_no,
        },
    )


def _verify_running_balance(
    rows: list[tuple[int, int, int]], label: str, warnings: list[str]
) -> int:
    """rows = [(row_no, movement_paise, closing_balance_paise)]. Returns rows verified.

    Checks a difference, not an absolute: the opening balance in a statement
    header is often stale or formatted differently, while consecutive closing
    balances are always internally consistent. One row cannot be verified (there
    is nothing before it), which is why the count is len-1 and not len.
    """
    verified = 0
    for i in range(1, len(rows)):
        row_no, movement, closing = rows[i]
        _, _, prev_closing = rows[i - 1]
        if closing - prev_closing != movement:
            raise ParseError(
                f"{label}: running balance broken at row {row_no}: "
                f"prev={prev_closing} + movement={movement} != closing={closing}. "
                "A column was misread; refusing to reconcile against it."
            )
        verified += 1
    if not rows:
        warnings.append(f"{label}: no rows found")
    return verified


def _find_header_row(rows: list[list[str]], required: Iterable[str]) -> int:
    """Locate the header among a statement's preamble lines.

    HDFC exports put an account block above the table. Scanning for the header
    rather than assuming row 0 is what makes the parser survive a bank adding a
    line to its own boilerplate.
    """
    needed = [r.lower() for r in required]
    for idx, row in enumerate(rows[:40]):
        cells = [c.strip().lower() for c in row]
        if all(any(n in c for c in cells) for n in needed):
            return idx
    raise ParseError(f"could not find a header row containing {list(required)}")


def _cell(row: list[str], idx: int) -> str:
    return row[idx].strip() if idx < len(row) else ""


# ---------------------------------------------------------------------------
# HDFC-style CSV
# ---------------------------------------------------------------------------

def parse_hdfc(path: Path) -> StatementParse:
    out = StatementParse(source_file=path.name, bank_format="hdfc")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    hdr_idx = _find_header_row(rows, ["narration", "value dt", "closing balance"])
    header = [c.strip().lower() for c in rows[hdr_idx]]
    col = {name: i for i, name in enumerate(header)}

    def need(*names: str) -> int:
        for n in names:
            for h, i in col.items():
                if n in h:
                    return i
        raise ParseError(f"HDFC statement missing column {names[0]!r}")

    c_date, c_narr = need("date"), need("narration")
    c_ref, c_value = need("chq", "ref"), need("value dt")
    c_wd, c_dp, c_bal = need("withdrawal"), need("deposit"), need("closing")

    balance_rows: list[tuple[int, int, int]] = []
    for n, row in enumerate(rows[hdr_idx + 1 :], start=hdr_idx + 2):
        if not any(c.strip() for c in row):
            continue
        try:
            posted = datetime.strptime(_cell(row, c_date), "%d/%m/%y").date()
            value = datetime.strptime(_cell(row, c_value), "%d/%m/%y").date()
        except ValueError as exc:
            raise ParseError(f"{path.name} row {n}: bad HDFC date ({exc})") from exc

        wd, dp = _cell(row, c_wd), _cell(row, c_dp)
        if bool(wd) == bool(dp):
            raise ParseError(
                f"{path.name} row {n}: expected exactly one of withdrawal/deposit, "
                f"got wd={wd!r} dp={dp!r}"
            )
        amount = from_rupee_string(dp) if dp else -from_rupee_string(wd)
        closing = from_rupee_string(_cell(row, c_bal))

        out.txns.append(
            _make_txn(
                external_id=_cell(row, c_ref),
                amount_paise=amount,
                value_date=value,
                posted_date=posted,
                narration=" ".join(_cell(row, c_narr).split()),
                bank_format="hdfc",
                row_no=n,
                source_file=path.name,
            )
        )
        balance_rows.append((n, amount, closing))

    out.balance_rows_verified = _verify_running_balance(balance_rows, path.name, out.warnings)
    out.balance_checked = True
    return out


# ---------------------------------------------------------------------------
# ICICI-style CSV
# ---------------------------------------------------------------------------

def parse_icici(path: Path) -> StatementParse:
    out = StatementParse(source_file=path.name, bank_format="icici")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    hdr_idx = _find_header_row(rows, ["value date", "transaction remarks", "balance"])
    header = [c.strip().lower() for c in rows[hdr_idx]]

    def need(*names: str) -> int:
        for n in names:
            for i, h in enumerate(header):
                if n in h:
                    return i
        raise ParseError(f"ICICI statement missing column {names[0]!r}")

    c_value, c_txn = need("value date"), need("transaction date")
    c_ref, c_remarks = need("cheque"), need("remarks")
    c_wd, c_dp, c_bal = need("withdrawal"), need("deposit"), need("balance")

    balance_rows: list[tuple[int, int, int]] = []
    for n, row in enumerate(rows[hdr_idx + 1 :], start=hdr_idx + 2):
        if not any(c.strip() for c in row):
            continue
        try:
            value = datetime.strptime(_cell(row, c_value), "%d-%m-%Y").date()
            posted = datetime.strptime(_cell(row, c_txn), "%d-%m-%Y").date()
        except ValueError as exc:
            raise ParseError(f"{path.name} row {n}: bad ICICI date ({exc})") from exc

        # ICICI writes 0.00 rather than blank in the unused column, so presence
        # cannot be used as the sign test the way it is for HDFC.
        wd = from_rupee_string(_cell(row, c_wd) or "0")
        dp = from_rupee_string(_cell(row, c_dp) or "0")
        if wd and dp:
            raise ParseError(f"{path.name} row {n}: both withdrawal and deposit non-zero")
        if not wd and not dp:
            continue  # a genuinely zero-value line: informational, not money
        amount = dp if dp else -wd
        closing = from_rupee_string(_cell(row, c_bal))

        out.txns.append(
            _make_txn(
                external_id=_cell(row, c_ref),
                amount_paise=amount,
                value_date=value,
                posted_date=posted,
                narration=" ".join(_cell(row, c_remarks).split()),
                bank_format="icici",
                row_no=n,
                source_file=path.name,
            )
        )
        balance_rows.append((n, amount, closing))

    out.balance_rows_verified = _verify_running_balance(balance_rows, path.name, out.warnings)
    out.balance_checked = True
    return out


# ---------------------------------------------------------------------------
# MT940
# ---------------------------------------------------------------------------

_MT_61 = re.compile(
    r"^:61:(?P<value>\d{6})(?P<entry>\d{4})?(?P<mark>RC|RD|C|D)"
    r"(?P<amount>[\d,]+)(?P<code>[A-Z]{4})(?P<rest>.*)$"
)
_MT_BAL = re.compile(r"^:6[02]F:(?P<mark>[CD])(?P<date>\d{6})(?P<ccy>[A-Z]{3})(?P<amount>[\d,]+)$")


def _mt940_amount_to_paise(text: str) -> int:
    """MT940 uses ``,`` as the decimal separator: ``41208,00`` is Rs 41,208.00.

    Handing this string to the CSV amount parser would read it as
    ``4120800`` rupees -- a 100x error that is *also* a plausible amount, so
    nothing downstream would flag it. Hence a separate function rather than a
    shared "smart" one.
    """
    whole, _, frac = text.partition(",")
    frac = (frac + "00")[:2]
    if not whole.isdigit() or not frac.isdigit():
        raise ParseError(f"bad MT940 amount {text!r}")
    return int(whole) * 100 + int(frac)


def _mt940_entry_date(value_date: date, mmdd: str | None) -> date:
    """MT940 entry dates carry no year, so it has to be inferred.

    A December value date with a January entry date has rolled over; the
    +/-180-day correction handles both directions. Without it, a statement
    spanning New Year silently produces dates a year off, which reads as extreme
    timing drift rather than as a parser bug.
    """
    if not mmdd:
        return value_date
    month, day = int(mmdd[:2]), int(mmdd[2:])
    for year in (value_date.year, value_date.year + 1, value_date.year - 1):
        try:
            cand = date(year, month, day)
        except ValueError:
            continue
        if abs((cand - value_date).days) <= 180:
            return cand
    return value_date


def parse_mt940(path: Path) -> StatementParse:
    out = StatementParse(source_file=path.name, bank_format="mt940")
    lines = path.read_text(encoding="utf-8").splitlines()

    opening: int | None = None
    closing: int | None = None
    movements = 0
    pending: dict | None = None
    n_txn = 0

    def flush(narration_parts: list[str]) -> None:
        nonlocal pending, movements, n_txn
        if pending is None:
            return
        narration = " ".join(" ".join(narration_parts).split())
        out.txns.append(
            _make_txn(
                external_id=pending["ref"],
                amount_paise=pending["amount"],
                value_date=pending["value"],
                posted_date=pending["entry"],
                narration=narration,
                bank_format="mt940",
                row_no=pending["line_no"],
                source_file=path.name,
            )
        )
        movements += pending["amount"]
        n_txn += 1
        pending = None

    narration_parts: list[str] = []
    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line or line == "-":
            continue

        if line.startswith(":61:"):
            flush(narration_parts)
            narration_parts = []
            m = _MT_61.match(line)
            if not m:
                raise ParseError(f"{path.name} line {line_no}: unparseable :61: {line!r}")
            value = datetime.strptime(m.group("value"), "%y%m%d").date()
            amount = _mt940_amount_to_paise(m.group("amount"))
            if m.group("mark") in ("D", "RC"):
                # D is a debit. RC is a *reversal of a credit*, which is also a
                # debit -- a detail that is easy to miss and inverts a sign.
                amount = -amount
            rest = m.group("rest")
            ref = rest.split("//", 1)[1].strip() if "//" in rest else rest[:16].strip()
            if not ref:
                raise ParseError(f"{path.name} line {line_no}: :61: has no reference")
            pending = {
                "ref": ref,
                "amount": amount,
                "value": value,
                "entry": _mt940_entry_date(value, m.group("entry")),
                "line_no": line_no,
            }
        elif line.startswith(":86:"):
            narration_parts.append(line[4:])
        elif line.startswith((":60F:", ":62F:")):
            m = _MT_BAL.match(line)
            if not m:
                raise ParseError(f"{path.name} line {line_no}: unparseable balance {line!r}")
            val = _mt940_amount_to_paise(m.group("amount"))
            if m.group("mark") == "D":
                val = -val
            if line.startswith(":60F:"):
                opening = val
            else:
                closing = val
    flush(narration_parts)

    if opening is not None and closing is not None:
        if closing - opening != movements:
            raise ParseError(
                f"{path.name}: MT940 balances inconsistent -- opening {opening} + "
                f"movements {movements} != closing {closing}"
            )
        out.balance_checked = True
        out.balance_rows_verified = n_txn
    else:
        out.warnings.append(f"{path.name}: missing :60F:/:62F:, balance not verified")
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PARSERS: dict[str, Callable[[Path], StatementParse]] = {
    "hdfc": parse_hdfc,
    "icici": parse_icici,
    "mt940": parse_mt940,
}


def sniff_format(path: Path) -> str:
    """Pick a parser from the file's own content, not its name.

    Filenames are the merchant's business, not a contract. Sniffing on content
    means ``march_statement_final_v2.csv`` still parses, and an unrecognised file
    raises rather than being fed to a default parser that would misread it.
    """
    head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    if ":61:" in head or head.lstrip().startswith(":20:"):
        return "mt940"
    lowered = head.lower()
    if "transaction remarks" in lowered or "s no." in lowered:
        return "icici"
    if "narration" in lowered and "value dt" in lowered:
        return "hdfc"
    raise ParseError(f"{path.name}: unrecognised bank statement format")


def parse_statement(path: Path) -> StatementParse:
    return PARSERS[sniff_format(path)](path)


def parse_all_statements(paths: Iterable[Path]) -> tuple[list[NormalizedTxn], list[StatementParse]]:
    """Merge every account into one canonical stream.

    Ordering is by (value_date, external_id) rather than file order so a run over
    the same three statements presented in a different order produces identical
    results. Reconciliation output that depends on which file was read first is
    not reproducible, and irreproducible output cannot be audited.
    """
    parses = [parse_statement(p) for p in paths]
    txns = [t for p in parses for t in p.txns]
    txns.sort(key=lambda t: (t.value_date, t.external_id))

    seen: dict[str, NormalizedTxn] = {}
    for t in txns:
        if t.external_id in seen:
            raise ParseError(
                f"bank reference {t.external_id!r} appears in two statements; "
                "line identity is ambiguous and matching would double-count"
            )
        seen[t.external_id] = t
    return txns, parses
