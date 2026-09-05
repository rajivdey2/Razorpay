"""Ingestion layer: raw source files in, one canonical stream out.

    from app.ingestion import load_batch
    batch = load_batch(Path("data/eval"))

Everything downstream sees ``NormalizedTxn`` and nothing else. Adding a fourth
bank is a new function in ``bank.PARSERS``; the matcher does not learn about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.schema import NormalizedTxn

from .bank import ParseError, StatementParse, parse_all_statements, parse_statement, sniff_format
from .books import BooksIngest, parse_books
from .gateway import GatewayIngest, SettlementComponent, parse_gateway
from .refs import best_reference, extract_references

__all__ = [
    "IngestedBatch", "load_batch", "ParseError", "StatementParse",
    "parse_statement", "parse_all_statements", "sniff_format",
    "parse_books", "BooksIngest", "parse_gateway", "GatewayIngest",
    "SettlementComponent", "extract_references", "best_reference",
]

#: Bank statements are found by extension, not by name, so a merchant can drop in
#: ``current_account_march.csv`` without editing code. Format detection is then
#: content-based (``bank.sniff_format``).
BANK_GLOBS = ("*.csv", "*.mt940", "*.sta", "*.txt")
NON_BANK_NAMES = {"books_receivables.csv"}


@dataclass
class IngestedBatch:
    gateway: GatewayIngest
    books: BooksIngest
    bank: list[NormalizedTxn] = field(default_factory=list)
    statements: list[StatementParse] = field(default_factory=list)
    source_dir: str = ""

    @property
    def all_txns(self) -> list[NormalizedTxn]:
        return [*self.gateway.settlements, *self.bank, *self.books.entries]

    def summary(self) -> dict:
        return {
            "gateway_settlements": len(self.gateway.settlements),
            "bank_lines": len(self.bank),
            "book_entries": len(self.books.entries),
            "statements": [
                {
                    "file": s.source_file,
                    "format": s.bank_format,
                    "lines": len(s.txns),
                    "balance_verified_rows": s.balance_rows_verified,
                    "balance_checked": s.balance_checked,
                    "warnings": s.warnings,
                }
                for s in self.statements
            ],
            "order_keys_available": len(self.gateway.order_to_settlement),
            "book_entries_with_order_ref": sum(
                1 for e in self.books.entries if e.provenance.get("order_ref")
            ),
        }


def load_batch(directory: Path) -> IngestedBatch:
    directory = Path(directory)
    gateway = parse_gateway(
        directory / "gateway_settlements.json",
        directory / "gateway_payments.json",
    )
    books = parse_books(directory / "books_receivables.csv")

    candidates: list[Path] = []
    for pattern in BANK_GLOBS:
        for p in sorted(directory.glob(pattern)):
            if p.name in NON_BANK_NAMES or p.name.startswith("ground_truth"):
                continue
            candidates.append(p)
    # De-duplicate while preserving the sorted order (``*.csv`` and ``*.txt`` can
    # both match on case-insensitive filesystems).
    seen: set[str] = set()
    paths = [p for p in candidates if not (p.name in seen or seen.add(p.name))]

    bank, statements = parse_all_statements(paths)
    return IngestedBatch(
        gateway=gateway, books=books, bank=bank,
        statements=statements, source_dir=str(directory),
    )
