"""Canonical transaction schema -- the narrow waist of the system.

Every source (Razorpay settlements API, three different bank CSV dialects, the
merchant's books) is normalised into ``NormalizedTxn`` before anything else runs.
The matching engine, the event store, and the UI only ever see this shape, which
is why adding a fourth bank format later is an ingestion-layer change and not a
matcher change.

Design notes worth stating out loud:

* ``amount_paise`` is **signed**: credits positive, debits/refunds negative.
  Refund netting (break pattern #3) then falls out of ordinary subset summation
  instead of needing a special case in the matcher.
* ``txn_id`` is derived, not random: ``{source}:{external_id}``. Deterministic
  ids mean re-ingesting the same file produces the same ids, which is what makes
  idempotency checkable rather than aspirational.
* ``raw_payload_hash`` is over the *source bytes*, so a bank that re-exports the
  same line with a different narration is a genuinely new fact and gets a new
  event, while a byte-identical re-export is a no-op.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .money import check_paise, inr

Source = Literal["gateway", "bank", "books"]

# The eleven break patterns from the problem statement. These strings are the
# vocabulary shared by the generator (which injects them), the eval harness
# (which slices metrics by them), and the workbench (which shows them).
BreakPattern = Literal[
    "clean",
    "batch_settlement",      # 1  N payments -> 1 payout
    "split_settlement",      # 2  1 payment  -> N payouts
    "refund_netting",        # 3  batch nets out refunds
    "fee_tax_drift",         # 4  MDR/GST rounding vs merchant assumption
    "tds_mismatch",          # 5  statutory deduction the gateway never sees
    "duplicate_bank_entry",  # 6  reversal + re-settlement shows twice
    "orphan_bank_credit",    # 7  money in bank, no gateway record
    "missing_settlement",    # 8  gateway processed, bank has not credited
    "timing_drift",          # 9  bank holiday / value-date != statement date
    "narration_noise",       # 10 UTR truncated or merged into free text
    "fx_mismatch",           # 11 invoiced in USD, settled at a different rate
]

ALL_BREAK_PATTERNS: tuple[str, ...] = (
    "batch_settlement",
    "split_settlement",
    "refund_netting",
    "fee_tax_drift",
    "tds_mismatch",
    "duplicate_bank_entry",
    "orphan_bank_credit",
    "missing_settlement",
    "timing_drift",
    "narration_noise",
    "fx_mismatch",
)


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace.

    Every hash in this system (payload hashes, the event chain, projection
    fingerprints) goes through here, so "same content" always means the same
    bytes regardless of dict insertion order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_hash(source: str, external_id: str, payload: dict[str, Any]) -> str:
    return sha256_hex(canonical_json({"s": source, "e": external_id, "p": payload}))


class NormalizedTxn(BaseModel):
    """One line of money as seen by one source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Source
    external_id: str
    amount_paise: int          # signed; credit +, debit/refund -
    currency: str = "INR"
    utr: str | None = None
    narration: str | None = None
    value_date: date
    fees_paise: int = 0
    tax_paise: int = 0
    raw_payload_hash: str
    #: Every reference-shaped token ingestion could find, best first. ``utr`` is
    #: just ``ref_candidates[0]`` promoted for display. Keeping the alternatives
    #: means a parser that ranks wrong costs readability rather than recall --
    #: the scoring layer sees all of them and picks with the amount and date
    #: evidence in hand.
    ref_candidates: tuple[str, ...] = ()
    # Free-form provenance: which file/row, which API page. Never used for
    # matching decisions -- only for showing a human where a number came from.
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("amount_paise", "fees_paise", "tax_paise")
    @classmethod
    def _money_is_int(cls, v: int, info: Any) -> int:
        return check_paise(v, info.field_name)

    @field_validator("currency")
    @classmethod
    def _currency_upper(cls, v: str) -> str:
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"currency must be a 3-letter code, got {v!r}")
        return v.upper()

    @field_validator("utr")
    @classmethod
    def _normalise_utr(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().upper()
        return v or None

    @property
    def txn_id(self) -> str:
        return f"{self.source}:{self.external_id}"

    @property
    def gross_paise(self) -> int:
        """Amount before the gateway took its cut.

        For a gateway settlement, ``amount_paise`` is the **net** credited to the
        bank; gross is what the customer paid. Getting this direction wrong is
        the single most common way to mis-model Razorpay's settlement entity, so
        it lives in one property rather than being re-derived at three call
        sites.
        """
        return self.amount_paise + self.fees_paise + self.tax_paise

    def __str__(self) -> str:  # pragma: no cover - debug affordance
        return f"<{self.txn_id} {inr(self.amount_paise)} {self.value_date} utr={self.utr}>"


class GroundTruthLink(BaseModel):
    """What the synthetic generator knows and the matcher must discover.

    A link is many-to-many by construction: ``left_ids`` x ``right_ids``. A clean
    1:1 is just the degenerate case, which keeps batch (N:1) and split (1:N)
    from being second-class citizens in the evaluation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    leg: Literal["gateway_bank", "gateway_books"]
    left_ids: tuple[str, ...]
    right_ids: tuple[str, ...]
    pattern: str = "clean"

    @property
    def cardinality(self) -> str:
        n, m = len(self.left_ids), len(self.right_ids)
        if n == 1 and m == 1:
            return "1:1"
        if m == 1:
            return "N:1"
        if n == 1:
            return "1:N"
        return "N:M"

    def pair_keys(self) -> set[tuple[str, str]]:
        """Flatten to the set of (left, right) pairs this link asserts.

        Pair-level flattening is how precision/recall are computed. It is the
        strict reading: a proposed N:1 group only scores full credit if every
        constituent pair is correct, and a group that gets 3 of 4 legs right
        earns 3 true positives and 1 false negative rather than a free pass.
        """
        return {(l, r) for l in self.left_ids for r in self.right_ids}


class UnmatchableTxn(BaseModel):
    """Ground truth for lines that *should* end up as exceptions.

    Recording these explicitly is what makes "honest exception list" measurable:
    a system that flags exactly these and nothing else is perfect, and one that
    quietly matches them is wrong even if its match rate looks better.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    txn_id: str
    reason: str
    leg: Literal["gateway_bank", "gateway_books"]


class Dataset(BaseModel):
    """A generated batch plus its answer key."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    generated_at: str
    gateway: list[NormalizedTxn]
    bank: list[NormalizedTxn]
    books: list[NormalizedTxn]
    links: list[GroundTruthLink]
    unmatchable: list[UnmatchableTxn]
    injection_log: list[dict[str, Any]] = Field(default_factory=list)

    def by_id(self) -> dict[str, NormalizedTxn]:
        out: dict[str, NormalizedTxn] = {}
        for t in (*self.gateway, *self.bank, *self.books):
            out[t.txn_id] = t
        return out

    def links_for(self, leg: str) -> list[GroundTruthLink]:
        return [l for l in self.links if l.leg == leg]
