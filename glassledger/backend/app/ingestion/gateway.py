"""Gateway ingestion: Razorpay ``settlement`` entities + the settlement recon view.

Two files, because a real integration reads two endpoints:

``gateway_settlements.json``
    ``GET /v1/settlements`` -- the payout itself. ``amount`` is the **net**
    credited, ``fees``/``tax`` are what was withheld, everything in paise,
    ``created_at`` a Unix timestamp.

``gateway_payments.json``
    the settlement recon report -- which payments and refunds rolled into which
    payout, each payment carrying the merchant's ``order_id``.

The recon view is what makes deterministic matching possible on the books leg: an
``order_id`` present in the merchant's ERP resolves transitively to a settlement.
Roughly 60% of book entries carry one. The other 40%, plus every credit note,
withholding accrual and FX adjustment, have no key at all and can only be resolved
as part of a summed group.

``created_at`` is a Unix timestamp and is converted with an explicit UTC
interpretation. Razorpay timestamps are IST-relative in the dashboard and epoch in
the API, and a naive ``datetime.fromtimestamp`` silently applies the *server's*
local zone -- so the same file reconciled on a laptop in Berhampur and a container
in Virginia would produce value dates one day apart, near midnight, for a subset of
rows. That is a genuinely miserable bug to find, so the conversion is pinned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.core.schema import NormalizedTxn, payload_hash


@dataclass
class SettlementComponent:
    """One payment or refund inside a payout, for evidence and residual maths."""

    kind: str                  # payment | refund
    component_id: str
    amount_paise: int          # signed: payments +, refunds -
    order_ref: str | None
    method: str | None
    invoice_currency: str
    invoice_amount_minor: int | None


@dataclass
class GatewayIngest:
    settlements: list[NormalizedTxn] = field(default_factory=list)
    #: settlement_id -> its constituent payments/refunds
    components: dict[str, list[SettlementComponent]] = field(default_factory=dict)
    #: order_id -> settlement_id, the deterministic bridge into the books leg
    order_to_settlement: dict[str, str] = field(default_factory=dict)
    unsettled_orders: set[str] = field(default_factory=set)

    def gross_paise(self, settlement_id: str) -> int:
        return sum(c.amount_paise for c in self.components.get(settlement_id, []))


def _epoch_to_date(ts: int):
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date()


def parse_gateway(settlements_path: Path, payments_path: Path | None = None) -> GatewayIngest:
    out = GatewayIngest()

    doc = json.loads(settlements_path.read_text(encoding="utf-8"))
    items = doc["items"] if isinstance(doc, dict) and "items" in doc else doc
    for raw in items:
        if raw.get("entity") not in (None, "settlement"):
            continue
        sid = raw["id"]
        amount = int(raw["amount"])
        fees = int(raw.get("fees") or 0)
        tax = int(raw.get("tax") or 0)
        utr = (raw.get("utr") or "").strip().upper() or None
        value_date = _epoch_to_date(raw["created_at"])
        out.settlements.append(
            NormalizedTxn(
                source="gateway",
                external_id=sid,
                amount_paise=amount,
                currency="INR",
                utr=utr,
                narration=f"settlement {sid} status={raw.get('status')}",
                value_date=value_date,
                fees_paise=fees,
                tax_paise=tax,
                ref_candidates=(utr,) if utr else (),
                raw_payload_hash=payload_hash("gateway", sid, raw),
                provenance={
                    "status": raw.get("status"),
                    "source_file": settlements_path.name,
                },
            )
        )

    if payments_path is not None and payments_path.exists():
        pdoc = json.loads(payments_path.read_text(encoding="utf-8"))
        pitems = pdoc["items"] if isinstance(pdoc, dict) and "items" in pdoc else pdoc
        for raw in pitems:
            sid = raw.get("settlement_id")
            kind = raw.get("entity", "payment")
            order_ref = raw.get("order_id")
            comp = SettlementComponent(
                kind=kind,
                component_id=raw["id"],
                amount_paise=int(raw["amount"]),
                order_ref=order_ref,
                method=raw.get("method"),
                invoice_currency=raw.get("invoice_currency") or "INR",
                invoice_amount_minor=raw.get("invoice_amount"),
            )
            if sid:
                out.components.setdefault(sid, []).append(comp)
                if order_ref:
                    out.order_to_settlement[order_ref] = sid
            elif order_ref:
                # Captured but not yet settled. Tracked explicitly so a books
                # entry pointing here becomes "awaiting settlement" rather than
                # an unexplained orphan -- same open item, hugely different
                # message to whoever is clearing the queue.
                out.unsettled_orders.add(order_ref)

    # Enrich each settlement with its component count. Used by the ambiguity and
    # residual-attribution features, and by the workbench's evidence panel.
    enriched: list[NormalizedTxn] = []
    for t in out.settlements:
        comps = out.components.get(t.external_id, [])
        enriched.append(
            t.model_copy(
                update={
                    "provenance": {
                        **t.provenance,
                        "n_payments": sum(1 for c in comps if c.kind == "payment"),
                        "n_refunds": sum(1 for c in comps if c.kind == "refund"),
                        "gross_paise": sum(c.amount_paise for c in comps),
                    }
                }
            )
        )
    out.settlements = sorted(enriched, key=lambda t: (t.value_date, t.external_id))
    return out
