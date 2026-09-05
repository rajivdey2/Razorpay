"""FastAPI app: the CQRS query side, plus the two commands a human can issue.

Read endpoints serve from a projection rebuilt off the event stream. Write endpoints
(``resolve`` an exception, ``confirm``/``reject`` a proposed match) append events and
let the projection catch up -- they never mutate a read model directly, so a human
override is exactly as auditable as an agent decision and lands in the same trail.

Design notes:

* The projection is rebuilt on demand rather than held in a long-lived cache. At this
  scale a full rebuild is a few tens of milliseconds, and correctness-by-construction
  is worth more than the milliseconds. A real deployment would keep a warm projection
  and call ``catch_up``, which is already implemented and tested for equivalence with
  a cold rebuild.
* No auth. This is a local demo tool, and pretending otherwise with a hardcoded bearer
  token would be theatre. The endpoints that write are marked, and the deployment note
  in the README says what would need to change.
* Amounts cross the wire as integer paise with a formatted string alongside. The
  frontend never does money arithmetic; it renders what the backend computed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.core.money import inr
from app.events import EventStore
from app.projections import audit_trail, rebuild

FRONTEND = Path(__file__).resolve().parents[3] / "frontend"


class ResolveRequest(BaseModel):
    resolution: str = Field(
        description="accept | reject | reassign | no_action | escalate"
    )
    note: str = ""
    resolved_by: str = "human"
    reassign_to: str | None = None


class MatchActionRequest(BaseModel):
    action: str = Field(description="confirm | reject")
    note: str = ""
    actor: str = "human"


def create_app(db_path: Path) -> FastAPI:
    app = FastAPI(
        title="GlassLedger",
        description="Multi-source reconciliation with a calibrated, auditable agent",
        version="1.0.0",
    )

    def store() -> EventStore:
        return EventStore(db_path)

    def _money(paise: int) -> dict:
        return {"paise": paise, "display": inr(paise)}

    # -- reads ---------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        s = store()
        try:
            v = s.verify_chain()
            return {
                "ok": True,
                "events": s.count(),
                "chain_ok": v.ok,
                "chain_detail": v.detail,
                "db": str(db_path),
            }
        finally:
            s.close()

    @app.get("/api/metrics")
    def metrics() -> dict:
        s = store()
        try:
            p = rebuild(s)
            m = p.metrics()
            v = s.verify_chain()
            balanced, drift = p.ledger_balanced()
            return {
                **m,
                "bank_leg_reconciled": _money(m["bank_leg_reconciled_paise"]),
                "pending_human": _money(m["pending_human_paise"]),
                "exception_amount": _money(m["exception_paise"]),
                "ledger_display": {
                    k: inr(v_) for k, v_ in sorted(p.ledger.items())
                },
                "integrity": {
                    "chain_ok": v.ok,
                    "chain_detail": v.detail,
                    "replay_fingerprint": p.fingerprint()[:24],
                    "ledger_balanced": balanced,
                    "ledger_drift_paise": drift,
                },
                "event_types": s.type_counts(),
            }
        finally:
            s.close()

    @app.get("/api/exceptions")
    def exceptions(status: str = "open", limit: int = 200) -> dict:
        s = store()
        try:
            p = rebuild(s)
            items = (
                p.open_exceptions()
                if status == "open"
                else sorted(p.exceptions.values(), key=lambda e: -e.priority)
            )
            return {
                "count": len(items),
                "total": _money(sum(abs(e.amount_paise) for e in items)),
                "items": [
                    {**e.to_json(), "amount": _money(e.amount_paise)}
                    for e in items[:limit]
                ],
            }
        finally:
            s.close()

    @app.get("/api/matches")
    def matches(status: str = "proposed", limit: int = 200) -> dict:
        s = store()
        try:
            p = rebuild(s)
            if status == "proposed":
                items = p.pending_matches()
            elif status == "confirmed":
                items = sorted(
                    p.confirmed_matches(),
                    key=lambda m: -max(abs(m.left_amount_paise), abs(m.right_amount_paise)),
                )
            else:
                items = list(p.matches.values())
            return {
                "count": len(items),
                "items": [
                    {
                        **m.to_json(),
                        "exposure": _money(
                            max(abs(m.left_amount_paise), abs(m.right_amount_paise))
                        ),
                        "left_display": inr(m.left_amount_paise),
                        "right_display": inr(m.right_amount_paise),
                        "residual_display": inr(m.residual_paise),
                    }
                    for m in items[:limit]
                ],
            }
        finally:
            s.close()

    @app.get("/api/txn/{txn_id:path}")
    def txn(txn_id: str) -> dict:
        s = store()
        try:
            p = rebuild(s)
            t = p.txns.get(txn_id)
            if t is None:
                raise HTTPException(404, f"unknown transaction {txn_id}")
            return {
                **t.to_json(),
                "amount": _money(t.amount_paise),
                "event_seqs": p.txn_index.get(txn_id, []),
            }
        finally:
            s.close()

    @app.get("/api/audit/{txn_id:path}")
    def audit(txn_id: str) -> dict:
        """The replayed history of one transaction. The demo's centrepiece."""
        s = store()
        try:
            trail = audit_trail(s, txn_id)
            if not trail:
                raise HTTPException(404, f"no events touch {txn_id}")
            return {
                "txn_id": txn_id,
                "events": len(trail),
                "trail": trail,
                "note": "reconstructed by replaying the append-only event log",
            }
        finally:
            s.close()

    # -- writes (append events; never mutate a projection) --------------

    @app.post("/api/exceptions/{exception_id}/resolve")
    def resolve(exception_id: str, req: ResolveRequest) -> dict:
        s = store()
        try:
            p = rebuild(s)
            e = p.exceptions.get(exception_id)
            if e is None:
                raise HTTPException(404, f"unknown exception {exception_id}")
            if e.status == "resolved":
                # Idempotent by nature: the event is already there, so re-posting
                # returns the current state rather than appending a second
                # resolution with a different author.
                return {"exception_id": exception_id, "status": "already_resolved",
                        "resolution": e.resolution}
            ev = s.append(
                "ExceptionResolved",
                exception_id,
                {
                    "resolution": req.resolution,
                    "resolved_by": req.resolved_by,
                    "note": req.note,
                    "reassign_to": req.reassign_to,
                    "category": e.category,
                    "txn_ids": e.txn_ids,
                    "amount_paise": e.amount_paise,
                    "resolved_at": datetime.now(timezone.utc).replace(
                        microsecond=0
                    ).isoformat(),
                },
                idempotency_key=f"resolve:{exception_id}",
            )
            return {"exception_id": exception_id, "status": "resolved",
                    "event_seq": ev.seq, "chain_hash": ev.hash[:16]}
        finally:
            s.close()

    @app.post("/api/matches/{match_key:path}/action")
    def match_action(match_key: str, req: MatchActionRequest) -> dict:
        """Human confirmation or rejection of a proposed match.

        A rejection appends ``MatchRejected`` rather than deleting the proposal.
        The wrong answer stays in the history next to the right one, which is the
        whole reason this is event-sourced: an auto-match that a human overturned
        is the most useful training signal the system will ever get, and deleting
        it throws that away.
        """
        s = store()
        try:
            p = rebuild(s)
            m = p.matches.get(match_key)
            if m is None:
                raise HTTPException(404, f"unknown match {match_key}")
            if req.action == "confirm":
                ev = s.append(
                    "MatchConfirmed", match_key,
                    {
                        "leg": m.leg, "confirmed_by": req.actor,
                        "confidence": m.confidence, "tier": m.tier,
                        "algorithm": m.algorithm,
                        "rationale": req.note or "confirmed by human review",
                        "gate": "human_approval",
                    },
                    idempotency_key=f"human_confirm:{match_key}",
                )
            elif req.action == "reject":
                ev = s.append(
                    "MatchRejected", match_key,
                    {"leg": m.leg, "rejected_by": req.actor,
                     "reason": req.note or "rejected by human review",
                     "gate": "human_approval"},
                    idempotency_key=f"human_reject:{match_key}",
                )
            else:
                raise HTTPException(400, "action must be 'confirm' or 'reject'")
            return {"match_key": match_key, "action": req.action,
                    "event_seq": ev.seq, "chain_hash": ev.hash[:16]}
        finally:
            s.close()

    @app.get("/api/integrity")
    def integrity() -> dict:
        """Everything a sceptic would want to check, in one call."""
        s = store()
        try:
            v = s.verify_chain()
            p1 = rebuild(s)
            p2 = rebuild(s)
            balanced, drift = p1.ledger_balanced()
            mutation_blocked = True
            detail = ""
            try:
                s._conn.execute("UPDATE events SET event_type='x' WHERE seq=1")
                s._conn.commit()
                mutation_blocked = False
            except Exception as exc:
                detail = str(exc)
            return {
                "hash_chain": {"ok": v.ok, "events": v.events_checked,
                               "detail": v.detail},
                "replay_deterministic": p1.fingerprint() == p2.fingerprint(),
                "replay_fingerprint": p1.fingerprint(),
                "ledger_balanced": balanced,
                "ledger_drift_paise": drift,
                "append_only_enforced": mutation_blocked,
                "append_only_detail": detail,
            }
        finally:
            s.close()

    # -- static workbench ------------------------------------------------

    @app.get("/")
    def index():
        f = FRONTEND / "index.html"
        if not f.exists():
            return JSONResponse({"error": "frontend not built", "looked_in": str(f)}, 404)
        return FileResponse(f)

    @app.get("/app.js")
    def appjs():
        return FileResponse(FRONTEND / "app.js", media_type="application/javascript")

    @app.get("/app.css")
    def appcss():
        return FileResponse(FRONTEND / "app.css", media_type="text/css")

    @app.get("/favicon.ico")
    def favicon():
        # An empty 204 rather than a 404: the browser asks unprompted, and a 404
        # in the console is noise that hides real errors during a demo.
        from fastapi import Response

        return Response(status_code=204)

    return app
