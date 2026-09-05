"""End-to-end tests over a real generated batch, plus the ingestion round trip.

Session-scoped fixtures generate one small dataset and run the pipeline once, so the
whole file costs a few seconds. These are the tests that would catch a regression in
the *composition* of the parts, which the unit tests cannot see.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from app.core.money import from_rupee_string
from app.events import EventStore, record_run
from app.ingestion import ParseError, load_batch, parse_statement, sniff_format
from app.matching import DummyScorer, ReconciliationEngine
from app.projections import audit_trail, catch_up, rebuild

from data_generator.generate_synthetic import generate


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("gl-data")
    ds = generate(seed=1234, n_payments=180, days=14, out=out)
    return ds, out


@pytest.fixture(scope="session")
def batch(dataset):
    _, out = dataset
    return load_batch(out)


@pytest.fixture(scope="session")
def run(batch):
    return ReconciliationEngine(DummyScorer(), threshold=0.85).run(batch)


class TestIngestionRoundTrip:
    def test_every_line_survives_the_file_round_trip(self, dataset, batch):
        """Ids, amounts and dates must come back exactly through three dialects.

        The generator writes HDFC CSV, ICICI CSV and MT940; the parsers read them
        back. Anything lost here is lost before the matcher ever runs, and would
        show up as an accuracy ceiling nobody could explain.
        """
        ds, _ = dataset
        truth = {t.txn_id: t for t in ds.bank}
        got = {t.txn_id: t for t in batch.bank}
        assert set(truth) == set(got), "bank line identity did not survive ingestion"
        for k, t in truth.items():
            assert got[k].amount_paise == t.amount_paise, f"amount drifted on {k}"
            assert got[k].value_date == t.value_date, f"value date drifted on {k}"

    def test_all_three_dialects_were_exercised(self, batch):
        formats = {s.bank_format for s in batch.statements}
        assert formats == {"hdfc", "icici", "mt940"}

    def test_running_balances_verify(self, batch):
        for s in batch.statements:
            assert s.balance_checked, f"{s.source_file} balance not verified"
            assert not s.warnings, f"{s.source_file}: {s.warnings}"

    def test_gateway_and_books_round_trip(self, dataset, batch):
        ds, _ = dataset
        assert {t.txn_id for t in ds.gateway} == {
            t.txn_id for t in batch.gateway.settlements
        }
        assert {t.txn_id for t in ds.books} == {
            t.txn_id for t in batch.books.entries
        }

    def test_format_sniffing_is_content_based(self, dataset):
        _, out = dataset
        assert sniff_format(out / "bank_hdfc.csv") == "hdfc"
        assert sniff_format(out / "bank_icici.csv") == "icici"
        assert sniff_format(out / "bank_generic.mt940") == "mt940"

    def test_unrecognised_file_raises_rather_than_guessing(self, tmp_path):
        """A default parser on an unknown format misreads silently.

        Refusing is the only safe behaviour: a misparsed statement produces numbers
        that look plausible and are wrong, which is worse than no numbers.
        """
        f = tmp_path / "mystery.csv"
        f.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
        with pytest.raises(ParseError, match="unrecognised"):
            parse_statement(f)

    def test_broken_running_balance_is_caught(self, dataset, tmp_path):
        """Corrupt one deposit and the parser must refuse the whole statement.

        Read and rewrite with the csv module rather than splitting on commas: the
        amounts are quoted *because* they contain commas, and a naive split
        produced a malformed file that failed for the wrong reason.
        """
        import csv as _csv

        _, out = dataset
        with (out / "bank_hdfc.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(_csv.reader(fh))
        hdr = next(i for i, r in enumerate(rows)
                   if any("narration" in c.strip().lower() for c in r))
        # Pick a *later* deposit row: the check compares consecutive closing
        # balances, so the first data row has nothing before it to disagree with and
        # corrupting it is undetectable by construction.
        target = next(
            i for i in range(hdr + 2, len(rows))
            if len(rows[i]) > 6 and rows[i][5].strip() and not rows[i][4].strip()
        )
        rows[target][5] = "9,99,999.00"   # deposit no longer matches the balance move

        bad = tmp_path / "bank_hdfc.csv"
        with bad.open("w", newline="", encoding="utf-8") as fh:
            _csv.writer(fh).writerows(rows)
        with pytest.raises(ParseError, match="running balance"):
            parse_statement(bad)


class TestPipelineInvariants:
    def test_every_transaction_is_accounted_for(self, run, batch):
        """No silent drops. The honesty claim, made checkable.

        A transaction the engine neither resolved nor flagged has vanished, and the
        merchant has no way to know. This assertion is the difference between an
        exception list you can trust and one you hope is complete.
        """
        result = run.assert_complete(batch)
        assert result["complete"]
        assert result["accounted_for"] == result["ingested"]

    def test_no_transaction_is_double_confirmed(self, run):
        """One line cannot be auto-confirmed into two different matches on a leg.

        On gateway->bank both sides are exclusive. On books->gateway a settlement
        legitimately has many entries, so only the entry side is checked.
        """
        for leg, lr in run.legs.items():
            seen_left: set[str] = set()
            seen_right: set[str] = set()
            for d in lr.decisions.auto_confirmed:
                c = d.candidate
                assert not (seen_left & set(c.left_ids)), f"{leg}: left double-claimed"
                seen_left.update(c.left_ids)
                if leg == "gateway_bank":
                    assert not (seen_right & set(c.right_ids)), f"{leg}: right double-claimed"
                    seen_right.update(c.right_ids)

    def test_confirmed_gateway_bank_matches_balance_exactly(self, run):
        """A bank credit either equals the settlement net or it is not a match.

        Auto-confirming a near-miss on this leg would mean posting a journal entry
        whose cash line disagrees with the receivable it clears.
        """
        for d in run.legs["gateway_bank"].decisions.auto_confirmed:
            assert d.candidate.residual_paise == 0, (
                f"{d.candidate.match_key} confirmed with residual "
                f"{d.candidate.residual_paise}"
            )

    def test_currencies_are_never_crossed(self, run, batch):
        by_id = {t.txn_id: t for t in batch.all_txns}
        for lr in run.legs.values():
            for d in (*lr.decisions.auto_confirmed, *lr.decisions.proposed):
                ccy = {by_id[i].currency for i in d.candidate.all_ids() if i in by_id}
                assert len(ccy) <= 1, f"crossed currencies: {ccy}"

    def test_materiality_gate_holds(self, run):
        """No *inferred* match above the gate is ever auto-confirmed.

        Tier 1 identity matches are exempt by documented policy
        (``MATERIALITY_APPLIES_TO_TIER1``); this asserts the exemption is limited to
        exactly those and has not leaked to the probabilistic tiers.
        """
        from app.core.config import MATERIALITY_PAISE

        for lr in run.legs.values():
            for d in lr.decisions.auto_confirmed:
                if d.candidate.exposure_paise >= MATERIALITY_PAISE:
                    assert d.candidate.tier == 1, (
                        f"tier {d.candidate.tier} match auto-confirmed at "
                        f"{d.candidate.exposure_paise} paise, above the gate"
                    )

    def test_wash_pairs_net_to_zero(self, run):
        for c in run.all_wash_pairs():
            assert c.left_amount_paise + c.right_amount_paise == 0

    def test_engine_is_deterministic(self, batch):
        """Same input, same model, same output. Twice.

        Reconciliation output that varies run to run cannot be audited, and a dict
        iteration order or an unsorted tie-break is enough to break it.
        """
        a = ReconciliationEngine(DummyScorer(), threshold=0.85).run(batch)
        b = ReconciliationEngine(DummyScorer(), threshold=0.85).run(batch)
        for leg in a.legs:
            ka = sorted(d.candidate.match_key for d in a.legs[leg].decisions.auto_confirmed)
            kb = sorted(d.candidate.match_key for d in b.legs[leg].decisions.auto_confirmed)
            assert ka == kb


class TestEventSourcing:
    def test_recording_then_replay_is_deterministic(self, batch, run, tmp_path):
        store = EventStore(tmp_path / "e.db")
        try:
            record_run(store, batch, run)
            assert store.verify_chain().ok
            p1 = rebuild(store)
            p2 = rebuild(store)
            assert p1.fingerprint() == p2.fingerprint()
        finally:
            store.close()

    def test_recording_twice_is_idempotent(self, batch, run, tmp_path):
        """Re-running the whole pipeline appends nothing the second time.

        This is what makes "just run it again" a safe response to a partial failure.
        """
        store = EventStore(tmp_path / "e2.db")
        try:
            record_run(store, batch, run)
            n1 = store.count()
            fp1 = rebuild(store).fingerprint()
            record_run(store, batch, run)
            assert store.count() == n1, "second run appended events"
            assert rebuild(store).fingerprint() == fp1
        finally:
            store.close()

    def test_incremental_catch_up_equals_full_rebuild(self, batch, run, tmp_path):
        """The classic CQRS bug: an incremental path that diverges from a rebuild.

        Invisible until someone compares, which is what this does.
        """
        store = EventStore(tmp_path / "e3.db")
        try:
            record_run(store, batch, run)
            full = rebuild(store)
            incremental = rebuild(store, after_seq=0)
            # Fold half, then catch up on the rest.
            half = store.count() // 2
            partial = rebuild(store)
            partial.__init__()  # reset to empty
            for ev in store.stream(limit=half):
                from app.projections import apply_event

                apply_event(partial, ev)
            catch_up(partial, store)
            assert partial.fingerprint() == full.fingerprint()
        finally:
            store.close()

    def test_ledger_balances(self, batch, run, tmp_path):
        store = EventStore(tmp_path / "e4.db")
        try:
            record_run(store, batch, run)
            p = rebuild(store)
            balanced, drift = p.ledger_balanced()
            assert balanced, f"ledger out by {drift} paise"
        finally:
            store.close()

    def test_audit_trail_contains_proposal_and_conclusion(self, batch, run, tmp_path):
        store = EventStore(tmp_path / "e5.db")
        try:
            record_run(store, batch, run)
            d = run.legs["gateway_bank"].decisions.auto_confirmed[0]
            txn = d.candidate.left_ids[0]
            trail = audit_trail(store, txn)
            types = {e["event_type"] for e in trail}
            assert "TransactionIngested" in types
            assert "MatchCandidateProposed" in types
            assert "MatchConfirmed" in types, (
                "a transaction's audit trail must include its own conclusion"
            )
        finally:
            store.close()

    def test_every_confirmed_match_has_a_recorded_rationale(self, batch, run, tmp_path):
        store = EventStore(tmp_path / "e6.db")
        try:
            record_run(store, batch, run)
            p = rebuild(store)
            for m in p.confirmed_matches():
                assert m.rationale, f"{m.match_key} confirmed with no rationale"
        finally:
            store.close()


class TestGeneratorReproducibility:
    def test_same_seed_same_fingerprint(self, tmp_path):
        from data_generator.generate_synthetic import fingerprint

        a = generate(seed=99, n_payments=120, days=10, out=tmp_path / "a")
        b = generate(seed=99, n_payments=120, days=10, out=tmp_path / "b")
        assert fingerprint(a) == fingerprint(b)

    def test_different_seed_different_data(self, tmp_path):
        from data_generator.generate_synthetic import fingerprint

        a = generate(seed=1, n_payments=120, days=10, out=tmp_path / "c")
        b = generate(seed=2, n_payments=120, days=10, out=tmp_path / "d")
        assert fingerprint(a) != fingerprint(b)

    def test_all_eleven_break_patterns_appear(self, dataset):
        """A generator that silently stops injecting a pattern makes the metric
        for that pattern meaningless while still printing a number."""
        from app.core.schema import ALL_BREAK_PATTERNS

        ds, _ = dataset
        fired = {rec["pattern"] for rec in ds.injection_log}
        missing = set(ALL_BREAK_PATTERNS) - fired
        assert not missing, f"patterns never ran: {missing}"

    def test_ground_truth_money_is_conserved(self, dataset):
        """Every 1:N link's parts must sum to the whole, exactly.

        Splitting a settlement into three credits that sum to something else would
        make the answer key itself wrong, and every metric computed against it.
        """
        ds, _ = dataset
        by_id = ds.by_id()
        for link in ds.links_for("gateway_bank"):
            left = sum(by_id[i].amount_paise for i in link.left_ids)
            right = sum(by_id[i].amount_paise for i in link.right_ids)
            assert left == right, f"{link.pattern} link does not conserve money"

    def test_bank_refs_are_unique(self, dataset):
        ds, _ = dataset
        ids = [t.external_id for t in ds.bank]
        assert len(ids) == len(set(ids))
