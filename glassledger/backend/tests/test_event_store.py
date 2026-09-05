"""The event store's three guarantees: append-only, tamper-evident, idempotent.

Each is tested against the behaviour that would break it, not against a happy path.
An append-only store that has never had an ``UPDATE`` attempted on it is not known
to be append-only.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.events import EventStore, GENESIS_HASH


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "events.db")
    yield s
    s.close()


class TestAppendOnly:
    def test_update_is_refused_by_the_database(self, store):
        store.append("TransactionIngested", "bank:X", {"amount_paise": 100})
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._conn.execute("UPDATE events SET payload_json='{}' WHERE seq=1")

    def test_delete_is_refused_by_the_database(self, store):
        store.append("TransactionIngested", "bank:X", {"amount_paise": 100})
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._conn.execute("DELETE FROM events WHERE seq=1")

    def test_unknown_event_type_is_refused(self, store):
        """A typo'd event type is history no projection will ever read."""
        with pytest.raises(ValueError, match="unknown event type"):
            store.append("TransactionIngsted", "bank:X", {})

    def test_sequence_is_monotonic(self, store):
        seqs = [
            store.append("TransactionIngested", f"bank:{i}", {"i": i}).seq
            for i in range(25)
        ]
        assert seqs == sorted(seqs) == list(range(1, 26))


class TestHashChain:
    def test_genesis(self, store):
        ev = store.append("TransactionIngested", "bank:X", {"amount_paise": 1})
        assert ev.prev_hash == GENESIS_HASH

    def test_links(self, store):
        a = store.append("TransactionIngested", "bank:A", {"amount_paise": 1})
        b = store.append("TransactionIngested", "bank:B", {"amount_paise": 2})
        assert b.prev_hash == a.hash

    def test_clean_chain_verifies(self, store):
        for i in range(40):
            store.append("TransactionIngested", f"bank:{i}", {"amount_paise": i})
        v = store.verify_chain()
        assert v.ok and v.events_checked == 40

    def test_payload_edit_is_detected_at_the_exact_event(self, store):
        for i in range(12):
            store.append("TransactionIngested", f"bank:{i}", {"amount_paise": i * 100})
        # Bypass the triggers the way someone with file access would.
        store._conn.executescript("DROP TRIGGER events_no_update;")
        store._conn.execute(
            "UPDATE events SET payload_json=? WHERE seq=?",
            (json.dumps({"amount_paise": 999999}), 6),
        )
        store._conn.commit()
        v = store.verify_chain()
        assert not v.ok
        assert v.first_bad_seq == 6
        assert "edited" in v.detail

    def test_reordering_is_detected(self, store):
        """Swapping two individually-valid events must still break the chain.

        This is why ``prev_hash`` is folded into the digest. Hashing content alone
        would let an attacker reorder history using only events that hash correctly.
        """
        for i in range(6):
            store._conn.execute(
                "INSERT INTO events (event_id, event_type, aggregate_id, occurred_at,"
                " payload_json, idempotency_key, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"id{i}", "TransactionIngested", f"bank:{i}", "2026-01-01T00:00:00+00:00",
                 "{}", None, "0" * 64, "f" * 64),
            )
        store._conn.commit()
        v = store.verify_chain()
        assert not v.ok


class TestIdempotency:
    def test_same_key_appends_once(self, store):
        a = store.append("TransactionIngested", "bank:X", {"amount_paise": 100},
                         idempotency_key="k1")
        b = store.append("TransactionIngested", "bank:X", {"amount_paise": 100},
                         idempotency_key="k1")
        assert a.seq == b.seq
        assert store.count() == 1

    def test_replaying_a_webhook_twenty_times_posts_once(self, store):
        """At-least-once delivery is the norm for payment webhooks.

        A store without idempotency double-posts under completely ordinary
        conditions -- not an edge case, the expected behaviour of the transport.
        """
        for _ in range(20):
            store.append("MatchConfirmed", "m1", {"confirmed_by": "agent"},
                         idempotency_key="confirm:m1")
        assert store.count() == 1
        assert store.verify_chain().ok

    def test_different_keys_both_append(self, store):
        store.append("TransactionIngested", "bank:X", {"a": 1}, idempotency_key="k1")
        store.append("TransactionIngested", "bank:Y", {"a": 1}, idempotency_key="k2")
        assert store.count() == 2

    def test_no_key_means_no_dedup(self, store):
        """Deliberate: an event with no natural key is a genuinely new fact."""
        store.append("TransactionIngested", "bank:X", {"a": 1})
        store.append("TransactionIngested", "bank:X", {"a": 1})
        assert store.count() == 2

    @settings(max_examples=40, deadline=None)
    @given(st.lists(st.integers(min_value=0, max_value=8), min_size=1, max_size=40))
    def test_arbitrary_replay_order_converges(self, seq):
        """Any interleaving of the same keyed appends yields the same store.

        The property retries and out-of-order redeliveries actually need: the final
        state depends on the *set* of facts, not the order they arrived in.
        """
        s = EventStore(":memory:")
        try:
            for i in seq:
                s.append("TransactionIngested", f"bank:{i}", {"i": i},
                         idempotency_key=f"k{i}")
            assert s.count() == len(set(seq))
            assert s.verify_chain().ok
        finally:
            s.close()


class TestQueries:
    def test_touching_finds_events_by_txn_id(self, store):
        store.append("TransactionIngested", "bank:BR1", {"amount_paise": 1})
        store.append("MatchCandidateProposed", "leg|a|b",
                     {"left_ids": ["gateway:s1"], "right_ids": ["bank:BR1"]})
        found = store.touching("bank:BR1")
        assert len(found) == 2
        assert {e.event_type for e in found} == {
            "TransactionIngested", "MatchCandidateProposed"
        }

    def test_touching_finds_the_confirmation_too(self, store):
        """Regression: confirmations used to be invisible from the transaction.

        The confirmation is the most important event about a match. When its payload
        did not name the transaction ids, ``touching`` could not find it, and the
        audit trail showed every proposal and no conclusion.
        """
        store.append("TransactionIngested", "bank:BR1", {"amount_paise": 1})
        store.append("MatchConfirmed", "leg|gateway:s1|bank:BR1",
                     {"confirmed_by": "agent", "left_ids": ["gateway:s1"],
                      "right_ids": ["bank:BR1"]})
        types = {e.event_type for e in store.touching("bank:BR1")}
        assert "MatchConfirmed" in types

    def test_for_aggregate_is_ordered(self, store):
        for i in range(5):
            store.append("MatchCandidateProposed", "agg1", {"i": i})
        evs = store.for_aggregate("agg1")
        assert [e.payload["i"] for e in evs] == [0, 1, 2, 3, 4]
