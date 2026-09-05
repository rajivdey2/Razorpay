"""Money arithmetic. The foundation everything else assumes is correct.

These tests exist because a single float in a currency path produces errors that
are individually invisible and collectively a restatement. The property tests are
the important ones: they assert the *laws* money has to obey rather than a handful
of examples someone thought of.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.core.money import (
    MoneyTypeError,
    apply_bps,
    check_paise,
    from_rupee_string,
    indian_grouping,
    inr,
    rupees,
)

paise = st.integers(min_value=-10**13, max_value=10**13)


class TestNoFloatMoney:
    def test_float_amount_is_rejected(self):
        with pytest.raises(MoneyTypeError):
            check_paise(12.5)

    def test_bool_is_rejected(self):
        """``True`` is an ``int`` subclass worth exactly 1 paise.

        A stray flag flowing into an amount field would be a silent 1-paise
        movement, which is precisely the kind of error that survives every review
        because it never looks wrong.
        """
        with pytest.raises(MoneyTypeError):
            check_paise(True)

    def test_string_is_rejected(self):
        with pytest.raises(MoneyTypeError):
            check_paise("100")


class TestParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1,23,456.78", 12345678),      # Indian lakh grouping
            ("123,456.78", 12345678),       # Western grouping, same value
            ("1234.5", 123450),             # one decimal place
            ("1234", 123400),               # no decimal point
            ("(1234.00)", -123400),         # accounting negative
            ("1,234.00 CR", 123400),
            ("1,234.00 DR", -123400),
            ("-1,234.00", -123400),
            ("0.01", 1),
            ("₹ 5,000.00", 500000),
        ],
    )
    def test_bank_statement_shapes(self, text, expected):
        assert from_rupee_string(text) == expected

    def test_the_float_trap(self):
        """Find a two-decimal amount where the naive float conversion is wrong.

        Searched rather than hard-coded: which specific values misbehave depends on
        binary rounding, so asserting one magic number makes the test a statement
        about this machine. Asserting that witnesses *exist* -- and that the exact
        parser gets every one of them right -- is the claim that actually matters.

        Each witness is a real amount a bank could print, off by one paise in the
        direction that accumulates.
        """
        witnesses = []
        for cents in range(0, 200_000):
            text = f"{cents // 100}.{cents % 100:02d}"
            if int(float(text) * 100) != cents:
                witnesses.append(text)
                if len(witnesses) >= 5:
                    break
        assert witnesses, "no float witnesses found -- did the platform change?"
        for text in witnesses:
            exact = int(text.replace(".", ""))
            assert int(float(text) * 100) != exact          # the trap
            assert from_rupee_string(text) == exact         # the parser

    def test_sub_paise_is_refused_rather_than_rounded(self):
        """``8.615`` has no exact paise value, so the parser refuses it.

        Rounding would be defensible for a report and indefensible here: a bank
        feed cannot contain sub-paise, so an input that does means the field was
        misparsed, and rounding turns a format bug into a rounding difference
        nobody can trace.
        """
        with pytest.raises(ValueError, match="sub-paise"):
            from_rupee_string("8.615")

    def test_sub_paise_is_refused_not_rounded(self):
        with pytest.raises(ValueError, match="sub-paise"):
            from_rupee_string("100.123")

    def test_trailing_zeros_beyond_paise_are_fine(self):
        assert from_rupee_string("100.1200") == 10012

    @pytest.mark.parametrize("bad", ["", "abc", "1.2.3", "--5"])
    def test_garbage_raises(self, bad):
        with pytest.raises(ValueError):
            from_rupee_string(bad)


class TestProperties:
    @given(paise)
    def test_format_parse_round_trip(self, p):
        """Formatting then parsing must return the identical integer.

        The single most important property in the file. If it ever fails, some
        amount in the system can be displayed and re-read as a different number.
        """
        assert from_rupee_string(rupees(p)) == p

    @given(paise)
    def test_inr_round_trips_too(self, p):
        assert from_rupee_string(inr(p)) == p

    @given(st.lists(paise, min_size=1, max_size=60))
    def test_summation_is_exact_and_order_independent(self, xs):
        """Integer money sums identically regardless of order. Floats do not.

        Reconciliation is built on comparing sums computed in different orders --
        a settlement total against invoices in one order, bank credits in another.
        With floats those two totals can differ, and the difference looks exactly
        like a real break.
        """
        assert sum(xs) == sum(reversed(xs))
        assert sum(sorted(xs)) == sum(xs)

    @given(st.integers(min_value=0, max_value=10**11), st.integers(0, 1000))
    def test_apply_bps_is_monotone(self, amount, bps):
        assert apply_bps(amount, bps) <= apply_bps(amount, bps + 1)

    @given(st.integers(min_value=0, max_value=10**11))
    def test_apply_bps_zero_and_full(self, amount):
        assert apply_bps(amount, 0) == 0
        assert apply_bps(amount, 10_000) == amount

    @given(st.integers(min_value=0, max_value=10**12))
    def test_indian_grouping_preserves_digits(self, n):
        assert indian_grouping(n).replace(",", "") == str(n)

    @pytest.mark.parametrize(
        "n,expected",
        [(1, "1"), (999, "999"), (1000, "1,000"), (99999, "99,999"),
         (100000, "1,00,000"), (10000000, "1,00,00,000")],
    )
    def test_indian_grouping_examples(self, n, expected):
        assert indian_grouping(n) == expected
