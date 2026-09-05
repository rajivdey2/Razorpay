"""Money primitives.

One rule, enforced here rather than remembered everywhere: currency amounts are
Python ``int`` in the smallest unit (paise for INR). No float ever touches a
money value, because float arithmetic is not associative and reconciliation is
built entirely on summing things in different orders and asking whether the
totals agree.

    >>> 0.1 + 0.2 == 0.3
    False

That single line is why a reconciliation engine that stores money as float will
eventually report a break that does not exist -- or worse, silently net two
breaks against each other.

Where a ratio is genuinely needed (an MDR percentage, an FX rate) we keep it in
basis points / micro-units as ``int`` and do the division last, with explicit
banker-free floor/round semantics stated at the call site.
"""

from __future__ import annotations

from typing import Final

PAISE_PER_RUPEE: Final[int] = 100
BPS_DENOMINATOR: Final[int] = 10_000  # 1 bp = 0.01%


class MoneyTypeError(TypeError):
    """Raised when a non-integer sneaks into a money field."""


def check_paise(value: object, field: str = "amount_paise") -> int:
    """Assert *value* is a plain ``int`` number of paise and return it.

    ``bool`` is rejected explicitly: it is an ``int`` subclass, and a stray
    ``True`` flowing into an amount field would be worth exactly 1 paise while
    reading like a flag.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise MoneyTypeError(
            f"{field} must be int paise, got {type(value).__name__}({value!r}). "
            "Money is never a float in this system."
        )
    return value


def indian_grouping(whole: int) -> str:
    """Digit grouping the Indian way: ``12345678 -> '1,23,45,678'``.

    Last three digits, then pairs. This is the convention every Indian bank
    statement, invoice and accounting package uses, and getting it wrong is a
    small tell that the system was built for somewhere else. ``f"{n:,}"`` gives
    Western thousands grouping, so it cannot be used.
    """
    s = str(abs(whole))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def rupees(paise: int) -> str:
    """Format paise for humans: ``12345678 -> '1,23,456.78'``.

    Presentation only. Never parse this back into a number.
    """
    check_paise(paise)
    sign = "-" if paise < 0 else ""
    whole, frac = divmod(abs(paise), PAISE_PER_RUPEE)
    return f"{sign}{indian_grouping(whole)}.{frac:02d}"


def inr(paise: int) -> str:
    """``-₹0.01``, not ``₹-0.01``. The sign precedes the currency mark."""
    body = rupees(paise)
    return ("-₹" + body[1:]) if body.startswith("-") else ("₹" + body)


def from_rupee_string(text: str) -> int:
    """Parse a bank-statement amount string into paise, exactly.

    Handles the shapes real Indian bank exports actually emit::

        '1,23,456.78'   lakh-grouped
        '1234.5'        one decimal place
        '1234'          no decimal point
        '(1234.00)'     accounting negative
        '1234.00 CR'    trailing indicator
        '-1,234.00'

    Deliberately string-based. ``int(round(float(text) * 100))`` is the obvious
    one-liner and it is wrong: ``float('8.615') * 100 == 861.4999999999999``,
    which rounds to 861 instead of 862 and puts a 1-paise phantom break into the
    ledger. We split on the decimal point and pad instead, so the conversion is
    exact for every input a bank can produce.
    """
    s = text.strip().upper()
    if not s:
        raise ValueError("empty amount string")

    # Currency marks come off FIRST. Doing the sign test before stripping "₹"
    # meant "₹-0.01" parsed as positive: the leading "₹" hid the minus from
    # ``startswith('-')``, and the later ``replace`` removed the symbol without
    # ever reconsidering the sign. A silently sign-flipped refund is about the
    # worst single-character bug available in this domain, and the round-trip
    # property test is what caught it.
    for mark in ("INR", "RS.", "RS", "₹"):
        s = s.replace(mark, "")
    s = s.strip()
    if not s:
        raise ValueError(f"no digits in amount string: {text!r}")

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    for tag in (" CR", "CR", " DR", "DR"):
        if s.endswith(tag):
            if tag.strip() == "DR":
                negative = True
            s = s[: -len(tag)].strip()
            break
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()
    if s.startswith("+"):
        s = s[1:].strip()

    s = s.replace(",", "").strip()
    if not s:
        raise ValueError(f"no digits in amount string: {text!r}")

    if "." in s:
        whole, _, frac = s.partition(".")
        if "." in frac:
            raise ValueError(f"multiple decimal points: {text!r}")
    else:
        whole, frac = s, ""

    whole = whole or "0"
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise ValueError(f"non-numeric amount string: {text!r}")

    if len(frac) > 2:
        # Sub-paise precision cannot exist in a bank feed; refuse rather than
        # round, so a format misparse surfaces loudly instead of as a rounding
        # difference nobody can trace six months later.
        if set(frac[2:]) != {"0"}:
            raise ValueError(f"sub-paise precision in amount: {text!r}")
        frac = frac[:2]

    paise = int(whole) * PAISE_PER_RUPEE + int(frac.ljust(2, "0"))
    return -paise if negative else paise


def apply_bps(paise: int, bps: int) -> int:
    """Percentage of an amount, in paise, rounded half-up.

    Used for MDR and GST-on-fee. Half-up (not Python's banker's rounding) is
    what payment processors bill with, so matching their arithmetic matters more
    than statistical neutrality here.
    """
    check_paise(paise)
    numerator = paise * bps
    # Half-up on a positive denominator.
    return (numerator + BPS_DENOMINATOR // 2) // BPS_DENOMINATOR
