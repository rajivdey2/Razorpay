"""Pulling references out of bank narration text.

There is no UTR column. There is a 40-to-65-character string a bank wrote for a
human, and somewhere in it -- maybe -- is the reference that ties this credit to a
settlement. Examples from the emitted statements::

    RAZORPAY SETTLEMENT KKBK260302085514
    IMPS/P2A/KKBK2603O3729207/RAZORPAY/SETTL     <- keyed 0 as O
    NEFT-HDFC260304-RAZORPAY SOFTWARE PVT        <- truncated to 10 chars
    BY TRANSFER-NEFT*UTIB2603020084*RZP
    REV/RETURN NEFT SBIN260305773821 BENEFICIARY

Two decisions here are load-bearing.

**Return candidates, not a winner.** ``extract_references`` returns every token
that could plausibly be a reference, ranked. A single-value ``utr`` field forces
the parser to guess, and when it guesses wrong the matcher has no way to recover
because the alternative is already discarded. Handing the matcher a ranked list
lets the *scoring* layer decide, with the amount and date evidence in hand, which
candidate is real -- and lets it score a partial match on the second-best
candidate rather than scoring zero.

**Never repair, only report.** It is tempting to un-garble ``KKBK2603O3729207``
back to ``...03729207`` by reversing the O/0 confusion. Do not: the same
transformation applied to a genuinely different reference manufactures a false
match at maximum confidence, which is the single worst outcome available to a
reconciliation system. The fuzzy comparison in ``matching.features`` treats
confusable characters as a near-miss rather than an equality, so the evidence
degrades smoothly instead of a repair step lying about certainty.
"""

from __future__ import annotations

import re

#: A UTR-shaped token: bank-ish alpha prefix then mostly digits. Matches the real
#: 16-char NEFT/RTGS shape and its truncations.
_UTR_LIKE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,6}[0-9][A-Z0-9]{4,16})(?![A-Z0-9])")

#: Any longish alphanumeric run -- a fallback for references that do not follow
#: the bank-prefix convention at all.
_ALNUM_RUN = re.compile(r"(?<![A-Z0-9])([A-Z0-9]{8,24})(?![A-Z0-9])")

#: Tokens that look reference-shaped but never are. Kept short and specific: an
#: over-eager stopword list silently drops real references, and a reference lost
#: at ingestion is invisible for the rest of the pipeline.
_STOPWORDS = frozenset(
    {
        "RAZORPAY", "SETTLEMENT", "BENEFICIARY", "TRANSFER", "CUSTOMER",
        "DIRECTORS", "REFUND", "DEPOSIT", "SOFTWARE", "RESETTLEMENT",
    }
)


def _digit_ratio(token: str) -> float:
    if not token:
        return 0.0
    return sum(c.isdigit() for c in token) / len(token)


def score_candidate(token: str) -> float:
    """How reference-like a token is, in [0, 1]. Used only for ranking.

    Deliberately crude. This ranking picks which candidate lands in the
    human-facing ``utr`` field; every candidate survives into
    ``ref_candidates`` regardless, so a mis-ranking costs readability, not
    recall.
    """
    if token in _STOPWORDS or len(token) < 8:
        return 0.0
    d = _digit_ratio(token)
    if d < 0.25:
        return 0.0
    score = 0.35 + 0.35 * min(d / 0.75, 1.0)
    # A 2-6 letter prefix followed by digits is the shape UTRs actually take.
    if re.match(r"^[A-Z]{2,6}[0-9]", token):
        score += 0.20
    # 16 characters is the NEFT UTR length; being close to it is evidence.
    score += 0.10 * max(0.0, 1.0 - abs(len(token) - 16) / 10.0)
    return min(1.0, score)


def extract_references(text: str | None) -> tuple[str, ...]:
    """Ranked reference candidates from a narration string.

    Splits on the separators banks actually use (``/ - * _ space .``) as well as
    running the UTR-shaped regex, because ``NEFT-HDFC260304560105-RAZORPAY``
    hides its reference between hyphens while ``TRF FROM RAZORPAY REF X CR``
    delimits with spaces.
    """
    if not text:
        return ()
    up = text.upper()
    found: dict[str, float] = {}

    for match in _UTR_LIKE.finditer(up):
        tok = match.group(1)
        found[tok] = max(found.get(tok, 0.0), score_candidate(tok))
    for match in _ALNUM_RUN.finditer(up):
        tok = match.group(1)
        found[tok] = max(found.get(tok, 0.0), score_candidate(tok))
    # Separator-delimited fields the regexes may have merged with neighbours.
    for tok in re.split(r"[^A-Z0-9]+", up):
        if tok:
            s = score_candidate(tok)
            if s > 0:
                found[tok] = max(found.get(tok, 0.0), s)

    ranked = sorted(
        (t for t, s in found.items() if s > 0),
        key=lambda t: (-found[t], -len(t), t),
    )
    return tuple(ranked[:6])


def best_reference(text: str | None) -> str | None:
    refs = extract_references(text)
    return refs[0] if refs else None
