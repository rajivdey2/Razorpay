"""Feature extraction for a candidate correspondence.

Eighteen numbers, one per entry in ``config.FEATURE_NAMES``, computed identically
at training time and at inference time -- one function, no parallel
implementations. Feature skew between train and serve is the classic silent ML
failure, and in a money system it manifests as a calibration curve that looked
honest in the eval and is over-confident in production.

Three of these features deserve their own note, because they are what make the
confidence score *mean* something rather than just correlate with correctness:

``utr_similarity``
    Confusion-aware edit distance. Substituting ``0`` for ``O`` costs 0.35 of an
    edit rather than a full one, because a keyed/OCR'd digit-letter swap is
    overwhelmingly more likely than a genuinely different reference that happens
    to differ in exactly that way. Truncation is handled by a separate
    common-prefix term, since a bank cutting a narration at 40 characters
    preserves the prefix perfectly and edit distance punishes the missing tail as
    though it were corruption.

``left_ambiguity`` / ``right_ambiguity``
    How many *other* plausible partners each side has. Without these, two
    identical candidates both score ~0.95 and the engine confidently picks the
    wrong one -- the exact failure the problem statement warns about in its
    postmortem section. With them, the model learns that a perfect-looking match
    in a crowded window is worth less than the same match in an empty one, which
    is the whole reason a *calibrated* score beats a similarity score.

``within_fee_band``
    A structural test, not a threshold on noise. It asks whether the residual is
    the size fees, GST-on-fee and withholding could plausibly explain given the
    gross, which is a different question from "is the residual small".
"""

from __future__ import annotations

import math
from datetime import date

from app.core.config import FEATURE_NAMES, LegConfig
from app.core.schema import NormalizedTxn

#: Character pairs a human or an OCR pass confuses. Cost below a full edit,
#: never zero -- treating them as equal would manufacture false identities.
_CONFUSABLE = {
    frozenset("O0"), frozenset("I1"), frozenset("Il"), frozenset("S5"),
    frozenset("B8"), frozenset("Z2"), frozenset("G6"), frozenset("l1"),
}
_CONFUSION_COST = 0.35


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _CONFUSION_COST if frozenset((a, b)) in _CONFUSABLE else 1.0


def weighted_levenshtein(a: str, b: str) -> float:
    """Edit distance with a reduced cost for confusable substitutions.

    Two rows of a DP table rather than the full matrix: reference strings are
    short, but this runs over every candidate edge in every window, and the
    allocation churn of a full matrix showed up in the profile.
    """
    if a == b:
        return 0.0
    if not a:
        return float(len(b))
    if not b:
        return float(len(a))
    prev = [float(i) for i in range(len(b) + 1)]
    for i, ca in enumerate(a, start=1):
        cur = [float(i)] + [0.0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1.0,               # deletion
                cur[j - 1] + 1.0,            # insertion
                prev[j - 1] + _sub_cost(ca, cb),
            )
        prev = cur
    return prev[-1]


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if _sub_cost(ca, cb) > _CONFUSION_COST:
            break
        n += 1
    return n


def reference_similarity(left_refs: tuple[str, ...], right_refs: tuple[str, ...]) -> tuple[float, str, str]:
    """Best similarity over the cross product of candidate references.

    Returns ``(similarity, left_ref, right_ref)`` so the workbench can show a
    human *which* two strings the score came from. A bare number is not evidence;
    "HDFC260304 vs HDFC260304560105, truncated, 0.79" is.
    """
    best = (0.0, "", "")
    for lr in left_refs:
        if not lr:
            continue
        for rr in right_refs:
            if not rr:
                continue
            longest = max(len(lr), len(rr))
            edit_sim = 1.0 - weighted_levenshtein(lr, rr) / longest
            # Truncation-tolerant: a shared prefix as long as the shorter string
            # is strong evidence even when the tail is missing entirely.
            pref = _common_prefix(lr, rr)
            prefix_sim = pref / min(len(lr), len(rr)) if min(len(lr), len(rr)) else 0.0
            # Discount a prefix match by how much of the longer string is
            # unexplained, so a 4-character overlap on two 16-char refs does not
            # read as a near-identity.
            prefix_sim *= min(len(lr), len(rr)) / longest
            sim = max(edit_sim, prefix_sim)
            if sim > best[0]:
                best = (sim, lr, rr)
    return best


def char_ngram_cosine(a: str | None, b: str | None, n: int = 3) -> float:
    """Cosine over character n-gram counts.

    Character n-grams rather than word TF-IDF because bank narrations are not
    prose: ``IMPS/P2A/KKBK2603O3729207/RAZORPAY/SETTL`` tokenises into garbage
    under a word tokeniser, while its trigrams still overlap heavily with the
    reference it contains. No corpus-wide IDF is fitted, deliberately -- an IDF
    table is state that has to be versioned alongside the model, and it bought
    nothing measurable here.
    """
    if not a or not b:
        return 0.0
    a, b = a.upper(), b.upper()
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    va: dict[str, int] = {}
    vb: dict[str, int] = {}
    for i in range(len(a) - n + 1):
        va[a[i : i + n]] = va.get(a[i : i + n], 0) + 1
    for i in range(len(b) - n + 1):
        vb[b[i : i + n]] = vb.get(b[i : i + n], 0) + 1
    dot = sum(v * vb.get(k, 0) for k, v in va.items())
    if not dot:
        return 0.0
    sa = sum(v * v for v in va.values())
    sb = sum(v * v for v in vb.values())
    # One sqrt of the product, not the product of two sqrts. For identical vectors
    # sa == sb == dot, so sqrt(sa * sb) is exactly sa and the ratio is exactly 1.0;
    # sqrt(sa) * sqrt(sb) lands a few ulps off and a self-comparison comes back as
    # 0.9999999999999998. Every consumer treats this as a bounded feature, and a
    # value that is *almost* 1 for an exact match is a bad thing to debug inside a
    # model artefact.
    return min(1.0, max(0.0, dot / math.sqrt(sa * sb)))


def _logit(p: float, floor: float = 1e-4) -> float:
    p = min(max(p, floor), 1.0 - floor)
    return math.log(p / (1.0 - p))


def _repr_dates(left: list[NormalizedTxn], right: list[NormalizedTxn]) -> tuple[date, date]:
    """One date per side, for a group that may span several.

    Left takes its *latest* date and right its *earliest*: the question a
    settlement cycle answers is "how long after the last constituent event did
    the money show up", and taking means would let one old invoice in a batch
    drag the delta into a region the prior calls implausible.
    """
    return max(t.value_date for t in left), min(t.value_date for t in right)


def extract(
    left: list[NormalizedTxn],
    right: list[NormalizedTxn],
    leg: LegConfig,
    *,
    left_ambiguity: int = 1,
    right_ambiguity: int = 1,
    amount_peers: int = 1,
    gross_paise: int | None = None,
) -> dict[str, float]:
    """Feature vector for the hypothesis "these left items == these right items".

    ``left_ambiguity`` / ``right_ambiguity`` / ``amount_peers`` come from the
    blocking graph, not from the pair itself: they are properties of the
    neighbourhood, and computing them here would mean re-scanning the window per
    edge.
    """
    la = sum(t.amount_paise for t in left)
    ra = sum(t.amount_paise for t in right)
    scale = max(abs(la), abs(ra), 1)
    residual = ra - la

    ldate, rdate = _repr_dates(left, right)
    delta_days = (rdate - ldate).days

    left_refs: tuple[str, ...] = tuple(
        r for t in left for r in (t.ref_candidates or ((t.utr,) if t.utr else ()))
    )
    right_refs: tuple[str, ...] = tuple(
        r for t in right for r in (t.ref_candidates or ((t.utr,) if t.utr else ()))
    )
    sim, _, _ = reference_similarity(left_refs, right_refs)
    exact = 1.0 if (set(left_refs) & set(right_refs)) else 0.0

    right_text = " ".join(filter(None, (t.narration for t in right)))
    left_text = " ".join(filter(None, (t.narration for t in left)))
    in_narration = 0.0
    for lr in left_refs:
        if lr and len(lr) >= 8 and lr in right_text.upper():
            in_narration = 1.0
            break

    # "Explainable" is measured against gross where known, because fees and
    # withholding are percentages of gross, not of the net that reached the bank.
    basis = abs(gross_paise) if gross_paise else scale
    band = max(
        leg.explainable_floor_paise,
        (basis * leg.explainable_bps) // 10_000,
    )

    prior = leg.cycle_prior.get(delta_days, 0.0025)

    feats = {
        "amount_rel_delta": min(abs(residual) / scale, 1.0),
        "amount_abs_delta_log": math.log1p(abs(residual)) / 20.0,
        "amount_exact": 1.0 if residual == 0 else 0.0,
        "within_fee_band": 1.0 if abs(residual) <= band else 0.0,
        "date_delta_days": max(-30.0, min(30.0, float(delta_days))),
        "date_delta_abs": min(abs(delta_days), 30) / 30.0,
        "cycle_prior_logit": _logit(prior) / 10.0,
        "utr_exact": exact,
        "utr_similarity": sim,
        "utr_in_narration": in_narration,
        "narration_cosine": char_ngram_cosine(
            " ".join(filter(None, (left_text, *left_refs))),
            " ".join(filter(None, (right_text, *right_refs))),
        ),
        "currency_match": 1.0 if len({t.currency for t in (*left, *right)}) == 1 else 0.0,
        "subset_size": min(len(left) + len(right), 24) / 24.0,
        "is_subset": 1.0 if (len(left) > 1 or len(right) > 1) else 0.0,
        # Ambiguity enters as a log so the difference between 1 and 2 competitors
        # matters far more than between 9 and 10 -- which is how it behaves in
        # practice.
        "left_ambiguity": math.log1p(max(0, left_ambiguity - 1)) / 3.0,
        "right_ambiguity": math.log1p(max(0, right_ambiguity - 1)) / 3.0,
        "amount_uniqueness": 1.0 / max(1, amount_peers),
        "left_amount_log": math.log1p(abs(la)) / 20.0,
    }

    missing = set(FEATURE_NAMES) - set(feats)
    extra = set(feats) - set(FEATURE_NAMES)
    if missing or extra:
        raise AssertionError(
            f"feature vector does not match FEATURE_NAMES (missing={missing}, extra={extra})"
        )
    return feats


def to_vector(feats: dict[str, float]) -> list[float]:
    """Dict to ordered list. The single place feature ordering is decided."""
    return [float(feats[name]) for name in FEATURE_NAMES]
