"""Calibrated confidence model.

Two public surfaces:

``DummyScorer``
    Heuristic-only, no training required. Used before the real model is fit, in
    tests, and as the naive baseline in the eval table.

``CalibratedScorer``
    A gradient-boosted classifier wrapped with isotonic-regression calibration.
    The calibration step is what earns the right to publish a precision/recall
    table: without it, the raw model's confidence in the ambiguous band is
    systematically over-stated, and a reliability diagram drawn from it would
    show a hockey-stick bend above 0.7 that makes every number above that line
    a lie.

Training contract
-----------------
Train on the seed-7 dataset, evaluate on the seed-42 dataset, and never touch
the eval set during training. This is the two-dataset discipline described in
``generate_synthetic.py``, and it is the only thing that separates "the numbers
look good in the notebook" from "the numbers mean something".

Threshold selection
-------------------
The threshold is *derived* rather than chosen. ``fit`` returns the lowest
probability at which the calibrated model's precision clears ``TARGET_PRECISION``
on a held-out validation slice of the training data. That number becomes the
auto-confirm threshold, and it is stored in the model artefact alongside the
feature schema version -- so the threshold published in the eval table came from
the same training run that produced the model file, not from iterating until the
eval numbers looked good.

What ``score_many`` returns is a calibrated probability, not a rank. When it says
0.82 it means "our best estimate is that 82% of matches we would confirm at this
score are correct". That claim is validated by the reliability diagram in
``eval/run_eval.py``, which is the right way to know whether it is true. The
validation is not optional: an overconfident score that looks honest because
nobody checked is exactly the failure mode a calibrated system is supposed to
prevent.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np

from app.core.config import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MODEL,
    TARGET_PRECISION,
)


class DummyScorer:
    """Heuristic score from the feature dict -- no training, no calibration.

    Serves as the Tier-1-and-2-without-ML baseline in the eval table, and as the
    fallback when the model file has not been built yet. The heuristic is
    deliberately crude so the calibrated model has something to beat, not so
    sophisticated that it blurs the comparison.

    The formula is a weighted average of the six features that were most
    predictive in the first training run, with the others zeroed. Keeping it
    transparent and reproducible from the feature names matters because the
    baseline is what every number in the paper is relative to.
    """

    name = "dummy_heuristic"
    trained = False
    threshold: float = 0.85

    def score_many(self, feature_dicts: list[dict[str, float]]) -> list[float]:
        return [self._score(f) for f in feature_dicts]

    def _score(self, f: dict[str, float]) -> float:
        s = (
            0.35 * f.get("utr_exact", 0.0)
            + 0.20 * f.get("utr_similarity", 0.0)
            + 0.15 * f.get("amount_exact", 0.0)
            + 0.15 * f.get("within_fee_band", 0.0)
            + 0.10 * (1.0 - f.get("date_delta_abs", 0.0))
            + 0.05 * f.get("narration_cosine", 0.0)
        )
        # Ambiguity discount: even a perfect-looking match means less when three
        # equally-perfect alternatives exist.
        amb_penalty = 1.0 - 0.1 * (
            f.get("left_ambiguity", 0.0) + f.get("right_ambiguity", 0.0)
        )
        return max(0.0, min(1.0, s * max(0.4, amb_penalty)))


class CalibratedScorer:
    """``sklearn.HistGradientBoostingClassifier`` + isotonic calibration.

    The model is a gradient-boosted tree. Isotonic regression is the calibration
    method. The choice is pragmatic: on datasets this size (a few thousand rows)
    isotonic beats Platt scaling on the reliability diagram because it is
    piecewise linear and can adapt to the kink in the raw model's output around
    0.7 without assuming any functional form.

    Feature schema version is stored and checked on load. A mismatch raises
    immediately: a silently wrong feature at column 7 would produce a model that
    looks fine until it doesn't, and "until it doesn't" in a finance context is
    a misposted journal entry nobody can trace.
    """

    def __init__(self) -> None:
        self._clf = None
        self._cal = None
        self.threshold: float = 0.90
        self.trained = False
        self.name = "calibrated_gbm"
        self._schema_version: str = FEATURE_SCHEMA_VERSION
        #: How the threshold was arrived at, and whether the precision target was
        #: actually met. Surfaced in the eval report -- a threshold that silently
        #: fell back to a default is the kind of thing that turns a published
        #: metric into a fiction.
        self.threshold_report: dict = {}

    def fit(self, X: list[dict[str, float]], y: list[int]) -> "CalibratedScorer":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        n = len(X)
        if n < 40:
            raise ValueError(
                f"training set too small ({n} rows); "
                "the calibration curve would be meaningless"
            )
        if len(set(y)) < 2:
            raise ValueError(
                "training set has a single class; a classifier fitted on it would "
                "be a constant and its calibration curve meaningless"
            )
        Xv = np.array([[f.get(name, 0.0) for name in FEATURE_NAMES] for f in X])

        # Three-way split: fit / calibrate / choose-threshold. The third slice is
        # the one people forget, and skipping it means the threshold is chosen on
        # data the calibrator already saw -- which biases it optimistically in
        # exactly the high-confidence region the auto-confirm rule depends on.
        cal_frac = MODEL.calibration_fraction
        thr_frac = MODEL.threshold_fraction
        X_trn, X_rest, y_trn, y_rest = train_test_split(
            Xv, y, test_size=cal_frac + thr_frac,
            random_state=MODEL.random_state, stratify=y,
        )
        X_cal, X_thr, y_cal, y_thr = train_test_split(
            X_rest, y_rest,
            test_size=thr_frac / (cal_frac + thr_frac),
            random_state=MODEL.random_state + 1, stratify=y_rest,
        )

        base = HistGradientBoostingClassifier(
            max_depth=MODEL.max_depth,
            max_iter=MODEL.max_iter,
            learning_rate=MODEL.learning_rate,
            l2_regularization=MODEL.l2_regularization,
            min_samples_leaf=MODEL.min_samples_leaf,
            random_state=MODEL.random_state,
        )
        base.fit(X_trn, y_trn)

        try:  # sklearn >= 1.6
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(
                FrozenEstimator(base), method=MODEL.calibration_method
            )
        except ImportError:  # pragma: no cover - older sklearn
            calibrated = CalibratedClassifierCV(
                base, method=MODEL.calibration_method, cv="prefit"
            )
        calibrated.fit(X_cal, y_cal)
        self._clf = calibrated

        self.threshold, self.threshold_report = self._choose_threshold(
            calibrated.predict_proba(X_thr)[:, 1], list(y_thr)
        )
        self.threshold_report["train_rows"] = len(X_trn)
        self.threshold_report["calibration_rows"] = len(X_cal)
        self.threshold_report["threshold_rows"] = len(X_thr)
        self.threshold_report["positive_rate"] = round(sum(y) / len(y), 4)
        self.trained = True
        return self

    @staticmethod
    def _choose_threshold(probs, labels: list[int]) -> tuple[float, dict]:
        """Lowest calibrated probability whose precision meets the target.

        Sweeping upward and stopping at the first threshold that clears the target
        takes the most permissive threshold we can justify, maximising recall while
        holding precision -- rather than the most conservative one we could get away
        with.

        Every candidate threshold is required to confirm a *meaningful* number of
        matches (``MIN_CONFIRMED``). Without that floor the sweep happily returns a
        threshold that confirms three matches at 100% precision, which satisfies the
        target arithmetically and is useless.

        If no threshold meets the target, this does **not** quietly return 1.0. It
        returns the best achievable threshold together with ``target_met: False``
        and the precision it actually reached, so the eval can print the shortfall.
        A system that silently disables its own automation when it cannot hit a
        quality bar looks, from the outside, exactly like a system that has no
        automation -- and nobody would know which they had.
        """
        import numpy as _np

        probs = _np.asarray(probs, dtype=float)
        labels_a = _np.asarray(labels, dtype=int)
        n_pos = int(labels_a.sum())
        min_confirmed = max(10, n_pos // 10)

        sweep = []
        for thr in _np.linspace(0.30, 0.995, 400):
            preds = probs >= thr
            confirmed = int(preds.sum())
            if confirmed < min_confirmed:
                continue
            precision = float((preds & (labels_a == 1)).sum()) / confirmed
            recall = float((preds & (labels_a == 1)).sum()) / max(1, n_pos)
            sweep.append((float(thr), precision, recall, confirmed))

        if not sweep:
            return 0.90, {
                "method": "fallback_no_viable_threshold",
                "target_met": False,
                "reason": f"no threshold confirmed at least {min_confirmed} matches",
                "chosen": 0.90,
            }

        for thr, precision, recall, confirmed in sweep:
            if precision >= TARGET_PRECISION:
                return thr, {
                    "method": "lowest_threshold_meeting_target_precision",
                    "target_precision": TARGET_PRECISION,
                    "target_met": True,
                    "chosen": round(thr, 4),
                    "precision_at_chosen": round(precision, 4),
                    "recall_at_chosen": round(recall, 4),
                    "confirmed_at_chosen": confirmed,
                    "min_confirmed_required": min_confirmed,
                }

        best = max(sweep, key=lambda s: (s[1], s[2]))
        return best[0], {
            "method": "best_available_target_not_met",
            "target_precision": TARGET_PRECISION,
            "target_met": False,
            "chosen": round(best[0], 4),
            "precision_at_chosen": round(best[1], 4),
            "recall_at_chosen": round(best[2], 4),
            "confirmed_at_chosen": best[3],
            "min_confirmed_required": min_confirmed,
            "note": "target precision unreachable on the threshold slice; "
                    "using the best available and reporting the shortfall",
        }

    def score_many(self, feature_dicts: list[dict[str, float]]) -> list[float]:
        if not self.trained or not feature_dicts:
            return DummyScorer().score_many(feature_dicts)
        Xv = np.array([[f.get(name, 0.0) for name in FEATURE_NAMES] for f in feature_dicts])
        return self._clf.predict_proba(Xv)[:, 1].tolist()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "clf": self._clf,
                    "threshold": self.threshold,
                    "threshold_report": self.threshold_report,
                    "schema_version": self._schema_version,
                    "feature_names": list(FEATURE_NAMES),
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "CalibratedScorer":
        with Path(path).open("rb") as fh:
            data = pickle.load(fh)
        if data.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"model artefact has feature schema {data.get('schema_version')!r} "
                f"but current code is {FEATURE_SCHEMA_VERSION!r}; "
                "retrain before running inference"
            )
        if data.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError(
                "model artefact feature names differ from current FEATURE_NAMES; "
                "retrain before running inference"
            )
        obj = cls()
        obj._clf = data["clf"]
        obj.threshold = data["threshold"]
        obj.threshold_report = data.get("threshold_report", {})
        obj.trained = True
        return obj


def build_training_data(
    candidates: list,       # list[Candidate]
    ground_truth_pairs: set[tuple[str, str]],
) -> tuple[list[dict[str, float]], list[int]]:
    """Convert a list of ``Candidate`` objects into (X, y) for the classifier.

    The ground truth is a set of ``(left_id, right_id)`` pairs. A candidate is
    a positive example if and only if *all* its pairs appear in the ground truth.
    A candidate that gets three out of four legs right is a wrong answer, not a
    partial credit, because confirming it posts a wrong journal entry.
    """
    X: list[dict[str, float]] = []
    y: list[int] = []
    for c in candidates:
        if not c.features:
            continue
        label = 1 if c.pair_keys().issubset(ground_truth_pairs) else 0
        X.append(c.features)
        y.append(label)
    return X, y


def reliability_diagram_points(
    probs: list[float], labels: list[int], n_bins: int = 10
) -> list[dict]:
    """Calibration curve data for the eval report.

    Computes mean predicted probability and actual positive rate per bin. A
    well-calibrated model's points sit on the diagonal. Any systematic departure
    -- especially the common "S-curve" where the model is over-confident in the
    high range -- is visible here and is the thing that justifies the calibration
    step rather than trusting the raw model output.
    """
    bins = [[] for _ in range(n_bins)]
    for p, l in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, l))

    points = []
    for i, b in enumerate(bins):
        if not b:
            continue
        mean_pred = sum(p for p, _ in b) / len(b)
        frac_pos = sum(l for _, l in b) / len(b)
        points.append(
            {
                "bin_center": (i + 0.5) / n_bins,
                "mean_predicted": round(mean_pred, 4),
                "fraction_positive": round(frac_pos, 4),
                "count": len(b),
            }
        )
    return points


def brier_score(probs: list[float], labels: list[int]) -> float:
    """Mean squared error between predicted probability and true label.

    0.0 is perfect, 0.25 is what you get from a constant-0.5 predictor. A
    calibrated model on real data tends to land around 0.05-0.12 depending on
    how much genuine ambiguity there is. This is the single number that
    summarises both discrimination and calibration in a way AUC-ROC does not,
    because AUC-ROC is insensitive to calibration error.
    """
    if not probs:
        return float("nan")
    return sum((p - l) ** 2 for p, l in zip(probs, labels)) / len(probs)


def expected_calibration_error(
    probs: list[float], labels: list[int], n_bins: int = 10
) -> float:
    """Weighted mean gap between confidence and accuracy across bins.

    The Brier score conflates being *uncertain* with being *miscalibrated*: a
    model that correctly reports 0.5 on a genuinely coin-flip case is penalised
    the same as one that reports 0.9 and is wrong half the time. ECE isolates the
    part that matters for a decision threshold -- when the system says 0.82, is it
    right 82% of the time -- which is exactly the claim the auto-confirm policy
    rests on.
    """
    if not probs:
        return float("nan")
    total = 0.0
    for pt in reliability_diagram_points(probs, labels, n_bins):
        weight = pt["count"] / len(probs)
        total += weight * abs(pt["mean_predicted"] - pt["fraction_positive"])
    return total


def calibration_by_slice(
    probs: list[float], labels: list[int], slices: list[str], n_bins: int = 10
) -> dict[str, dict]:
    """Per-subgroup calibration, because aggregate calibration hides subgroups.

    The problem statement predicts this failure and it is worth taking seriously:
    a model can be beautifully calibrated overall while being badly over-confident
    on one break pattern, because the well-behaved majority averages the error
    away. Reporting ECE per break pattern is how that gets caught -- and it is the
    difference between "our confidence is trustworthy" and "our confidence is
    trustworthy on the cases that were already easy".
    """
    groups: dict[str, tuple[list[float], list[int]]] = {}
    for p, l, s in zip(probs, labels, slices):
        g = groups.setdefault(s, ([], []))
        g[0].append(p)
        g[1].append(l)

    out: dict[str, dict] = {}
    for name, (ps, ls) in sorted(groups.items()):
        if len(ps) < 5:
            continue
        out[name] = {
            "n": len(ps),
            "positives": sum(ls),
            "mean_confidence": round(sum(ps) / len(ps), 4),
            "accuracy": round(sum(ls) / len(ls), 4),
            "brier": round(brier_score(ps, ls), 4),
            "ece": round(expected_calibration_error(ps, ls, n_bins), 4),
        }
    return out
