"""Matching engine: tiered reconciliation with calibrated confidence.

    from app.matching import ReconciliationEngine, CalibratedScorer
    engine = ReconciliationEngine(CalibratedScorer.load(path))
    run = engine.run(batch)

Layer map:

    features.py    18 features per hypothesis, one implementation for train + serve
    blocking.py    candidate generation; the reason this terminates
    subsetsum.py   bounded exhaustive N:1 search with admissible pruning
    tier1.py       deterministic rules: wash pairs, exact refs, order-key joins
    tier2.py       Hungarian assignment + weighted set packing
    tier3.py       bounded LLM arbitration, proposer-only, never above the gate
    confidence.py  gradient-boosted classifier + isotonic calibration
    residuals.py   is this residual explainable by fees/withholding/FX?
    policy.py      the four decision rules, including the materiality gate
    exceptions.py  the ranked, evidence-backed list of what is left
    engine.py      per-leg orchestration
"""

from __future__ import annotations

from .blocking import BlockingGraph, build_graph, build_subset_pools
from .confidence import (
    CalibratedScorer,
    DummyScorer,
    brier_score,
    build_training_data,
    calibration_by_slice,
    expected_calibration_error,
    reliability_diagram_points,
)
from .engine import LegRun, ReconciliationEngine, ReconciliationRun
from .exceptions import ExceptionReport, build_exceptions
from .policy import Policy, PolicyResult, is_auto_confirmable
from .residuals import ResidualAttribution
from .subsetsum import find_subsets, find_subsets_exact_first
from .types import Candidate, Decision, Exception_

__all__ = [
    "BlockingGraph", "build_graph", "build_subset_pools",
    "CalibratedScorer", "DummyScorer", "brier_score", "build_training_data",
    "calibration_by_slice", "expected_calibration_error",
    "reliability_diagram_points",
    "LegRun", "ReconciliationEngine", "ReconciliationRun",
    "ExceptionReport", "build_exceptions",
    "Policy", "PolicyResult", "is_auto_confirmable",
    "ResidualAttribution",
    "find_subsets", "find_subsets_exact_first",
    "Candidate", "Decision", "Exception_",
]
