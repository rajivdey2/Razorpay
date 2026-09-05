"""Property tests for the matching engine's algorithmic core.

These target the properties that are hard to check by example and expensive to get
wrong: subset-sum completeness, the admissibility of its pruning, the optimality of
the assignment, and the invariants of the confusion-aware edit distance.

The most valuable test here is ``test_pruning_never_loses_a_solution``. It compares
the pruned search against brute-force enumeration over every subset, which is the
only way to know that a bound is admissible rather than merely fast. A pruning bug
does not crash -- it silently reduces recall, and looks like a matcher that is not
very good.
"""

from __future__ import annotations

from itertools import combinations

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from app.matching.features import (
    char_ngram_cosine,
    reference_similarity,
    weighted_levenshtein,
)
from app.matching.subsetsum import find_subsets, find_subsets_exact_first

amounts = st.lists(
    st.integers(min_value=-500_00, max_value=900_00).filter(lambda x: x != 0),
    min_size=1, max_size=12,
)


def brute_force(xs: list[int], target: int, tol: int, max_size: int) -> set[tuple[int, ...]]:
    out = set()
    for size in range(1, min(max_size, len(xs)) + 1):
        for idx in combinations(range(len(xs)), size):
            if abs(sum(xs[i] for i in idx) - target) <= tol:
                out.add(idx)
    return out


class TestSubsetSum:
    @settings(max_examples=250, deadline=None,
              suppress_health_check=[HealthCheck.filter_too_much])
    @given(amounts, st.integers(min_value=-1000_00, max_value=1000_00))
    def test_pruning_never_loses_a_solution(self, xs, target):
        """Every subset brute force finds, the pruned search must also find.

        This is what makes the bounds *admissible* rather than heuristic. Note the
        negative amounts in the strategy: refunds make the running sum non-monotone,
        which is exactly the case a single suffix-sum bound gets wrong, so the
        two-sided bound is under test here and not just decoration.
        """
        xs = sorted(xs, reverse=True)
        res = find_subsets(xs, target, 0, max_size=6, max_solutions=10**6,
                           node_budget=10**7)
        assume(res.exhausted)
        expected = brute_force(xs, target, 0, 6)
        found = {s.indices for s in res.solutions}
        assert expected == found, f"lost {expected - found}, invented {found - expected}"

    @settings(max_examples=200, deadline=None)
    @given(amounts, st.integers(min_value=0, max_value=200_00))
    def test_every_returned_subset_actually_sums(self, xs, tol):
        xs = sorted(xs, reverse=True)
        target = sum(xs[:2])
        res = find_subsets(xs, target, tol, max_size=8)
        for s in res.solutions:
            assert sum(xs[i] for i in s.indices) == s.total_paise
            assert abs(s.total_paise - target) <= tol

    @settings(max_examples=150, deadline=None)
    @given(amounts)
    def test_no_index_is_used_twice(self, xs):
        xs = sorted(xs, reverse=True)
        res = find_subsets(xs, sum(xs) // 2, 100, max_size=8)
        for s in res.solutions:
            assert len(set(s.indices)) == len(s.indices)

    def test_node_budget_is_honoured(self):
        """A pathological pool must terminate and *say* it was truncated.

        The saying-so is the point. A search that quietly returns partial results
        is indistinguishable from one that found everything, and the caller would
        report full coverage on a window it never finished.
        """
        # The target has to sit in the *middle* of the reachable range, or the
        # suffix bound prunes the whole tree at the root and the search finishes
        # in one node -- which is what the first version of this test accidentally
        # measured.
        xs = sorted([i * 7 + 1 for i in range(40)], reverse=True)
        target = sum(xs) // 2
        res = find_subsets(xs, target, 5000, max_size=20,
                           max_solutions=10**6, node_budget=500)
        assert res.nodes_visited <= 520
        assert res.exhausted is False, (
            "a truncated search must report itself; reporting full coverage on a "
            "window it never finished is the failure this guards"
        )

    def test_exact_first_prefers_the_exact_answer(self):
        """With an exact solution available, the loose tolerance never runs.

        Otherwise a wide tolerance manufactures plausible alternatives to an answer
        that was already exactly right, and ambiguity is invented from nothing.
        """
        xs = [500_00, 300_00, 200_00, 199_99]
        res = find_subsets_exact_first(xs, 500_00, 100)
        assert all(s.residual_paise == 0 for s in res.solutions)

    def test_ambiguity_is_reported_not_hidden(self):
        """Four equal amounts summing to a two-item target: 6 real answers.

        The count is the output that matters. Returning the first one silently is
        how a matcher confidently posts one of six equally-likely explanations.
        """
        xs = [100_00, 100_00, 100_00, 100_00]
        res = find_subsets(xs, 200_00, 0, max_size=4)
        assert res.ambiguity == 6

    def test_empty_pool(self):
        res = find_subsets([], 1000, 0)
        assert res.solutions == [] and res.exhausted

    def test_smallest_explanation_first(self):
        """Occam: two invoices beat five that happen to hit the same total."""
        xs = sorted([600_00, 400_00, 250_00, 200_00, 150_00, 100_00], reverse=True)
        res = find_subsets(xs, 1000_00, 0, max_size=6)
        assert res.solutions[0].size <= res.solutions[-1].size


class TestReferenceSimilarity:
    @given(st.text(alphabet="ABCDEFGH0123456789", min_size=1, max_size=18))
    def test_identity(self, s):
        assert weighted_levenshtein(s, s) == 0.0

    @given(
        st.text(alphabet="ABC0123", min_size=0, max_size=10),
        st.text(alphabet="ABC0123", min_size=0, max_size=10),
    )
    def test_symmetry(self, a, b):
        assert weighted_levenshtein(a, b) == weighted_levenshtein(b, a)

    @given(
        st.text(alphabet="ABC0123", min_size=0, max_size=7),
        st.text(alphabet="ABC0123", min_size=0, max_size=7),
        st.text(alphabet="ABC0123", min_size=0, max_size=7),
    )
    def test_triangle_inequality(self, a, b, c):
        ab = weighted_levenshtein(a, b)
        bc = weighted_levenshtein(b, c)
        ac = weighted_levenshtein(a, c)
        assert ac <= ab + bc + 1e-9

    def test_confusable_substitution_costs_less_than_a_real_one(self):
        """``O``->``0`` is a keying error; ``O``->``X`` is a different reference.

        Scoring them identically means either treating real mismatches as noise or
        treating OCR noise as a real mismatch. Both are wrong in the same system.
        """
        confusable = weighted_levenshtein("HDFC260304", "HDFC26O304")
        genuine = weighted_levenshtein("HDFC260304", "HDFC26X304")
        assert confusable < genuine
        assert confusable > 0, "a confusable pair must never be treated as identical"

    def test_truncation_scores_high_but_not_perfect(self):
        """A bank cutting a narration preserves the prefix exactly.

        Edit distance alone punishes the missing tail as though it were corruption,
        which loses the strongest signal available on a truncated line.
        """
        sim, _, _ = reference_similarity(("HDFC260304560105",), ("HDFC260304",))
        assert 0.5 < sim < 1.0

    def test_unrelated_references_score_low(self):
        sim, _, _ = reference_similarity(("HDFC260304560105",), ("ICIC991231000001",))
        assert sim < 0.5

    def test_empty_candidates(self):
        assert reference_similarity((), ("ABC",)) == (0.0, "", "")


class TestNarrationSimilarity:
    @given(st.text(min_size=3, max_size=40))
    def test_self_similarity_is_exactly_one(self, s):
        """Exactly 1.0, not 1.0000000000000002.

        The function clamps, because ``sqrt(x) * sqrt(x) != x`` in floating point
        and every consumer treats this as a bounded feature.
        """
        assume(len(s) >= 3)
        assert char_ngram_cosine(s, s) == 1.0

    @given(st.text(min_size=0, max_size=30), st.text(min_size=0, max_size=30))
    def test_bounded(self, a, b):
        assert 0.0 <= char_ngram_cosine(a, b) <= 1.0 + 1e-9

    def test_shared_reference_dominates_boilerplate(self):
        """Two narrations sharing a UTR score higher than two sharing only wording."""
        same_ref = char_ngram_cosine(
            "NEFT-HDFC260304560105-RAZORPAY SOFTWARE PVT",
            "RAZORPAY SETTLEMENT HDFC260304560105",
        )
        diff_ref = char_ngram_cosine(
            "NEFT-HDFC260304560105-RAZORPAY SOFTWARE PVT",
            "RAZORPAY SETTLEMENT ICIC991231000001",
        )
        assert same_ref > diff_ref
