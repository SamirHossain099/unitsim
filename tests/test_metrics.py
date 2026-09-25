import numpy as np
import pytest

from unitsim.metrics import (
    DegenerateDistribution,
    boundary_prf,
    brevity_fit,
    cluster_agreement,
    lexicon_prf,
    loglog_r2,
    loglog_r2_or_none,
    pooled_unit_counts,
    token_prf,
    unit_counts,
    units,
)


def S(*positions, n):
    s = np.zeros(n, dtype=bool)
    s[list(positions)] = True
    return s


def test_perfect_segmentation_scores_one_everywhere():
    e = np.array([0, 1, 2, 0, 1, 3, 3])
    t = S(0, 3, 5, n=7)
    assert boundary_prf(t, t).f1 == token_prf(t, t).f1 == lexicon_prf(e, t, t).f1 == 1.0


def test_hand_worked_example():
    # true [0 1 2][0 1][3 3]   predicted [0 1][2 0 1][3 3]
    e = np.array([0, 1, 2, 0, 1, 3, 3])
    t, p = S(0, 3, 5, n=7), S(0, 2, 5, n=7)
    b = boundary_prf(p, t)                       # interior true {3,5}, predicted {2,5}
    assert (b.precision, b.recall) == (0.5, 0.5)
    tok = token_prf(p, t)                        # only span (5,7) matches
    assert (tok.precision, tok.recall) == pytest.approx((1 / 3, 1 / 3))
    lex = lexicon_prf(e, p, t)                   # types (0,1) and (3,3) shared
    assert (lex.precision, lex.recall) == pytest.approx((2 / 3, 2 / 3))


def test_no_predicted_boundaries_scores_zero_not_nan():
    b = boundary_prf(S(0, n=6), S(0, 3, n=6))
    assert (b.precision, b.recall, b.f1) == (0.0, 0.0, 0.0)


def test_single_unit_ground_truth_is_degenerate():
    with pytest.raises(DegenerateDistribution):
        boundary_prf(S(0, 2, n=4), S(0, n=4))


@pytest.mark.parametrize("bad", [S(2, n=4), np.array([1, 0, 1, 0]), np.zeros(0, dtype=bool)])
def test_malformed_starts_raise(bad):
    with pytest.raises(ValueError):
        boundary_prf(bad, S(0, 2, n=4))


def test_units_and_their_counts():
    e = np.array([0, 1, 0, 1, 2])
    s = S(0, 2, 4, n=5)
    assert units(e, s) == [(0, 1), (0, 1), (2,)]
    types, counts = unit_counts(e, s)
    assert types == [(0, 1), (2,)]
    assert counts.tolist() == [2, 1]


def test_loglog_r2_is_one_on_an_exact_power_law():
    f = loglog_r2(np.round(1e6 * np.arange(1, 51) ** -1.0))
    assert f.r2 > 0.99999
    assert f.slope == pytest.approx(-1.0, abs=1e-3)
    assert f.n_types == 50


def test_loglog_r2_ignores_zero_counts():
    assert loglog_r2([100, 50, 0, 33, 0]).n_types == 3


@pytest.mark.parametrize("counts", [[5, 5, 5, 5], [10, 3], [0, 0, 7]])
def test_loglog_r2_refuses_undefined_inputs(counts):
    with pytest.raises(DegenerateDistribution):
        loglog_r2(counts)


def test_cluster_agreement_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    cases = [(rng.integers(0, 6, 400), rng.integers(0, 9, 400)) for _ in range(20)]
    cases += [(np.zeros(50, int), rng.integers(0, 3, 50)),   # one true class
              (rng.integers(0, 3, 50), np.zeros(50, int))]   # one cluster
    for t, k in cases:
        ours = cluster_agreement(t, k)
        ref = skm.homogeneity_completeness_v_measure(t, k)
        assert (ours.homogeneity, ours.completeness, ours.v_measure) == pytest.approx(ref, abs=1e-10)


def test_loglog_r2_or_none_returns_none_only_when_undefined():
    assert loglog_r2_or_none([1, 1, 1, 1]) is None          # all singletons: the F2 case
    assert loglog_r2_or_none([9, 4, 2, 1]) == loglog_r2([9, 4, 2, 1]).r2
    with pytest.raises(ValueError):
        loglog_r2_or_none([3, -1, 2])                        # malformed input is not swallowed


def test_pooled_counts_never_join_units_across_sequences():
    seqs = [np.array([0, 1]), np.array([0, 1])]
    starts = [S(0, n=2), S(0, n=2)]
    types, counts = pooled_unit_counts(seqs, starts)
    assert types == [(0, 1)] and counts.tolist() == [2]
    with pytest.raises(ValueError):
        pooled_unit_counts(seqs, starts[:1])


def test_brevity_fit_detects_frequent_short_units():
    types = [(0,), (1, 2), (3, 4, 5), (6, 7, 8, 9)]
    counts = [1000, 100, 10, 1]                       # length rises exactly as log count falls
    b = brevity_fit(types, counts)
    assert b.r2 == pytest.approx(1.0)
    assert b.slope < 0
    assert b.slope == pytest.approx(-1 / np.log(10))


def test_brevity_fit_on_generated_words_follows_the_generator():
    from unitsim.generate import build_lexicon, generate_stream
    rng = np.random.default_rng(3)
    for brevity, sign in ((True, -1), (False, 0)):
        slopes = []
        for _ in range(30):
            lex = build_lexicon(60, 12, rng=rng, law="zipf", brevity=brevity)
            counts = generate_stream(lex, 5000, rng=rng).word_counts()
            present = counts > 0
            slopes.append(brevity_fit([w for w, p in zip(lex.words, present) if p],
                                      counts[present]).slope)
        if sign < 0:
            assert np.mean(slopes) < -0.2
        else:
            assert abs(np.mean(slopes)) < 0.1


@pytest.mark.parametrize("types,counts", [
    ([(0, 1), (2, 3), (4, 5)], [9, 3, 1]),            # constant length
    ([(0,), (1, 2), (3, 4, 5)], [4, 4, 4]),           # constant count
    ([(0,), (1, 2)], [5, 1]),                         # too few types
])
def test_brevity_fit_refuses_undefined_inputs(types, counts):
    with pytest.raises(DegenerateDistribution):
        brevity_fit(types, counts)


def test_cluster_agreement_ignores_label_names():
    a = cluster_agreement([0, 0, 1, 1, 2, 2], [5, 5, 3, 3, 9, 9])
    assert a.v_measure == pytest.approx(1.0)
