import numpy as np
import pytest

from unitsim.baselines import (
    bigram_pseudo,
    compare_to_baseline,
    mean_over_groups,
    rotate_cuts,
    shuffle_within,
)
from unitsim.metrics import DegenerateDistribution


def test_shuffle_permutes_each_sequence_independently():
    seqs = [np.arange(50), np.array([3, 3, 1, 2])]
    out = shuffle_within(seqs, rng=np.random.default_rng(0))
    for a, b in zip(seqs, out):
        assert np.array_equal(np.sort(a), np.sort(b))
    assert not np.array_equal(out[0], seqs[0])


def _bool(positions, n):
    s = np.zeros(n, dtype=bool)
    s[list(positions)] = True
    return s


def test_rotation_is_a_cyclic_shift_of_the_original_cuts():
    rng = np.random.default_rng(1)
    for _ in range(200):
        length = int(rng.integers(2, 40))
        cuts = np.flatnonzero(rng.random(length - 1) < 0.3) + 1
        st = _bool([0, *cuts.tolist()], length)
        (new,) = rotate_cuts([st], rng=rng)
        new_cuts = set(np.flatnonzero(new).tolist())
        assert new[0]
        if cuts.size == 0:
            assert new_cuts == {0}
            continue
        assert any(new_cuts == {0} | {(c + k) % length for c in cuts} for k in range(1, length)), \
            "rotated cuts are not a cyclic shift of the originals"


def test_shared_offset_moves_every_sequence_by_the_same_amount():
    rng = np.random.default_rng(2)
    starts = [_bool([0, 3, 7], 20), _bool([0, 5], 20)]
    a, b = rotate_cuts(starts, rng=rng, per_sequence=False)
    shift_a = {(c - 3) % 20 for c in np.flatnonzero(a) if c} | {(c - 7) % 20 for c in np.flatnonzero(a) if c}
    shift_b = {(c - 5) % 20 for c in np.flatnonzero(b) if c}
    assert shift_a & shift_b


def test_rotation_rejects_malformed_starts():
    with pytest.raises(ValueError):
        rotate_cuts([np.array([0, 1, 0])], rng=np.random.default_rng(0))


def test_bigram_pseudo_on_a_deterministic_cycle_is_the_same_cycle():
    target = np.tile([0, 1, 2, 3], 10)
    pseudo = bigram_pseudo(target, 50, rng=np.random.default_rng(3))
    assert pseudo.shape == (50, 40)
    succ = {0: 1, 1: 2, 2: 3, 3: 0}
    for row in pseudo:
        assert all(succ[int(a)] == int(b) for a, b in zip(row[:-1], row[1:]))


def test_bigram_pseudo_uses_only_target_bigrams_or_restarts():
    rng = np.random.default_rng(4)
    target = rng.integers(0, 5, 60)
    target[-1] = 9                     # 9 occurs only at the end: forces restarts
    bigrams = set(zip(target[:-1].tolist(), target[1:].tolist()))
    heads = set(target[:-1].tolist())
    pseudo = bigram_pseudo(target, 200, rng=rng)
    for row in pseudo:
        for a, b in zip(row[:-1].tolist(), row[1:].tolist()):
            assert (a, b) in bigrams or a not in heads


def test_bigram_pseudo_matches_bigram_frequencies_in_expectation():
    target = np.array([0, 1, 0, 2, 0, 1, 0, 1, 2, 0] * 5)
    pseudo = bigram_pseudo(target, 3000, rng=np.random.default_rng(5))
    real_after_0 = np.bincount(target[1:][target[:-1] == 0], minlength=3) / np.sum(target[:-1] == 0)
    rows, cols = np.nonzero(pseudo[:, :-1] == 0)
    fake_after_0 = np.bincount(pseudo[rows, cols + 1], minlength=3) / rows.size
    assert fake_after_0 == pytest.approx(real_after_0, abs=0.01)


def test_mean_over_groups_raises_by_default_on_an_undefined_group():
    with pytest.raises(DegenerateDistribution):
        mean_over_groups([0.9, None, 0.8])
    assert mean_over_groups([0.9, 0.8]).mean == pytest.approx(0.85)


def test_mean_over_groups_skip_and_zero_policies():
    skip = mean_over_groups([0.9, None, 0.6], "skip")
    zero = mean_over_groups([0.9, None, 0.6], "zero")
    assert (skip.mean, skip.n_undefined, skip.n_total) == (pytest.approx(0.75), 1, 3)
    assert (zero.mean, zero.n_undefined) == (pytest.approx(0.5), 1)
    assert mean_over_groups([None, None], "skip").mean is None
    assert mean_over_groups([None, None], "zero").mean == 0.0


@pytest.mark.parametrize("values,policy", [([0.5, float("nan")], "skip"), ([], "skip"),
                                           ([0.5], "median")])
def test_mean_over_groups_rejects_bad_input(values, policy):
    with pytest.raises(ValueError):
        mean_over_groups(values, policy)


def test_compare_to_baseline_reports_z_and_empirical_p():
    c = compare_to_baseline(0.93, [0.5, 0.6, 0.7, 0.55, 0.65])
    b = np.array([0.5, 0.6, 0.7, 0.55, 0.65])
    assert c.z == pytest.approx((0.93 - b.mean()) / b.std(ddof=1))
    assert c.p_empirical == pytest.approx(1 / 6)


@pytest.mark.parametrize("observed,baseline,err", [
    (float("nan"), [0.1, 0.2], ValueError),
    (0.5, [0.1, float("nan")], ValueError),
    (0.5, [0.3], ValueError),
    (0.5, [0.3, 0.3, 0.3], DegenerateDistribution),
])
def test_compare_to_baseline_refuses_undefined_inputs(observed, baseline, err):
    with pytest.raises(err):
        compare_to_baseline(observed, baseline)
