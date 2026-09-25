import numpy as np
import pytest

from unitsim.corrupt import confuse, oversplit, undersplit


@pytest.fixture
def labels():
    return np.random.default_rng(0).integers(0, 8, 50_000)


def test_confuse_changes_the_requested_share_and_never_keeps_the_label(labels):
    out = confuse(labels, 0.2, 8, rng=np.random.default_rng(1))
    assert np.mean(out != labels) == pytest.approx(0.2, abs=0.01)
    assert out.min() >= 0 and out.max() < 8


def test_confuse_at_zero_is_the_identity(labels):
    assert np.array_equal(confuse(labels, 0.0, 8, rng=np.random.default_rng(2)), labels)


def test_confused_labels_are_spread_over_the_other_types():
    e = np.zeros(80_000, dtype=np.int64)
    out = confuse(e, 1.0, 5, rng=np.random.default_rng(3))
    assert not np.any(out == 0)
    assert np.bincount(out, minlength=5)[1:] / out.size == pytest.approx([0.25] * 4, abs=0.01)


def test_oversplit_is_a_refinement(labels):
    out, n, split = oversplit(labels, 0.25, 8, rng=np.random.default_rng(4))
    assert n == 10 and split.size == 2
    for lab in np.unique(out):
        assert np.unique(labels[out == lab]).size == 1  # each new label maps to one true type


def test_undersplit_is_a_coarsening(labels):
    out, n, pairs = undersplit(labels, 0.5, 8, rng=np.random.default_rng(5))
    assert n == 6 and pairs.shape == (2, 2)
    assert set(np.unique(out)) == set(range(6))
    for lab in np.unique(labels):
        assert np.unique(out[labels == lab]).size == 1  # each true type maps to one label


@pytest.mark.parametrize("call", [
    lambda rng: confuse(np.array([0, 9]), 0.1, 8, rng=rng),
    lambda rng: confuse(np.array([0, 1]), 1.5, 8, rng=rng),
    lambda rng: confuse(np.array([0.0, 1.0]), 0.1, 8, rng=rng),
    lambda rng: oversplit(np.array([0, 1]), float("nan"), 8, rng=rng),
    lambda rng: undersplit(np.array([-1, 1]), 0.5, 8, rng=rng),
])
def test_invalid_input_raises(call):
    with pytest.raises(ValueError):
        call(np.random.default_rng(6))
