"""Element-label errors of the kind unsupervised clustering makes.

Each function changes which element a token is labelled as, never where it sits. The ground-truth
word boundaries of a stream therefore stay valid, and a pipeline run on corrupted labels can still be
scored against them.

    confuse     a token takes a different label with probability `rate`
    oversplit   one true type becomes two labels (a common HDBSCAN outcome on graded calls)
    undersplit  two true types share one label
"""
import numpy as np


def _check(elements, n_elements):
    e = np.asarray(elements)
    if e.ndim != 1:
        raise ValueError(f"elements must be 1-D, got shape {e.shape}")
    if e.size and not np.issubdtype(e.dtype, np.integer):
        raise ValueError(f"elements must be integer labels, got dtype {e.dtype}")
    if e.size and (e.min() < 0 or e.max() >= n_elements):
        raise ValueError(
            f"labels must lie in [0, {n_elements}), got range [{e.min()}, {e.max()}]")
    return e.astype(np.int64, copy=False)


def _check_fraction(name, value):
    if not 0 <= value <= 1:  # also rejects NaN
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def confuse(elements, rate, n_elements, *, rng):
    """Relabel each token, with probability `rate`, as a different element chosen uniformly."""
    e = _check(elements, n_elements)
    _check_fraction("rate", rate)
    if rate > 0 and n_elements < 2:
        raise ValueError("confusion needs at least 2 element types")
    out = e.copy()
    hit = rng.random(e.size) < rate
    offset = rng.integers(1, n_elements, size=int(hit.sum())) if n_elements > 1 else 0
    out[hit] = (e[hit] + offset) % n_elements
    return out


def oversplit(elements, fraction, n_elements, *, rng):
    """Split round(fraction * n_elements) types, each into itself and one new label.

    Each token of a split type keeps its label or takes the new one with probability 1/2.
    Returns (labels, n_labels, split_types).
    """
    e = _check(elements, n_elements)
    _check_fraction("fraction", fraction)
    k = int(round(fraction * n_elements))
    split = rng.choice(n_elements, size=k, replace=False)
    out = e.copy()
    for j, t in enumerate(split):
        idx = np.flatnonzero(e == t)
        out[idx[rng.random(idx.size) < 0.5]] = n_elements + j
    return out, n_elements + k, np.sort(split)


def undersplit(elements, fraction, n_elements, *, rng):
    """Merge round(fraction * n_elements / 2) disjoint pairs of types into one label each.

    Labels are renumbered 0..n_labels-1. Returns (labels, n_labels, merged_pairs).
    """
    e = _check(elements, n_elements)
    _check_fraction("fraction", fraction)
    k = int(round(fraction * n_elements / 2))
    pairs = rng.permutation(n_elements)[:2 * k].reshape(k, 2)
    mapping = np.arange(n_elements)
    for a, b in pairs:
        mapping[b] = a
    uniq, compact = np.unique(mapping, return_inverse=True)
    return compact[e], int(uniq.size), pairs
