"""The published baselines, reproduced so they can be run where the truth is known.

shuffle  Arnon et al. 2025; Kirby et al. 2026. Elements shuffled within each song or bout, then the
         whole pipeline reruns. Keeps element frequencies; destroys sequence.
rotate   Arnon et al. 2025; Kirby et al. 2026. Cut positions shifted by a random number of steps and
         units re-read from the unchanged elements. Keeps element order and unit lengths; breaks the
         alignment between cuts and TP dips. The finch paper draws an offset per bout and wraps cuts
         past the end back to the start. The whale paper describes one offset per year's dataset and
         does not say how the end is handled.
bigram   Kirby et al. 2026. Each bout regenerated from its own bigrams. Keeps first-order transitions,
         the cue TP segmentation uses; destroys longer structure.

See VERIFICATION.md for the published text behind each construction and the interpretation calls.
"""
from dataclasses import dataclass

import numpy as np

from .metrics import DegenerateDistribution


def shuffle_within(sequences, *, rng):
    """A copy of `sequences` with the elements of each sequence independently permuted."""
    return [rng.permutation(np.asarray(s)) for s in sequences]


def rotate_cuts(starts, *, rng, per_sequence=True):
    """Shift every interior cut by an offset, wrapping modulo the sequence length.

    per_sequence=True draws an offset in 1..L-1 for each sequence (Kirby et al. 2026). False draws one
    offset for the whole dataset and applies it to every sequence modulo its length (one reading of
    Arnon et al. 2025). A cut that lands on position 0 merges with the sequence start, so a sequence
    can lose one unit. Returns new boolean start arrays.
    """
    starts = [np.asarray(s) for s in starts]
    for i, s in enumerate(starts):
        if s.dtype != bool or s.ndim != 1 or s.size == 0 or not s[0]:
            raise ValueError(f"starts[{i}] must be a non-empty boolean array beginning with True")
    shared = None if per_sequence else int(rng.integers(1, max(s.size for s in starts) + 1))
    out = []
    for s in starts:
        length = s.size
        cuts = np.flatnonzero(s[1:]) + 1
        new = np.zeros(length, dtype=bool)
        new[0] = True
        if length > 1 and cuts.size:
            k = int(rng.integers(1, length)) if per_sequence else shared % length
            new[(cuts + k) % length] = True
        out.append(new)
    return out


def bigram_pseudo(sequence, n_replicates, *, rng):
    """Pseudo sequences of the same length that keep the target's own bigram statistics.

    Follows the three steps of Kirby et al. (2026), vectorised across replicates:
      (i)   start empty;
      (ii)  append the element at a uniformly random position of the target;
      (iii) take the occurrences, in the target, of the current last element that have a successor;
            pick one uniformly and append its successor. With no such occurrence, go to (ii).
      Repeat (iii) until the target length is reached.

    Two interpretation calls. The paper says to find occurrences "in the pseudo song bout"; a
    successor can only be read from the target, so the target is searched. An element found both
    mid-sequence and at the end is continued from an occurrence that has a successor.

    Returns an int array of shape (n_replicates, len(sequence)).
    """
    s = np.asarray(sequence)
    if s.ndim != 1 or s.size == 0 or not np.issubdtype(s.dtype, np.integer):
        raise ValueError("sequence must be a non-empty 1-D array of integer labels")
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    length = s.size
    order = np.argsort(s[:-1], kind="stable")
    heads = s[:-1][order]                 # element at each position that has a successor, sorted
    out = np.empty((n_replicates, length), dtype=s.dtype)
    out[:, 0] = s[rng.integers(0, length, size=n_replicates)]
    for t in range(1, length):
        last = out[:, t - 1]
        lo = np.searchsorted(heads, last, side="left")
        hi = np.searchsorted(heads, last, side="right")
        has = hi > lo
        pick = lo + np.floor(rng.random(n_replicates) * np.maximum(hi - lo, 1)).astype(np.int64)
        restart = s[rng.integers(0, length, size=n_replicates)]
        out[:, t] = np.where(has, s[order[np.minimum(pick, heads.size - 1)] + 1], restart)
    return out


@dataclass(frozen=True)
class BaselineComparison:
    observed: float
    baseline_mean: float
    baseline_sd: float
    z: float
    p_empirical: float   # (1 + #baseline >= observed) / (1 + replicates)
    n_replicates: int


def compare_to_baseline(observed, baseline):
    """z-score of an observed statistic against baseline replicates, as the papers report it.

    Raises on non-finite input or a baseline with no spread, rather than returning NaN: a NaN compared
    with anything is False, which would silently report maximal significance.
    """
    b = np.asarray(baseline, dtype=float)
    if b.ndim != 1 or b.size < 2:
        raise ValueError("need at least 2 baseline replicates")
    if not np.isfinite(observed) or not np.all(np.isfinite(b)):
        raise ValueError("observed and baseline values must all be finite")
    sd = float(b.std(ddof=1))
    if sd == 0:
        raise DegenerateDistribution("every baseline replicate is identical; z is undefined")
    mean = float(b.mean())
    return BaselineComparison(float(observed), mean, sd, (float(observed) - mean) / sd,
                              float((1 + np.sum(b >= observed)) / (1 + b.size)), int(b.size))


UNDEFINED_POLICIES = ("raise", "skip", "zero")


@dataclass(frozen=True)
class Aggregate:
    mean: float | None   # None only under "skip" when no group is defined
    n_undefined: int
    n_total: int


def mean_over_groups(values, undefined="raise"):
    """Mean of per-group statistics, some of which may be undefined (None).

    A group's log-log R^2 does not exist when every discovered unit occurs equally often -- in practice
    when every unit is unique, which shuffled baselines produce at small corpora (FINDINGS.md F2). The
    published code is not available, so the right treatment is unknown and this makes it a stated choice:

    raise  any undefined group raises DegenerateDistribution. The default: never decide silently.
    skip   average the defined groups only, as a pandas mean with skipna=True treats NaN.
    zero   count an undefined group as R^2 = 0: no rank-frequency relationship.
    """
    if undefined not in UNDEFINED_POLICIES:
        raise ValueError(f"undefined must be one of {UNDEFINED_POLICIES}, got {undefined!r}")
    vals = list(values)
    if not vals:
        raise ValueError("no group statistics given")
    for v in vals:
        if v is not None and not np.isfinite(v):
            raise ValueError("a group statistic is non-finite; mark undefined groups with None")
    n_undefined = sum(v is None for v in vals)
    if n_undefined and undefined == "raise":
        raise DegenerateDistribution(
            f"{n_undefined} of {len(vals)} groups have an undefined statistic; choose a policy")
    if undefined == "zero":
        mean = float(np.mean([0.0 if v is None else v for v in vals]))
    else:
        defined = [v for v in vals if v is not None]
        mean = float(np.mean(defined)) if defined else None
    return Aggregate(mean, n_undefined, len(vals))
