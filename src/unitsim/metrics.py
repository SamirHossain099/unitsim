"""Scores comparing a pipeline's output with the generator's ground truth.

Segmentation scores follow Goldwater, Griffiths & Johnson (2009): boundary, token and lexicon
precision, recall and F1. Cluster scores are the homogeneity, completeness and V-measure of Rosenberg
& Hirschberg (2007), which Sainburg, Thielk & Gentner (2020) use against hand labels. `loglog_r2`
reproduces the fit statistic of Arnon et al. (2025) as published, weaknesses included; the
Clauset-Shalizi-Newman alternative is in `powerlaw_fit`.

A segmentation is a boolean `starts` array: True where a unit begins. starts[0] must be True.

Convention: a pipeline that predicts nothing scores precision 0, not NaN. A NaN compared with
anything is False, so it would silently drop out of any count built on comparisons.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np


class DegenerateDistribution(ValueError):
    """A statistic is undefined for this input. Raised rather than returning NaN."""


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float


def _prf(tp, n_pred, n_true):
    if n_true == 0:
        raise DegenerateDistribution("ground truth contains nothing to recover")
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return PRF(float(precision), float(recall), float(f1))


def _check_starts(starts):
    s = np.asarray(starts)
    if s.dtype != bool:
        raise ValueError(f"starts must be a boolean array, got dtype {s.dtype}")
    if s.ndim != 1 or s.size == 0:
        raise ValueError("starts must be a non-empty 1-D array")
    if not s[0]:
        raise ValueError("starts[0] must be True: position 0 always begins a unit")
    return s


def _pair(pred_starts, true_starts):
    p, t = _check_starts(pred_starts), _check_starts(true_starts)
    if p.size != t.size:
        raise ValueError(f"segmentations differ in length: {p.size} vs {t.size}")
    return p, t


def boundary_prf(pred_starts, true_starts):
    """Precision/recall/F1 over interior boundaries (position 0 is excluded: it is always one)."""
    p, t = _pair(pred_starts, true_starts)
    pb, tb = p[1:], t[1:]
    return _prf(int(np.sum(pb & tb)), int(pb.sum()), int(tb.sum()))


def _spans(starts):
    idx = np.flatnonzero(starts)
    return set(zip(idx.tolist(), np.append(idx[1:], starts.size).tolist()))


def token_prf(pred_starts, true_starts):
    """A predicted unit token is correct only if both of its edges match a true token."""
    p, t = _pair(pred_starts, true_starts)
    ps, ts = _spans(p), _spans(t)
    return _prf(len(ps & ts), len(ps), len(ts))


def units(elements, starts):
    """Split an element sequence at `starts` into tuples, in order."""
    e = np.asarray(elements)
    s = _check_starts(starts)
    if e.shape != s.shape:
        raise ValueError(f"elements and starts differ in shape: {e.shape} vs {s.shape}")
    return [tuple(a.tolist()) for a in np.split(e, np.flatnonzero(s)[1:])]


def lexicon_prf(elements, pred_starts, true_starts):
    """Precision/recall/F1 over distinct unit types found versus types actually present."""
    _pair(pred_starts, true_starts)
    pred_types = set(units(elements, pred_starts))
    true_types = set(units(elements, true_starts))
    return _prf(len(pred_types & true_types), len(pred_types), len(true_types))


def unit_counts(elements, starts):
    """Each distinct unit and its frequency, most frequent first (ties broken by the unit)."""
    items = sorted(Counter(units(elements, starts)).items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items], np.array([v for _, v in items], dtype=np.int64)


def pooled_unit_counts(sequences, starts_list):
    """Units from several sequences pooled into one inventory, most frequent first.

    This is how the published analyses count: over every recording in a year (whale) or every bout
    in a recording (finch). A unit never spans a sequence boundary.
    """
    if len(sequences) != len(starts_list):
        raise ValueError(f"{len(sequences)} sequences but {len(starts_list)} segmentations")
    counter = Counter()
    for e, s in zip(sequences, starts_list):
        counter.update(units(e, s))
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items], np.array([v for _, v in items], dtype=np.int64)


@dataclass(frozen=True)
class RankFrequencyFit:
    r2: float
    slope: float
    n_types: int


def loglog_r2(counts):
    """Squared Pearson correlation between log rank and log frequency.

    The statistic Arnon et al. (2025) report as the fit to a power law ("mean R2=0.93", Pearson
    product moment correlation). Types are ranked by descending count; tied counts take consecutive
    ranks. Kept exactly as published so the published pipeline can be reproduced. It is not a test
    of whether a distribution is a power law.
    """
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {c.shape}")
    if not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("counts must be finite and non-negative")
    c = np.sort(c[c > 0])[::-1]
    if c.size < 3:
        raise DegenerateDistribution(f"need at least 3 types with non-zero count, got {c.size}")
    if c[0] == c[-1]:
        raise DegenerateDistribution("every type has the same count; the correlation is undefined")
    x = np.log(np.arange(1, c.size + 1))
    y = np.log(c)
    r = np.corrcoef(x, y)[0, 1]
    slope = np.polyfit(x, y, 1)[0]
    if not (np.isfinite(r) and np.isfinite(slope)):
        raise DegenerateDistribution("log-log fit is not finite")
    return RankFrequencyFit(float(r * r), float(slope), int(c.size))


@dataclass(frozen=True)
class BrevityFit:
    r2: float
    slope: float   # change in unit length per unit of log frequency; negative = frequent units shorter
    n_types: int


def brevity_fit(types, counts):
    """Zipf's law of brevity as Arnon et al. (2025) report it: the squared Pearson correlation between
    each unit type's length and its log frequency (whale song, Fig. 3A, R^2 = 0.62). One point per type.

    The slope is from regressing length on log frequency, so brevity shows as a negative slope.
    """
    lengths = np.array([len(t) for t in types], dtype=float)
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1 or c.size != lengths.size:
        raise ValueError(f"{lengths.size} types but counts of shape {c.shape}")
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("counts must be finite and positive")
    if c.size < 3:
        raise DegenerateDistribution(f"need at least 3 types, got {c.size}")
    if np.ptp(lengths) == 0 or np.ptp(c) == 0:
        raise DegenerateDistribution("unit lengths or counts are constant; the correlation is undefined")
    x = np.log(c)
    r = np.corrcoef(x, lengths)[0, 1]
    slope = np.polyfit(x, lengths, 1)[0]
    if not (np.isfinite(r) and np.isfinite(slope)):
        raise DegenerateDistribution("brevity fit is not finite")
    return BrevityFit(float(r * r), float(slope), int(c.size))


def loglog_r2_or_none(counts):
    """`loglog_r2(counts).r2`, or None where the statistic does not exist.

    Only DegenerateDistribution is caught; malformed input still raises. Returns None, never NaN, so an
    undefined value cannot slip silently through a comparison. Callers must decide what None means.
    """
    try:
        return loglog_r2(counts).r2
    except DegenerateDistribution:
        return None


@dataclass(frozen=True)
class ClusterAgreement:
    homogeneity: float
    completeness: float
    v_measure: float


def _entropy(counts):
    counts = counts[counts > 0].astype(float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def cluster_agreement(true_labels, pred_labels):
    """Homogeneity, completeness and V-measure. Invariant to how either labelling is numbered."""
    t, k = np.asarray(true_labels), np.asarray(pred_labels)
    if t.ndim != 1 or t.shape != k.shape or t.size == 0:
        raise ValueError("labelings must be non-empty 1-D arrays of equal length")
    _, ti = np.unique(t, return_inverse=True)
    _, ki = np.unique(k, return_inverse=True)
    n = t.size
    cont = np.zeros((ti.max() + 1, ki.max() + 1))
    np.add.at(cont, (ti, ki), 1)
    row, col = cont.sum(1), cont.sum(0)
    h_c, h_k = _entropy(row), _entropy(col)
    r_idx, c_idx = np.nonzero(cont)
    n_ck = cont[r_idx, c_idx]
    h_c_given_k = float(-np.sum(n_ck / n * np.log(n_ck / col[c_idx])))
    h_k_given_c = float(-np.sum(n_ck / n * np.log(n_ck / row[r_idx])))
    homogeneity = 1.0 if h_c == 0 else 1.0 - h_c_given_k / h_c
    completeness = 1.0 if h_k == 0 else 1.0 - h_k_given_c / h_k
    denom = homogeneity + completeness
    v = 0.0 if denom == 0 else 2 * homogeneity * completeness / denom
    return ClusterAgreement(float(homogeneity), float(completeness), float(v))
