"""Reading a frequency law off the statistics a published pipeline reports.

FINDINGS.md F6-F16 show that no single statistic computed after transitional-probability segmentation
identifies the frequency law of the source: each responds to how concentrated the commonest units are, to the
realised vocabulary and to the segmentation threshold. That does not make the question hopeless. It makes it a
question about a whole statistic vector, compared with what each candidate law produces for a corpus of this
size under settings nobody can read off real song.

This module implements that comparison and, as importantly, its validation.

    build_reference(X, y)   reference distributions, one per candidate law, over simulated datasets that match
                            the target corpus in size and settings and vary the nuisance parameters over a prior
    posterior(ref, x)       probability of each law given one observed statistic vector, under a flat prior
    verdict(post)           the law, or None for "not identifiable", when no law reaches `min_probability`
    evaluate(ref, X, y)     discrimination and probability calibration on held-out simulated datasets

The statistics are the ones a real analysis reports, so the method applies to a real corpus without ground
truth: the mean log-log R^2 over groups, the mean rank-frequency slope, the mean number of unit types per
group, and the mean number of cuts per sequence. Boundary accuracy is deliberately absent: it needs the truth.

Densities are Gaussian kernel density estimates on standardised features (Scott's rule), which is enough for
four features and a few hundred reference datasets per law. The output is only as good as the prior over
nuisance parameters, so `evaluate` reports calibration as well as accuracy: a method that claims 90%
confidence should be right about 90% of the time, and where it is not, the honest report is "not identifiable".
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

#: Statistics a published analysis reports and a real corpus can supply.
FEATURES = ("observed_mean_r2", "observed_mean_slope", "observed_mean_unit_types", "cuts_per_sequence")


class NotEnoughReference(Exception):
    """A law has too few reference datasets, or a feature does not vary, so no density can be fitted."""


@dataclass(frozen=True)
class Reference:
    laws: tuple
    features: tuple
    centre: np.ndarray
    scale: np.ndarray
    densities: dict
    counts: dict

    def standardise(self, x):
        return (np.asarray(x, dtype=float) - self.centre) / self.scale


def _matrix(rows, features):
    x = np.asarray([[float(r[f]) for f in features] for r in rows], dtype=float)
    if x.ndim != 2 or x.shape[1] != len(features):
        raise ValueError(f"expected {len(features)} features, got array of shape {x.shape}")
    return x


def build_reference(rows, laws=None, features=FEATURES, min_per_law=40):
    """Reference densities from simulated datasets of known law.

    `rows` is a sequence of mappings with the feature keys and a "law" key.
    """
    rows = [r for r in rows if all(np.isfinite(float(r[f])) for f in features)]
    laws = tuple(laws) if laws else tuple(sorted({r["law"] for r in rows}))
    x = _matrix(rows, features)
    centre, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
    if not np.all(scale > 0):
        raise NotEnoughReference(f"feature(s) constant across the reference: {np.asarray(features)[scale <= 0]}")
    z = (x - centre) / scale
    densities, counts = {}, {}
    for law in laws:
        take = np.array([r["law"] == law for r in rows])
        if take.sum() < min_per_law:
            raise NotEnoughReference(f"{law}: {int(take.sum())} reference datasets, need {min_per_law}")
        densities[law] = gaussian_kde(z[take].T)
        counts[law] = int(take.sum())
    return Reference(laws=laws, features=tuple(features), centre=centre, scale=scale, densities=densities,
                     counts=counts)


def posterior(reference, observation):
    """Probability of each law given one observed statistic vector, under a flat prior over laws."""
    if isinstance(observation, dict):
        observation = [observation[f] for f in reference.features]
    z = reference.standardise(observation)
    if not np.all(np.isfinite(z)):
        raise ValueError("observation contains a non-finite feature")
    logp = np.array([reference.densities[law].logpdf(z)[0] for law in reference.laws])
    logp -= logp.max()
    p = np.exp(logp)
    return dict(zip(reference.laws, (p / p.sum()).tolist()))


def verdict(post, min_probability=0.7):
    """The most probable law, or None when no law reaches `min_probability` ("not identifiable")."""
    law, p = max(post.items(), key=lambda kv: kv[1])
    return law if p >= min_probability else None


def reliability(confidences, correct, n_bins=5):
    """Mean confidence against the share correct, in equal-width bins of confidence."""
    confidences, correct = np.asarray(confidences, dtype=float), np.asarray(correct, dtype=bool)
    edges = np.linspace(confidences.min(), 1.0, n_bins + 1) if confidences.size else np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (confidences >= lo) & (confidences <= hi if hi == edges[-1] else confidences < hi)
        if take.sum():
            out.append({"from": float(lo), "to": float(hi), "n": int(take.sum()),
                        "mean_confidence": float(confidences[take].mean()),
                        "share_correct": float(correct[take].mean())})
    return out


def evaluate(reference, rows, min_probability=0.7, n_bins=5):
    """Discrimination and probability calibration on held-out simulated datasets of known law."""
    rows = [r for r in rows if all(np.isfinite(float(r[f])) for f in reference.features)]
    if not rows:
        raise ValueError("no usable held-out rows")
    laws = reference.laws
    posteriors = [posterior(reference, r) for r in rows]
    truth = [r["law"] for r in rows]
    top = [max(p.items(), key=lambda kv: kv[1]) for p in posteriors]
    correct = [law == t for (law, _), t in zip(top, truth)]
    confidence = [p for _, p in top]
    decided = [p >= min_probability for p in confidence]
    brier = float(np.mean([sum((post[law] - (law == t)) ** 2 for law in laws) for post, t in zip(posteriors, truth)]))
    confusion = {t: {law: 0 for law in laws} for t in laws}
    for t, (law, _) in zip(truth, top):
        confusion[t][law] += 1
    per_law = {}
    for law in laws:
        p_law = np.array([post[law] for post in posteriors])
        is_law = np.array([t == law for t in truth])
        if is_law.any() and not is_law.all():
            pos, neg = p_law[is_law], p_law[~is_law]
            wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            per_law[law] = {"auc": float(wins / (pos.size * neg.size)),
                            "recall": float(np.mean([c for c, t in zip(correct, truth) if t == law])),
                            "n": int(is_law.sum())}
    decided_correct = [c for c, d in zip(correct, decided) if d]
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "chance": 1.0 / len(laws),
        "mean_confidence": float(np.mean(confidence)),
        "brier": brier,
        "min_probability": min_probability,
        "share_decided": float(np.mean(decided)),
        "accuracy_when_decided": float(np.mean(decided_correct)) if decided_correct else None,
        "per_law": per_law,
        "confusion": confusion,
        "reliability": reliability(confidence, correct, n_bins=n_bins),
    }
