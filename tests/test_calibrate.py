"""unitsim.calibrate turns statistics into a verdict about a frequency law, so its failure modes matter more
than its successes: it must report low confidence when the laws overlap, and its probabilities must be honest."""
import numpy as np
import pytest

from unitsim.calibrate import (
    FEATURES,
    NotEnoughReference,
    build_reference,
    evaluate,
    posterior,
    reliability,
    verdict,
)


def rows_from(mean, n, law, rng, spread=1.0):
    return [{**{f: float(v) for f, v in zip(FEATURES, rng.normal(mean, spread))}, "law": law} for _ in range(n)]


def separable(rng, n=120):
    centres = {"zipf": [10, 10, 10, 10], "uniform": [-10, -10, -10, -10],
               "lognormal": [10, -10, 10, -10], "geometric": [-10, 10, -10, 10]}
    return [r for law, c in centres.items() for r in rows_from(c, n, law, rng)]


def overlapping(rng, n=120):
    return [r for law in ("zipf", "uniform", "lognormal", "geometric")
            for r in rows_from([0, 0, 0, 0], n, law, rng)]


def test_posterior_is_a_distribution_and_finds_the_obvious_answer():
    rng = np.random.default_rng(0)
    ref = build_reference(separable(rng))
    post = posterior(ref, dict(zip(FEATURES, [10, 10, 10, 10])))
    assert set(post) == {"zipf", "uniform", "lognormal", "geometric"}
    assert sum(post.values()) == pytest.approx(1.0)
    assert max(post, key=post.get) == "zipf" and post["zipf"] > 0.9
    assert verdict(post) == "zipf"


def test_overlapping_laws_give_no_verdict():
    rng = np.random.default_rng(1)
    ref = build_reference(overlapping(rng))
    post = posterior(ref, dict(zip(FEATURES, [0, 0, 0, 0])))
    assert max(post.values()) < 0.7
    assert verdict(post) is None
    assert verdict(post, min_probability=0.2) is not None       # the threshold is the caller's choice


def test_evaluate_reports_accuracy_confidence_and_calibration():
    rng = np.random.default_rng(2)
    ref = build_reference(separable(rng))
    good = evaluate(ref, separable(np.random.default_rng(3)))
    assert good["accuracy"] > 0.95 and good["chance"] == 0.25
    assert good["share_decided"] > 0.9
    assert good["brier"] < 0.1
    assert set(good["per_law"]) == {"zipf", "uniform", "lognormal", "geometric"}
    assert min(v["auc"] for v in good["per_law"].values()) > 0.95
    bad = evaluate(ref, overlapping(np.random.default_rng(4)))
    assert bad["accuracy"] < 0.6
    assert sum(sum(row.values()) for row in bad["confusion"].values()) == bad["n"]


def test_reliability_bins_confidence_against_correctness():
    bins = reliability([0.3, 0.35, 0.9, 0.95], [False, True, True, True], n_bins=2)
    assert [b["n"] for b in bins] == [2, 2]
    assert bins[0]["share_correct"] == 0.5 and bins[1]["share_correct"] == 1.0
    assert bins[0]["mean_confidence"] == pytest.approx(0.325)


def test_too_few_or_constant_reference_datasets_raise():
    rng = np.random.default_rng(5)
    with pytest.raises(NotEnoughReference):
        build_reference(separable(rng, n=10))
    constant = [{**{f: 1.0 for f in FEATURES}, "law": law} for law in ("zipf", "uniform") for _ in range(60)]
    with pytest.raises(NotEnoughReference):
        build_reference(constant)


def test_non_finite_features_are_dropped_or_rejected():
    rng = np.random.default_rng(6)
    rows = separable(rng)
    rows.append({**{f: float("nan") for f in FEATURES}, "law": "zipf"})
    ref = build_reference(rows)
    assert sum(ref.counts.values()) == len(rows) - 1
    with pytest.raises(ValueError):
        posterior(ref, dict(zip(FEATURES, [float("nan"), 0, 0, 0])))
