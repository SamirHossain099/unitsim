"""Tests for the Clauset-Shalizi-Newman discrete power-law fit.

The sampler and the likelihood are the parts most likely to be wrong in a way that still produces
plausible numbers, so they are checked against closed forms and the reference `powerlaw` package.
Whether the bootstrap p-value is calibrated is a validation run with its own results file, not a
unit test: a single seeded p-value from a true power law is uniform on [0, 1] and proves nothing.
"""
import numpy as np
import pytest
from scipy.special import zeta

from unitsim.metrics import DegenerateDistribution
from unitsim.powerlaw_fit import (
    PowerLawFit,
    _cdf,
    _lognormal_logpmf,
    compare_lognormal,
    fit_alpha,
    fit_powerlaw,
    gof_pvalue,
    ks_distance,
    sample_powerlaw,
)


@pytest.mark.parametrize("alpha,xmin", [(2.5, 1), (2.0, 5), (1.7, 1)])
def test_sampler_matches_the_pmf(alpha, xmin):
    s = sample_powerlaw(200_000, alpha, xmin, rng=np.random.default_rng(1))
    assert s.min() >= xmin
    k = np.arange(xmin, xmin + 6)
    pmf = k.astype(float) ** -alpha / zeta(alpha, xmin)
    emp = np.array([(s == v).mean() for v in k])
    assert np.all(np.abs(emp - pmf) < 5 * np.sqrt(pmf * (1 - pmf) / s.size))


@pytest.mark.parametrize("alpha", [1.8, 2.0, 2.5, 3.0])
def test_mle_recovers_the_exponent(alpha):
    s = sample_powerlaw(50_000, alpha, 1, rng=np.random.default_rng(2))
    assert fit_alpha(s, 1) == pytest.approx(alpha, abs=0.03)


def test_mle_agrees_with_the_reference_implementation():
    powerlaw = pytest.importorskip("powerlaw")
    s = sample_powerlaw(5_000, 2.4, 1, rng=np.random.default_rng(3))
    ref = powerlaw.Fit(s, discrete=True, xmin=1)
    assert fit_alpha(s, 1) == pytest.approx(ref.power_law.alpha, abs=5e-3)


def test_ks_distance_is_the_exact_supremum_over_integers():
    x = sample_powerlaw(300, 2.6, 1, rng=np.random.default_rng(4))
    a = fit_alpha(x, 1)
    grid = np.arange(1, x.max() + 1)
    emp = np.searchsorted(np.sort(x), grid, side="right") / x.size
    assert ks_distance(x, a, 1) == pytest.approx(np.max(np.abs(emp - _cdf(a, grid, 1))), abs=1e-12)


def test_ks_distance_catches_a_gap_that_data_points_alone_miss():
    x = np.array([1] * 50 + [100] * 50)
    at_points = np.max(np.abs(np.array([0.5, 1.0]) - _cdf(1.5, np.array([1, 100]), 1)))
    assert ks_distance(x, 1.5, 1) > at_points + 0.1   # the peak sits at 99, not a data point


def test_xmin_selection_finds_where_the_tail_starts():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.integers(1, 10, size=2000), sample_powerlaw(3000, 2.5, 10, rng=rng)])
    f = fit_powerlaw(x)
    assert 8 <= f.xmin <= 14
    assert f.alpha == pytest.approx(2.5, abs=0.2)
    assert not f.xmin_fixed


def test_degenerate_tail_raises():
    with pytest.raises(DegenerateDistribution):
        fit_alpha(np.array([3, 3, 3, 3]), 3)


def test_no_usable_tail_raises():
    with pytest.raises(DegenerateDistribution):
        fit_powerlaw(np.array([1, 2, 3]), min_tail=10)


@pytest.mark.parametrize("bad", [np.array([0, 1, 2]), np.array([1.5, 2.0]), np.array([])])
def test_invalid_counts_raise(bad):
    with pytest.raises(ValueError):
        fit_powerlaw(bad)


def test_gof_refuses_a_nan_observed_statistic():
    fit = PowerLawFit(alpha=2.0, xmin=1, ks=float("nan"), n_tail=10, n=10, xmin_fixed=True)
    with pytest.raises(ValueError):
        gof_pvalue(np.arange(1, 11), fit, rng=np.random.default_rng(0), n_boot=5)


def test_gof_rejects_a_lognormal_outright():
    rng = np.random.default_rng(6)
    x = np.floor(rng.lognormal(2.0, 0.5, size=2000)).astype(np.int64)
    x = x[x >= 1]
    g = gof_pvalue(x, fit_powerlaw(x, xmin=1), rng=rng, n_boot=50)
    assert g.p == 0.0 and g.n_boot == 50 and g.n_failed == 0


def test_likelihood_ratio_favours_the_generating_model():
    rng = np.random.default_rng(7)
    ln = np.floor(rng.lognormal(2.0, 0.8, size=20_000)).astype(np.int64)
    ln = ln[ln >= 1]
    lr = compare_lognormal(ln, fit_powerlaw(ln, xmin=1))
    assert lr.R < 0 and lr.p < 1e-6
    assert lr.mu == pytest.approx(2.0, abs=0.05)
    assert lr.sigma == pytest.approx(0.8, abs=0.05)
    assert not lr.at_bound                  # a genuine lognormal has an interior optimum

    pl = sample_powerlaw(5_000, 2.2, 1, rng=rng)
    lr_pl = compare_lognormal(pl, fit_powerlaw(pl, xmin=1))
    assert lr_pl.R > 0 or lr_pl.p > 0.05   # never significantly favours the wrong model


@pytest.mark.parametrize("c", [-1.0, -1.5, -0.6])
def test_lognormal_ridge_tends_to_a_binned_power_law_not_k_to_the_minus_alpha(c):
    """Why the lognormal comparison at xmin = 1 is also a comparison of discretisations (FINDINGS F4).

    With mu = c * sigma^2 and sigma large, the discrete lognormal P(k) = F(k+1) - F(k) becomes
    proportional to the integral of x^(c-1) over [k, k+1], i.e. ((k+1)^c - k^c) / c. The discrete power
    law with the same tail exponent, alpha = 1 - c, is k^-alpha, which has a different shape at small k.
    """
    k = np.arange(1, 7, dtype=float)
    sigma = 1e3
    logp = _lognormal_logpmf(k, c * sigma ** 2, sigma, 1)
    ridge = np.exp(logp - logp[0])
    binned = ((k + 1) ** c - k ** c) / c
    discrete = k ** (c - 1)
    assert ridge == pytest.approx(binned / binned[0], rel=1e-4)
    assert not np.allclose(ridge, discrete / discrete[0], rtol=0.02)


@pytest.mark.parametrize("seed", range(6))
def test_lognormal_comparison_on_heavy_tails_at_xmin_1_is_polished_to_tolerance(seed):
    """Heavy tails at xmin = 1 put the lognormal optimum on the flat ridge, where L-BFGS-B can stop early
    or report success short of the supremum (FINDINGS F4). The polish must deliver a fit within
    tolerance rather than raise or return a silently worse optimum."""
    x = sample_powerlaw(519, 1.77, 1, rng=np.random.default_rng(100 + seed))
    lr = compare_lognormal(x, fit_powerlaw(x, xmin=1))
    assert np.isfinite(lr.R) and np.isfinite(lr.p)
    assert 0.0 <= lr.optimisation_gap < 1e-2


def test_widened_bounds_are_numerically_finite_on_the_ridge():
    k = np.arange(1, 2000, dtype=float)
    assert np.all(np.isfinite(_lognormal_logpmf(k, -1e6, 1e4 * 0.999, 1)))


def test_likelihood_ratio_agrees_in_sign_with_the_reference_implementation():
    powerlaw = pytest.importorskip("powerlaw")
    rng = np.random.default_rng(8)
    ln = np.floor(rng.lognormal(2.0, 0.8, size=5_000)).astype(np.int64)
    ln = ln[ln >= 1]
    ours = compare_lognormal(ln, fit_powerlaw(ln, xmin=1))
    ref_R, ref_p = powerlaw.Fit(ln, discrete=True, xmin=1).distribution_compare(
        "power_law", "lognormal", normalized_ratio=True)
    assert np.sign(ours.normalized_R) == np.sign(ref_R) == -1
    assert ours.p < 0.01 and ref_p < 0.01
