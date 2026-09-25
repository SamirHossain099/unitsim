"""Discrete power-law fitting after Clauset, Shalizi & Newman (2009), SIAM Review 51:661.

What a log-log R^2 cannot do, this does: estimate the exponent by maximum likelihood, choose the
lower cutoff xmin by minimising the Kolmogorov-Smirnov distance, test goodness of fit by
semi-parametric bootstrap, and compare the power law against a lognormal with Vuong's
likelihood-ratio test.

Input is the count of each type -- how many times each distinct unit occurred. A Zipf rank-frequency
exponent `a` corresponds to a frequency-distribution exponent alpha = 1 + 1/a, so Zipf's law (a = 1)
is alpha = 2.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erfc, zeta
from scipy.stats import norm

from .metrics import DegenerateDistribution

ALPHA_BOUNDS = (1.0 + 1e-6, 10.0)


@dataclass(frozen=True)
class PowerLawFit:
    alpha: float
    xmin: int
    ks: float
    n_tail: int
    n: int
    xmin_fixed: bool


def _as_counts(x):
    a = np.asarray(x)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("expected a non-empty 1-D array of counts")
    if not np.issubdtype(a.dtype, np.integer):
        if (not np.issubdtype(a.dtype, np.floating) or not np.all(np.isfinite(a))
                or np.any(a != np.round(a))):
            raise ValueError("counts must be integers")
    a = a.astype(np.int64)
    if np.any(a < 1):
        raise ValueError("counts must be >= 1; drop zero-count types first")
    return a


def _nll(alpha, log_sum, n, xmin):
    return n * np.log(zeta(alpha, xmin)) + alpha * log_sum


def fit_alpha(x, xmin):
    """Maximum-likelihood exponent of a discrete power law on [xmin, inf).

    log zeta(alpha, xmin) is convex in alpha, so the negative log-likelihood is convex and a bounded
    scalar search finds the global optimum.
    """
    x = _as_counts(x)
    xmin = int(xmin)
    if xmin < 1:
        raise ValueError(f"xmin must be >= 1, got {xmin}")
    tail = x[x >= xmin]
    if tail.size == 0:
        raise DegenerateDistribution(f"no values at or above xmin={xmin}")
    if np.all(tail == xmin):
        raise DegenerateDistribution(f"every tail value equals xmin={xmin}; the MLE diverges")
    res = minimize_scalar(_nll, bounds=ALPHA_BOUNDS, method="bounded",
                          args=(float(np.log(tail).sum()), tail.size, xmin),
                          options={"xatol": 1e-10})
    alpha = float(res.x)
    if not (res.success and np.isfinite(alpha)):
        raise RuntimeError(f"alpha optimisation failed: {res.message}")
    if alpha > ALPHA_BOUNDS[1] - 1e-3:
        raise DegenerateDistribution(
            f"alpha reached the upper bound {ALPHA_BOUNDS[1]}; the tail is too light to fit")
    return alpha


def _cdf(alpha, xs, xmin):
    """P(X <= xs) for a discrete power law on [xmin, inf)."""
    return 1.0 - zeta(alpha, np.asarray(xs, dtype=float) + 1.0) / zeta(alpha, xmin)


def ks_distance(x, alpha, xmin):
    """Largest gap between the empirical and model CDFs over the tail.

    Both CDFs are step functions on the integers. Between observed values u_i < u_{i+1} the empirical
    CDF is flat while the model CDF keeps rising, so the gap can peak at u_{i+1} - 1, which is not a
    data point. Evaluating only at observed values, as some implementations do, can miss it.
    """
    x = np.asarray(x)
    tail = x[x >= xmin]
    if tail.size == 0:
        raise DegenerateDistribution(f"no values at or above xmin={xmin}")
    uniq, cnt = np.unique(tail, return_counts=True)
    emp = np.cumsum(cnt) / tail.size
    gaps = [np.abs(emp - _cdf(alpha, uniq, xmin))]
    before = uniq[1:] - 1
    has_gap = before > uniq[:-1]
    if has_gap.any():
        gaps.append(np.abs(emp[:-1][has_gap] - _cdf(alpha, before[has_gap], xmin)))
    if uniq[0] > xmin:
        gaps.append(np.atleast_1d(_cdf(alpha, uniq[0] - 1, xmin)))
    return float(max(g.max() for g in gaps))


def fit_powerlaw(x, *, xmin=None, min_tail=10):
    """Fit alpha by MLE. If `xmin` is None, choose it by minimising the KS distance.

    Candidate cutoffs are the distinct observed values that leave at least `min_tail` values in the
    tail. Candidates whose tail cannot be fitted (all values equal) are skipped.
    """
    x = _as_counts(x)
    if xmin is not None:
        alpha = fit_alpha(x, xmin)
        return PowerLawFit(alpha, int(xmin), ks_distance(x, alpha, xmin),
                           int(np.sum(x >= xmin)), int(x.size), True)
    best = None
    for c in np.unique(x):
        tail = x[x >= c]
        if tail.size < min_tail:
            break
        try:
            a = fit_alpha(tail, c)
        except DegenerateDistribution:
            continue
        d = ks_distance(tail, a, c)
        if best is None or d < best.ks:
            best = PowerLawFit(a, int(c), d, int(tail.size), int(x.size), False)
    if best is None:
        raise DegenerateDistribution(
            f"no xmin leaves a fittable tail of at least {min_tail} values (n={x.size})")
    return best


def sample_powerlaw(size, alpha, xmin, *, rng, max_value=1e15):
    """Exact draws from a discrete power law on [xmin, inf) by inverse CDF.

    Uses P(X >= x) = zeta(alpha, x) / zeta(alpha, xmin): bracket by doubling, then bisect on the
    integers. The closed-form approximation of Clauset et al. (2009, appendix D) is poor at xmin = 1,
    the usual case for repertoire counts.
    """
    size = int(size)
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    if not alpha > 1:
        raise ValueError(f"alpha must be > 1, got {alpha!r}")
    if int(xmin) < 1:
        raise ValueError(f"xmin must be >= 1, got {xmin!r}")
    if size == 0:
        return np.empty(0, dtype=np.int64)
    u = rng.random(size)
    z0 = zeta(alpha, xmin)
    lo = np.full(size, float(xmin))
    hi = np.full(size, float(xmin))
    active = np.ones(size, dtype=bool)  # invariant for active entries: S(hi) >= u
    while active.any():
        lo[active] = hi[active]
        hi[active] *= 2.0
        if hi.max() > max_value:
            raise OverflowError(
                f"a draw exceeded {max_value:g}; alpha={alpha} is too close to 1 for this size")
        active = zeta(alpha, hi) / z0 >= u
    # now S(lo) >= u > S(hi) everywhere; bisect to the largest integer x with S(x) >= u
    while True:
        need = hi - lo > 1
        if not need.any():
            break
        mid = np.floor((lo + hi) / 2.0)
        ge = zeta(alpha, mid) / z0 >= u
        lo = np.where(need & ge, mid, lo)
        hi = np.where(need & ~ge, mid, hi)
    return lo.astype(np.int64)


@dataclass(frozen=True)
class GoodnessOfFit:
    p: float
    n_boot: int
    n_failed: int


def gof_pvalue(x, fit, *, rng, n_boot=500, min_tail=10, max_failed_fraction=0.0):
    """Semi-parametric bootstrap p-value (Clauset et al. 2009, section 4.1).

    Each replicate draws the tail from the fitted power law and the body, with replacement, from the
    observed values below xmin, then repeats the SAME fitting procedure: xmin is re-selected if the
    original fit selected it and held fixed if it was fixed. p is the share of replicates whose KS
    distance is at least the observed one. Small p rejects the power law. Large p means plausible,
    not correct.
    """
    x = _as_counts(x)
    if not np.isfinite(fit.ks):
        raise ValueError("observed KS distance is not finite; a p-value built on it is meaningless")
    if x.size != fit.n:
        raise ValueError(f"fit was made on {fit.n} values but {x.size} were passed")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    below = x[x < fit.xmin]
    exceed = failed = 0
    for _ in range(n_boot):
        n_tail = int(rng.binomial(fit.n, fit.n_tail / fit.n))
        tail = sample_powerlaw(n_tail, fit.alpha, fit.xmin, rng=rng)
        if fit.n > n_tail:
            body = rng.choice(below, size=fit.n - n_tail, replace=True)
        else:
            body = np.empty(0, dtype=np.int64)
        try:
            f = fit_powerlaw(np.concatenate([body, tail]),
                             xmin=fit.xmin if fit.xmin_fixed else None, min_tail=min_tail)
        except DegenerateDistribution:
            failed += 1
            continue
        if not np.isfinite(f.ks):
            raise FloatingPointError("a bootstrap replicate produced a non-finite KS distance")
        exceed += int(f.ks >= fit.ks)
    valid = n_boot - failed
    if valid == 0 or failed > max_failed_fraction * n_boot:
        raise RuntimeError(
            f"{failed} of {n_boot} bootstrap replicates could not be fitted (allowed fraction "
            f"{max_failed_fraction}); the p-value would be biased")
    return GoodnessOfFit(exceed / valid, valid, failed)


@dataclass(frozen=True)
class LikelihoodRatio:
    R: float             # log-likelihood ratio, power law minus lognormal; > 0 favours power law
    normalized_R: float
    p: float             # Vuong two-sided p-value for the sign of R
    mu: float
    sigma: float
    at_bound: bool             # lognormal MLE sits on a parameter bound (see compare_lognormal)
    optimisation_gap: float    # log-likelihood gained by the last Nelder-Mead polish; < POLISH_TOL


def _log_interval(a, b):
    """log(Phi(b) - Phi(a)) for a < b elementwise, stable in both tails."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.empty(a.shape)
    left = b <= 0
    right = a >= 0
    mid = ~(left | right)
    lb, la = norm.logcdf(b[left]), norm.logcdf(a[left])
    out[left] = lb + np.log1p(-np.exp(la - lb))
    sa, sb = norm.logsf(a[right]), norm.logsf(b[right])
    out[right] = sa + np.log1p(-np.exp(sb - sa))
    out[mid] = np.log(norm.cdf(b[mid]) - norm.cdf(a[mid]))
    return out


def _lognormal_logpmf(x, mu, sigma, xmin):
    """Discrete lognormal on [xmin, inf): P(X = k) proportional to F(k+1) - F(k), F lognormal CDF.

    This is exactly the law of floor(L) for L lognormal, truncated at xmin.
    """
    x = np.asarray(x, dtype=float)
    a = (np.log(x) - mu) / sigma
    b = (np.log(x + 1.0) - mu) / sigma
    return _log_interval(a, b) - norm.logsf((np.log(xmin) - mu) / sigma)


LOG_SIGMA_BOUNDS = (float(np.log(0.01)), float(np.log(1e4)))
MU_LOWER = -1e6
POLISH_TOL = 1e-2      # log-likelihood; an error this size moves R by less than 0.01
POLISH_ROUNDS = 5


def compare_lognormal(x, fit):
    """Vuong likelihood-ratio test of the fitted power law against a discrete lognormal.

    Both are fitted by maximum likelihood on the same tail (x >= fit.xmin). R > 0 favours the power
    law; p is the probability of |R| this large if the two fit equally well.

    On heavy-tailed data the lognormal likelihood can keep improving as mu falls and sigma grows with
    mu / sigma^2 held near a constant c. That ridge leads to a *binned* power law, P(k) proportional to
    the integral of x^(c-1) over [k, k+1] -- not to the discrete power law k^-alpha fitted here. The two
    differ most at small k, so at xmin = 1 this test also compares discretisation conventions, and R can
    stay clearly positive at the lognormal's supremum (FINDINGS.md F4).

    The fit is bounded (mu >= -1e6, sigma <= 1e4) and started from three points, one on that ridge. The
    ridge is nearly flat, so a gradient optimiser can stop short of the supremum -- and can also report
    success at a point measurably worse than the ridge optimum (FINDINGS.md F4). Optimiser flags are
    therefore not trusted: the best start is polished with Nelder-Mead until one more polish gains less
    than POLISH_TOL in log-likelihood, and `optimisation_gap` reports that last gain. `at_bound` reports
    an optimum on a bound. A fit still improving after POLISH_ROUNDS polishes, away from any bound, raises.
    """
    x = _as_counts(x)
    tail = x[x >= fit.xmin].astype(float)
    n = tail.size
    logs = np.log(tail)
    mu_upper = float(logs.max()) + 10.0
    bounds = [(MU_LOWER, mu_upper), LOG_SIGMA_BOUNDS]

    def nll(theta):
        val = -_lognormal_logpmf(tail, theta[0], np.exp(theta[1]), fit.xmin).sum()
        return val if np.isfinite(val) else 1e300

    starts = [
        [float(np.clip(logs.mean(), MU_LOWER, mu_upper)),
         float(np.clip(np.log(max(logs.std(), 1e-3)), *LOG_SIGMA_BOUNDS))],
        [float(np.log(fit.xmin)) - 5.0, float(np.log(3.0))],  # heavy-tailed: mass mostly below xmin
        [-1e4, float(np.log(100.0))],                         # on the ridge, mu / sigma^2 = -1
    ]
    best = None
    for s in starts:
        r = minimize(nll, x0=s, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    if not np.isfinite(best.fun) or best.fun >= 1e300:
        raise RuntimeError(f"lognormal fit produced no finite likelihood: {best.message}")

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    theta, value = np.clip(best.x, lo, hi), float(nll(np.clip(best.x, lo, hi)))
    gap = np.inf
    for _ in range(POLISH_ROUNDS):
        p = minimize(nll, x0=theta, method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
        candidate = np.clip(p.x, lo, hi)
        candidate_value = float(nll(candidate))     # re-evaluated after clipping, not trusted
        gap = max(value - candidate_value, 0.0)
        if candidate_value < value:
            theta, value = candidate, candidate_value
        if gap < POLISH_TOL:
            break
    mu, log_sigma = float(theta[0]), float(theta[1])
    tol = 1e-6
    at_bound = (mu <= MU_LOWER + tol or mu >= mu_upper - tol
                or log_sigma <= LOG_SIGMA_BOUNDS[0] + tol or log_sigma >= LOG_SIGMA_BOUNDS[1] - tol)
    if gap >= POLISH_TOL and not at_bound:
        raise RuntimeError(
            f"lognormal fit still gaining {gap:.3g} log-likelihood after {POLISH_ROUNDS} polishes, "
            "away from any bound")
    sigma = float(np.exp(log_sigma))

    l_pl = -fit.alpha * logs - np.log(zeta(fit.alpha, fit.xmin))
    l_ln = _lognormal_logpmf(tail, mu, sigma, fit.xmin)
    d = l_pl - l_ln
    if not np.all(np.isfinite(d)):
        raise FloatingPointError("non-finite per-point log-likelihood ratio")
    R, sd = float(d.sum()), float(d.std())
    if sd == 0:
        raise DegenerateDistribution("the two models give identical per-point likelihoods")
    return LikelihoodRatio(R, R / (sd * np.sqrt(n)), float(erfc(abs(R) / (np.sqrt(2 * n) * sd))),
                           mu, sigma, bool(at_bound), float(gap))
