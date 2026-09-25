"""C3: do Clauset-style tests separate the generating law where log-log R^2 does not?

Every analysis group is scored twice -- on the units TP segmentation discovers, and on the true word counts -- so the
effect of the segmentation step on each verdict is visible side by side. Groups are the ones C4 analyses:

    whale  8 years sized as Table S1, independent words; factor column `mean_repeats`
    finch  6 birds x 5 recordings x 10 bouts, calibrated primary configuration by default; `mean_repeats`
    song   8 whale years of song-structured streams (compose_song, generate_song); `stickiness`

    loglog_r2                           the published statistic
    csn_xmin, csn_alpha, csn_tail,      Clauset fit with KS-selected xmin; tail_fraction = csn_tail /
    tail_fraction                       number of unit types
    gof_p, gof_failed                   semi-parametric bootstrap p (Clauset et al. 2009, section 4.1)
    lr_R, lr_normalized_R, lr_p,        Vuong test against a discrete lognormal at the selected xmin
    lr_at_bound, lr_mu, lr_sigma, lr_gap

FINDINGS.md F4 applies to every lr_* value: at small xmin the ratio also compares discretisation conventions. Tail
fraction is reported beside every p-value because small tails are rarely rejected.

A statistic that cannot be computed is left empty, with the reason in `error`. Nothing is imputed.

The whale regime draws exactly the random numbers it drew before the finch and song regimes were added, so
results/c3_whale.csv reproduces.

Writes results/c3_<regime>.csv (one row per dataset, group and source) and a .json of run metadata.

Usage
-----
    python src/exp_c3_estimators.py --regime whale --datasets 30 --n-boot 100
    python src/exp_c3_estimators.py --regime finch --datasets 1 --n-boot 10 --laws zipf uniform   # smoke
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

import unitsim
from exp_c4_inference_unit import FINCH, LAWS, WHALE_YEARS, law_params, shape
from unitsim.generate import build_lexicon, compose_song, generate_song, generate_stream
from unitsim.metrics import DegenerateDistribution, loglog_r2_or_none, pooled_unit_counts
from unitsim.powerlaw_fit import compare_lognormal, fit_powerlaw, gof_pvalue
from unitsim.tp import segment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCES = ("discovered", "true")
REGIMES = ("whale", "finch", "song")
FACTOR = {"whale": "mean_repeats", "finch": "mean_repeats", "song": "stickiness"}
FIRST_YEAR = 2010


def _present(streams, n_words):
    counts = np.bincount(np.concatenate([s.tokens for s in streams]), minlength=n_words)
    return counts[counts > 0]


def simulate_year(lex, n_rec, n_elem, mean_repeats, rng):
    tokens = max(2, int(round(n_elem / n_rec / float(np.dot(lex.lengths, lex.probs)))))
    streams = [generate_stream(lex, tokens, rng=rng, mean_repeats=mean_repeats) for _ in range(n_rec)]
    return [s.elements for s in streams], _present(streams, lex.n_words)


def groups(regime, law, factor, rng, a):
    """Yield (element sequences, true word counts) for each analysis group of one dataset, in draw order."""
    sh = shape(a)
    if regime in ("whale", "song"):
        n_words = a.n_words if regime == "whale" else a.n_themes * a.variants_per_theme
        for n_rec, n_elem in WHALE_YEARS:
            lex = build_lexicon(n_words, a.n_elements, rng=rng, length_range=tuple(a.length_range),
                                brevity=False, **law_params(law, n_words, **sh))
            if regime == "whale":
                yield simulate_year(lex, n_rec, n_elem, factor, rng)
                continue
            song = compose_song(lex, a.n_themes, rng=rng)
            tokens = max(2, int(round(n_elem / n_rec / float(np.dot(lex.lengths, lex.probs)))))
            streams = [generate_song(lex, song, tokens, rng=rng, phrases_per_cycle=a.phrases_per_cycle,
                                     stickiness=factor, random_start=True) for _ in range(n_rec)]
            yield [s.elements for s in streams], _present(streams, lex.n_words)
        return
    for _ in range(FINCH["birds"]):
        lex = build_lexicon(a.finch_n_words, a.finch_n_elements, rng=rng,
                            length_range=tuple(a.finch_length_range), brevity=False,
                            **law_params(law, a.finch_n_words, **sh))
        for _ in range(FINCH["recordings_per_bird"]):
            streams = [generate_stream(lex, a.finch_tokens_per_bout, rng=rng, mean_repeats=factor)
                       for _ in range(FINCH["bouts_per_recording"])]
            yield [s.elements for s in streams], _present(streams, lex.n_words)


def csn_stats(counts, rng, n_boot, max_failed_fraction):
    out = {"n_types": int(counts.size), "loglog_r2": loglog_r2_or_none(counts),
           "csn_xmin": None, "csn_alpha": None, "csn_tail": None, "tail_fraction": None,
           "gof_p": None, "gof_failed": None, "lr_R": None, "lr_normalized_R": None, "lr_p": None,
           "lr_at_bound": None, "lr_mu": None, "lr_sigma": None, "lr_gap": None, "error": ""}
    try:
        fit = fit_powerlaw(counts)
    except DegenerateDistribution as e:
        out["error"] = f"fit: {e}"
        return out
    out.update(csn_xmin=fit.xmin, csn_alpha=fit.alpha, csn_tail=fit.n_tail,
               tail_fraction=fit.n_tail / counts.size)
    try:
        lr = compare_lognormal(counts, fit)
        out.update(lr_R=lr.R, lr_normalized_R=lr.normalized_R, lr_p=lr.p, lr_at_bound=lr.at_bound,
                   lr_mu=lr.mu, lr_sigma=lr.sigma, lr_gap=lr.optimisation_gap)
    except (DegenerateDistribution, RuntimeError, FloatingPointError) as e:
        out["error"] += f"lr: {e}; "
    try:
        g = gof_pvalue(counts, fit, rng=rng, n_boot=n_boot, max_failed_fraction=max_failed_fraction)
        out.update(gof_p=g.p, gof_failed=g.n_failed)
    except (RuntimeError, OverflowError) as e:   # OverflowError: alpha near 1 makes bootstrap draws unbounded
        out["error"] += f"gof: {e}; "
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=REGIMES, default="whale")
    ap.add_argument("--datasets", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--max-failed-fraction", type=float, default=0.05)
    ap.add_argument("--laws", nargs="+", choices=LAWS, default=list(LAWS))
    ap.add_argument("--repeats", type=float, nargs="+", default=None,
                    help="whale default 1 5; finch default 1")
    ap.add_argument("--stickiness", type=float, nargs="+", default=[0.0, 0.8])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-words", type=int, default=100)
    ap.add_argument("--n-elements", type=int, default=30)
    ap.add_argument("--length-range", type=int, nargs=2, default=[2, 6])
    ap.add_argument("--n-themes", type=int, default=8)
    ap.add_argument("--variants-per-theme", type=int, default=3)
    ap.add_argument("--phrases-per-cycle", type=float, default=40.0)
    ap.add_argument("--finch-n-words", type=int, default=10)
    ap.add_argument("--finch-n-elements", type=int, default=8)
    ap.add_argument("--finch-length-range", type=int, nargs=2, default=[2, 4])
    ap.add_argument("--finch-tokens-per-bout", type=int, default=15)
    ap.add_argument("--zipf-exponent", type=float, default=1.0)
    ap.add_argument("--lognormal-sigma", type=float, default=1.0)
    ap.add_argument("--geometric-tail", type=float, default=0.05)
    ap.add_argument("--base-seed", type=int, default=20260915)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(ROOT, "results", f"c3_{a.regime}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if a.regime == "song":
        factors = a.stickiness
    else:
        factors = a.repeats or ([1.0, 5.0] if a.regime == "whale" else [1.0])
    factor_name = FACTOR[a.regime]

    resources.report("resources: ")
    t0 = time.time()
    rows = []
    conditions = [(law, f, d) for law in a.laws for f in factors for d in range(a.datasets)]
    for k, (law, f, d) in enumerate(conditions, 1):
        tc = time.time()
        if a.regime == "whale":
            key = [a.base_seed, LAWS.index(law), int(round(f * 100)), d]
        else:
            key = [a.base_seed, 100 + REGIMES.index(a.regime), LAWS.index(law), int(round(f * 100)), d]
        rng = np.random.default_rng(key)
        start = len(rows)
        for g, (seqs, true_counts) in enumerate(groups(a.regime, law, f, rng, a)):
            discovered = pooled_unit_counts(seqs, segment(seqs, threshold=a.threshold))[1]
            for s, counts in zip(SOURCES, (discovered, true_counts)):
                brng = np.random.default_rng(key + [g, SOURCES.index(s)])  # bootstrap never touches data
                rows.append({"regime": a.regime, "law": law, factor_name: f, "dataset": d, "group": g,
                             "year": None if a.regime == "finch" else FIRST_YEAR + g, "source": s,
                             **shape(a), **csn_stats(counts, brng, a.n_boot, a.max_failed_fraction)})
        errors = sum(bool(row["error"]) for row in rows[start:])
        print(f"[{k}/{len(conditions)}] {law:9s} {factor_name}={f:g} d={d}  errors={errors}  "
              f"{time.time() - tc:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    meta = {"script": os.path.basename(__file__), "args": vars(a), "whale_years": WHALE_YEARS,
            "finch": FINCH, "unitsim_version": unitsim.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "runtime_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(df)}
    with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out_path} ({len(df)} rows) in {meta['runtime_s']}s")
    cols = ["loglog_r2", "tail_fraction", "gof_p", "lr_normalized_R"]
    print(df.groupby(["source", "law", factor_name])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
