"""C2: do element-label errors at empirically anchored rates move the published statistics by more than
seed-to-seed variability?

Anchors (VERIFICATION.md, error anchors): humpback element classification has a 19.19% random-forest
out-of-bag error (Arnon et al. 2025 supplement); finch coders agreed 93.3% of the time, so disagreed 6.7%
(Kirby et al. 2026). Neither is literally a per-token confusion rate. They set the scale of plausible error
and are labelled as anchors, not measurements of these simulated data.

Design: whale inference unit, 8 years sized as Table S1. Each simulated dataset's element sequences are
analysed five ways -- clean labels, 6.7% confusion, 19.19% confusion, 25% of element types over-split, 25%
under-split -- so every shift is paired within a dataset. Over- and under-splitting map the same types in
every recording of a year. Statistics are computed per year and averaged over years, as published:

    loglog_r2, loglog_slope     rank-frequency fit of the discovered units (the published R^2)
    brevity_r2, brevity_slope   unit length against log frequency (the published whale brevity R^2 of 0.62 was withdrawn by erratum; VERIFICATION.md)
    csn_alpha                   Clauset frequency-distribution exponent of discovered-unit counts
    boundary_f1                 recovered boundaries against the true word boundaries

The noise floor for each statistic is its between-dataset sd under clean labels in the same cell. A shift is
only ever reported against it (shared lesson 2). Years where a statistic is undefined are skipped in that
statistic's mean, and the count is recorded.

Writes results/c2_whale.csv (one row per dataset and condition) and .json.

Usage
-----
    python src/exp_c2_label_error.py --datasets 30
    python src/exp_c2_label_error.py --datasets 2      # smoke
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
from exp_c4_inference_unit import LAWS, WHALE_YEARS, law_params
from unitsim.baselines import mean_over_groups
from unitsim.corrupt import confuse, oversplit, undersplit
from unitsim.generate import build_lexicon, generate_stream
from unitsim.metrics import (
    DegenerateDistribution,
    boundary_prf,
    brevity_fit,
    loglog_r2,
    pooled_unit_counts,
)
from unitsim.powerlaw_fit import fit_powerlaw
from unitsim.tp import segment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONDITIONS = ("clean", "confuse_6.7", "confuse_19.19", "oversplit_25", "undersplit_25")
STATS = ("loglog_r2", "loglog_slope", "brevity_r2", "brevity_slope", "csn_alpha", "boundary_f1")


def simulate_years(law, brevity, mean_repeats, rng, a):
    years = []
    for n_rec, n_elem in WHALE_YEARS:
        lex = build_lexicon(a.n_words, a.n_elements, rng=rng, length_range=tuple(a.length_range),
                            brevity=brevity, **law_params(law, a.n_words))
        tokens = max(2, int(round(n_elem / n_rec / float(np.dot(lex.lengths, lex.probs)))))
        streams = [generate_stream(lex, tokens, rng=rng, mean_repeats=mean_repeats)
                   for _ in range(n_rec)]
        years.append({"seqs": [s.elements for s in streams], "truth": [s.starts for s in streams]})
    return years


def corrupt_year(seqs, n_elements, condition, rng):
    if condition == "clean":
        return seqs
    if condition.startswith("confuse_"):
        rate = float(condition.split("_")[1]) / 100.0
        return [confuse(s, rate, n_elements, rng=rng) for s in seqs]
    lengths = [s.size for s in seqs]
    split = oversplit if condition.startswith("oversplit_") else undersplit
    labels, _, _ = split(np.concatenate(seqs), float(condition.split("_")[1]) / 100.0, n_elements,
                         rng=rng)
    return np.split(labels, np.cumsum(lengths)[:-1])


def year_stats(seqs, truth, threshold):
    starts = segment(seqs, threshold=threshold)
    types, counts = pooled_unit_counts(seqs, starts)
    out = dict.fromkeys(STATS)
    try:
        f = loglog_r2(counts)
        out["loglog_r2"], out["loglog_slope"] = f.r2, f.slope
    except DegenerateDistribution:
        pass
    try:
        b = brevity_fit(types, counts)
        out["brevity_r2"], out["brevity_slope"] = b.r2, b.slope
    except DegenerateDistribution:
        pass
    try:
        out["csn_alpha"] = fit_powerlaw(counts).alpha
    except DegenerateDistribution:
        pass
    out["boundary_f1"] = float(np.mean([boundary_prf(p, t).f1 for p, t in zip(starts, truth)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", type=int, default=30)
    ap.add_argument("--laws", nargs="+", choices=LAWS, default=["zipf", "uniform"])
    ap.add_argument("--brevity", type=int, nargs="+", choices=[0, 1], default=[1, 0])
    ap.add_argument("--repeats", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-words", type=int, default=100)
    ap.add_argument("--n-elements", type=int, default=30)
    ap.add_argument("--length-range", type=int, nargs=2, default=[2, 6])
    ap.add_argument("--base-seed", type=int, default=20260914)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "c2_whale.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    resources.report("resources: ")
    t0 = time.time()
    rows = []
    cells = [(law, bool(bv), r) for law in a.laws for bv in a.brevity for r in a.repeats]
    for k, (law, brevity, r) in enumerate(cells, 1):
        tc = time.time()
        for d in range(a.datasets):
            key = [a.base_seed, LAWS.index(law), int(brevity), int(round(r * 100)), d]
            years = simulate_years(law, brevity, r, np.random.default_rng(key), a)
            for ci, condition in enumerate(CONDITIONS):
                crng = np.random.default_rng(key + [ci])      # corruption draws never touch the data
                per_year = [year_stats(corrupt_year(y["seqs"], a.n_elements, condition, crng),
                                       y["truth"], a.threshold) for y in years]
                row = {"law": law, "brevity": brevity, "mean_repeats": r, "dataset": d,
                       "condition": condition}
                for s in STATS:
                    agg = mean_over_groups([py[s] for py in per_year], "skip")
                    row[s] = agg.mean
                    row[f"{s}_undefined_years"] = agg.n_undefined
                rows.append(row)
        print(f"[{k}/{len(cells)}] law={law:8s} brevity={brevity!s:5s} repeats={r:g}  "
              f"{time.time() - tc:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    meta = {"script": os.path.basename(__file__), "args": vars(a), "conditions": CONDITIONS,
            "whale_years": WHALE_YEARS, "unitsim_version": unitsim.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "runtime_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": len(df)}
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {a.out} ({len(df)} rows) in {meta['runtime_s']}s")

    keys = ["law", "brevity", "mean_repeats", "dataset"]
    clean = df[df.condition == "clean"].set_index(keys)[list(STATS)]
    floor = clean.groupby(level=["law", "brevity", "mean_repeats"]).std(ddof=1)
    print("\n|mean paired shift| / clean between-dataset sd (above 1 = shift exceeds the noise floor):")
    for condition in CONDITIONS[1:]:
        shifted = df[df.condition == condition].set_index(keys)[list(STATS)]
        mean_shift = (shifted - clean).groupby(level=["law", "brevity", "mean_repeats"]).mean()
        print(f"\n  {condition}")
        print((mean_shift.abs() / floor).round(2).to_string())


if __name__ == "__main__":
    main()
