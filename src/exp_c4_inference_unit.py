"""C4 at the published inference unit: can the published baselines tell a Zipfian source from a
non-Zipfian one?

Each simulated DATASET stands in for one whole published study and is analysed the way that study was.

whale  8 years sized exactly as Table S1 of Arnon et al. 2025: recordings and sound elements per year.
       A fresh lexicon per year, since song content turns over between years. TPs pooled over a year's
       recordings, units never crossing a recording, R^2 per year, statistic = mean R^2 over 8 years.
       Baseline replicates: every recording shuffled; cuts rotated with one offset per year; every
       recording regenerated from its own bigrams (the whale study did not run this last one).
finch  6 birds x 5 recordings x 10 bouts, as Kirby et al. 2026. One lexicon per bird over 8 syllables,
       shared by that bird's recordings. TPs pooled over a recording's bouts, units never crossing a
       bout, R^2 per recording, statistic = mean over 30 recordings. Baseline replicates: shuffle within
       each bout, rotate per bout with wrap-around, bigram per bout. Finch bout length and lexicon size
       are not reported: calibrate with src/calibrate_finch.py before trusting a finch result.

The statistic and every baseline are means over the dataset's years or recordings, so the z written
here is comparable in construction to the published z.

Undefined R^2 (FINDINGS.md F2). A group's log-log R^2 does not exist when every discovered unit occurs
equally often, which shuffled groups produce at small corpora. The published code is not available, so
--undefined must be stated: raise, skip (average the defined groups, as a pandas mean with skipna), or
zero. Every row records how often groups were undefined, so the choice's effect can be measured.

Writes results/c4_<regime>_<policy>.csv (one row per dataset) and a .json of run metadata. Besides the
statistic and the baselines, each row carries the mean log-log slope and mean number of unit types over the
dataset's years or recordings, for the calibration step (CALIBRATION-DESIGN.md).

Usage
-----
    python src/exp_c4_inference_unit.py --regime whale --undefined skip --datasets 30 --replicates 200
    python src/exp_c4_inference_unit.py --regime whale --undefined zero --datasets 1 --replicates 20
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
from unitsim.baselines import (
    UNDEFINED_POLICIES,
    bigram_pseudo,
    compare_to_baseline,
    mean_over_groups,
    rotate_cuts,
    shuffle_within,
)
from unitsim.generate import build_lexicon, generate_stream
from unitsim.metrics import (
    DegenerateDistribution,
    boundary_prf,
    loglog_r2,
    loglog_r2_or_none,
    pooled_unit_counts,
)
from unitsim.tp import segment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: (recordings, sound elements) per year, 2010-2017. Arnon et al. 2025, Table S1.
WHALE_YEARS = ((4, 1053), (6, 2028), (6, 3351), (6, 2664), (7, 5765), (12, 13251), (7, 8469), (7, 9179))
#: Kirby et al. 2026: 6 birds x 5 recordings; 299 bouts over 30 recordings.
FINCH = {"birds": 6, "recordings_per_bird": 5, "bouts_per_recording": 10}
LAWS = ("zipf", "uniform", "lognormal", "geometric")
REGIMES = ("whale", "finch")
BASELINES = ("shuffle", "rotate", "bigram")


def law_params(law, n_words, *, zipf_exponent=1.0, lognormal_sigma=1.0, geometric_tail=0.05):
    """Generator parameters for a law. Defaults are the settings of every run before the 2026-09-12 sweeps."""
    if law == "zipf":
        return {"law": "zipf", "exponent": float(zipf_exponent)}
    if law == "uniform":
        return {"law": "uniform"}
    if law == "lognormal":
        return {"law": "lognormal", "sigma": float(lognormal_sigma)}
    if law == "geometric":
        # least frequent word `geometric_tail` times as likely as the most frequent, whatever the lexicon size
        return {"law": "geometric", "ratio": float(geometric_tail ** (1.0 / (n_words - 1)))}
    raise ValueError(f"unknown law {law!r}")


def shape(a):
    """A run's law-shape settings, falling back to the defaults for callers whose arguments lack them."""
    return {"zipf_exponent": getattr(a, "zipf_exponent", 1.0),
            "lognormal_sigma": getattr(a, "lognormal_sigma", 1.0),
            "geometric_tail": getattr(a, "geometric_tail", 0.05)}


def simulate_whale(law, mean_repeats, rng, a):
    groups = []
    for n_rec, n_elem in WHALE_YEARS:
        lex = build_lexicon(a.n_words, a.n_elements, rng=rng, length_range=tuple(a.length_range),
                            brevity=False, **law_params(law, a.n_words, **shape(a)))
        expected_len = float(np.dot(lex.lengths, lex.probs))
        tokens = max(2, int(round(n_elem / n_rec / expected_len)))
        streams = [generate_stream(lex, tokens, rng=rng, mean_repeats=mean_repeats)
                   for _ in range(n_rec)]
        groups.append({"seqs": [s.elements for s in streams], "truth": [s.starts for s in streams]})
    return groups


def simulate_finch(law, mean_repeats, rng, a):
    groups = []
    for _ in range(FINCH["birds"]):
        lex = build_lexicon(a.finch_n_words, a.finch_n_elements, rng=rng,
                            length_range=tuple(a.finch_length_range), brevity=False,
                            **law_params(law, a.finch_n_words, **shape(a)))
        for _ in range(FINCH["recordings_per_bird"]):
            streams = [generate_stream(lex, a.finch_tokens_per_bout, rng=rng,
                                       mean_repeats=mean_repeats)
                       for _ in range(FINCH["bouts_per_recording"])]
            groups.append({"seqs": [s.elements for s in streams],
                           "truth": [s.starts for s in streams]})
    return groups


def group_r2(seqs, threshold):
    starts = segment(seqs, threshold=threshold)
    return loglog_r2_or_none(pooled_unit_counts(seqs, starts)[1]), starts


def analyse(groups, regime, threshold, replicates, undefined, rng):
    real, slopes, n_types, starts_all, f1, cuts = [], [], [], [], [], []
    for g in groups:
        st = segment(g["seqs"], threshold=threshold)       # exactly what group_r2 does, kept apart so the
        counts = pooled_unit_counts(g["seqs"], st)[1]      # counts also yield slope and type count
        r2 = loglog_r2_or_none(counts)
        real.append(r2)
        slopes.append(None if r2 is None else loglog_r2(counts).slope)
        n_types.append(float(counts.size))
        starts_all.append(st)
        f1.extend(boundary_prf(p, t).f1 for p, t in zip(st, g["truth"]))
        cuts.extend(int(p[1:].sum()) for p in st)
    obs = mean_over_groups(real, undefined)
    if obs.mean is None:
        raise DegenerateDistribution("no group of the observed dataset has a defined R^2")
    # slope has no meaningful value to stand in for an undefined group, so it always skips them
    row = {"observed_mean_r2": obs.mean, "observed_undefined_groups": obs.n_undefined,
           "observed_mean_slope": mean_over_groups(slopes, "skip").mean,
           "observed_mean_unit_types": float(np.mean(n_types)),
           "boundary_f1": float(np.mean(f1)), "cuts_per_sequence": float(np.mean(cuts)),
           "elements_total": int(sum(s.size for g in groups for s in g["seqs"])),
           "n_groups": len(groups)}

    per_sequence_rotation = regime == "finch"   # finch: per bout; whale: one offset per year
    pseudo = [[bigram_pseudo(s, replicates, rng=rng) for s in g["seqs"]] for g in groups]
    samples = {name: [] for name in BASELINES}
    undefined_groups = dict.fromkeys(BASELINES, 0)
    dropped = dict.fromkeys(BASELINES, 0)
    for b in range(replicates):
        evaluations = {
            "shuffle": [group_r2(shuffle_within(g["seqs"], rng=rng), threshold)[0] for g in groups],
            "rotate": [loglog_r2_or_none(pooled_unit_counts(
                g["seqs"], rotate_cuts(st, rng=rng, per_sequence=per_sequence_rotation))[1])
                for g, st in zip(groups, starts_all)],
            "bigram": [group_r2([p[b] for p in ps], threshold)[0] for ps in pseudo],
        }
        for name, values in evaluations.items():
            agg = mean_over_groups(values, undefined)
            undefined_groups[name] += agg.n_undefined
            if agg.mean is None:
                dropped[name] += 1
            else:
                samples[name].append(agg.mean)

    for name in BASELINES:
        if len(samples[name]) < 2:
            raise DegenerateDistribution(
                f"{name}: fewer than 2 of {replicates} replicates have a defined mean")
        c = compare_to_baseline(obs.mean, samples[name])
        row.update({f"{name}_mean": c.baseline_mean, f"{name}_sd": c.baseline_sd,
                    f"{name}_z": c.z, f"{name}_p": c.p_empirical,
                    f"{name}_undefined_rate": undefined_groups[name] / (replicates * len(groups)),
                    f"{name}_dropped_replicates": dropped[name]})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=REGIMES, required=True)
    ap.add_argument("--undefined", choices=UNDEFINED_POLICIES, required=True,
                    help="how a group with an undefined R^2 enters a mean (FINDINGS.md F2)")
    ap.add_argument("--datasets", type=int, default=30)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--laws", nargs="+", choices=LAWS, default=list(LAWS))
    ap.add_argument("--repeats", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--n-words", type=int, default=100)
    ap.add_argument("--n-elements", type=int, default=30)
    ap.add_argument("--length-range", type=int, nargs=2, default=[2, 6])
    ap.add_argument("--finch-n-words", type=int, default=15)
    ap.add_argument("--finch-n-elements", type=int, default=8)
    ap.add_argument("--finch-length-range", type=int, nargs=2, default=[2, 4])
    ap.add_argument("--finch-tokens-per-bout", type=int, default=10)
    ap.add_argument("--zipf-exponent", type=float, default=1.0)
    ap.add_argument("--lognormal-sigma", type=float, default=1.0)
    ap.add_argument("--geometric-tail", type=float, default=0.05,
                    help="least frequent word's probability relative to the most frequent")
    ap.add_argument("--base-seed", type=int, default=20260912)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", f"c4_{a.regime}_{a.undefined}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    simulate = simulate_whale if a.regime == "whale" else simulate_finch
    resources.report("resources: ")
    t0 = time.time()
    rows = []
    conditions = [(law, r, d) for law in a.laws for r in a.repeats for d in range(a.datasets)]
    for k, (law, r, d) in enumerate(conditions, 1):
        tc = time.time()
        rng = np.random.default_rng([a.base_seed, REGIMES.index(a.regime), LAWS.index(law),
                                     int(round(r * 100)), d])
        groups = simulate(law, r, rng, a)
        row = {"regime": a.regime, "undefined_policy": a.undefined, "law": law,
               "mean_repeats": r, "dataset": d, "threshold": a.threshold, **shape(a)}
        row.update(analyse(groups, a.regime, a.threshold, a.replicates, a.undefined, rng))
        rows.append(row)
        print(f"[{k}/{len(conditions)}] {law:9s} repeats={r:g} d={d}  "
              f"R2={row['observed_mean_r2']:.3f}  z(shuf/rot/bi)="
              f"{row['shuffle_z']:.1f}/{row['rotate_z']:.1f}/{row['bigram_z']:.1f}  "
              f"undef(shuf/rot/bi)={row['shuffle_undefined_rate']:.2f}/"
              f"{row['rotate_undefined_rate']:.2f}/{row['bigram_undefined_rate']:.2f}  "
              f"cuts/seq={row['cuts_per_sequence']:.1f}  {time.time() - tc:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = {"script": os.path.basename(__file__), "args": vars(a), "whale_years": WHALE_YEARS,
            "finch": FINCH, "unitsim_version": unitsim.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "runtime_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(df)}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out} ({len(df)} rows) in {meta['runtime_s']}s")
    cols = ["observed_mean_r2", "boundary_f1", "cuts_per_sequence", "shuffle_z", "rotate_z",
            "bigram_z", "shuffle_undefined_rate"]
    print(df.groupby(["law", "mean_repeats"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
