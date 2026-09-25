"""Pilot for claims C1 and C4 of the accompanying paper. Sizes the full experiment; not for quoting.

C1  Does TP segmentation return a Zipf-like log-log R^2 from structured sources whose word
    frequencies are not Zipfian?
C4  Do such sources pass the published shuffled, rotated and bigram baselines?

Sources differ only in the word-frequency law. Every other parameter is held fixed and sized to the
humpback corpus (Arnon et al. 2025, Table S1): one simulated year is 7 recordings of about 800
elements, 30 element types, word lengths 2-6. Each condition is one simulated year, analysed as the
whale paper analyses a year: TPs pooled over its recordings, units never crossing a recording boundary.

Writes results/pilot_c1_c4.csv (one row per condition and threshold) and a .json of run metadata.

Usage
-----
    python src/exp_pilot_c1_c4.py                       # defaults below
    python src/exp_pilot_c1_c4.py --seeds 2 --replicates 50   # smoke run
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
from unitsim.baselines import bigram_pseudo, compare_to_baseline, rotate_cuts, shuffle_within
from unitsim.generate import build_lexicon, generate_stream
from unitsim.metrics import boundary_prf, loglog_r2, pooled_unit_counts
from unitsim.tp import segment

LAWS = {
    "zipf": {"law": "zipf", "exponent": 1.0},
    "uniform": {"law": "uniform"},
    "lognormal": {"law": "lognormal", "sigma": 1.0},
    "geometric": {"law": "geometric", "ratio": 0.97},
}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pipeline_r2(seqs, threshold):
    starts = segment(seqs, threshold=threshold)
    _, counts = pooled_unit_counts(seqs, starts)
    return loglog_r2(counts).r2, starts, counts


def run_condition(law_name, mean_repeats, rep, a, law_index, repeats_index):
    rng = np.random.default_rng([a.base_seed, law_index, repeats_index, rep])
    lex = build_lexicon(a.n_words, a.n_elements, rng=rng, length_range=(2, 6), brevity=False,
                        **LAWS[law_name])
    streams = [generate_stream(lex, a.tokens_per_recording, rng=rng, mean_repeats=mean_repeats)
               for _ in range(a.n_recordings)]
    seqs = [s.elements for s in streams]
    true_counts = np.bincount(np.concatenate([s.tokens for s in streams]), minlength=lex.n_words)

    common = {
        "law": law_name, "mean_repeats": mean_repeats, "rep": rep,
        "n_elements_total": int(sum(s.size for s in seqs)),
        "true_word_types": int(np.sum(true_counts > 0)),
        "true_r2": loglog_r2(true_counts).r2,
    }
    rows = []
    for thr in a.thresholds:
        r2, starts, counts = pipeline_r2(seqs, thr)
        row = dict(common, threshold=thr, pipeline_r2=r2, n_unit_types=int(counts.size),
                   n_units=int(counts.sum()),
                   boundary_f1=float(np.mean([boundary_prf(p, s.starts).f1
                                              for p, s in zip(starts, streams)])))
        if thr == a.baseline_threshold:
            shuffled = [pipeline_r2(shuffle_within(seqs, rng=rng), thr)[0]
                        for _ in range(a.replicates)]
            rotated = [loglog_r2(pooled_unit_counts(
                seqs, rotate_cuts(starts, rng=rng, per_sequence=False))[1]).r2
                for _ in range(a.replicates)]
            pseudo = [bigram_pseudo(s, a.replicates, rng=rng) for s in seqs]
            bigram = [pipeline_r2([p[b] for p in pseudo], thr)[0] for b in range(a.replicates)]
            for name, values in (("shuffle", shuffled), ("rotate", rotated), ("bigram", bigram)):
                c = compare_to_baseline(r2, values)
                row[f"{name}_mean"] = c.baseline_mean
                row[f"{name}_sd"] = c.baseline_sd
                row[f"{name}_z"] = c.z
                row[f"{name}_p"] = c.p_empirical
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--baseline-threshold", type=float, default=0.5)
    ap.add_argument("--repeats", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--n-words", type=int, default=100)
    ap.add_argument("--n-elements", type=int, default=30)
    ap.add_argument("--n-recordings", type=int, default=7)
    ap.add_argument("--tokens-per-recording", type=int, default=200)
    ap.add_argument("--base-seed", type=int, default=20260912)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pilot_c1_c4.csv"))
    a = ap.parse_args()
    if a.baseline_threshold not in a.thresholds:
        ap.error("--baseline-threshold must be one of --thresholds")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    resources.report("resources: ")
    t0 = time.time()
    rows = []
    conditions = [(ln, li, r, ri, rep) for li, ln in enumerate(LAWS)
                  for ri, r in enumerate(a.repeats) for rep in range(a.seeds)]
    for k, (ln, li, r, ri, rep) in enumerate(conditions, 1):
        tc = time.time()
        rows.extend(run_condition(ln, r, rep, a, li, ri))
        print(f"[{k}/{len(conditions)}] law={ln:9s} repeats={r:g} rep={rep}  "
              f"{time.time() - tc:5.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    meta = {"script": os.path.basename(__file__), "args": vars(a), "laws": LAWS,
            "unitsim_version": unitsim.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "runtime_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(df)}
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {a.out} ({len(df)} rows) in {meta['runtime_s']}s")

    at = df[df.threshold == a.baseline_threshold]
    cols = ["true_r2", "pipeline_r2", "boundary_f1", "shuffle_z", "rotate_z", "bigram_z"]
    print(at.groupby(["law", "mean_repeats"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
