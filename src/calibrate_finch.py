"""Calibrate the finch regime against the published cut rate.

Kirby et al. (2026) report 10.72 cuts per bout at threshold 0.5 on real Bengalese finch song. Bout length
and lexicon size are not reported, so this sweeps the generator settings that control them. For each
setting it records the pipeline's cuts per bout on simulated song, and how often a recording's R^2 is
undefined, both unshuffled and after shuffling (FINDINGS.md F2).

Writes results/calibration_finch.csv. Use it to choose settings whose cut rate brackets 10.72, and report
that choice with its sensitivity rather than one tuned value.
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from exp_c4_inference_unit import FINCH, law_params
from unitsim.baselines import shuffle_within
from unitsim.generate import build_lexicon, generate_stream
from unitsim.metrics import loglog_r2_or_none, pooled_unit_counts
from unitsim.tp import segment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUBLISHED_CUTS_PER_BOUT = 10.72


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens-per-bout", type=int, nargs="+", default=[8, 10, 12, 15, 20])
    ap.add_argument("--length-ranges", nargs="+", default=["1-3", "2-3", "2-4"])
    ap.add_argument("--n-words", type=int, nargs="+", default=[10, 15, 25])
    ap.add_argument("--repeats", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--laws", nargs="+", default=["zipf", "uniform"])
    ap.add_argument("--n-elements", type=int, default=8)
    ap.add_argument("--datasets", type=int, default=3)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--base-seed", type=int, default=20260913)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "calibration_finch.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    resources.report("resources: ")
    t0 = time.time()
    rows = []
    grid = list(itertools.product(a.laws, a.tokens_per_bout, a.length_ranges, a.n_words, a.repeats))
    for k, (law, tpb, lr, nw, rep) in enumerate(grid):
        lo, hi = (int(v) for v in lr.split("-"))
        cuts, bout_len = [], []
        undefined_real = undefined_shuffled = recordings = 0
        for d in range(a.datasets):
            rng = np.random.default_rng([a.base_seed, k, d])
            for _ in range(FINCH["birds"]):
                lex = build_lexicon(nw, a.n_elements, rng=rng, length_range=(lo, hi),
                                    brevity=False, **law_params(law, nw))
                for _ in range(FINCH["recordings_per_bird"]):
                    seqs = [generate_stream(lex, tpb, rng=rng, mean_repeats=rep).elements
                            for _ in range(FINCH["bouts_per_recording"])]
                    st = segment(seqs, threshold=a.threshold)
                    cuts.extend(int(s[1:].sum()) for s in st)
                    bout_len.extend(int(s.size) for s in seqs)
                    undefined_real += loglog_r2_or_none(pooled_unit_counts(seqs, st)[1]) is None
                    for _ in range(a.shuffles):
                        sh = shuffle_within(seqs, rng=rng)
                        undefined_shuffled += loglog_r2_or_none(
                            pooled_unit_counts(sh, segment(sh, threshold=a.threshold))[1]) is None
                    recordings += 1
        rows.append({"law": law, "tokens_per_bout": tpb, "length_range": lr, "n_words": nw,
                     "mean_repeats": rep, "cuts_per_bout": float(np.mean(cuts)),
                     "bout_length": float(np.mean(bout_len)),
                     "undefined_real_rate": undefined_real / recordings,
                     "undefined_shuffled_rate": undefined_shuffled / (recordings * a.shuffles)})

    df = pd.DataFrame(rows)
    df["gap_to_published"] = df.cuts_per_bout - PUBLISHED_CUTS_PER_BOUT
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(df)} settings) in {time.time() - t0:.0f}s")
    near = df.loc[df.gap_to_published.abs().sort_values().index].head(12)
    print(f"\nsettings nearest the published {PUBLISHED_CUTS_PER_BOUT} cuts per bout:")
    print(near.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
