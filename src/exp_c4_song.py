"""C4 on song-structured data: does the answer survive theme order and phrase repetition?

Every earlier C4 run drew words independently. Humpback song is not like that: phrases repeat within
themes and themes follow a fixed order, shared by the singers of a year. This reruns C4 at the whale
inference unit on streams from `unitsim.generate.generate_song`, which adds that structure while keeping
every phrase type's expected frequency exactly proportional to the chosen law, so structure and frequency
law stay separable.

One Song (themes and their order) is composed per year and shared by that year's recordings. Year sizes
follow Table S1, and the analysis -- mean-R^2 statistic, three baselines, undefined policy -- is imported
unchanged from exp_c4_inference_unit.py.

Song parameters are assumptions, not calibrated values, and are written to every row: themes per year,
phrase variants per theme, renditions per cycle, stickiness. Humpback themes typically hold few phrase
variants, hence the small lexicon (themes x variants).

Writes results/c4_whale_song_<policy>.csv and .json. Summarise with
    python src/analyze_c4.py --regime whale --tag song --factor stickiness

Usage
-----
    python src/exp_c4_song.py --undefined skip --datasets 30 --replicates 200
    python src/exp_c4_song.py --undefined skip --datasets 1 --replicates 20 --laws zipf uniform   # smoke
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
from exp_c4_inference_unit import LAWS, WHALE_YEARS, analyse, law_params, shape
from unitsim.baselines import UNDEFINED_POLICIES
from unitsim.generate import build_lexicon, compose_song, generate_song

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def simulate_song_years(law, stickiness, rng, a):
    n_words = a.n_themes * a.variants_per_theme
    groups = []
    for n_rec, n_elem in WHALE_YEARS:
        lex = build_lexicon(n_words, a.n_elements, rng=rng, length_range=tuple(a.length_range),
                            brevity=False, **law_params(law, n_words, **shape(a)))
        song = compose_song(lex, a.n_themes, rng=rng)
        tokens = max(2, int(round(n_elem / n_rec / float(np.dot(lex.lengths, lex.probs)))))
        streams = [generate_song(lex, song, tokens, rng=rng, phrases_per_cycle=a.phrases_per_cycle,
                                 stickiness=stickiness, random_start=True)
                   for _ in range(n_rec)]
        groups.append({"seqs": [s.elements for s in streams], "truth": [s.starts for s in streams]})
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--undefined", choices=UNDEFINED_POLICIES, required=True)
    ap.add_argument("--datasets", type=int, default=30)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--laws", nargs="+", choices=LAWS, default=list(LAWS))
    ap.add_argument("--stickiness", type=float, nargs="+", default=[0.0, 0.8])
    ap.add_argument("--n-themes", type=int, default=8)
    ap.add_argument("--variants-per-theme", type=int, default=3)
    ap.add_argument("--phrases-per-cycle", type=float, default=40.0)
    ap.add_argument("--n-elements", type=int, default=30)
    ap.add_argument("--length-range", type=int, nargs=2, default=[2, 6])
    ap.add_argument("--zipf-exponent", type=float, default=1.0)
    ap.add_argument("--lognormal-sigma", type=float, default=1.0)
    ap.add_argument("--geometric-tail", type=float, default=0.05)
    ap.add_argument("--base-seed", type=int, default=20260916)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", f"c4_whale_song_{a.undefined}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    resources.report("resources: ")
    t0 = time.time()
    rows = []
    conditions = [(law, s, d) for law in a.laws for s in a.stickiness for d in range(a.datasets)]
    for k, (law, s, d) in enumerate(conditions, 1):
        tc = time.time()
        rng = np.random.default_rng([a.base_seed, LAWS.index(law), int(round(s * 100)), d])
        groups = simulate_song_years(law, s, rng, a)
        row = {"regime": "whale", "structure": "song", "undefined_policy": a.undefined, "law": law,
               "stickiness": s, "dataset": d, "threshold": a.threshold, "n_themes": a.n_themes,
               "variants_per_theme": a.variants_per_theme, "phrases_per_cycle": a.phrases_per_cycle,
               **shape(a)}
        row.update(analyse(groups, "whale", a.threshold, a.replicates, a.undefined, rng))
        rows.append(row)
        print(f"[{k}/{len(conditions)}] {law:9s} stickiness={s:g} d={d}  "
              f"R2={row['observed_mean_r2']:.3f}  z(shuf/rot/bi)="
              f"{row['shuffle_z']:.1f}/{row['rotate_z']:.1f}/{row['bigram_z']:.1f}  "
              f"F1={row['boundary_f1']:.2f}  {time.time() - tc:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = {"script": os.path.basename(__file__), "args": vars(a), "whale_years": WHALE_YEARS,
            "unitsim_version": unitsim.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "runtime_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(df)}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out} ({len(df)} rows) in {meta['runtime_s']}s")
    cols = ["observed_mean_r2", "boundary_f1", "shuffle_z", "rotate_z", "bigram_z"]
    print(df.groupby(["law", "stickiness"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
