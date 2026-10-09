"""Reference and held-out datasets matched to the real humpback corpus of Arnon et al. 2025 (Data S2).

The calibration of `exp_calibration.py` matched the whale years to Table S1 of that paper and drew the element
repertoire from a prior (20 to 40 types). The deposited sequences reveal more than Table S1: each year's exact
recording and element counts, and its number of element types (33 to 92). Revealed quantities are matched here,
year by year; only what the data cannot reveal is drawn from a prior: the number of themes and phrase variants
(so the lexicon size), word lengths, phrases per cycle, stickiness, and the shape of each law.

Real annotations also carry label error, which the simulated words of the paper did not: the humpback element
classifier's out-of-bag error is 19.19% and the finch coders disagree on 6.7% of syllables. Section 3.6 of the
paper shows that such error raises the number of distinct units and flattens the rank-frequency slope. The
`_labels` priors therefore add a per-token confusion rate drawn from 0 to 0.2, a range set by those published
rates and not fitted to the corpus.

    song           the song prior of the paper (`exp_calibration.PRIORS["song"]`), repertoire matched to the corpus
    song_labels    the same, plus label confusion
    broad_labels   more phrase variants per theme, longer words and stickier phrases, plus label confusion

Half the datasets of each law are `reference` and half `holdout`, as in `exp_calibration.py`, so the procedure
is validated under each prior before it is applied to the real corpus (`analyze_real_whale.py`).

    python src/exp_real_whale.py --prior song --n-per-law 1000 --workers 8
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import unitsim
from exp_c4_inference_unit import LAWS
from exp_calibration import SHAPE_PRIOR, draw_shape, features, law_kwargs
from real_whale_features import load_groups
from unitsim.calibrate import FEATURES
from unitsim.corrupt import confuse
from unitsim.generate import build_lexicon, compose_song, generate_song

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PRIORS = {
    "song": {"n_themes": [4, 12], "variants_per_theme": [2, 5], "length_range": [[2, 4], [2, 5], [2, 6], [3, 6]],
             "phrases_per_cycle": [20.0, 80.0], "stickiness": [0.0, 0.8]},
    "song_labels": {"n_themes": [4, 12], "variants_per_theme": [2, 5],
                    "length_range": [[2, 4], [2, 5], [2, 6], [3, 6]], "phrases_per_cycle": [20.0, 80.0],
                    "stickiness": [0.0, 0.8], "confusion": [0.0, 0.2]},
    "broad_labels": {"n_themes": [4, 12], "variants_per_theme": [2, 20],
                     "length_range": [[2, 4], [2, 5], [2, 6], [3, 6], [2, 8], [3, 8]],
                     "phrases_per_cycle": [20.0, 120.0], "stickiness": [0.0, 0.95], "confusion": [0.0, 0.2]},
}


def corpus_years():
    """(recordings, elements, element types) per year of the deposited corpus."""
    _, sizes = load_groups()
    return tuple((s["recordings"], s["elements"], s["element_types"]) for s in sizes)


def draw_nuisance(prior, rng):
    p = PRIORS[prior]
    n_themes = int(rng.integers(p["n_themes"][0], p["n_themes"][1] + 1))
    variants = int(rng.integers(p["variants_per_theme"][0], p["variants_per_theme"][1] + 1))
    return {"n_themes": n_themes, "variants_per_theme": variants, "n_words": n_themes * variants,
            "length_range": tuple(p["length_range"][int(rng.integers(len(p["length_range"])))]),
            "phrases_per_cycle": float(rng.uniform(*p["phrases_per_cycle"])),
            "stickiness": float(rng.uniform(*p["stickiness"])),
            "confusion": float(rng.uniform(*p["confusion"])) if "confusion" in p else 0.0}


def simulate(law, nuisance, years, rng):
    """Song-structured years with each year's recording count, element count and repertoire matched."""
    kw = law_kwargs(law, nuisance["n_words"], nuisance["shape"])
    groups = []
    for n_rec, n_elem, n_types in years:
        lex = build_lexicon(nuisance["n_words"], n_types, rng=rng, length_range=nuisance["length_range"],
                            brevity=False, **kw)
        song = compose_song(lex, nuisance["n_themes"], rng=rng)
        tokens = max(2, int(round(n_elem / n_rec / float(np.dot(lex.lengths, lex.probs)))))
        streams = [generate_song(lex, song, tokens, rng=rng, phrases_per_cycle=nuisance["phrases_per_cycle"],
                                 stickiness=nuisance["stickiness"], random_start=True) for _ in range(n_rec)]
        groups.append([confuse(s.elements, nuisance["confusion"], n_types, rng=rng) for s in streams])
    return groups


def one(job):
    prior, law, split, offset, i, base_seed, threshold, years = job
    rng = np.random.default_rng([base_seed, list(PRIORS).index(prior), LAWS.index(law), offset, i])
    nuisance = draw_nuisance(prior, rng)
    nuisance["shape"] = draw_shape(law, rng, SHAPE_PRIOR)
    groups = simulate(law, nuisance, years, rng)
    row = {"prior": prior, "law": law, "set": split, "dataset": i, "threshold": threshold,
           **{k: v for k, v in nuisance.items() if k != "shape"},
           **{f"shape_{k}": v for k, v in nuisance["shape"].items()}, **features(groups, threshold)}
    row["length_range"] = f"{nuisance['length_range'][0]}-{nuisance['length_range'][1]}"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prior", choices=tuple(PRIORS), default="song")
    ap.add_argument("--n-per-law", type=int, default=1000, help="reference datasets per law; as many held out")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--base-seed", type=int, default=20261005)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", f"real_whale_reference_{a.prior}.csv")
    years = corpus_years()
    jobs = [(a.prior, law, split, offset, i, a.base_seed, a.threshold, years)
            for law in LAWS for split, offset in (("reference", 0), ("holdout", 1)) for i in range(a.n_per_law)]
    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(one, jobs, chunksize=16))
    else:
        rows = [one(j) for j in jobs]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = {"script": os.path.basename(__file__), "args": vars(a), "prior": PRIORS[a.prior],
            "shape_prior": SHAPE_PRIOR, "features": list(FEATURES), "years": years,
            "unitsim_version": unitsim.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "runtime_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(df)}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out} ({len(df)} rows) in {meta['runtime_s']}s")
    print(df.groupby(["law", "set"])[list(FEATURES)].mean().round(3).to_string())


if __name__ == "__main__":
    main()
