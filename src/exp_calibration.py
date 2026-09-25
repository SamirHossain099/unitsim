"""Reference and held-out datasets for the calibration of CALIBRATION-DESIGN.md.

Simulates datasets that match a target corpus in size and segmentation settings while drawing the parameters a
real corpus does not reveal from a prior: lexicon size, element repertoire, word lengths, repetition or
stickiness, song structure, and the shape parameter of the law itself. Every dataset is analysed with the
published pipeline, and the statistics a real analysis reports are written out.

    whale   8 years sized as Table S1 of Arnon et al. 2025, independent words
    finch   6 birds x 5 recordings x 10 bouts, bout length from the calibration of FINDINGS.md F3
    song    whale years with theme structure (compose_song, generate_song)

Half the datasets of each law are marked `reference` and half `holdout`, drawn from the same prior with
different seeds, so `analyze_calibration.py` can fit on one half and measure discrimination and probability
calibration on the other. Nothing here uses ground truth: the recorded statistics are mean log-log R^2, mean
rank-frequency slope, mean unit types per group and mean cuts per sequence, all computable on a real corpus.

The prior is stated in PRIORS below and written to the run's .json. It is an assumption, not an estimate, and
the validation in analyze_calibration.py is what says whether a verdict under it can be trusted.

Writes results/calibration_<regime>.csv and .json.

Usage
-----
    python src/exp_calibration.py --regime whale --n-per-law 400
    python src/exp_calibration.py --regime finch --n-per-law 20      # smoke
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
from exp_c4_inference_unit import FINCH, LAWS, WHALE_YEARS
from unitsim.baselines import mean_over_groups
from unitsim.calibrate import FEATURES
from unitsim.generate import build_lexicon, compose_song, generate_song, generate_stream
from unitsim.metrics import loglog_r2, loglog_r2_or_none, pooled_unit_counts
from unitsim.tp import segment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REGIMES = ("whale", "finch", "song")

#: What a real corpus does not reveal. Ranges are inclusive; integers are drawn uniformly over the range.
PRIORS = {
    "whale": {"n_words": [60, 160], "n_elements": [20, 40], "length_range": [[2, 4], [2, 5], [2, 6], [3, 6]],
              "mean_repeats": [1.0, 5.0]},
    "finch": {"n_words": [8, 20], "n_elements": [6, 10], "length_range": [[1, 3], [2, 3], [2, 4]],
              "mean_repeats": [1.0, 2.0], "tokens_per_bout": [10, 20]},
    "song": {"n_themes": [4, 12], "variants_per_theme": [2, 5], "n_elements": [20, 40],
             "length_range": [[2, 4], [2, 5], [2, 6], [3, 6]], "phrases_per_cycle": [20.0, 80.0],
             "stickiness": [0.0, 0.8]},
}
#: A deliberately wider prior, for the stress test of analyze_calibration.py: every range is extended past the
#: reference prior, so held-out datasets drawn from it sit where the calibration was never fitted.
WIDE_PRIORS = {
    "whale": {"n_words": [40, 220], "n_elements": [12, 60], "length_range": [[1, 3], [2, 4], [2, 8], [4, 8]],
              "mean_repeats": [1.0, 9.0]},
    "finch": {"n_words": [5, 30], "n_elements": [4, 14], "length_range": [[1, 2], [1, 4], [2, 5], [3, 5]],
              "mean_repeats": [1.0, 4.0], "tokens_per_bout": [6, 30]},
    "song": {"n_themes": [2, 16], "variants_per_theme": [1, 8], "n_elements": [12, 60],
             "length_range": [[1, 3], [2, 4], [2, 8], [4, 8]], "phrases_per_cycle": [8.0, 160.0],
             "stickiness": [0.0, 0.95]},
}
#: Shape of each law, also unknown for a real corpus.
SHAPE_PRIOR = {"zipf": {"exponent": [0.8, 1.2]}, "uniform": {},
               "lognormal": {"sigma": [0.5, 2.0]}, "geometric": {"log10_tail": [-2.0, -0.7]}}
WIDE_SHAPE_PRIOR = {"zipf": {"exponent": [0.5, 1.6]}, "uniform": {},
                    "lognormal": {"sigma": [0.2, 3.5]}, "geometric": {"log10_tail": [-3.0, -0.3]}}


def draw_shape(law, rng, prior=None):
    prior = prior or SHAPE_PRIOR
    if law == "zipf":
        return {"exponent": float(rng.uniform(*prior["zipf"]["exponent"]))}
    if law == "lognormal":
        return {"sigma": float(rng.uniform(*prior["lognormal"]["sigma"]))}
    if law == "geometric":
        return {"tail": float(10.0 ** rng.uniform(*prior["geometric"]["log10_tail"]))}
    return {}


def law_kwargs(law, n_words, shape):
    if law == "zipf":
        return {"law": "zipf", "exponent": shape["exponent"]}
    if law == "uniform":
        return {"law": "uniform"}
    if law == "lognormal":
        return {"law": "lognormal", "sigma": shape["sigma"]}
    return {"law": "geometric", "ratio": float(shape["tail"] ** (1.0 / (n_words - 1)))}


def possible_words(n_elements, length_range):
    """How many distinct words exist over this repertoire and length range (repeats allowed, as in build_lexicon)."""
    lo, hi = length_range
    return sum(n_elements ** length for length in range(lo, hi + 1))


def draw_nuisance(regime, rng, priors=None):
    """One draw from the prior. The lexicon is capped at the number of words the repertoire can supply, which
    binds only under the wide prior of the stress test: the matched prior cannot ask for an impossible lexicon."""
    p = (priors or PRIORS)[regime]
    out = {"n_elements": int(rng.integers(p["n_elements"][0], p["n_elements"][1] + 1)),
           "length_range": tuple(p["length_range"][int(rng.integers(len(p["length_range"])))])}
    if regime == "song":
        out.update(n_themes=int(rng.integers(p["n_themes"][0], p["n_themes"][1] + 1)),
                   variants_per_theme=int(rng.integers(p["variants_per_theme"][0], p["variants_per_theme"][1] + 1)),
                   phrases_per_cycle=float(rng.uniform(*p["phrases_per_cycle"])),
                   stickiness=float(rng.uniform(*p["stickiness"])))
        out["n_words"] = out["n_themes"] * out["variants_per_theme"]
        out["n_words"] = max(2, min(out["n_words"], possible_words(out["n_elements"], out["length_range"])))
        out["variants_per_theme"] = max(1, out["n_words"] // out["n_themes"])
        out["n_words"] = out["n_themes"] * out["variants_per_theme"]
    else:
        out.update(n_words=int(rng.integers(p["n_words"][0], p["n_words"][1] + 1)),
                   mean_repeats=float(rng.uniform(*p["mean_repeats"])))
        if regime == "finch":
            out["tokens_per_bout"] = int(rng.integers(p["tokens_per_bout"][0], p["tokens_per_bout"][1] + 1))
        out["n_words"] = max(2, min(out["n_words"], possible_words(out["n_elements"], out["length_range"])))
    return out


def simulate(regime, law, nuisance, rng):
    """Element sequences of one dataset, grouped as the published analysis groups them."""
    shape = nuisance["shape"]
    kw = law_kwargs(law, nuisance["n_words"], shape)
    groups = []
    if regime in ("whale", "song"):
        for n_rec, n_elem in WHALE_YEARS:
            lex = build_lexicon(nuisance["n_words"], nuisance["n_elements"], rng=rng,
                                length_range=nuisance["length_range"], brevity=False, **kw)
            tokens = max(2, int(round(n_elem / n_rec / float(np.dot(lex.lengths, lex.probs)))))
            if regime == "whale":
                streams = [generate_stream(lex, tokens, rng=rng, mean_repeats=nuisance["mean_repeats"])
                           for _ in range(n_rec)]
            else:
                song = compose_song(lex, nuisance["n_themes"], rng=rng)
                streams = [generate_song(lex, song, tokens, rng=rng,
                                         phrases_per_cycle=nuisance["phrases_per_cycle"],
                                         stickiness=nuisance["stickiness"], random_start=True)
                           for _ in range(n_rec)]
            groups.append([s.elements for s in streams])
        return groups
    for _ in range(FINCH["birds"]):
        lex = build_lexicon(nuisance["n_words"], nuisance["n_elements"], rng=rng,
                            length_range=nuisance["length_range"], brevity=False, **kw)
        for _ in range(FINCH["recordings_per_bird"]):
            streams = [generate_stream(lex, nuisance["tokens_per_bout"], rng=rng,
                                       mean_repeats=nuisance["mean_repeats"])
                       for _ in range(FINCH["bouts_per_recording"])]
            groups.append([s.elements for s in streams])
    return groups


def features(groups, threshold):
    """The statistics a published analysis reports. No ground truth is used."""
    r2, slopes, n_types, cuts = [], [], [], []
    for seqs in groups:
        starts = segment(seqs, threshold=threshold)
        counts = pooled_unit_counts(seqs, starts)[1]
        value = loglog_r2_or_none(counts)
        r2.append(value)
        slopes.append(None if value is None else loglog_r2(counts).slope)
        n_types.append(float(counts.size))
        cuts.extend(int(s[1:].sum()) for s in starts)
    mean_r2 = mean_over_groups(r2, "skip")
    return {"observed_mean_r2": mean_r2.mean, "observed_undefined_groups": mean_r2.n_undefined,
            "observed_mean_slope": mean_over_groups(slopes, "skip").mean,
            "observed_mean_unit_types": float(np.mean(n_types)),
            "cuts_per_sequence": float(np.mean(cuts)),
            "elements_total": int(sum(s.size for g in groups for s in g))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=REGIMES, default="whale")
    ap.add_argument("--n-per-law", type=int, default=400, help="reference datasets per law; as many held out")
    ap.add_argument("--laws", nargs="+", choices=LAWS, default=list(LAWS))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--prior", choices=("matched", "wide"), default="matched",
                    help="matched: reference and held-out halves from the same prior. wide: every dataset from "
                         "WIDE_PRIORS and marked holdout, for the prior-mismatch stress test")
    ap.add_argument("--base-seed", type=int, default=20260925)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suffix = "" if a.prior == "matched" else "_wide"
    out = a.out or os.path.join(ROOT, "results", f"calibration_{a.regime}{suffix}.csv")
    priors = PRIORS if a.prior == "matched" else WIDE_PRIORS
    shape_prior = SHAPE_PRIOR if a.prior == "matched" else WIDE_SHAPE_PRIOR
    splits = (("reference", 0), ("holdout", 1)) if a.prior == "matched" else (("holdout", 2),)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    resources.report("resources: ")
    t0 = time.time()
    rows = []
    for law in a.laws:
        for split, offset in splits:
            for i in range(a.n_per_law):
                rng = np.random.default_rng([a.base_seed, REGIMES.index(a.regime), LAWS.index(law), offset, i])
                nuisance = draw_nuisance(a.regime, rng, priors)
                nuisance["shape"] = draw_shape(law, rng, shape_prior)
                groups = simulate(a.regime, law, nuisance, rng)
                row = {"regime": a.regime, "law": law, "set": split, "dataset": i,
                       "threshold": a.threshold, **{k: v for k, v in nuisance.items() if k != "shape"},
                       **{f"shape_{k}": v for k, v in nuisance["shape"].items()},
                       **features(groups, a.threshold)}
                row["length_range"] = f"{nuisance['length_range'][0]}-{nuisance['length_range'][1]}"
                rows.append(row)
            print(f"[{law} {split}] {a.n_per_law} datasets, {time.time() - t0:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = {"script": os.path.basename(__file__), "args": vars(a), "priors": priors,
            "shape_prior": shape_prior, "features": list(FEATURES), "whale_years": WHALE_YEARS, "finch": FINCH,
            "unitsim_version": unitsim.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "runtime_s": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": len(df)}
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out} ({len(df)} rows) in {meta['runtime_s']}s")
    print(df.groupby(["law", "set"])[list(FEATURES)].mean().round(3).to_string())


if __name__ == "__main__":
    main()
