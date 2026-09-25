"""Summarise the C1/C4 pilot: how well each statistic separates a Zipfian source from each other law.

Reads results/pilot_c1_c4.csv and .json. Writes results/pilot_c1_c4_summary.json.

The question is discriminability, not whether a number "looks Zipfian". A finite sample from a uniform
lexicon already has a sizeable log-log R^2 from sampling noise alone, so a high R^2 on its own says
little. AUC is P(statistic from a Zipf source > statistic from the other source), from the Mann-Whitney
U statistic, with a percentile bootstrap CI over simulated years. AUC 0.5: the statistic cannot tell
the two apart. 1.0: perfect separation.

`population_r2` is the log-log R^2 of the generating probabilities themselves, recomputed by rebuilding
each lexicon from its seed. It is undefined for a uniform law (every probability equal) and is stored
as null there.

Per-year baseline z is NOT comparable to published z: the papers compare the MEAN R^2 over recordings
(finch, verified) or years (whale) against the mean over each pseudo-dataset. The full experiment
reproduces that inference unit; this pilot does not.
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from exp_pilot_c1_c4 import LAWS
from unitsim.generate import build_lexicon
from unitsim.metrics import DegenerateDistribution, loglog_r2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEPARATORS = ["pipeline_r2", "true_r2", "shuffle_z", "rotate_z", "bigram_z"]
DESCRIBE = ["population_r2", "true_r2", "pipeline_r2", "boundary_f1", "n_unit_types",
            "shuffle_z", "rotate_z", "bigram_z"]


def auc(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC needs both groups non-empty")
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(neg))):
        raise ValueError("AUC input contains non-finite values")
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (pos.size * neg.size))


def auc_ci(pos, neg, rng, n_boot):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    boots = [auc(rng.choice(pos, pos.size), rng.choice(neg, neg.size)) for _ in range(n_boot)]
    return [float(v) for v in np.percentile(boots, [2.5, 97.5])]


def population_r2(args, law, law_index, repeats_index, rep):
    rng = np.random.default_rng([args["base_seed"], law_index, repeats_index, rep])
    lex = build_lexicon(args["n_words"], args["n_elements"], rng=rng, length_range=(2, 6),
                        brevity=False, **LAWS[law])
    try:
        return loglog_r2(lex.probs).r2
    except DegenerateDistribution:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "pilot_c1_c4.csv"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    with open(os.path.splitext(a.csv)[0] + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    args = meta["args"]
    law_index = {law: i for i, law in enumerate(LAWS)}
    repeats_index = {r: i for i, r in enumerate(args["repeats"])}

    keys = df[["law", "mean_repeats", "rep"]].drop_duplicates()
    pop = {(r.law, r.mean_repeats, r.rep): population_r2(
        args, r.law, law_index[r.law], repeats_index[r.mean_repeats], int(r.rep))
        for r in keys.itertuples()}
    df["population_r2"] = [pop[(r.law, r.mean_repeats, r.rep)] for r in df.itertuples()]

    base = df[df.threshold == args["baseline_threshold"]]
    rng = np.random.default_rng(a.seed)
    out = {"source": os.path.basename(a.csv), "baseline_threshold": args["baseline_threshold"],
           "seeds_per_condition": args["seeds"], "replicates": args["replicates"],
           "describe": {}, "auc_zipf_vs": {}, "pipeline_r2_by_threshold": {}}

    for (law, rep), g in base.groupby(["law", "mean_repeats"]):
        key = f"{law}|repeats={rep:g}"
        out["describe"][key] = {
            col: {"mean": float(g[col].dropna().mean()) if g[col].notna().any() else None,
                  "sd": float(g[col].dropna().std(ddof=1)) if g[col].notna().sum() > 1 else None,
                  "min": float(g[col].dropna().min()) if g[col].notna().any() else None,
                  "max": float(g[col].dropna().max()) if g[col].notna().any() else None}
            for col in DESCRIBE}

    for rep in args["repeats"]:
        z = base[(base.law == "zipf") & (base.mean_repeats == rep)]
        for law in LAWS:
            if law == "zipf":
                continue
            o = base[(base.law == law) & (base.mean_repeats == rep)]
            out["auc_zipf_vs"][f"{law}|repeats={rep:g}"] = {
                s: {"auc": auc(z[s], o[s]), "ci95": auc_ci(z[s], o[s], rng, a.n_boot)}
                for s in SEPARATORS}

    for (law, rep, thr), g in df.groupby(["law", "mean_repeats", "threshold"]):
        out["pipeline_r2_by_threshold"][f"{law}|repeats={rep:g}|thr={thr:g}"] = {
            "mean": float(g.pipeline_r2.mean()), "sd": float(g.pipeline_r2.std(ddof=1))}

    dest = os.path.splitext(a.csv)[0] + "_summary.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dest}")

    print("\nAUC, Zipf source vs other law (0.5 = indistinguishable):")
    for k, v in out["auc_zipf_vs"].items():
        cells = "  ".join(f"{s}={v[s]['auc']:.2f}[{v[s]['ci95'][0]:.2f},{v[s]['ci95'][1]:.2f}]"
                          for s in SEPARATORS)
        print(f"  {k:26s} {cells}")


if __name__ == "__main__":
    main()
