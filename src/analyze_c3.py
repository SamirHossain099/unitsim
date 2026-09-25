"""Summarise C3: which statistic separates the generating law, on discovered units and on true words?

Reads results/c3_whale.csv. Writes results/c3_whale_summary.json.

Discriminability uses the simulated dataset as the unit: each continuous statistic is averaged over the 8
years first, matching the published inference unit. Verdict rates are per year, because a Clauset test is
run and read one year at a time.

Verdicts per year, among years where that test produced a value:
    pl_plausible        gof_p >= 0.1, the convention of Clauset et al. (2009)
    pl_plausible_broad  gof_p >= 0.1 and tail_fraction >= 0.5: plausible over at least half the types
    lr_favours_pl       lr_R > 0 and lr_p < 0.1
    lr_favours_ln       lr_R < 0 and lr_p < 0.1
Likelihood-ratio verdicts are also given with at-bound lognormal fits excluded (FINDINGS.md F4).
Every rate carries its denominator, so a rate computed over few years is visible as such.
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from analyze_pilot_c1_c4 import auc, auc_ci
from exp_c3_estimators import SOURCES
from exp_c4_inference_unit import LAWS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTINUOUS = ["loglog_r2", "tail_fraction", "gof_p", "lr_normalized_R", "csn_alpha"]


def num(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def as_bool(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return {"True": True, "False": False}[str(v)]


def rate(hit, valid):
    n = int(valid.sum())
    return {"rate": float((hit & valid).sum() / n) if n else None, "n": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "c3_whale.csv"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--factor", default="mean_repeats",
                    help="column splitting conditions besides law, e.g. stickiness for song runs")
    a = ap.parse_args()
    label = {"mean_repeats": "repeats"}.get(a.factor, a.factor)

    df = pd.read_csv(a.csv)
    at_bound = df.lr_at_bound.map(as_bool)
    df["at_bound_true"] = at_bound.eq(True)
    df["at_bound_false"] = at_bound.eq(False)
    df["error"] = df.error.fillna("")
    out = {"source": os.path.basename(a.csv), "verdicts": {}, "describe": {}, "auc_zipf_vs": {}}

    for (src, law, rep), g in df.groupby(["source", "law", a.factor]):
        key = f"{src}|{law}|{label}={rep:g}"
        has_gof = g.gof_p.notna()
        has_lr = g.lr_R.notna()
        interior = has_lr & g.at_bound_false
        pl_sig = (g.lr_R > 0) & (g.lr_p < 0.1)
        ln_sig = (g.lr_R < 0) & (g.lr_p < 0.1)
        out["verdicts"][key] = {
            "pl_plausible": rate(g.gof_p >= 0.1, has_gof),
            "pl_plausible_broad": rate((g.gof_p >= 0.1) & (g.tail_fraction >= 0.5), has_gof),
            "lr_favours_pl": rate(pl_sig, has_lr),
            "lr_favours_ln": rate(ln_sig, has_lr),
            "lr_favours_pl_interior": rate(pl_sig, interior),
            "lr_favours_ln_interior": rate(ln_sig, interior),
            "at_bound": rate(g.at_bound_true, has_lr),
            "error": {"rate": float((g.error != "").mean()), "n": int(len(g))},
        }
        out["describe"][key] = {c: {"mean": num(g[c].mean()), "sd": num(g[c].std(ddof=1))}
                                for c in CONTINUOUS}

    per_dataset = (df.groupby(["source", "law", a.factor, "dataset"])[CONTINUOUS]
                   .mean().reset_index())
    rng = np.random.default_rng(a.seed)
    for src in SOURCES:
        for rep in sorted(per_dataset[a.factor].unique()):
            base = per_dataset[(per_dataset.source == src) & (per_dataset[a.factor] == rep)]
            zipf = base[base.law == "zipf"]
            for law in LAWS:
                other = base[base.law == law]
                if law == "zipf" or zipf.empty or other.empty:
                    continue
                cell = {}
                for c in CONTINUOUS:
                    pos, neg = zipf[c].dropna(), other[c].dropna()
                    if pos.size < 2 or neg.size < 2:
                        cell[c] = None
                        continue
                    cell[c] = {"auc": auc(pos, neg), "ci95": auc_ci(pos, neg, rng, a.n_boot),
                               "n": [int(pos.size), int(neg.size)]}
                out["auc_zipf_vs"][f"{src}|{law}|{label}={rep:g}"] = cell

    dest = os.path.splitext(a.csv)[0] + "_summary.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dest}")

    show = ["pl_plausible", "pl_plausible_broad", "lr_favours_pl_interior", "lr_favours_ln_interior",
            "at_bound"]
    print("\nper-year verdict rates (n in brackets):")
    for key, v in out["verdicts"].items():
        cells = "  ".join(f"{s}={v[s]['rate']:.2f}[{v[s]['n']}]" if v[s]["rate"] is not None
                          else f"{s}=--[0]" for s in show)
        print(f"  {key:32s} {cells}")
    print("\ndataset-level AUC, Zipf vs other law:")
    for key, v in out["auc_zipf_vs"].items():
        cells = "  ".join(f"{c}={v[c]['auc']:.2f}[{v[c]['ci95'][0]:.2f},{v[c]['ci95'][1]:.2f}]"
                          if v[c] else f"{c}=--" for c in CONTINUOUS)
        print(f"  {key:32s} {cells}")


if __name__ == "__main__":
    main()
