"""Summarise C2: paired shifts from element-label error, against the clean noise floor.

Reads results/c2_whale.csv. Writes results/c2_whale_summary.json.

For each cell (law, brevity, repeats), corruption condition and statistic:

    floor_sd          between-dataset sd of the statistic under clean labels in that cell
    mean_shift        mean over datasets of (corrupted - clean), paired within dataset
    ratio             |mean_shift| / floor_sd. Above 1, label error moves the statistic by more than the
                      natural spread across datasets, so a single dataset's value cannot be read without
                      knowing its error rate
    share_beyond      share of datasets whose own |shift| exceeds floor_sd
    sign_consistency  share of datasets whose shift has the sign of mean_shift

Rows where the statistic was undefined in any year are counted, never silently averaged in.
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

from exp_c2_label_error import CONDITIONS, STATS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEYS = ["law", "brevity", "mean_repeats", "dataset"]
CELL = ["law", "brevity", "mean_repeats"]
SHOW = ["loglog_r2", "loglog_slope", "brevity_slope", "csn_alpha", "boundary_f1"]


def num(x):
    x = float(x)
    return x if math.isfinite(x) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "c2_whale.csv"))
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    clean = df[df.condition == "clean"].set_index(KEYS)
    if clean.index.duplicated().any():
        raise AssertionError("duplicate clean rows for a dataset")
    out = {"source": os.path.basename(a.csv), "cells": {}}
    table = []

    for cell, cg in clean.groupby(level=CELL):
        name = f"{cell[0]}|brevity={cell[1]}|repeats={cell[2]:g}"
        floor = cg[list(STATS)].std(ddof=1)
        entry = {"n_datasets": int(len(cg)),
                 "clean_mean": {s: num(v) for s, v in cg[list(STATS)].mean().items()},
                 "floor_sd": {s: num(v) for s, v in floor.items()}, "conditions": {}}
        for condition in CONDITIONS[1:]:
            shifted = df[df.condition == condition].set_index(KEYS)
            missing = cg.index.difference(shifted.index)
            if len(missing):
                raise AssertionError(f"{condition} lacks {len(missing)} clean datasets in {name}")
            shifted = shifted.loc[cg.index]
            diff = shifted[list(STATS)] - cg[list(STATS)]
            res = {}
            for s in STATS:
                col = diff[s].dropna()
                f = floor[s]
                ms = col.mean() if col.size else float("nan")
                usable = col.size > 1 and math.isfinite(f) and f > 0
                res[s] = {
                    "mean_shift": num(ms),
                    "sd_shift": num(col.std(ddof=1)) if col.size > 1 else None,
                    "ratio": num(abs(ms) / f) if usable else None,
                    "share_beyond": num((col.abs() > f).mean()) if usable else None,
                    "sign_consistency": num((np.sign(col) == np.sign(ms)).mean()) if col.size else None,
                    "n": int(col.size),
                    "rows_with_undefined_years": int((shifted[f"{s}_undefined_years"] > 0).sum()
                                                     + (cg[f"{s}_undefined_years"] > 0).sum()),
                }
            entry["conditions"][condition] = res
            table.append({"cell": name, "condition": condition,
                          **{s: res[s]["ratio"] for s in SHOW}})
        out["cells"][name] = entry

    dest = os.path.splitext(a.csv)[0] + "_summary.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dest}")
    print("\n|mean paired shift| / clean between-dataset sd (above 1 = beyond the noise floor):")
    print(pd.DataFrame(table).set_index(["cell", "condition"]).round(2).to_string())


if __name__ == "__main__":
    main()
