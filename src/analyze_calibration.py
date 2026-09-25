"""Validate the calibration before anyone uses it (CALIBRATION-DESIGN.md, step 4).

Fits reference densities on the `reference` half of results/calibration_<regime>.csv and scores the `holdout`
half, which was drawn from the same prior with different seeds. Reports, for the whole held-out set and for
each law:

    accuracy, chance                 how often the most probable law is the true one
    share_decided                    share of datasets reaching the confidence threshold, the rest being
    accuracy_when_decided            reported as "not identifiable"
    brier, reliability               whether a stated confidence means what it says
    per_law auc and recall
    ablation                         the same on each single statistic, to show which one carries the law

A calibration that is accurate but overconfident is not usable, so the reliability table is the point of this
script as much as the accuracy is.

Writes results/calibration_<regime>_summary.json.

Usage
-----
    python src/analyze_calibration.py --regime whale
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import os

import pandas as pd

from unitsim.calibrate import FEATURES, build_reference, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def split(df):
    rows = df.to_dict("records")
    return ([r for r in rows if r["set"] == "reference"], [r for r in rows if r["set"] == "holdout"])


def run(reference_rows, holdout_rows, features, thresholds):
    ref = build_reference(reference_rows, features=features)
    out = {"features": list(features), "reference_n": ref.counts}
    for t in thresholds:
        out[f"at_{t:g}"] = evaluate(ref, holdout_rows, min_probability=t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", default="whale")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--holdout-csv", default=None,
                    help="score these datasets instead of the file's own holdout half; the _wide run measures "
                         "what a prior that misses the truth costs")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    a = ap.parse_args()
    csv = a.csv or os.path.join(ROOT, "results", f"calibration_{a.regime}.csv")
    df = pd.read_csv(csv)
    reference_rows, holdout_rows = split(df)
    if a.holdout_csv:
        holdout_rows = pd.read_csv(a.holdout_csv).to_dict("records")
    if not reference_rows or not holdout_rows:
        raise SystemExit(f"{csv} has no reference/holdout split")

    out = {"source": os.path.basename(csv),
           "holdout_source": os.path.basename(a.holdout_csv) if a.holdout_csv else os.path.basename(csv),
           "regime": a.regime, "n_reference": len(reference_rows),
           "n_holdout": len(holdout_rows), "all_features": run(reference_rows, holdout_rows, FEATURES,
                                                               a.thresholds),
           "ablation": {}}
    for f in FEATURES:
        out["ablation"][f] = run(reference_rows, holdout_rows, (f,), a.thresholds)

    dest = os.path.splitext(a.holdout_csv or csv)[0] + "_summary.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dest}")

    main_cell = out["all_features"][f"at_{a.thresholds[1]:g}"]
    print(f"\n{a.regime}: {out['n_reference']} reference and {out['n_holdout']} held-out datasets")
    print(f"  accuracy {main_cell['accuracy']:.2f} (chance {main_cell['chance']:.2f}), "
          f"Brier {main_cell['brier']:.3f}")
    for t in a.thresholds:
        cell = out["all_features"][f"at_{t:g}"]
        acc = cell["accuracy_when_decided"]
        print(f"  confidence >= {t:g}: decided {cell['share_decided']:.2f} of datasets, "
              f"accuracy when decided {acc if acc is None else round(acc, 2)}")
    print("  per law (AUC / recall):", {law: (round(v["auc"], 2), round(v["recall"], 2))
                                        for law, v in main_cell["per_law"].items()})
    print("  reliability:", [(round(b["mean_confidence"], 2), round(b["share_correct"], 2), b["n"])
                             for b in main_cell["reliability"]])
    print("  single-statistic accuracy:",
          {f: round(out["ablation"][f][f"at_{a.thresholds[1]:g}"]["accuracy"], 2) for f in FEATURES})


if __name__ == "__main__":
    main()
