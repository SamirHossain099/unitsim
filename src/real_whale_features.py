"""The four calibration statistics computed on the real humpback corpus of Arnon et al. 2025.

Data S2 of that paper (science.adq7055_data_s2.csv, in refs/arnon2025_data/, downloaded 2026-10-05 from the
Science supplementary materials) holds one row per recording: the year ("song_type") and the sequence of
element labels ("string"). Its terms: "Data used in the analysis is available for the purposes of reproducing
or extending the Zipfian analysis."

The corpus is grouped as the published analysis groups it, recordings within a year, and run through the same
pipeline as every simulated dataset (`exp_calibration.features`): transitional probabilities pooled over a year's
recordings, cuts at the threshold, unit counts per year. Nothing here sees a frequency law.

Writes results/real_whale_features.json.

    python src/real_whale_features.py
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd

from exp_calibration import features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "refs", "arnon2025_data", "science.adq7055_data_s2.csv")
OUT = os.path.join(ROOT, "results", "real_whale_features.json")
THRESHOLDS = (0.25, 0.5, 0.75)


def load_groups(path=DATA):
    """One list of integer-coded recordings per year, in year order, with each year's labels coded separately."""
    df = pd.read_csv(path)
    groups, sizes = [], []
    for year, rows in df.groupby("song_type", sort=True):
        recs = [ast.literal_eval(s) for s in rows["string"]]
        labels = {lab: i for i, lab in enumerate(sorted({lab for r in recs for lab in r}))}
        groups.append([np.array([labels[lab] for lab in r], dtype=np.int64) for r in recs])
        sizes.append({"year": int(year), "recordings": len(recs), "elements": int(sum(len(r) for r in recs)),
                      "element_types": len(labels)})
    return groups, sizes


def main():
    groups, sizes = load_groups()
    sha = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
    out = {"source": os.path.relpath(DATA, ROOT).replace(os.sep, "/"), "sha256": sha, "years": sizes,
           "elements_total": sum(s["elements"] for s in sizes), "by_threshold": {}}
    for t in THRESHOLDS:
        f = features(groups, t)
        cuts_per_year = f["cuts_per_sequence"] * sum(s["recordings"] for s in sizes) / len(sizes)
        out["by_threshold"][str(t)] = {**f, "cuts_per_year": cuts_per_year}
        print(f"threshold {t}: " + ", ".join(f"{k}={v:.4g}" for k, v in f.items() if isinstance(v, float))
              + f", cuts per year {cuts_per_year:.1f}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
