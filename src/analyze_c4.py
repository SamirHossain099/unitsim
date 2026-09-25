"""Summarise C4 at the published inference unit.

Reads results/c4_<regime>_<policy>.csv for each policy present. Writes results/c4_<regime>_summary.json.

Three questions per regime:
  1. Discriminability: AUC of each statistic for a Zipfian dataset against each non-Zipfian law, with a
     bootstrap CI over simulated datasets. 0.5 = cannot tell them apart; below 0.5 = points the wrong way.
  2. The published bar: what share of datasets from each law clears the z the paper reported.
  3. Policy sensitivity: datasets are paired across policies (same seed, same random draws; only the
     aggregation differs), so the per-dataset change in z between `skip` and `zero` is reported directly.
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import os

import numpy as np
import pandas as pd

from analyze_pilot_c1_c4 import auc, auc_ci
from exp_c4_inference_unit import BASELINES, LAWS, REGIMES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: z reported at threshold 0.5, each read verbatim (VERIFICATION.md). Whale: Arnon et al. 2025, AAM lines
#: 155-157 (shuffled) and 169 (rotated); no bigram baseline. Finch: Kirby et al. 2026, results text.
PUBLISHED_Z = {
    "whale": {"shuffle": 23.09, "rotate": 14.43},
    "finch": {"shuffle": 17.05, "rotate": 15.69, "bigram": 6.77},
}
STATISTICS = ["observed_mean_r2", "shuffle_z", "rotate_z", "bigram_z"]
DESCRIBE = STATISTICS + ["boundary_f1", "cuts_per_sequence", "observed_undefined_groups"] + \
    [f"{b}_undefined_rate" for b in BASELINES]
#: Recorded only by runs made after exp_c4_inference_unit.py gained them; used when present.
OPTIONAL = ["observed_mean_slope", "observed_mean_unit_types"]


def describe(series):
    s = series.dropna()
    return {"mean": float(s.mean()), "sd": float(s.std(ddof=1)) if s.size > 1 else None,
            "min": float(s.min()), "max": float(s.max()), "n": int(s.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=REGIMES, required=True)
    ap.add_argument("--tag", default=None,
                    help="configuration tag in the file name, e.g. primary for c4_finch_primary_skip.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--factor", default="mean_repeats",
                    help="column splitting conditions besides law, e.g. stickiness for song runs")
    ap.add_argument("--dir", default=os.path.join(ROOT, "results"), help="directory holding the CSVs")
    a = ap.parse_args()
    label = {"mean_repeats": "repeats"}.get(a.factor, a.factor)

    stem = f"c4_{a.regime}_{a.tag}_" if a.tag else f"c4_{a.regime}_"
    paths = {}
    for policy in ("raise", "skip", "zero"):
        p = os.path.join(a.dir, f"{stem}{policy}.csv")
        if os.path.exists(p):
            paths[policy] = p
    if not paths:
        raise SystemExit(f"no results/{stem}<policy>.csv files found")
    frames = {}
    for policy, p in paths.items():
        df = pd.read_csv(p)
        found = set(df.undefined_policy.unique())
        if found != {policy}:
            raise AssertionError(f"{os.path.basename(p)} holds policies {found}, expected {policy}")
        frames[policy] = df
    rng = np.random.default_rng(a.seed)
    out = {"regime": a.regime, "tag": a.tag, "factor": a.factor, "published_z": PUBLISHED_Z[a.regime],
           "policies": {}}

    for policy, df in frames.items():
        pol = {"file": os.path.basename(paths[policy]),
               "describe": {}, "auc_zipf_vs": {}, "share_clearing_published_z": {}}
        if a.factor not in df.columns:
            raise SystemExit(f"{os.path.basename(paths[policy])} has no column {a.factor!r}")
        stats = STATISTICS + [c for c in OPTIONAL if c in df.columns]
        describe_cols = DESCRIBE + [c for c in OPTIONAL if c in df.columns]
        for (law, rep), g in df.groupby(["law", a.factor]):
            pol["describe"][f"{law}|{label}={rep:g}"] = {c: describe(g[c]) for c in describe_cols}
            pol["share_clearing_published_z"][f"{law}|{label}={rep:g}"] = {
                b: float((g[f"{b}_z"] > z).mean()) for b, z in PUBLISHED_Z[a.regime].items()}
        for rep in sorted(df[a.factor].unique()):
            zipf = df[(df.law == "zipf") & (df[a.factor] == rep)]
            for law in LAWS:
                if law == "zipf":
                    continue
                other = df[(df.law == law) & (df[a.factor] == rep)]
                if zipf.empty or other.empty:
                    continue
                pol["auc_zipf_vs"][f"{law}|{label}={rep:g}"] = {
                    s: {"auc": auc(zipf[s], other[s]),
                        "ci95": auc_ci(zipf[s], other[s], rng, a.n_boot)} for s in stats}
        out["policies"][policy] = pol

    if {"skip", "zero"} <= set(frames):
        keys = ["law", a.factor, "dataset"]
        m = frames["skip"].merge(frames["zero"], on=keys, suffixes=("_skip", "_zero"))
        if len(m) != len(frames["skip"]):
            raise AssertionError("skip and zero runs do not cover the same datasets")
        # With no undefined observed group both policies compute the same observed mean, so any
        # difference on those datasets means the two runs did not see the same data.
        clean = m.observed_undefined_groups_skip == 0
        if not np.allclose(m.observed_mean_r2_skip[clean], m.observed_mean_r2_zero[clean]):
            raise AssertionError("observed R^2 differs between policies on datasets with no undefined "
                                 "group: the skip and zero runs are not paired")
        out["policy_sensitivity"] = {
            b: {"mean_abs_dz": float((m[f"{b}_z_skip"] - m[f"{b}_z_zero"]).abs().mean()),
                "max_abs_dz": float((m[f"{b}_z_skip"] - m[f"{b}_z_zero"]).abs().max())}
            for b in BASELINES}

    dest = os.path.join(a.dir, f"{stem}summary.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dest}")
    for policy, pol in out["policies"].items():
        print(f"\n[{policy}] share of datasets clearing the published z:")
        for k, v in pol["share_clearing_published_z"].items():
            print(f"  {k:24s} " + "  ".join(f"{b}>{PUBLISHED_Z[a.regime][b]}: {s:.2f}"
                                             for b, s in v.items()))
        print(f"[{policy}] AUC, Zipf vs other law:")
        for k, v in pol["auc_zipf_vs"].items():
            print(f"  {k:24s} " + "  ".join(
                f"{s}={v[s]['auc']:.2f}[{v[s]['ci95'][0]:.2f},{v[s]['ci95'][1]:.2f}]"
                for s in STATISTICS))
    if "policy_sensitivity" in out:
        print("\npolicy sensitivity |z_skip - z_zero|:", json.dumps(out["policy_sensitivity"]))


if __name__ == "__main__":
    main()
