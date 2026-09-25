"""Collect every C4 summary (base runs and the 2026-09-12 sweeps) into one long table.

Reads results/c4_*_summary.json as written by analyze_c4.py. Writes results/c4_sweeps_long.csv with one row per
regime, sweep, undefined policy, law, factor level and statistic:

    kind        whale | song | finch
    config      finch generator configuration (primary, sensA, sensB); empty otherwise
    sweep       base | rep1000 | thr<x> | sigma<x> | tail<x> | zipfexp<x> | themes<n> | variants<n> | ppc<n>
    auc, lo, hi Zipf-vs-law AUC and its bootstrap 95% CI for that statistic
    r2_zipf, r2_law, cuts_zipf, cuts_law    mean observed R^2 and cuts per sequence, for confound checks
    share_zipf, share_law                   share of datasets clearing every published z for the regime

Sweeps vary one thing at a time against the base run of the same regime, so a row is compared with its base row.

Usage
-----
    python src/analyze_sweeps.py
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import glob
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SWEEP = re.compile(r"^(rep\d+|thr[\d.]+|sigma[\d.]+|tail[\d.]+|zipfexp[\d.]+|themes\d+|variants\d+|ppc\d+)$")


def parse_name(path):
    """(kind, config, sweep) from a summary file name, or None for files that are not C4 summaries."""
    m = re.fullmatch(r"c4_(.+)_summary\.json", os.path.basename(path))
    if not m:
        return None
    parts = m.group(1).split("_")
    if parts[:2] == ["whale", "song"]:
        kind, config, rest = "song", "", parts[2:]
    elif parts[0] == "whale":
        kind, config, rest = "whale", "", parts[1:]
    elif parts[0] == "finch" and len(parts) >= 2:
        kind, config, rest = "finch", parts[1], parts[2:]
    else:
        return None
    if not rest:
        return kind, config, "base"
    if len(rest) == 1 and SWEEP.match(rest[0]):
        return kind, config, rest[0]
    return None


def collect(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "c4_*_summary.json"))):
        name = parse_name(path)
        if name is None:
            continue
        kind, config, sweep = name
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)
        for policy, pol in summary["policies"].items():
            for key, stats in pol["auc_zipf_vs"].items():
                law, level = key.split("|")
                zipf_key = f"zipf|{level}"
                d_z, d_l = pol["describe"].get(zipf_key, {}), pol["describe"].get(key, {})
                s_z = pol["share_clearing_published_z"].get(zipf_key, {})
                s_l = pol["share_clearing_published_z"].get(key, {})
                for stat, cell in stats.items():
                    rows.append({
                        "kind": kind, "config": config, "sweep": sweep, "policy": policy, "law": law,
                        "level": level, "statistic": stat, "auc": cell["auc"],
                        "lo": cell["ci95"][0], "hi": cell["ci95"][1],
                        "r2_zipf": d_z.get("observed_mean_r2", {}).get("mean"),
                        "r2_law": d_l.get("observed_mean_r2", {}).get("mean"),
                        "cuts_zipf": d_z.get("cuts_per_sequence", {}).get("mean"),
                        "cuts_law": d_l.get("cuts_per_sequence", {}).get("mean"),
                        "share_zipf": min(s_z.values()) if s_z else None,
                        "share_law": min(s_l.values()) if s_l else None,
                    })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    df = collect(a.dir)
    if df.empty:
        raise SystemExit("no C4 summaries found")
    dest = os.path.join(a.dir, "c4_sweeps_long.csv")
    df.to_csv(dest, index=False)
    print(f"wrote {dest} ({len(df)} rows)")
    skip = df[df.policy == "skip"]
    wide = skip.pivot_table(index=["kind", "config", "sweep", "law", "level"], columns="statistic",
                            values="auc").round(2)
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print("\nZipf-vs-law AUC, undefined=skip:")
        print(wide.to_string())


if __name__ == "__main__":
    main()
