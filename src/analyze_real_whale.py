"""Apply the calibration to the real humpback corpus, prior by prior, validating each prior first.

For each reference set this reports, in order:

  1. validation on held-out simulated corpora drawn from the same prior: accuracy, the share given a verdict at
     confidence 0.7 and 0.9 and the accuracy of those verdicts, and how often a held-out corpus is wrongly
     reported as outside the reference set (the containment check's false-alarm rate);
  2. the containment check on the real corpus (`unitsim.calibrate.typicality`);
  3. only then, the posterior and the verdict for the real corpus.

Reference sets: the song reference of the paper (`results/calibration_song.csv`, years sized from Table S1, element
repertoire drawn from 20 to 40 types), and the corpus-matched references of `exp_real_whale.py`.

Writes results/real_whale_calibration.json.

    python src/analyze_real_whale.py
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import json
import os

import numpy as np
import pandas as pd

from unitsim.calibrate import (
    FEATURES,
    build_reference,
    contains,
    evaluate,
    posterior,
    typicality,
    verdict,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SETS = {
    "paper_song": "calibration_song.csv",
    "matched_song": "real_whale_reference_song.csv",
    "matched_song_labels": "real_whale_reference_song_labels.csv",
    "matched_broad_labels": "real_whale_reference_broad_labels.csv",
}
MIN_TYPICALITY = 0.01


def records(df):
    return [{**{f: r[f] for f in FEATURES}, "law": r["law"]} for _, r in df.iterrows()]


def analyse(name, path, observation):
    df = pd.read_csv(path)
    ref_rows, hold_rows = records(df[df["set"] == "reference"]), records(df[df["set"] == "holdout"])
    ref = build_reference(ref_rows)
    val = {str(t): evaluate(ref, hold_rows, min_probability=t) for t in (0.7, 0.9)}
    hold = [r for r in hold_rows if all(np.isfinite(float(r[f])) for f in FEATURES)]
    false_outside = float(np.mean([not contains(ref, r, MIN_TYPICALITY) for r in hold]))
    typ = typicality(ref, observation)
    post = posterior(ref, observation)
    inside = max(typ.values()) >= MIN_TYPICALITY
    ranges = {f: {"min": float(df[f].min()), "max": float(df[f].max()),
                  "share_at_or_above_observed": float(np.mean(df[f] >= observation[f]))} for f in FEATURES}
    out = {"reference": os.path.basename(path), "n_reference": len(ref_rows), "n_holdout": len(hold_rows),
           "validation": {t: {k: v[k] for k in ("accuracy", "share_decided", "accuracy_when_decided", "brier")}
                          for t, v in val.items()},
           "holdout_reported_outside": false_outside,
           "real_typicality": typ, "real_inside": inside, "real_posterior": post,
           "real_verdict_0.7": verdict(post, 0.7) if inside else None,
           "real_verdict_0.9": verdict(post, 0.9) if inside else None,
           "feature_ranges": ranges}
    print(f"\n== {name} ({out['reference']})")
    for t, v in out["validation"].items():
        print(f"   validation at {t}: " + ", ".join(f"{k} {v[k]:.3f}" for k in v if v[k] is not None))
    print(f"   held-out corpora reported outside: {false_outside:.3f}")
    print("   real corpus typicality: " + ", ".join(f"{k} {v:.3f}" for k, v in typ.items())
          + f"  -> {'inside' if inside else 'OUTSIDE'}")
    print("   real corpus posterior: " + ", ".join(f"{k} {v:.3f}" for k, v in post.items())
          + f"  -> verdict 0.7: {out['real_verdict_0.7']}, 0.9: {out['real_verdict_0.9']}")
    return out


def main():
    obs = json.load(open(os.path.join(RES, "real_whale_features.json"), encoding="utf-8"))["by_threshold"]["0.5"]
    observation = {f: obs[f] for f in FEATURES}
    print("real corpus: " + ", ".join(f"{f} {observation[f]:.4g}" for f in FEATURES))
    out = {"observation": observation, "min_typicality": MIN_TYPICALITY, "sets": {}}
    for name, fname in SETS.items():
        path = os.path.join(RES, fname)
        if os.path.exists(path):
            out["sets"][name] = analyse(name, path, observation)
        else:
            print(f"\n== {name}: {fname} not found, skipped")
    with open(os.path.join(RES, "real_whale_calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/real_whale_calibration.json")


if __name__ == "__main__":
    main()
