"""The paper's supplementary tables, written from results/ so no number is typed by hand.

    S1  AUC with bootstrap 95% interval for Zipf against each alternative at the published settings (main Table 1)
    S2  Song parameters, one factor at a time: AUC with interval for mean R^2, the three baseline z and the slope
    S3  Threshold: AUC with interval, whale words, whale song and finch (main Figure 2, Table 2)
    S4  Shape of the alternative law: AUC with interval for mean R^2 and the baselines (main Figure 3)
    S5  Power-law tests on true words and discovered units: AUC with interval (main Figure 4)
    S6  Calibration validation under the matched and the wide prior (main Figure 5)
    S7  Calibration: per-law discrimination and the single-statistic ablation

Writes SUPPLEMENT-TABLES.md and tables/table_s{n}.csv. Every value is read from a summary JSON; the markdown is a
rendering of the CSVs.

Usage
-----
    python src/tables_paper.py
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "tables")
LAWS = ("uniform", "lognormal", "geometric")
HEAVY = ("lognormal", "geometric")
C4_STATS = [("observed_mean_r2", "mean R²"), ("shuffle_z", "shuffled z"), ("rotate_z", "rotated z"),
            ("bigram_z", "bigram z")]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


def c4_cell(summary, key, stat):
    c = load(summary)["policies"]["skip"]["auc_zipf_vs"][key][stat]
    return c["auc"], c["ci95"][0], c["ci95"][1]


def fmt(v):
    auc, lo, hi = v
    return f"{auc:.2f} [{lo:.2f}, {hi:.2f}]"


def rows_from(spec, stats, cell):
    """spec: list of (label columns dict, summary, key). Returns tidy rows and wide display rows."""
    tidy, wide = [], []
    for labels, summary, key in spec:
        show = dict(labels)
        for stat, name in stats:
            auc, lo, hi = cell(summary, key, stat)
            tidy.append({**labels, "statistic": stat, "auc": auc, "ci_low": lo, "ci_high": hi})
            show[name] = fmt((auc, lo, hi))
        wide.append(show)
    return tidy, wide


def md_table(wide):
    cols = list(wide[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in wide]
    return "\n".join(lines)


def table_s1():
    regimes = [("whale, independent words, m = 1", "c4_whale_summary.json", "repeats=1"),
               ("whale, independent words, m = 5", "c4_whale_summary.json", "repeats=5"),
               ("finch", "c4_finch_primary_summary.json", "repeats=1"),
               ("whale song, s = 0", "c4_whale_song_summary.json", "stickiness=0"),
               ("whale song, s = 0.8", "c4_whale_song_summary.json", "stickiness=0.8")]
    spec = [({"regime": r, "alternative": law}, s, f"{law}|{lv}") for r, s, lv in regimes for law in LAWS]
    return rows_from(spec, C4_STATS, c4_cell)


def table_s2():
    settings = [("8 themes x 3 variants, 40 renditions", "c4_whale_song_summary.json"),
                ("4 themes", "c4_whale_song_themes4_summary.json"),
                ("12 themes", "c4_whale_song_themes12_summary.json"),
                ("2 variants per theme", "c4_whale_song_variants2_summary.json"),
                ("5 variants per theme", "c4_whale_song_variants5_summary.json"),
                ("20 renditions per cycle", "c4_whale_song_ppc20_summary.json"),
                ("80 renditions per cycle", "c4_whale_song_ppc80_summary.json")]
    spec = [({"song": name, "stickiness": s, "alternative": law}, f, f"{law}|stickiness={s}")
            for name, f in settings for s in ("0", "0.8") for law in LAWS]
    return rows_from(spec, C4_STATS + [("observed_mean_slope", "slope")], c4_cell)


def table_s3():
    runs = [("whale, independent words", "repeats=1", "m = 1",
             [("0.25", "c4_whale_thr0.25_summary.json"), ("0.5", "c4_whale_summary.json"),
              ("0.75", "c4_whale_thr0.75_summary.json")]),
            ("whale, independent words", "repeats=5", "m = 5",
             [("0.25", "c4_whale_thr0.25_summary.json"), ("0.5", "c4_whale_summary.json"),
              ("0.75", "c4_whale_thr0.75_summary.json")]),
            ("whale song", "stickiness=0", "s = 0",
             [("0.25", "c4_whale_song_thr0.25_summary.json"), ("0.5", "c4_whale_song_summary.json"),
              ("0.75", "c4_whale_song_thr0.75_summary.json")]),
            ("whale song", "stickiness=0.8", "s = 0.8",
             [("0.25", "c4_whale_song_thr0.25_summary.json"), ("0.5", "c4_whale_song_summary.json"),
              ("0.75", "c4_whale_song_thr0.75_summary.json")]),
            ("finch", "repeats=1", "", [("0.3", "c4_finch_primary_thr0.3_summary.json"),
                                        ("0.5", "c4_finch_primary_summary.json"),
                                        ("0.7", "c4_finch_primary_thr0.7_summary.json")])]
    spec = [({"regime": reg, "level": lab, "threshold": thr, "alternative": law}, f, f"{law}|{lv}")
            for reg, lv, lab, thr_files in runs for thr, f in thr_files for law in LAWS]
    return rows_from(spec, C4_STATS, c4_cell)


def table_s4():
    shapes = [("lognormal sigma 0.5", "sigma0.5", ["lognormal"]), ("lognormal sigma 1.5", "sigma1.5", ["lognormal"]),
              ("lognormal sigma 2", "sigma2", ["lognormal"]), ("geometric tail 0.2", "tail0.2", ["geometric"]),
              ("geometric tail 0.01", "tail0.01", ["geometric"]), ("Zipf exponent 0.8", "zipfexp0.8", list(LAWS)),
              ("Zipf exponent 1.2", "zipfexp1.2", list(LAWS))]
    regimes = [("whale, independent words", "c4_whale", ("repeats=1", "m = 1"), ("repeats=5", "m = 5")),
               ("finch", "c4_finch_primary", ("repeats=1", ""),),
               ("whale song", "c4_whale_song", ("stickiness=0", "s = 0"), ("stickiness=0.8", "s = 0.8"))]
    spec = []
    for reg, prefix, *levels in regimes:
        for name, tag, laws in shapes:
            for lv, lab in levels:
                for law in laws:
                    spec.append(({"regime": reg, "level": lab, "change": name, "alternative": law},
                                 f"{prefix}_{tag}_summary.json", f"{law}|{lv}"))
    return rows_from(spec, C4_STATS, c4_cell)


def c3_cell(summary, key, stat):
    c = load(summary)["auc_zipf_vs"][key][stat]
    return c["auc"], c["ci95"][0], c["ci95"][1]


def table_s5():
    regimes = [("whale, independent words, m = 1", "c3_whale_summary.json", "repeats=1"),
               ("whale, independent words, m = 5", "c3_whale_summary.json", "repeats=5"),
               ("finch", "c3_finch_summary.json", "repeats=1"),
               ("whale song, s = 0", "c3_song_summary.json", "stickiness=0"),
               ("whale song, s = 0.8", "c3_song_summary.json", "stickiness=0.8")]
    stats = [("loglog_r2", "log-log R²"), ("tail_fraction", "tail fraction"), ("gof_p", "bootstrap p"),
             ("lr_normalized_R", "likelihood ratio"), ("csn_alpha", "exponent")]
    spec = [({"regime": r, "units": src, "alternative": law}, s, f"{src}|{law}|{lv}")
            for r, s, lv in regimes for src in ("true", "discovered") for law in LAWS]
    return rows_from(spec, stats, c3_cell)



def table_s6():
    """Calibration validation, matched and wide prior, at three confidence thresholds."""
    rows = []
    for regime, label in (("whale", "whale, independent words"), ("finch", "finch"), ("song", "whale song")):
        for prior, name in (("", "matched"), ("_wide", "wide")):
            s = load(f"calibration_{regime}{prior}_summary.json")["all_features"]
            for t in ("0.5", "0.7", "0.9"):
                cell = s[f"at_{t}"]
                rows.append({"regime": label, "prior": name, "confidence threshold": t,
                             "held-out datasets": cell["n"], "accuracy": round(cell["accuracy"], 3),
                             "Brier": round(cell["brier"], 3),
                             "share decided": round(cell["share_decided"], 3),
                             "accuracy when decided": None if cell["accuracy_when_decided"] is None
                             else round(cell["accuracy_when_decided"], 3)})
    return rows, rows


def table_s7():
    """Per-law discrimination and the single-statistic ablation, matched prior."""
    rows = []
    for regime, label in (("whale", "whale, independent words"), ("finch", "finch"), ("song", "whale song")):
        s = load(f"calibration_{regime}_summary.json")
        cell = s["all_features"]["at_0.7"]
        for law, v in sorted(cell["per_law"].items()):
            rows.append({"regime": label, "law": law, "AUC": round(v["auc"], 3), "recall": round(v["recall"], 3),
                         "accuracy, all four statistics": round(cell["accuracy"], 3),
                         **{f"accuracy, {f} only": round(s["ablation"][f]["at_0.7"]["accuracy"], 3)
                            for f in ("observed_mean_r2", "observed_mean_slope", "observed_mean_unit_types",
                                      "cuts_per_sequence")}})
    return rows, rows

TABLES = [
    ("S1", table_s1, "AUC for Zipfian against non-Zipfian sources at the published settings, with bootstrap 95% "
     "intervals over 30 datasets per law (main Table 1). Undefined groups skipped."),
    ("S2", table_s2, "Song parameters varied one at a time around the default song (8 themes of 3 phrase variants, "
     "40 renditions per cycle). Whale-sized years, threshold 0.5. AUC with bootstrap 95% interval; slope is the mean "
     "rank-frequency slope."),
    ("S3", table_s3, "Segmentation threshold. AUC with bootstrap 95% interval at the published thresholds. Datasets "
     "are identical across thresholds; only the segmentation changes. Undefined groups skipped."),
    ("S4", table_s4, "Shape of the alternative law. Each row changes one shape parameter from the defaults (Zipf "
     "exponent 1, lognormal sigma 1, geometric tail ratio 0.05); Zipf-exponent rows keep the alternatives at their "
     "defaults. AUC with bootstrap 95% interval, threshold 0.5."),
    ("S5", table_s5, "Maximum-likelihood power-law tests on true word counts and on discovered units. AUC with "
     "bootstrap 95% interval; statistics averaged over the years or recordings of each dataset before the AUC. For the "
     "exponent, an AUC near 0 means separation in the opposite direction."),
    ("S6", table_s6, "Calibration validation (paper, section 3.7). Held-out simulated datasets of known law, "
     "scored against reference densities fitted on an independent half. The wide prior draws held-out datasets from "
     "ranges the reference never covered. Chance accuracy is 0.25."),
    ("S7", table_s7, "Calibration, matched prior, confidence threshold 0.7: discrimination for each law "
     "(one-vs-rest AUC over the posterior, and recall), beside the accuracy of the full four-statistic vector and of "
     "each statistic on its own."),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = ["# Electronic supplementary material: tables S1-S7", "",
             "Supplement to *Calibrating Zipfian inference from segmented animal vocal sequences with known sources*. "
             "Every value is read from the results files in the accompanying code archive by `src/tables_paper.py`, "
             "and the archive's tests check that these tables regenerate identically. AUC: the probability that a "
             "randomly chosen Zipfian dataset scores higher than a randomly chosen dataset of the alternative law. "
             "0.5: no separation; below 0.5: ranked backwards.", ""]
    for name, build, caption in TABLES:
        tidy, wide = build()
        pd.DataFrame(tidy).to_csv(os.path.join(OUT, f"table_{name.lower()}.csv"), index=False)
        parts += [f"## Table {name}", "", caption, "", md_table(wide), ""]
        print(f"Table {name}: {len(wide)} rows")
    with open(os.path.join(ROOT, "SUPPLEMENT-TABLES.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(parts))
    print("wrote SUPPLEMENT-TABLES.md and tables/*.csv")


if __name__ == "__main__":
    main()
