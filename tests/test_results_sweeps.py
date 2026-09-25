"""Pins the numbers quoted in FINDINGS.md F12 (replicate count), F13 (threshold), F14 (song parameters) and F15 (C3 at
finch scale and under song structure) to their results files (RESEARCH.md standing rule 4). Red means FINDINGS.md is
stale."""
import json
import os

import pandas as pd
import pytest
from scipy.stats import kruskal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LAWS = ("uniform", "lognormal", "geometric")
HEAVY = ("lognormal", "geometric")
ZSTATS = ("shuffle_z", "rotate_z", "bigram_z")
TABLE_STATS = ("observed_mean_r2",) + ZSTATS
TOL = 5.01e-3


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        pytest.skip(f"results/{name} not present")
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return pd.read_csv(path)


def auc(name, policy="skip"):
    return load(name)["policies"][policy]["auc_zipf_vs"]


# ---- F12 replicate count ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base,rep,n_cells,largest,n_big,share", [
    ("c4_whale_summary.json", "c4_whale_rep1000_summary.json", 18, 0.15, 2, 0.13),
    ("c4_whale_song_summary.json", "c4_whale_song_rep1000_summary.json", 18, 0.07, 0, 0.13),
    ("c4_finch_primary_summary.json", "c4_finch_primary_rep1000_summary.json", 9, 0.13, 1, 0.10),
])
def test_f12_replicate_count(base, rep, n_cells, largest, n_big, share):
    b, r = load(base)["policies"]["skip"], load(rep)["policies"]["skip"]
    dz = [abs(r["auc_zipf_vs"][k][s]["auc"] - b["auc_zipf_vs"][k][s]["auc"])
          for k in r["auc_zipf_vs"] for s in ZSTATS]
    assert len(dz) == n_cells
    assert max(dz) == pytest.approx(largest, abs=TOL)
    assert sum(d > 0.10 for d in dz) == n_big
    for k in r["auc_zipf_vs"]:
        assert r["auc_zipf_vs"][k]["observed_mean_r2"]["auc"] == b["auc_zipf_vs"][k]["observed_mean_r2"]["auc"]
    ds = [abs(v - b["share_clearing_published_z"][k][bl])
          for k, cells in r["share_clearing_published_z"].items() for bl, v in cells.items()]
    assert max(ds) == pytest.approx(share, abs=TOL)


def test_f12_readings():
    rep = auc("c4_whale_rep1000_summary.json")
    base = auc("c4_whale_summary.json")["geometric|repeats=1"]["shuffle_z"]
    geo = rep["geometric|repeats=1"]["shuffle_z"]
    assert geo["auc"] == pytest.approx(0.77, abs=TOL) and geo["ci95"][0] > 0.5
    assert base["auc"] == pytest.approx(0.62, abs=TOL) and base["ci95"][0] < 0.5
    cells = [rep["lognormal|repeats=1"][s] for s in ZSTATS]
    assert all(c["ci95"][0] < 0.5 < c["ci95"][1] for c in cells)
    assert min(c["auc"] for c in cells) == pytest.approx(0.50, abs=TOL)
    assert max(c["auc"] for c in cells) == pytest.approx(0.61, abs=TOL)
    shifts = {("c4_whale_summary.json", "c4_whale_rep1000_summary.json", "uniform|repeats=1"): (0.89, 0.99),
              ("c4_whale_song_summary.json", "c4_whale_song_rep1000_summary.json", "geometric|stickiness=0.8"):
                  (0.25, 0.18),
              ("c4_finch_primary_summary.json", "c4_finch_primary_rep1000_summary.json", "uniform|repeats=1"):
                  (0.49, 0.36)}
    for (b, r, key), (before, after) in shifts.items():
        assert auc(b)[key]["shuffle_z"]["auc"] == pytest.approx(before, abs=TOL)
        assert auc(r)[key]["shuffle_z"]["auc"] == pytest.approx(after, abs=TOL)


# ---- F13 threshold ---------------------------------------------------------------------------------------------------

THRESHOLD_AUC = {
    "c4_whale_thr0.25_summary.json": {
        "repeats=1": {"uniform": (1.00, 0.99, 0.88, 0.01), "lognormal": (1.00, 0.99, 0.86, 0.31),
                      "geometric": (1.00, 1.00, 0.96, 0.34)},
        "repeats=5": {"uniform": (1.00, 1.00, 0.69, 0.03), "lognormal": (0.97, 0.83, 0.67, 0.27),
                      "geometric": (0.99, 0.99, 0.60, 0.08)}},
    "c4_whale_summary.json": {
        "repeats=1": {"uniform": (1.00, 0.89, 1.00, 0.87), "lognormal": (1.00, 0.58, 0.59, 0.51),
                      "geometric": (1.00, 0.62, 0.88, 0.77)},
        "repeats=5": {"uniform": (1.00, 0.30, 0.59, 0.00), "lognormal": (0.94, 0.32, 0.55, 0.09),
                      "geometric": (0.99, 0.25, 0.46, 0.00)}},
    "c4_whale_thr0.75_summary.json": {
        "repeats=1": {"uniform": (1.00, 1.00, 1.00, 1.00), "lognormal": (1.00, 0.85, 0.99, 1.00),
                      "geometric": (1.00, 1.00, 1.00, 1.00)},
        "repeats=5": {"uniform": (1.00, 0.77, 0.99, 0.95), "lognormal": (0.99, 0.50, 0.74, 0.89),
                      "geometric": (1.00, 0.65, 0.87, 0.95)}},
    "c4_whale_song_thr0.25_summary.json": {
        "stickiness=0": {"uniform": (0.96, 0.35, 0.67, 0.04), "lognormal": (0.61, 0.44, 0.54, 0.35),
                         "geometric": (0.70, 0.29, 0.53, 0.18)},
        "stickiness=0.8": {"uniform": (0.90, 0.24, 0.18, 0.00), "lognormal": (0.50, 0.42, 0.48, 0.31),
                           "geometric": (0.60, 0.17, 0.28, 0.10)}},
    "c4_whale_song_summary.json": {
        "stickiness=0": {"uniform": (1.00, 0.97, 0.96, 0.80), "lognormal": (0.50, 0.35, 0.43, 0.44),
                         "geometric": (0.61, 0.26, 0.51, 0.46)},
        "stickiness=0.8": {"uniform": (0.99, 0.34, 0.39, 0.09), "lognormal": (0.28, 0.35, 0.27, 0.38),
                           "geometric": (0.31, 0.25, 0.19, 0.26)}},
    "c4_whale_song_thr0.75_summary.json": {
        "stickiness=0": {"uniform": (1.00, 1.00, 1.00, 1.00), "lognormal": (0.78, 0.60, 0.58, 0.70),
                         "geometric": (0.97, 0.72, 0.74, 0.89)},
        "stickiness=0.8": {"uniform": (1.00, 1.00, 1.00, 1.00), "lognormal": (0.67, 0.49, 0.35, 0.47),
                           "geometric": (0.94, 0.56, 0.28, 0.63)}},
    "c4_finch_primary_thr0.3_summary.json": {
        "repeats=1": {"uniform": (0.86, 0.80, 0.60, 0.17), "lognormal": (0.53, 0.61, 0.64, 0.43),
                      "geometric": (0.65, 0.57, 0.72, 0.48)}},
    "c4_finch_primary_summary.json": {
        "repeats=1": {"uniform": (0.99, 0.49, 0.59, 0.38), "lognormal": (0.64, 0.64, 0.57, 0.60),
                      "geometric": (0.63, 0.70, 0.66, 0.67)}},
    "c4_finch_primary_thr0.7_summary.json": {
        "repeats=1": {"uniform": (1.00, 0.94, 0.96, 0.97), "lognormal": (0.73, 0.69, 0.68, 0.68),
                      "geometric": (0.59, 0.67, 0.60, 0.57)}},
}


@pytest.mark.parametrize("name", sorted(THRESHOLD_AUC))
def test_f13_auc_tables(name):
    cells = auc(name)
    for level, laws in THRESHOLD_AUC[name].items():
        for law, expected in laws.items():
            got = tuple(cells[f"{law}|{level}"][s]["auc"] for s in TABLE_STATS)
            assert got == pytest.approx(expected, abs=TOL), (name, level, law)


def test_f13_readings_ranges():
    t = THRESHOLD_AUC
    bigram_025 = [t["c4_whale_thr0.25_summary.json"]["repeats=1"][law][3] for law in LAWS]
    assert (min(bigram_025), max(bigram_025)) == (0.01, 0.34)
    whale_r2 = [v[0] for name in ("c4_whale_thr0.25_summary.json", "c4_whale_summary.json",
                                  "c4_whale_thr0.75_summary.json")
                for laws in t[name].values() for v in laws.values()]
    assert min(whale_r2) == 0.94
    for name, lo, hi in (("c4_whale_song_thr0.75_summary.json", 0.67, 0.97),
                         ("c4_whale_song_thr0.25_summary.json", 0.50, 0.70),
                         ("c4_whale_song_summary.json", 0.28, 0.61)):
        r2 = [laws[law][0] for laws in t[name].values() for law in HEAVY]
        assert (min(r2), max(r2)) == (lo, hi), name


def test_f13_whale_cuts_and_negative_z():
    cuts = {name: load(name).query("law == 'zipf' and mean_repeats == 1").cuts_per_sequence.mean()
            for name in ("c4_whale_thr0.25_skip.csv", "c4_whale_skip.csv", "c4_whale_thr0.75_skip.csv")}
    assert list(cuts.values()) == pytest.approx([145.8, 232.2, 321.2], abs=0.06)
    z = load("c4_whale_thr0.75_skip.csv").query("law == 'zipf' and mean_repeats == 5")
    assert [z.shuffle_z.mean(), z.rotate_z.mean(), z.bigram_z.mean()] == pytest.approx([18.31, -3.83, -10.41], abs=6e-3)
    song = load("c4_whale_song_thr0.75_skip.csv").query("law == 'zipf'").groupby("stickiness")
    means = song[["shuffle_z", "rotate_z", "bigram_z"]].mean()
    assert means.loc[0.0].tolist() == pytest.approx([2.92, -28.54, -36.67], abs=6e-3)
    assert means.loc[0.8].tolist() == pytest.approx([5.81, -19.56, -30.39], abs=6e-3)


FINCH_CUTS = {
    "c4_finch_primary_thr0.3_skip.csv": {"zipf": 6.56, "uniform": 5.88, "lognormal": 6.32, "geometric": 6.30},
    "c4_finch_primary_skip.csv": {"zipf": 11.40, "uniform": 10.83, "lognormal": 11.15, "geometric": 11.54},
    "c4_finch_primary_thr0.7_skip.csv": {"zipf": 15.57, "uniform": 15.47, "lognormal": 15.53, "geometric": 15.96},
}


def test_f13_finch_cuts_confounds_and_calibration_gap():
    for name, expected in FINCH_CUTS.items():
        df = load(name)
        assert df.groupby("law").cuts_per_sequence.mean().to_dict() == pytest.approx(expected, abs=6e-3)
    thr03, thr07 = load("c4_finch_primary_thr0.3_skip.csv"), load("c4_finch_primary_thr0.7_skip.csv")
    k03 = kruskal(*[g.cuts_per_sequence for _, g in thr03.groupby("law")])
    k07 = kruskal(*[g.cuts_per_sequence for _, g in thr07.groupby("law")])
    assert k03.statistic == pytest.approx(17.27, abs=TOL) and k03.pvalue == pytest.approx(0.0006, abs=5e-5)
    assert k07.statistic == pytest.approx(2.86, abs=TOL) and k07.pvalue == pytest.approx(0.41, abs=TOL)
    gaps = [abs(FINCH_CUTS[n]["zipf"] - pub) for n, pub in (("c4_finch_primary_thr0.3_skip.csv", 5.01),
                                                           ("c4_finch_primary_skip.csv", 10.72),
                                                           ("c4_finch_primary_thr0.7_skip.csv", 16.75))]
    assert gaps == pytest.approx([1.55, 0.68, 1.18], abs=1e-9)


def test_f13_finch_undefined_and_policy():
    und = load("c4_finch_primary_thr0.3_skip.csv").groupby("law").shuffle_undefined_rate.mean()
    assert und.to_dict() == pytest.approx({"zipf": 0.525, "uniform": 0.779, "lognormal": 0.557, "geometric": 0.514},
                                          abs=6e-4)
    base = load("c4_finch_primary_skip.csv").groupby("law").shuffle_undefined_rate.mean()
    assert (base.min(), base.max()) == pytest.approx((0.026, 0.070), abs=6e-4)
    assert load("c4_finch_primary_thr0.7_skip.csv").shuffle_undefined_rate.max() == pytest.approx(0.0003, abs=5e-5)
    for name, mean, largest in (("c4_finch_primary_thr0.3_summary.json", 9.20, 26.77),
                                ("c4_finch_primary_summary.json", 1.22, None),
                                ("c4_finch_primary_thr0.7_summary.json", 0.003, None)):
        sens = load(name)["policy_sensitivity"]["shuffle"]
        assert sens["mean_abs_dz"] == pytest.approx(mean, abs=TOL if mean > 0.01 else 5e-4)
        if largest is not None:
            assert sens["max_abs_dz"] == pytest.approx(largest, abs=TOL)


FINCH_PUBLISHED_Z = {"thr0.3": {"shuffle": 16.32, "rotate": 11.9, "bigram": 9.22},
                     "base": {"shuffle": 17.05, "rotate": 15.69, "bigram": 6.77},
                     "thr0.7": {"rotate": 9.661, "bigram": 1.87}}


@pytest.mark.parametrize("tag,policy,shares", [
    ("thr0.3", "skip", {"zipf": 0.37, "uniform": 0.20, "lognormal": 0.27, "geometric": 0.37}),
    ("thr0.3", "zero", {"zipf": 0.43, "uniform": 0.97, "lognormal": 0.57, "geometric": 0.50}),
    ("base", "skip", {"zipf": 0.73, "uniform": 0.73, "lognormal": 0.53, "geometric": 0.33}),
    ("base", "zero", {"zipf": 0.70, "uniform": 0.73, "lognormal": 0.53, "geometric": 0.37}),
    ("thr0.7", "skip", {"zipf": 0.27, "uniform": 0.00, "lognormal": 0.13, "geometric": 0.27}),
    ("thr0.7", "zero", {"zipf": 0.27, "uniform": 0.00, "lognormal": 0.13, "geometric": 0.27}),
])
def test_f13_finch_shares_against_threshold_specific_z(tag, policy, shares):
    name = f"c4_finch_primary_{policy}.csv" if tag == "base" else f"c4_finch_primary_{tag}_{policy}.csv"
    df = load(name)
    clears = pd.concat([df[f"{b}_z"] > z for b, z in FINCH_PUBLISHED_Z[tag].items()], axis=1).all(axis=1)
    assert clears.groupby(df.law).mean().to_dict() == pytest.approx(shares, abs=TOL)


# ---- F14 song parameters ---------------------------------------------------------------------------------------------

SONG_SETTINGS = {  # summary: {law: (mean R^2 0, 0.8), (best z 0, 0.8), (slope 0, 0.8)}
    "c4_whale_song_summary.json": {"lognormal": ((0.50, 0.28), (0.44, 0.38), (0.79, 0.76)),
                                   "geometric": ((0.61, 0.31), (0.51, 0.26), (0.92, 0.83))},
    "c4_whale_song_themes4_summary.json": {"lognormal": ((0.35, 0.35), (0.31, 0.38), (0.66, 0.46)),
                                           "geometric": ((0.20, 0.26), (0.24, 0.34), (0.75, 0.52))},
    "c4_whale_song_themes12_summary.json": {"lognormal": ((0.54, 0.45), (0.47, 0.37), (0.92, 0.77)),
                                            "geometric": ((0.67, 0.41), (0.44, 0.26), (0.99, 0.87))},
    "c4_whale_song_variants2_summary.json": {"lognormal": ((0.30, 0.24), (0.34, 0.36), (0.66, 0.64)),
                                             "geometric": ((0.36, 0.17), (0.44, 0.23), (0.83, 0.66))},
    "c4_whale_song_variants5_summary.json": {"lognormal": ((0.78, 0.38), (0.57, 0.34), (0.91, 0.79)),
                                             "geometric": ((0.92, 0.36), (0.62, 0.21), (0.98, 0.89))},
    "c4_whale_song_ppc20_summary.json": {"lognormal": ((0.49, 0.43), (0.50, 0.52), (0.88, 0.88)),
                                         "geometric": ((0.62, 0.44), (0.63, 0.37), (0.98, 0.90))},
    "c4_whale_song_ppc80_summary.json": {"lognormal": ((0.52, 0.40), (0.42, 0.42), (0.89, 0.66)),
                                         "geometric": ((0.64, 0.28), (0.50, 0.28), (0.95, 0.66))},
}
LEVELS = ("stickiness=0", "stickiness=0.8")


@pytest.mark.parametrize("name", sorted(SONG_SETTINGS))
def test_f14_table(name):
    cells = auc(name)
    for law, (r2, best, slope) in SONG_SETTINGS[name].items():
        for i, level in enumerate(LEVELS):
            c = cells[f"{law}|{level}"]
            assert c["observed_mean_r2"]["auc"] == pytest.approx(r2[i], abs=TOL)
            assert max(c[s]["auc"] for s in ZSTATS) == pytest.approx(best[i], abs=TOL)
            assert c["observed_mean_slope"]["auc"] == pytest.approx(slope[i], abs=TOL)


def test_f14_readings():
    best, lower_bounds, sticky_r2 = [], [], []
    for name in SONG_SETTINGS:
        cells = auc(name)
        for law in HEAVY:
            for level in LEVELS:
                c = cells[f"{law}|{level}"]
                best += [c[s]["auc"] for s in ZSTATS]
                lower_bounds += [c[s]["ci95"][0] for s in ZSTATS]
            sticky_r2.append(cells[f"{law}|stickiness=0.8"]["observed_mean_r2"]["auc"])
    assert max(best) == pytest.approx(0.63, abs=TOL)
    assert max(lower_bounds) <= 0.5
    assert (min(sticky_r2), max(sticky_r2)) == pytest.approx((0.17, 0.45), abs=TOL)
    spans = [("c4_whale_song_themes4_summary.json", "lognormal"), ("c4_whale_song_themes4_summary.json", "geometric"),
             ("c4_whale_song_variants2_summary.json", "lognormal")]
    for name, law in spans:
        ci = auc(name)[f"{law}|stickiness=0.8"]["observed_mean_slope"]["ci95"]
        assert ci[0] < 0.5 < ci[1], (name, law)


def test_f14_confounds():
    tags = ("themes4", "themes12", "variants2", "variants5", "ppc20", "ppc80")
    low_elements, cut_p = [], []
    for tag in tags:
        df = load(f"c4_whale_song_{tag}_skip.csv")
        for level, g in df.groupby("stickiness"):
            by_law = g.groupby("law")
            p_elem = kruskal(*[x.elements_total for _, x in by_law]).pvalue
            if p_elem < 0.05:
                means = by_law.elements_total.mean()
                low_elements.append((tag, level, round(p_elem, 3), round(means.max() - means.min())))
            cut_p.append(kruskal(*[x.cuts_per_sequence for _, x in by_law]).pvalue)
    assert low_elements == [("variants5", 0.8, 0.029, 173)]
    assert max(cut_p) == pytest.approx(0.013, abs=6e-4)


# ---- F15 C3 finch and song -------------------------------------------------------------------------------------------


def test_f15_run_health():
    finch = load("c3_finch.csv")
    err = finch.error.fillna("")
    assert len(finch) == 7200
    assert (err != "").sum() == 336
    assert set(finch.loc[err != "", "source"]) == {"true"}
    assert err.str.startswith("fit").sum() == 316
    assert err.str.contains("gof").sum() == 20
    assert err.str.contains("exceeded").sum() == 3
    assert err.str.contains("lr:").sum() == 0
    assert (finch.lr_at_bound.astype(str) == "True").sum() == 0
    disc = finch[finch.source == "discovered"].groupby("law")
    assert disc.csn_xmin.median().eq(1.0).all()
    song = load("c3_song.csv")
    serr = song.error.fillna("")
    assert len(song) == 3840 and (serr != "").sum() == 33
    assert set(map(tuple, song.loc[serr != "", ["source", "law"]].values)) == {("true", "uniform")}
    assert serr[serr != ""].str.startswith("gof").all()
    assert (song.lr_at_bound.astype(str) == "True").sum() == 0
    assert song.lr_gap.max() == pytest.approx(0.0088, abs=5e-5)
    med = song[song.source == "discovered"].groupby("stickiness").csn_xmin.apply(
        lambda s: s.groupby(song.loc[s.index, "law"]).median())
    assert (med.loc[0.0].min(), med.loc[0.0].max()) == (3.0, 5.0)
    assert (med.loc[0.8].min(), med.loc[0.8].max()) == (1.0, 2.0)


C3_STATS = ("loglog_r2", "tail_fraction", "gof_p", "lr_normalized_R", "csn_alpha")
C3_DISCOVERED = {
    ("c3_finch_summary.json", "repeats=1"): {"uniform": (0.92, 0.06, 0.91, 0.57, 0.58),
                                             "lognormal": (0.53, 0.35, 0.42, 0.42, 0.70),
                                             "geometric": (0.55, 0.33, 0.28, 0.36, 0.65)},
    ("c3_song_summary.json", "stickiness=0"): {"uniform": (1.00, 0.31, 0.75, 0.71, 0.10),
                                               "lognormal": (0.48, 0.36, 0.62, 0.58, 0.67),
                                               "geometric": (0.48, 0.38, 0.67, 0.60, 0.74)},
    ("c3_song_summary.json", "stickiness=0.8"): {"uniform": (0.91, 0.15, 0.73, 0.36, 0.50),
                                                 "lognormal": (0.29, 0.30, 0.52, 0.45, 0.68),
                                                 "geometric": (0.30, 0.38, 0.48, 0.38, 0.66)},
}


@pytest.mark.parametrize("name,level", sorted(C3_DISCOVERED))
def test_f15_discovered_auc(name, level):
    cells = load(name)["auc_zipf_vs"]
    for law, expected in C3_DISCOVERED[(name, level)].items():
        got = tuple(cells[f"discovered|{law}|{level}"][s]["auc"] for s in C3_STATS)
        assert got == pytest.approx(expected, abs=TOL), (name, level, law)


def test_f15_true_words_and_alpha_direction():
    finch = load("c3_finch_summary.json")["auc_zipf_vs"]
    song = load("c3_song_summary.json")["auc_zipf_vs"]
    f_gof = [finch[f"true|{law}|repeats=1"]["gof_p"]["auc"] for law in LAWS]
    f_lr = [finch[f"true|{law}|repeats=1"]["lr_normalized_R"]["auc"] for law in LAWS]
    s_keys = [f"true|{law}|{level}" for law in LAWS for level in LEVELS]
    s_gof = [song[k]["gof_p"]["auc"] for k in s_keys]
    s_lr = [song[k]["lr_normalized_R"]["auc"] for k in s_keys]
    assert (min(f_gof), max(f_gof)) == pytest.approx((0.95, 0.98), abs=TOL)
    assert (min(f_lr), max(f_lr)) == pytest.approx((0.96, 1.00), abs=TOL)
    assert (min(s_gof), max(s_gof)) == pytest.approx((0.97, 1.00), abs=TOL)
    assert (min(s_lr), max(s_lr)) == pytest.approx((0.85, 1.00), abs=TOL)
    assert all(finch[f"true|{law}|repeats=1"]["loglog_r2"]["auc"] >= 0.995 for law in LAWS)
    assert all(song[k]["loglog_r2"]["auc"] >= 0.995 for k in s_keys)
    alpha = [song[f"discovered|{law}|{level}"]["csn_alpha"] for law in HEAVY for level in LEVELS]
    assert all(a["ci95"][0] > 0.5 for a in alpha)
    assert (min(a["auc"] for a in alpha), max(a["auc"] for a in alpha)) == pytest.approx((0.66, 0.74), abs=TOL)
    whale = load("c3_whale_summary.json")["auc_zipf_vs"]
    w_alpha = [whale[f"discovered|{law}|repeats={r}"]["csn_alpha"]["auc"] for law in HEAVY for r in (1, 5)]
    assert (min(w_alpha), max(w_alpha)) == pytest.approx((0.24, 0.39), abs=TOL)


def test_f15_verdicts():
    fv = load("c3_finch_summary.json")["verdicts"]
    sv = load("c3_song_summary.json")["verdicts"]
    assert fv["discovered|zipf|repeats=1"]["pl_plausible"]["rate"] == pytest.approx(0.88, abs=TOL)
    others = [fv[f"discovered|{law}|repeats=1"]["pl_plausible"]["rate"] for law in LAWS]
    assert (min(others), max(others)) == pytest.approx((0.77, 0.90), abs=TOL)
    lr = [fv[f"discovered|{law}|repeats=1"][v]["rate"] for law in LAWS + ("zipf",)
          for v in ("lr_favours_pl_interior", "lr_favours_ln_interior")]
    assert max(lr) == pytest.approx(0.03, abs=TOL)
    assert fv["true|zipf|repeats=1"]["pl_plausible"]["rate"] == pytest.approx(0.58, abs=TOL)
    true_others = [fv[f"true|{law}|repeats=1"]["pl_plausible"]["rate"] for law in LAWS]
    assert (min(true_others), max(true_others)) == pytest.approx((0.33, 0.41), abs=TOL)
    ln = [sv[f"discovered|zipf|{level}"]["lr_favours_ln_interior"]["rate"] for level in LEVELS]
    assert ln == pytest.approx([0.25, 0.26], abs=TOL)
    assert all(sv[f"true|zipf|{level}"]["lr_favours_ln_interior"]["rate"] == 0.0 for level in LEVELS)
