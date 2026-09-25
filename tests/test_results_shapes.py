"""Pins the numbers quoted in FINDINGS.md F16 (the shape of the alternative law) to their results files and to the
generator (RESEARCH.md standing rule 4). Red means FINDINGS.md is stale."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kruskal, pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))

import exp_c4_inference_unit as c4  # noqa: E402
from unitsim.generate import word_probabilities  # noqa: E402

ZSTATS = ("shuffle_z", "rotate_z", "bigram_z")
TOL = 5.01e-3
DRAWS = 200


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        pytest.skip(f"results/{name} not present")
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return pd.read_csv(path)


def auc(name):
    return load(name)["policies"]["skip"]["auc_zipf_vs"]


def probs(law, n_words, seed, **shape):
    params = c4.law_params(law, n_words, **shape)
    params.pop("law")
    return np.sort(word_probabilities(n_words, law, rng=np.random.default_rng(seed), **params))[::-1]


def top_word(law, n_words, **shape):
    return float(np.mean([probs(law, n_words, s, **shape)[0] for s in range(DRAWS)]))


def loglog(p):
    return pearsonr(np.log(np.arange(1, p.size + 1)), np.log(p))[0] ** 2


# ---- the generating distributions ------------------------------------------------------------------------------------


def test_f16_word_level_r2_does_not_depend_on_shape():
    families = {"zipf": [{"zipf_exponent": e} for e in (0.8, 1.0, 1.2)],
                "lognormal": [{"lognormal_sigma": s} for s in (0.5, 1.0, 1.5, 2.0)],
                "geometric": [{"geometric_tail": t} for t in (0.2, 0.05, 0.01)]}
    expected = {"zipf": 1.000, "lognormal": 0.839, "geometric": 0.803}
    for law, shapes in families.items():
        for seed in range(3):
            values = [loglog(probs(law, 100, seed, **sh)) for sh in shapes]
            assert max(values) - min(values) < 1e-9, law
        mean = np.mean([loglog(probs(law, 100, s)) for s in range(300)])
        assert mean == pytest.approx(expected[law], abs=6e-4)


TOP_WORDS = {
    ("zipf", ()): 0.193, ("zipf", (("zipf_exponent", 0.8),)): 0.123, ("zipf", (("zipf_exponent", 1.2),)): 0.278,
    ("lognormal", (("lognormal_sigma", 0.5),)): 0.031, ("lognormal", ()): 0.076,
    ("lognormal", (("lognormal_sigma", 1.5),)): 0.151, ("lognormal", (("lognormal_sigma", 2.0),)): 0.244,
    ("geometric", (("geometric_tail", 0.2),)): 0.020, ("geometric", ()): 0.031,
    ("geometric", (("geometric_tail", 0.01),)): 0.046,
}
FINCH_TOP_WORDS = {
    ("zipf", ()): 0.341, ("lognormal", (("lognormal_sigma", 0.5),)): 0.193, ("lognormal", ()): 0.309,
    ("lognormal", (("lognormal_sigma", 1.5),)): 0.420, ("lognormal", (("lognormal_sigma", 2.0),)): 0.513,
    ("geometric", (("geometric_tail", 0.2),)): 0.197, ("geometric", ()): 0.294,
    ("geometric", (("geometric_tail", 0.01),)): 0.403,
}


@pytest.mark.parametrize("n_words,table", [(100, TOP_WORDS), (10, FINCH_TOP_WORDS)])
def test_f16_top_word_probabilities(n_words, table):
    for (law, shape), expected in table.items():
        assert top_word(law, n_words, **dict(shape)) == pytest.approx(expected, abs=6e-4), (law, shape)


# ---- whale -----------------------------------------------------------------------------------------------------------

WHALE = {  # file: {law: {repeats: (mean R^2, shuffle, rotate, bigram, slope)}}
    "c4_whale_sigma0.5_summary.json": {"lognormal": {1: (1.00, 0.80, 0.92, 0.75, 0.00), 5: (1.00, 0.24, 0.48, 0.00, 0.00)}},
    "c4_whale_sigma1.5_summary.json": {"lognormal": {1: (0.60, 0.38, 0.63, 0.87, 1.00), 5: (0.55, 0.39, 0.51, 0.50, 0.78)}},
    "c4_whale_sigma2_summary.json": {"lognormal": {1: (0.22, 0.65, 0.87, 0.98, 1.00), 5: (0.20, 0.72, 0.57, 0.91, 0.96)}},
    "c4_whale_tail0.2_summary.json": {"geometric": {1: (1.00, 0.77, 0.97, 0.69, 0.00), 5: (1.00, 0.31, 0.58, 0.01, 0.01)}},
    "c4_whale_tail0.01_summary.json": {"geometric": {1: (1.00, 0.77, 0.88, 0.96, 1.00), 5: (0.92, 0.24, 0.47, 0.04, 0.46)}},
    "c4_whale_zipfexp0.8_summary.json": {"lognormal": {1: (0.94, 0.65, 0.59, 0.80, 0.92), 5: (0.71, 0.44, 0.60, 0.31, 0.50)},
                                         "geometric": {1: (1.00, 0.68, 0.89, 0.93, 0.61), 5: (0.91, 0.35, 0.49, 0.07, 0.25)}},
    "c4_whale_zipfexp1.2_summary.json": {"lognormal": {1: (1.00, 0.68, 0.36, 0.08, 0.00), 5: (0.98, 0.25, 0.41, 0.00, 0.03)},
                                         "geometric": {1: (1.00, 0.71, 0.76, 0.23, 0.00), 5: (1.00, 0.17, 0.30, 0.00, 0.01)}},
}
WHALE_STATS = ("observed_mean_r2",) + ZSTATS + ("observed_mean_slope",)


@pytest.mark.parametrize("name", sorted(WHALE))
def test_f16_whale_auc(name):
    cells = auc(name)
    for law, levels in WHALE[name].items():
        for rep, expected in levels.items():
            got = tuple(cells[f"{law}|repeats={rep}"][s]["auc"] for s in WHALE_STATS)
            assert got == pytest.approx(expected, abs=TOL), (name, law, rep)


def test_f16_whale_unit_types_and_confounds():
    types = {"sigma0.5": ("lognormal", 407.0), "sigma1.5": ("lognormal", 277.9), "sigma2": ("lognormal", 219.9),
             "tail0.2": ("geometric", 407.0), "tail0.01": ("geometric", 305.0)}
    element_p = []
    for tag, (law, expected) in types.items():
        df = load(f"c4_whale_{tag}_skip.csv")
        means = df[df.mean_repeats == 1].groupby("law").observed_mean_unit_types.mean()
        assert means[law] == pytest.approx(expected, abs=0.06), tag
        assert means["zipf"] == pytest.approx(326.6, abs=0.06)
    for tag in ("sigma0.5", "sigma1.5", "sigma2", "tail0.2", "tail0.01", "zipfexp0.8", "zipfexp1.2"):
        df = load(f"c4_whale_{tag}_skip.csv")
        for _, g in df.groupby("mean_repeats"):
            element_p.append(kruskal(*[x.elements_total for _, x in g.groupby("law")]).pvalue)
    assert min(element_p) == pytest.approx(0.060, abs=6e-4)
    for tag, p in (("sigma1.5", 0.19), ("sigma2", 0.89), ("tail0.01", 0.66)):
        g = load(f"c4_whale_{tag}_skip.csv").query("mean_repeats == 1").groupby("law")
        assert kruskal(*[x.cuts_per_sequence for _, x in g]).pvalue == pytest.approx(p, abs=TOL), tag
    for tag, law, cuts in (("sigma0.5", "lognormal", 212), ("tail0.2", "geometric", 211)):
        m = load(f"c4_whale_{tag}_skip.csv").query("mean_repeats == 1").groupby("law").cuts_per_sequence.mean()
        assert (round(m[law]), round(m["zipf"])) == (cuts, 232), tag


def test_f16_whale_mean_r2_follows_the_commonest_word():
    """AUC above 0.5 exactly when the Zipf source's commonest word is more probable than the alternative's."""
    default_top = {"uniform": 0.01, "lognormal": top_word("lognormal", 100), "geometric": top_word("geometric", 100)}
    comparisons = []  # (summary, law, zipf top word, alternative top word)
    zipf = top_word("zipf", 100)
    for law in ("uniform", "lognormal", "geometric"):
        comparisons.append(("c4_whale_summary.json", law, zipf, default_top[law]))
    for tag, law, shape in (("sigma0.5", "lognormal", {"lognormal_sigma": 0.5}),
                            ("sigma1.5", "lognormal", {"lognormal_sigma": 1.5}),
                            ("sigma2", "lognormal", {"lognormal_sigma": 2.0}),
                            ("tail0.2", "geometric", {"geometric_tail": 0.2}),
                            ("tail0.01", "geometric", {"geometric_tail": 0.01})):
        comparisons.append((f"c4_whale_{tag}_summary.json", law, zipf, top_word(law, 100, **shape)))
    for tag, exponent in (("zipfexp0.8", 0.8), ("zipfexp1.2", 1.2)):
        z = top_word("zipf", 100, zipf_exponent=exponent)
        for law in ("uniform", "lognormal", "geometric"):
            comparisons.append((f"c4_whale_{tag}_summary.json", law, z, default_top[law]))
    checked = 0
    for name, law, z_top, alt_top in comparisons:
        for rep in (1, 5):
            a = auc(name)[f"{law}|repeats={rep}"]["observed_mean_r2"]["auc"]
            assert (a > 0.5) == (z_top > alt_top), (name, law, rep, a, z_top, alt_top)
            checked += 1
    assert checked == 28


# ---- finch -----------------------------------------------------------------------------------------------------------

FINCH = {  # file: (law, unit types, mean R^2, shuffle, rotate, bigram, unit-type AUC)
    "c4_finch_primary_sigma0.5_summary.json": ("lognormal", 40.4, 0.83, 0.50, 0.40, 0.39, 0.25),
    "c4_finch_primary_summary.json|lognormal": ("lognormal", 35.9, 0.64, 0.64, 0.57, 0.60, 0.70),
    "c4_finch_primary_sigma1.5_summary.json": ("lognormal", 31.3, 0.69, 0.71, 0.81, 0.80, 0.95),
    "c4_finch_primary_sigma2_summary.json": ("lognormal", 27.3, 0.76, 0.79, 0.93, 0.84, 0.98),
    "c4_finch_primary_tail0.2_summary.json": ("geometric", 41.0, 0.81, 0.54, 0.45, 0.37, 0.20),
    "c4_finch_primary_summary.json|geometric": ("geometric", 35.9, 0.63, 0.70, 0.66, 0.67, 0.69),
    "c4_finch_primary_tail0.01_summary.json": ("geometric", 30.2, 0.56, 0.83, 0.88, 0.87, 0.99),
}
FINCH_STATS = ("observed_mean_r2",) + ZSTATS + ("observed_mean_unit_types",)


@pytest.mark.parametrize("key", sorted(FINCH))
def test_f16_finch(key):
    name = key.split("|")[0]
    law, types, *expected = FINCH[key]
    cell = auc(name)[f"{law}|repeats=1"]
    assert tuple(cell[s]["auc"] for s in FINCH_STATS) == pytest.approx(tuple(expected), abs=TOL)
    df = load(name.replace("_summary.json", "_skip.csv"))
    means = df.groupby("law").observed_mean_unit_types.mean()
    assert means[law] == pytest.approx(types, abs=0.06)
    assert means["zipf"] == pytest.approx(37.9, abs=0.06)


def test_f16_finch_readings_and_confounds():
    separating = ("c4_finch_primary_sigma1.5_summary.json", "c4_finch_primary_sigma2_summary.json",
                  "c4_finch_primary_tail0.01_summary.json")
    inside = [v for k, v in FINCH.items() if k.split("|")[0] in separating]
    outside = [v for k, v in FINCH.items() if k.split("|")[0] not in separating]
    z_in = [x for v in inside for x in v[3:6]]
    assert (min(z_in), max(z_in)) == (0.71, 0.93)
    assert max(x for v in outside for x in v[3:6]) < 0.71
    assert (min(v[6] for v in inside), max(v[6] for v in inside)) == (0.95, 0.99)
    for tag, cuts, p in (("sigma1.5", 10.39, 0.0011), ("sigma2", 9.32, None)):
        df = load(f"c4_finch_primary_{tag}_skip.csv")
        g = df.groupby("law")
        assert g.cuts_per_sequence.mean()["lognormal"] == pytest.approx(cuts, abs=6e-3)
        assert g.cuts_per_sequence.mean()["zipf"] == pytest.approx(11.40, abs=6e-3)
        pv = kruskal(*[x.cuts_per_sequence for _, x in g]).pvalue
        assert (pv == pytest.approx(p, abs=5e-5)) if p else pv < 1e-4
    others = [kruskal(*[x.cuts_per_sequence for _, x in load(f"c4_finch_primary_{t}_skip.csv").groupby("law")]).pvalue
              for t in ("sigma0.5", "tail0.2", "tail0.01")]
    assert min(others) > 0.18
    # the whale ordering rule fails here: more concentrated alternative, AUC still above 0.5
    assert FINCH["c4_finch_primary_sigma2_summary.json"][2] == 0.76
    assert top_word("lognormal", 10, lognormal_sigma=2.0) > top_word("zipf", 10)


# ---- song ------------------------------------------------------------------------------------------------------------

SONG = {  # file: {law: {stickiness: (mean R^2, shuffle, rotate, bigram)}}
    "c4_whale_song_sigma0.5_summary.json": {"lognormal": {"0": (1.00, 0.80, 0.95, 0.87), "0.8": (0.83, 0.36, 0.42, 0.26)}},
    "c4_whale_song_sigma1.5_summary.json": {"lognormal": {"0": (0.06, 0.27, 0.14, 0.13), "0.8": (0.11, 0.36, 0.17, 0.34)}},
    "c4_whale_song_sigma2_summary.json": {"lognormal": {"0": (0.04, 0.40, 0.08, 0.06), "0.8": (0.11, 0.50, 0.23, 0.43)}},
    "c4_whale_song_tail0.2_summary.json": {"geometric": {"0": (1.00, 0.89, 0.95, 0.87), "0.8": (0.76, 0.31, 0.26, 0.16)}},
    "c4_whale_song_tail0.01_summary.json": {"geometric": {"0": (0.03, 0.08, 0.13, 0.19), "0.8": (0.06, 0.22, 0.10, 0.35)}},
}


@pytest.mark.parametrize("name", sorted(SONG))
def test_f16_song_auc(name):
    cells = auc(name)
    for law, levels in SONG[name].items():
        for level, expected in levels.items():
            got = tuple(cells[f"{law}|stickiness={level}"][s]["auc"] for s in ("observed_mean_r2",) + ZSTATS)
            assert got == pytest.approx(expected, abs=TOL), (name, law, level)


def test_f16_song_zipf_exponent_and_ranges():
    for tag, expected in (("zipfexp0.8", {"lognormal": (0.16, 0.18), "geometric": (0.21, 0.20)}),
                          ("zipfexp1.2", {"lognormal": (0.86, 0.49), "geometric": (0.92, 0.51)})):
        cells = auc(f"c4_whale_song_{tag}_summary.json")
        for law, (s0, s8) in expected.items():
            got = (cells[f"{law}|stickiness=0"]["observed_mean_r2"]["auc"],
                   cells[f"{law}|stickiness=0.8"]["observed_mean_r2"]["auc"])
            assert got == pytest.approx((s0, s8), abs=TOL), (tag, law)
    song_r2 = [lv[0] for f in SONG.values() for law in f.values() for lv in law.values()]
    song_rot = [lv[2] for f in SONG.values() for law in f.values() for lv in law.values()]
    whale_r2 = [lv[0] for f in WHALE.values() for law in f.values() for lv in law.values()]
    whale_slope = [lv[4] for f in WHALE.values() for law in f.values() for lv in law.values()]
    assert (max(song_r2), min(song_r2)) == (1.00, 0.03)
    assert SONG["c4_whale_song_sigma2_summary.json"]["lognormal"]["0"][0] == 0.04
    assert (max(song_rot), min(song_rot)) == (0.95, 0.08)
    assert (max(whale_r2), min(whale_r2)) == (1.00, 0.20)
    assert (min(whale_slope), max(whale_slope)) == (0.00, 1.00)
