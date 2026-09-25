"""Pins the numbers quoted in FINDINGS.md F7 (finch), F8 (song structure) and F9 (label error) to their results files
(RESEARCH.md standing rule 4). Red means FINDINGS.md is stale."""
import json
import os

import pandas as pd
import pytest
from scipy.stats import kruskal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        pytest.skip(f"results/{name} not present")
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return pd.read_csv(path)


def auc(summary, policy, key, stat):
    return summary["policies"][policy]["auc_zipf_vs"][key][stat]["auc"]


# ---- F7 finch -------------------------------------------------------------------------------------------------------


def test_f7_calibration_and_confounds():
    df = load("c4_finch_primary_skip.csv")
    assert len(df) == 120
    g = df.groupby("law")
    cuts = g.cuts_per_sequence.mean()
    assert cuts.to_dict() == pytest.approx({"zipf": 11.40, "uniform": 10.83, "lognormal": 11.15,
                                            "geometric": 11.54}, abs=5e-3)
    assert (cuts - 10.72).abs().max() <= 0.82 + 5e-3
    assert g.shuffle_undefined_rate.mean()["uniform"] == pytest.approx(0.070, abs=5e-4)
    h = kruskal(*[x.elements_total for _, x in g])
    assert (h.statistic, h.pvalue) == pytest.approx((0.49, 0.92), abs=5e-3)
    h = kruskal(*[x.cuts_per_sequence for _, x in g])
    assert h.statistic == pytest.approx(13.15, abs=5e-3) and h.pvalue == pytest.approx(0.0043, abs=5e-5)


@pytest.mark.parametrize("key,stat,expected", [
    ("uniform|repeats=1", "observed_mean_r2", 0.99), ("lognormal|repeats=1", "observed_mean_r2", 0.64),
    ("geometric|repeats=1", "observed_mean_r2", 0.63), ("uniform|repeats=1", "bigram_z", 0.38),
    ("geometric|repeats=1", "shuffle_z", 0.70), ("lognormal|repeats=1", "observed_mean_unit_types", 0.70),
    ("uniform|repeats=1", "observed_mean_slope", 0.19), ("geometric|repeats=1", "observed_mean_slope", 0.68),
])
def test_f7_auc(key, stat, expected):
    assert auc(load("c4_finch_primary_summary.json"), "skip", key, stat) == pytest.approx(expected, abs=5e-3)


def test_f7_policy_sensitivity_and_shares():
    s = load("c4_finch_primary_summary.json")
    sens = s["policy_sensitivity"]
    assert sens["shuffle"]["mean_abs_dz"] == pytest.approx(1.22, abs=5e-3)
    assert sens["shuffle"]["max_abs_dz"] == pytest.approx(3.33, abs=5e-3)
    assert sens["rotate"]["max_abs_dz"] == pytest.approx(4.26, abs=5e-3)
    share = s["policies"]["skip"]["share_clearing_published_z"]
    assert share["geometric|repeats=1"]["bigram"] == pytest.approx(0.50, abs=5e-3)
    assert share["uniform|repeats=1"]["shuffle"] == pytest.approx(0.97, abs=5e-3)
    assert s["policies"]["zero"]["share_clearing_published_z"]["lognormal|repeats=1"]["shuffle"] == \
        pytest.approx(0.63, abs=5e-3)


# ---- F8 song ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key,stat,expected", [
    ("lognormal|stickiness=0.8", "observed_mean_r2", 0.28), ("geometric|stickiness=0.8", "observed_mean_r2", 0.31),
    ("lognormal|stickiness=0", "observed_mean_r2", 0.50), ("uniform|stickiness=0.8", "bigram_z", 0.09),
    ("geometric|stickiness=0.8", "rotate_z", 0.19), ("geometric|stickiness=0", "observed_mean_slope", 0.92),
    ("lognormal|stickiness=0.8", "observed_mean_slope", 0.76), ("uniform|stickiness=0.8", "observed_mean_unit_types",
                                                                 0.30),
])
def test_f8_auc(key, stat, expected):
    assert auc(load("c4_whale_song_summary.json"), "skip", key, stat) == pytest.approx(expected, abs=5e-3)


def test_f8_slope_cis_all_above_half_for_heavy_tails():
    s = load("c4_whale_song_summary.json")["policies"]["skip"]["auc_zipf_vs"]
    for key in ("lognormal|stickiness=0", "geometric|stickiness=0", "lognormal|stickiness=0.8",
                "geometric|stickiness=0.8"):
        assert s[key]["observed_mean_slope"]["ci95"][0] > 0.5


def test_f8_means_confounds_and_rotate_bar():
    df = load("c4_whale_song_skip.csv")
    assert len(df) == 240
    means = df.groupby(["law", "stickiness"])[["observed_mean_r2", "boundary_f1", "rotate_z"]].mean()
    assert means.loc[("uniform", 0.0), "observed_mean_r2"] == pytest.approx(0.796, abs=5e-4)
    assert means.loc[("lognormal", 0.8), "observed_mean_r2"] == pytest.approx(0.933, abs=5e-4)
    assert means.loc[("zipf", 0.8), "boundary_f1"] == pytest.approx(0.431, abs=5e-4)
    assert means.loc[("uniform", 0.0), "rotate_z"] == pytest.approx(-12.1, abs=0.05)
    assert (df.rotate_z > 14.43).sum() == 0
    for st, (h, p) in {0.0: (1.24, 0.74), 0.8: (9.70, 0.021)}.items():
        sub = df[df.stickiness == st]
        res = kruskal(*[x.elements_total for _, x in sub.groupby("law")])
        assert (res.statistic, res.pvalue) == pytest.approx((h, p), abs=5e-3)


# ---- F9 label error --------------------------------------------------------------------------------------------------


def test_f9_ratio_ranges_and_directions():
    cells = load("c2_whale_summary.json")["cells"]
    assert len(cells) == 8

    def ratios(cond, stat):
        return [c["conditions"][cond][stat]["ratio"] for c in cells.values()]

    assert min(ratios("confuse_19.19", "loglog_r2")) == pytest.approx(21.08, abs=5e-3)
    assert max(ratios("confuse_19.19", "loglog_r2")) == pytest.approx(49.54, abs=5e-3)
    assert min(ratios("confuse_6.7", "loglog_r2")) == pytest.approx(5.45, abs=5e-3)
    assert max(ratios("confuse_19.19", "loglog_slope")) == pytest.approx(36.76, abs=5e-3)
    for c in cells.values():
        for cond in ("confuse_6.7", "confuse_19.19"):
            r2 = c["conditions"][cond]["loglog_r2"]
            slope = c["conditions"][cond]["loglog_slope"]
            assert r2["mean_shift"] < 0 and r2["sign_consistency"] == 1.0
            assert slope["mean_shift"] > 0 and slope["sign_consistency"] == 1.0
        for cond in c["conditions"]:
            for stat in c["conditions"][cond].values():
                assert stat["rows_with_undefined_years"] == 0
    shifts_1919 = [c["conditions"]["confuse_19.19"]["loglog_r2"]["mean_shift"] for c in cells.values()]
    assert min(shifts_1919) == pytest.approx(-0.188, abs=5e-4) and max(shifts_1919) == pytest.approx(-0.134, abs=5e-4)
    split = [abs(c["conditions"][k]["loglog_r2"]["mean_shift"]) for c in cells.values()
             for k in ("oversplit_25", "undersplit_25")]
    assert max(split) <= 0.029 + 5e-4
