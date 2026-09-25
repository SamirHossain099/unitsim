"""Pins the numbers quoted in FINDINGS.md F10 (C3) and F11 (finch sensitivity) to their results files
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


# ---- F10 C3 ----------------------------------------------------------------------------------------------------------


def test_f10_run_health():
    df = load("c3_whale.csv")
    assert len(df) == 3840
    assert set(df.groupby(["source", "law", "mean_repeats"]).size()) == {240}
    assert (df.error.fillna("") != "").sum() == 0
    assert df.gof_failed.max() == 0
    assert df.lr_gap.max() <= 0.0105
    assert (df.lr_at_bound.astype(str) == "True").sum() == 5
    disc = df[df.source == "discovered"].groupby(["law", "mean_repeats"])
    assert disc.csn_xmin.median().drop(("uniform", 1.0)).eq(1.0).all()
    assert disc.tail_fraction.mean().loc[("uniform", 1.0)] == pytest.approx(0.306, abs=5e-4)


@pytest.mark.parametrize("key,stat,expected", [
    ("discovered|lognormal|repeats=1", "lr_normalized_R", 0.00),
    ("discovered|geometric|repeats=1", "lr_normalized_R", 0.00),
    ("discovered|geometric|repeats=5", "lr_normalized_R", 0.00),
    ("discovered|lognormal|repeats=1", "gof_p", 0.97),
    ("discovered|uniform|repeats=1", "gof_p", 0.31),
    ("discovered|lognormal|repeats=5", "loglog_r2", 0.90),
    ("discovered|uniform|repeats=5", "tail_fraction", 0.47),
    ("true|lognormal|repeats=1", "lr_normalized_R", 0.97),
    ("true|lognormal|repeats=5", "gof_p", 0.79),
    ("true|lognormal|repeats=5", "tail_fraction", 0.83),
])
def test_f10_auc(key, stat, expected):
    s = load("c3_whale_summary.json")
    assert s["auc_zipf_vs"][key][stat]["auc"] == pytest.approx(expected, abs=5e-3)


@pytest.mark.parametrize("key,verdict,rate,n", [
    ("true|zipf|repeats=1", "pl_plausible", 0.99, 240),
    ("true|zipf|repeats=1", "pl_plausible_broad", 0.70, 240),
    ("true|geometric|repeats=1", "lr_favours_ln_interior", 0.30, 240),
    ("discovered|geometric|repeats=1", "lr_favours_pl_interior", 0.51, 236),
    ("discovered|lognormal|repeats=1", "lr_favours_pl_interior", 0.46, 239),
    ("discovered|zipf|repeats=1", "lr_favours_pl_interior", 0.03, 240),
    ("discovered|uniform|repeats=1", "pl_plausible", 0.74, 240),
])
def test_f10_verdicts(key, verdict, rate, n):
    v = load("c3_whale_summary.json")["verdicts"][key][verdict]
    assert v["rate"] == pytest.approx(rate, abs=5e-3) and v["n"] == n


def test_f10_true_counts_never_favour_a_power_law():
    verdicts = load("c3_whale_summary.json")["verdicts"]
    for key, v in verdicts.items():
        if key.startswith("true|"):
            assert v["lr_favours_pl_interior"]["rate"] == 0.0


# ---- F11 finch sensitivity -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tag,cuts,gap,elem_p,cut_h", [
    ("sensA", {"zipf": 11.12, "uniform": 10.27, "lognormal": 10.62, "geometric": 11.67}, 0.95, 0.93, 16.61),
    ("sensB", {"zipf": 11.45, "uniform": 11.18, "lognormal": 11.70, "geometric": 11.82}, 1.10, 0.38, 5.18),
])
def test_f11_calibration_and_confounds(tag, cuts, gap, elem_p, cut_h):
    df = load(f"c4_finch_{tag}_skip.csv")
    assert len(df) == 60
    g = df.groupby("law")
    assert g.cuts_per_sequence.mean().to_dict() == pytest.approx(cuts, abs=5e-3)
    assert (g.cuts_per_sequence.mean() - 10.72).abs().max() == pytest.approx(gap, abs=5e-3)
    assert kruskal(*[x.elements_total for _, x in g]).pvalue == pytest.approx(elem_p, abs=5e-3)
    assert kruskal(*[x.cuts_per_sequence for _, x in g]).statistic == pytest.approx(cut_h, abs=5e-3)


@pytest.mark.parametrize("tag,key,stat,expected", [
    ("sensA", "geometric|repeats=1", "observed_mean_r2", 0.45),
    ("sensA", "geometric|repeats=1", "observed_mean_slope", 0.73),
    ("sensA", "uniform|repeats=1", "observed_mean_unit_types", 0.00),
    ("sensB", "lognormal|repeats=2", "observed_mean_r2", 0.44),
    ("sensB", "uniform|repeats=2", "bigram_z", 0.14),
    ("sensB", "geometric|repeats=2", "observed_mean_slope", 0.65),
])
def test_f11_auc(tag, key, stat, expected):
    s = load(f"c4_finch_{tag}_summary.json")
    assert s["policies"]["skip"]["auc_zipf_vs"][key][stat]["auc"] == pytest.approx(expected, abs=5e-3)


def test_f11_only_two_heavy_tail_lower_bounds_clear_half():
    above = []
    for tag, rep in (("sensA", 1), ("sensB", 2)):
        cells = load(f"c4_finch_{tag}_summary.json")["policies"]["skip"]["auc_zipf_vs"]
        for law in ("lognormal", "geometric"):
            for stat, v in cells[f"{law}|repeats={rep}"].items():
                if v["ci95"][0] > 0.5:
                    above.append((tag, law, stat, round(v["ci95"][0], 2)))
    assert sorted(above) == [("sensA", "geometric", "observed_mean_slope", 0.53),
                             ("sensA", "geometric", "observed_mean_unit_types", 0.54)]


@pytest.mark.parametrize("tag,law,shares", [
    ("sensA", "zipf", (0.13, 0.33, 0.07)), ("sensB", "zipf", (0.80, 0.00, 0.07)),
    ("sensA", "geometric", (0.33, 0.20, 0.07)), ("sensB", "uniform", (0.93, 0.00, 0.20)),
])
def test_f11_share_clearing_the_published_z(tag, law, shares):
    s = load(f"c4_finch_{tag}_summary.json")
    rep = 1 if tag == "sensA" else 2
    got = s["policies"]["skip"]["share_clearing_published_z"][f"{law}|repeats={rep}"]
    assert (got["shuffle"], got["rotate"], got["bigram"]) == pytest.approx(shares, abs=5e-3)


def test_f11_policy_sensitivity():
    for tag, mean, largest in (("sensA", 0.98, 3.50), ("sensB", 1.03, 3.67)):
        sens = load(f"c4_finch_{tag}_summary.json")["policy_sensitivity"]["shuffle"]
        assert sens["mean_abs_dz"] == pytest.approx(mean, abs=5e-3)
        assert sens["max_abs_dz"] == pytest.approx(largest, abs=5e-3)
