"""Pins the numbers quoted in FINDINGS.md F6 to the results files they came from (RESEARCH.md standing rule 4).

If a results file is regenerated, overwritten or rerun with other settings and a quoted number moves, this goes
red, and FINDINGS.md must be updated before anything else is written from it. Literals here are the quoted values;
the computed values come from results/.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kruskal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "c4_whale_skip.csv")
SUMMARY = os.path.join(ROOT, "results", "c4_whale_summary.json")

pytestmark = pytest.mark.skipif(not os.path.exists(CSV), reason="results/c4_whale_skip.csv not present")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def skip_summary():
    if not os.path.exists(SUMMARY):
        pytest.skip("results/c4_whale_summary.json not present")
    with open(SUMMARY, encoding="utf-8") as fh:
        return json.load(fh)["policies"]["skip"]


# (law, repeats): mean R^2, boundary F1, shuffle z, rotate z, bigram z -- FINDINGS.md F6, first table
F6_MEANS = {
    ("zipf", 1.0): (0.943, 0.711, 27.34, 21.56, 13.35),
    ("uniform", 1.0): (0.833, 0.713, 24.64, 14.78, 11.92),
    ("lognormal", 1.0): (0.917, 0.741, 26.96, 20.93, 13.26),
    ("geometric", 1.0): (0.892, 0.743, 26.53, 18.32, 12.36),
    ("zipf", 5.0): (0.934, 0.371, 26.76, 8.64, 8.50),
    ("uniform", 5.0): (0.903, 0.359, 28.02, 8.08, 13.65),
    ("lognormal", 5.0): (0.923, 0.375, 28.10, 8.39, 10.72),
    ("geometric", 5.0): (0.918, 0.375, 28.38, 8.99, 12.19),
}


def test_design_is_complete(df):
    assert len(df) == 240
    assert set(df.groupby(["law", "mean_repeats"]).size()) == {30}
    assert set(df.undefined_policy) == {"skip"}


@pytest.mark.parametrize("cell", sorted(F6_MEANS))
def test_cell_means_match_findings(df, cell):
    g = df[(df.law == cell[0]) & (df.mean_repeats == cell[1])]
    r2, f1, shuffle, rotate, bigram = F6_MEANS[cell]
    assert g.observed_mean_r2.mean() == pytest.approx(r2, abs=5e-4)
    assert g.boundary_f1.mean() == pytest.approx(f1, abs=5e-4)
    assert g.shuffle_z.mean() == pytest.approx(shuffle, abs=5e-3)
    assert g.rotate_z.mean() == pytest.approx(rotate, abs=5e-3)
    assert g.bigram_z.mean() == pytest.approx(bigram, abs=5e-3)


def test_undefined_groups_were_rare_and_nothing_was_dropped(df):
    assert df.shuffle_undefined_rate.max() == pytest.approx(0.00625)
    assert df.rotate_undefined_rate.max() == pytest.approx(0.0125)
    assert df.bigram_undefined_rate.max() == 0.0
    assert df.observed_undefined_groups.max() == 0
    assert df[[c for c in df.columns if c.endswith("_dropped_replicates")]].to_numpy().sum() == 0


def test_confound_check_corpus_size_does_not_vary_with_law(df):
    means = df.groupby(["law", "mean_repeats"]).elements_total.mean()
    assert means.min() == pytest.approx(45636, abs=0.5) and means.max() == pytest.approx(45918, abs=0.5)
    for repeats, (h, p) in {1.0: (1.44, 0.70), 5.0: (6.42, 0.093)}.items():
        sub = df[df.mean_repeats == repeats]
        res = kruskal(*[sub[sub.law == law].elements_total for law in sorted(sub.law.unique())])
        assert res.statistic == pytest.approx(h, abs=5e-3)
        assert res.pvalue == pytest.approx(p, abs=5e-3)


@pytest.mark.parametrize("key,stat,expected", [
    ("uniform|repeats=5", "bigram_z", 0.00),
    ("geometric|repeats=5", "bigram_z", 0.00),
    ("lognormal|repeats=5", "bigram_z", 0.09),
    ("uniform|repeats=5", "shuffle_z", 0.30),
    ("lognormal|repeats=5", "shuffle_z", 0.32),
    ("geometric|repeats=5", "shuffle_z", 0.25),
    ("lognormal|repeats=5", "observed_mean_r2", 0.94),
    ("geometric|repeats=5", "observed_mean_r2", 0.99),
    ("lognormal|repeats=1", "shuffle_z", 0.58),
    ("lognormal|repeats=1", "bigram_z", 0.51),
])
def test_auc_matches_findings(skip_summary, key, stat, expected):
    assert skip_summary["auc_zipf_vs"][key][stat]["auc"] == pytest.approx(expected, abs=5e-3)


def test_every_mean_r2_auc_is_at_least_0_94(skip_summary):
    aucs = [v["observed_mean_r2"]["auc"] for v in skip_summary["auc_zipf_vs"].values()]
    assert len(aucs) == 6 and min(aucs) >= 0.935


@pytest.mark.parametrize("key,shuffle,rotate", [
    ("uniform|repeats=1", 0.83, 0.67), ("zipf|repeats=5", 0.97, 0.00),
    ("lognormal|repeats=1", 1.00, 1.00), ("geometric|repeats=5", 1.00, 0.00),
])
def test_share_clearing_the_published_z_matches_findings(skip_summary, key, shuffle, rotate):
    share = skip_summary["share_clearing_published_z"][key]
    assert share["shuffle"] == pytest.approx(shuffle, abs=5e-3)
    assert share["rotate"] == pytest.approx(rotate, abs=5e-3)


def test_zero_policy_addendum_matches_findings():
    zero_csv = os.path.join(ROOT, "results", "c4_whale_zero.csv")
    if not (os.path.exists(zero_csv) and os.path.exists(SUMMARY)):
        pytest.skip("zero-policy results not present")
    with open(SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    zero = summary["policies"]["zero"]
    assert len(pd.read_csv(zero_csv)) == 240
    assert zero["share_clearing_published_z"]["uniform|repeats=1"]["shuffle"] == pytest.approx(0.70, abs=5e-3)
    assert zero["auc_zipf_vs"]["uniform|repeats=1"]["shuffle_z"]["auc"] == pytest.approx(0.93, abs=5e-3)
    assert zero["auc_zipf_vs"]["geometric|repeats=1"]["shuffle_z"]["auc"] == pytest.approx(0.69, abs=5e-3)
    sens = summary["policy_sensitivity"]
    assert sens["shuffle"]["mean_abs_dz"] == pytest.approx(0.094, abs=5e-4)
    assert sens["shuffle"]["max_abs_dz"] == pytest.approx(1.82, abs=5e-3)
    assert sens["rotate"]["mean_abs_dz"] == pytest.approx(0.032, abs=5e-4)
    assert sens["rotate"]["max_abs_dz"] == pytest.approx(6.06, abs=5e-3)
    assert sens["bigram"]["max_abs_dz"] == 0.0
    skip = summary["policies"]["skip"]["auc_zipf_vs"]
    for key in skip:
        if key.endswith("repeats=5"):
            for stat in ("observed_mean_r2", "shuffle_z", "rotate_z", "bigram_z"):
                assert round(skip[key][stat]["auc"], 2) == round(zero["auc_zipf_vs"][key][stat]["auc"], 2)


def test_quoted_rounding_is_honest(df):
    """F6 quotes 26.53 and 28.38 for values that sit on a rounding half; check they are not overstated."""
    g = df.groupby(["law", "mean_repeats"])
    assert np.round(g.shuffle_z.mean()[("geometric", 1.0)], 2) in (26.52, 26.53)
    assert np.round(g.shuffle_z.mean()[("geometric", 5.0)], 2) in (28.37, 28.38)
