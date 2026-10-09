"""Pins the numbers quoted in FINDINGS.md F18 (the calibration applied to the deposited humpback corpus) to their
results files (RESEARCH.md standing rule 4). Red means FINDINGS.md or the manuscript is stale."""
import json
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LAWS = ("zipf", "uniform", "lognormal", "geometric")
TOL = 5.01e-4


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        pytest.skip(f"results/{name} not present")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_f18_corpus_and_reproduction():
    f = load("real_whale_features.json")
    years = f["years"]
    assert len(years) == 8 and sum(y["recordings"] for y in years) == 55
    assert f["elements_total"] == 45749
    assert min(y["element_types"] for y in years) == 33 and max(y["element_types"] for y in years) == 92
    rec = {y["year"]: y["recordings"] for y in years}
    assert rec[2015] == 13 and rec[2016] == 6
    t = f["by_threshold"]["0.5"]
    # the paper: mean R^2 0.93, "an average of 1562 cuts on the data from each year"
    assert t["observed_mean_r2"] == pytest.approx(0.9303, abs=TOL)
    assert round(t["observed_mean_r2"], 2) == 0.93
    assert t["cuts_per_year"] == pytest.approx(1561.6, abs=0.05)
    assert round(t["cuts_per_year"]) == 1562
    assert t["observed_mean_slope"] == pytest.approx(-0.877, abs=TOL)
    assert t["observed_mean_unit_types"] == pytest.approx(423.25, abs=TOL)
    assert t["cuts_per_sequence"] == pytest.approx(227.1, abs=0.05)


@pytest.mark.parametrize("prior", ["song", "song_labels", "broad_labels"])
def test_f18_reference_runs(prior):
    path = os.path.join(RES, f"real_whale_reference_{prior}.csv")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    df = pd.read_csv(path)
    assert len(df) == 8000 and set(df.groupby(["law", "set"]).size()) == {1000}
    meta = load(f"real_whale_reference_{prior}.json")
    assert meta["args"]["threshold"] == 0.5
    if prior.endswith("_labels"):
        assert meta["prior"]["confusion"] == [0.0, 0.2]
        assert df["confusion"].between(0.0, 0.2).all()
    else:
        assert (df["confusion"] == 0).all()


@pytest.mark.parametrize("name,inside,largest", [
    ("paper_song", False, 0.000), ("matched_song", False, 0.001),
    ("matched_song_labels", True, 0.041), ("matched_broad_labels", True, 0.218),
])
def test_f18_containment(name, inside, largest):
    s = load("real_whale_calibration.json")["sets"][name]
    assert s["real_inside"] is inside
    assert max(s["real_typicality"].values()) == pytest.approx(largest, abs=TOL)
    assert 0.002 - TOL <= s["holdout_reported_outside"] <= 0.008 + TOL


def test_f18_outside_sets_would_have_said_uniform():
    sets = load("real_whale_calibration.json")["sets"]
    for name in ("paper_song", "matched_song"):
        assert sets[name]["real_posterior"]["uniform"] == pytest.approx(1.0, abs=TOL)
        assert sets[name]["real_verdict_0.7"] is None


@pytest.mark.parametrize("name,post,v07,decided,right", [
    ("matched_song_labels", {"zipf": 0.282, "lognormal": 0.514, "geometric": 0.180, "uniform": 0.024}, None,
     0.095, 0.839),
    ("matched_broad_labels", {"zipf": 0.170, "lognormal": 0.7004, "geometric": 0.121, "uniform": 0.008},
     "lognormal", 0.047, 0.794),
])
def test_f18_posteriors_and_validation(name, post, v07, decided, right):
    s = load("real_whale_calibration.json")["sets"][name]
    for law, p in post.items():
        assert s["real_posterior"][law] == pytest.approx(p, abs=TOL)
    assert s["real_verdict_0.7"] == v07 and s["real_verdict_0.9"] is None
    assert s["validation"]["0.7"]["share_decided"] == pytest.approx(decided, abs=TOL)
    assert s["validation"]["0.7"]["accuracy_when_decided"] == pytest.approx(right, abs=TOL)
    # Zipf is never the leading law where the reference set contains the corpus
    assert max(s["real_posterior"], key=s["real_posterior"].get) != "zipf"


def test_f18_matched_song_validation_and_ranges():
    sets = load("real_whale_calibration.json")["sets"]
    ms = sets["matched_song"]["validation"]["0.7"]
    assert ms["share_decided"] == pytest.approx(0.257, abs=TOL)
    assert ms["accuracy_when_decided"] == pytest.approx(0.906, abs=TOL)
    r = sets["paper_song"]["feature_ranges"]
    assert r["observed_mean_unit_types"]["max"] == pytest.approx(328.88, abs=0.005)
    assert round(r["observed_mean_unit_types"]["max"]) == 329          # as the manuscript rounds it, beside 423
    obs = load("real_whale_calibration.json")["observation"]
    assert round(obs["observed_mean_unit_types"]) == 423
    assert r["observed_mean_slope"]["max"] == pytest.approx(-0.88, abs=0.005)
    assert r["observed_mean_slope"]["share_at_or_above_observed"] == 0.0
    m = sets["matched_song"]["feature_ranges"]
    assert m["observed_mean_unit_types"]["max"] == pytest.approx(209.75, abs=0.005)
    assert m["observed_mean_slope"]["share_at_or_above_observed"] == pytest.approx(0.0005, abs=1e-6)
    for name, share in (("matched_song_labels", 0.0149), ("matched_broad_labels", 0.0109)):
        assert sets[name]["feature_ranges"]["observed_mean_r2"]["share_at_or_above_observed"] == \
            pytest.approx(share, abs=1e-4)


def test_f18_the_only_verdict_clears_the_threshold_by_its_quoted_margin():
    """The manuscript quotes 0.7004, not 0.700: the verdict flips on which side of 0.7 the number falls."""
    p = load("real_whale_calibration.json")["sets"]["matched_broad_labels"]["real_posterior"]["lognormal"]
    assert 0.7 < p < 0.7005
    assert round(p, 4) == 0.7004
