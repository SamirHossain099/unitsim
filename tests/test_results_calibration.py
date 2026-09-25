"""Pins the numbers quoted in FINDINGS.md F17 (the calibration and its validation) to their results files
(RESEARCH.md standing rule 4). Red means FINDINGS.md is stale."""
import json
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
REGIMES = ("whale", "finch", "song")
LAWS = ("zipf", "uniform", "lognormal", "geometric")
FEATURES = ("observed_mean_r2", "observed_mean_slope", "observed_mean_unit_types", "cuts_per_sequence")
TOL = 5.01e-4


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        pytest.skip(f"results/{name} not present")
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return pd.read_csv(path)


@pytest.mark.parametrize("regime,rows,wide_rows", [(r, 8000, 2000) for r in REGIMES])
def test_f17_run_shape(regime, rows, wide_rows):
    df = load(f"calibration_{regime}.csv")
    assert len(df) == rows
    counts = df.groupby(["law", "set"]).size()
    assert set(counts) == {1000}
    assert set(df.law) == set(LAWS) and set(df["set"]) == {"reference", "holdout"}
    assert df[list(FEATURES)].notna().all().all()
    wide = load(f"calibration_{regime}_wide.csv")
    assert len(wide) == wide_rows and set(wide["set"]) == {"holdout"}
    meta = load(f"calibration_{regime}.json")
    assert meta["args"]["prior"] == "matched" and meta["args"]["threshold"] == 0.5
    assert load(f"calibration_{regime}_wide.json")["args"]["prior"] == "wide"


@pytest.mark.parametrize("regime,accuracy,brier,decided7,acc7,decided9,acc9", [
    ("whale", 0.617, 0.495, 0.295, 0.834, 0.110, 0.923),
    ("finch", 0.457, 0.653, 0.049, 0.796, 0.014, 0.821),
    ("song", 0.619, 0.495, 0.251, 0.917, 0.127, 0.972),
])
def test_f17_matched_prior(regime, accuracy, brier, decided7, acc7, decided9, acc9):
    s = load(f"calibration_{regime}_summary.json")["all_features"]
    assert s["at_0.7"]["n"] == 4000 and s["at_0.7"]["chance"] == 0.25
    assert s["at_0.7"]["accuracy"] == pytest.approx(accuracy, abs=TOL)
    assert s["at_0.7"]["brier"] == pytest.approx(brier, abs=TOL)
    assert s["at_0.7"]["share_decided"] == pytest.approx(decided7, abs=TOL)
    assert s["at_0.7"]["accuracy_when_decided"] == pytest.approx(acc7, abs=TOL)
    assert s["at_0.9"]["share_decided"] == pytest.approx(decided9, abs=TOL)
    assert s["at_0.9"]["accuracy_when_decided"] == pytest.approx(acc9, abs=TOL)


PER_LAW = {
    "whale": {"zipf": (0.895, 0.785), "uniform": (0.963, 0.865), "lognormal": (0.739, 0.259),
              "geometric": (0.784, 0.558)},
    "finch": {"zipf": (0.744, 0.651), "uniform": (0.822, 0.579), "lognormal": (0.662, 0.228),
              "geometric": (0.650, 0.371)},
    "song": {"zipf": (0.844, 0.674), "uniform": (0.968, 0.885), "lognormal": (0.750, 0.342),
             "geometric": (0.798, 0.574)},
}


@pytest.mark.parametrize("regime", REGIMES)
def test_f17_per_law(regime):
    cells = load(f"calibration_{regime}_summary.json")["all_features"]["at_0.7"]["per_law"]
    for law, (auc, recall) in PER_LAW[regime].items():
        assert cells[law]["auc"] == pytest.approx(auc, abs=TOL), (regime, law)
        assert cells[law]["recall"] == pytest.approx(recall, abs=TOL), (regime, law)
        assert cells[law]["n"] == 1000


ABLATION = {"whale": (0.555, 0.433, 0.367, 0.333), "finch": (0.317, 0.337, 0.341, 0.265),
            "song": (0.438, 0.346, 0.296, 0.395)}


@pytest.mark.parametrize("regime", REGIMES)
def test_f17_no_single_statistic_matches_the_vector(regime):
    s = load(f"calibration_{regime}_summary.json")
    full = s["all_features"]["at_0.7"]["accuracy"]
    for feature, expected in zip(FEATURES, ABLATION[regime]):
        got = s["ablation"][feature]["at_0.7"]["accuracy"]
        assert got == pytest.approx(expected, abs=TOL), (regime, feature)
        assert got < full, (regime, feature)


RELIABILITY = {"whale": [(0.386, 0.395), (0.496, 0.495), (0.631, 0.675), (0.784, 0.766), (0.943, 0.899)],
               "song": [(0.367, 0.384), (0.482, 0.518), (0.621, 0.658), (0.777, 0.846), (0.957, 0.964)]}


@pytest.mark.parametrize("regime", ("whale", "song"))
def test_f17_reliability_matched(regime):
    bins = load(f"calibration_{regime}_summary.json")["all_features"]["at_0.7"]["reliability"]
    got = [(round(b["mean_confidence"], 3), round(b["share_correct"], 3)) for b in bins]
    assert got == pytest.approx(RELIABILITY[regime], abs=TOL)


def test_f17_finch_top_bin_is_the_one_overconfident_cell():
    bins = load("calibration_finch_summary.json")["all_features"]["at_0.7"]["reliability"]
    top = bins[-1]
    assert (round(top["mean_confidence"], 2), round(top["share_correct"], 2), top["n"]) == (0.95, 0.78, 72)
    for b in bins[:-1]:
        assert abs(b["mean_confidence"] - b["share_correct"]) < 0.12


@pytest.mark.parametrize("regime,accuracy,brier,decided9,acc9", [
    ("whale", 0.424, 0.874, 0.386, 0.468),
    ("finch", 0.388, 0.880, 0.311, 0.468),
    ("song", 0.441, 0.841, 0.412, 0.554),
])
def test_f17_wide_prior_is_confidently_wrong(regime, accuracy, brier, decided9, acc9):
    s = load(f"calibration_{regime}_wide_summary.json")["all_features"]
    matched = load(f"calibration_{regime}_summary.json")["all_features"]
    assert s["at_0.7"]["n"] == 2000
    assert s["at_0.7"]["accuracy"] == pytest.approx(accuracy, abs=TOL)
    assert s["at_0.7"]["brier"] == pytest.approx(brier, abs=TOL)
    assert s["at_0.9"]["share_decided"] == pytest.approx(decided9, abs=TOL)
    assert s["at_0.9"]["accuracy_when_decided"] == pytest.approx(acc9, abs=TOL)
    # confidently wrong: it decides more often than under the matched prior and is right far less often
    assert s["at_0.9"]["share_decided"] > matched["at_0.9"]["share_decided"]
    assert s["at_0.9"]["accuracy_when_decided"] < matched["at_0.9"]["accuracy_when_decided"] - 0.3


def test_f17_wide_whale_top_bin():
    top = load("calibration_whale_wide_summary.json")["all_features"]["at_0.7"]["reliability"][-1]
    assert round(top["mean_confidence"], 2) == 0.98 and round(top["share_correct"], 2) == 0.47
