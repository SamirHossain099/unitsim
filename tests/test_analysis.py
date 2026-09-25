"""The analysis scripts produce every number that reaches FINDINGS.md, so they are tested on inputs whose
answers can be worked out by hand. Each script runs as a subprocess, exactly as run_all.py runs it.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from exp_c2_label_error import CONDITIONS, STATS
from exp_c3_estimators import SOURCES
from exp_c4_inference_unit import BASELINES, LAWS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "src", script), *args],
                          capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "UNITSIM_THREADS": "1"})


def ok(result):
    assert result.returncode == 0, f"script failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    return result


# ---- analyze_c4 ------------------------------------------------------------------------------------


def c4_frame(policy, factor="mean_repeats", n=6):
    rows = []
    for law in LAWS:
        for d in range(n):
            row = {"law": law, factor: 1.0, "dataset": d, "undefined_policy": policy,
                   "observed_mean_r2": 0.95 if law == "zipf" else 0.80 + 0.01 * d,
                   "observed_undefined_groups": 0, "boundary_f1": 0.7, "cuts_per_sequence": 10.0}
            for b in BASELINES:
                row.update({f"{b}_z": (30.0 if law == "zipf" else 10.0) + d, f"{b}_mean": 0.5,
                            f"{b}_sd": 0.01, f"{b}_p": 0.001, f"{b}_undefined_rate": 0.0,
                            f"{b}_dropped_replicates": 0})
            rows.append(row)
    return pd.DataFrame(rows)


def test_c4_auc_and_share_clearing_the_published_z(tmp_path):
    c4_frame("skip").to_csv(tmp_path / "c4_whale_skip.csv", index=False)
    ok(run("analyze_c4.py", "--regime", "whale", "--dir", str(tmp_path), "--n-boot", "50"))
    out = json.loads((tmp_path / "c4_whale_summary.json").read_text())
    pol = out["policies"]["skip"]
    for law in ("uniform", "lognormal", "geometric"):
        assert pol["auc_zipf_vs"][f"{law}|repeats=1"]["observed_mean_r2"]["auc"] == 1.0
        assert pol["auc_zipf_vs"][f"{law}|repeats=1"]["shuffle_z"]["auc"] == 1.0
    assert pol["share_clearing_published_z"]["zipf|repeats=1"]["shuffle"] == 1.0      # 30..35 > 23.09
    assert pol["share_clearing_published_z"]["uniform|repeats=1"]["shuffle"] == 0.0   # 10..15 < 23.09


def test_c4_policy_sensitivity_is_the_exact_paired_difference(tmp_path):
    skip = c4_frame("skip")
    zero = c4_frame("zero")
    zero["shuffle_z"] += 2.0
    skip.to_csv(tmp_path / "c4_whale_skip.csv", index=False)
    zero.to_csv(tmp_path / "c4_whale_zero.csv", index=False)
    ok(run("analyze_c4.py", "--regime", "whale", "--dir", str(tmp_path), "--n-boot", "20"))
    sens = json.loads((tmp_path / "c4_whale_summary.json").read_text())["policy_sensitivity"]
    assert sens["shuffle"]["mean_abs_dz"] == pytest.approx(2.0)
    assert sens["shuffle"]["max_abs_dz"] == pytest.approx(2.0)
    assert sens["rotate"]["mean_abs_dz"] == 0.0


def test_c4_refuses_runs_that_are_not_paired(tmp_path):
    skip = c4_frame("skip")
    zero = c4_frame("zero")
    zero.loc[3, "observed_mean_r2"] += 0.01          # no undefined groups, so it must match skip
    skip.to_csv(tmp_path / "c4_whale_skip.csv", index=False)
    zero.to_csv(tmp_path / "c4_whale_zero.csv", index=False)
    r = run("analyze_c4.py", "--regime", "whale", "--dir", str(tmp_path), "--n-boot", "20")
    assert r.returncode != 0 and "not paired" in r.stderr


def test_c4_refuses_a_file_whose_policy_disagrees_with_its_name(tmp_path):
    c4_frame("zero").to_csv(tmp_path / "c4_whale_skip.csv", index=False)
    r = run("analyze_c4.py", "--regime", "whale", "--dir", str(tmp_path), "--n-boot", "20")
    assert r.returncode != 0 and "expected skip" in r.stderr


def test_c4_splits_on_another_factor_and_tag(tmp_path):
    c4_frame("skip", factor="stickiness").to_csv(tmp_path / "c4_whale_song_skip.csv", index=False)
    ok(run("analyze_c4.py", "--regime", "whale", "--tag", "song", "--factor", "stickiness",
           "--dir", str(tmp_path), "--n-boot", "20"))
    out = json.loads((tmp_path / "c4_whale_song_summary.json").read_text())
    assert out["factor"] == "stickiness"
    assert "uniform|stickiness=1" in out["policies"]["skip"]["auc_zipf_vs"]


def test_c4_uses_slope_and_type_count_when_a_run_recorded_them(tmp_path):
    frame = c4_frame("skip")
    frame["observed_mean_slope"] = [-1.0 if law == "zipf" else -0.5 for law in frame.law]
    frame["observed_mean_unit_types"] = [100.0 if law == "zipf" else 300.0 for law in frame.law]
    frame.to_csv(tmp_path / "c4_whale_skip.csv", index=False)
    ok(run("analyze_c4.py", "--regime", "whale", "--dir", str(tmp_path), "--n-boot", "20"))
    pol = json.loads((tmp_path / "c4_whale_summary.json").read_text())["policies"]["skip"]
    cell = pol["auc_zipf_vs"]["uniform|repeats=1"]
    assert cell["observed_mean_slope"]["auc"] == 0.0         # zipf's -1.0 is always below -0.5
    assert cell["observed_mean_unit_types"]["auc"] == 0.0    # 100 always below 300
    assert "observed_mean_slope" in pol["describe"]["zipf|repeats=1"]


def test_c4_leaves_slope_and_type_count_out_when_a_run_lacks_them(tmp_path):
    c4_frame("skip").to_csv(tmp_path / "c4_whale_skip.csv", index=False)
    ok(run("analyze_c4.py", "--regime", "whale", "--dir", str(tmp_path), "--n-boot", "20"))
    pol = json.loads((tmp_path / "c4_whale_summary.json").read_text())["policies"]["skip"]
    assert "observed_mean_slope" not in pol["auc_zipf_vs"]["uniform|repeats=1"]
    assert "observed_mean_unit_types" not in pol["describe"]["zipf|repeats=1"]


# ---- analyze_c2 ------------------------------------------------------------------------------------


def test_c2_ratio_is_the_mean_paired_shift_over_the_clean_sd(tmp_path):
    clean_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = []
    for d, v in enumerate(clean_values):
        for condition in CONDITIONS:
            shift = 3.0 if condition == "confuse_19.19" else 0.0
            row = {"law": "zipf", "brevity": True, "mean_repeats": 1.0, "dataset": d,
                   "condition": condition}
            for s in STATS:
                row[s] = v + shift
                row[f"{s}_undefined_years"] = 0
            rows.append(row)
    csv = tmp_path / "c2_whale.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    ok(run("analyze_c2.py", "--csv", str(csv)))
    cell = json.loads((tmp_path / "c2_whale_summary.json").read_text())["cells"]["zipf|brevity=True|repeats=1"]
    floor = float(np.std(clean_values, ddof=1))
    assert cell["floor_sd"]["loglog_r2"] == pytest.approx(floor)
    moved = cell["conditions"]["confuse_19.19"]["loglog_r2"]
    assert moved["mean_shift"] == pytest.approx(3.0)
    assert moved["ratio"] == pytest.approx(3.0 / floor)
    assert moved["share_beyond"] == 1.0 and moved["sign_consistency"] == 1.0
    still = cell["conditions"]["confuse_6.7"]["loglog_r2"]
    assert still["mean_shift"] == 0.0 and still["ratio"] == 0.0


# ---- analyze_c3 ------------------------------------------------------------------------------------


def test_c3_verdict_denominators_exclude_missing_tests_and_at_bound_fits(tmp_path):
    rows = []
    for law in ("zipf", "uniform"):
        for d in range(3):
            for y in range(4):
                for src in SOURCES:
                    rows.append({
                        "law": law, "mean_repeats": 1.0, "dataset": d, "year": 2010 + y, "source": src,
                        "n_types": 100, "loglog_r2": 0.9, "csn_xmin": 2, "csn_alpha": 2.0,
                        "csn_tail": 60, "tail_fraction": 0.6,
                        "gof_p": None if y == 3 else (0.5 if law == "zipf" else 0.01),
                        "gof_failed": 0, "lr_R": 1.0, "lr_normalized_R": 1.0, "lr_p": 0.05,
                        "lr_at_bound": y == 2, "lr_mu": 0.0, "lr_sigma": 1.0, "lr_gap": 0.0,
                        "error": "gof: failed" if y == 3 else ""})
    csv = tmp_path / "c3_whale.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    ok(run("analyze_c3.py", "--csv", str(csv), "--n-boot", "20"))
    out = json.loads((tmp_path / "c3_whale_summary.json").read_text())
    zipf = out["verdicts"]["discovered|zipf|repeats=1"]
    assert zipf["pl_plausible"] == {"rate": 1.0, "n": 9}         # the 3 years without a p are excluded
    assert out["verdicts"]["discovered|uniform|repeats=1"]["pl_plausible"]["rate"] == 0.0
    assert zipf["lr_favours_pl"] == {"rate": 1.0, "n": 12}
    assert zipf["lr_favours_pl_interior"]["n"] == 9               # at-bound years excluded
    assert zipf["at_bound"]["rate"] == pytest.approx(0.25)
    assert zipf["error"]["rate"] == pytest.approx(0.25)
    auc = out["auc_zipf_vs"]["discovered|uniform|repeats=1"]
    assert auc["gof_p"]["auc"] == 1.0                             # 0.5 vs 0.01 in every dataset
    assert auc["loglog_r2"]["auc"] == 0.5                         # identical values: indistinguishable
