"""analyze_sweeps.py assigns each summary to a regime and sweep by file name; a wrong parse silently mislabels rows."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import analyze_sweeps as sw  # noqa: E402


def test_parse_name_covers_every_summary_the_driver_writes():
    cases = {
        "c4_whale_summary.json": ("whale", "", "base"),
        "c4_whale_song_summary.json": ("song", "", "base"),
        "c4_finch_primary_summary.json": ("finch", "primary", "base"),
        "c4_finch_sensB_summary.json": ("finch", "sensB", "base"),
        "c4_whale_thr0.25_summary.json": ("whale", "", "thr0.25"),
        "c4_whale_song_ppc80_summary.json": ("song", "", "ppc80"),
        "c4_whale_song_rep1000_summary.json": ("song", "", "rep1000"),
        "c4_finch_primary_thr0.7_summary.json": ("finch", "primary", "thr0.7"),
        "c4_finch_primary_zipfexp1.2_summary.json": ("finch", "primary", "zipfexp1.2"),
    }
    for name, expected in cases.items():
        assert sw.parse_name(name) == expected, name
    for name in ("c3_whale_summary.json", "c4_whale_bogus_summary.json", "c4_whale_skip.csv"):
        assert sw.parse_name(name) is None, name


def test_collect_reads_auc_and_confound_columns(tmp_path):
    cell = {"auc": 0.7, "ci95": [0.6, 0.8]}
    describe = {"observed_mean_r2": {"mean": 0.9}, "cuts_per_sequence": {"mean": 10.0}}
    summary = {"policies": {"skip": {
        "auc_zipf_vs": {"uniform|repeats=1": {"shuffle_z": cell}},
        "describe": {"zipf|repeats=1": describe, "uniform|repeats=1": describe},
        "share_clearing_published_z": {"zipf|repeats=1": {"shuffle": 1.0, "rotate": 0.5},
                                       "uniform|repeats=1": {"shuffle": 0.2, "rotate": 0.4}}}}}
    (tmp_path / "c4_whale_thr0.75_summary.json").write_text(json.dumps(summary))
    df = sw.collect(str(tmp_path))
    row = df.iloc[0].to_dict()
    assert len(df) == 1
    assert (row["kind"], row["sweep"], row["law"], row["level"], row["statistic"]) == \
        ("whale", "thr0.75", "uniform", "repeats=1", "shuffle_z")
    assert (row["auc"], row["lo"], row["hi"], row["cuts_law"]) == (0.7, 0.6, 0.8, 10.0)
    assert (row["share_zipf"], row["share_law"]) == (0.5, 0.2)    # every published z must be cleared
