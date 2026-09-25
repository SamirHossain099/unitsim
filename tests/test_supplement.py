"""SUPPLEMENT-TABLES.md is generated from results/; it must be current, and main Table 1 must agree with Table S1."""
import os
import re
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPP = os.path.join(ROOT, "SUPPLEMENT-TABLES.md")
DRAFT = os.path.join(ROOT, "PAPER-DRAFT.md")


def test_supplement_is_regenerated_identically(tmp_path):
    if not os.path.exists(SUPP):
        pytest.skip("supplement not built")
    before = open(SUPP, encoding="utf-8").read()
    out = subprocess.run([sys.executable, os.path.join(ROOT, "src", "tables_paper.py")], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert open(SUPP, encoding="utf-8").read() == before, "SUPPLEMENT-TABLES.md was stale; rerun src/tables_paper.py"


def test_main_table1_matches_table_s1():
    if not os.path.exists(DRAFT):
        pytest.skip("no draft")
    s1 = pd.read_csv(os.path.join(ROOT, "tables", "table_s1.csv"))
    text = open(DRAFT, encoding="utf-8").read()
    table = text.split("**Table 1.**", 1)[1].split("**Figure 1**", 1)[0]
    rows = [r for r in table.splitlines() if r.startswith("|") and re.search(r"\d\.\d\d", r)]
    regime = None
    checked = 0
    stats = ["observed_mean_r2", "shuffle_z", "rotate_z", "bigram_z"]
    for r in rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        regime = cells[0] or regime
        law, values = cells[1], [float(v) for v in cells[2:]]
        for stat, v in zip(stats, values):
            hit = s1[(s1.regime == regime) & (s1.alternative == law) & (s1.statistic == stat)]
            assert len(hit) == 1, (regime, law, stat)
            assert round(float(hit.auc.iloc[0]), 2) == pytest.approx(v, abs=5.01e-3), (regime, law, stat)
            checked += 1
    assert checked == 60
