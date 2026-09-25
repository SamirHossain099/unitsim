"""`resources` must be imported before numpy in every entry point.

numpy reads OMP_NUM_THREADS and friends at import time, and `resources` sets them. If a formatter
reorders the import block so numpy lands first, the thread cap silently stops working. This happened in
project 01; the fence and this test are why it cannot happen here.
"""
import glob
import os
import re

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
HEAVY = {name: re.compile(rf"^\s*(?:import {name}|from {name})", re.M)
         for name in ("numpy", "pandas", "scipy", "sklearn", "umap", "unitsim")}


def entry_points():
    out = []
    for p in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        if os.path.basename(p) == "resources.py":
            continue
        out.append((p, open(p, encoding="utf-8").read()))
    return out


def test_there_are_entry_points_to_check():
    assert entry_points(), "no entry points found under src/ -- did they move?"


def test_every_entry_point_imports_resources():
    for p, s in entry_points():
        assert re.search(r"^import resources\b", s, re.M), (
            f"{os.path.basename(p)} does not import resources; its BLAS threads are uncapped")


def test_resources_precedes_heavy_imports():
    for p, s in entry_points():
        r = re.search(r"^import resources\b", s, re.M).start()
        for name, rx in HEAVY.items():
            m = rx.search(s)
            if m:
                assert r < m.start(), (
                    f"{os.path.basename(p)}: `import resources` comes after `{name}`; "
                    "the thread cap will not take effect")


def test_isort_fence_is_present():
    for p, s in entry_points():
        assert "# isort: off" in s and "# isort: on" in s, (
            f"{os.path.basename(p)}: missing the isort fence around the resources import")
