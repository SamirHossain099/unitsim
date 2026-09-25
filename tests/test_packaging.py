"""The artifact must contain the library and nothing else.

Adapted from project 01, where both failure modes happened: a wheel that built "successfully" with no
modules in it, and an sdist that swept in the manuscript, working notes, results and CLAUDE.md. This
project sits beside NOVELTY.md, VERIFICATION.md, CLAUDE.md and third-party papers in refs/, so the same
leak is one missing allowlist line away. These tests build the real distributions and inspect them,
because that is the only check that would catch either failure.
"""
import glob
import os
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Substring-matched against every archive entry.
FORBIDDEN = ("CLAUDE.md", "NOVELTY.md", "VERIFICATION.md", "FINDINGS.md",
             "results/", "refs/", ".venv", "validation/", "tests/")

REQUIRED = ("unitsim/__init__.py", "unitsim/generate.py", "unitsim/corrupt.py",
            "unitsim/metrics.py", "unitsim/powerlaw_fit.py", "unitsim/tp.py",
            "unitsim/baselines.py")


@pytest.fixture(scope="module")
def dists(tmp_path_factory):
    pytest.importorskip("build", reason="`pip install build` to run packaging tests")
    out = tmp_path_factory.mktemp("dist")
    r = subprocess.run([sys.executable, "-m", "build", "--outdir", str(out), ROOT],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    wheels = glob.glob(os.path.join(str(out), "*.whl"))
    sdists = glob.glob(os.path.join(str(out), "*.tar.gz"))
    assert wheels, "build produced no wheel"
    assert sdists, "build produced no sdist"
    return wheels[0], sdists[0]


def _assert_clean(names, what):
    leaked = sorted({n for n in names for f in FORBIDDEN if f in n})
    assert not leaked, (
        f"{what} ships {len(leaked)} file(s) that must stay private:\n  " + "\n  ".join(leaked[:20])
        + "\nThe sdist allowlist in pyproject.toml has stopped working.")


def test_wheel_contains_the_package(dists):
    names = zipfile.ZipFile(dists[0]).namelist()
    assert any(n.endswith(".py") for n in names), f"wheel has no Python modules: {names}"
    for required in REQUIRED:
        assert required in names, f"{required} missing from the wheel"


def test_wheel_does_not_ship_working_files(dists):
    _assert_clean(zipfile.ZipFile(dists[0]).namelist(), "wheel")


def test_sdist_does_not_ship_working_files(dists):
    _assert_clean(tarfile.open(dists[1]).getnames(), "sdist")


def test_sdist_is_small(dists):
    mb = os.path.getsize(dists[1]) / 1e6
    assert mb < 0.5, f"sdist is {mb:.2f} MB; it has almost certainly swept in refs/ or results/"


def test_declared_dependencies_are_actually_imported():
    """Hard dependencies must be ones the library imports; everything else belongs in an extra."""
    try:
        import tomllib
    except ImportError:
        pytest.skip("tomllib needs Python 3.11+")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        cfg = tomllib.load(fh)
    declared = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower()
                for d in cfg["project"]["dependencies"]}
    src = "".join(open(f, encoding="utf-8").read()
                  for f in glob.glob(os.path.join(ROOT, "src", "unitsim", "*.py")))
    for dep in declared:
        assert f"import {dep}" in src or f"from {dep}" in src, (
            f"`{dep}` is a hard dependency but nothing in src/unitsim imports it")
