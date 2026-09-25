"""The parallel scheduler decides which summaries run on which data, so its ordering and skip rules are tested."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_all  # noqa: E402

OK = [sys.executable, "-c", "import time; time.sleep(0.3)"]
BAD = [sys.executable, "-c", "raise SystemExit(3)"]


def test_prerequisites_gate_dependents_and_failures_skip_them(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "LOGS", str(tmp_path))
    monkeypatch.setattr(run_all, "DEPS", {"a_child": ["a"], "b_child": ["b"], "grandchild": ["b_child"]})
    steps = [("a_child", OK, 0.0, ""), ("a", OK, 0.0, ""), ("b", BAD, 0.0, ""),
             ("b_child", OK, 0.0, ""), ("grandchild", OK, 0.0, ""), ("free", OK, 0.0, "")]
    results = run_all.run_parallel(steps, workers=3, poll_s=0.05)
    assert results == {"a": True, "a_child": True, "b": False, "b_child": False, "grandchild": False,
                       "free": True}
    assert not (tmp_path / "step_b_child.log").exists()           # a skipped step never started
    assert os.path.getmtime(tmp_path / "step_a_child.log") >= os.path.getmtime(tmp_path / "step_a.log")


def test_prerequisites_outside_the_plan_do_not_block(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all, "LOGS", str(tmp_path))
    monkeypatch.setattr(run_all, "DEPS", {"summary": ["experiment_run_earlier"]})
    assert run_all.run_parallel([("summary", OK, 0.0, "")], workers=2, poll_s=0.05) == {"summary": True}


def test_every_sweep_summary_depends_on_steps_that_exist():
    names = {s[0] for s in run_all.STEPS}
    assert len(names) == len(run_all.STEPS), "duplicate step names"
    for summary, deps in run_all.DEPS.items():
        assert summary in names and all(d in names for d in deps), summary
