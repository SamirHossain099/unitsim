"""A bootstrap that cannot sample (alpha too close to 1) must leave gof_p empty with a reason, not end the run."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import exp_c3_estimators as c3  # noqa: E402


def test_gof_overflow_is_recorded_not_raised(monkeypatch):
    def overflow(*args, **kwargs):
        raise OverflowError("a draw exceeded 1e+15; alpha=1.35 is too close to 1 for this size")

    monkeypatch.setattr(c3, "gof_pvalue", overflow)
    counts = np.array([200, 90, 60, 40, 30, 22, 18, 15, 12, 10, 8, 7, 6, 5, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1])
    out = c3.csn_stats(counts, np.random.default_rng(0), n_boot=10, max_failed_fraction=0.05)
    assert out["gof_p"] is None and out["gof_failed"] is None
    assert "gof: a draw exceeded" in out["error"]
    assert out["csn_alpha"] is not None and out["loglog_r2"] is not None   # the rest of the group survives
