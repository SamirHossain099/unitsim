"""Sequential driver: runs experiment steps one at a time, never two at once (shared lesson 8).

A lock file refuses a second concurrent invocation. Each step logs to results/logs/step_<name>.log and is
skipped with a message if its RAM estimate exceeds 60% of free memory.

Nothing runs by default: every step is long or already done, so name what you want.

Usage
-----
    python run_all.py --dry-run
    python run_all.py --only c4_whale_skip c4_whale_zero c4_whale_summary
"""
import argparse
import fnmatch
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)
import resources  # noqa: E402

PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable
LOGS = os.path.join(HERE, "results", "logs")
LOCK = os.path.join(HERE, "results", ".run_all.lock")


def S(name):
    return os.path.join(SRC, name)


def c4(regime, policy, datasets=30, replicates=200):
    return [PY, S("exp_c4_inference_unit.py"), "--regime", regime, "--undefined", policy,
            "--datasets", str(datasets), "--replicates", str(replicates)]


#: tag -> (tokens per bout, word lengths, lexicon size, mean repeats, datasets). Chosen from
#: results/calibration_finch.csv so zipf and uniform both cut within 0.75 of the published 10.72 cuts
#: per bout (primary: within 0.35). FINDINGS.md F3.
FINCH_CONFIGS = {
    "primary": (15, (2, 4), 10, 1.0, 30),
    "sensA": (20, (1, 3), 15, 1.0, 15),
    "sensB": (20, (2, 3), 15, 2.0, 15),
}


def c4_finch(tag, policy, replicates=200):
    tokens, (lo, hi), n_words, repeats, datasets = FINCH_CONFIGS[tag]
    return [PY, S("exp_c4_inference_unit.py"), "--regime", "finch", "--undefined", policy,
            "--datasets", str(datasets), "--replicates", str(replicates), "--repeats", str(repeats),
            "--finch-tokens-per-bout", str(tokens), "--finch-length-range", str(lo), str(hi),
            "--finch-n-words", str(n_words),
            "--out", os.path.join(HERE, "results", f"c4_finch_{tag}_{policy}.csv")]


# (name, argv, rough peak RAM in GB, note)
STEPS = [
    ("pilot", [PY, S("exp_pilot_c1_c4.py"), "--seeds", "20", "--replicates", "200"], 1.0,
     "C1/C4 pilot at the per-year unit (FINDINGS F1)"),
    ("pilot_summary", [PY, S("analyze_pilot_c1_c4.py")], 0.5, "pilot AUC summary"),
    ("calibrate_finch", [PY, S("calibrate_finch.py")], 0.5,
     "finch generator settings against the published 10.72 cuts per bout"),
    ("c4_whale_skip", c4("whale", "skip"), 2.0, "C4, whale inference unit, undefined groups skipped"),
    ("c4_whale_zero", c4("whale", "zero"), 2.0, "C4, whale inference unit, undefined groups as 0"),
    ("c4_whale_summary", [PY, S("analyze_c4.py"), "--regime", "whale"], 0.5,
     "C4 whale: AUC, share clearing the published z, policy sensitivity"),
    ("fig_c4_whale", [PY, S("figures_c4.py"), "--regime", "whale", "--policy", "skip"], 0.5,
     "Figure: AUC of each statistic, Zipf vs other laws, whale inference unit"),
    ("c2_whale", [PY, S("exp_c2_label_error.py"), "--datasets", "30"], 1.0,
     "C2: element-label error against the clean noise floor, whale inference unit"),
    ("c2_whale_summary", [PY, S("analyze_c2.py")], 0.5,
     "C2 summary: paired shifts as multiples of the clean between-dataset sd"),
    ("c3_whale", [PY, S("exp_c3_estimators.py"), "--datasets", "30", "--n-boot", "100"], 1.0,
     "C3: Clauset fit, bootstrap and lognormal ratio vs log-log R^2, discovered and true units"),
    ("c3_whale_summary", [PY, S("analyze_c3.py")], 0.5,
     "C3 summary: per-year verdict rates and dataset-level AUC, by source"),
    ("c4_whale_song_skip", [PY, S("exp_c4_song.py"), "--undefined", "skip", "--datasets", "30",
                            "--replicates", "200"], 2.0,
     "C4 on song-structured whale years (fixed theme order, repetition), undefined groups skipped"),
    ("c4_whale_song_zero", [PY, S("exp_c4_song.py"), "--undefined", "zero", "--datasets", "30",
                            "--replicates", "200"], 2.0,
     "C4 on song-structured whale years, undefined groups as 0"),
    ("c4_whale_song_summary", [PY, S("analyze_c4.py"), "--regime", "whale", "--tag", "song",
                               "--factor", "stickiness"], 0.5,
     "C4 song summary: AUC, share clearing the published z, policy sensitivity"),
]
for _tag in FINCH_CONFIGS:
    for _policy in ("skip", "zero"):
        STEPS.append((f"c4_finch_{_tag}_{_policy}", c4_finch(_tag, _policy), 1.0,
                      f"C4, finch inference unit, config {_tag}, undefined groups {_policy}"))
    STEPS.append((f"c4_finch_{_tag}_summary",
                  [PY, S("analyze_c4.py"), "--regime", "finch", "--tag", _tag], 0.5,
                  f"C4 finch summary, config {_tag}"))

#: Prerequisites, enforced by the parallel scheduler among steps in the same plan.
DEPS = {
    "pilot_summary": ["pilot"],
    "c4_whale_summary": ["c4_whale_skip", "c4_whale_zero"],
    "fig_c4_whale": ["c4_whale_summary"],
    "c2_whale_summary": ["c2_whale"],
    "c3_whale_summary": ["c3_whale"],
    "c4_whale_song_summary": ["c4_whale_song_skip", "c4_whale_song_zero"],
}
for _tag in FINCH_CONFIGS:
    DEPS[f"c4_finch_{_tag}_summary"] = [f"c4_finch_{_tag}_skip", f"c4_finch_{_tag}_zero"]


# ---- Sweeps closing the reviewer gaps (2026-09-12): 1,000 replicates, the published alternative thresholds, law
# shapes, song parameters, and C3 at finch scale and under song structure. Experiment steps are named
# c4_<kind>_<tag>_<policy> and their summaries c4_<kind>_<tag>_summary; finch sweeps use the primary configuration.

def _out(name):
    return os.path.join(HERE, "results", f"{name}.csv")


def _sweep_argv(kind, file_tag, policy, extra, replicates):
    stem = {"whale": "c4_whale", "finch": "c4_finch", "song": "c4_whale_song"}[kind]
    common = ["--undefined", policy, "--replicates", str(replicates), *extra,
              "--out", _out(f"{stem}_{file_tag}_{policy}")]
    if kind == "whale":
        return [PY, S("exp_c4_inference_unit.py"), "--regime", "whale", "--datasets", "30", *common]
    if kind == "song":
        return [PY, S("exp_c4_song.py"), "--datasets", "30", *common]
    tokens, (lo, hi), n_words, repeats, datasets = FINCH_CONFIGS["primary"]
    return [PY, S("exp_c4_inference_unit.py"), "--regime", "finch", "--datasets", str(datasets),
            "--repeats", str(repeats), "--finch-tokens-per-bout", str(tokens),
            "--finch-length-range", str(lo), str(hi), "--finch-n-words", str(n_words), *common]


def _add_sweep(kind, name, extra, policies=("skip",), replicates=200, gb=1.0):
    file_tag = f"primary_{name}" if kind == "finch" else name
    base = f"c4_{kind}_{file_tag}"
    experiments = []
    for policy in policies:
        step = f"{base}_{policy}"
        STEPS.append((step, _sweep_argv(kind, file_tag, policy, extra, replicates), gb,
                      f"sweep {kind} {name}, undefined {policy}"))
        experiments.append(step)
    argv = [PY, S("analyze_c4.py"), "--regime", "finch" if kind == "finch" else "whale",
            "--tag", f"song_{file_tag}" if kind == "song" else file_tag]
    if kind == "song":
        argv += ["--factor", "stickiness"]
    STEPS.append((f"{base}_summary", argv, 0.5, f"sweep summary {kind} {name}"))
    DEPS[f"{base}_summary"] = experiments


_add_sweep("whale", "rep1000", [], replicates=1000, gb=1.5)
_add_sweep("song", "rep1000", [], replicates=1000, gb=1.5)
_add_sweep("finch", "rep1000", [], policies=("skip", "zero"), replicates=1000)
for _t in (0.25, 0.75):
    _add_sweep("whale", f"thr{_t:g}", ["--threshold", str(_t)])
    _add_sweep("song", f"thr{_t:g}", ["--threshold", str(_t)])
for _t in (0.3, 0.7):
    _add_sweep("finch", f"thr{_t:g}", ["--threshold", str(_t)], policies=("skip", "zero"))
_SHAPES = {
    "sigma0.5": ["--laws", "zipf", "lognormal", "--lognormal-sigma", "0.5"],
    "sigma1.5": ["--laws", "zipf", "lognormal", "--lognormal-sigma", "1.5"],
    "sigma2": ["--laws", "zipf", "lognormal", "--lognormal-sigma", "2.0"],
    "tail0.2": ["--laws", "zipf", "geometric", "--geometric-tail", "0.2"],
    "tail0.01": ["--laws", "zipf", "geometric", "--geometric-tail", "0.01"],
    "zipfexp0.8": ["--zipf-exponent", "0.8"],
    "zipfexp1.2": ["--zipf-exponent", "1.2"],
}
for _kind in ("whale", "finch", "song"):
    for _name, _extra in _SHAPES.items():
        _add_sweep(_kind, _name, _extra)
for _name, _extra in {"themes4": ["--n-themes", "4"], "themes12": ["--n-themes", "12"],
                      "variants2": ["--variants-per-theme", "2"],
                      "variants5": ["--variants-per-theme", "5"],
                      "ppc20": ["--phrases-per-cycle", "20"], "ppc80": ["--phrases-per-cycle", "80"]}.items():
    _add_sweep("song", _name, _extra)
STEPS += [
    ("c3_finch", [PY, S("exp_c3_estimators.py"), "--regime", "finch", "--datasets", "30", "--n-boot", "100"],
     1.0, "C3 at the finch inference unit"),
    ("c3_finch_summary", [PY, S("analyze_c3.py"), "--csv", _out("c3_finch")], 0.5, "C3 finch summary"),
    ("c3_song", [PY, S("exp_c3_estimators.py"), "--regime", "song", "--datasets", "30", "--n-boot", "100"],
     1.0, "C3 on song-structured whale years"),
    ("c3_song_summary", [PY, S("analyze_c3.py"), "--csv", _out("c3_song"), "--factor", "stickiness"], 0.5,
     "C3 song summary"),
]
DEPS["c3_finch_summary"] = ["c3_finch"]
DEPS["c3_song_summary"] = ["c3_song"]

# ---- The calibration of CALIBRATION-DESIGN.md: reference and held-out datasets per regime, then its validation.
N_PER_LAW = 1000
for _regime in ("whale", "finch", "song"):
    STEPS.append((f"cal_{_regime}",
                  [PY, S("exp_calibration.py"), "--regime", _regime, "--n-per-law", str(N_PER_LAW)], 1.0,
                  f"calibration reference and held-out datasets, {_regime}"))
    STEPS.append((f"cal_{_regime}_summary",
                  [PY, S("analyze_calibration.py"), "--regime", _regime], 0.5,
                  f"calibration validation, {_regime}"))
    DEPS[f"cal_{_regime}_summary"] = [f"cal_{_regime}"]
    STEPS.append((f"cal_{_regime}_wide",
                  [PY, S("exp_calibration.py"), "--regime", _regime, "--n-per-law", str(N_PER_LAW // 2),
                   "--prior", "wide"], 1.0,
                  f"calibration stress test, held-out datasets from a wider prior, {_regime}"))
    STEPS.append((f"cal_{_regime}_wide_summary",
                  [PY, S("analyze_calibration.py"), "--regime", _regime,
                   "--holdout-csv", _out(f"calibration_{_regime}_wide")], 0.5,
                  f"calibration under prior mismatch, {_regime}"))
    DEPS[f"cal_{_regime}_wide_summary"] = [f"cal_{_regime}", f"cal_{_regime}_wide"]


def run_step(name, argv, need_gb, note):
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"step_{name}.log")
    free = resources.available_ram_bytes() / 1e9
    print(f"\n{'=' * 72}\n[{name}] {note}\n  needs ~{need_gb:.1f} GB | free {free:.1f} GB | log {log}",
          flush=True)
    if need_gb > 0.6 * free:
        print(f"  SKIPPED: needs {need_gb:.1f} GB, cap is 60% of {free:.1f} GB free")
        return False
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    ok = p.returncode == 0
    print(f"  {'OK' if ok else f'FAILED (exit {p.returncode})'} in {time.time() - t0:.0f}s", flush=True)
    if not ok:
        with open(log, encoding="utf-8") as fh:
            for line in fh.read().strip().splitlines()[-12:]:
                print("   ", line)
    return ok


def run_parallel(steps, workers, poll_s=5.0):
    """Run steps concurrently, at most `workers` at once, each capped at one BLAS thread.

    A step starts only when every prerequisite in DEPS that is also in this plan has succeeded, and is skipped if
    one failed. A step whose RAM estimate exceeds 60% of free memory waits for running steps to finish, and is
    skipped only if it still does not fit with nothing else running. Returns {name: succeeded}.
    """
    os.makedirs(LOGS, exist_ok=True)
    in_plan = {s[0] for s in steps}
    pending, running, results = list(steps), {}, {}
    env = {**os.environ, "UNITSIM_THREADS": "1"}
    while pending or running:
        changed = False
        for step in list(pending):
            if len(running) >= workers:
                break
            name, argv, need_gb, note = step
            deps = [d for d in DEPS.get(name, ()) if d in in_plan]
            if any(results.get(d) is False for d in deps):
                pending.remove(step)
                results[name] = False
                changed = True
                print(f"[{name}] SKIPPED: a prerequisite failed", flush=True)
                continue
            if not all(results.get(d) for d in deps):
                continue
            free = resources.available_ram_bytes() / 1e9
            if need_gb > 0.6 * free:
                if running:
                    continue
                pending.remove(step)
                results[name] = False
                changed = True
                print(f"[{name}] SKIPPED: needs {need_gb:.1f} GB, cap is 60% of {free:.1f} GB free", flush=True)
                continue
            log = open(os.path.join(LOGS, f"step_{name}.log"), "w", encoding="utf-8")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, env=env)
            running[name] = (proc, log, time.time())
            pending.remove(step)
            changed = True
            print(f"[{name}] started ({len(running)} running, {len(pending)} pending): {note}", flush=True)
        for name, (proc, log, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            results[name] = rc == 0
            del running[name]
            changed = True
            print(f"[{name}] {'OK' if rc == 0 else f'FAILED (exit {rc})'} in {time.time() - t0:.0f}s",
                  flush=True)
        if pending and not running and not changed:
            for name, *_ in pending:
                results[name] = False
                print(f"[{name}] SKIPPED: could not be scheduled", flush=True)
            pending = []
        if running:
            time.sleep(poll_s)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", default=[], help="step names to run, in listed order")
    ap.add_argument("--match", nargs="+", default=[],
                    help="fnmatch patterns; matching steps join the plan after --only, in STEPS order")
    ap.add_argument("--workers", type=int, default=1,
                    help="run up to this many steps at once, each on one BLAS thread (1 = sequential)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    by_name = {s[0]: s for s in STEPS}
    unknown = [n for n in a.only if n not in by_name]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; known: {sorted(by_name)}")
    if len(set(a.only)) != len(a.only):
        ap.error("a step is named more than once")
    if a.workers < 1:
        ap.error("--workers must be at least 1")
    names = list(a.only)
    for s in STEPS:
        if s[0] not in names and any(fnmatch.fnmatch(s[0], p) for p in a.match):
            names.append(s[0])
    if a.match and len(names) == len(a.only):
        ap.error(f"no step matches {a.match}")
    steps = [by_name[n] for n in names]   # --only in the order given, then --match in STEPS order
    resources.report("resources: ")
    print("plan:", " -> ".join(s[0] for s in steps) or "(nothing; use --only)")
    if a.dry_run or not steps:
        for n, argv, gb, note in steps:
            print(f"  {n:18s} ~{gb:.1f} GB  {' '.join(argv[1:])}")
        return 0

    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"another run_all appears to be running ({LOCK} exists). "
              "If it is stale, delete the file and retry.")
        return 2
    try:
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        os.close(fd)
        if a.workers > 1:
            results = run_parallel(steps, a.workers)
        else:
            results = {}
            for n, argv, gb, note in steps:
                results[n] = run_step(n, argv, gb, note)
                if not results[n]:
                    print(f"stopping: {n} failed, and later steps may depend on it")
                    break
    finally:
        os.remove(LOCK)
    print(f"\n{'=' * 72}\nSUMMARY")
    for n, ok in results.items():
        print(f"  {n:18s} {'OK' if ok else 'FAILED/SKIPPED'}")
    return 0 if results and all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
