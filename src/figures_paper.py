"""The paper's figures.

Reads the C4 and C3 summaries and CSVs in results/ and writes figures/fig{1..4}_*.png (600 dpi) and .pdf, each with a
.json caching every plotted number, so a figure's numbers exist outside the image.

    fig1_published_settings   AUC for Zipf against lognormal and geometric at the published settings, four statistics
    fig2_threshold            bigram z (whale words) and rotated z (whale song) AUC by threshold; finch undefined share
    fig3_shape                mean R^2 AUC against the difference in most-frequent-word probability, whale and song
    fig4_powerlaw_tests       bootstrap p and likelihood-ratio AUC, true words against discovered units
    fig5_calibration          calibration validation: stated confidence against share correct, and the
                              accuracy bought by abstaining, under the matched and the wide prior

Design (dataviz skill): at most three hues on screen, the reference palette's first three categorical slots, which
validate all-pairs in light mode. No text is smaller than 7.5 pt (Royal Society figure rule). Text uses ink tokens; gridlines are hairlines; a vertical or horizontal line marks
AUC 0.5; panel letters are in the artwork. Static manuscript figures, so no hover layer.

Usage
-----
    python src/figures_paper.py            # all four
    python src/figures_paper.py --only 3
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import exp_c4_inference_unit as c4  # noqa: E402
from unitsim.generate import word_probabilities  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

SURFACE = "#fcfcfb"      # reference palette, chart chrome and ink, light
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")   # categorical slots 1-3, validated all-pairs
LAW_COLOUR = {"uniform": SLOTS[0], "lognormal": SLOTS[1], "geometric": SLOTS[2]}
WIDTH = 7.2   # inches, two-column width

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(name):
    path = os.path.join(RES, name)
    if name.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return pd.read_csv(path)


def auc_cell(summary, key, stat, policy="skip"):
    cell = load(summary)["policies"][policy]["auc_zipf_vs"][key][stat]
    return {"auc": cell["auc"], "lo": cell["ci95"][0], "hi": cell["ci95"][1]}


def letter(ax, text, title):
    ax.set_title(f"({text})  {title}", loc="left", color=INK, pad=6)


def save(fig, stem, data):
    os.makedirs(OUT, exist_ok=True)
    png, pdf = (os.path.join(OUT, f"{stem}.{ext}") for ext in ("png", "pdf"))
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT, f"{stem}.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"wrote {png}, .pdf and .json")


def half_line(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(0.5, color=INK_2, lw=0.8, zorder=1)


def auc_axis(ax, vertical=True):
    lim = (-0.03, 1.03)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    if vertical:
        ax.set_xlim(*lim)
        ax.set_xticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    else:
        ax.set_ylim(*lim)
        ax.set_yticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)


def interval(ax, x, y, lo, hi, colour, horizontal=True, filled=True):
    if horizontal:
        ax.plot([lo, hi], [y, y], color=colour, lw=1.4, solid_capstyle="round", zorder=2)
    else:
        ax.plot([x, x], [lo, hi], color=colour, lw=1.4, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], marker="o", ms=4.8, color=colour if filled else SURFACE, mec=colour if not filled else SURFACE,
            mew=1.3 if not filled else 1.0, zorder=3, linestyle="none")


# ---- figure 1 --------------------------------------------------------------------------------------------------------

REGIMES = [("whale words, m = 1", "c4_whale_summary.json", "repeats=1"),
           ("whale words, m = 5", "c4_whale_summary.json", "repeats=5"),
           ("finch", "c4_finch_primary_summary.json", "repeats=1"),
           ("whale song, s = 0", "c4_whale_song_summary.json", "stickiness=0"),
           ("whale song, s = 0.8", "c4_whale_song_summary.json", "stickiness=0.8")]
STATS = [("observed_mean_r2", "Mean R²"), ("shuffle_z", "Shuffled z"), ("rotate_z", "Rotated z"),
         ("bigram_z", "Bigram z")]


def figure1():
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.9), sharey=True)
    data = {}
    ys = np.arange(len(REGIMES))[::-1]
    for i, (ax, (stat, title)) in enumerate(zip(axes, STATS)):
        half_line(ax)
        for y, (label, summary, level) in zip(ys, REGIMES):
            for law, dy in (("lognormal", 0.16), ("geometric", -0.16)):
                c = auc_cell(summary, f"{law}|{level}", stat)
                data[f"{stat}|{label}|{law}"] = c
                interval(ax, c["auc"], y + dy, c["lo"], c["hi"], LAW_COLOUR[law])
        auc_axis(ax)
        ax.set_yticks(ys, [r[0] for r in REGIMES])
        ax.tick_params(axis="y", length=0, colors=INK_2)
        ax.set_xlabel("AUC, Zipf vs alternative")
        letter(ax, "abcd"[i], title)
    handles = [Line2D([0], [0], lw=0, marker="o", ms=4.8, color=LAW_COLOUR[law], label=law)
               for law in ("lognormal", "geometric")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 1.04))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig1_published_settings", data)


# ---- figure 2 --------------------------------------------------------------------------------------------------------

def figure2():
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    data = {}
    whale = [(0.25, "c4_whale_thr0.25_summary.json"), (0.5, "c4_whale_summary.json"),
             (0.75, "c4_whale_thr0.75_summary.json")]
    song = [(0.25, "c4_whale_song_thr0.25_summary.json"), (0.5, "c4_whale_song_summary.json"),
            (0.75, "c4_whale_song_thr0.75_summary.json")]
    panels = [(axes[0], whale, "repeats=1", "bigram_z", "Whale words, bigram z", (-0.012, 0.0, 0.012)),
              (axes[1], song, "stickiness=0.8", "rotate_z", "Whale song, rotated z", (-0.012, 0.0, 0.012))]
    for k, (ax, runs, level, stat, title, offsets) in enumerate(panels):
        half_line(ax, vertical=False)
        for law, dx in zip(("uniform", "lognormal", "geometric"), offsets):
            cells = [auc_cell(summary, f"{law}|{level}", stat) for _, summary in runs]
            xs = [t + dx for t, _ in runs]
            ax.plot(xs, [c["auc"] for c in cells], color=LAW_COLOUR[law], lw=1.0, zorder=2)
            for x, c in zip(xs, cells):
                interval(ax, x, c["auc"], c["lo"], c["hi"], LAW_COLOUR[law], horizontal=False)
            data[f"{title}|{law}"] = {str(t): c for (t, _), c in zip(runs, cells)}
        auc_axis(ax, vertical=False)
        ax.set_xticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"])
        ax.set_xlim(0.19, 0.81)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("AUC, Zipf vs alternative")
        letter(ax, "ab"[k], title)
    ax = axes[2]
    finch = [(0.3, "c4_finch_primary_thr0.3_skip.csv"), (0.5, "c4_finch_primary_skip.csv"),
             (0.7, "c4_finch_primary_thr0.7_skip.csv")]
    for law in ("zipf", "uniform", "lognormal", "geometric"):
        shares = [float(load(f).query("law == @law").shuffle_undefined_rate.mean()) for _, f in finch]
        data[f"finch undefined|{law}"] = {str(t): v for (t, _), v in zip(finch, shares)}
        if law == "zipf":
            ax.plot([t for t, _ in finch], shares, color=INK_2, lw=1.0, ls=(0, (3, 2)), marker="o", ms=3.5, zorder=3)
        else:
            ax.plot([t for t, _ in finch], shares, color=LAW_COLOUR[law], lw=1.0, marker="o", ms=4.2, zorder=2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_xticks([0.3, 0.5, 0.7], ["0.3", "0.5", "0.7"])
    ax.set_xlim(0.25, 0.75)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Shuffled recordings undefined")
    letter(ax, "c", "Finch, undefined shuffled R²")
    handles = [Line2D([0], [0], lw=1.0, marker="o", ms=4.2, color=LAW_COLOUR[law], label=law)
               for law in ("uniform", "lognormal", "geometric")]
    handles.append(Line2D([0], [0], lw=1.0, ls=(0, (3, 2)), marker="o", ms=3.5, color=INK_2, label="zipf (panel c)"))
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 1.06))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig2_threshold", data)


# ---- figure 3 --------------------------------------------------------------------------------------------------------

DRAWS = 200


def top_word(law, n_words, **shape):
    if law == "uniform":
        return 1.0 / n_words
    tops = []
    for seed in range(DRAWS):
        params = c4.law_params(law, n_words, **shape)
        params.pop("law")
        tops.append(np.max(word_probabilities(n_words, law, rng=np.random.default_rng(seed), **params)))
    return float(np.mean(tops))


def shape_comparisons(prefix):
    """(summary, law, zipf shape, alternative shape) for every shape comparison of a regime."""
    out = [(f"{prefix}_summary.json", law, {}, {}) for law in ("uniform", "lognormal", "geometric")]
    for tag, law, shape in (("sigma0.5", "lognormal", {"lognormal_sigma": 0.5}),
                            ("sigma1.5", "lognormal", {"lognormal_sigma": 1.5}),
                            ("sigma2", "lognormal", {"lognormal_sigma": 2.0}),
                            ("tail0.2", "geometric", {"geometric_tail": 0.2}),
                            ("tail0.01", "geometric", {"geometric_tail": 0.01})):
        out.append((f"{prefix}_{tag}_summary.json", law, {}, shape))
    for tag, exponent in (("zipfexp0.8", 0.8), ("zipfexp1.2", 1.2)):
        for law in ("uniform", "lognormal", "geometric"):
            out.append((f"{prefix}_{tag}_summary.json", law, {"zipf_exponent": exponent}, {}))
    return out


def figure3():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharey=True)
    data = {}
    regimes = [(axes[0], "c4_whale", 100, ("repeats=1", "repeats=5"), ("m = 1", "m = 5"), "Whale words"),
               (axes[1], "c4_whale_song", 24, ("stickiness=0", "stickiness=0.8"), ("s = 0", "s = 0.8"), "Whale song")]
    for k, (ax, prefix, n_words, levels, level_labels, title) in enumerate(regimes):
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        half_line(ax, vertical=False)
        points = []
        for summary, law, zipf_shape, alt_shape in shape_comparisons(prefix):
            diff = top_word("zipf", n_words, **zipf_shape) - top_word(law, n_words, **alt_shape)
            for j, level in enumerate(levels):
                c = auc_cell(summary, f"{law}|{level}", "observed_mean_r2")
                points.append((diff, c, j))
                data[f"{title}|{summary}|{law}|{level}"] = {"top_word_difference": diff, **c}
        for diff, c, j in points:
            dx = (-0.002, 0.002)[j]
            interval(ax, diff + dx, c["auc"], c["lo"], c["hi"], SLOTS[j], horizontal=False)
        auc_axis(ax, vertical=False)
        ax.set_xlabel("Most-frequent-word probability, Zipf minus alternative")
        if k == 0:
            ax.set_ylabel("Mean R² AUC, Zipf vs alternative")
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        letter(ax, "ab"[k], title)
        handles = [Line2D([0], [0], lw=0, marker="o", ms=4.8, color=SLOTS[j], label=level_labels[j]) for j in (0, 1)]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_shape", data)


# ---- figure 4 --------------------------------------------------------------------------------------------------------

C3_ROWS = [("whale words, m = 1", "c3_whale_summary.json", "repeats=1"),
           ("whale words, m = 5", "c3_whale_summary.json", "repeats=5"),
           ("finch", "c3_finch_summary.json", "repeats=1"),
           ("whale song, s = 0", "c3_song_summary.json", "stickiness=0"),
           ("whale song, s = 0.8", "c3_song_summary.json", "stickiness=0.8")]


def figure4():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), sharey=True)
    data = {}
    rows = [(label, summary, level, law) for label, summary, level in C3_ROWS for law in ("lognormal", "geometric")]
    ys = np.arange(len(rows))[::-1]
    for k, (ax, stat, title) in enumerate(((axes[0], "gof_p", "Bootstrap goodness-of-fit p"),
                                           (axes[1], "lr_normalized_R", "Likelihood ratio, power law vs lognormal"))):
        half_line(ax)
        for y, (label, summary, level, law) in zip(ys, rows):
            cells = load(summary)["auc_zipf_vs"]
            for source, dy, filled, colour in (("true", 0.17, True, SLOTS[0]), ("discovered", -0.17, True, SLOTS[1])):
                raw = cells[f"{source}|{law}|{level}"][stat]
                c = {"auc": raw["auc"], "lo": raw["ci95"][0], "hi": raw["ci95"][1]}
                data[f"{stat}|{label}|{law}|{source}"] = c
                interval(ax, c["auc"], y + dy, c["lo"], c["hi"], colour, filled=filled)
        auc_axis(ax)
        ax.set_yticks(ys, [f"{label}, {law}" for label, _, _, law in rows])
        ax.tick_params(axis="y", length=0, colors=INK_2)
        ax.set_xlabel("AUC, Zipf vs alternative")
        letter(ax, "ab"[k], title)
    handles = [Line2D([0], [0], lw=0, marker="o", ms=4.8, color=SLOTS[0], label="true word counts"),
               Line2D([0], [0], lw=0, marker="o", ms=4.8, color=SLOTS[1], label="discovered units")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 1.03))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig4_powerlaw_tests", data)



# ---- figure 5 --------------------------------------------------------------------------------------------------------

CAL_REGIMES = (("whale", "whale words"), ("finch", "finch"), ("song", "whale song"))
CAL_COLOUR = {"whale": SLOTS[0], "finch": SLOTS[1], "song": SLOTS[2]}
CAL_THRESHOLDS = np.round(np.arange(0.26, 0.99, 0.02), 2)


def _calibration_scores(regime):
    """Posterior confidence and correctness for the held-out datasets, under the matched and the wide prior."""
    from unitsim.calibrate import build_reference, posterior

    rows = pd.read_csv(os.path.join(RES, f"calibration_{regime}.csv")).to_dict("records")
    wide = pd.read_csv(os.path.join(RES, f"calibration_{regime}_wide.csv")).to_dict("records")
    ref = build_reference([r for r in rows if r["set"] == "reference"])
    out = {}
    for name, held in (("matched", [r for r in rows if r["set"] == "holdout"]), ("wide", wide)):
        conf, correct = [], []
        for r in held:
            post = posterior(ref, r)
            law, p = max(post.items(), key=lambda kv: kv[1])
            conf.append(p)
            correct.append(law == r["law"])
        out[name] = (np.array(conf), np.array(correct, dtype=bool))
    return out


def figure5():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    data = {}
    scores = {regime: _calibration_scores(regime) for regime, _ in CAL_REGIMES}
    ax = axes[0]
    ax.plot([0.25, 1], [0.25, 1], color=INK_2, lw=0.8, zorder=1)
    edges = np.array([0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
    for regime, label in CAL_REGIMES:
        for prior, style in (("matched", dict(ls="-", marker="o")), ("wide", dict(ls=(0, (3, 2)), marker="o"))):
            conf, correct = scores[regime][prior]
            xs, ys, ns = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                take = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
                if take.sum() >= 20:
                    xs.append(float(conf[take].mean()))
                    ys.append(float(correct[take].mean()))
                    ns.append(int(take.sum()))
            data[f"reliability|{regime}|{prior}"] = {"confidence": xs, "share_correct": ys, "n": ns}
            face = CAL_COLOUR[regime] if prior == "matched" else SURFACE
            ax.plot(xs, ys, color=CAL_COLOUR[regime], lw=1.2, ms=4.8, mfc=face,
                    mec=CAL_COLOUR[regime], mew=1.3, zorder=2, **style)
    ax.set_xlim(0.22, 1.02)
    ax.set_ylim(0.22, 1.02)
    ax.set_xticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"])
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("Share correct")
    letter(ax, "a", "Reliability")
    ax = axes[1]
    for regime, label in CAL_REGIMES:
        for prior, style in (("matched", dict(ls="-")), ("wide", dict(ls=(0, (3, 2))))):
            conf, correct = scores[regime][prior]
            xs, ys = [], []
            for t in CAL_THRESHOLDS:
                take = conf >= t
                if take.sum() >= 20:
                    xs.append(float(take.mean()))
                    ys.append(float(correct[take].mean()))
            data[f"decision|{regime}|{prior}"] = {"share_decided": xs, "accuracy_when_decided": ys,
                                                  "thresholds": CAL_THRESHOLDS.tolist()[:len(xs)]}
            ax.plot(xs, ys, color=CAL_COLOUR[regime], lw=1.4, zorder=2, **style)
    ax.axhline(0.25, color=INK_2, lw=0.8, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.22, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0.25, 0.5, 0.75, 1], ["0.25", "0.5", "0.75", "1"])
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Share of datasets given a verdict")
    ax.set_ylabel("Accuracy of those verdicts")
    letter(ax, "b", "Accuracy against coverage")
    handles = [Line2D([0], [0], color=CAL_COLOUR[r], lw=1.4, label=lab) for r, lab in CAL_REGIMES]
    handles += [Line2D([0], [0], color=INK_2, lw=1.4, label="matched prior"),
                Line2D([0], [0], color=INK_2, lw=1.4, ls=(0, (3, 2)), label="wide prior")]
    fig.legend(handles=handles, loc="upper right", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(1.0, 1.04))
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig5_calibration", data)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", type=int, nargs="+", choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    for n in a.only:
        {1: figure1, 2: figure2, 3: figure3, 4: figure4, 5: figure5}[n]()


if __name__ == "__main__":
    main()
