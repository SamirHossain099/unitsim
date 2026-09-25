"""Figure: which statistic can tell a Zipfian source from another frequency law? (C4)

Reads results/c4_<regime>[_<tag>]_summary.json. Writes figures/fig_c4_<regime>[_<tag>]_<policy>.png and a .json
beside it caching every plotted number (a figure's numbers must exist outside the PNG).

One panel per statistic, lettered (a)-(d) in the artwork. Rows are the non-Zipfian laws. Each row has two dodged
marks, one per level of the run's repetition factor. x is the AUC for Zipf against that law with its bootstrap 95%
CI; a hairline at 0.5 marks "cannot tell them apart", and below it the statistic ranks the laws backwards.

Design (dataviz skill): panels by statistic rather than four coloured series, so only two hues are ever on screen,
the reference palette's first two categorical slots, validated all-pairs in light mode with validate_palette.js.
Text uses ink tokens, never series colours; the legend is always shown; gridlines are solid hairlines. A static
manuscript figure, so it has no hover layer.
"""
# isort: off
import resources  # noqa: F401  -- caps BLAS threads; must precede numpy
# isort: on

import argparse
import json
import os

import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SURFACE = "#fcfcfb"      # reference palette, chart chrome & ink, light
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")        # categorical slots 1 and 2, reference palette, light mode
LAWS = ("uniform", "lognormal", "geometric")
STATS = (("observed_mean_r2", "Mean R²"), ("shuffle_z", "Shuffle z"),
         ("rotate_z", "Rotate z"), ("bigram_z", "Bigram z"))
FACTOR_WORDS = {"repeats": "mean repeats", "stickiness": "stickiness"}


def parse(summary, policy):
    cells = summary["policies"][policy]["auc_zipf_vs"]
    levels, label = set(), None
    for key in cells:
        law, level = key.split("|")
        label, value = level.split("=")
        levels.add(float(value))
    levels = sorted(levels)
    if len(levels) not in (1, 2):
        raise SystemExit(f"expected one or two levels of the repetition factor, found {levels}")
    data = {}
    for stat, _ in STATS:
        data[stat] = {}
        for law in LAWS:
            data[stat][law] = {}
            for value in levels:
                cell = cells[f"{law}|{label}={value:g}"][stat]
                data[stat][law][f"{value:g}"] = {"auc": cell["auc"], "ci95": cell["ci95"]}
    return label, levels, data


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", default="whale")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--policy", default="skip")
    ap.add_argument("--dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()

    stem = f"c4_{a.regime}_{a.tag}" if a.tag else f"c4_{a.regime}"
    src = os.path.join(a.dir, f"{stem}_summary.json")
    with open(src, encoding="utf-8") as fh:
        summary = json.load(fh)
    label, levels, data = parse(summary, a.policy)
    words = FACTOR_WORDS.get(label, label)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.9), sharey=True, facecolor=SURFACE)
    y_of = {law: i for i, law in enumerate(reversed(LAWS))}
    dodge = (0.14, -0.14) if len(levels) == 2 else (0.0,)

    for k, (ax, (stat, title)) in enumerate(zip(axes, STATS)):
        ax.set_facecolor(SURFACE)
        for x in (0.0, 0.25, 0.75, 1.0):
            ax.axvline(x, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0.5, color=INK_2, lw=0.8, zorder=1)
        for j, value in enumerate(levels):
            for law in LAWS:
                cell = data[stat][law][f"{value:g}"]
                y = y_of[law] + dodge[j]
                ax.plot(cell["ci95"], [y, y], color=SERIES[j], lw=1.6, solid_capstyle="round",
                        zorder=2)
                ax.plot([cell["auc"]], [y], marker="o", ms=6.5, color=SERIES[j], mec=SURFACE,
                        mew=1.4, zorder=3)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(-0.6, len(LAWS) - 0.4)
        ax.set_yticks(range(len(LAWS)))
        ax.set_yticklabels(list(reversed(LAWS)))
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.set_title(f"({chr(ord('a') + k)})  {title}", loc="left", fontsize=9.5, color=INK,
                     pad=6)
        ax.set_xlabel("AUC, Zipf vs law")
    axes[0].text(0.5, len(LAWS) - 0.47, "cannot separate", ha="center", va="bottom", fontsize=7.5,
                 color=INK_2)

    # dot-only keys: a line through a ringed marker renders as a broken, dashed-looking stroke
    handles = [Line2D([0], [0], lw=0, marker="o", ms=6.5, color=SERIES[j], mec=SURFACE, mew=1.4,
                      label=f"{words} {value:g}") for j, value in enumerate(levels)]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.995, 1.0), labelcolor=INK, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    os.makedirs(a.out, exist_ok=True)
    name = f"fig_{stem}_{a.policy}"
    png = os.path.join(a.out, f"{name}.png")
    fig.savefig(png, dpi=300, facecolor=SURFACE)
    cache = {"source": os.path.relpath(src, ROOT), "policy": a.policy, "factor": label,
             "levels": [f"{v:g}" for v in levels], "series_colours": SERIES, "panels": data}
    with open(os.path.join(a.out, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)
    print(f"wrote {png} and its .json cache")


if __name__ == "__main__":
    main()
