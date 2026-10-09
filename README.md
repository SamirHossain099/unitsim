# unitsim

Ground-truth simulation for unit discovery in animal vocal sequences, and the analysis code for the paper
*Calibrating Zipfian inference from segmented animal vocal sequences with known sources*.

Claims that animal song units follow Zipf's law are made about units a pipeline recovered, not units anyone
observed. Recent work segments humpback whale and Bengalese finch song at dips in transitional probability,
fits a rank-frequency law to the resulting units, and supports the fit with shuffled, rotated and
bigram-preserving baselines. Whether that pipeline can tell a Zipfian source from a source with some other
frequency law cannot be checked on real song, whose lexicon is unknown. It can be checked in simulation.

`unitsim` generates communication systems whose lexicon, frequency law and song structure are set by design,
runs the published pipeline over them, and scores what comes back.

## What is here

| Path | Contents |
|---|---|
| `src/unitsim/` | The library: `generate`, `corrupt`, `tp` (the published segmentation), `baselines` (shuffled, rotated, bigram), `metrics`, `powerlaw_fit`, `calibrate` |
| `src/exp_*.py` | The experiments, one file per question, each writing a CSV and a JSON of run metadata to `results/` |
| `src/analyze_*.py` | Summaries: discrimination with bootstrap intervals, verdict rates, calibration validation |
| `src/figures_paper.py`, `src/tables_paper.py` | The paper's figures and supplementary tables, with every plotted number cached beside the image |
| `run_all.py` | The driver. Sequential by default, `--workers N` to run steps in parallel, `--match` to select them |
| `tests/` | Unit tests, a line-for-line reference port of the published segmentation, a packaging guard, and tests that pin every number in the paper to a file in `results/` |
| `results/`, `figures/`, `tables/` | The outputs the paper reports |

## Install

```
pip install -e ".[analysis,dev]"
```

Python 3.10 or newer. The library itself needs only numpy and scipy; pandas and matplotlib are used by the
experiments and figures. Everything runs on CPU.

## Reproduce

```
python run_all.py --dry-run              # list the steps
python run_all.py --workers 6            # about a day on 20 cores; results/ is overwritten
python -m pytest -q                      # every number in the paper, checked against results/
```

Individual pieces:

```
python src/exp_c4_inference_unit.py --regime whale --undefined skip --datasets 30
python src/exp_calibration.py --regime whale --n-per-law 1000
python src/analyze_calibration.py --regime whale
python src/figures_paper.py
```

## The segmentation is a reference port

`unitsim.tp` reproduces the cut rule of the published pipeline, and `tests/test_tp_reference_port.py`
requires bit-identical transitional probabilities and identical cuts against a port of the authors' public
code, for bigram and trigram contexts at seven thresholds. If that test fails, nothing downstream is valid.

## Calibration

`unitsim.calibrate` compares the statistics a real analysis reports (mean log-log R^2, mean rank-frequency
slope, mean unit types per group, mean cuts per sequence) with simulated references under each candidate law,
drawn over a prior on what real song does not reveal. It returns a probability for each law and abstains when
none is probable enough. Its validation, including what happens when the prior misses the truth, is in
`src/analyze_calibration.py` and in the paper.

Before reading a posterior on a real corpus, check that the reference set contains it:
`unitsim.calibrate.typicality` gives, for each law, the share of that law's reference datasets (leave-one-out
density) that are no more typical than the observation, and `contains` applies a threshold. A posterior is a
ratio of densities, so without this check it names a law even for a corpus no law could have produced.

## Application to the published humpback corpus

The humpback sequences are Data S2 of Arnon et al. (2025, *Science* 387:649), in the paper's supplementary
materials; they are not redistributed here. Save `science.adq7055_data_s2.csv` to `refs/arnon2025_data/`, then:

```
python src/real_whale_features.py                                     # statistics of the real corpus
python src/exp_real_whale.py --prior song_labels --n-per-law 1000 --workers 8
python src/analyze_real_whale.py                                      # validation, containment, posterior
```

`exp_real_whale.py` takes `--prior song`, `song_labels` or `broad_labels`. The results files are included, so
the tests check the reported numbers without the data.

## Citing

See `CITATION.cff`. Archived on Zenodo: concept DOI [10.5281/zenodo.22960677](https://doi.org/10.5281/zenodo.22960677), which always resolves to the latest version.

MIT licence.
