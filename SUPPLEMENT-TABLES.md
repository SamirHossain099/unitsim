# Electronic supplementary material: tables S1-S7

Supplement to *Calibrating Zipfian inference from segmented animal vocal sequences with known sources*. Every value is read from the results files in the accompanying code archive by `src/tables_paper.py`, and the archive's tests check that these tables regenerate identically. AUC: the probability that a randomly chosen Zipfian dataset scores higher than a randomly chosen dataset of the alternative law. 0.5: no separation; below 0.5: ranked backwards.

## Table S1

AUC for Zipfian against non-Zipfian sources at the published settings, with bootstrap 95% intervals over 30 datasets per law (main Table 1). Undefined groups skipped.

| regime | alternative | mean R² | shuffled z | rotated z | bigram z |
|---|---|---|---|---|---|
| whale, independent words, m = 1 | uniform | 1.00 [1.00, 1.00] | 0.89 [0.80, 0.96] | 1.00 [0.99, 1.00] | 0.87 [0.76, 0.96] |
| whale, independent words, m = 1 | lognormal | 1.00 [1.00, 1.00] | 0.58 [0.42, 0.72] | 0.59 [0.44, 0.74] | 0.51 [0.35, 0.66] |
| whale, independent words, m = 1 | geometric | 1.00 [1.00, 1.00] | 0.62 [0.48, 0.76] | 0.88 [0.78, 0.95] | 0.77 [0.65, 0.89] |
| whale, independent words, m = 5 | uniform | 1.00 [1.00, 1.00] | 0.30 [0.17, 0.43] | 0.59 [0.45, 0.73] | 0.00 [0.00, 0.00] |
| whale, independent words, m = 5 | lognormal | 0.94 [0.87, 0.99] | 0.32 [0.19, 0.45] | 0.55 [0.41, 0.69] | 0.09 [0.03, 0.16] |
| whale, independent words, m = 5 | geometric | 0.99 [0.96, 1.00] | 0.25 [0.13, 0.37] | 0.46 [0.31, 0.61] | 0.00 [0.00, 0.00] |
| finch | uniform | 0.99 [0.96, 1.00] | 0.49 [0.35, 0.64] | 0.59 [0.45, 0.74] | 0.38 [0.24, 0.52] |
| finch | lognormal | 0.64 [0.49, 0.77] | 0.64 [0.49, 0.78] | 0.57 [0.41, 0.71] | 0.60 [0.45, 0.74] |
| finch | geometric | 0.63 [0.48, 0.77] | 0.70 [0.56, 0.83] | 0.66 [0.52, 0.80] | 0.67 [0.53, 0.81] |
| whale song, s = 0 | uniform | 1.00 [1.00, 1.00] | 0.97 [0.91, 1.00] | 0.96 [0.91, 1.00] | 0.80 [0.68, 0.91] |
| whale song, s = 0 | lognormal | 0.50 [0.36, 0.65] | 0.35 [0.21, 0.50] | 0.43 [0.28, 0.57] | 0.44 [0.29, 0.58] |
| whale song, s = 0 | geometric | 0.61 [0.47, 0.75] | 0.26 [0.14, 0.40] | 0.51 [0.36, 0.66] | 0.46 [0.31, 0.61] |
| whale song, s = 0.8 | uniform | 0.99 [0.96, 1.00] | 0.34 [0.21, 0.48] | 0.39 [0.24, 0.53] | 0.09 [0.02, 0.19] |
| whale song, s = 0.8 | lognormal | 0.28 [0.15, 0.42] | 0.35 [0.22, 0.48] | 0.27 [0.16, 0.41] | 0.38 [0.23, 0.52] |
| whale song, s = 0.8 | geometric | 0.31 [0.17, 0.45] | 0.25 [0.13, 0.38] | 0.19 [0.09, 0.30] | 0.26 [0.14, 0.40] |

## Table S2

Song parameters varied one at a time around the default song (8 themes of 3 phrase variants, 40 renditions per cycle). Whale-sized years, threshold 0.5. AUC with bootstrap 95% interval; slope is the mean rank-frequency slope.

| song | stickiness | alternative | mean R² | shuffled z | rotated z | bigram z | slope |
|---|---|---|---|---|---|---|---|
| 8 themes x 3 variants, 40 renditions | 0 | uniform | 1.00 [1.00, 1.00] | 0.97 [0.91, 1.00] | 0.96 [0.91, 1.00] | 0.80 [0.68, 0.91] | 0.85 [0.74, 0.93] |
| 8 themes x 3 variants, 40 renditions | 0 | lognormal | 0.50 [0.36, 0.65] | 0.35 [0.21, 0.50] | 0.43 [0.28, 0.57] | 0.44 [0.29, 0.58] | 0.79 [0.68, 0.89] |
| 8 themes x 3 variants, 40 renditions | 0 | geometric | 0.61 [0.47, 0.75] | 0.26 [0.14, 0.40] | 0.51 [0.36, 0.66] | 0.46 [0.31, 0.61] | 0.92 [0.85, 0.98] |
| 8 themes x 3 variants, 40 renditions | 0.8 | uniform | 0.99 [0.96, 1.00] | 0.34 [0.21, 0.48] | 0.39 [0.24, 0.53] | 0.09 [0.02, 0.19] | 0.80 [0.68, 0.90] |
| 8 themes x 3 variants, 40 renditions | 0.8 | lognormal | 0.28 [0.15, 0.42] | 0.35 [0.22, 0.48] | 0.27 [0.16, 0.41] | 0.38 [0.23, 0.52] | 0.76 [0.64, 0.88] |
| 8 themes x 3 variants, 40 renditions | 0.8 | geometric | 0.31 [0.17, 0.45] | 0.25 [0.13, 0.38] | 0.19 [0.09, 0.30] | 0.26 [0.14, 0.40] | 0.83 [0.73, 0.93] |
| 4 themes | 0 | uniform | 0.98 [0.94, 1.00] | 0.80 [0.67, 0.90] | 0.90 [0.81, 0.97] | 0.72 [0.58, 0.85] | 0.75 [0.60, 0.87] |
| 4 themes | 0 | lognormal | 0.35 [0.21, 0.49] | 0.31 [0.19, 0.46] | 0.31 [0.18, 0.44] | 0.31 [0.18, 0.44] | 0.66 [0.52, 0.80] |
| 4 themes | 0 | geometric | 0.20 [0.10, 0.32] | 0.22 [0.12, 0.35] | 0.23 [0.11, 0.35] | 0.24 [0.12, 0.38] | 0.75 [0.62, 0.87] |
| 4 themes | 0.8 | uniform | 0.89 [0.79, 0.96] | 0.49 [0.35, 0.64] | 0.53 [0.38, 0.68] | 0.48 [0.33, 0.63] | 0.74 [0.61, 0.85] |
| 4 themes | 0.8 | lognormal | 0.35 [0.21, 0.49] | 0.38 [0.24, 0.53] | 0.27 [0.15, 0.40] | 0.35 [0.22, 0.50] | 0.46 [0.32, 0.62] |
| 4 themes | 0.8 | geometric | 0.26 [0.14, 0.40] | 0.26 [0.14, 0.40] | 0.19 [0.09, 0.31] | 0.34 [0.21, 0.48] | 0.52 [0.38, 0.68] |
| 12 themes | 0 | uniform | 1.00 [1.00, 1.00] | 0.97 [0.93, 1.00] | 0.99 [0.97, 1.00] | 0.88 [0.77, 0.96] | 0.85 [0.75, 0.93] |
| 12 themes | 0 | lognormal | 0.54 [0.40, 0.69] | 0.45 [0.31, 0.60] | 0.47 [0.32, 0.63] | 0.41 [0.27, 0.55] | 0.92 [0.84, 0.98] |
| 12 themes | 0 | geometric | 0.67 [0.53, 0.80] | 0.36 [0.23, 0.50] | 0.44 [0.30, 0.59] | 0.31 [0.18, 0.44] | 0.99 [0.98, 1.00] |
| 12 themes | 0.8 | uniform | 0.98 [0.96, 1.00] | 0.34 [0.21, 0.48] | 0.32 [0.19, 0.46] | 0.00 [0.00, 0.00] | 0.61 [0.46, 0.76] |
| 12 themes | 0.8 | lognormal | 0.45 [0.30, 0.59] | 0.37 [0.24, 0.52] | 0.24 [0.12, 0.36] | 0.29 [0.17, 0.42] | 0.77 [0.65, 0.88] |
| 12 themes | 0.8 | geometric | 0.41 [0.27, 0.56] | 0.26 [0.14, 0.39] | 0.11 [0.04, 0.20] | 0.10 [0.04, 0.19] | 0.87 [0.76, 0.94] |
| 2 variants per theme | 0 | uniform | 1.00 [0.99, 1.00] | 0.86 [0.75, 0.94] | 0.92 [0.84, 0.97] | 0.76 [0.63, 0.88] | 0.88 [0.77, 0.95] |
| 2 variants per theme | 0 | lognormal | 0.30 [0.17, 0.45] | 0.24 [0.13, 0.38] | 0.29 [0.16, 0.43] | 0.34 [0.20, 0.48] | 0.66 [0.51, 0.79] |
| 2 variants per theme | 0 | geometric | 0.36 [0.22, 0.51] | 0.22 [0.11, 0.36] | 0.35 [0.20, 0.49] | 0.44 [0.30, 0.59] | 0.83 [0.71, 0.92] |
| 2 variants per theme | 0.8 | uniform | 0.93 [0.86, 0.98] | 0.52 [0.37, 0.67] | 0.44 [0.29, 0.59] | 0.21 [0.11, 0.34] | 0.89 [0.80, 0.96] |
| 2 variants per theme | 0.8 | lognormal | 0.24 [0.11, 0.37] | 0.36 [0.22, 0.51] | 0.19 [0.09, 0.30] | 0.31 [0.18, 0.45] | 0.64 [0.49, 0.77] |
| 2 variants per theme | 0.8 | geometric | 0.17 [0.07, 0.28] | 0.23 [0.11, 0.35] | 0.10 [0.03, 0.19] | 0.23 [0.11, 0.36] | 0.66 [0.52, 0.79] |
| 5 variants per theme | 0 | uniform | 1.00 [1.00, 1.00] | 0.98 [0.95, 1.00] | 1.00 [0.98, 1.00] | 0.94 [0.88, 0.98] | 0.74 [0.61, 0.85] |
| 5 variants per theme | 0 | lognormal | 0.78 [0.65, 0.89] | 0.42 [0.27, 0.56] | 0.57 [0.42, 0.71] | 0.51 [0.35, 0.65] | 0.91 [0.83, 0.97] |
| 5 variants per theme | 0 | geometric | 0.92 [0.84, 0.98] | 0.48 [0.34, 0.63] | 0.62 [0.48, 0.76] | 0.47 [0.32, 0.62] | 0.98 [0.95, 1.00] |
| 5 variants per theme | 0.8 | uniform | 1.00 [1.00, 1.00] | 0.33 [0.20, 0.47] | 0.21 [0.11, 0.33] | 0.00 [0.00, 0.00] | 0.47 [0.32, 0.62] |
| 5 variants per theme | 0.8 | lognormal | 0.38 [0.24, 0.53] | 0.34 [0.20, 0.49] | 0.27 [0.15, 0.40] | 0.19 [0.10, 0.30] | 0.79 [0.67, 0.89] |
| 5 variants per theme | 0.8 | geometric | 0.36 [0.22, 0.51] | 0.21 [0.10, 0.33] | 0.10 [0.03, 0.19] | 0.06 [0.01, 0.12] | 0.89 [0.80, 0.96] |
| 20 renditions per cycle | 0 | uniform | 1.00 [1.00, 1.00] | 0.99 [0.98, 1.00] | 0.99 [0.97, 1.00] | 0.94 [0.86, 0.99] | 0.80 [0.68, 0.90] |
| 20 renditions per cycle | 0 | lognormal | 0.49 [0.33, 0.64] | 0.44 [0.29, 0.59] | 0.50 [0.34, 0.65] | 0.49 [0.34, 0.63] | 0.88 [0.78, 0.96] |
| 20 renditions per cycle | 0 | geometric | 0.62 [0.48, 0.75] | 0.42 [0.28, 0.57] | 0.63 [0.48, 0.78] | 0.56 [0.41, 0.71] | 0.98 [0.92, 1.00] |
| 20 renditions per cycle | 0.8 | uniform | 0.98 [0.95, 1.00] | 0.64 [0.49, 0.77] | 0.81 [0.69, 0.91] | 0.34 [0.21, 0.48] | 0.94 [0.87, 0.99] |
| 20 renditions per cycle | 0.8 | lognormal | 0.43 [0.28, 0.58] | 0.40 [0.25, 0.55] | 0.48 [0.33, 0.62] | 0.52 [0.37, 0.66] | 0.88 [0.78, 0.95] |
| 20 renditions per cycle | 0.8 | geometric | 0.44 [0.29, 0.58] | 0.24 [0.12, 0.38] | 0.32 [0.19, 0.46] | 0.37 [0.23, 0.51] | 0.90 [0.81, 0.96] |
| 80 renditions per cycle | 0 | uniform | 1.00 [1.00, 1.00] | 0.94 [0.87, 0.98] | 0.99 [0.97, 1.00] | 0.89 [0.79, 0.96] | 0.92 [0.83, 0.98] |
| 80 renditions per cycle | 0 | lognormal | 0.52 [0.37, 0.67] | 0.33 [0.20, 0.47] | 0.42 [0.28, 0.58] | 0.40 [0.26, 0.54] | 0.89 [0.79, 0.96] |
| 80 renditions per cycle | 0 | geometric | 0.64 [0.50, 0.78] | 0.30 [0.18, 0.43] | 0.49 [0.34, 0.64] | 0.50 [0.35, 0.64] | 0.95 [0.90, 0.99] |
| 80 renditions per cycle | 0.8 | uniform | 0.96 [0.91, 0.99] | 0.53 [0.37, 0.68] | 0.42 [0.27, 0.56] | 0.09 [0.03, 0.16] | 0.52 [0.37, 0.65] |
| 80 renditions per cycle | 0.8 | lognormal | 0.40 [0.25, 0.55] | 0.36 [0.23, 0.51] | 0.30 [0.18, 0.45] | 0.42 [0.29, 0.58] | 0.66 [0.52, 0.80] |
| 80 renditions per cycle | 0.8 | geometric | 0.28 [0.16, 0.42] | 0.28 [0.15, 0.41] | 0.12 [0.05, 0.22] | 0.23 [0.12, 0.35] | 0.66 [0.51, 0.79] |

## Table S3

Segmentation threshold. AUC with bootstrap 95% interval at the published thresholds. Datasets are identical across thresholds; only the segmentation changes. Undefined groups skipped.

| regime | level | threshold | alternative | mean R² | shuffled z | rotated z | bigram z |
|---|---|---|---|---|---|---|---|
| whale, independent words | m = 1 | 0.25 | uniform | 1.00 [1.00, 1.00] | 0.99 [0.95, 1.00] | 0.88 [0.78, 0.96] | 0.01 [0.00, 0.03] |
| whale, independent words | m = 1 | 0.25 | lognormal | 1.00 [0.99, 1.00] | 0.99 [0.98, 1.00] | 0.86 [0.75, 0.95] | 0.31 [0.18, 0.46] |
| whale, independent words | m = 1 | 0.25 | geometric | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.96 [0.89, 1.00] | 0.34 [0.21, 0.48] |
| whale, independent words | m = 1 | 0.5 | uniform | 1.00 [1.00, 1.00] | 0.89 [0.80, 0.96] | 1.00 [0.99, 1.00] | 0.87 [0.76, 0.96] |
| whale, independent words | m = 1 | 0.5 | lognormal | 1.00 [1.00, 1.00] | 0.58 [0.42, 0.72] | 0.59 [0.44, 0.74] | 0.51 [0.35, 0.66] |
| whale, independent words | m = 1 | 0.5 | geometric | 1.00 [1.00, 1.00] | 0.62 [0.48, 0.76] | 0.88 [0.78, 0.95] | 0.77 [0.65, 0.89] |
| whale, independent words | m = 1 | 0.75 | uniform | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| whale, independent words | m = 1 | 0.75 | lognormal | 1.00 [1.00, 1.00] | 0.85 [0.75, 0.94] | 0.99 [0.97, 1.00] | 1.00 [1.00, 1.00] |
| whale, independent words | m = 1 | 0.75 | geometric | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| whale, independent words | m = 5 | 0.25 | uniform | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.69 [0.54, 0.81] | 0.03 [0.00, 0.08] |
| whale, independent words | m = 5 | 0.25 | lognormal | 0.97 [0.93, 1.00] | 0.83 [0.72, 0.93] | 0.67 [0.52, 0.80] | 0.27 [0.16, 0.40] |
| whale, independent words | m = 5 | 0.25 | geometric | 0.99 [0.97, 1.00] | 0.99 [0.97, 1.00] | 0.60 [0.45, 0.74] | 0.08 [0.02, 0.16] |
| whale, independent words | m = 5 | 0.5 | uniform | 1.00 [1.00, 1.00] | 0.30 [0.17, 0.43] | 0.59 [0.45, 0.73] | 0.00 [0.00, 0.00] |
| whale, independent words | m = 5 | 0.5 | lognormal | 0.94 [0.87, 0.99] | 0.32 [0.19, 0.45] | 0.55 [0.41, 0.69] | 0.09 [0.03, 0.16] |
| whale, independent words | m = 5 | 0.5 | geometric | 0.99 [0.96, 1.00] | 0.25 [0.13, 0.37] | 0.46 [0.31, 0.61] | 0.00 [0.00, 0.00] |
| whale, independent words | m = 5 | 0.75 | uniform | 1.00 [1.00, 1.00] | 0.77 [0.64, 0.88] | 0.99 [0.97, 1.00] | 0.95 [0.88, 0.99] |
| whale, independent words | m = 5 | 0.75 | lognormal | 0.99 [0.95, 1.00] | 0.50 [0.35, 0.65] | 0.74 [0.61, 0.86] | 0.89 [0.80, 0.96] |
| whale, independent words | m = 5 | 0.75 | geometric | 1.00 [0.99, 1.00] | 0.65 [0.51, 0.79] | 0.87 [0.77, 0.95] | 0.95 [0.89, 1.00] |
| whale song | s = 0 | 0.25 | uniform | 0.96 [0.91, 0.99] | 0.35 [0.22, 0.50] | 0.67 [0.53, 0.81] | 0.04 [0.00, 0.11] |
| whale song | s = 0 | 0.25 | lognormal | 0.61 [0.46, 0.75] | 0.44 [0.30, 0.59] | 0.54 [0.39, 0.68] | 0.35 [0.21, 0.49] |
| whale song | s = 0 | 0.25 | geometric | 0.70 [0.56, 0.82] | 0.29 [0.17, 0.42] | 0.53 [0.38, 0.68] | 0.18 [0.08, 0.30] |
| whale song | s = 0 | 0.5 | uniform | 1.00 [1.00, 1.00] | 0.97 [0.91, 1.00] | 0.96 [0.91, 1.00] | 0.80 [0.68, 0.91] |
| whale song | s = 0 | 0.5 | lognormal | 0.50 [0.36, 0.65] | 0.35 [0.21, 0.50] | 0.43 [0.28, 0.57] | 0.44 [0.29, 0.58] |
| whale song | s = 0 | 0.5 | geometric | 0.61 [0.47, 0.75] | 0.26 [0.14, 0.40] | 0.51 [0.36, 0.66] | 0.46 [0.31, 0.61] |
| whale song | s = 0 | 0.75 | uniform | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| whale song | s = 0 | 0.75 | lognormal | 0.78 [0.65, 0.89] | 0.60 [0.44, 0.74] | 0.58 [0.43, 0.73] | 0.70 [0.57, 0.83] |
| whale song | s = 0 | 0.75 | geometric | 0.97 [0.93, 1.00] | 0.72 [0.58, 0.84] | 0.74 [0.60, 0.86] | 0.89 [0.80, 0.96] |
| whale song | s = 0.8 | 0.25 | uniform | 0.90 [0.81, 0.97] | 0.24 [0.14, 0.36] | 0.18 [0.09, 0.31] | 0.00 [0.00, 0.01] |
| whale song | s = 0.8 | 0.25 | lognormal | 0.50 [0.35, 0.66] | 0.42 [0.28, 0.55] | 0.48 [0.33, 0.63] | 0.31 [0.18, 0.45] |
| whale song | s = 0.8 | 0.25 | geometric | 0.60 [0.45, 0.73] | 0.17 [0.07, 0.28] | 0.28 [0.15, 0.41] | 0.10 [0.03, 0.18] |
| whale song | s = 0.8 | 0.5 | uniform | 0.99 [0.96, 1.00] | 0.34 [0.21, 0.48] | 0.39 [0.24, 0.53] | 0.09 [0.02, 0.19] |
| whale song | s = 0.8 | 0.5 | lognormal | 0.28 [0.15, 0.42] | 0.35 [0.22, 0.48] | 0.27 [0.16, 0.41] | 0.38 [0.23, 0.52] |
| whale song | s = 0.8 | 0.5 | geometric | 0.31 [0.17, 0.45] | 0.25 [0.13, 0.38] | 0.19 [0.09, 0.30] | 0.26 [0.14, 0.40] |
| whale song | s = 0.8 | 0.75 | uniform | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] |
| whale song | s = 0.8 | 0.75 | lognormal | 0.67 [0.52, 0.80] | 0.49 [0.34, 0.65] | 0.35 [0.21, 0.50] | 0.47 [0.32, 0.62] |
| whale song | s = 0.8 | 0.75 | geometric | 0.94 [0.88, 0.99] | 0.56 [0.41, 0.71] | 0.28 [0.16, 0.41] | 0.63 [0.48, 0.76] |
| finch |  | 0.3 | uniform | 0.86 [0.75, 0.95] | 0.80 [0.68, 0.90] | 0.60 [0.44, 0.73] | 0.17 [0.07, 0.29] |
| finch |  | 0.3 | lognormal | 0.53 [0.38, 0.68] | 0.61 [0.46, 0.76] | 0.64 [0.50, 0.78] | 0.43 [0.29, 0.58] |
| finch |  | 0.3 | geometric | 0.65 [0.50, 0.77] | 0.57 [0.41, 0.71] | 0.72 [0.59, 0.84] | 0.48 [0.34, 0.64] |
| finch |  | 0.5 | uniform | 0.99 [0.96, 1.00] | 0.49 [0.35, 0.64] | 0.59 [0.45, 0.74] | 0.38 [0.24, 0.52] |
| finch |  | 0.5 | lognormal | 0.64 [0.49, 0.77] | 0.64 [0.49, 0.78] | 0.57 [0.41, 0.71] | 0.60 [0.45, 0.74] |
| finch |  | 0.5 | geometric | 0.63 [0.48, 0.77] | 0.70 [0.56, 0.83] | 0.66 [0.52, 0.80] | 0.67 [0.53, 0.81] |
| finch |  | 0.7 | uniform | 1.00 [1.00, 1.00] | 0.94 [0.88, 0.99] | 0.96 [0.91, 0.99] | 0.97 [0.92, 1.00] |
| finch |  | 0.7 | lognormal | 0.73 [0.61, 0.85] | 0.69 [0.55, 0.82] | 0.68 [0.54, 0.81] | 0.68 [0.54, 0.81] |
| finch |  | 0.7 | geometric | 0.59 [0.45, 0.73] | 0.67 [0.53, 0.82] | 0.60 [0.46, 0.75] | 0.57 [0.41, 0.71] |

## Table S4

Shape of the alternative law. Each row changes one shape parameter from the defaults (Zipf exponent 1, lognormal sigma 1, geometric tail ratio 0.05); Zipf-exponent rows keep the alternatives at their defaults. AUC with bootstrap 95% interval, threshold 0.5.

| regime | level | change | alternative | mean R² | shuffled z | rotated z | bigram z |
|---|---|---|---|---|---|---|---|
| whale, independent words | m = 1 | lognormal sigma 0.5 | lognormal | 1.00 [1.00, 1.00] | 0.80 [0.68, 0.90] | 0.92 [0.84, 0.98] | 0.75 [0.61, 0.87] |
| whale, independent words | m = 5 | lognormal sigma 0.5 | lognormal | 1.00 [1.00, 1.00] | 0.24 [0.13, 0.37] | 0.48 [0.33, 0.63] | 0.00 [0.00, 0.00] |
| whale, independent words | m = 1 | lognormal sigma 1.5 | lognormal | 0.60 [0.45, 0.74] | 0.38 [0.24, 0.54] | 0.63 [0.48, 0.77] | 0.87 [0.76, 0.95] |
| whale, independent words | m = 5 | lognormal sigma 1.5 | lognormal | 0.55 [0.41, 0.70] | 0.39 [0.26, 0.54] | 0.51 [0.36, 0.65] | 0.50 [0.34, 0.64] |
| whale, independent words | m = 1 | lognormal sigma 2 | lognormal | 0.22 [0.11, 0.35] | 0.65 [0.50, 0.78] | 0.87 [0.76, 0.95] | 0.98 [0.96, 1.00] |
| whale, independent words | m = 5 | lognormal sigma 2 | lognormal | 0.20 [0.10, 0.32] | 0.72 [0.59, 0.84] | 0.57 [0.43, 0.71] | 0.91 [0.84, 0.97] |
| whale, independent words | m = 1 | geometric tail 0.2 | geometric | 1.00 [1.00, 1.00] | 0.77 [0.64, 0.87] | 0.97 [0.92, 1.00] | 0.69 [0.54, 0.82] |
| whale, independent words | m = 5 | geometric tail 0.2 | geometric | 1.00 [0.99, 1.00] | 0.31 [0.18, 0.45] | 0.58 [0.43, 0.71] | 0.01 [0.00, 0.02] |
| whale, independent words | m = 1 | geometric tail 0.01 | geometric | 1.00 [1.00, 1.00] | 0.77 [0.64, 0.88] | 0.88 [0.78, 0.96] | 0.96 [0.91, 1.00] |
| whale, independent words | m = 5 | geometric tail 0.01 | geometric | 0.92 [0.85, 0.98] | 0.24 [0.13, 0.36] | 0.47 [0.32, 0.62] | 0.04 [0.01, 0.10] |
| whale, independent words | m = 1 | Zipf exponent 0.8 | uniform | 1.00 [1.00, 1.00] | 0.90 [0.81, 0.96] | 1.00 [1.00, 1.00] | 0.99 [0.97, 1.00] |
| whale, independent words | m = 1 | Zipf exponent 0.8 | lognormal | 0.94 [0.88, 0.99] | 0.65 [0.50, 0.79] | 0.59 [0.43, 0.73] | 0.80 [0.67, 0.90] |
| whale, independent words | m = 1 | Zipf exponent 0.8 | geometric | 1.00 [1.00, 1.00] | 0.68 [0.55, 0.81] | 0.89 [0.80, 0.96] | 0.93 [0.86, 0.98] |
| whale, independent words | m = 5 | Zipf exponent 0.8 | uniform | 1.00 [1.00, 1.00] | 0.41 [0.26, 0.56] | 0.65 [0.49, 0.78] | 0.02 [0.00, 0.06] |
| whale, independent words | m = 5 | Zipf exponent 0.8 | lognormal | 0.71 [0.58, 0.84] | 0.44 [0.29, 0.58] | 0.60 [0.45, 0.74] | 0.31 [0.19, 0.45] |
| whale, independent words | m = 5 | Zipf exponent 0.8 | geometric | 0.91 [0.84, 0.97] | 0.35 [0.21, 0.50] | 0.49 [0.34, 0.64] | 0.07 [0.01, 0.16] |
| whale, independent words | m = 1 | Zipf exponent 1.2 | uniform | 1.00 [1.00, 1.00] | 0.92 [0.84, 0.98] | 1.00 [1.00, 1.00] | 0.30 [0.17, 0.44] |
| whale, independent words | m = 1 | Zipf exponent 1.2 | lognormal | 1.00 [1.00, 1.00] | 0.68 [0.53, 0.81] | 0.36 [0.22, 0.50] | 0.08 [0.03, 0.15] |
| whale, independent words | m = 1 | Zipf exponent 1.2 | geometric | 1.00 [1.00, 1.00] | 0.71 [0.56, 0.82] | 0.76 [0.64, 0.87] | 0.23 [0.11, 0.36] |
| whale, independent words | m = 5 | Zipf exponent 1.2 | uniform | 1.00 [1.00, 1.00] | 0.20 [0.09, 0.32] | 0.46 [0.31, 0.61] | 0.00 [0.00, 0.00] |
| whale, independent words | m = 5 | Zipf exponent 1.2 | lognormal | 0.98 [0.96, 1.00] | 0.25 [0.13, 0.38] | 0.41 [0.26, 0.56] | 0.00 [0.00, 0.00] |
| whale, independent words | m = 5 | Zipf exponent 1.2 | geometric | 1.00 [0.99, 1.00] | 0.17 [0.07, 0.29] | 0.30 [0.18, 0.44] | 0.00 [0.00, 0.00] |
| finch |  | lognormal sigma 0.5 | lognormal | 0.83 [0.72, 0.92] | 0.50 [0.35, 0.65] | 0.40 [0.26, 0.54] | 0.39 [0.26, 0.54] |
| finch |  | lognormal sigma 1.5 | lognormal | 0.69 [0.55, 0.82] | 0.71 [0.58, 0.83] | 0.81 [0.70, 0.91] | 0.80 [0.68, 0.91] |
| finch |  | lognormal sigma 2 | lognormal | 0.76 [0.63, 0.88] | 0.79 [0.67, 0.89] | 0.93 [0.86, 0.98] | 0.84 [0.72, 0.93] |
| finch |  | geometric tail 0.2 | geometric | 0.81 [0.69, 0.91] | 0.54 [0.39, 0.69] | 0.45 [0.31, 0.60] | 0.37 [0.23, 0.51] |
| finch |  | geometric tail 0.01 | geometric | 0.56 [0.41, 0.72] | 0.83 [0.70, 0.93] | 0.88 [0.78, 0.95] | 0.87 [0.75, 0.95] |
| finch |  | Zipf exponent 0.8 | uniform | 0.95 [0.89, 1.00] | 0.45 [0.31, 0.60] | 0.63 [0.48, 0.76] | 0.41 [0.27, 0.56] |
| finch |  | Zipf exponent 0.8 | lognormal | 0.49 [0.35, 0.63] | 0.62 [0.47, 0.76] | 0.62 [0.46, 0.76] | 0.63 [0.48, 0.76] |
| finch |  | Zipf exponent 0.8 | geometric | 0.49 [0.34, 0.64] | 0.67 [0.53, 0.81] | 0.71 [0.58, 0.83] | 0.70 [0.55, 0.83] |
| finch |  | Zipf exponent 1.2 | uniform | 0.99 [0.97, 1.00] | 0.35 [0.21, 0.49] | 0.52 [0.37, 0.67] | 0.26 [0.13, 0.39] |
| finch |  | Zipf exponent 1.2 | lognormal | 0.72 [0.60, 0.84] | 0.53 [0.39, 0.68] | 0.49 [0.34, 0.65] | 0.46 [0.30, 0.60] |
| finch |  | Zipf exponent 1.2 | geometric | 0.70 [0.56, 0.82] | 0.56 [0.41, 0.70] | 0.60 [0.46, 0.75] | 0.55 [0.40, 0.70] |
| whale song | s = 0 | lognormal sigma 0.5 | lognormal | 1.00 [0.99, 1.00] | 0.80 [0.68, 0.90] | 0.95 [0.89, 0.99] | 0.87 [0.77, 0.95] |
| whale song | s = 0.8 | lognormal sigma 0.5 | lognormal | 0.83 [0.71, 0.92] | 0.36 [0.23, 0.51] | 0.42 [0.28, 0.56] | 0.26 [0.14, 0.39] |
| whale song | s = 0 | lognormal sigma 1.5 | lognormal | 0.06 [0.01, 0.12] | 0.27 [0.15, 0.41] | 0.14 [0.05, 0.24] | 0.13 [0.05, 0.22] |
| whale song | s = 0.8 | lognormal sigma 1.5 | lognormal | 0.11 [0.02, 0.22] | 0.36 [0.23, 0.51] | 0.17 [0.07, 0.28] | 0.34 [0.20, 0.47] |
| whale song | s = 0 | lognormal sigma 2 | lognormal | 0.04 [0.00, 0.10] | 0.40 [0.26, 0.55] | 0.08 [0.03, 0.16] | 0.06 [0.01, 0.12] |
| whale song | s = 0.8 | lognormal sigma 2 | lognormal | 0.11 [0.04, 0.19] | 0.50 [0.35, 0.65] | 0.23 [0.11, 0.35] | 0.43 [0.29, 0.58] |
| whale song | s = 0 | geometric tail 0.2 | geometric | 1.00 [1.00, 1.00] | 0.89 [0.80, 0.96] | 0.95 [0.89, 0.99] | 0.87 [0.78, 0.95] |
| whale song | s = 0.8 | geometric tail 0.2 | geometric | 0.76 [0.63, 0.88] | 0.31 [0.18, 0.44] | 0.26 [0.14, 0.39] | 0.16 [0.06, 0.28] |
| whale song | s = 0 | geometric tail 0.01 | geometric | 0.03 [0.00, 0.07] | 0.08 [0.02, 0.15] | 0.13 [0.05, 0.23] | 0.19 [0.08, 0.30] |
| whale song | s = 0.8 | geometric tail 0.01 | geometric | 0.06 [0.01, 0.13] | 0.22 [0.11, 0.35] | 0.10 [0.03, 0.19] | 0.35 [0.22, 0.50] |
| whale song | s = 0 | Zipf exponent 0.8 | uniform | 1.00 [0.99, 1.00] | 0.92 [0.84, 0.98] | 0.85 [0.74, 0.93] | 0.65 [0.50, 0.78] |
| whale song | s = 0 | Zipf exponent 0.8 | lognormal | 0.16 [0.06, 0.27] | 0.28 [0.15, 0.41] | 0.19 [0.09, 0.30] | 0.22 [0.11, 0.36] |
| whale song | s = 0 | Zipf exponent 0.8 | geometric | 0.21 [0.10, 0.35] | 0.19 [0.08, 0.31] | 0.25 [0.13, 0.38] | 0.23 [0.12, 0.36] |
| whale song | s = 0.8 | Zipf exponent 0.8 | uniform | 0.85 [0.75, 0.94] | 0.38 [0.24, 0.52] | 0.40 [0.26, 0.55] | 0.09 [0.02, 0.19] |
| whale song | s = 0.8 | Zipf exponent 0.8 | lognormal | 0.18 [0.09, 0.30] | 0.39 [0.25, 0.54] | 0.28 [0.16, 0.42] | 0.35 [0.22, 0.50] |
| whale song | s = 0.8 | Zipf exponent 0.8 | geometric | 0.20 [0.10, 0.32] | 0.28 [0.15, 0.41] | 0.21 [0.10, 0.33] | 0.27 [0.14, 0.40] |
| whale song | s = 0 | Zipf exponent 1.2 | uniform | 1.00 [1.00, 1.00] | 0.95 [0.89, 1.00] | 1.00 [0.99, 1.00] | 0.95 [0.88, 0.99] |
| whale song | s = 0 | Zipf exponent 1.2 | lognormal | 0.86 [0.75, 0.94] | 0.32 [0.18, 0.47] | 0.76 [0.63, 0.87] | 0.80 [0.66, 0.90] |
| whale song | s = 0 | Zipf exponent 1.2 | geometric | 0.92 [0.84, 0.98] | 0.22 [0.10, 0.35] | 0.79 [0.66, 0.90] | 0.78 [0.66, 0.89] |
| whale song | s = 0.8 | Zipf exponent 1.2 | uniform | 1.00 [0.99, 1.00] | 0.29 [0.17, 0.43] | 0.45 [0.31, 0.61] | 0.11 [0.03, 0.21] |
| whale song | s = 0.8 | Zipf exponent 1.2 | lognormal | 0.49 [0.34, 0.64] | 0.31 [0.18, 0.45] | 0.32 [0.19, 0.47] | 0.43 [0.29, 0.58] |
| whale song | s = 0.8 | Zipf exponent 1.2 | geometric | 0.51 [0.35, 0.65] | 0.20 [0.09, 0.33] | 0.23 [0.12, 0.36] | 0.32 [0.20, 0.47] |

## Table S5

Maximum-likelihood power-law tests on true word counts and on discovered units. AUC with bootstrap 95% interval; statistics averaged over the years or recordings of each dataset before the AUC. For the exponent, an AUC near 0 means separation in the opposite direction.

| regime | units | alternative | log-log R² | tail fraction | bootstrap p | likelihood ratio | exponent |
|---|---|---|---|---|---|---|---|
| whale, independent words, m = 1 | true | uniform | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| whale, independent words, m = 1 | true | lognormal | 1.00 [1.00, 1.00] | 0.99 [0.97, 1.00] | 0.98 [0.95, 1.00] | 0.97 [0.92, 1.00] | 0.00 [0.00, 0.00] |
| whale, independent words, m = 1 | true | geometric | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| whale, independent words, m = 1 | discovered | uniform | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.31 [0.18, 0.45] | 0.95 [0.86, 1.00] | 0.00 [0.00, 0.00] |
| whale, independent words, m = 1 | discovered | lognormal | 1.00 [1.00, 1.00] | 0.36 [0.24, 0.51] | 0.97 [0.92, 1.00] | 0.00 [0.00, 0.00] | 0.39 [0.25, 0.54] |
| whale, independent words, m = 1 | discovered | geometric | 1.00 [1.00, 1.00] | 0.36 [0.23, 0.50] | 0.99 [0.97, 1.00] | 0.00 [0.00, 0.00] | 0.24 [0.12, 0.36] |
| whale, independent words, m = 5 | true | uniform | 1.00 [1.00, 1.00] | 0.99 [0.96, 1.00] | 0.84 [0.73, 0.94] | 0.97 [0.92, 1.00] | 0.00 [0.00, 0.00] |
| whale, independent words, m = 5 | true | lognormal | 1.00 [1.00, 1.00] | 0.83 [0.71, 0.94] | 0.79 [0.68, 0.89] | 0.91 [0.82, 0.98] | 0.03 [0.00, 0.09] |
| whale, independent words, m = 5 | true | geometric | 1.00 [1.00, 1.00] | 0.89 [0.79, 0.96] | 0.93 [0.86, 0.98] | 1.00 [0.98, 1.00] | 0.00 [0.00, 0.00] |
| whale, independent words, m = 5 | discovered | uniform | 1.00 [1.00, 1.00] | 0.47 [0.33, 0.61] | 0.66 [0.52, 0.79] | 0.02 [0.00, 0.06] | 0.14 [0.06, 0.24] |
| whale, independent words, m = 5 | discovered | lognormal | 0.90 [0.82, 0.97] | 0.36 [0.23, 0.51] | 0.64 [0.50, 0.78] | 0.01 [0.00, 0.04] | 0.33 [0.19, 0.48] |
| whale, independent words, m = 5 | discovered | geometric | 0.97 [0.93, 1.00] | 0.37 [0.25, 0.50] | 0.71 [0.56, 0.83] | 0.00 [0.00, 0.01] | 0.27 [0.14, 0.41] |
| finch | true | uniform | 1.00 [1.00, 1.00] | 0.50 [0.50, 0.50] | 0.98 [0.95, 1.00] | 0.98 [0.94, 1.00] | 0.00 [0.00, 0.00] |
| finch | true | lognormal | 1.00 [1.00, 1.00] | 0.50 [0.50, 0.50] | 0.95 [0.89, 0.99] | 0.96 [0.91, 1.00] | 1.00 [0.98, 1.00] |
| finch | true | geometric | 1.00 [1.00, 1.00] | 0.50 [0.50, 0.50] | 0.97 [0.92, 1.00] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] |
| finch | discovered | uniform | 0.92 [0.86, 0.98] | 0.06 [0.01, 0.12] | 0.91 [0.83, 0.97] | 0.57 [0.42, 0.73] | 0.58 [0.43, 0.72] |
| finch | discovered | lognormal | 0.53 [0.38, 0.67] | 0.35 [0.22, 0.49] | 0.42 [0.27, 0.56] | 0.42 [0.28, 0.56] | 0.70 [0.57, 0.83] |
| finch | discovered | geometric | 0.55 [0.40, 0.69] | 0.33 [0.19, 0.48] | 0.28 [0.15, 0.41] | 0.36 [0.23, 0.50] | 0.65 [0.50, 0.78] |
| whale song, s = 0 | true | uniform | 1.00 [1.00, 1.00] | 0.44 [0.29, 0.60] | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| whale song, s = 0 | true | lognormal | 1.00 [1.00, 1.00] | 0.93 [0.86, 0.98] | 0.97 [0.92, 1.00] | 0.88 [0.77, 0.96] | 0.16 [0.06, 0.27] |
| whale song, s = 0 | true | geometric | 1.00 [1.00, 1.00] | 0.79 [0.67, 0.90] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.35 [0.20, 0.51] |
| whale song, s = 0 | discovered | uniform | 1.00 [1.00, 1.00] | 0.31 [0.18, 0.45] | 0.75 [0.62, 0.87] | 0.71 [0.57, 0.85] | 0.10 [0.02, 0.21] |
| whale song, s = 0 | discovered | lognormal | 0.48 [0.32, 0.63] | 0.36 [0.23, 0.51] | 0.62 [0.47, 0.76] | 0.58 [0.44, 0.73] | 0.67 [0.53, 0.80] |
| whale song, s = 0 | discovered | geometric | 0.48 [0.33, 0.63] | 0.38 [0.24, 0.52] | 0.67 [0.53, 0.80] | 0.60 [0.45, 0.75] | 0.74 [0.60, 0.86] |
| whale song, s = 0.8 | true | uniform | 1.00 [1.00, 1.00] | 0.76 [0.64, 0.87] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 0.00 [0.00, 0.00] |
| whale song, s = 0.8 | true | lognormal | 1.00 [1.00, 1.00] | 0.80 [0.67, 0.90] | 0.97 [0.91, 1.00] | 0.85 [0.74, 0.94] | 0.16 [0.07, 0.27] |
| whale song, s = 0.8 | true | geometric | 1.00 [1.00, 1.00] | 0.68 [0.54, 0.81] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.26 [0.14, 0.38] |
| whale song, s = 0.8 | discovered | uniform | 0.91 [0.83, 0.96] | 0.15 [0.06, 0.26] | 0.73 [0.60, 0.85] | 0.36 [0.22, 0.50] | 0.50 [0.35, 0.65] |
| whale song, s = 0.8 | discovered | lognormal | 0.29 [0.17, 0.42] | 0.30 [0.18, 0.43] | 0.52 [0.37, 0.68] | 0.45 [0.30, 0.60] | 0.68 [0.53, 0.82] |
| whale song, s = 0.8 | discovered | geometric | 0.30 [0.17, 0.44] | 0.38 [0.23, 0.52] | 0.48 [0.33, 0.63] | 0.38 [0.25, 0.54] | 0.66 [0.52, 0.79] |

## Table S6

Calibration validation (paper, section 3.7). Held-out simulated datasets of known law, scored against reference densities fitted on an independent half. The wide prior draws held-out datasets from ranges the reference never covered. Chance accuracy is 0.25.

| regime | prior | confidence threshold | held-out datasets | accuracy | Brier | share decided | accuracy when decided |
|---|---|---|---|---|---|---|---|
| whale, independent words | matched | 0.5 | 4000 | 0.617 | 0.495 | 0.673 | 0.701 |
| whale, independent words | matched | 0.7 | 4000 | 0.617 | 0.495 | 0.295 | 0.834 |
| whale, independent words | matched | 0.9 | 4000 | 0.617 | 0.495 | 0.11 | 0.923 |
| whale, independent words | wide | 0.5 | 2000 | 0.424 | 0.874 | 0.799 | 0.443 |
| whale, independent words | wide | 0.7 | 2000 | 0.424 | 0.874 | 0.566 | 0.472 |
| whale, independent words | wide | 0.9 | 2000 | 0.424 | 0.874 | 0.386 | 0.468 |
| finch | matched | 0.5 | 4000 | 0.457 | 0.653 | 0.23 | 0.684 |
| finch | matched | 0.7 | 4000 | 0.457 | 0.653 | 0.049 | 0.796 |
| finch | matched | 0.9 | 4000 | 0.457 | 0.653 | 0.014 | 0.821 |
| finch | wide | 0.5 | 2000 | 0.388 | 0.88 | 0.627 | 0.419 |
| finch | wide | 0.7 | 2000 | 0.388 | 0.88 | 0.426 | 0.434 |
| finch | wide | 0.9 | 2000 | 0.388 | 0.88 | 0.311 | 0.468 |
| whale song | matched | 0.5 | 4000 | 0.619 | 0.495 | 0.57 | 0.748 |
| whale song | matched | 0.7 | 4000 | 0.619 | 0.495 | 0.251 | 0.917 |
| whale song | matched | 0.9 | 4000 | 0.619 | 0.495 | 0.127 | 0.972 |
| whale song | wide | 0.5 | 2000 | 0.441 | 0.841 | 0.83 | 0.474 |
| whale song | wide | 0.7 | 2000 | 0.441 | 0.841 | 0.606 | 0.506 |
| whale song | wide | 0.9 | 2000 | 0.441 | 0.841 | 0.412 | 0.554 |

## Table S7

Calibration, matched prior, confidence threshold 0.7: discrimination for each law (one-vs-rest AUC over the posterior, and recall), beside the accuracy of the full four-statistic vector and of each statistic on its own.

| regime | law | AUC | recall | accuracy, all four statistics | accuracy, observed_mean_r2 only | accuracy, observed_mean_slope only | accuracy, observed_mean_unit_types only | accuracy, cuts_per_sequence only |
|---|---|---|---|---|---|---|---|---|
| whale, independent words | geometric | 0.784 | 0.558 | 0.617 | 0.555 | 0.433 | 0.367 | 0.333 |
| whale, independent words | lognormal | 0.739 | 0.259 | 0.617 | 0.555 | 0.433 | 0.367 | 0.333 |
| whale, independent words | uniform | 0.963 | 0.865 | 0.617 | 0.555 | 0.433 | 0.367 | 0.333 |
| whale, independent words | zipf | 0.895 | 0.785 | 0.617 | 0.555 | 0.433 | 0.367 | 0.333 |
| finch | geometric | 0.65 | 0.371 | 0.457 | 0.317 | 0.337 | 0.341 | 0.265 |
| finch | lognormal | 0.662 | 0.228 | 0.457 | 0.317 | 0.337 | 0.341 | 0.265 |
| finch | uniform | 0.822 | 0.579 | 0.457 | 0.317 | 0.337 | 0.341 | 0.265 |
| finch | zipf | 0.744 | 0.651 | 0.457 | 0.317 | 0.337 | 0.341 | 0.265 |
| whale song | geometric | 0.798 | 0.574 | 0.619 | 0.438 | 0.346 | 0.296 | 0.395 |
| whale song | lognormal | 0.75 | 0.342 | 0.619 | 0.438 | 0.346 | 0.296 | 0.395 |
| whale song | uniform | 0.968 | 0.885 | 0.619 | 0.438 | 0.346 | 0.296 | 0.395 |
| whale song | zipf | 0.844 | 0.674 | 0.619 | 0.438 | 0.346 | 0.296 | 0.395 |
