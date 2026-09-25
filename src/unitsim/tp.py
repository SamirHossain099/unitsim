"""Transitional-probability segmentation, reproducing the published pipeline.

The method (Arnon & Kirby 2024; Arnon et al. 2025; Kirby et al. 2026) scans each sequence and cuts
before an element whenever the transitional probability (TP) into it falls below `threshold` times
the TP before it. The first TP of a sequence never cuts.

Defaults reproduce Kirby's public implementation, `transition_data` and `cut_sequences_sliding` in
github.com/smkirby/cultural_evolution_of_statistical_structure. `tests/test_tp_reference_port.py`
requires bit-identical TPs and identical cuts against a port of that code:

    TP_j = (c(s[j:j+n]) / N_n) / (c(key_j) / N_{n-1})

Counts c are pooled over every sequence passed in and never span a sequence boundary. N_k is the total
number of k-gram windows. key_j is the n-1 elements before the last (forward) or after the first
(backward). The (n-1)-gram table counts every window, including ones at a sequence end with no
successor. `denominator="conditional"` divides instead by occurrences that do have a neighbour on the
predicted side, the textbook TP. The two differ only through sequence-edge positions.

A cut from TP_j places a unit boundary before element j + n - 1.

Everything is vectorised with numpy: the baselines rerun this pipeline thousands of times.
"""
import numpy as np

DIRECTIONS = ("forward", "backward")
DENOMINATORS = ("kirby", "conditional")


def _flatten(sequences):
    seqs = [np.asarray(s) for s in sequences]
    if not seqs:
        raise ValueError("no sequences given")
    for i, s in enumerate(seqs):
        if s.ndim != 1 or s.size == 0:
            raise ValueError(f"sequence {i} must be a non-empty 1-D array")
        if not np.issubdtype(s.dtype, np.integer):
            raise ValueError(f"sequence {i} must hold integer element labels, got {s.dtype}")
        if s.min() < 0:
            raise ValueError(f"sequence {i} has negative labels")
    lengths = np.array([s.size for s in seqs], dtype=np.int64)
    x = np.concatenate(seqs).astype(np.int64)
    seq_id = np.repeat(np.arange(len(seqs)), lengths)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return x, seq_id, lengths, offsets


def _windows(x, seq_id, k, base):
    """Integer code and start index of every length-k window inside a single sequence."""
    m = x.size - k + 1
    if m <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.arange(m)
    starts = starts[seq_id[starts] == seq_id[starts + k - 1]]
    code = np.zeros(starts.size, dtype=np.int64)
    for i in range(k):
        code = code * base + x[starts + i]
    return code, starts


def _lookup(table_codes, table_counts, keys):
    idx = np.searchsorted(table_codes, keys)
    if idx.size and (idx.max() >= table_codes.size or np.any(table_codes[idx] != keys)):
        raise AssertionError("an n-gram key is missing from its (n-1)-gram table")
    return table_counts[idx]


def _check_options(n, direction, denominator):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if denominator not in DENOMINATORS:
        raise ValueError(f"denominator must be one of {DENOMINATORS}, got {denominator!r}")


def _tp_flat(x, seq_id, n, direction, denominator):
    base = int(x.max()) + 1
    if base ** n >= 2 ** 62:
        raise OverflowError(f"{base} labels with n={n} overflow 64-bit n-gram codes")
    ng_code, ws = _windows(x, seq_id, n, base)
    if ng_code.size == 0:
        return np.empty(0), ws
    ng_u, ng_inv, ng_cnt = np.unique(ng_code, return_inverse=True, return_counts=True)
    ng_count = ng_cnt[ng_inv]
    shift = base ** (n - 1)
    keys = ng_code // base if direction == "forward" else ng_code % shift
    if denominator == "kirby":
        sub_code, _ = _windows(x, seq_id, n - 1, base)
        sub_u, sub_cnt = np.unique(sub_code, return_counts=True)
        tp = (ng_count / ng_code.size) / (_lookup(sub_u, sub_cnt, keys) / sub_code.size)
    else:
        key_u, key_inv, key_cnt = np.unique(keys, return_inverse=True, return_counts=True)
        tp = ng_count / key_cnt[key_inv]
    return tp, ws


def transition_probabilities(sequences, *, n=2, direction="forward", denominator="kirby"):
    """TP for every n-element window, counts pooled over `sequences`.

    Returns one float array per sequence, of length max(len - (n - 1), 0).

    forward   probability of a window's last element given the n-1 before it.
    backward  probability of a window's first element given the n-1 after it. Arnon et al. (2025)
              report a backward pipeline, "the probability of each sound element given the following
              one"; its code is not public, so this is the direct mirror of the forward rule.
    """
    _check_options(n, direction, denominator)
    x, seq_id, lengths, _ = _flatten(sequences)
    tp, _ = _tp_flat(x, seq_id, n, direction, denominator)
    per_seq = np.maximum(lengths - (n - 1), 0)
    return np.split(tp, np.cumsum(per_seq)[:-1])


def segment(sequences, *, threshold=0.5, n=2, direction="forward", denominator="kirby"):
    """Unit starts for each sequence by the published sliding-ratio rule.

    Cut before element j + n - 1 when TP_j < threshold * TP_{j-1}, with TP_{-1} = 0 at the start of
    every sequence so the first TP never cuts. Position 0 always starts a unit. Returns a list of
    boolean arrays aligned with `sequences`.
    """
    if not (np.isfinite(threshold) and threshold > 0):
        raise ValueError(f"threshold must be a positive finite number, got {threshold!r}")
    _check_options(n, direction, denominator)
    x, seq_id, lengths, offsets = _flatten(sequences)
    tp, ws = _tp_flat(x, seq_id, n, direction, denominator)
    starts = np.zeros(x.size, dtype=bool)
    starts[offsets] = True
    if tp.size:
        prev = np.empty_like(tp)
        prev[0] = 0.0
        prev[1:] = tp[:-1]
        first_of_seq = np.ones(tp.size, dtype=bool)
        first_of_seq[1:] = seq_id[ws[1:]] != seq_id[ws[:-1]]
        prev[first_of_seq] = 0.0
        starts[ws[tp < prev * threshold] + n - 1] = True
    return [starts[o:o + ln] for o, ln in zip(offsets, lengths)]
