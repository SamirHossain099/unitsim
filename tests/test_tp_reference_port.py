"""`unitsim.tp` must reproduce the published segmentation exactly.

The reference below is a port of three functions from Kirby's public code,
github.com/smkirby/cultural_evolution_of_statistical_structure, file
cultural_evolution_of_statistical_structure.Rmd (read 2026-09-12): `ngram_table` (lines 61-68),
`transition_data` (114-126) and `cut_sequences_sliding` (194-221).

The only changes are to the plumbing: pandas filtering by generation, chain and sentence_number is
replaced by passing a list of strings and a list of TP lists, and the bookkeeping that does not affect
cuts (word_lengths, within_TPs, across_TPs, errors) is dropped. Every arithmetic expression and every
branch is kept character for character, so TPs must match bit for bit, not approximately.
"""
from collections import defaultdict

import numpy as np
import pytest

from unitsim.generate import Lexicon, generate_stream
from unitsim.metrics import boundary_prf
from unitsim.tp import segment, transition_probabilities

# ---- port of Kirby's code -------------------------------------------------------------------------


def kirby_ngram_table(strings, n):
    ngram_counts = defaultdict(int)
    for s in strings:
        for i in range(len(s) - (n - 1)):
            ngram = s[i:i+n]
            ngram_counts[str(ngram)] = ngram_counts[str(ngram)] + 1
    return ngram_counts


def kirby_transition_data(strings, n):
    ngrams = kirby_ngram_table(strings, n)
    ngrams_sum = sum(ngrams.values())
    smaller_grams = kirby_ngram_table(strings, n - 1)
    smaller_grams_sum = sum(smaller_grams.values())
    transitions_list = []
    for string in strings:
        transitions = []
        for i in range(len(string) - (n - 1)):
            transitions.append((ngrams[string[i:i+n]]/ngrams_sum) / (smaller_grams[string[i:i+(n-1)]]/smaller_grams_sum))
        transitions_list.append(transitions)
    return transitions_list


def kirby_cut_sequences_sliding(strings, td, n, cutoff_ratio):
    sentences = []
    for i in range(len(strings)):
        ts = td[i]
        words = []
        word = str(strings[i])[0:(n-1)]
        last_tp = 0
        for j in range(len(ts)):
            if ts[j] < last_tp * cutoff_ratio:
                words.append(word)
                word = ''
            word += str(strings[i])[j+(n-1)]
            last_tp = ts[j]
        words.append(word)
        sentences.append(words)
    return sentences

# ---- helpers ---------------------------------------------------------------------------------------


ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def random_corpus(rng, n_seqs, alphabet_size, max_len):
    ints = [rng.integers(0, alphabet_size, size=int(rng.integers(1, max_len + 1)))
            for _ in range(n_seqs)]
    strings = ["".join(ALPHABET[v] for v in s) for s in ints]
    return ints, strings


def cuts_from_words(words):
    return np.cumsum([len(w) for w in words])[:-1].tolist()


def cuts_from_starts(starts):
    return np.flatnonzero(starts[1:]).__add__(1).tolist()

# ---- tests -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("alphabet_size", [2, 4, 9, 30])
def test_transition_probabilities_are_bit_identical_to_kirby(n, alphabet_size):
    rng = np.random.default_rng(100 * n + alphabet_size)
    for _ in range(25):
        ints, strings = random_corpus(rng, int(rng.integers(1, 8)), alphabet_size, 60)
        ours = transition_probabilities(ints, n=n)
        ref = kirby_transition_data(strings, n)
        assert len(ours) == len(ref)
        for a, b in zip(ours, ref):
            assert np.array_equal(a, np.array(b, dtype=float)), "TPs differ from Kirby's code"


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("threshold", [0.25, 0.3, 0.41, 0.5, 0.7, 0.75, 1.0])
def test_cuts_are_identical_to_kirby(n, threshold):
    rng = np.random.default_rng(int(1000 * threshold) + n)
    for _ in range(40):
        ints, strings = random_corpus(rng, int(rng.integers(1, 8)), int(rng.integers(2, 12)), 60)
        ours = segment(ints, threshold=threshold, n=n)
        ref = kirby_cut_sequences_sliding(strings, kirby_transition_data(strings, n), n, threshold)
        for s, words, st in zip(strings, ref, ours):
            assert "".join(words) == s
            assert cuts_from_starts(st) == cuts_from_words(words)


def test_conditional_denominator_is_the_textbook_tp():
    rng = np.random.default_rng(7)
    ints, _ = random_corpus(rng, 5, 4, 40)
    ours = transition_probabilities(ints, n=2, denominator="conditional")
    followed = defaultdict(int)
    pair = defaultdict(int)
    for s in ints:
        for a, b in zip(s[:-1], s[1:]):
            followed[a] += 1
            pair[(a, b)] += 1
    for s, tps in zip(ints, ours):
        expected = [pair[(a, b)] / followed[a] for a, b in zip(s[:-1], s[1:])]
        assert tps == pytest.approx(expected, abs=1e-15)


def test_backward_mirrors_forward_on_reversed_input():
    """Backward TP of the window (a, b) is P(a | b): forward TP of (b, a) on the reversed corpus."""
    rng = np.random.default_rng(8)
    ints, _ = random_corpus(rng, 6, 5, 50)
    back = transition_probabilities(ints, n=2, direction="backward", denominator="conditional")
    fwd_rev = transition_probabilities([s[::-1] for s in ints], n=2, denominator="conditional")
    for b, f in zip(back, fwd_rev):
        assert b == pytest.approx(f[::-1], abs=1e-15)


def test_first_transition_never_cuts_and_nothing_spans_sequences():
    seqs = [np.array([0, 1, 0, 1, 2]), np.array([2, 2]), np.array([3])]
    for st in segment(seqs, threshold=1e9):   # absurdly high threshold: every eligible TP cuts
        assert st[0]
        if st.size > 1:
            assert not st[1] or st.size == 2 and not st[1]


def test_recovers_saffran_words_perfectly():
    """Every element belongs to exactly one word: within-word TP is ~1, across-word TP is ~1/4."""
    words = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    lex = Lexicon(words=words, probs=np.full(4, 0.25), n_elements=12, law="uniform", brevity=False)
    rng = np.random.default_rng(9)
    streams = [generate_stream(lex, 400, rng=rng) for _ in range(3)]
    predicted = segment([s.elements for s in streams], threshold=0.5)
    for st, p in zip(streams, predicted):
        assert boundary_prf(p, st.starts).f1 == 1.0


@pytest.mark.parametrize("bad", [
    dict(sequences=[np.array([0, 1])], n=1),
    dict(sequences=[np.array([0, 1])], direction="sideways"),
    dict(sequences=[np.array([0, 1])], denominator="nope"),
    dict(sequences=[np.array([0, 1])], threshold=0.0),
    dict(sequences=[np.array([0, 1])], threshold=float("nan")),
    dict(sequences=[np.array([], dtype=int)]),
    dict(sequences=[np.array([0.5, 1.0])]),
    dict(sequences=[]),
])
def test_invalid_input_raises(bad):
    seqs = bad.pop("sequences")
    with pytest.raises(ValueError):
        segment(seqs, **bad)
