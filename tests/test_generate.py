"""The generator is the ground truth every result is scored against, so it gets checked first."""
import numpy as np
import pytest
from scipy.stats import spearmanr

from unitsim.generate import FREQUENCY_LAWS, build_lexicon, generate_stream, word_probabilities


@pytest.mark.parametrize("law", FREQUENCY_LAWS)
def test_probabilities_form_a_descending_distribution(law):
    p = word_probabilities(40, law, rng=np.random.default_rng(0))
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p >= 0)
    assert np.all(np.diff(p) <= 1e-15)


def test_zipf_probabilities_have_exactly_the_requested_slope():
    p = word_probabilities(100, "zipf", exponent=1.3)
    slope = np.polyfit(np.log(np.arange(1, 101)), np.log(p), 1)[0]
    assert slope == pytest.approx(-1.3, abs=1e-10)


def test_uniform_is_flat():
    assert np.ptp(word_probabilities(30, "uniform")) == 0


@pytest.mark.parametrize("law,kwargs", [
    ("nope", {}),
    ("zipf", {"exponent": 0}),
    ("zipf", {"exponent": float("nan")}),
    ("geometric", {"ratio": 1.0}),
    ("lognormal", {}),                      # no rng
    ("lognormal", {"sigma": -1.0, "rng": np.random.default_rng(0)}),
])
def test_bad_parameters_raise(law, kwargs):
    with pytest.raises(ValueError):
        word_probabilities(10, law, **kwargs)


def test_lexicon_words_are_distinct_and_in_range():
    lex = build_lexicon(60, 12, rng=np.random.default_rng(1), length_range=(2, 5))
    assert len(set(lex.words)) == 60
    assert all(2 <= len(w) <= 5 for w in lex.words)
    assert all(0 <= e < 12 for w in lex.words for e in w)


def test_no_immediate_repeats_when_disallowed():
    lex = build_lexicon(80, 6, rng=np.random.default_rng(2), allow_immediate_repeat=False)
    assert all(a != b for w in lex.words for a, b in zip(w, w[1:]))


def test_brevity_puts_short_words_in_the_frequent_slots():
    lex = build_lexicon(50, 10, rng=np.random.default_rng(3), brevity=True)
    assert np.all(np.diff(lex.lengths) >= 0)


def test_without_brevity_length_is_unrelated_to_rank():
    rho = [spearmanr(np.arange(50), build_lexicon(50, 10, rng=np.random.default_rng(s)).lengths)[0]
           for s in range(200)]
    assert abs(np.mean(rho)) < 0.03  # ~3 standard errors


def test_impossible_lexicon_raises_before_trying():
    # 2 elements, lengths 2..3: only 4 + 8 = 12 distinct words exist
    with pytest.raises(ValueError, match="only 12 exist"):
        build_lexicon(100, 2, rng=np.random.default_rng(0), length_range=(2, 3))


def test_stream_ground_truth_is_internally_consistent():
    rng = np.random.default_rng(4)
    lex = build_lexicon(30, 10, rng=rng)
    st = generate_stream(lex, 500, rng=rng)
    assert st.starts[0]
    assert st.starts.sum() == 500 == st.tokens.size
    rebuilt = np.concatenate([np.array(lex.words[t]) for t in st.tokens])
    assert np.array_equal(rebuilt, st.elements)
    for i, t in zip(np.flatnonzero(st.starts)[:50], st.tokens[:50]):
        assert tuple(st.elements[i:i + len(lex.words[t])].tolist()) == lex.words[t]


def test_stream_frequencies_converge_to_the_lexicon_even_with_repeats():
    rng = np.random.default_rng(5)
    lex = build_lexicon(20, 10, rng=rng, law="zipf", exponent=1.0)
    st = generate_stream(lex, 200_000, rng=rng, mean_repeats=3.0)
    share = st.word_counts() / st.tokens.size
    assert np.max(np.abs(share - lex.probs)) < 0.01


def test_repeats_create_runs_of_the_same_word():
    rng = np.random.default_rng(6)
    lex = build_lexicon(40, 10, rng=rng, law="uniform")
    same_1 = np.mean(np.diff(generate_stream(lex, 20_000, rng=rng).tokens) == 0)
    same_5 = np.mean(np.diff(generate_stream(lex, 20_000, rng=rng, mean_repeats=5.0).tokens) == 0)
    assert same_1 < 0.06   # 1/40 expected
    assert same_5 > 0.7    # ~0.8 expected


@pytest.mark.parametrize("kwargs", [{"n_tokens": 0}, {"n_tokens": 10, "mean_repeats": 0.5},
                                    {"n_tokens": 10, "mean_repeats": float("nan")}])
def test_stream_bad_parameters_raise(kwargs):
    lex = build_lexicon(10, 5, rng=np.random.default_rng(7))
    n = kwargs.pop("n_tokens")
    with pytest.raises(ValueError):
        generate_stream(lex, n, rng=np.random.default_rng(8), **kwargs)
