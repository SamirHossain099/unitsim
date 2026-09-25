"""Song-structured generation must add theme order and repetition without losing control of the
frequency law -- otherwise a C4 result on song-structured data would confound structure with law."""
import numpy as np
import pytest

from unitsim.generate import Song, build_lexicon, compose_song, generate_song, generate_stream


def lexicon(seed=0, n_words=24, law="zipf"):
    return build_lexicon(n_words, 12, rng=np.random.default_rng(seed), law=law)


def song_for(lex, n_themes, seed):
    return compose_song(lex, n_themes, rng=np.random.default_rng(seed))


@pytest.mark.parametrize("stickiness", [0.0, 0.8])
def test_phrase_frequencies_follow_the_lexicon(stickiness):
    lex = lexicon(1)
    st = generate_song(lex, song_for(lex, 8, 20), 200_000, rng=np.random.default_rng(2),
                       phrases_per_cycle=40, stickiness=stickiness, random_start=False)
    share = st.word_counts() / st.tokens.size
    assert np.max(np.abs(share - lex.probs)) < 0.01


def test_themes_are_sung_in_the_songs_fixed_cyclic_order():
    lex = lexicon(3)
    song = song_for(lex, 6, 21)
    # large cycles make a skipped theme vanishingly unlikely, so every cycle holds every theme
    st = generate_song(lex, song, 30_000, rng=np.random.default_rng(4), phrases_per_cycle=400,
                       random_start=False)
    runs = st.themes[np.r_[True, st.themes[1:] != st.themes[:-1]]]
    assert np.array_equal(runs[:6], song.order)
    assert np.array_equal(runs, np.tile(song.order, len(runs) // 6 + 1)[:len(runs)])


def test_recordings_of_one_song_share_its_themes_and_order():
    lex = lexicon(5)
    song = song_for(lex, 7, 22)
    rng = np.random.default_rng(6)
    theme_of_word = {}
    for _ in range(4):
        st = generate_song(lex, song, 3_000, rng=rng, phrases_per_cycle=40)
        for w, t in zip(st.tokens.tolist(), st.themes.tolist()):
            assert theme_of_word.setdefault(w, t) == t
    assert np.array_equal(np.sort(np.concatenate(song.themes)), np.arange(lex.n_words))


def test_stickiness_creates_runs_of_one_phrase_type():
    lex = lexicon(8, law="uniform")
    song = song_for(lex, 4, 23)
    loose = generate_song(lex, song, 20_000, rng=np.random.default_rng(9), phrases_per_cycle=40)
    sticky = generate_song(lex, song, 20_000, rng=np.random.default_rng(9), phrases_per_cycle=40,
                           stickiness=0.9)
    same = lambda s: np.mean(np.diff(s.tokens) == 0)   # noqa: E731
    assert same(sticky) > same(loose) + 0.3


def test_ground_truth_is_internally_consistent():
    lex = lexicon(10)
    st = generate_song(lex, song_for(lex, 5, 24), 3_000, rng=np.random.default_rng(11),
                       phrases_per_cycle=30)
    assert st.tokens.size == st.themes.size == st.starts.sum() == 3_000
    rebuilt = np.concatenate([np.array(lex.words[t]) for t in st.tokens])
    assert np.array_equal(rebuilt, st.elements)


def test_random_start_can_begin_mid_cycle():
    lex = lexicon(12)
    song = song_for(lex, 6, 25)
    at_cycle_start = []
    for seed in range(20):
        st = generate_song(lex, song, 5_000, rng=np.random.default_rng(seed), phrases_per_cycle=400)
        at_cycle_start.append(bool(np.all(st.themes[:5] == song.order[0])))
    assert not all(at_cycle_start)


def test_the_iid_generator_is_deterministic_and_themeless():
    lex = lexicon(13)
    a = generate_stream(lex, 400, rng=np.random.default_rng(14))
    b = generate_stream(lex, 400, rng=np.random.default_rng(14))
    assert np.array_equal(a.elements, b.elements) and a.themes is None


@pytest.mark.parametrize("n_themes", [0, 25, 4.5, True])
def test_compose_rejects_bad_theme_counts(n_themes):
    with pytest.raises(ValueError):
        compose_song(lexicon(15), n_themes, rng=np.random.default_rng(16))


@pytest.mark.parametrize("kwargs", [{"phrases_per_cycle": 0}, {"phrases_per_cycle": float("nan")},
                                    {"phrases_per_cycle": 10, "stickiness": 1.0}])
def test_generate_rejects_bad_parameters(kwargs):
    lex = lexicon(17)
    with pytest.raises(ValueError):
        generate_song(lex, song_for(lex, 4, 26), 100, rng=np.random.default_rng(18), **kwargs)


def test_a_song_composed_for_another_lexicon_is_refused():
    small, big = lexicon(19, n_words=12), lexicon(20, n_words=24)
    with pytest.raises(ValueError, match="cover every word"):
        generate_song(big, song_for(small, 4, 27), 100, rng=np.random.default_rng(21),
                      phrases_per_cycle=40)
    bad_order = Song(themes=song_for(big, 4, 28).themes, order=np.array([0, 0, 1, 2]))
    with pytest.raises(ValueError, match="permutation"):
        generate_song(big, bad_order, 100, rng=np.random.default_rng(22), phrases_per_cycle=40)
