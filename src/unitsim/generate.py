"""Synthetic communication systems with known ground truth.

A system has three levels, following how the humpback literature describes song
(Arnon et al. 2025, Science 387:649):

    element  the smallest annotated sound type ("sound element")
    word     a fixed sequence of elements -- the unit a segmentation pipeline tries to recover
    stream   words emitted one after another, optionally repeated in runs, the way a humpback
             repeats a phrase to build a theme

A real recording hides every property below. Here each is a parameter, so a pipeline's output can be
scored against the truth:

    law           how word frequencies are distributed: zipf, uniform, lognormal, geometric
    brevity       whether frequent words are short (Zipf's law of brevity) or length is unrelated
    n_elements    repertoire size; with the lexicon size it sets how many words share each element,
                  and therefore how deep the transitional-probability dips at word boundaries are
    mean_repeats  how long runs of the same word are
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FREQUENCY_LAWS = ("zipf", "uniform", "lognormal", "geometric")


def word_probabilities(n_words, law, *, exponent=1.0, sigma=1.0, ratio=0.8, rng=None):
    """Probability of each word, most frequent first.

    zipf       p(r) proportional to r**-exponent; exponent=1 is Zipf's law.
    uniform    every word equally likely; no frequency structure.
    lognormal  weights drawn from a lognormal and sorted. Heavy-tailed and close to straight on a
               log-log plot, but not a power law -- the standard confound for log-log fits
               (Clauset, Shalizi & Newman 2009). Needs `rng`.
    geometric  p(r) proportional to ratio**(r-1); light-tailed.
    """
    if isinstance(n_words, bool) or not isinstance(n_words, (int, np.integer)) or n_words < 1:
        raise ValueError(f"n_words must be a positive integer, got {n_words!r}")
    ranks = np.arange(1, n_words + 1, dtype=float)
    if law == "zipf":
        if not exponent > 0:  # also rejects NaN
            raise ValueError(f"zipf exponent must be > 0, got {exponent!r}")
        w = ranks ** (-float(exponent))
    elif law == "uniform":
        w = np.ones(n_words)
    elif law == "lognormal":
        if rng is None:
            raise ValueError("lognormal weights are random; pass rng")
        if not sigma > 0:
            raise ValueError(f"lognormal sigma must be > 0, got {sigma!r}")
        w = np.sort(rng.lognormal(mean=0.0, sigma=float(sigma), size=n_words))[::-1]
    elif law == "geometric":
        if not 0 < ratio < 1:
            raise ValueError(f"geometric ratio must be in (0, 1), got {ratio!r}")
        w = float(ratio) ** (ranks - 1)
    else:
        raise ValueError(f"unknown law {law!r}; expected one of {FREQUENCY_LAWS}")
    p = w / w.sum()
    if not np.all(np.isfinite(p)):
        raise FloatingPointError(f"non-finite word probabilities for law={law!r}")
    return p


def _n_possible_words(n_elements, lo, hi, allow_immediate_repeat):
    total = 0
    for length in range(lo, hi + 1):
        if allow_immediate_repeat:
            total += n_elements ** length
        else:
            total += n_elements * (n_elements - 1) ** (length - 1)
    return total


@dataclass(frozen=True)
class Lexicon:
    """A fixed set of distinct words and how often each is used. Index 0 is the most frequent."""

    words: tuple
    probs: np.ndarray
    n_elements: int
    law: str
    brevity: bool

    @property
    def n_words(self):
        return len(self.words)

    @property
    def lengths(self):
        return np.array([len(w) for w in self.words], dtype=np.int64)


def build_lexicon(n_words, n_elements, *, rng, law="zipf", length_range=(2, 5), brevity=False,
                  allow_immediate_repeat=True, **law_params):
    """Draw `n_words` distinct words over an `n_elements` repertoire.

    Word i receives probability probs[i]. With brevity=True the shortest words take the most
    frequent slots. With brevity=False the assignment is a random permutation, so length carries no
    information about frequency -- the null a pipeline must not turn into a brevity law.

    `length_range` is the support of word lengths. Lengths are drawn uniformly and duplicates are
    rejected, so when short words are nearly exhausted the realised lengths skew long.
    """
    lo, hi = (int(v) for v in length_range)
    if not 1 <= lo <= hi:
        raise ValueError(f"length_range must satisfy 1 <= lo <= hi, got {length_range!r}")
    if n_elements < 1:
        raise ValueError(f"n_elements must be >= 1, got {n_elements!r}")
    if not allow_immediate_repeat and hi > 1 and n_elements < 2:
        raise ValueError("avoiding immediate repeats needs at least 2 elements")
    possible = _n_possible_words(n_elements, lo, hi, allow_immediate_repeat)
    if n_words > possible:
        raise ValueError(
            f"{n_words} distinct words requested but only {possible} exist with "
            f"{n_elements} elements and lengths {lo}..{hi}")

    probs = word_probabilities(n_words, law, rng=rng, **law_params)

    seen, words = set(), []
    max_attempts = 50 * n_words + 1000
    attempts = 0
    while len(words) < n_words:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not draw {n_words} distinct words from {possible} possible in "
                f"{max_attempts} attempts; enlarge n_elements or length_range")
        length = int(rng.integers(lo, hi + 1))
        if allow_immediate_repeat:
            w = tuple(int(e) for e in rng.integers(0, n_elements, size=length))
        else:
            seq = [int(rng.integers(0, n_elements))]
            for _ in range(length - 1):
                nxt = int(rng.integers(0, n_elements - 1))
                seq.append(nxt + (nxt >= seq[-1]))
            w = tuple(seq)
        if w not in seen:
            seen.add(w)
            words.append(w)

    if brevity:
        # tie-break noise < 1 keeps the length ordering strict while shuffling equal lengths
        keys = np.array([len(w) for w in words], dtype=float) + 0.5 * rng.random(n_words)
        order = np.argsort(keys, kind="stable")
    else:
        order = rng.permutation(n_words)
    words = tuple(words[i] for i in order)

    return Lexicon(words=words, probs=probs, n_elements=int(n_elements), law=law,
                   brevity=bool(brevity))


@dataclass(frozen=True)
class Stream:
    """A generated element sequence with its ground-truth segmentation."""

    elements: np.ndarray  # (n,) element id at each position
    starts: np.ndarray    # (n,) True where a word token begins; starts[0] is True
    tokens: np.ndarray    # (n_tokens,) word id of each token, in order
    lexicon: Lexicon
    themes: np.ndarray | None = None   # (n_tokens,) theme of each token; song-structured streams only

    def word_counts(self):
        """Occurrences of each word; index matches lexicon.words."""
        return np.bincount(self.tokens, minlength=self.lexicon.n_words)


def _assemble(lexicon, tokens, themes=None):
    word_arrays = [np.asarray(w, dtype=np.int64) for w in lexicon.words]
    elements = np.concatenate([word_arrays[t] for t in tokens])
    starts = np.zeros(elements.size, dtype=bool)
    starts[np.concatenate(([0], np.cumsum(lexicon.lengths[tokens])[:-1]))] = True
    return Stream(elements=elements, starts=starts, tokens=tokens, lexicon=lexicon, themes=themes)


def generate_stream(lexicon, n_tokens, *, rng, mean_repeats=1.0):
    """Emit `n_tokens` word tokens.

    Word types are drawn independently from lexicon.probs. Each draw is emitted a geometric number
    of times with mean `mean_repeats`: 1 gives no forced repetition, larger values give runs of the
    same word. Run length is independent of word type, so lexicon.probs remains each word's expected
    share of tokens.
    """
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, (int, np.integer)) or n_tokens < 1:
        raise ValueError(f"n_tokens must be a positive integer, got {n_tokens!r}")
    if not (np.isfinite(mean_repeats) and mean_repeats >= 1):
        raise ValueError(f"mean_repeats must be a finite number >= 1, got {mean_repeats!r}")

    tokens = np.empty(n_tokens, dtype=np.int64)
    filled = 0
    while filled < n_tokens:
        batch = int((n_tokens - filled) / mean_repeats) + 16
        types = rng.choice(lexicon.n_words, size=batch, p=lexicon.probs)
        if mean_repeats > 1:
            runs = rng.geometric(1.0 / mean_repeats, size=batch)
            types = np.repeat(types, runs)
        take = min(types.size, n_tokens - filled)
        tokens[filled:filled + take] = types[:take]
        filled += take

    return _assemble(lexicon, tokens)


@dataclass(frozen=True)
class Song:
    """One song's structure: the phrase types each theme uses, and the order the themes are sung in."""

    themes: tuple        # one sorted array of word ids per theme
    order: np.ndarray    # theme indices in singing order

    @property
    def n_themes(self):
        return len(self.themes)


def compose_song(lexicon, n_themes, *, rng):
    """Split the lexicon's words at random into `n_themes` themes of near-equal size and fix a random
    singing order.

    Compose once per song -- per year, for humpbacks, whose singers share one song -- and generate every
    recording of that song from the same Song. Composing per recording would give each recording a
    different song and weaken the structure the pipeline pools across recordings.
    """
    if isinstance(n_themes, bool) or not isinstance(n_themes, (int, np.integer)):
        raise ValueError(f"n_themes must be an integer, got {n_themes!r}")
    if not 1 <= n_themes <= lexicon.n_words:
        raise ValueError(f"n_themes must be in 1..{lexicon.n_words}, got {n_themes}")
    parts = np.array_split(rng.permutation(lexicon.n_words), n_themes)
    return Song(themes=tuple(np.sort(p) for p in parts), order=rng.permutation(n_themes))


def generate_song(lexicon, song, n_tokens, *, rng, phrases_per_cycle, stickiness=0.0,
                  random_start=True):
    """Emit `n_tokens` phrase tokens with song structure: phrases repeated within themes, themes sung in a
    fixed order, the cycle repeated -- as Arnon et al. (2025) describe humpback song.

    Each word of `lexicon` is a phrase type; `song` (from `compose_song`) says which theme each belongs to
    and the order themes are sung in. In every cycle theme t is sung Poisson(phrases_per_cycle * P_t)
    times, P_t being the total probability of its phrase types; a theme drawn zero times is skipped that
    cycle. Each rendition picks a phrase type of its theme in proportion to lexicon.probs, or with
    probability `stickiness` repeats the previous rendition's type.

    Every rendition's marginal type is therefore the within-theme categorical, and the expected count of
    each phrase type is proportional to lexicon.probs -- exactly as in `generate_stream`. The frequency law
    stays under control while the sequence gains theme order and repetition.

    random_start drops a uniform number of tokens from the front of the first cycle, as a recording starts
    mid-song. The returned Stream carries each token's theme in `themes`.
    """
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, (int, np.integer)) or n_tokens < 1:
        raise ValueError(f"n_tokens must be a positive integer, got {n_tokens!r}")
    if not (np.isfinite(phrases_per_cycle) and phrases_per_cycle > 0):
        raise ValueError(f"phrases_per_cycle must be positive and finite, got {phrases_per_cycle!r}")
    if not 0 <= stickiness < 1:
        raise ValueError(f"stickiness must be in [0, 1), got {stickiness!r}")
    covered = np.sort(np.concatenate(song.themes)) if song.themes else np.empty(0, dtype=int)
    if not np.array_equal(covered, np.arange(lexicon.n_words)):
        raise ValueError("song themes must cover every word of the lexicon exactly once")
    if sorted(np.asarray(song.order).tolist()) != list(range(song.n_themes)):
        raise ValueError("song order must be a permutation of its theme indices")
    themes, order = song.themes, np.asarray(song.order)
    theme_p = np.array([lexicon.probs[t].sum() for t in themes])
    within = [lexicon.probs[t] / p if p > 0 else np.full(t.size, 1.0 / t.size)
              for t, p in zip(themes, theme_p)]

    def one_cycle():
        tok, th = [], []
        for t in order:
            n = int(rng.poisson(phrases_per_cycle * theme_p[t]))
            if n == 0:
                continue
            draws = rng.choice(themes[t], size=n, p=within[t])
            if stickiness > 0 and n > 1:
                stay = rng.random(n) < stickiness
                stay[0] = False
                source = np.where(stay, 0, np.arange(n))
                draws = draws[np.maximum.accumulate(source)]   # copy the latest fresh draw
            tok.append(draws)
            th.append(np.full(n, t))
        if not tok:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(tok).astype(np.int64), np.concatenate(th).astype(np.int64)

    tokens, theme_ids, total = [], [], 0
    max_cycles = 10 * n_tokens + 1000
    for c in range(max_cycles):
        tok, th = one_cycle()
        if c == 0 and random_start and tok.size:
            drop = int(rng.integers(0, tok.size))
            tok, th = tok[drop:], th[drop:]
        tokens.append(tok)
        theme_ids.append(th)
        total += tok.size
        if total >= n_tokens:
            break
    else:
        raise RuntimeError(f"no {n_tokens} tokens after {max_cycles} cycles; raise phrases_per_cycle")
    tokens = np.concatenate(tokens)[:n_tokens]
    theme_ids = np.concatenate(theme_ids)[:n_tokens]
    return _assemble(lexicon, tokens, themes=theme_ids)
