"""Synthetic keystrokes and the two quantities of a window (speed and rest), from key-press times only.

A keystroke is its press time in ms and nothing else: no key, no character, no release (decision 3 of
.steering/20261031-keystroke-design). The texts decide only where the presses fall, never what is recorded.

Two generators (decision 4 of the same file), compared on equal terms:
- from_text: one press per character of a text (a line break is one press). The gap before a press is drawn from a
  lognormal whose median depends on where the press falls: inside a word, at the start of a word, after the end of a
  sentence, after the end of a paragraph.
- regimes: no text. The typing switches between a fast and a slow regime (geometric run lengths), and rests of two
  seconds or more are put in at random with a fixed chance per press.

Two series made from a series of presses (decision 5):
- shuffled: the same gaps in a random order (the distribution of gaps is kept, their order in time is broken).

Window quantities (after GSE src-tauri/src/analysis/features.rs, F1 and F5, read on 2026-10-31):
- speed: the median of the gaps shorter than the rest threshold whose two presses lie in the window (GSE F1 drops
  gaps of 2000 ms or more as outliers);
- rest: the share of the window's time inside gaps of the rest threshold or more (the part of each such gap that lies
  in the window). GSE F5 counts these gaps; the share is used here so that a rest across a window edge is split
  between the two windows instead of counted twice or not at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# times are sums of float gaps, so a gap drawn as exactly rest_ms may come back a hair short of it
TOL_MS = 1e-6
SENTENCE_END = re.compile(r"[.!?。！？]$")


@dataclass(frozen=True)
class Lognormal:
    median_ms: float
    sigma: float  # standard deviation of the log

    def draw(self, rng: np.random.Generator, n: int | None = None):
        return self.median_ms * np.exp(self.sigma * rng.standard_normal(n))


@dataclass(frozen=True)
class TextTiming:
    within_word: Lognormal
    word_start: Lognormal
    sentence_end: Lognormal
    paragraph_end: Lognormal


def positions(text: str) -> list[str]:
    """Where each press of an English text falls, one label per press.

    Paragraphs are separated by blank lines and a paragraph's lines are joined by one space. Presses: every character
    of a paragraph (spaces included) and one Enter between paragraphs. The label is the kind of gap before the press:
    'start' (the first press, no gap), 'word_start' (the first character of a word), 'sentence_end' (the first character
    of a word after a word that ends a sentence), 'paragraph_end' (the first character of a paragraph after the first)
    and 'within_word' (every other press, including the space after a word and the Enter)."""
    paras = [p.split() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    out: list[str] = []
    for pi, words in enumerate(paras):
        if pi > 0:
            out.append("within_word")  # the Enter
        for wi, word in enumerate(words):
            if wi > 0:
                out.append("within_word")  # the space
            for ci in range(len(word)):
                if ci > 0:
                    out.append("within_word")
                elif not out:
                    out.append("start")
                elif wi == 0:
                    out.append("paragraph_end")
                elif SENTENCE_END.search(words[wi - 1]):
                    out.append("sentence_end")
                else:
                    out.append("word_start")
    return out


def from_text(text: str, timing: TextTiming, rng: np.random.Generator) -> np.ndarray:
    """Press times (ms, the first at 0) of the text typed with the timing."""
    labels = positions(text)
    gaps = np.empty(len(labels) - 1)
    for k, lab in enumerate(labels[1:]):
        gaps[k] = getattr(timing, lab).draw(rng)
    return np.concatenate([[0.0], np.cumsum(gaps)])


@dataclass(frozen=True)
class Regimes:
    fast: Lognormal
    slow: Lognormal
    mean_run_fast: float  # mean number of presses before switching, geometric
    mean_run_slow: float
    rest: Lognormal  # drawn and then floored at the rest threshold
    rest_chance: float  # chance per gap that the gap is a rest


def from_regimes(n_presses: int, g: Regimes, rest_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Press times (ms, the first at 0) of n_presses without a text."""
    gaps = np.empty(n_presses - 1)
    fast = bool(rng.integers(2))
    left = rng.geometric(1.0 / (g.mean_run_fast if fast else g.mean_run_slow))
    for k in range(n_presses - 1):
        if left == 0:
            fast = not fast
            left = rng.geometric(1.0 / (g.mean_run_fast if fast else g.mean_run_slow))
        left -= 1
        if rng.random() < g.rest_chance:
            gaps[k] = max(rest_ms, float(g.rest.draw(rng)))
        else:
            gaps[k] = float((g.fast if fast else g.slow).draw(rng))
    return np.concatenate([[0.0], np.cumsum(gaps)])


def shuffled(times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same gaps in a random order, starting at 0."""
    gaps = np.diff(times)
    return np.concatenate([[0.0], np.cumsum(rng.permutation(gaps))])


@dataclass(frozen=True)
class Window:
    start_ms: float
    presses: int
    speed_ms: float | None  # None when no gap shorter than the rest threshold lies in the window
    rest_share: float


def windows(times: np.ndarray, width_ms: float, rest_ms: float) -> list[Window]:
    """Consecutive windows [k w, (k + 1) w) from 0 to the last press (the last window may be cut short: its rest share
    is over its own length). A gap belongs to the speed of a window when both its presses lie in it."""
    t = np.asarray(times, dtype=float)
    if t.size < 2 or np.any(np.diff(t) < 0) or t[0] != 0.0:
        raise ValueError("times must start at 0, rise and hold at least two presses")
    end = float(t[-1])
    n = int(np.ceil(end / width_ms)) if end > 0 else 1
    out = []
    gaps = np.diff(t)
    is_rest = gaps >= rest_ms - TOL_MS
    for k in range(n):
        a, b = k * width_ms, min((k + 1) * width_ms, end)
        inside = (t >= a) & ((t < b) if k < n - 1 else (t <= b))
        both = inside[:-1] & inside[1:]
        short = gaps[both & ~is_rest]
        lo, hi = t[:-1][is_rest], t[1:][is_rest]
        rest = float(np.sum(np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)))
        out.append(Window(a, int(inside.sum()), float(np.median(short)) if short.size else None, rest / (b - a)))
    return out


def unit(x: float, lo: float, hi: float) -> float:
    """x placed on [0, 1] between lo and hi, saturated outside."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------- v2 (prototype_v2; the functions above stay as v1 used them)


@dataclass(frozen=True)
class Mixture:
    """Lognormals with weights, drawn below or at-or-above a threshold (by rejection)."""
    parts: tuple[tuple[float, Lognormal], ...]

    def draw(self, rng: np.random.Generator, below: float | None = None, at_least: float | None = None) -> float:
        w = np.array([a for a, _ in self.parts], dtype=float)
        if not (w.size and np.all(w >= 0) and w.sum() > 0):
            raise ValueError("a mixture needs non-negative weights with a positive sum")
        w = w / w.sum()
        for _ in range(100_000):
            x = float(self.parts[rng.choice(len(w), p=w)][1].draw(rng))
            if (below is None or x < below) and (at_least is None or x >= at_least):
                return x
        raise RuntimeError("rejection did not find a draw in 100000 tries")


def share_below(d: Lognormal, x: float) -> float:
    """P(gap < x) for a lognormal gap."""
    from math import erf, log, sqrt

    if d.sigma == 0:
        return 1.0 if d.median_ms < x else 0.0
    return 0.5 * (1.0 + erf((log(x) - log(d.median_ms)) / (d.sigma * sqrt(2.0))))


@dataclass(frozen=True)
class MatchedRegimes:
    fast: Mixture
    slow: Mixture
    rest: Mixture
    rest_chance: float
    mean_run_fast: float
    mean_run_slow: float


def matched_regimes(text: str, timing: TextTiming, rest_ms: float, mean_run_fast: float) -> MatchedRegimes:
    """Regimes whose gaps have, in expectation, the distribution of the gaps of from_text(text, timing): the same
    chance of a rest, the same distribution of rest lengths, and the same mixture of shorter gaps. What differs is
    the order: short gaps come in runs of a fast regime (gaps inside words) and a slow one (the other kinds below
    the threshold), with geometric run lengths whose means give the text's share of slow gaps; rests fall at random.
    """
    labels = positions(text)[1:]
    kinds = ("within_word", "word_start", "sentence_end", "paragraph_end")
    n = {k: labels.count(k) for k in kinds}
    below = {k: n[k] * share_below(getattr(timing, k), rest_ms) for k in kinds}
    above = {k: n[k] - below[k] for k in kinds}
    slow_kinds = kinds[1:]
    short = sum(below.values())
    slow_share = sum(below[k] for k in slow_kinds) / short
    if not 0 < slow_share < 1:
        raise ValueError("the text gives no mix of fast and slow gaps")
    return MatchedRegimes(
        # weights are the counts: drawing below or above the threshold by rejection multiplies each part by its
        # chance of falling there, which gives below[k] and above[k]
        fast=Mixture(((float(n["within_word"]), timing.within_word),)),
        slow=Mixture(tuple((float(n[k]), getattr(timing, k)) for k in slow_kinds if n[k] > 0)),
        rest=Mixture(tuple((float(n[k]), getattr(timing, k)) for k in kinds if above[k] > 0)),
        rest_chance=sum(above.values()) / len(labels),
        mean_run_fast=mean_run_fast,
        mean_run_slow=mean_run_fast * slow_share / (1.0 - slow_share),
    )


def from_matched_regimes(n_presses: int, g: MatchedRegimes, rest_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Press times (ms, the first at 0) of n_presses without a text. A gap is a rest with chance rest_chance;
    otherwise it is drawn below the threshold from the current regime, whose run lengths count short gaps only."""
    if n_presses < 2:
        raise ValueError("at least two presses")
    gaps = np.empty(n_presses - 1)
    fast = rng.random() < g.mean_run_fast / (g.mean_run_fast + g.mean_run_slow)  # start in proportion
    left = rng.geometric(1.0 / (g.mean_run_fast if fast else g.mean_run_slow))
    for k in range(n_presses - 1):
        if rng.random() < g.rest_chance:
            gaps[k] = g.rest.draw(rng, at_least=rest_ms)
            continue
        if left == 0:
            fast = not fast
            left = rng.geometric(1.0 / (g.mean_run_fast if fast else g.mean_run_slow))
        left -= 1
        gaps[k] = (g.fast if fast else g.slow).draw(rng, below=rest_ms)
    return np.concatenate([[0.0], np.cumsum(gaps)])


def check_times(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)) or t[0] != 0.0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite, start at 0, rise strictly and hold at least two presses")
    return t


def full_windows(times: np.ndarray, width_ms: float, rest_ms: float) -> tuple[list[Window], float]:
    """The full windows [k w, (k + 1) w), k = 0 .. floor(end / w) - 1, all half-open; the time after the last full
    window is dropped and returned. Speed: median of the gaps < rest_ms with both presses in the window (None if
    none). Rest: share of the window's time inside gaps >= rest_ms."""
    t = check_times(times)
    if not (np.isfinite(width_ms) and width_ms > 0 and np.isfinite(rest_ms) and rest_ms > 0):
        raise ValueError("window width and rest threshold must be positive")
    gaps = np.diff(t)
    is_rest = gaps >= rest_ms - TOL_MS
    lo, hi = t[:-1][is_rest], t[1:][is_rest]
    n = int(np.floor(t[-1] / width_ms))
    out = []
    for k in range(n):
        a, b = k * width_ms, (k + 1) * width_ms
        inside = (t >= a) & (t < b)
        short = gaps[inside[:-1] & inside[1:] & ~is_rest]
        rest = float(np.sum(np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)))
        out.append(Window(a, int(inside.sum()), float(np.median(short)) if short.size else None, rest / (b - a)))
    return out, float(t[-1] - n * width_ms)
