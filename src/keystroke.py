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
