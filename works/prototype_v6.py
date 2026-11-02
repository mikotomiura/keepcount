"""Prototype v6: the audit of the compression on human keystrokes (decision 3 of .steering/20261031-keystroke-concept;
the choices in decisions 1-10 of .steering/20261031-compression-audit-v6).

Usage:  uv run python works/prototype_v6.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6.json (committed before this script is first run), and v4's conditions for
        the mapping and the relief
Input:  KUPA-KEYS TASK-1 and TASK-2 (external/kupa-keys), the Aalto 136M zip (external/aalto-136m; read, not
        extracted); none of them in git, all checked by SHA-256
Output: results/prototype_v6/record.json and figures (written to a temporary directory beside it and renamed when
        complete). No press time, gap, key, text, per-person vector or person id is written.

Steps: counted presses per person (KUPA: per task; Aalto: per presented sentence) -> 200 eligible persons per dataset
drawn with a fixed seed -> tiles per sentence and per 10 s window (speed and rest -> p, q with v4's external scale ->
v4's tile on field 42 -> depth and fineness of v5) -> per person and part a vector per stage (S0 quantiles of ln gap,
S1 median and IQR of p and q, S2 median and IQR of depth and fineness) -> count 2 (identification across tasks or
parts) and count 1 (the operations on a series against the person's own halves and another person).
"""

from __future__ import annotations

import atexit
import csv
import datetime as dt
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
atexit.register(shutil.rmtree, sys.pycache_prefix, ignore_errors=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

import _common as cm  # noqa: E402
import keystroke as ks  # noqa: E402
import prototype_v4 as v4  # noqa: E402
import prototype_v5 as v5  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6.json"
OUT_DIR = ROOT / "results" / "prototype_v6"
CODE = ["works/prototype_v6.py", "works/prototype_v5.py", "works/prototype_v4.py", "works/_common.py",
        "src/keystroke.py", "uv.lock"]
SEED = 20261031
N = 200
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
FIELD = 42
WINDOW_MS = 10000.0
MIN_TILES_COUNT1 = 8
N_BOOT, N_PERM = 2000, 1000
DATASET = {"kupa": 0, "aalto": 1}
TASK = {"copy": 0, "essay": 1, "session": 0}
STAGE = {"S0": 0, "S1": 1, "S2": 2}
UNIT = {"sentence": 0, "window_10s": 1}
TILE_PX = {"sentence": 120, "window_10s": 80}
SENT_END, KUPA_NAMED = ".!?", ("Enter", "Backspace", "Delete")
ROWS = {"kupa": {"sentence": ["text/before", "text/after", "text/none", "inside/before", "shuffled/before",
                              "shuffled_in_halves/before"],
                 "window_10s": ["text", "inside", "shuffled", "shuffled_in_halves"]},
        "aalto": {"sentence": ["text", "shuffled", "shuffled_in_halves"],
                  "window_10s": ["text", "shuffled", "shuffled_in_halves"]}}
S0_TOL = 1e-9  # relative: a shuffled series is rebuilt by a cumulative sum, so its gaps differ in the last bits


# ---------------------------------------------------------------- reading


def kupa_kind(key: str) -> str | None:
    """The kind of a counted KUPA press, or None if the key is not counted (decision 4 of
    .steering/20261031-human-keystroke-datasets)."""
    if key in KUPA_NAMED:
        return "space" if key == "Enter" else "other"
    if len(key) != 1:
        return None
    if key in SENT_END:
        return "end"
    if key.isalnum():
        return "alnum"
    return "space" if key == " " else "other"


def clean(times: list[float], kinds: list[str]) -> tuple[np.ndarray, list[str], int]:
    """Stable sort by time, then drop a press at the same time as the previous one; times from 0."""
    if not all(math.isfinite(x) for x in times):
        raise SystemExit("a press time is not a finite number")
    order = sorted(range(len(times)), key=lambda i: times[i])
    t = np.array([times[i] for i in order], dtype=float)
    k = [kinds[i] for i in order]
    keep = np.concatenate([[True], np.diff(t) > 0]) if t.size else np.array([], dtype=bool)
    dropped = int((~keep).sum())
    t, k = t[keep], [x for x, kk in zip(k, keep) if kk]
    return (t - t[0] if t.size else t), k, dropped


def read_kupa(path: Path) -> tuple[dict[str, dict], set[str]]:
    """Per id: the counted presses (times from 0) and their kinds, and the number dropped for equal times; and every
    id with any event."""
    raw: dict[str, tuple[list[float], list[str]]] = {}
    seen: set[str] = set()
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add(row["id"])
            if row["type"] != "down" or row["is_repeat"] == "True":
                continue
            kind = kupa_kind(row["key"])
            if kind is None:
                continue
            ts, ks_ = raw.setdefault(row["id"], ([], []))
            ts.append(float(row["time"]))
            ks_.append(kind)
    out = {}
    for pid, (ts, kinds) in raw.items():
        t, k, dropped = clean(ts, kinds)
        out[pid] = {"times": t, "kinds": k, "dropped": dropped}
    return out, seen


def kupa_starts(kinds: list[str]) -> list[int]:
    """Press 0 and the first alphanumeric press after a sentence-end key."""
    starts, after_end = [0], False
    for i, k in enumerate(kinds):
        if k == "alnum" and after_end and i > 0:
            starts.append(i)
            after_end = False
        elif k == "end":
            after_end = True
    return starts


def every_sentence_holds_a_gap(s: "Series") -> bool:
    """Under each of the countings before, after and none, every sentence holds at least one gap (decision 11)."""
    return all(hi > lo for c in ("before", "after", "none") for lo, hi in ks.counted_ranges(s.starts, s.times.size, c))


def kupa_word_starts(kinds: list[str]) -> list[int]:
    return [i for i in range(1, len(kinds)) if kinds[i] == "alnum" and kinds[i - 1] == "space"]


def aalto_ids(zf: zipfile.ZipFile) -> dict[int, bool]:
    """Every PARTICIPANT_ID of the metadata, mapped to whether its KEYBOARD_TYPE is 'full' or 'laptop'."""
    text = io.TextIOWrapper(zf.open("Keystrokes/files/metadata_participants.txt"), encoding="utf-8", errors="replace")
    header = text.readline().rstrip("\n").split("\t")
    i_id, i_kb = header.index("PARTICIPANT_ID"), header.index("KEYBOARD_TYPE")
    out = {}
    for line in text:
        f = line.rstrip("\n").split("\t")
        if len(f) > max(i_id, i_kb):
            out[int(f[i_id])] = f[i_kb] in ("full", "laptop")
    return out


def read_aalto(zf: zipfile.ZipFile, pid: int, tally: dict) -> list[np.ndarray] | None:
    """The press times of each presented sentence (counted presses, from the sentence's first press), in the order of
    first appearance; None if the file is missing or the person is not eligible. Fields are read from both ends of a
    line, so a tab inside SENTENCE or USER_INPUT does not shift them."""
    name = f"Keystrokes/files/{pid}_keystrokes.txt"
    try:
        info = zf.getinfo(name)
    except KeyError:
        tally["missing_files"] += 1
        return None
    sents: dict[str, list[float]] = {}
    digits = 0
    with io.TextIOWrapper(zf.open(info), encoding="utf-8", errors="replace") as f:
        for line in f:
            x = line.rstrip("\r\n").split("\t")
            if x[0] == "PARTICIPANT_ID" or len(x) < 9:
                if x[0] != "PARTICIPANT_ID":
                    tally["short_lines"] += 1
                continue
            if len(x) != 9:
                tally["lines_read_from_ends"] += 1
            letter = x[-2]
            presses = sents.setdefault(x[1], [])  # every section seen, counted presses or not
            if not (len(letter) == 1 or letter == "BKSP"):
                tally["not_counted"] += 1
                continue
            if letter.isdigit():
                digits += 1
            presses.append(float(x[-4]))
    if len(sents) != 15:
        return None
    out = []
    for ts in sents.values():
        t, _, dropped = clean(ts, ["c"] * len(ts))
        tally["dropped_equal_times"] += dropped
        if t.size < 2:
            return None
        out.append(t)
    for part in (out[:7], out[7:]):  # each part needs a full 10 s window (decision 11)
        if int(np.floor(aalto_series(part).times[-1] / WINDOW_MS)) < 1:
            tally["part_without_window"] += 1
            return None
    tally["digit_letters_of_eligible"] += digits
    return out


# ---------------------------------------------------------------- series and tiles


class Series:
    """A person's series in one task: press times from 0, sentence gap ranges per counting, word starts (KUPA)."""

    def __init__(self, times: np.ndarray, starts: list[int] | None = None, word_starts: list[int] | None = None,
                 sentence_ranges: list[tuple[int, int]] | None = None):
        self.times = ks.check_times(times)
        self.starts = starts
        self.word_starts = word_starts
        self.fixed_ranges = sentence_ranges  # Aalto: the presented sentences

    def ranges(self, counting: str) -> list[tuple[int, int]]:
        if self.fixed_ranges is not None:
            return self.fixed_ranges
        return ks.counted_ranges(self.starts, self.times.size, counting)

    def with_times(self, times: np.ndarray) -> "Series":
        return Series(times, self.starts, self.word_starts, self.fixed_ranges)


def aalto_series(sentences: list[np.ndarray]) -> Series:
    """The within-sentence gaps of the sentences joined in order (the gap between sentences dropped)."""
    gaps, ranges, at = [], [], 0
    for t in sentences:
        g = np.diff(t)
        gaps.append(g)
        ranges.append((at, at + g.size))
        at += g.size
    return Series(np.concatenate([[0.0], np.cumsum(np.concatenate(gaps))]), sentence_ranges=ranges)


class Tiles:
    """The tiles of one series in one unit: p, q, depth, fineness and, per tile, the gap indices it holds (for S0)."""

    def __init__(self, s: Series, unit: str, counting: str, m: dict, r: dict, rest_ms: float):
        gaps = np.diff(s.times)
        if unit == "sentence":
            rg = s.ranges(counting)
            segs = ks.segments_of_gaps(gaps, rg, rest_ms)
            self.held = [np.arange(lo, hi) for lo, hi in rg]
        else:
            segs, _ = ks.full_windows(s.times, WINDOW_MS, rest_ms)
            first = s.times[:-1]
            self.held = [np.flatnonzero((first >= w.start_ms) & (first < w.start_ms + WINDOW_MS)) for w in segs]
            self.window_starts = [w.start_ms for w in segs]
        self.gaps = gaps
        self.n = len(segs)
        if self.n == 0:
            raise ValueError("no tile")
        p, q, no_speed = v4.pq(segs, m, "external")
        self.p, self.q, self.no_speed = p, q, len(no_speed)
        grey = [v4.tile(a, b, r, FIELD, TILE_PX[unit]) for a, b in zip(p, q)]
        self.depth = np.array([v5.depth(g) for g in grey])
        self.fineness = np.array([v5.fineness(g) for g in grey])

    def vectors(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        """Stage vectors of the tiles idx."""
        held = np.concatenate([self.held[i] for i in idx]) if len(idx) else np.array([], dtype=int)
        if held.size == 0:
            raise ValueError("the tiles hold no gap")
        lg = np.log(self.gaps[held])
        return {"S0": np.quantile(lg, QUANTILES, method="linear"),
                "S1": np.concatenate([med_iqr(self.p[idx]), med_iqr(self.q[idx])]),
                "S2": np.concatenate([med_iqr(self.depth[idx]), med_iqr(self.fineness[idx])])}

    def all(self) -> dict[str, np.ndarray]:
        return self.vectors(np.arange(self.n))


def med_iqr(v: np.ndarray) -> np.ndarray:
    q1, q2, q3 = np.percentile(v, [25, 50, 75], method="linear")
    return np.array([q2, q3 - q1])


def split_index(tiles: Tiles, s: Series, unit: str, h: int) -> int:
    """The gap index that separates half A (tiles < h) from half B, for shuffled_in_halves: sentences, the first gap
    held by tile h (gaps before it are held by A under the counting 'before' or by the presented sentences); windows,
    the first gap whose first press is at or after the start of window h (the gaps before it have their first press in
    A's windows or before them)."""
    if unit == "sentence":
        return int(tiles.held[h][0])
    return int(np.searchsorted(s.times[:-1], tiles.window_starts[h], side="left"))


# ---------------------------------------------------------------- counts


def standardise(g: np.ndarray, e: np.ndarray, how: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Registration and evaluation sets standardised per feature; features with standard deviation 0 in either set
    dropped (their indices returned)."""
    if how == "ranks":
        g = np.column_stack([rankdata(g[:, j]) for j in range(g.shape[1])])
        e = np.column_stack([rankdata(e[:, j]) for j in range(e.shape[1])])
    sg, se = g.std(axis=0), e.std(axis=0)
    drop = [j for j in range(g.shape[1]) if sg[j] == 0 or se[j] == 0]
    keep = [j for j in range(g.shape[1]) if j not in drop]
    g, e = g[:, keep], e[:, keep]
    if how == "registration_scale":
        mu, sd = g.mean(axis=0), g.std(axis=0)
        return (g - mu) / sd, (e - mu) / sd, drop
    return (g - g.mean(axis=0)) / g.std(axis=0), (e - e.mean(axis=0)) / e.std(axis=0), drop


def distances(e: np.ndarray, g: np.ndarray) -> np.ndarray:
    """d[i, j] = Euclidean distance of evaluation i to registration j."""
    return np.sqrt(((e[:, None, :] - g[None, :, :]) ** 2).sum(axis=2))


def per_person(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """top1 and rank_auc of each evaluation person (the genuine registration of i is i)."""
    n = d.shape[0]
    gen = np.diag(d)
    off = ~np.eye(n, dtype=bool)
    imp = d[off].reshape(n, n - 1)
    rauc = ((imp > gen[:, None]).sum(axis=1) + 0.5 * (imp == gen[:, None]).sum(axis=1)) / (n - 1)
    mn = imp.min(axis=1)
    tied = (imp == gen[:, None]).sum(axis=1)
    top1 = np.where(gen < mn, 1.0, np.where(gen == mn, 1.0 / (1 + tied), 0.0))
    return top1, rauc


def pooled(d: np.ndarray) -> tuple[float, float]:
    """Pooled ROC-AUC (genuine against all impostors, ties 1/2) and TPR at FAR 0.01."""
    n = d.shape[0]
    gen = np.diag(d)
    imp = d[~np.eye(n, dtype=bool)]
    s = np.sort(imp)
    lo = np.searchsorted(s, gen, side="left")
    hi = np.searchsorted(s, gen, side="right")
    auc = float(np.mean((imp.size - hi + 0.5 * (hi - lo)) / imp.size))
    thr = float(np.quantile(imp, 0.01, method="linear"))
    return auc, float(np.mean(gen <= thr)), thr, float(np.mean(imp <= thr))


def ci(values: np.ndarray, idx: np.ndarray) -> list[float]:
    m = values[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def reading(lo: float, hi: float) -> dict:
    """Two axes (decisions 4 and 8): detection against 0.5 and size against 0.60."""
    return {"detection": "detectable" if lo > 0.5 else "not detected",
            "size": "below 0.60" if hi < 0.60 else ("above 0.60" if lo > 0.60 else "undecided")}


def identify(g: np.ndarray, e: np.ndarray, idx: np.ndarray, perm_rng: np.random.Generator) -> dict:
    d = distances(e, g)
    top1, rauc = per_person(d)
    pa, tpr, thr, far = pooled(d)
    null_top1, null_rauc = [], []
    for _ in range(N_PERM):
        pi = perm_rng.permutation(d.shape[1])
        t, r_ = per_person(d[:, pi])
        null_top1.append(t.mean())
        null_rauc.append(r_.mean())
    c_r = ci(rauc, idx)
    return {"rank_auc": float(rauc.mean()), "rank_auc_ci": c_r, "reading": reading(*c_r),
            "top1": float(top1.mean()), "top1_ci": ci(top1, idx), "pooled_auc": pa, "tpr_at_far_0.01": tpr,
            "far_threshold": thr, "far_realised": far,
            "p_rank_auc": (1 + int(np.sum(np.array(null_rauc) >= rauc.mean()))) / (N_PERM + 1),
            "p_top1": (1 + int(np.sum(np.array(null_top1) >= top1.mean()))) / (N_PERM + 1),
            "_rauc": rauc}


def count2(reg: dict, ev: dict, ds: str, direction: int) -> dict:
    """reg[unit][stage], ev[unit][stage]: (200, k) arrays in the same person order."""
    out: dict = {}
    for unit in UNIT:
        n = reg[unit]["S0"].shape[0]
        idx = {st: np.random.default_rng([SEED, 6, 2, DATASET[ds], direction, STAGE[st], UNIT[unit]])
               .integers(0, n, (N_BOOT, n)) for st in STAGE}
        res: dict = {}
        for how in ("main", "registration_scale", "ranks"):
            res[how] = {}
            for st in STAGE:
                g, e, drop = standardise(reg[unit][st], ev[unit][st], how)
                r_ = identify(g, e, idx[st], np.random.default_rng([SEED, 6, 3, DATASET[ds], direction, STAGE[st],
                                                                     UNIT[unit]]))
                r_["dropped_features"] = drop
                res[how][st] = r_
            for a, b in (("S0", "S1"), ("S1", "S2")):
                diff = res[how][a]["_rauc"] - res[how][b]["_rauc"]
                res[how][f"{a}-{b}"] = {"rank_auc_difference": float(diff.mean()), "ci": ci(diff, idx[b])}
            for st in STAGE:
                del res[how][st]["_rauc"]
        out[unit] = res
    return out


def count1(halves: dict, ds: str, task: str, unit: str) -> dict:
    """halves[row] = list over persons of (vectors A, vectors B) per stage."""
    rows = ROWS[ds][unit]
    first = halves[rows[0]]
    n = len(first)
    if n < 2:
        raise SystemExit(f"count 1 {ds} {task} {unit}: fewer than 2 persons")
    out: dict = {"persons": n}
    for st in STAGE:
        t_all = np.array([v[h][st] for v in first for h in (0, 1)])
        sd = t_all.std(axis=0)
        keep = sd > 0
        out.setdefault("dropped_features", {})[st] = [int(j) for j in np.flatnonzero(~keep)]

        def z(x):
            return x[keep] / sd[keep]

        tA = np.array([z(v[0][st]) for v in first])
        tB = np.array([z(v[1][st]) for v in first])
        base = np.linalg.norm(tA - tB, axis=1)
        dAB = distances(tA, tB)
        between = np.array([np.median(np.delete(dAB[i], i)) for i in range(n)])
        for ri, row in enumerate(rows[1:], start=1):
            rA = np.array([z(v[0][st]) for v in halves[row]])
            rB = np.array([z(v[1][st]) for v in halves[row]])
            change = (np.linalg.norm(rA - tA, axis=1) + np.linalg.norm(rB - tB, axis=1)) / 2
            cross = (np.linalg.norm(rA - tB, axis=1) + np.linalg.norm(rB - tA, axis=1)) / 2
            own = np.where(change > base, 1.0, np.where(change == base, 0.5, 0.0))
            other = np.where(change > between, 1.0, np.where(change == between, 0.5, 0.0))
            idx = np.random.default_rng([SEED, 6, 6, DATASET[ds], TASK[task], UNIT[unit], STAGE[st], ri]) \
                .integers(0, n, (N_BOOT, n))
            c_own = ci(own, idx)
            pos = base > 0
            out.setdefault(row, {})[st] = {
                "share_vs_own": float(own.mean()), "share_vs_own_ci": c_own, "reading": reading(*c_own),
                "share_vs_another": float(other.mean()), "share_vs_another_ci": ci(other, idx),
                "median_base": float(np.median(base)), "median_change": float(np.median(change)),
                "median_cross": float(np.median(cross)), "median_between": float(np.median(between)),
                "median_change_over_base": float(np.median(change[pos] / base[pos])) if pos.any() else None,
                "base_zero": int((~pos).sum())}
    return out


# ---------------------------------------------------------------- rows


def make_rows(ds: str, s: Series, unit: str, h_tiles: Tiles, key: list[int], m: dict, r: dict,
              rest_ms: float) -> tuple[dict[str, Tiles], dict]:
    """The tiles of every row of a person in one unit (the first row's tiles given), and the checks."""
    pid_key = key  # [dataset, task, person id]
    out = {ROWS[ds][unit][0]: h_tiles}
    info = {"inside_left": 0}
    h = h_tiles.n // 2
    for row in ROWS[ds][unit][1:]:
        kind, _, counting = row.partition("/")
        counting = counting or "before"
        if kind == "text":
            out[row] = Tiles(s, unit, counting, m, r, rest_ms)
            continue
        if kind == "shuffled":
            t = ks.shuffled(s.times, np.random.default_rng([SEED, 6, 4, *pid_key]))
        elif kind == "inside":
            t, left = ks.rests_inside_at(s.times, s.starts, s.word_starts,
                                         np.random.default_rng([SEED, 6, 5, *pid_key]))
            info["inside_left"] += left
        elif kind == "shuffled_in_halves":
            k = split_index(h_tiles, s, unit, h)
            t = ks.shuffled_in_two(s.times, k, np.random.default_rng([SEED, 6, 7, pid_key[0], pid_key[1], UNIT[unit],
                                                                     pid_key[2]]))
        else:
            raise ValueError(row)
        out[row] = Tiles(s.with_times(t), unit, counting, m, r, rest_ms)
    return out, info


def check_s0(rows: dict[str, Tiles], unit: str) -> None:
    """Implementation check of the conditions: a whole-series shuffle keeps S0 of the whole series; a shuffle in
    halves keeps S0 of each half (sentence unit)."""
    first = next(iter(rows.values()))
    for name, t in rows.items():
        kind = name.split("/")[0]
        if kind == "shuffled" and unit == "sentence":
            if not np.allclose(t.all()["S0"], first.all()["S0"], rtol=S0_TOL, atol=1e-12):
                raise SystemExit("a whole-series shuffle changed S0 of the whole series")
        if kind == "shuffled_in_halves" and unit == "sentence":
            h = first.n // 2
            for idx in (np.arange(h), np.arange(h, first.n)):
                if not np.allclose(t.vectors(idx)["S0"], first.vectors(idx)["S0"], rtol=S0_TOL, atol=1e-12):
                    raise SystemExit("a shuffle in halves changed S0 of a half")


# ---------------------------------------------------------------- figures


def scatter(points: list[tuple[float, float]], size: int = 420, margin: int = 40) -> Image.Image:
    """Median p (x) against median q (y), axes [0, 1], origin lower left; no labels but the axis names."""
    im = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(im)
    span = size - 2 * margin
    d.rectangle([margin, margin, margin + span, margin + span], outline=0)
    for k in range(1, 4):
        x = margin + span * k / 4
        d.line([x, margin + span, x, margin + span + 4], fill=0)
        d.line([margin - 4, margin + span - span * k / 4, margin, margin + span - span * k / 4], fill=0)
    for p, q in points:
        x, y = margin + span * p, margin + span * (1 - q)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=0)
    d.text((margin + span / 2, size - margin / 2 - 6), "p", fill=0)
    d.text((margin / 2 - 4, margin + span / 2), "q", fill=0)
    return im


# ---------------------------------------------------------------- run


def draw_order(ids: list, d: int) -> list:
    return [ids[i] for i in np.random.default_rng([SEED, 6, d]).permutation(len(ids))]


def run(c: dict, m: dict, r: dict, out_dir: Path) -> dict:
    rest_ms = c["units"]["rest_ms"]
    if rest_ms != 2000.0:
        raise SystemExit("rest_ms differs from the conditions read by this script")
    rec: dict = {"datasets": {}}
    figs: dict = {}

    # KUPA
    kp = c["data"]["kupa"]["files"]
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = read_kupa(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    place = {pid: i for i, pid in enumerate(ids_sorted)}
    chosen, looked = [], 0
    series: dict = {}
    for pid in draw_order(ids_sorted, 0):
        looked += 1
        ok, ser = True, {}
        for task in ("copy", "essay"):
            x = kupa[task].get(pid)
            if x is None or x["times"].size < 2:
                ok = False
                break
            st = kupa_starts(x["kinds"])
            s = Series(x["times"], st, kupa_word_starts(x["kinds"]))
            if len(st) < 5 or int(np.floor(s.times[-1] / WINDOW_MS)) < 5 or not every_sentence_holds_a_gap(s):
                ok = False
                break
            ser[task] = s
        if ok:
            chosen.append(pid)
            series[pid] = ser
            if len(chosen) == N:
                break
    if len(chosen) < N:
        raise SystemExit(f"KUPA: only {len(chosen)} eligible ids")
    rec["datasets"]["kupa"] = kupa_rec = {
        "ids_read": {t: len(v) for t, v in seen.items()}, "ids_with_counted_presses": {t: len(v) for t, v in kupa.items()},
        "ids_looked_at": looked,
        "ids_sha256": hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest(),
        "dropped_equal_times": {t: int(sum(kupa[t][p]["dropped"] for p in chosen)) for t in kupa},
        "presses": {t: int(sum(series[p][t].times.size for p in chosen)) for t in kupa}}
    del kupa

    # Aalto
    za = ROOT / c["data"]["aalto"]["file"]
    tally = {"missing_files": 0, "short_lines": 0, "lines_read_from_ends": 0, "not_counted": 0,
             "dropped_equal_times": 0, "part_without_window": 0, "digit_letters_of_eligible": 0, "keyboard_not_eligible": 0}
    with zipfile.ZipFile(za) as zf:
        kb = aalto_ids(zf)
        a_ids = sorted(kb)
        a_chosen, a_looked, a_series = [], 0, {}
        for pid in draw_order(a_ids, 1):
            a_looked += 1
            if not kb[pid]:
                tally["keyboard_not_eligible"] += 1
                continue
            sents = read_aalto(zf, pid, tally)
            if sents is None:
                continue
            a_chosen.append(pid)
            a_series[pid] = sents
            if len(a_chosen) == N:
                break
    if len(a_chosen) < N:
        raise SystemExit(f"Aalto: only {len(a_chosen)} eligible ids")
    rec["datasets"]["aalto"] = aalto_rec = {
        "ids_in_metadata": len(a_ids), "ids_with_keyboard": int(sum(kb.values())), "ids_looked_at": a_looked,
        "ids_sha256": hashlib.sha256("\n".join(str(x) for x in sorted(a_chosen)).encode()).hexdigest(),
        "tally_over_files_looked_at": tally,
        "presses": int(sum(sum(t.size for t in v) for v in a_series.values()))}

    # tiles, rows and halves
    pq_points: dict = {}
    vec2: dict = {}   # (dataset, task) -> unit -> stage -> list over chosen persons (whole part)
    half: dict = {}   # (dataset, task, unit) -> row -> list of (A, B)
    meta: dict = {}
    jobs = [("kupa", task, pid, series[pid][task], place[pid]) for task in ("copy", "essay") for pid in chosen] + \
           [("aalto", "session", pid, aalto_series(a_series[pid]), pid) for pid in a_chosen]
    tile_rows_fig: dict = {}
    for ds, task, pid, s, pnum in jobs:
        key = [DATASET[ds], TASK[task], pnum]
        for unit in UNIT:
            first_row = ROWS[ds][unit][0]
            counting = first_row.partition("/")[2] or "before"
            tt = Tiles(s, unit, counting, m, r, rest_ms)
            mk = meta.setdefault((ds, task, unit), {"tiles": [], "no_speed": 0, "inside_left": 0, "persons_count1": 0,
                                                    "left_out_half_without_gap": 0})
            mk["tiles"].append(tt.n)
            mk["no_speed"] += tt.no_speed
            if ds == "kupa":
                for st, v in tt.all().items():
                    vec2.setdefault((ds, task), {}).setdefault(unit, {}).setdefault(st, []).append(v)
            if unit == "sentence":
                pq_points.setdefault((ds, task), []).append((float(np.median(tt.p)), float(np.median(tt.q))))
                if len(tile_rows_fig.setdefault((ds, task), [])) < 4:
                    tile_rows_fig[(ds, task)].append(
                        [v4.tile(a, b, r, FIELD, TILE_PX[unit]) for a, b in zip(tt.p[:24], tt.q[:24])])
            h = tt.n // 2
            halves_hold = tt.n >= MIN_TILES_COUNT1 and all(
                sum(tt.held[i].size for i in part) > 0 for part in (range(h), range(h, tt.n)))
            if tt.n >= MIN_TILES_COUNT1 and not halves_hold:
                mk["left_out_half_without_gap"] += 1
            if halves_hold:
                rows, info = make_rows(ds, s, unit, tt, key, m, r, rest_ms)
                check_s0(rows, unit)
                mk["inside_left"] += info["inside_left"]
                mk["persons_count1"] += 1
                iA, iB = np.arange(h), np.arange(h, tt.n)
                for row, t_ in rows.items():
                    if t_.n < tt.n:
                        raise SystemExit(f"{ds} {task} {unit} {row}: fewer tiles than the first row")
                    half.setdefault((ds, task, unit), {}).setdefault(row, []).append((t_.vectors(iA), t_.vectors(iB)))
        if ds == "aalto":
            sents = a_series[pid]
            for part, sl in (("A", slice(0, 7)), ("B", slice(7, 15))):
                sp = aalto_series(sents[sl])
                for unit in UNIT:
                    tp = Tiles(sp, unit, "before", m, r, rest_ms)
                    mp = meta.setdefault(("aalto", f"part_{part}", unit), {"tiles": [], "no_speed": 0, "inside_left": 0,
                                                                          "persons_count1": 0,
                                                                          "left_out_half_without_gap": 0})
                    mp["tiles"].append(tp.n)
                    mp["no_speed"] += tp.no_speed
                    for st, v in tp.all().items():
                        vec2.setdefault((ds, part), {}).setdefault(unit, {}).setdefault(st, []).append(v)

    # count 2
    arr = {k: {u: {st: np.array(v) for st, v in uv.items()} for u, uv in d.items()} for k, d in vec2.items()}
    rec["count_2_person"] = {
        "kupa_copy_to_essay": count2(arr[("kupa", "copy")], arr[("kupa", "essay")], "kupa", 0),
        "kupa_essay_to_copy": count2(arr[("kupa", "essay")], arr[("kupa", "copy")], "kupa", 1),
        "aalto_first7_to_last8": count2(arr[("aalto", "A")], arr[("aalto", "B")], "aalto", 0)}

    # count 1
    rec["count_1_operations"] = {f"{ds}_{task}_{unit}": count1(hv, ds, task, unit)
                                 for (ds, task, unit), hv in half.items()}
    rec["tiles"] = {f"{ds}_{task}_{unit}": {"persons": len(v["tiles"]), "tiles_total": int(sum(v["tiles"])),
                                            "tiles_median": float(np.median(v["tiles"])), "no_speed": v["no_speed"],
                                            "persons_count1": v["persons_count1"], "inside_left": v["inside_left"],
                                            "left_out_half_without_gap": v["left_out_half_without_gap"]}
                    for (ds, task, unit), v in meta.items()}
    rec["stage_summary"] = {f"{ds}_{part}_{unit}": {st: {"median": np.median(a, axis=0).tolist(),
                                                         "iqr": (np.percentile(a, 75, axis=0)
                                                                 - np.percentile(a, 25, axis=0)).tolist()}
                                                    for st, a in uv.items()}
                            for (ds, part), d in arr.items() for unit, uv in d.items()}

    # figures
    for (ds, task), rows in tile_rows_fig.items():
        figs[f"{ds}_{task}_tiles"] = v4.figure(rows, TILE_PX["sentence"], 6)
    for (ds, task), pts in pq_points.items():
        figs[f"{ds}_{task}_scatter"] = scatter(pts)
    rec["figures"] = {}
    for name, im in figs.items():
        path = out_dir / f"{name}.png"
        im.save(path)
        rec["figures"][name] = {"file": f"results/prototype_v6/{path.name}", "sha256": cm.sha256(path)}
    return rec


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    cm.check_generator(c["generator"])
    from importlib import metadata
    for pkg in ("numpy", "pillow", "scipy"):
        if metadata.version(pkg) != c[pkg]:
            raise SystemExit(f"{pkg} is {metadata.version(pkg)}, not {c[pkg]}")
    m4 = c["mapping"]["v4_conditions"]
    raw4 = cm.check_file(ROOT / m4["file"], m4["sha256"], "v4's conditions")
    c4 = json.loads(raw4)
    if c4["generator"] != c["generator"]:
        raise SystemExit("the generator of v4's conditions differs from v6's")
    if c4["units"]["rest_ms"] != c["units"]["rest_ms"]:
        raise SystemExit("rest_ms differs from v4's")
    if [c4["units"]["sentence"]["tile_px"], c4["units"]["windows"]["tile_px"]] != [TILE_PX["sentence"],
                                                                                   TILE_PX["window_10s"]]:
        raise SystemExit("tile sizes differ from v4's")
    if c["participants"]["n"] != N:
        raise SystemExit("n differs")
    paths = [COND.relative_to(ROOT).as_posix(), m4["file"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    data = {"kupa_copy": c["data"]["kupa"]["files"]["copy"], "kupa_essay": c["data"]["kupa"]["files"]["essay"],
            "aalto": {"file": c["data"]["aalto"]["file"], "sha256": c["data"]["aalto"]["sha256"]}}
    for n, x in data.items():
        if cm.sha256(ROOT / x["file"]) != x["sha256"]:
            raise SystemExit(f"{x['file']} does not match the SHA-256 of the conditions")
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6-", dir=OUT_DIR.parent))
    try:
        rec = run(c, c4["mapping"], c4["relief"], tmp)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "v4_conditions_sha256": hashlib.sha256(raw4).hexdigest(),
                         "data_sha256": {n: x["sha256"] for n, x in data.items()},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "atact_library_sha256": cm.library_sha256(),
                         "versions": cm.versions("numpy", "pillow", "scipy"),
                         "bit_generator": type(np.random.default_rng(0).bit_generator).__name__,
                         "sensitivity_rng": "the sensitivities (registration_scale, ranks) reuse the bootstrap and "
                                            "permutation keys of the main standardisation"}
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    main_cell = rec["count_2_person"]["kupa_copy_to_essay"]["sentence"]["main"]
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("rank_auc", "rank_auc_ci", "reading")}
                      for k, v in main_cell.items() if k in STAGE}))


if __name__ == "__main__":
    main()
