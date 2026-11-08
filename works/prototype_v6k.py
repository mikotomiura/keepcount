"""Prototype v6k: KeepCount's procedure applied to summaries that writing-process research reports: the sentence
summary without the external scale (U), P-bursts (B, Bsd, Bsd_complete) and KLiCKe's pause means (I), at fixed
thresholds of 2 s, 1 s and 500 ms and at a threshold per record from a mixture of lognormals, on persons not used before
(decision 5 of .steering/20261108-paper-a-jowr-reframe; decisions 1-5 of .steering/20261108-v6k-standard-summaries).

Usage:  uv run python works/prototype_v6k.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6k.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6k/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value, per-person threshold or id is written.
"""

from __future__ import annotations

import atexit
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
atexit.register(shutil.rmtree, sys.pycache_prefix, ignore_errors=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import keystroke as ks  # noqa: E402
import prototype_v4 as v4  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v6b as v6b  # noqa: E402
import prototype_v6c as v6c  # noqa: E402
import prototype_v6d as v6d  # noqa: E402
import prototype_v6e as v6e  # noqa: E402
import prototype_v6i as v6i  # noqa: E402
import prototype_v6j as v6j  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6k.json"
OUT_DIR = ROOT / "results" / "prototype_v6k"
CODE = ["works/prototype_v6k.py", *v6j.CODE]
SEED = v6.SEED
REST_MS = 2000.0

THRESH = {"2000": 2000.0, "1000": 1000.0, "500": 500.0, "writer": None, "writer_fixed": None}
TH_IX = {"2000": 0, "1000": 1, "500": 2, "writer": 3, "writer_fixed": 4}
FIXED = ("2000", "1000", "500")
STAGE = {"S0": 0, "S1": 1, "U": 2, "B": 3, "Bsd": 4, "I": 5, "R": 6, "Bsd_complete": 7}
TH_STAGES = ("U", "B", "Bsd")
ROWS = [*v6.ROWS["kupa"]["sentence"], "random_half"]  # text/before first
ROW_IX = {row: i for i, row in enumerate(ROWS)}
COUNTING = {"text/after": "after", "text/none": "none"}
DATASET = {"kupa": 0, "lwp": 2}
TASK = {"copy": 0, "essay": 1, "lwp": 0}
SKIP = {"kupa": 437, "lwp": 204}
V6_LOOKED = 217
MAIN = (("kupa", "essay"), ("lwp", "lwp"))
MAIN_ROWS = ("inside/before", "shuffled/before", "shuffled_in_halves/before")
RULE_STAGES = ("U@2000", "B@2000", "Bsd@2000", "I")
BESIDE_RULE = ("Bsd_complete@2000",)
BY_CONSTRUCTION = {"U": {("inside/before", "oddeven"), ("inside/before", "halves")},
                   "B": {("inside/before", "oddeven"), ("inside/before", "halves"),
                         ("shuffled_in_halves/before", "halves")}}
KEEPS_PART_MULTISET = {("inside/before", "oddeven"), ("inside/before", "halves"), ("shuffled_in_halves/before", "halves")}
MIN_RULE_CELLS = 4
SIZE = 0.60
I_MS = 200.0
CONSTR_TOL = 1e-9
W_MIN_PER_K, W_STARTS, W_MAXIT, W_TOL, W_SD_FLOOR, W_MIN_WEIGHT, W_BIC_TIE = 20, 5, 2000, 1e-10, 0.01, 0.02, 1e-9
PAIRS = (("U@2000", "S1"), ("B@2000", "S1"), ("I", "S1"), ("S0", "U@2000"), ("U@1000", "U@2000"), ("U@500", "U@2000"),
         ("U@writer", "U@2000"))
SCALE = {"main": 0, "registration_scale": 1}
DIRECTIONS = {"kupa_copy_to_essay": ("copy", "essay", 0), "kupa_essay_to_copy": ("essay", "copy", 1)}


def sk(stage: str, th: str | None = None) -> str:
    return stage if th is None else f"{stage}@{th}"


def base_of(key: str) -> tuple[str, str | None]:
    st, _, th = key.partition("@")
    return st, (th or None)


def all_keys(count2: bool = False) -> list[str]:
    ths = ("2000", "1000", "500", "writer") if count2 else tuple(THRESH)
    return ["S0", "S1", "I", "R", *[sk(st, th) for st in TH_STAGES for th in ths], "Bsd_complete@2000"]


# ================================================================ the writer's threshold


def em(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, w: np.ndarray) -> dict | None:
    """EM for a one-dimensional Gaussian mixture from the given start; None if the log-likelihood is not finite."""
    mu, sd, w = mu.astype(float).copy(), np.maximum(sd.astype(float), W_SD_FLOOR), w.astype(float).copy()
    prev, it, reached = -np.inf, 0, False
    while True:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            logp = np.log(w) - 0.5 * math.log(2 * math.pi) - np.log(sd) - 0.5 * ((x[:, None] - mu) / sd) ** 2
            top = logp.max(axis=1, keepdims=True)
            lse = top + np.log(np.exp(logp - top).sum(axis=1, keepdims=True))
        ll = float(lse.sum())
        if not math.isfinite(ll):
            return None
        if it > 0 and ll - prev < W_TOL:
            break
        if it == W_MAXIT:
            reached = True
            break
        resp = np.exp(logp - lse)
        nk = resp.sum(axis=0)
        if np.any(nk <= 0):
            return None
        w = nk / x.size
        mu = (resp * x[:, None]).sum(axis=0) / nk
        sd = np.maximum(np.sqrt((resp * (x[:, None] - mu) ** 2).sum(axis=0) / nk), W_SD_FLOOR)
        prev, it = ll, it + 1
    return {"ll": ll, "mu": mu, "sd": sd, "w": w, "iterations": it, "reached": reached}


def normal_ll(x: np.ndarray, mu: float, sd: float) -> float:
    return float(np.sum(-0.5 * math.log(2 * math.pi) - math.log(sd) - 0.5 * ((x - mu) / sd) ** 2))


def fit_writer(x: np.ndarray, key: list[int]) -> dict | None:
    """The conditions' mixture on x (ln of the 'word' gaps of one record); key = [dataset, task, place, row index]."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return None
    sd1 = float(x.std())
    floored = sd1 == 0.0
    if floored:  # all gaps equal: the conditions are silent; the floor of EM is used (CHOICES)
        sd1 = W_SD_FLOOR
    fits = {1: {"ll": normal_ll(x, float(x.mean()), sd1), "mu": np.array([x.mean()]), "sd": np.array([sd1]),
                "w": np.array([1.0]), "iterations": 0, "reached": False}}
    discarded = reached = 0
    for k in (2, 3):
        if n < W_MIN_PER_K * k:
            continue
        sd0 = np.full(k, x.std() / k)
        w0 = np.full(k, 1.0 / k)
        starts = [np.quantile(x, [(2 * i + 1) / (2 * k) for i in range(k)], method="linear")]
        for s in range(1, W_STARTS):
            idx = np.random.default_rng([SEED, 92, *key, k, s]).choice(n, k, replace=False)
            starts.append(x[idx])
        best = None
        for mu0 in starts:
            f = em(x, np.asarray(mu0), sd0, w0)
            if f is not None:
                reached += int(f["reached"])
            if f is None or np.any(f["w"] < W_MIN_WEIGHT):
                discarded += 1
                continue
            if best is None or f["ll"] > best["ll"]:
                best = f
        if best is not None:
            fits[k] = best
    bic = {k: -2 * f["ll"] + (3 * k - 1) * math.log(n) for k, f in fits.items()}
    low = min(bic.values())
    chosen = min(k for k, b in bic.items() if b <= low + W_BIC_TIE)
    f = fits[chosen]
    order = np.argsort(f["mu"], kind="stable")
    j = order[1] if chosen >= 2 else order[0]
    return {"theta": float(math.exp(f["mu"][j] + 3 * f["sd"][j])), "k": chosen, "weight_j": float(f["w"][j]), "n": n,
            "discarded": discarded, "reached": reached, "k1_floored": floored}


# ================================================================ summaries of a part


class Rec:
    """One record under one counting: gaps, the sentences' held gap indices, labels and v4's p and q at 2 s."""

    def __init__(self, times: np.ndarray, ranges: list[tuple[int, int]], lab: np.ndarray, m: dict):
        self.gaps = np.diff(ks.check_times(times))
        self.ranges = ranges
        self.held = [np.arange(lo, hi) for lo, hi in ranges]
        self.n = len(ranges)
        self.lab = lab
        segs = ks.segments_of_gaps(self.gaps, ranges, REST_MS)
        self.p, self.q, _ = v4.pq(segs, m, "external")
        self._u: dict = {}

    def u_tiles(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Per sentence u_speed (nan without a short gap) and u_rest at theta."""
        if theta not in self._u:
            segs = ks.segments_of_gaps(self.gaps, self.ranges, theta)
            self._u[theta] = (np.array([math.log(s.speed_ms) if s.speed_ms is not None else np.nan for s in segs]),
                              np.array([s.rest_share for s in segs]))
        return self._u[theta]

    def held_of(self, idx: np.ndarray) -> np.ndarray:
        return np.concatenate([self.held[i] for i in idx]) if len(idx) else np.array([], dtype=int)


def runs(held: np.ndarray) -> list[np.ndarray]:
    """held (increasing gap indices) split into runs of consecutive indices."""
    if held.size == 0:
        return []
    cut = np.flatnonzero(np.diff(held) != 1) + 1
    return np.split(held, cut)


def bursts(gaps: np.ndarray, held: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Burst lengths (presses) over the runs of held, the lengths of the bursts bounded by long gaps at both ends, and
    the number of bursts cut by a run's edge."""
    lengths, complete, cut = [], [], 0
    for r in runs(held):
        g = gaps[r]
        long_at = np.flatnonzero(g >= theta - ks.TOL_MS)
        if long_at.size == 0:
            lengths.append(g.size + 1)
            cut += 1
            continue
        lengths.append(int(long_at[0]) + 1)
        mid = np.diff(long_at)
        lengths.extend(int(v) for v in mid)
        complete.extend(int(v) for v in mid)
        lengths.append(g.size - int(long_at[-1]))
        cut += 2
    return np.array(lengths, dtype=float), np.array(complete, dtype=float), cut


def b_identity(gaps: np.ndarray, held: np.ndarray, theta: float) -> float:
    """(G + r) / (L + r): the mean burst length from the counts alone."""
    r = len(runs(held))
    return (held.size + r) / (int(np.sum(gaps[held] >= theta - ks.TOL_MS)) + r)


def part_vectors(rec: Rec, idx: np.ndarray, th_values: dict[str, float | None]) -> tuple[dict, dict]:
    """The stage vectors of the sentences idx of a record, and per-part tallies (edge share of bursts at 2 s)."""
    held = rec.held_of(idx)
    if held.size == 0:
        raise ValueError("the part holds no gap")
    g = rec.gaps[held]
    out: dict = {"S0": np.quantile(np.log(g), v6.QUANTILES, method="linear"),
                 "S1": np.concatenate([v6.med_iqr(rec.p[idx]), v6.med_iqr(rec.q[idx])]),
                 "R": v6b.loc(rec, idx, rec.lab)}
    lab = rec.lab[held]
    over = g > I_MS
    a, wi, wo = g[over], g[over & (lab == "within")], g[over & (lab == "word")]
    out["I"] = None if min(a.size, wi.size, wo.size) == 0 else np.array([a.mean(), wi.mean(), wo.mean()]) / 1000.0
    tally = {"I0": float(a.mean()) / 1000.0 if a.size else None}
    for th, theta in th_values.items():
        if theta is None:
            for st in TH_STAGES:
                out[sk(st, th)] = None
            continue
        sp, rest = rec.u_tiles(theta)
        s = sp[idx]
        s = s[~np.isnan(s)]
        out[sk("U", th)] = None if s.size == 0 else np.concatenate([v6.med_iqr(s), v6.med_iqr(rest[idx])])
        lengths, complete, cut = bursts(rec.gaps, held, theta)
        mean = float(lengths.mean())
        if abs(mean - b_identity(rec.gaps, held, theta)) > CONSTR_TOL * max(1.0, mean):
            raise SystemExit("the mean burst length differs from (G + r) / (L + r)")
        out[sk("B", th)] = np.array([mean])
        out[sk("Bsd", th)] = np.array([float(lengths.std())])
        if th == "2000":
            out["Bsd_complete@2000"] = np.array([float(complete.std())]) if complete.size >= 2 else None
            tally["edge_share_2000"] = cut / lengths.size
    return out, tally


# ================================================================ persons and rows


def kupa_eligible(kupa: dict, pid: str) -> dict | None:
    """v6b's rule: both tasks."""
    ser = {}
    for task in ("copy", "essay"):
        x = kupa[task].get(pid)
        if x is None or x["times"].size < 2:
            return None
        k6 = v6b.v6_kinds(x["kinds"])
        st = v6.kupa_starts(k6)
        s = v6.Series(x["times"], st, v6.kupa_word_starts(k6))
        if len(st) < 5 or int(np.floor(s.times[-1] / v6.WINDOW_MS)) < 5 or not v6.every_sentence_holds_a_gap(s):
            return None
        ser[task] = (s, x["kinds"])
    return ser


def lwp_eligible(lwp: dict, pid: str) -> tuple | None:
    """v6e's rule (v6e.choose)."""
    x = lwp.get(pid)
    if x is None or x["times"].size < 2:
        return None
    k6 = v6b.v6_kinds(x["kinds"])
    st = v6.kupa_starts(k6)
    s = v6.Series(x["times"], st, v6.kupa_word_starts(k6))
    if len(st) < 5 or int(np.floor(s.times[-1] / v6.WINDOW_MS)) < 5 or not v6.every_sentence_holds_a_gap(s):
        return None
    return s, x["kinds"]


def draw(order: list, eligible, start: int, stop: int | None, n: int) -> tuple[list, dict, int]:
    """The first n eligible ids of order[start:stop] (all eligible ones if n is None), their series and the number
    looked at."""
    chosen, series, looked = [], {}, 0
    for pid in order[start:stop]:
        looked += 1
        x = eligible(pid)
        if x is not None:
            chosen.append(pid)
            series[pid] = x
            if n is not None and len(chosen) == n:
                break
    return chosen, series, looked


def ids_sha(ids: list) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def row_times(s: v6.Series, row: str, key: list[int]) -> np.ndarray:
    """The press times of a row (v6.make_rows' operations and keys; random_half as v6j); key = [dataset, task, place]."""
    if row.startswith("text/"):
        return s.times
    if row == "shuffled/before":
        return ks.shuffled(s.times, np.random.default_rng([SEED, 6, 4, *key]))
    if row == "inside/before":
        return ks.rests_inside_at(s.times, s.starts, s.word_starts, np.random.default_rng([SEED, 6, 5, *key]))[0]
    held = [np.arange(lo, hi) for lo, hi in s.ranges("before")]
    if row == "shuffled_in_halves/before":
        k = int(held[len(held) // 2][0])
        return ks.shuffled_in_two(s.times, k, np.random.default_rng([SEED, 6, 7, key[0], key[1], v6.UNIT["sentence"],
                                                                    key[2]]))
    if row == "random_half":
        n = len(held)
        half = np.random.default_rng([SEED, 6, v6j.KEY_RECEIVING, *key, v6j.CONTROL["random_half"]]).choice(
            n, n // 2, replace=False)
        g, _, _ = v6j.move_rests(np.diff(s.times), held, half, None, REST_MS,
                                 np.random.default_rng([SEED, 6, v6j.KEY_DRAW, *key, v6j.CONTROL["random_half"]]))
        return np.concatenate([[0.0], np.cumsum(g)])
    raise ValueError(row)


def count1_ok(s: v6.Series) -> bool:
    held = [np.arange(lo, hi) for lo, hi in s.ranges("before")]
    n = len(held)
    if n < v6.MIN_TILES_COUNT1:
        return False
    return all(sum(held[i].size for i in part) > 0 for how in v6b.BASELINE for part in v6b.parts(n, how))


def person_rows(s: v6.Series, kinds: list[str], key: list[int], m: dict, rows: list[str],
                writer: bool = True) -> tuple[dict, dict, dict]:
    """Per row: per baseline the (A, B) vectors; per row the writer fit; per row and baseline the tallies."""
    lab = v6b.labels(s, kinds)
    word = lab == "word"
    n = len(s.ranges("before"))
    out, fits, tallies = {}, {}, {}
    text_fit = None
    for row in rows:
        t = row_times(s, row, key)
        rec = Rec(t, s.ranges(COUNTING.get(row, "before")), lab, m)
        if rec.n != n:
            raise SystemExit(f"{row}: {rec.n} sentences, not {n}")
        if writer:
            if row.startswith("text/"):
                if text_fit is None:
                    text_fit = fit_writer(np.log(rec.gaps[word]), [*key, ROW_IX["text/before"]])
                fit = text_fit
            else:
                fit = fit_writer(np.log(rec.gaps[word]), [*key, ROW_IX[row]])
            fits[row] = fit
            tw = text_fit["theta"] if text_fit is not None else None
            th_values = {**{th: THRESH[th] for th in FIXED}, "writer": fit["theta"] if fit is not None else None,
                         "writer_fixed": tw}
        else:
            th_values = {th: THRESH[th] for th in FIXED}
        out[row], tallies[row] = {}, {}
        for how in v6b.BASELINE:
            A, B = v6b.parts(n, how)
            va, ta = part_vectors(rec, A, th_values)
            vb, tb = part_vectors(rec, B, th_values)
            out[row][how] = (va, vb)
            tallies[row][how] = (ta, tb)
    return out, fits, tallies


def check_construction(vec: dict, tallies: dict) -> None:
    """The identities of stages.by_construction, per person (fixed thresholds and writer_fixed); I's first number is
    taken from the tallies, so that it is compared even where I's vector is missing."""
    for row, how in KEEPS_PART_MULTISET:
        if row not in vec:
            continue
        for h in (0, 1):
            a, b = vec["text/before"][how][h], vec[row][how][h]
            pairs = {sk("B", th): (a[sk("B", th)], b[sk("B", th)]) for th in (*FIXED, "writer_fixed")}
            pairs["I0"] = (tallies["text/before"][how][h]["I0"], tallies[row][how][h]["I0"])
            if row == "inside/before":
                pairs.update({sk("U", th): (a[sk("U", th)], b[sk("U", th)]) for th in (*FIXED, "writer_fixed")})
                pairs["S1"] = (a["S1"], b["S1"])
            for nm, (x, y) in pairs.items():
                if x is None and y is None:
                    continue
                if (x is None) != (y is None):
                    raise SystemExit(f"{row} {how} {nm}: a value exists on one side only")
                if np.max(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))) > CONSTR_TOL:
                    raise SystemExit(f"{row} {how} {nm}: changed where it cannot by construction")


# ================================================================ count 1


def one_cell(hv: dict, row: str, how: str, key_name: str, rng_key: list[int], extra: str | None = None) -> dict:
    """v6b.one on the persons with key_name (and extra, if given) for the text and the row."""
    first_all, sel_all = hv[how]["text/before"], hv[how][row]
    n_all = len(first_all)
    need = [key_name] + ([extra] if extra else [])
    ok = [i for i in range(n_all) if all(v6b.has(first_all[i], k) and v6b.has(sel_all[i], k) for k in need)]
    if base_of(key_name)[0] == "R" and row == "text/none":
        if ok:
            raise SystemExit("text/none has R for some persons")
        return {"defined": False, "persons": 0}
    if len(ok) < 2:
        return {"too_few_persons": True, "persons": len(ok), "left_out_no_vector": n_all - len(ok)}
    res = v6b.one([first_all[i] for i in ok], [sel_all[i] for i in ok], key_name, rng_key)
    res["left_out_no_vector"] = n_all - len(ok)
    return res


def key1(ds: str, task: str, key_name: str, row: str, how: str, gate: int) -> list[int]:
    st, th = base_of(key_name)
    return [SEED, 90, DATASET[ds], TASK[task], STAGE[st], ROW_IX[row], v6b.BASELINE[how], TH_IX[th] if th else 0,
            gate]


def count1(hv: dict, ds: str, task: str) -> dict:
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for row in ROWS[1:]:
            out[how][row] = {k: one_cell(hv, row, how, k, key1(ds, task, k, row, how, 0)) for k in all_keys()}
    return out


def rule_persons(hv: dict, ds: str, task: str) -> dict:
    """Each rule stage and R on the persons who have both (main rows, both baselines)."""
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for row in MAIN_ROWS:
            cell = {}
            for k in (*RULE_STAGES, *BESIDE_RULE):
                gate = 1 + STAGE[base_of(k)[0]]
                cell[k] = {"stage": one_cell(hv, row, how, k, key1(ds, task, k, row, how, gate), extra="R"),
                           "R": one_cell(hv, row, how, "R", key1(ds, task, "R", row, how, gate), extra=k)}
            out[how][row] = cell
    return out


def size_class(res: dict) -> str:
    if "share_vs_own_ci" not in res:
        return "no count"
    lo, hi = res["share_vs_own_ci"]
    return "above" if lo > SIZE else ("below" if hi < SIZE else "undecided")


def label(classes: list[str]) -> str:
    if len(classes) < MIN_RULE_CELLS:
        return "not evaluable"
    s = set(classes)
    if s == {"above"}:
        return "shows these operations beyond the smallest difference"
    if s == {"below"}:
        return "does not show them beyond it"
    if s == {"undecided"}:
        return "undecided"
    if "above" in s and "below" in s:
        return "mixed directions"
    return "not beyond, partly undecided" if "below" in s else "beyond, partly undecided"


def evaluate_rule(rp: dict) -> dict:
    """rp[f'{ds}_{task}'][how][row][stage] = {'stage': ..., 'R': ...} on the rule persons."""
    out: dict = {}
    for k in (*RULE_STAGES, *BESIDE_RULE):
        st = base_of(k)[0]
        cells = {"read": {}, "status": {}, "left_out_of_rule_persons": {}}
        classes = []
        for ds, task in MAIN:
            nm = f"{ds}_{task}"
            for how in v6b.BASELINE:
                for row in MAIN_ROWS:
                    c = rp[nm][how][row][k]
                    where = f"{nm} | {how} | {row}"
                    cells["left_out_of_rule_persons"][where] = c["stage"]["left_out_no_vector"]
                    if (row, how) in BY_CONSTRUCTION.get(st, set()):
                        cells["status"][where] = "fixed by construction"
                        continue
                    if size_class(c["R"]) == "no count" or size_class(c["stage"]) == "no count":
                        cells["status"][where] = "fewer than 2 rule persons"
                        continue
                    if size_class(c["R"]) != "above":
                        cells["status"][where] = "control not met"
                        continue
                    cls = size_class(c["stage"])
                    cells["read"][where] = cls
                    cells["status"][where] = f"read: {cls}"
                    classes.append(cls)
        tally: dict = {}
        for where, cls in cells["read"].items():
            nm, _, row = where.split(" | ")
            tally.setdefault(f"{nm} | {row}", {"above": 0, "below": 0, "undecided": 0})[cls] += 1
        lab = label(classes) if k in RULE_STAGES else "beside the rule: no label"
        out[k] = {"label": lab, "cells_read": len(classes), "classes_by_dataset_and_row": tally, **cells}
    return out


def thresholds_part2(cnt: dict) -> dict:
    out: dict = {}
    for st in TH_STAGES:
        out[st] = {}
        ref = {}
        for ds, task in MAIN:
            for how in v6b.BASELINE:
                for row in MAIN_ROWS:
                    ref[(ds, task, how, row)] = size_class(cnt[f"{ds}_{task}"][how][row][sk(st, "2000")])
        for th in ("1000", "500", "writer", "writer_fixed"):
            classes = {}
            for (ds, task, how, row), r0 in ref.items():
                c = cnt[f"{ds}_{task}"][how][row][sk(st, th)]
                classes[f"{ds}_{task} | {how} | {row}"] = size_class(c)
            same = sum(classes[f"{ds}_{task} | {how} | {row}"] == r0 != "no count"
                       for (ds, task, how, row), r0 in ref.items())
            no_count = sum("no count" in (classes[f"{ds}_{task} | {how} | {row}"], r0)
                           for (ds, task, how, row), r0 in ref.items())
            out[st][th] = {"classes": classes, "same_as_2000": same, "cells_without_a_count": no_count,
                           "cells": len(ref)}
    return out


# ================================================================ count 2


def key2(direction: int, key_name: str, scale: str, paired: int) -> list[int]:
    st, th = base_of(key_name)
    return [SEED, 91, direction, STAGE[st], SCALE[scale], TH_IX[th] if th else 0, paired]


def distances(g: np.ndarray, e: np.ndarray, scale: str) -> tuple[np.ndarray | None, list[int]]:
    gs, es, drop = v6.standardise(g, e, scale)
    if gs.shape[1] == 0:
        return None, drop
    return v6.distances(es, gs), drop


def count2(vec2: dict) -> dict:
    """vec2[task] = list over the 200 persons of the whole-record vectors."""
    out: dict = {}
    n_all = len(vec2["copy"])
    for name, (a, b, direction) in DIRECTIONS.items():
        out[name] = {}
        for k in [x for x in all_keys(count2=True) if x != "Bsd_complete@2000"]:
            ok = [i for i in range(n_all) if vec2[a][i][k] is not None and vec2[b][i][k] is not None]
            g = np.array([vec2[a][i][k] for i in ok])
            e = np.array([vec2[b][i][k] for i in ok])
            cell = {"persons": len(ok), "left_out_no_vector": n_all - len(ok)}
            for scale in SCALE:
                if len(ok) < 2:
                    cell[scale] = {"too_few_persons": True}
                    continue
                d, drop = distances(g, e, scale)
                if d is None:
                    cell[scale] = {"constant": True, "dropped_features": drop}
                    continue
                res, _, _ = v6i.full_d(d, key2(direction, k, scale, 0))
                res["dropped_features"] = drop
                cell[scale] = res
            out[name][k] = cell
        out[name]["paired"] = {}
        for pi, (x, y) in enumerate(PAIRS):
            ok = [i for i in range(n_all) if all(vec2[t][i][s] is not None for t in (a, b) for s in (x, y))]
            pair = {"persons": len(ok)}
            for scale in SCALE:
                if len(ok) < 2:
                    pair[scale] = {"too_few_persons": True}
                    continue
                rr = {}
                for s in (x, y):
                    d, _ = distances(np.array([vec2[a][i][s] for i in ok]), np.array([vec2[b][i][s] for i in ok]),
                                     scale)
                    if d is None:
                        break
                    v6c.finite(d)
                    rr[s] = v6.per_person(d)[1]
                if len(rr) < 2:
                    pair[scale] = {"constant": True}
                    continue
                idx = np.random.default_rng(key2(direction, y, scale, 1 + pi)).integers(0, len(ok), (v6.N_BOOT, len(ok)))
                diff = rr[x] - rr[y]
                pair[scale] = {"rank_auc_difference": float(diff.mean()), "ci": v6.ci(diff, idx)}
            out[name]["paired"][f"{x} - {y}"] = pair
    return out


# ================================================================ checks on earlier persons


def check_v6_count2(series: dict, ids: list, m: dict, rec6: dict) -> None:
    vec = {task: [] for task in ("copy", "essay")}
    for pid in ids:
        for task in vec:
            s, kinds = series[pid][task]
            rec = Rec(s.times, s.ranges("before"), v6b.labels(s, kinds), m)
            v, _ = part_vectors(rec, np.arange(rec.n), {"2000": REST_MS})
            vec[task].append(v)
    for name, (a, b, _) in DIRECTIONS.items():
        for st in ("S0", "S1"):
            d, _ = distances(np.array([v[st] for v in vec[a]]), np.array([v[st] for v in vec[b]]), "main")
            r = float(v6.per_person(d)[1].mean())
            ref = rec6["count_2_person"][name]["sentence"]["main"][st]["rank_auc"]
            if not abs(r - ref) < v6c.CHECK_TOL:
                raise SystemExit(f"{name} {st}: rank_auc {r} differs from v6's {ref}")


def check_count1(hv: dict, ref: dict, rows: list[str], stages: dict[str, tuple[str, int]], prefix: list[int],
                 where: str) -> None:
    """hv[how][row] for the earlier persons; stages: my key -> (the earlier stage name, its stage index); the earlier
    key is prefix + [stage index, row index, baseline]. S-stages on every person, R on those with R (as v6b/v6e)."""
    for how in v6b.BASELINE:
        for row in rows:
            for mine, (theirs, si) in stages.items():
                r = ref[how][row][theirs]
                if mine == "R" and row == "text/none":
                    if r.get("defined", True):
                        raise SystemExit(f"{where}: text/none R defined in the earlier record")
                    continue
                first, sel = hv[how]["text/before"], hv[how][row]
                ok = [i for i in range(len(first)) if v6b.has(first[i], mine) and v6b.has(sel[i], mine)]
                if mine != "R" and len(ok) != len(first):
                    raise SystemExit(f"{where} {how} {row} {mine}: a vector is missing")
                res = v6b.one([first[i] for i in ok], [sel[i] for i in ok], mine,
                              [*prefix, si, ROW_IX[row], v6b.BASELINE[how]])
                for k in ("persons", "share_vs_own", "share_vs_own_ci"):
                    if res[k] != r[k]:
                        raise SystemExit(f"{where} {how} {row} {mine}: {k} {res[k]} differs from {r[k]}")


def hv_for_check(series: dict, ids: list, task: str | None, ds: str, place: dict, m: dict,
                 rows: list[str]) -> dict:
    hv = {how: {row: [] for row in rows} for how in v6b.BASELINE}
    for pid in ids:
        s, kinds = series[pid][task] if task else series[pid]
        if not count1_ok(s):
            continue
        vec, _, _ = person_rows(s, kinds, [DATASET[ds], TASK[task or "lwp"], place[pid]], m, rows, writer=False)
        for how in v6b.BASELINE:
            for row in rows:
                hv[how][row].append(vec[row][how])
    return hv


# ================================================================ the new persons


def build(series: dict, ids: list, ds: str, task: str, place: dict, m: dict) -> tuple[dict, dict]:
    """hv[row][how] over count persons, with the writer diagnostics and lengths."""
    hv = {row: {how: [] for how in v6b.BASELINE} for row in ROWS}
    fits_text, dev = [], {row: [] for row in ROWS[1:]}
    fits_rows = {row: [] for row in ROWS[1:] if not row.startswith("text/")}
    edge = {how: {"A": [], "B": []} for how in v6b.BASELINE}
    no_fit = {row: 0 for row in ROWS}
    left, sentences = 0, []
    for pid in ids:
        s, kinds = series[pid][task] if ds == "kupa" else series[pid]
        if not count1_ok(s):
            left += 1
            continue
        vec, fits, tallies = person_rows(s, kinds, [DATASET[ds], TASK[task], place[pid]], m, ROWS)
        check_construction(vec, tallies)
        sentences.append(len(s.ranges("before")))
        for row in ROWS:
            for how in v6b.BASELINE:
                hv[row][how].append(vec[row][how])
            if fits[row] is None:
                no_fit[row] += 1
        for row in fits_rows:
            if fits[row] is not None:
                fits_rows[row].append(fits[row])
        if fits["text/before"] is not None:
            fits_text.append(fits["text/before"])
            for row in ROWS[1:]:
                if fits[row] is not None:
                    dev[row].append(abs(math.log(fits[row]["theta"]) - math.log(fits["text/before"]["theta"])))
        for how in v6b.BASELINE:
            for part, t in zip("AB", tallies["text/before"][how]):
                edge[how][part].append(t["edge_share_2000"])
    diag = {"count_persons": len(sentences), "left_out_fewer_than_8_or_part_without_gap": left,
            "sentences": v6d.five(sentences), "records_without_theta_w": no_fit,
            "theta_w_text": writer_summary(fits_text),
            "theta_w_operated_rows": {row: writer_summary(f) for row, f in fits_rows.items()},
            "abs_ln_theta_w_row_minus_text": {row: v6d.five(v) if v else None for row, v in dev.items()},
            "edge_share_of_bursts_2000_text": {how: {part: v6d.five(v) for part, v in d.items()}
                                               for how, d in edge.items()}}
    return {how: {row: hv[row][how] for row in ROWS} for how in v6b.BASELINE}, diag


def writer_summary(fits: list[dict]) -> dict | None:
    if not fits:
        return None
    ks_ = [f["k"] for f in fits]
    return {"theta_ms": v6d.five([f["theta"] for f in fits]), "weight_j": v6d.five([f["weight_j"] for f in fits]),
            "gaps_fitted": v6d.five([f["n"] for f in fits]), "k": {str(k): ks_.count(k) for k in (1, 2, 3)},
            "starts_discarded": int(sum(f["discarded"] for f in fits)),
            "starts_reaching_max_iterations": int(sum(f["reached"] for f in fits)),
            "k1_floored": int(sum(f["k1_floored"] for f in fits))}


def vectors2(series: dict, ids: list, place: dict, m: dict) -> tuple[dict, dict]:
    vec = {task: [] for task in ("copy", "essay")}
    fits = {task: [] for task in vec}
    for pid in ids:
        for task in vec:
            s, kinds = series[pid][task]
            lab = v6b.labels(s, kinds)
            rec = Rec(s.times, s.ranges("before"), lab, m)
            fit = fit_writer(np.log(rec.gaps[lab == "word"]), [DATASET["kupa"], TASK[task], place[pid],
                                                               ROW_IX["text/before"]])
            if fit is not None:
                fits[task].append(fit)
            th = {**{t: THRESH[t] for t in FIXED}, "writer": fit["theta"] if fit is not None else None}
            v, _ = part_vectors(rec, np.arange(rec.n), th)
            vec[task].append(v)
    return vec, {task: writer_summary(f) for task, f in fits.items()}


# ================================================================ predictions and main cells


def predictions(rec: dict) -> dict:
    c1, c2 = rec["count_1"], rec["count_2"]

    def main_cells(stage_key, rows):
        return [c1[f"{ds}_{task}"][how][row][stage_key] for ds, task in MAIN for how in v6b.BASELINE for row in rows]

    def rp_cells(stage_key, rows):
        return [rec["rule_persons"][f"{ds}_{task}"][how][row][stage_key]["stage"] for ds, task in MAIN
                for how in v6b.BASELINE for row in rows]

    shuffles = ("shuffled/before", "shuffled_in_halves/before")
    p1 = all(size_class(c) == "above" for c in main_cells("R", shuffles))
    p2 = all(size_class(c) in ("below", "undecided") for c in rp_cells("U@2000", shuffles))
    b_changeable = [rec["rule_persons"][f"{ds}_{task}"][how][row]["B@2000"]["stage"] for ds, task in MAIN
                    for how in v6b.BASELINE for row in MAIN_ROWS if (row, how) not in BY_CONSTRUCTION["B"]]
    p3 = len(b_changeable) == 6 and all(size_class(c) == "below" for c in b_changeable)
    p4 = sum(size_class(c) == "above" for c in rp_cells("Bsd@2000", shuffles)) >= 4
    p5 = (sum(size_class(c) == "above" for c in rp_cells("I", ("inside/before",))) >= 3
          and all(size_class(c) == "above" for c in rp_cells("I", shuffles)))
    pair = c2["kupa_copy_to_essay"]["paired"]["U@2000 - S1"]["main"]
    p6 = "ci" in pair and pair["ci"][0] > 0
    med = rec["writer"]["kupa_count_2_records"]["essay"]["theta_ms"]["median"]
    p8 = all(rec["rule"]["part_2"][st][th]["same_as_2000"] >= 8 for st in TH_STAGES for th in ("1000", "500"))
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6,
            "P7": 500.0 <= med <= 2000.0, "P8": p8}


def main_cells_out(rec: dict) -> dict:
    out = {}
    for k in (*RULE_STAGES, *BESIDE_RULE):
        r = rec["rule"]["part_1"][k]
        out[f"rule | {k}"] = {"label": r["label"], "cells_read": r["cells_read"],
                              "classes": r["classes_by_dataset_and_row"]}
        for ds, task in MAIN:
            for how in v6b.BASELINE:
                for row in MAIN_ROWS:
                    c = rec["rule_persons"][f"{ds}_{task}"][how][row][k]
                    out[f"rule | {k} | {ds}_{task} | {how} | {row}"] = {
                        nm: ([round(c[nm]["share_vs_own"], 3), [round(x, 3) for x in c[nm]["share_vs_own_ci"]]]
                             if "share_vs_own_ci" in c[nm] else c[nm]) for nm in ("stage", "R")}
    c2 = rec["count_2"]["kupa_copy_to_essay"]
    for k in ("S0", "S1", "U@2000", "B@2000", "Bsd@2000", "I"):
        m = c2[k]["main"]
        out[f"count 2 | kupa_copy_to_essay | main | {k}"] = ([round(m["rank_auc"], 3),
                                                              [round(x, 3) for x in m["rank_auc_ci"]]]
                                                             if "rank_auc" in m else m)
    out["count 2 | kupa_copy_to_essay | main | paired"] = {k: v["main"] for k, v in c2["paired"].items()}
    return out


# ================================================================ run


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    if not OUT_DIR.parent.is_dir():
        raise SystemExit(f"{OUT_DIR.parent} does not exist")
    raw, c = cm.load_conditions(COND)
    cm.check_generator(c["generator"])
    from importlib import metadata
    for pkg in ("numpy", "pillow", "scipy"):
        if metadata.version(pkg) != c[pkg]:
            raise SystemExit(f"{pkg} is {metadata.version(pkg)}, not {c[pkg]}")
    for p, h in c["earlier"].items():
        cm.check_file(ROOT / p, h, p)
    c6 = json.loads((ROOT / "works/conditions/prototype_v6.json").read_bytes())
    c4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_bytes())
    if c6["units"]["rest_ms"] != REST_MS:
        raise SystemExit("rest_ms of v6's conditions differs")
    d_lwp = json.loads((ROOT / "works/conditions/prototype_v6e.json").read_text())["data"]["lwp"]
    kp = c6["data"]["kupa"]["files"]
    for f, h in [(d_lwp["file"], d_lwp["sha256"])] + [(x["file"], x["sha256"]) for x in kp.values()]:
        if cm.sha256(ROOT / f) != h:
            raise SystemExit(f"{f} does not match its SHA-256")
    recs = {k: json.loads((ROOT / f"results/prototype_{k}/record.json").read_text()) for k in ("v6", "v6b", "v6e")}
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    m = c4["mapping"]
    rec: dict = {"persons": {}, "checks": {}}

    # KUPA
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6b.read_kupa_b(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    kplace = {pid: i for i, pid in enumerate(ids_sorted)}
    order = v6.draw_order(ids_sorted, 0)
    elig = lambda pid: kupa_eligible(kupa, pid)  # noqa: E731
    v6_ids, v6_series, _ = draw(order, elig, 0, V6_LOOKED, None)
    v6b_ids, v6b_series, _ = draw(order, elig, V6_LOOKED, SKIP["kupa"], None)
    if ids_sha(v6_ids) != recs["v6"]["datasets"]["kupa"]["ids_sha256"] or len(v6_ids) != v6.N:
        raise SystemExit("positions 0-216 do not give v6's persons")
    if ids_sha(v6b_ids) != recs["v6b"]["persons"]["ids_sha256"] or len(v6b_ids) != v6.N \
            or recs["v6b"]["persons"]["ids_looked_at"] != SKIP["kupa"]:
        raise SystemExit("positions 217-436 do not give v6b's persons")
    k_ids, k_series, k_looked = draw(order, elig, SKIP["kupa"], None, v6.N)
    if len(k_ids) < v6.N or set(k_ids) & (set(v6_ids) | set(v6b_ids)):
        raise SystemExit("not 200 new eligible KUPA persons")
    rec["persons"]["kupa"] = {"ids_looked_at_from_437": k_looked, "ids_sha256": ids_sha(k_ids), "overlap": 0}
    del kupa

    # checks on earlier KUPA persons (before the new persons are counted)
    check_v6_count2(v6_series, v6_ids, m, recs["v6"])
    rec["checks"]["v6_count_2_S0_S1"] = "equal to v6's record"
    rows6 = v6.ROWS["kupa"]["sentence"]
    hv = hv_for_check(v6b_series, v6b_ids, "essay", "kupa", kplace, m, rows6)
    check_count1(hv, recs["v6b"]["count"]["essay_sentence"], rows6[1:], {"S0": ("S0", 0), "S1": ("S1", 1),
                                                                        "R": ("LOC", 4)},
                 [20261031, 66, v6.TASK["essay"], v6.UNIT["sentence"]], "v6b")
    rec["checks"]["v6b_count_1_essay_sentence_S0_S1_R"] = "equal to v6b's record"
    del v6_series, v6b_series, hv

    # LWP
    lwp, lids, tally = v6e.read_lwp(ROOT / d_lwp["file"], None)
    lplace = {pid: i for i, pid in enumerate(sorted(lids))}
    lorder = v6.draw_order(sorted(lids), 2)
    lelig = lambda pid: lwp_eligible(lwp, pid)  # noqa: E731
    e_ids, e_series, _ = draw(lorder, lelig, 0, SKIP["lwp"], None)
    if ids_sha(e_ids) != recs["v6e"]["persons"]["ids_sha256"] or len(e_ids) != v6.N \
            or recs["v6e"]["persons"]["ids_looked_at"] != SKIP["lwp"]:
        raise SystemExit("positions 0-203 do not give v6e's persons")
    l_ids, l_series, l_looked = draw(lorder, lelig, SKIP["lwp"], None, v6.N)
    if len(l_ids) < v6.N or set(l_ids) & set(e_ids):
        raise SystemExit("not 200 new eligible LWP persons")
    rec["persons"]["lwp"] = {"ids_looked_at_from_204": l_looked, "ids_sha256": ids_sha(l_ids), "overlap": 0,
                             "tally_of_reading": tally}
    del lwp
    rows_e = [rows6[0], *MAIN_ROWS]
    hv = hv_for_check(e_series, e_ids, None, "lwp", lplace, m, rows_e)
    check_count1(hv, recs["v6e"]["count"]["sentence"], list(MAIN_ROWS), {"S1": ("S1", 1), "R": ("LOC", 4)},
                 [20261031, 69, v6.UNIT["sentence"]], "v6e")
    rec["checks"]["v6e_count_1_sentence_S1_R"] = "equal to v6e's record"
    del e_series, hv

    # the new persons: count 1
    hvs, rec["writer"], rec["count_1"], rec["rule_persons"] = {}, {}, {}, {}
    for ds, task, series, ids, place in (("kupa", "essay", k_series, k_ids, kplace),
                                         ("kupa", "copy", k_series, k_ids, kplace),
                                         ("lwp", "lwp", l_series, l_ids, lplace)):
        nm = f"{ds}_{task}"
        hv, rec["writer"][nm] = build(series, ids, ds, task, place, m)
        rec["count_1"][nm] = count1(hv, ds, task)
        if (ds, task) in MAIN:
            rec["rule_persons"][nm] = rule_persons(hv, ds, task)
        del hv
    rec["checks"]["by_construction"] = "every identity held for every count person"
    rec["rule"] = {"part_1": evaluate_rule(rec["rule_persons"]), "part_2": thresholds_part2(rec["count_1"])}

    # count 2
    vec2, rec["writer"]["kupa_count_2_records"] = vectors2(k_series, k_ids, kplace, m)
    rec["count_2"] = count2(vec2)
    del vec2
    rec["predictions"] = predictions(rec)
    rec = {"main_cells": main_cells_out(rec), **rec}
    v6d.no_per_person(rec)
    v6i.no_long_list(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "data_sha256": {"kupa_copy": kp["copy"]["sha256"], "kupa_essay": kp["essay"]["sha256"],
                                         "lwp": d_lwp["sha256"]},
                         "choices_in_code": CHOICES,
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6k-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "text_rows_fit": "text/after and text/none are the text's record with another counting: they reuse the text's "
                     "fit (key row index 0)",
    "count_2_fit": "the whole-record fits of count 2 are the text fits of the same task with the same key",
    "k1_floor": "the closed-form k = 1 fit floors its standard deviation at 0.01 as EM does",
    "em_start_index": "start (i) is the quantile start; starts (ii-v) use start indices 1-4 in the key",
    "em_iterations": "an iteration is one M-step; the log-likelihood is computed before each and the loop stops when "
                     "it rises by less than 1e-10 or after 2000 M-steps (then it is computed once more)",
    "em_empty_component": "a start whose component gets a total responsibility of 0 is discarded with the non-finite",
    "discarded_count": "starts discarded and starts reaching 2000 iterations are summed over the records of a set",
    "construction_check": "at the fixed thresholds and writer_fixed: B under the three rows/baselines that keep each "
                          "part's runs' multisets, U and S1 under inside, and I's first number under the same three",
    "b_identity": "every mean burst length is also checked against (G + r) / (L + r) (relative 1e-9)",
    "rule_counts": "the rule persons' counts are reported per main cell under rule_persons; the counts on all "
                   "persons of a stage (gate 0) under count_1",
    "predictions": "P1 uses R's counts on all persons (gate 0); P2-P5 use the rule persons' cells directly (P2: no "
                   "shuffle cell above, whatever the control; P3: all 6 changeable cells below); P7 uses the 200 "
                   "KUPA essay records of count 2; P8 uses gate 0 and counts a cell as agreeing only if both classes "
                   "are counts",
    "k1_zero_variance": "a k = 1 fit with standard deviation 0 (all gaps equal; the conditions are silent) uses EM's "
                        "floor of 0.01; such fits are counted (k1_floored)",
    "bic_tie": "the smallest k whose BIC is within 1e-9 of the lowest BIC",
    "bsd_complete": "Bsd_complete@2000 is counted on its persons (gate 0) and on the rule persons with R (gate 8), and "
                    "gets no label; it is not counted in count 2",
    "construction_check_I0": "I's first number is compared from the part tallies, also where I's vector is missing; "
                             "S1 is checked under inside too",
    "fits_summed": "the text fits are summarised per dataset and task; the operated rows' fits per row "
                   "(theta_w_operated_rows)",
    "edge_share_parts": "the edge share is summarised per baseline and part (A, B) over count persons",
    "new_persons_overlap": "the new persons come from positions after those of v6, v6b and v6e in the same order, so "
                           "the overlap check cannot fail; it is kept as a guard",
    "lwp_task_code": "LWP's task code is 0 in every key, as v6e",
    "too_few_persons": "a count with fewer than 2 persons is recorded as too_few_persons instead of stopping the run "
                       "(v6b.one stops); in part 1 such a cell is 'fewer than 2 rule persons' and not read",
    "edge_share": "the share of bursts cut by a run's edge is summarised over the text's parts at 2 s",
}


if __name__ == "__main__":
    main()
