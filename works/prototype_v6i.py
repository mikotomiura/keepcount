"""Prototype v6i: (a) the dispersion of p and q over the sentences of a record against exchangeable series H (and
circular shifts C), person by person, as a calibrated percentile and a ratio instead of a win rate; (b) the match of
section 6 (v6c's measures) counted at the stages nearest the image: SEQ, the tile sequence aligned by relative position
(IMG5, IMG3), an order-discarded control, DTW and the number of tiles (decision 7 of .steering/20261101-paper-a-draft;
decisions 1 and 2 of .steering/20261101-v6i-ceiling-and-image-id).

Usage:  uv run python works/prototype_v6i.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6i.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2, the 136M Keystrokes zip and the LWP logs (external/; not in git, checked by
        SHA-256; LWP not redistributed)
Output: results/prototype_v6i/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value or id is written.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
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
import prototype_v6f as v6f  # noqa: E402
import prototype_v6g as v6g  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6i.json"
OUT_DIR = ROOT / "results" / "prototype_v6i"
CODE = ["works/prototype_v6i.py", "works/prototype_v6g.py", "works/prototype_v6f.py", "works/prototype_v6e.py",
        "works/prototype_v6d.py", "works/prototype_v6c.py", "works/prototype_v6b.py", "works/prototype_v6.py",
        "works/prototype_v5.py", "works/prototype_v4.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
SEED = v6.SEED

# ---- part (a)
K = 1000
KEY_H, KEY_C, KEY_GATHERED = 16, 30, 8
NUMBERS_A = ("p", "q")
STAT = {"raw": 0, "detrended": 1, "mad": 2}
NULLS = ("H", "C")
MAIN_A = (("kupa", "essay"), ("lwp", "lwp"))
TAIL = 0.025

# ---- part (b)
STAGE_B = {"S0": 0, "S1": 1, "S2": 2, "SEQ": 3, "IMG5_pq": 4, "IMG5_df": 5, "IMG5_centred": 6, "IMG3_pq": 7, "NT": 8,
           "IMG5_pq_NT": 9, "IMG5_pq_scrambled": 10, "DTW_pq": 11}
NEW_VECTOR = ("SEQ", "IMG5_pq", "IMG5_df", "IMG5_centred", "IMG3_pq", "NT", "IMG5_pq_NT", "IMG5_pq_scrambled")
NEEDS_5 = ("SEQ", "IMG5_pq", "IMG5_df", "IMG5_centred", "IMG3_pq", "IMG5_pq_NT", "IMG5_pq_scrambled")
PAIRED = (("IMG5_pq", "S1"), ("IMG5_df", "S2"), ("SEQ", "S1"), ("IMG5_pq_NT", "IMG5_pq"),
          ("IMG5_pq", "IMG5_pq_scrambled"), ("DTW_pq", "S1"))
PART = {"copy": 0, "essay": 1, "A": 0, "B": 1}


# ================================================================ part (a): dispersion against exchangeable series


def pq_of_gaps(gaps: np.ndarray, ranges: list[tuple[int, int]], m: dict, rest_ms: float) -> tuple[np.ndarray, ...]:
    """p and q of the sentence tiles cut at ranges (v4's external scale), without the relief."""
    p, q, _ = v4.pq(ks.segments_of_gaps(gaps, ranges, rest_ms), m, "external")
    return p, q


def dispersion(x: np.ndarray) -> dict[str, float]:
    """T raw (standard deviation, ddof 0), T detrended (of the residuals of a least-squares line on the tile index)
    and the mean absolute deviation from the mean."""
    x = np.asarray(x, dtype=float)
    k = np.arange(x.size, dtype=float)
    if np.ptp(x) == 0:
        res = np.zeros_like(x)
    else:
        b, a = np.polyfit(k, x, 1)
        res = x - (a + b * k)
    return {"raw": float(x.std()), "detrended": float(res.std()), "mad": float(np.mean(np.abs(x - x.mean())))}


def mid_rank(t: float, null: np.ndarray) -> float:
    """(number of null values below t + 1/2 number equal) / number of null values; exact equality."""
    return float((np.sum(null < t) + 0.5 * np.sum(null == t)) / null.size)


def circular(gaps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The gaps rotated by an offset drawn uniformly from 1..G-1."""
    if gaps.size < 2:
        raise SystemExit("a circular shift needs at least 2 gaps")
    return np.roll(gaps, int(rng.integers(1, gaps.size)))


def eligible_a(ranges: list[tuple[int, int]]) -> str | None:
    """v6h's count persons: at least 8 tiles and a gap in every part of both baselines; else the reason."""
    n = len(ranges)
    if n < v6.MIN_TILES_COUNT1:
        return "fewer_than_8"
    size = np.array([hi - lo for lo, hi in ranges])
    if any(size[part].sum() == 0 for how in v6b.BASELINE for part in v6b.parts(n, how)):
        return "part_without_gap"
    return None


def person_a(s: v6.Series, key: list[int], m: dict, rest_ms: float) -> dict:
    """Per number and statistic: T of text and of gathered, and T of the K series of each null."""
    g = np.diff(s.times)
    ranges = s.ranges("before")
    held = [np.arange(lo, hi) for lo, hi in ranges]
    out = {"text": {}, "gathered": {}, "H": {}, "C": {}, "no_speed": 0}
    p, q = pq_of_gaps(g, ranges, m, rest_ms)
    out["no_speed"] = int(sum(1 for lo, hi in ranges if not (g[lo:hi] < rest_ms - ks.TOL_MS).any()))
    out["text"] = {"p": dispersion(p), "q": dispersion(q)}
    out["pq_text"] = (p, q)
    tg, _, _ = v6f.gathered(s.times, held, rest_ms, np.random.default_rng([SEED, 6, KEY_GATHERED, *key]))
    pg, qg = pq_of_gaps(np.diff(tg), ranges, m, rest_ms)
    out["gathered"] = {"p": dispersion(pg), "q": dispersion(qg)}
    for null, kk in (("H", KEY_H), ("C", KEY_C)):
        acc = {x: {st: np.empty(K) for st in STAT} for x in NUMBERS_A}
        for k in range(K):
            rng = np.random.default_rng([SEED, 6, kk, *key, k])
            gn = rng.permutation(g) if null == "H" else circular(g, rng)
            pn, qn = pq_of_gaps(gn, ranges, m, rest_ms)
            for x, v in (("p", pn), ("q", qn)):
                for st, val in dispersion(v).items():
                    acc[x][st][k] = val
        out[null] = acc
    return out


def per_person_a(pa: dict) -> dict:
    """r and R per number, statistic and null; and gathered's r and R against H (raw)."""
    out = {}
    for x in NUMBERS_A:
        for st in STAT:
            t = pa["text"][x][st]
            for null in NULLS:
                nv = pa[null][x][st]
                med = float(np.median(nv))
                out[(x, st, null)] = {"r": mid_rank(t, nv), "R": t / med if med > 0 else None,
                                      "degenerate": bool(np.all(nv == t)), "T": t, "median_null": med,
                                      "width_null": float(np.percentile(nv, 97.5) - np.percentile(nv, 2.5))}
        nv = pa["H"][x]["raw"]
        med = float(np.median(nv))
        t = pa["gathered"][x]["raw"]
        out[(x, "gathered")] = {"r": mid_rank(t, nv), "R": t / med if med > 0 else None}
    return out


def ci_mean(v: np.ndarray, idx: np.ndarray) -> list[float]:
    m = v[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def ci_median(v: np.ndarray, idx: np.ndarray) -> list[float]:
    m = np.median(v[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def r_cell(rs: np.ndarray, Rs: list, degenerate: np.ndarray, idx: np.ndarray) -> dict:
    """mean r with its interval, the tail shares (all persons and the non-degenerate), median R with its interval (on
    the persons with R defined, resampled as the same persons)."""
    nd = ~degenerate
    out = {"persons": int(rs.size), "mean_r": float(rs.mean()), "mean_r_ci": ci_mean(rs, idx),
           "share_r_at_or_above_0.975": float(np.mean(rs >= 1 - TAIL)),
           "share_r_at_or_below_0.025": float(np.mean(rs <= TAIL)),
           "persons_degenerate": int(degenerate.sum()),
           "non_degenerate": ({"persons": int(nd.sum()), "mean_r": float(rs[nd].mean()),
                               "share_r_at_or_above_0.975": float(np.mean(rs[nd] >= 1 - TAIL)),
                               "share_r_at_or_below_0.025": float(np.mean(rs[nd] <= TAIL))} if nd.any() else None)}
    defined = np.array([x is not None for x in Rs])
    out["persons_R_undefined"] = int((~defined).sum())
    if defined.any():
        R = np.array([x if x is not None else np.nan for x in Rs], dtype=float)
        meds = []
        for row in idx:
            v = R[row][defined[row]]
            if v.size:
                meds.append(np.median(v))
        out["median_R"] = float(np.median(R[defined]))
        out["median_R_ci"] = [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]
    return out


def count_a(pp: list[dict], ds: str, task: str) -> dict:
    """Per number, statistic and null: the cell of r and R; r_p - r_q paired; gathered; the null summaries."""
    n = len(pp)
    out: dict = {"persons": n}
    for st, si in STAT.items():
        idx = np.random.default_rng([SEED, 80, v6f.DATASET[ds], v6f.TASK[task], si]).integers(0, n, (v6.N_BOOT, n))
        for null in NULLS:
            cell = out.setdefault(null, {}).setdefault(st, {})
            for x in NUMBERS_A:
                rs = np.array([p[(x, st, null)]["r"] for p in pp])
                cell[x] = r_cell(rs, [p[(x, st, null)]["R"] for p in pp],
                                 np.array([p[(x, st, null)]["degenerate"] for p in pp]), idx)
                cell[x]["summaries"] = {k: v6d.five([p[(x, st, null)][k] for p in pp])
                                        for k in ("T", "median_null", "width_null")}
            d = np.array([p[("p", st, null)]["r"] - p[("q", st, null)]["r"] for p in pp])
            cell["r_p_minus_r_q"] = {"mean": float(d.mean()), "ci": ci_mean(d, idx)}
        if st == "raw":
            out["gathered"] = {}
            for x in NUMBERS_A:
                rs = np.array([p[(x, "gathered")]["r"] for p in pp])
                out["gathered"][x] = r_cell(rs, [p[(x, "gathered")]["R"] for p in pp], np.zeros(n, dtype=bool),
                                            idx)
                del out["gathered"][x]["non_degenerate"], out["gathered"][x]["persons_degenerate"]
    return out


def label_half(ci: list[float]) -> str:
    return "above 0.5" if ci[0] > 0.5 else ("below 0.5" if ci[1] < 0.5 else "contains 0.5")


def evaluate_a(cnt: dict) -> dict:
    names = [f"{ds}_{task}" for ds, task in MAIN_A]
    control = {nm: cnt[nm]["gathered"]["q"]["mean_r_ci"][0] > 0.5 for nm in names}

    def verdict(x: str, gate: bool) -> tuple[str, dict]:
        cells = {}
        for nm in names:
            lab = label_half(cnt[nm]["H"]["raw"][x]["mean_r_ci"])
            cells[nm] = "not evaluable" if gate and not control[nm] else lab
        labs = list(cells.values())
        if all(v == "above 0.5" for v in labs):
            v = ("the dispersion of q over the sentences of a record is larger than under exchangeable gaps with the "
                 "same number of gaps per sentence") if x == "q" else "above exchangeable gaps in both cells"
        elif all(v == "contains 0.5" for v in labs):
            v = "not confirmed"
        elif all(v == "not evaluable" for v in labs):
            v = "not evaluable"
        else:
            v = "mixed"
        return v, cells

    a1, c1 = verdict("q", True)
    a2, c2 = verdict("p", False)
    a3 = {}
    for nm in names:
        lo, hi = cnt[nm]["H"]["raw"]["r_p_minus_r_q"]["ci"]
        a3[nm] = "p above q" if lo > 0 else ("q above p" if hi < 0 else "no difference confirmed")
    c = {nm: cnt[nm]["H"]["raw"] for nm in names}
    pred = {"P1": label_half(c["kupa_essay"]["q"]["mean_r_ci"]) == "contains 0.5"
            and label_half(c["lwp_lwp"]["q"]["mean_r_ci"]) == "above 0.5",
            "P2": all(c[nm]["p"]["mean_r_ci"][0] > 0.5 for nm in names),
            "P3": all(c[nm]["r_p_minus_r_q"]["ci"][0] > 0 for nm in names),
            "P4": all(0.9 <= c[nm]["q"]["median_R"] <= 1.2 for nm in names),
            "P9": all(control.values())}
    return {"A1": a1, "A1_cells": c1, "A2": a2, "A2_cells": c2, "A3": a3, "positive_control_met": control,
            "predictions": pred}


def kupa_persons_a(c6: dict, recs: dict) -> tuple[list[str], dict, dict]:
    chosen, series, place = v6f.kupa_persons(c6, recs["v6"], recs["v6b"])
    return chosen, {pid: series[pid]["essay"][0] for pid in chosen}, place


def lwp_persons_a(path: Path, recs: dict) -> tuple[list[str], dict, dict]:
    chosen, series, place = v6f.lwp_persons(path, recs["v6e"])
    return chosen, {pid: series[pid]["lwp"][0] for pid in chosen}, place


def run_a(c6: dict, m: dict, r: dict, recs: dict, d_lwp: dict) -> dict:
    rest_ms = c6["units"]["rest_ms"]
    rec: dict = {"persons": {}, "left_out": {}, "count": {}, "tallies": {}}
    for ds, task in MAIN_A:
        name = f"{ds}_{task}"
        if ds == "kupa":
            chosen, series, place = kupa_persons_a(c6, recs)
        else:
            chosen, series, place = lwp_persons_a(ROOT / d_lwp["file"], recs)
        ids = hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest()
        if ids != recs["v6h"]["persons"][ds]["ids_sha256"]:
            raise SystemExit(f"{name}: the drawn persons differ from v6h's")
        rec["persons"][ds] = {"n": len(chosen), "ids_sha256": ids}
        left = {"fewer_than_8": 0, "part_without_gap": 0}
        pp, no_speed = [], []
        for pid in chosen:
            s = series[pid]
            why = eligible_a(s.ranges("before"))
            if why is not None:
                left[why] += 1
                continue
            pa = person_a(s, [v6f.DATASET[ds], v6f.TASK[task], place[pid]], m, rest_ms)
            tt = v6.Tiles(s, "sentence", "before", m, r, rest_ms)
            if not (np.array_equal(tt.p, pa["pq_text"][0]) and np.array_equal(tt.q, pa["pq_text"][1])):
                raise SystemExit(f"{name}: p and q without the relief differ from v6.Tiles'")
            no_speed.append(pa["no_speed"])
            pp.append(per_person_a(pa))
        if left != recs["v6h"]["left_out"][name]:
            raise SystemExit(f"{name}: the persons left out differ from v6h's")
        rec["left_out"][name] = left
        rec["count"][name] = count_a(pp, ds, task)
        rec["tallies"][name] = {"tiles_without_speed": v6d.five(no_speed)}
        del series, pp
    rec["rule"] = evaluate_a(rec["count"])
    return rec


# ================================================================ part (b): identification nearest the image


def groups(n: int, k: int) -> list[np.ndarray]:
    if n < k:
        raise SystemExit(f"{n} tiles, fewer than {k} groups")
    return np.array_split(np.arange(n), k)


def binned(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Per group of consecutive tiles (array_split into k), the mean of a and of b, in group order."""
    return np.array([f(x[gi]) for gi in groups(a.size, k) for f, x in ((np.mean, a), (np.mean, b))])


def new_vectors(t: v6.Tiles, scramble: np.random.Generator) -> dict[str, np.ndarray]:
    """The new vector stages of one part."""
    n = t.n
    if n < 5:
        raise SystemExit(f"{n} tiles in a part counted with groups of 5")
    img5 = binned(t.p, t.q, 5)
    level = np.tile([t.p.mean(), t.q.mean()], 5)
    o = scramble.permutation(n)
    return {"SEQ": v6b.seq(t, np.arange(n)), "IMG5_pq": img5, "IMG5_df": binned(t.depth, t.fineness, 5),
            "IMG5_centred": img5 - level, "IMG3_pq": binned(t.p, t.q, 3), "NT": np.array([float(n)]),
            "IMG5_pq_NT": np.concatenate([img5, [float(n)]]), "IMG5_pq_scrambled": binned(t.p[o], t.q[o], 5)}


def dtw(a: np.ndarray, gs: list[np.ndarray]) -> np.ndarray:
    """DTW of a (n, 2) against every g in gs: minimal path cost with Euclidean local cost, divided by n + m. The
    registrations are padded to the longest with an infinite cost, which only lies to the right of each end cell."""
    n, big_m = a.shape[0], max(g.shape[0] for g in gs)
    lens = np.array([g.shape[0] for g in gs])
    pad = np.full((len(gs), big_m, 2), np.nan)
    for i, g in enumerate(gs):
        pad[i, :g.shape[0]] = g
    cost = np.sqrt(((a[None, :, None, :] - pad[:, None, :, :]) ** 2).sum(axis=3))  # (G, n, M)
    cost = np.where(np.isnan(cost), np.inf, cost)
    d = np.full((len(gs), big_m), np.inf)
    for i in range(n):
        new = np.empty_like(d)
        for j in range(big_m):
            best = d[:, j] if i > 0 else np.full(len(gs), np.inf)
            if j > 0:
                best = np.minimum(best, new[:, j - 1])
                if i > 0:
                    best = np.minimum(best, d[:, j - 1])
            if i == 0 and j == 0:
                best = np.zeros(len(gs))
            new[:, j] = cost[:, i, j] + best
        d = new
    return d[np.arange(len(gs)), lens - 1] / (n + lens)


def dtw_matrix(reg: list[np.ndarray], ev: list[np.ndarray]) -> np.ndarray:
    """d[i, j] = DTW of evaluation i to registration j, each number standardised over all tiles of its set."""
    def z(seqs):
        allv = np.concatenate(seqs)
        sd = allv.std(axis=0)
        if np.any(sd == 0):
            raise SystemExit("a number of DTW has standard deviation 0 over a set")
        return [(s - allv.mean(axis=0)) / sd for s in seqs]

    rz, ez = z(reg), z(ev)
    return np.array([dtw(e, rz) for e in ez])


def full_d(d: np.ndarray, key: list[int]) -> tuple[dict, np.ndarray, np.ndarray]:
    """v6c's full set on a distance matrix: top1, rank_auc with its interval, Rank-n rates, EER."""
    v6c.finite(d)
    top1, rauc = v6.per_person(d)
    rates = v6c.rank_rates(d, v6c.RANKS)
    n = d.shape[0]
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    return {"top1": float(top1.mean()), "rank_auc": float(rauc.mean()), "rank_auc_ci": v6.ci(rauc, idx),
            "reading": v6.reading(*v6.ci(rauc, idx)),
            "rank_n": {str(k): {"rate": float(v.mean()), "ci": v6.ci(v, idx), "chance": k / n} for k, v in rates.items()},
            "eer": v6c.eer(d)}, rauc, idx


def stage_vectors(g: np.ndarray, e: np.ndarray) -> tuple[np.ndarray | None, list[int]]:
    gs, es, drop = v6.standardise(g, e, "main")
    if gs.shape[1] == 0:
        return None, drop
    return v6.distances(es, gs), drop


def tiles_b(c6: dict, m: dict, r: dict, v6rec: dict) -> dict:
    """(dataset, part) -> unit -> list of Tiles in v6's person order, drawn as v6c.vectors draws them."""
    rest_ms = c6["units"]["rest_ms"]
    kp = c6["data"]["kupa"]["files"]
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6.read_kupa(ROOT / kp[task]["file"])
    chosen, series = [], {}
    for pid in v6.draw_order(sorted(seen["essay"]), 0):
        ok, ser = True, {}
        for task in ("copy", "essay"):
            x = kupa[task].get(pid)
            if x is None or x["times"].size < 2:
                ok = False
                break
            st = v6.kupa_starts(x["kinds"])
            s = v6.Series(x["times"], st, v6.kupa_word_starts(x["kinds"]))
            if len(st) < 5 or int(np.floor(s.times[-1] / v6.WINDOW_MS)) < 5 or not v6.every_sentence_holds_a_gap(s):
                ok = False
                break
            ser[task] = s
        if ok:
            chosen.append(pid)
            series[pid] = ser
            if len(chosen) == v6.N:
                break
    if hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest() != v6rec["datasets"]["kupa"]["ids_sha256"]:
        raise SystemExit("KUPA: the persons differ from v6's")
    del kupa
    with zipfile.ZipFile(ROOT / c6["data"]["aalto"]["file"]) as zf:
        kb = v6.aalto_ids(zf)
        tally = {"missing_files": 0, "short_lines": 0, "lines_read_from_ends": 0, "not_counted": 0,
                 "dropped_equal_times": 0, "part_without_window": 0, "digit_letters_of_eligible": 0,
                 "keyboard_not_eligible": 0}
        a_chosen, a_series = [], {}
        for pid in v6.draw_order(sorted(kb), 1):
            if not kb[pid]:
                continue
            sents = v6.read_aalto(zf, pid, tally)
            if sents is None:
                continue
            a_chosen.append(pid)
            a_series[pid] = sents
            if len(a_chosen) == v6.N:
                break
    if hashlib.sha256("\n".join(str(x) for x in sorted(a_chosen)).encode()).hexdigest() != \
            v6rec["datasets"]["aalto"]["ids_sha256"]:
        raise SystemExit("Aalto: the persons differ from v6's")
    out: dict = {}
    for task in ("copy", "essay"):
        for pid in chosen:
            for unit in v6.UNIT:
                counting = v6.ROWS["kupa"][unit][0].partition("/")[2] or "before"
                out.setdefault(("kupa", task), {}).setdefault(unit, []).append(
                    v6.Tiles(series[pid][task], unit, counting, m, r, rest_ms))
    for pid in a_chosen:
        for part, sl in (("A", slice(0, 7)), ("B", slice(7, 15))):
            sp = v6.aalto_series(a_series[pid][sl])
            for unit in v6.UNIT:
                out.setdefault(("aalto", part), {}).setdefault(unit, []).append(v6.Tiles(sp, unit, "before", m, r,
                                                                                         rest_ms))
    return out


def stages_of(tl: list[v6.Tiles], ds: str, part: str, unit: str, with_new: bool) -> dict[str, np.ndarray]:
    """unit's vectors (n persons, k) of the kept stages and, where every part has 5 tiles, of the new ones."""
    vec: dict = {st: [] for st in ("S0", "S1", "S2")}
    for place, t in enumerate(tl):
        for st, v in t.all().items():
            vec[st].append(v)
        nv = new_vectors(t, np.random.default_rng([SEED, 64, 3, v6.DATASET[ds], PART[part], v6.UNIT[unit], place])) \
            if with_new else {"NT": np.array([float(t.n)])}
        for st, v in nv.items():
            vec.setdefault(st, []).append(v)
    return {st: np.array(v) for st, v in vec.items()}


def run_b(tiles: dict, v6crec: dict) -> dict:
    rec: dict = {"count": {}}
    for name, (ds, a, b, direction) in v6c.DIRECTIONS.items():
        rec["count"][name] = {}
        for unit in v6.UNIT:
            with_new = not (ds == "aalto" and unit == "window_10s")
            reg = stages_of(tiles[(ds, a)][unit], ds, a, unit, with_new)
            ev = stages_of(tiles[(ds, b)][unit], ds, b, unit, with_new)
            cell, raucs, idxs = {}, {}, {}
            for st in ("S0", "S1", "S2"):  # v6c's key, so that the check covers the intervals too
                ref = v6crec["full"][name][unit][st]
                mine = v6c.full(reg[st], ev[st], [SEED, 63, 1, v6.DATASET[ds], direction, v6.STAGE[st], v6.UNIT[unit]])
                for k in ("top1", "rank_auc"):
                    if not abs(mine[k] - ref[k]) < v6c.CHECK_TOL:
                        raise SystemExit(f"{name} {unit} {st} {k}: differs from v6c's")
                if mine["rank_n"] != ref["rank_n"] or mine["eer"] != ref["eer"]:
                    raise SystemExit(f"{name} {unit} {st}: Rank-n or EER differs from v6c's")
            stages = [st for st in STAGE_B if st in reg] + (["DTW_pq"] if unit == "sentence" and with_new else [])
            for st in stages:
                key = [SEED, 64, 1, v6.DATASET[ds], direction, STAGE_B[st], v6.UNIT[unit]]
                if st == "DTW_pq":
                    d = dtw_matrix([np.column_stack([t.p, t.q]) for t in tiles[(ds, a)][unit]],
                                   [np.column_stack([t.p, t.q]) for t in tiles[(ds, b)][unit]])
                    drop = []
                else:
                    d, drop = stage_vectors(reg[st], ev[st])
                if d is None:
                    cell[st] = {"constant": True, "dropped_features": drop}
                    continue
                cell[st], raucs[st], idxs[st] = full_d(d, key)
                cell[st]["dropped_features"] = drop
            cell["paired"] = {}
            for x, y in PAIRED:
                if x in raucs and y in raucs:
                    diff = raucs[x] - raucs[y]
                    cell["paired"][f"{x} - {y}"] = {"rank_auc_difference": float(diff.mean()),
                                                    "ci": v6.ci(diff, idxs[x])}
            cell["tiles_per_part"] = {p: v6d.five([t.n for t in tiles[(ds, p)][unit]]) for p in (a, b)}
            rec["count"][name][unit] = cell
    rec["predictions"] = predict_b(rec["count"])
    return rec


def predict_b(cnt: dict) -> dict:
    c = cnt["kupa_copy_to_essay"]["sentence"]
    ra = {st: c[st]["rank_auc"] for st in c if isinstance(c[st], dict) and "rank_auc" in c[st]}
    s1, s2 = ra["S1"], ra["S2"]
    nt = [cnt[nm][u]["NT"] for nm in ("kupa_copy_to_essay", "kupa_essay_to_copy") for u in v6.UNIT]

    def ci(x):  # a constant stage has no rates: it is not above 0.60 (its rank_auc would be 0.5)
        return x.get("rank_auc_ci", [0.5, 0.5])

    return {"P5": abs(ra["IMG5_pq"] - s1) <= 0.03 and abs(ra["IMG5_df"] - s2) <= 0.03,
            "P6": all(ci(c[st])[1] < 0.60 for st in ("SEQ", "IMG5_centred")),
            "P7": all(ci(x)[1] < 0.60 for x in nt[0::2]) and all(ci(x)[0] > 0.60 for x in nt[1::2]),
            "P8": c["IMG5_pq"]["rank_n"]["1"]["rate"] < 0.05,
            "P10": abs(ra["IMG5_pq_scrambled"] - ra["IMG5_pq"]) <= 0.03,
            "P11": ra["DTW_pq"] < s1}


# ================================================================ run


def main_cells(rec: dict) -> dict:
    out = {}
    for ds, task in MAIN_A:
        nm = f"{ds}_{task}"
        c = rec["a"]["count"][nm]["H"]["raw"]
        for x in ("q", "p"):
            out[f"a | {nm} | r_{x}"] = {k: c[x][k] for k in ("mean_r", "mean_r_ci", "median_R", "median_R_ci",
                                                             "share_r_at_or_above_0.975", "share_r_at_or_below_0.025")
                                        if k in c[x]}
        out[f"a | {nm} | r_p - r_q"] = c["r_p_minus_r_q"]
        out[f"a | {nm} | gathered r_q"] = {k: rec["a"]["count"][nm]["gathered"]["q"][k] for k in ("mean_r",
                                                                                                    "mean_r_ci")}
    c = rec["b"]["count"]["kupa_copy_to_essay"]["sentence"]
    for st in ("S1", "S2", "SEQ", "IMG5_pq", "IMG5_df", "IMG5_centred", "IMG5_pq_scrambled", "DTW_pq", "NT",
               "IMG5_pq_NT"):
        out[f"b | kupa_copy_to_essay | sentence | {st}"] = (
            {"rank_auc": c[st]["rank_auc"], "rank_auc_ci": c[st]["rank_auc_ci"],
             "rank1": c[st]["rank_n"]["1"]["rate"], "rank20": c[st]["rank_n"]["20"]["rate"], "eer": c[st]["eer"]["eer"]}
            if "rank_auc" in c[st] else {"constant": True})
    out["b | kupa_copy_to_essay | sentence | paired"] = c["paired"]
    return out


def no_long_list(x, path: str = "") -> None:
    """No list of numbers longer than a five-number summary, except the dropped features."""
    if isinstance(x, dict):
        for k, v in x.items():
            no_long_list(v, f"{path}/{k}")
    elif isinstance(x, (list, tuple)):
        if len(x) > 5 and not path.endswith("/dropped_features"):
            raise SystemExit(f"{path}: a list of {len(x)} values in the record")
        for v in x:
            no_long_list(v, path)


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
    if c6["mapping"]["v4_conditions"]["file"] != "works/conditions/prototype_v4.json":
        raise SystemExit("v4's conditions are not the file named in the conditions")
    c4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_bytes())
    if c6["generator"] != c["generator"] or c6["units"]["rest_ms"] != 2000.0:
        raise SystemExit("the generator or rest_ms of v6's conditions differs")
    d_lwp = json.loads((ROOT / "works/conditions/prototype_v6e.json").read_text())["data"]["lwp"]
    if cm.sha256(ROOT / d_lwp["file"]) != d_lwp["sha256"]:
        raise SystemExit(f"{d_lwp['file']} does not match the SHA-256 of v6e's conditions")
    for _, x in c6["data"]["kupa"]["files"].items():
        if cm.sha256(ROOT / x["file"]) != x["sha256"]:
            raise SystemExit(f"{x['file']} does not match the SHA-256 of v6's conditions")
    if cm.sha256(ROOT / c6["data"]["aalto"]["file"]) != c6["data"]["aalto"]["sha256"]:
        raise SystemExit("the Aalto zip does not match the SHA-256 of v6's conditions")
    recs = {k: json.loads((ROOT / f"results/prototype_{k}/record.json").read_text())
            for k in ("v6", "v6b", "v6c", "v6e", "v6h")}
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    m, r = c4["mapping"], c4["relief"]
    rec = {"a": run_a(c6, m, r, recs, d_lwp)}
    rec["b"] = run_b(tiles_b(c6, m, r, recs["v6"]), recs["v6c"])
    rec = {"main_cells": main_cells(rec), **rec}
    v6d.no_per_person(rec)
    no_long_list(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6i-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({"a": rec["a"]["rule"], "b": rec["b"]["predictions"]}))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
