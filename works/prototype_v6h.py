"""Prototype v6h: what the tile sequence's reading in v6g carries: a null of exchangeable gaps (homogeneous series H,
5 per person) under v6g's yardstick, operations that move only the place of the rests, only their lengths, only the
short gaps or only the order of the sentence tiles, and IMG split into p and q (decisions 1 and 2 of
.steering/20261101-v6h-what-tiles-carry; decision 4 of .steering/20261101-paper-a-draft).

Usage:  uv run python works/prototype_v6h.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6h.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-2 and the LWP logs (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6h/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value or id is written.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import keystroke as ks  # noqa: E402
import prototype_v6 as v6  # noqa: E402
import prototype_v6b as v6b  # noqa: E402
import prototype_v6d as v6d  # noqa: E402
import prototype_v6f as v6f  # noqa: E402
import prototype_v6g as v6g  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6h.json"
OUT_DIR = ROOT / "results" / "prototype_v6h"
CODE = ["works/prototype_v6h.py", "works/prototype_v6g.py", "works/prototype_v6f.py", "works/prototype_v6e.py",
        "works/prototype_v6d.py", "works/prototype_v6b.py", "works/prototype_v6.py", "works/prototype_v5.py",
        "works/prototype_v4.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
SEED = v6.SEED
UNIT = "sentence"
B = v6g.B
M = 5
STAGE = {"IMG_pq": 7, "IMG_df": 8, "IMG_p": 9, "IMG_q": 10}
NUMBERS = {"IMG_pq": ("p", "q"), "IMG_df": ("depth", "fineness"), "IMG_p": ("p",), "IMG_q": ("q",)}
ROWS = ["shuffled", "tiles_permuted", "rests_moved", "short_shuffled", "gathered", "rests_swapped"]
ROW_INDEX = {row: i + 1 for i, row in enumerate(ROWS)}
TEXT_ONLY = ("gathered",)
MAIN = (("kupa", "essay"), ("lwp", "lwp"))
KEY_H, KEY_BASE_H = 16, 21
NEAR_ZERO = v6g.NEAR_ZERO  # bases between 0 and this: the rounding of times rebuilt by cumsum (KUPA), counted beside 0
KEY_TEXT = {"tiles_permuted": 13, "rests_moved": 14, "short_shuffled": 15, "rests_swapped": 22}
KEY_ON_H = {"shuffled": 17, "tiles_permuted": 18, "rests_moved": 19, "short_shuffled": 20, "rests_swapped": 23}


# ---------------------------------------------------------------- operations on the gaps


def gaps_of(times: np.ndarray) -> np.ndarray:
    return np.diff(ks.check_times(times))


def from_gaps(gaps: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(gaps)])


def is_rest(gaps: np.ndarray, rest_ms: float) -> np.ndarray:
    return gaps >= rest_ms - ks.TOL_MS


def shuffle_all(times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """All gaps permuted uniformly (H, and shuffled on H); the new gaps."""
    return rng.permutation(gaps_of(times))


def rests_moved(times: np.ndarray, rest_ms: float, rng: np.random.Generator) -> np.ndarray:
    """The positions of the rests drawn again uniformly without replacement among all positions; the rest values put
    there in their original order, the short gaps in the other positions in their original order; the new gaps."""
    g = gaps_of(times)
    rest = is_rest(g, rest_ms)
    new = np.empty_like(g)
    pos = np.zeros(g.size, dtype=bool)
    pos[rng.choice(g.size, size=int(rest.sum()), replace=False)] = True
    new[pos] = g[rest]
    new[~pos] = g[~rest]
    return new


def rests_swapped(times: np.ndarray, rest_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Positions of the rests and the short gaps kept; the rest values permuted among the rest positions."""
    g = gaps_of(times).copy()
    rest = is_rest(g, rest_ms)
    g[rest] = rng.permutation(g[rest])
    return g


def short_shuffled(times: np.ndarray, rest_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Rests kept in position and value; the short gaps permuted among the short positions."""
    g = gaps_of(times).copy()
    short = ~is_rest(g, rest_ms)
    g[short] = rng.permutation(g[short])
    return g


def tiles_permuted(times: np.ndarray, held: list[np.ndarray], rng: np.random.Generator):
    """The gaps of each tile kept together in order and the tiles' order permuted; the new times and the sentence
    ranges of the moved blocks (the tiles follow their gaps); the new gaps and the ranges."""
    g = gaps_of(times)
    if not np.array_equal(np.concatenate(held), np.arange(g.size)):
        raise SystemExit("the tiles do not hold every gap in order; tiles_permuted is not defined")
    order = rng.permutation(len(held))
    ranges, at = [], 0
    for k in order:
        ranges.append((at, at + held[k].size))
        at += held[k].size
    return np.concatenate([g[held[k]] for k in order]), ranges


# ---------------------------------------------------------------- observables and counts


def observables(t: v6.Tiles) -> dict:
    return {st: np.column_stack([getattr(t, a) for a in NUMBERS[st]]).astype(float) for st in STAGE}


def own_of(z, text: np.ndarray, base: list, change: np.ndarray):
    """own over the B draws (1 change > base_b, 1/2 equal, 0), the per-draw strict indicators (change > base_b), the
    median change and base, whether a base is 0 and whether a base is below NEAR_ZERO."""
    zt = z(text)
    bs = np.array([v6f.rms(z(x), zt) for x in base])
    c = v6f.rms(z(change), zt)
    if not (np.all(np.isfinite(bs)) and np.isfinite(c)):
        raise SystemExit("a non-finite distance")
    ind = v6f.win(np.full(B, c), bs)
    return (float(ind.mean()), (c > bs).astype(float), c, float(np.median(bs)), bool((bs == 0).any()),
            bool((bs < NEAR_ZERO).any()))


def count_cell(b: dict, row: str, st: str, key: list[int], where: str) -> dict:
    texts = [p["text"][st] for p in b["persons"]]
    sd, keep = v6f.img_scale(texts, where)
    if not keep.all():
        raise SystemExit(f"a number with standard deviation 0 for {where}")

    def z(x):
        x = np.asarray(x, dtype=float)[:, keep] / sd[keep]
        if not np.all(np.isfinite(x)):
            raise SystemExit(f"a non-finite standardized value for {where}")
        return x

    n = len(texts)
    own_t, own_h = np.zeros(n), np.zeros(n)
    per_draw = np.zeros((n, B))
    share_h_by = np.zeros((n, M))
    med = {"change_text": [], "base_text": [], "change_H": [], "base_H": []}
    zero_t = zero_h = near_t = near_h = 0
    for i, p in enumerate(b["persons"]):
        o, ind, c, mb, z0, n0 = own_of(z, p["text"][st], [x[st] for x in p["base"]], p["rows"][row][st])
        own_t[i], per_draw[i] = o, ind
        med["change_text"].append(c)
        med["base_text"].append(mb)
        zero_t += int(z0)
        near_t += int(n0)
        if row in TEXT_ONLY:
            continue
        cs, bs, any0, anyn = [], [], False, False
        for h, hh in enumerate(p["H"]):
            oh, _, ch, mbh, z0h, n0h = own_of(z, hh["text"][st], [x[st] for x in hh["base"]], hh["rows"][row][st])
            share_h_by[i, h] = oh
            cs.append(ch)
            bs.append(mbh)
            any0 |= z0h
            anyn |= n0h
        own_h[i] = share_h_by[i].mean()
        med["change_H"].append(float(np.median(cs)))
        med["base_H"].append(float(np.median(bs)))
        zero_h += int(any0)
        near_h += int(anyn)
    if n < 2:
        raise SystemExit(f"fewer than 2 persons for {where}")
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    out = {"persons": n, "share_text": float(own_t.mean()), "share_text_ci": v6.ci(own_t, idx),
           "per_draw_share_of_persons_min": float(per_draw.mean(axis=0).min()),
           "per_draw_share_of_persons_max": float(per_draw.mean(axis=0).max()),
           "median_change_text": v6d.five(med["change_text"]), "median_base_text": v6d.five(med["base_text"]),
           "persons_unchanged_text": int(b["unchanged"][row]), "persons_with_a_zero_base_text": zero_t,
           "persons_with_a_base_below_1e-9_text": near_t}
    if row not in TEXT_ONLY:
        delta = own_t - own_h
        by_h = share_h_by.mean(axis=0)
        out.update({"share_H": float(own_h.mean()), "share_H_ci": v6.ci(own_h, idx),
                    "mean_delta": float(delta.mean()), "mean_delta_ci": v6.ci(delta, idx),
                    "share_H_by_h_min": float(by_h.min()), "share_H_by_h_max": float(by_h.max()),
                    "median_range_of_own_H_over_h": float(np.median(share_h_by.max(axis=1) - share_h_by.min(axis=1))),
                    "median_change_H": v6d.five(med["change_H"]), "median_base_H": v6d.five(med["base_H"]),
                    "person_h_pairs_unchanged_H": int(b["unchanged_H"][row]), "person_h_pairs_of": n * M,
                    "persons_with_a_zero_base_H": zero_h, "persons_with_a_base_below_1e-9_H": near_h})
    return out


def key_of(ds: str, task: str, st: str, row: str) -> list[int]:
    return [SEED, 72, v6f.DATASET[ds], v6f.TASK[task], STAGE[st], ROW_INDEX[row]]


# ---------------------------------------------------------------- one dataset-task


def operated(s: v6.Series, tt: v6.Tiles, row: str, key: list[int], kind: str, rest_ms: float, h: int | None,
             mk) -> tuple[v6.Tiles, bool]:
    """The tiles of one operation on a series (text or H_h) and whether its gaps equal the original."""
    k = KEY_TEXT if h is None else KEY_ON_H
    rng = np.random.default_rng([SEED, 6, k[row], *key] + ([] if h is None else [h]))
    g = gaps_of(s.times)
    ranges = None
    if row == "tiles_permuted":
        new_g, ranges = tiles_permuted(s.times, tt.held, rng)
    elif row == "shuffled":
        new_g = shuffle_all(s.times, rng)
    else:
        new_g = {"rests_moved": rests_moved, "short_shuffled": short_shuffled,
                 "rests_swapped": rests_swapped}[row](s.times, rest_ms, rng)
    if np.array_equal(new_g, g):  # unchanged: the original tiles themselves, so the change is exactly 0
        return tt, True
    sx = v6.Series(from_gaps(new_g), sentence_ranges=ranges) if ranges is not None else s.with_times(from_gaps(new_g))
    new = mk(sx)
    if new.n != tt.n:
        raise SystemExit(f"{kind} {row}: {new.n} tiles, not {tt.n}")
    return new, False


def build(ds: str, task: str, chosen: list[str], series: dict, place: dict, m: dict, r: dict, rest_ms: float) -> dict:
    """Per count person (v6f's in the sentence unit): the observables of text, its 20 redraws and its rows; and of
    each H_h, its 20 redraws and its rows."""
    out = {"persons": [], "unchanged": dict.fromkeys(ROWS, 0), "unchanged_H": dict.fromkeys(ROWS, 0),
           "left_out": {"fewer_than_8": 0, "part_without_gap": 0}, "dq0": []}
    for pid in chosen:
        s, _ = series[pid][task]
        tt = v6.Tiles(s, UNIT, "before", m, r, rest_ms)
        if tt.n < v6.MIN_TILES_COUNT1:
            out["left_out"]["fewer_than_8"] += 1
            continue
        if any(sum(tt.held[i].size for i in part) == 0 for how in v6b.BASELINE for part in v6b.parts(tt.n, how)):
            out["left_out"]["part_without_gap"] += 1
            continue
        key = [v6f.DATASET[ds], v6f.TASK[task], place[pid]]

        def mk(sx):
            return v6.Tiles(sx, UNIT, "before", m, r, rest_ms)

        def redraws(sx, txt, keys):
            res = []
            for kk in keys:
                t = mk(sx.with_times(v6g.resampled(sx.times, txt.held, np.random.default_rng(kk))))
                if t.n != tt.n:
                    raise SystemExit(f"{ds} {task}: a redraw has {t.n} tiles, not {tt.n}")
                res.append(observables(t))
            return res

        rw, _ = v6.make_rows("kupa", s, UNIT, tt, key, m, r, rest_ms)
        v6.check_s0(rw, UNIT)
        rows = {"shuffled": rw["shuffled/before"]}
        if not np.array_equal(rows["shuffled"].held[-1], tt.held[-1]) or rows["shuffled"].n != tt.n:
            raise SystemExit(f"{ds} {task}: the shuffled row's tiles differ from text's")
        tg, _, _ = v6f.gathered(s.times, tt.held, rest_ms, np.random.default_rng([SEED, 6, 8, *key]))
        rows["gathered"] = mk(s.with_times(tg))
        out["unchanged"]["shuffled"] += int(np.array_equal(rows["shuffled"].gaps, tt.gaps))
        out["unchanged"]["gathered"] += int(np.array_equal(rows["gathered"].gaps, tt.gaps))
        for row in KEY_TEXT:
            rows[row], same = operated(s, tt, row, key, f"{ds} {task} text", rest_ms, None, mk)
            out["unchanged"][row] += int(same)
        base = redraws(s, tt, [[SEED, 6, v6g.KEY_BASE[False], *key, b] for b in range(B)])
        out["dq0"].append(float(np.median(np.abs(mk(s.with_times(v6g.resampled(
            s.times, tt.held, np.random.default_rng([SEED, 6, v6g.KEY_BASE[False], *key, 0])))).q - tt.q))))
        hs = []
        for h in range(M):
            sh = s.with_times(from_gaps(shuffle_all(s.times, np.random.default_rng([SEED, 6, KEY_H, *key, h]))))
            th = mk(sh)
            if th.n != tt.n:
                raise SystemExit(f"{ds} {task}: H has {th.n} tiles, not {tt.n}")
            hrows = {}
            for row in KEY_ON_H:
                t, same = operated(sh, th, row, key, f"{ds} {task} H", rest_ms, h, mk)
                hrows[row] = observables(t)
                out["unchanged_H"][row] += int(same)
            hs.append({"text": observables(th), "rows": hrows,
                       "base": redraws(sh, th, [[SEED, 6, KEY_BASE_H, *key, h, b] for b in range(B)])})
        out["persons"].append({"text": observables(tt), "base": base, "H": hs,
                               "rows": {row: observables(t) for row, t in rows.items()}})
    return out


def count(b: dict, ds: str, task: str) -> dict:
    return {row: {st: count_cell(b, row, st, key_of(ds, task, st, row), f"{ds} {task} {row} {st}") for st in STAGE}
            for row in ROWS}


# ---------------------------------------------------------------- rule and predictions


def evaluate(cnt: dict) -> dict:
    names = [f"{ds}_{task}" for ds, task in MAIN]

    def c(nm, row, st):
        return cnt[nm][row][st]

    q1 = [(nm, "shuffled", st) for nm in names for st in ("IMG_pq", "IMG_df")]
    q2 = [(nm, "rests_moved", "IMG_q") for nm in names]
    if (len(q1), len(q2)) != (4, 2):
        raise SystemExit("the cells of the rule differ from the conditions")
    lo = [c(*x)["mean_delta_ci"][0] > 0 for x in q1]
    has0 = [c(*x)["mean_delta_ci"][0] <= 0 <= c(*x)["mean_delta_ci"][1] for x in q1]
    below = [" | ".join(x) for x in q1 if c(*x)["mean_delta_ci"][1] < 0]
    if all(lo):
        v1 = "shuffling shows in the tile sequence beyond what exchangeable gaps give"
    elif all(has0):
        v1 = "a difference from exchangeable gaps is not confirmed"
    else:
        v1 = "mixed"
    held2 = [c(*x)["share_text_ci"][0] > 0.60 and c(*x)["mean_delta_ci"][0] > 0 for x in q2]
    if all(held2):
        v2 = "moving the rests to other sentences shows in the q tile sequence beyond exchangeable gaps"
    elif all(c(*x)["mean_delta_ci"][0] <= 0 <= c(*x)["mean_delta_ci"][1] for x in q2):
        v2 = "not confirmed"
    else:
        v2 = "mixed"
    pred = {"P1": all(c(*x)["share_H_ci"][0] > 0.5 for x in q1),
            "P2": all(lo),
            "P3": all(held2),
            "P4": all(not c(nm, "short_shuffled", "IMG_q")["share_text_ci"][0] > 0.60 for nm in names),
            "P5": all(c(nm, "rests_swapped", "IMG_q")["share_text"] < c(nm, "rests_moved", "IMG_q")["share_text"]
                      for nm in names)}
    def label(x):
        a, z = c(*x)["mean_delta_ci"]
        return "above 0" if a > 0 else ("below 0" if z < 0 else "contains 0")

    return {"q1": v1, "q1_cells": {" | ".join(x): label(x) for x in q1}, "q1_cells_below": below, "q2": v2,
            "q2_cells_not_held": {" | ".join(x): "not held" for x, ok in zip(q2, held2) if not ok},
            "predictions": pred}


# ---------------------------------------------------------------- run


def run(c6: dict, m: dict, r: dict, recs: dict) -> dict:
    rest_ms = c6["units"]["rest_ms"]
    d_lwp = json.loads((ROOT / "works/conditions/prototype_v6e.json").read_text())["data"]["lwp"]
    if cm.sha256(ROOT / d_lwp["file"]) != d_lwp["sha256"]:
        raise SystemExit(f"{d_lwp['file']} does not match the SHA-256 of v6e's conditions")
    for _, x in c6["data"]["kupa"]["files"].items():
        if cm.sha256(ROOT / x["file"]) != x["sha256"]:
            raise SystemExit(f"{x['file']} does not match the SHA-256 of v6's conditions")
    rec: dict = {"persons": {}, "count": {}, "left_out": {}, "tallies": {}}
    for ds, task in MAIN:
        if ds == "kupa":
            chosen, series, place = v6f.kupa_persons(c6, recs["v6"], recs["v6b"])
        else:
            chosen, series, place = v6f.lwp_persons(ROOT / d_lwp["file"], recs["v6e"])
        rec["persons"][ds] = {"n": len(chosen), "ids_sha256": hashlib.sha256("\n".join(sorted(chosen)).encode())
                              .hexdigest()}
        name = f"{ds}_{task}"
        b = build(ds, task, chosen, series, place, m, r, rest_ms)
        n_g = recs["v6g"]["count"][name]["redrawn"]["IMG_pq"]["persons"]
        if len(b["persons"]) != n_g:
            raise SystemExit(f"{name}: {len(b['persons'])} count persons, not v6g's {n_g}")
        if b["left_out"] != recs["v6g"]["left_out"][name]:
            raise SystemExit(f"{name}: the persons left out differ from v6g's")
        if v6d.five(b["dq0"]) != recs["v6g"]["tallies"][name]["median_abs_dq_draw0"]:
            raise SystemExit(f"{name}: the redraws of text differ from v6g's")
        rec["count"][name] = count(b, ds, task)
        for row, grow in (("shuffled", "shuffled/before"), ("gathered", "gathered/before")):
            for st in ("IMG_pq", "IMG_df"):
                mine, theirs = rec["count"][name][row][st], recs["v6g"]["count"][name][grow][st]
                if mine["share_text"] != theirs["share_vs_own"] or not np.isclose(
                        mine["median_base_text"]["median"], theirs["median_base"], rtol=1e-12, atol=0):
                    raise SystemExit(f"{name} {row} {st}: the count against text differs from v6g's")
        rec["left_out"][name] = b["left_out"]
        rec["tallies"][name] = {"persons_unchanged_text": b["unchanged"], "persons_unchanged_H_of_5": b["unchanged_H"]}
        del b, series
    rec["rule"] = evaluate(rec["count"])
    return {"main_cells": main_cells(rec), **rec}


def main_cells(rec: dict) -> dict:
    """In the order of the conditions."""
    order = [("shuffled", st) for st in ("IMG_pq", "IMG_df", "IMG_p", "IMG_q")]
    order += [(row, st) for row in ("rests_moved", "rests_swapped", "short_shuffled") for st in ("IMG_q", "IMG_p")]
    order += [(row, st) for row in ("tiles_permuted", "gathered") for st in ("IMG_pq", "IMG_df")]
    keys = ("share_text", "share_text_ci", "share_H", "share_H_ci", "mean_delta", "mean_delta_ci")
    out = {f"{ds}_{task} | {row} | {st}": {k: rec["count"][f"{ds}_{task}"][row][st][k] for k in keys
                                            if k in rec["count"][f"{ds}_{task}"][row][st]}
           for ds, task in MAIN for row, st in order}
    for ds, task in MAIN:
        x = rec["count"][f"{ds}_{task}"]["shuffled"]["IMG_pq"]
        out[f"{ds}_{task} | shuffled | IMG_pq | medians"] = {k: x[k]["median"] for k in (
            "median_change_text", "median_base_text", "median_change_H", "median_base_H")}
    return out


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
    recs = {k: json.loads((ROOT / f"results/prototype_{k}/record.json").read_text())
            for k in ("v6", "v6b", "v6e", "v6g")}
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], "results/prototype_v6/record.json", *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    rec = run(c6, c4["mapping"], c4["relief"], recs)
    v6d.no_per_person(rec)
    v6g.no_long_list({k: v for k, v in rec.items() if k != "rule"})
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6h-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["rule"]))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps({x: round(y, 3) for x, y in v.items() if not isinstance(y, list)}))


if __name__ == "__main__":
    main()
