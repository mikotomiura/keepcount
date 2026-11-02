"""Prototype v6f: the tile sequence (IMG) beside the image summaries, positive controls of the image stages (rests
gathered into some tiles; rests scaled by 1.5, 2 and 3) and a reference of quarters, on v6b's KUPA persons and v6e's
LWP persons (decisions 1 and 2 of .steering/20261101-v6f-image-level; decision 3 of .steering/20261101-paper-a-draft).

Usage:  uv run python works/prototype_v6f.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6f.json (committed before this script is first run), and the conditions and
            records of v6, v6b and v6e named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 (external/kupa-keys) and the LWP logs (external/kaggle-lwp); not in git, checked by
        SHA-256; LWP is not redistributed
Output: results/prototype_v6f/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value or id is written.
"""

from __future__ import annotations

import atexit
import datetime as dt
import hashlib
import json
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
import prototype_v6 as v6  # noqa: E402
import prototype_v6b as v6b  # noqa: E402
import prototype_v6d as v6d  # noqa: E402
import prototype_v6e as v6e  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6f.json"
OUT_DIR = ROOT / "results" / "prototype_v6f"
CODE = ["works/prototype_v6f.py", "works/prototype_v6e.py", "works/prototype_v6d.py", "works/prototype_v6b.py",
        "works/prototype_v6.py", "works/prototype_v5.py", "works/prototype_v4.py", "works/_common.py",
        "src/keystroke.py", "uv.lock"]
SEED = v6.SEED
DATASET = {"kupa": 0, "lwp": 2}
TASK = {"copy": 0, "essay": 1, "lwp": 0}
STAGE = {"S0": 0, "S1": 1, "S2": 2, "SEQ": 3, "LOC": 4, "IMG_pq": 7, "IMG_df": 8}
SUMMARY = ("S0", "S1", "S2", "SEQ", "LOC")
IMG = {"IMG_pq": ("p", "q"), "IMG_df": ("depth", "fineness")}
GATHERED = {"sentence": "gathered/before", "window_10s": "gathered"}
SCALES = {"rests_x1.5/before": 1.5, "rests_x2/before": 2.0, "rests_x3/before": 3.0}
OPERATIONS = {"sentence": ["inside/before", "shuffled/before", "shuffled_in_halves/before"],
              "window_10s": ["inside", "shuffled", "shuffled_in_halves"]}
QUARTER_ROW = 90
MIN_TILES_QUARTERS = 16
PREFIX = 70
MAIN = (("kupa", "essay"), ("lwp", "lwp"))
DQ_TOL = 1e-9  # a change of q below this is the rounding of times rebuilt by cumsum, not a change


# ---------------------------------------------------------------- operations


def gathered(times: np.ndarray, held: list[np.ndarray], rest_ms: float,
             rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Every rest gap held by a giving tile (k mod 4 in {2, 3}), giving tiles in order, swapped with a non-rest gap
    held by a receiving tile (k mod 4 in {0, 1}) chosen uniformly without replacement; the new times (from 0), the
    rests moved and the rests left."""
    t = ks.check_times(times)
    gaps = np.diff(t).copy()
    rest = gaps >= rest_ms - ks.TOL_MS
    give = [int(j) for k, h in enumerate(held) if k % 4 in (2, 3) for j in h if rest[j]]
    pool = np.array([int(j) for k, h in enumerate(held) if k % 4 in (0, 1) for j in h if not rest[j]], dtype=int)
    take = rng.permutation(pool)[:len(give)]
    for a, b in zip(give, take):
        gaps[a], gaps[b] = gaps[b], gaps[a]
    return np.concatenate([[0.0], np.cumsum(gaps)]), int(take.size), len(give) - int(take.size)


def scaled(times: np.ndarray, factor: float, rest_ms: float) -> tuple[np.ndarray, int]:
    """Every rest gap multiplied by factor, the others kept; the new times (from 0) and the number of rests."""
    t = ks.check_times(times)
    gaps = np.diff(t)
    rest = gaps >= rest_ms - ks.TOL_MS
    return np.concatenate([[0.0], np.cumsum(np.where(rest, gaps * factor, gaps))]), int(rest.sum())


# ---------------------------------------------------------------- IMG


def features(t: v6.Tiles, stage: str) -> np.ndarray:
    a, b = IMG[stage]
    return np.column_stack([getattr(t, a), getattr(t, b)]).astype(float)


def rms(x: np.ndarray, y: np.ndarray) -> float:
    """Root mean square per tile of the distance between two tile sequences paired by position."""
    if x.shape != y.shape or x.shape[0] == 0:
        raise SystemExit("tile sequences of different or zero length")
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))


def paired(x: np.ndarray, y: np.ndarray) -> float:
    """rms of the k-th tiles of two parts for k below the shorter length."""
    m = min(len(x), len(y))
    return rms(x[:m], y[:m])


def img_scale(first: list[np.ndarray], where: str) -> tuple[np.ndarray, np.ndarray]:
    """The standard deviation of each number over all tiles of text of the persons, and the numbers kept (sd > 0)."""
    allt = np.concatenate(first)
    if not np.all(np.isfinite(allt)):
        raise SystemExit(f"a non-finite tile number for {where}")
    sd = allt.std(axis=0)
    keep = sd > 0
    if not keep.any():
        raise SystemExit(f"every number of the tiles is constant for {where}")
    return sd, keep


def shares(own: np.ndarray, other: np.ndarray, base: np.ndarray, change: np.ndarray, between: np.ndarray,
           key: list[int], dropped: list[int]) -> dict:
    """The fields of v6b's one() from the per-person terms."""
    n = own.size
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    c_own = v6.ci(own, idx)
    pos = base > 0
    return {"persons": n, "dropped_features": dropped,
            "share_vs_own": float(own.mean()), "share_vs_own_ci": c_own, "reading": v6.reading(*c_own),
            "share_vs_another": float(other.mean()), "share_vs_another_ci": v6.ci(other, idx),
            "median_base": float(np.median(base)), "median_change": float(np.median(change)),
            "median_between": float(np.median(between)),
            "median_change_over_base": float(np.median(change[pos] / base[pos])) if pos.any() else None,
            "base_zero": int((~pos).sum())}


def win(change: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.where(change > ref, 1.0, np.where(change == ref, 0.5, 0.0))


def img_terms(first: list[np.ndarray], sel: list[np.ndarray], how: str, where: str):
    """Per person: own (1, 1/2, 0), other, base, change, between of IMG; and the dropped numbers."""
    n = len(first)
    if n < 2:
        raise SystemExit(f"fewer than 2 persons for {where}")
    sd, keep = img_scale(first, where)
    if not np.all(np.isfinite(np.concatenate(sel))):
        raise SystemExit(f"a non-finite tile number for {where}")
    zt = [x[:, keep] / sd[keep] for x in first]
    zr = [x[:, keep] / sd[keep] for x in sel]
    parts = []
    for i in range(n):
        if zr[i].shape != zt[i].shape:
            raise SystemExit(f"{where}: a row has a different number of tiles from text")
        parts.append(v6b.parts(len(zt[i]), how))
    base = np.array([paired(zt[i][A], zt[i][B]) for i, (A, B) in enumerate(parts)])
    change = np.array([(rms(zr[i][A], zt[i][A]) + rms(zr[i][B], zt[i][B])) / 2 for i, (A, B) in enumerate(parts)])
    between = np.array([np.median([paired(zt[i][parts[i][0]], zt[j][parts[j][1]]) for j in range(n) if j != i])
                        for i in range(n)])
    for a in (base, change, between):
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"a non-finite value for {where}")
    return win(change, base), win(change, between), base, change, between, [int(j) for j in np.flatnonzero(~keep)]


def one_img(first: list[np.ndarray], sel: list[np.ndarray], how: str, key: list[int]) -> tuple[dict, np.ndarray]:
    own, other, base, change, between, dropped = img_terms(first, sel, how, str(key))
    return shares(own, other, base, change, between, key, dropped), own


def own_summary(first: list, sel: list, st: str) -> np.ndarray:
    """The per-person own term of v6b's one() (its share is checked against one() where it is used)."""
    t_all = np.array([v[h][st] for v in first for h in (0, 1)])
    sd = t_all.std(axis=0)
    keep = sd > 0

    def z(x):
        return x[keep] / sd[keep]

    tA = np.array([z(v[0][st]) for v in first])
    tB = np.array([z(v[1][st]) for v in first])
    rA = np.array([z(v[0][st]) for v in sel])
    rB = np.array([z(v[1][st]) for v in sel])
    base = np.linalg.norm(tA - tB, axis=1)
    change = (np.linalg.norm(rA - tA, axis=1) + np.linalg.norm(rB - tB, axis=1)) / 2
    return win(change, base)


# ---------------------------------------------------------------- quarters (reference outside the rule)


def quarters(n: int, how: str) -> list[np.ndarray]:
    if how == "oddeven":
        return [np.arange(j, n, 4) for j in range(4)]
    h = n // 2
    a, b = h // 2, h + (n - h) // 2
    return [np.arange(0, a), np.arange(a, h), np.arange(h, b), np.arange(b, n)]


QUARTER_PAIRS = {"oddeven": (0, 1, 2, 3), "halves": (0, 2, 1, 3)}  # base0 = d(Qa, Qb), change0 = (d(Qc, Qa) + d(Qd, Qb)) / 2


def quarter_terms(vq: list[list[np.ndarray]], how: str, img: bool, where: str) -> tuple[np.ndarray, list[int]]:
    """own0 per person from the four quarter vectors (summaries) or tile arrays (IMG) of text, and the dropped
    numbers."""
    a, b, c, d = QUARTER_PAIRS[how]
    if img:
        sd, keep = img_scale([np.concatenate(q) for q in vq], where)
        z = [[x[:, keep] / sd[keep] for x in q] for q in vq]
        dist = paired
    else:
        allv = np.array([x for q in vq for x in q])
        if not np.all(np.isfinite(allv)):
            raise SystemExit(f"a non-finite value for {where}")
        sd = allv.std(axis=0)
        keep = sd > 0
        if not keep.any():
            raise SystemExit(f"every feature is constant for {where}")
        z = [[x[keep] / sd[keep] for x in q] for q in vq]

        def dist(x, y):
            return float(np.linalg.norm(x - y))
    base0 = np.array([dist(q[a], q[b]) for q in z])
    change0 = np.array([(dist(q[c], q[a]) + dist(q[d], q[b])) / 2 for q in z])
    if not (np.all(np.isfinite(base0)) and np.all(np.isfinite(change0))):
        raise SystemExit(f"a non-finite value for {where}")
    return win(change0, base0), [int(j) for j in np.flatnonzero(~keep)]


# ---------------------------------------------------------------- persons


def kupa_persons(c6: dict, rec6: dict, rec6b: dict) -> tuple[list[str], dict, dict]:
    """v6b's 200 persons (as v6b.run and v6d.run_v6b), checked against v6b's record."""
    kp = c6["data"]["kupa"]["files"]
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6b.read_kupa_b(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    place = {pid: i for i, pid in enumerate(ids_sorted)}
    order = v6.draw_order(ids_sorted, 0)

    def eligible(pid):
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

    v6_ids = [pid for pid in order[:v6b.V6_LOOKED] if eligible(pid) is not None]
    if hashlib.sha256("\n".join(sorted(v6_ids)).encode()).hexdigest() != rec6["datasets"]["kupa"]["ids_sha256"]:
        raise SystemExit("the first 217 ids of the permutation do not give v6's persons")
    chosen, series = [], {}
    for pid in order[v6b.V6_LOOKED:]:
        ser = eligible(pid)
        if ser is not None:
            chosen.append(pid)
            series[pid] = ser
            if len(chosen) == v6.N:
                break
    if hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest() != rec6b["persons"]["ids_sha256"]:
        raise SystemExit("the persons differ from v6b's")
    return chosen, series, place


def lwp_persons(path: Path, rec6e: dict) -> tuple[list[str], dict, dict]:
    """v6e's 200 persons (as v6e.main), checked against v6e's record."""
    lwp, ids, _ = v6e.read_lwp(path, None)
    chosen, series, _ = v6e.choose(lwp, ids)
    if hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest() != rec6e["persons"]["ids_sha256"]:
        raise SystemExit("the persons differ from v6e's")
    place = {pid: i for i, pid in enumerate(sorted(ids))}
    return chosen, {pid: {"lwp": series[pid]} for pid in chosen}, place


# ---------------------------------------------------------------- one dataset-task


def rows_of(unit: str) -> list[str]:
    rows = list(v6.ROWS["kupa"][unit]) + [GATHERED[unit]]
    return rows + list(SCALES) if unit == "sentence" else rows


def build(ds: str, task: str, unit: str, chosen: list[str], series: dict, place: dict, m: dict, r: dict,
          rest_ms: float) -> dict:
    """Per count person: the part vectors of every row for each baseline (kept vectors of v6b or v6e), the IMG tile
    arrays of every row, the text tiles, and the tallies."""
    vec_fn = v6b.vectors if ds == "kupa" else v6e.vectors
    rows = rows_of(unit)
    kept = v6.ROWS["kupa"][unit]
    hv = {how: {row: [] for row in rows} for how in v6b.BASELINE}
    img = {st: {row: [] for row in rows} for st in IMG}
    text_tiles: list[v6.Tiles] = []
    tal = {"moved": [], "left": [], "rests": [], **{f"q_changed_{row}": [] for row in rows if row not in kept},
           **{f"median_abs_dq_{row}": [] for row in rows if row not in kept}}
    left = {"fewer_than_8": 0, "part_without_gap": 0}
    for pid in chosen:
        s, kinds = series[pid][task]
        tt = v6.Tiles(s, unit, "before", m, r, rest_ms)
        if tt.n < v6.MIN_TILES_COUNT1:
            left["fewer_than_8"] += 1
            continue
        if any(sum(tt.held[i].size for i in part) == 0 for how in v6b.BASELINE for part in v6b.parts(tt.n, how)):
            left["part_without_gap"] += 1
            continue
        key = [DATASET[ds], TASK[task], place[pid]]
        rw, _ = v6.make_rows("kupa", s, unit, tt, key, m, r, rest_ms)
        v6.check_s0(rw, unit)
        tg, moved, nleft = gathered(s.times, tt.held, rest_ms, np.random.default_rng([SEED, 6, 8, *key]))
        rw[GATHERED[unit]] = v6.Tiles(s.with_times(tg), unit, "before", m, r, rest_ms)
        tal["moved"].append(moved)
        tal["left"].append(nleft)
        if unit == "sentence":
            for row, f in SCALES.items():
                tx, nrest = scaled(s.times, f, rest_ms)
                rw[row] = v6.Tiles(s.with_times(tx), unit, "before", m, r, rest_ms)
            tal["rests"].append(nrest)
        for row in rows:
            if rw[row].n != tt.n:
                raise SystemExit(f"{ds} {task} {unit} {row}: {rw[row].n} tiles, not {tt.n}")
            if row not in kept:
                dq = np.abs(rw[row].q - tt.q)
                tal[f"q_changed_{row}"].append(int((dq > DQ_TOL).sum()))
                tal[f"median_abs_dq_{row}"].append(float(np.median(dq)))
        lab = v6b.labels(s, kinds)
        for how in v6b.BASELINE:
            A, B = v6b.parts(tt.n, how)
            for row in rows:
                hv[how][row].append((vec_fn(rw[row], A, lab, how), vec_fn(rw[row], B, lab, how)))
        for st in IMG:
            for row in rows:
                img[st][row].append(features(rw[row], st))
        text_tiles.append(tt)
    empty = [k for k, v in tal.items() if not v and not (k == "rests" and unit != "sentence")]
    if empty:
        raise SystemExit(f"{ds} {task} {unit}: no value for {empty}")
    return {"hv": hv, "img": img, "text_tiles": text_tiles, "left_out": left,
            "tallies": {k: v6d.five(v) for k, v in tal.items() if v}}


def expected_stages(ds: str, how: str, row: str) -> set[str]:
    """The stages of a kept row in v6b's (KUPA) or v6e's (LWP) record."""
    st = {"S0", "S1", "S2", "LOC"} | ({"SEQ"} if how == "halves" else set())
    if ds == "lwp":
        st |= {"R_pos", "R_lag"}
    if row == "text/none":
        st -= {"LOC", "R_pos", "R_lag"}
    return st


def reproduce(b: dict, ds: str, task: str, unit: str, ref: dict) -> tuple[int, dict]:
    """The kept rows counted as v6b (KUPA) or v6e (LWP) must equal their records; the number of counts compared and
    the counts (share, CI, reading and medians of each kept row and stage), per baseline."""
    kept = v6.ROWS["kupa"][unit]
    n, out = 0, {}
    for how in v6b.BASELINE:
        hv = {row: b["hv"][how][row] for row in kept}
        if ds == "kupa":
            got = v6b.count(hv, kept, task, unit, how)
        else:
            got = v6e.count(hv, kept, unit, how, tuple(v6e.STAGE), 69)
        out[how] = {}
        for row in kept[1:]:
            have = {st for st, x in ref[how][row].items() if isinstance(x, dict) and "share_vs_own" in x}
            if have != expected_stages(ds, how, row):
                raise SystemExit(f"{ds} {task} {unit} {how} {row}: the record holds stages {sorted(have)}")
            out[how][row] = {}
            for st in sorted(have):
                x, y = ref[how][row][st], got[row][st]
                for f in ("persons", "share_vs_own", "share_vs_own_ci"):
                    if y[f] != x[f]:
                        raise SystemExit(f"{ds} {task} {unit} {how} {row} {st}: {f} differs from the record")
                out[how][row][st] = {k: y[k] for k in ("persons", "share_vs_own", "share_vs_own_ci", "reading",
                                                       "median_base", "median_change", "median_change_over_base")}
                n += 1
        if got["persons"] != ref[how]["persons"]:
            raise SystemExit(f"{ds} {task} {unit} {how}: count persons differ from the record")
    want = sum(len(expected_stages(ds, how, row)) for how in v6b.BASELINE for row in kept[1:])
    if n != want:
        raise SystemExit(f"{ds} {task} {unit}: {n} counts compared, not {want}")
    return n, out


def key_of(ds: str, task: str, unit: str, st: str, ri: int, how: str) -> list[int]:
    return [SEED, PREFIX, DATASET[ds], TASK[task], v6.UNIT[unit], STAGE[st], ri, v6b.BASELINE[how]]


def old_key(ds: str, task: str, unit: str, st: str, ri: int, how: str) -> list[int]:
    """The bootstrap key of a kept row and stage in v6b's (KUPA) or v6e's (LWP) count."""
    if ds == "kupa":
        return [SEED, 66, v6.TASK[task], v6.UNIT[unit], v6b.STAGE[st], ri, v6b.BASELINE[how]]
    return [SEED, 69, v6.UNIT[unit], v6e.STAGE[st], ri, v6b.BASELINE[how]]


def count(b: dict, ds: str, task: str, unit: str) -> dict:
    """New counts: summaries for the new rows, IMG for every row."""
    rows = rows_of(unit)
    kept = v6.ROWS["kupa"][unit]
    out: dict = {}
    for how in v6b.BASELINE:
        hv = b["hv"][how]
        res: dict = {}
        n_all = len(hv[rows[0]])
        for ri, row in enumerate(rows[1:], start=1):
            cell: dict = {}
            if row not in kept:
                for st in SUMMARY:
                    if st == "SEQ" and how != "halves":
                        continue
                    ok = [i for i in range(n_all) if v6b.has(hv[rows[0]][i], st) and v6b.has(hv[row][i], st)]
                    if st != "LOC" and len(ok) != n_all:
                        raise SystemExit(f"{ds} {task} {unit} {row} {st}: a vector is missing")
                    cell[st] = v6b.one([hv[rows[0]][i] for i in ok], [hv[row][i] for i in ok], st,
                                       key_of(ds, task, unit, st, ri, how))
                    cell[st]["left_out_no_vector"] = n_all - len(ok)
            for st in IMG:
                cell[st], _ = one_img(b["img"][st][rows[0]], b["img"][st][row], how,
                                      key_of(ds, task, unit, st, ri, how))
            res[row] = cell
        res["persons"] = n_all
        out[how] = res
    return out


def reference_quarters(b: dict, ds: str, task: str, unit: str) -> dict:
    """share0 on the count persons with at least 16 tiles, and share_R - share0 on the same persons."""
    rows = rows_of(unit)
    kept = v6.ROWS["kupa"][unit]
    tt = b["text_tiles"]
    sel = [i for i, t in enumerate(tt) if t.n >= MIN_TILES_QUARTERS]
    out: dict = {"persons": len(sel),
                 "keys": "share0: [20261031, 70, dataset, task, unit, stage, 90, baseline]; share_R and the CI of "
                         "share_R - share0: the key of R's count (v6b's or v6e's for the summaries of the kept rows, "
                         "prefix 70 otherwise)"}
    if len(sel) < 2:
        raise SystemExit(f"{ds} {task} {unit}: fewer than 2 persons with 16 tiles")
    for how in v6b.BASELINE:
        res: dict = {}
        for st in ("S0", "S1", "S2", *IMG):
            img = st in IMG
            if img:
                vq = [[b["img"][st][rows[0]][i][q] for q in quarters(tt[i].n, how)] for i in sel]
            else:
                vq = [[tt[i].vectors(q)[st] for q in quarters(tt[i].n, how)] for i in sel]
            where = f"{ds} {task} {unit} {how} {st} quarters"
            own0, dropped0 = quarter_terms(vq, how, img, where)
            k0 = key_of(ds, task, unit, st, QUARTER_ROW, how)
            idx0 = np.random.default_rng(k0).integers(0, len(sel), (v6.N_BOOT, len(sel)))
            c0 = v6.ci(own0, idx0)
            st_res = {"share0": float(own0.mean()), "share0_ci": c0, "dropped_features": dropped0, "paired": {}}
            if st != "S0":
                for ri, row in enumerate(rows[1:], start=1):
                    if row.startswith("text/"):
                        continue
                    if img:
                        res_r, own_r = one_img([b["img"][st][rows[0]][i] for i in sel], [b["img"][st][row][i] for i in sel],
                                           how, key_of(ds, task, unit, st, ri, how))
                        kr = key_of(ds, task, unit, st, ri, how)
                    else:
                        first = [b["hv"][how][rows[0]][i] for i in sel]
                        rsel = [b["hv"][how][row][i] for i in sel]
                        kr = old_key(ds, task, unit, st, ri, how) if row in kept else key_of(ds, task, unit, st, ri, how)
                        chk = v6b.one(first, rsel, st, kr)
                        own_r = own_summary(first, rsel, st)
                        if float(own_r.mean()) != chk["share_vs_own"]:
                            raise SystemExit(f"{where} {row}: own_summary differs from v6b's one()")
                    idx = np.random.default_rng(kr).integers(0, len(sel), (v6.N_BOOT, len(sel)))
                    st_res["paired"][row] = {"share_R": float(own_r.mean()),
                                             "dropped_features": res_r["dropped_features"] if img else chk["dropped_features"],
                                             "share_R_minus_share0": float((own_r - own0).mean()),
                                             "ci": v6.ci(own_r - own0, idx)}
            res[st] = st_res
        out[how] = res
    return out


# ---------------------------------------------------------------- rule and predictions


def evaluate(cnt: dict, quarters_main: dict) -> dict:
    """The confirmation rule and the predictions of the conditions, read mechanically from the counts."""
    def ci(c):
        name, how, row, st = c
        return cnt[name]["sentence"][how][row][st]["share_vs_own_ci"]

    img_by_how = {"oddeven": ("S1", "S2", "IMG_pq", "IMG_df"), "halves": ("S1", "S2", "SEQ", "IMG_pq", "IMG_df")}
    names = [f"{ds}_{task}" for ds, task in MAIN]
    a = [(nm, how, GATHERED["sentence"], st) for nm in names for how, sts in img_by_how.items() for st in sts]
    b = [(nm, how, row, st) for nm in names for how in v6b.BASELINE for row in OPERATIONS["sentence"] for st in IMG]
    if (len(a), len(b)) != (18, 24):
        raise SystemExit("the cells of the rule differ from the conditions")
    a_held = [c for c in a if ci(c)[0] > 0.60]
    b_held = [c for c in b if ci(c)[1] < 0.60]
    if len(a_held) == 18:
        pc = "met"
    elif len(a_held) >= 15:
        pc = "partly met"
    else:
        pc = "not met"
    if pc == "not met":
        finding = "not evaluable"
    elif pc == "met" and len(b_held) == 24:
        finding = "extended"
    elif len(b_held) >= 20:
        finding = "partly extended"
    else:
        finding = "not extended"
    sensitive = {(nm, how, st) for nm, how, _, st in a_held}
    not_sensitive = [c for c in b if (c[0], c[1], c[3]) not in sensitive]

    four = ("S1", "S2", "IMG_pq", "IMG_df")
    p2_strong = [(nm, how, row, st) for nm in names for how in v6b.BASELINE for row in ("rests_x2/before",
                 "rests_x3/before") for st in four]
    p2_weak = [(nm, how, "rests_x1.5/before", st) for nm in names for how in v6b.BASELINE for st in four]
    p3_halves = [c for c in b if c[1] == "halves"]
    p3_oddeven = [c for c in b if c[1] == "oddeven"]
    p4 = [(nm, how, st) for nm in names for how in v6b.BASELINE for st in four]
    pred = {
        "P1": len(a_held) == 18,
        "P2": all(ci(c)[0] > 0.60 for c in p2_strong) and any(not ci(c)[0] > 0.60 for c in p2_weak),
        "P3": all(ci(c)[1] < 0.60 for c in p3_halves) and any(not ci(c)[1] < 0.60 for c in p3_oddeven),
        "P4": all(0.40 <= quarters_main[nm][how][st]["share0"] <= 0.60 for nm, how, st in p4),
    }

    def names_of(cells):
        return {" | ".join(c): "not held" for c in cells}

    return {"a_held": len(a_held), "a_of": 18, "positive_control": pc,
            "b_held": len(b_held), "b_of": 24, "finding_1_tile_sequence": finding,
            "cells_not_holding": {"a": names_of([c for c in a if c not in a_held]),
                                  "b": names_of([c for c in b if c not in b_held])},
            "b_cells_not_sensitive": {" | ".join(c): "not sensitive" for c in not_sensitive},
            "predictions": pred,
            "prediction_cells": {"P2_strong_held": sum(ci(c)[0] > 0.60 for c in p2_strong), "P2_strong_of": 32,
                                 "P2_weak_above": sum(ci(c)[0] > 0.60 for c in p2_weak), "P2_weak_of": 16,
                                 "P3_halves_below": sum(ci(c)[1] < 0.60 for c in p3_halves), "P3_halves_of": 12,
                                 "P3_oddeven_below": sum(ci(c)[1] < 0.60 for c in p3_oddeven), "P3_oddeven_of": 12,
                                 "P4_within": sum(0.40 <= quarters_main[nm][how][st]["share0"] <= 0.60
                                                  for nm, how, st in p4), "P4_of": 16}}


# ---------------------------------------------------------------- run


def run(c: dict, c6: dict, m: dict, r: dict, recs: dict) -> dict:
    rest_ms = c6["units"]["rest_ms"]
    d_lwp = json.loads((ROOT / "works/conditions/prototype_v6e.json").read_text())["data"]["lwp"]
    if cm.sha256(ROOT / d_lwp["file"]) != d_lwp["sha256"]:
        raise SystemExit(f"{d_lwp['file']} does not match the SHA-256 of v6e's conditions")
    for t, x in c6["data"]["kupa"]["files"].items():
        if cm.sha256(ROOT / x["file"]) != x["sha256"]:
            raise SystemExit(f"{x['file']} does not match the SHA-256 of v6's conditions")
    rec: dict = {"persons": {}, "reproduced_counts": {}, "count": {}, "left_out": {}, "tallies": {},
                 "reference_quarters": {}}
    plan = [("kupa", ("essay", "copy")), ("lwp", ("lwp",))]
    for ds, tasks in plan:
        if ds == "kupa":
            chosen, series, place = kupa_persons(c6, recs["v6"], recs["v6b"])
        else:
            chosen, series, place = lwp_persons(ROOT / d_lwp["file"], recs["v6e"])
        rec["persons"][ds] = {"n": len(chosen), "ids_sha256": hashlib.sha256("\n".join(sorted(chosen)).encode())
                              .hexdigest()}
        for task in tasks:
            name = f"{ds}_{task}"
            rec["count"][name], rec["left_out"][name], rec["tallies"][name] = {}, {}, {}
            for unit in v6.UNIT:
                b = build(ds, task, unit, chosen, series, place, m, r, rest_ms)
                ref = recs["v6b"]["count"][f"{task}_{unit}"] if ds == "kupa" else recs["v6e"]["count"][unit]
                n_rep, kept_counts = reproduce(b, ds, task, unit, ref)
                rec["reproduced_counts"][f"{name}_{unit}"] = n_rep
                rec["count"][name][unit] = count(b, ds, task, unit)
                for how, rows in kept_counts.items():
                    for row, sts in rows.items():
                        rec["count"][name][unit][how][row].update(sts)
                rec["left_out"][name][unit] = b["left_out"]
                rec["tallies"][name][unit] = b["tallies"]
                if (ds, task) in MAIN and unit == "sentence":
                    rec["reference_quarters"][name] = reference_quarters(b, ds, task, unit)
                del b
        del series
    rec["rule"] = evaluate(rec["count"], rec["reference_quarters"])
    return {"main_cells": main_cells(rec), **rec}


def main_cells(rec: dict) -> dict:
    """The main cells of the conditions, in their order (copies of the counts below)."""
    out: dict = {}
    for how in ("oddeven", "halves"):
        for ds, task in MAIN:
            cell = rec["count"][f"{ds}_{task}"]["sentence"][how]
            img = ("S1", "S2", "SEQ", *IMG) if how == "halves" else ("S1", "S2", *IMG)
            out[f"{ds}_{task} | {how}"] = {
                row: {st: {k: cell[row][st][k] for k in ("share_vs_own", "share_vs_own_ci")}
                      for st in (img if row in (GATHERED["sentence"], *SCALES) else IMG)}
                for row in [GATHERED["sentence"], *SCALES, *OPERATIONS["sentence"]]}
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
        raise SystemExit("v4's conditions are not the file named in v6f's conditions")
    c4 = json.loads((ROOT / c6["mapping"]["v4_conditions"]["file"]).read_bytes())
    if c6["generator"] != c["generator"] or c6["units"]["rest_ms"] != 2000.0:
        raise SystemExit("the generator or rest_ms of v6's conditions differs")
    recs = {"v6": json.loads((ROOT / "results/prototype_v6/record.json").read_text()),
            "v6b": json.loads((ROOT / "results/prototype_v6b/record.json").read_text()),
            "v6e": json.loads((ROOT / "results/prototype_v6e/record.json").read_text())}
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], "results/prototype_v6/record.json", *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    rec = run(c, c6, c4["mapping"], c4["relief"], recs)
    v6d.no_per_person(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "v6_record_sha256": cm.sha256(ROOT / "results/prototype_v6/record.json"),
                         "data_sha256": {**{t: x["sha256"] for t, x in c6["data"]["kupa"]["files"].items()},
                                         "lwp": json.loads((ROOT / "works/conditions/prototype_v6e.json")
                                                           .read_text())["data"]["lwp"]["sha256"]},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6f-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    rule = rec["rule"]
    print(json.dumps({k: rule[k] for k in ("a_held", "positive_control", "b_held", "finding_1_tile_sequence",
                                           "predictions")}))
    for ds, task in MAIN:
        cell = rec["count"][f"{ds}_{task}"]["sentence"]["oddeven"]
        print(ds, json.dumps({row: {st: round(cell[row][st]["share_vs_own"], 3) for st in ("S1", "S2", *IMG)
                                    if st in cell[row]}
                              for row in [GATHERED["sentence"], *SCALES, *OPERATIONS["sentence"]]}))


if __name__ == "__main__":
    main()
