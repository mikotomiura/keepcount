"""Prototype v6j: three computations that close readings of v6i: (a) positive controls of a concentration of rest in
particular sentences under the four nulls of v6i's reading (H and C, raw and detrended), (b) DTW with the tiles of each
record in random order, (c) IMG5, SEQ and DTW under the registration set's scale (decision 9 of
.steering/20261101-paper-a-draft; decisions 1 and 2 of .steering/20261101-v6j-controls-under-drift).

Usage:  uv run python works/prototype_v6j.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6j.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  as v6i (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6j/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value or id is written.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
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
import prototype_v6c as v6c  # noqa: E402
import prototype_v6d as v6d  # noqa: E402
import prototype_v6f as v6f  # noqa: E402
import prototype_v6i as v6i  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6j.json"
OUT_DIR = ROOT / "results" / "prototype_v6j"
CODE = ["works/prototype_v6j.py", *v6i.CODE]
SEED = v6.SEED
K = v6i.K

# ---- part (a)
CONTROL = {"gathered": 1, "random_half": 2, "random_half_weak": 3, "middle_block": 4}
NULL = {"H_raw": 0, "H_detrended": 1, "C_raw": 2, "C_detrended": 3}
KEY_C, KEY_RECEIVING, KEY_DRAW = 31, 32, 33
RULE_CONTROL = "random_half"

# ---- part (b)
S = 5
STAGE_SCR = 12
SCALE = {"main": 0, "registration": 1}
REG_VECTOR = ("S1", "SEQ", "IMG5_pq", "IMG5_df", "IMG5_pq_scrambled")
CHECK_V6I = ("SEQ", "IMG5_pq", "IMG5_df", "IMG5_pq_scrambled", "DTW_pq")


# ================================================================ part (a)


def is_rest(gaps: np.ndarray, rest_ms: float) -> np.ndarray:
    return gaps >= rest_ms - ks.TOL_MS


def move_rests(gaps: np.ndarray, held: list[np.ndarray], receiving: np.ndarray, n_move: int | None, rest_ms: float,
               rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """v6f's swap with a given set of receiving tiles: each rest gap of a giving tile (giving tiles in order) swapped
    with a non-rest gap of a receiving tile drawn uniformly without replacement; only the first n_move rests if
    n_move is given. The new gaps, the rests moved and the rests left (of those meant to move)."""
    g = np.asarray(gaps, dtype=float).copy()
    rest = is_rest(g, rest_ms)
    recv = np.zeros(len(held), dtype=bool)
    recv[receiving] = True
    give = [int(j) for k, h in enumerate(held) if not recv[k] for j in h if rest[j]]
    pool = np.array([int(j) for k, h in enumerate(held) if recv[k] for j in h if not rest[j]], dtype=int)
    take = rng.permutation(pool)
    if n_move is not None:
        give = give[:n_move]
    take = take[:len(give)]
    for a, b in zip(give, take):
        g[a], g[b] = g[b], g[a]
    return g, int(take.size), len(give) - int(take.size)


def controls(s: v6.Series, key: list[int], rest_ms: float) -> dict[str, tuple[np.ndarray, int, int]]:
    """The four controls' gaps, with the rests moved and left."""
    g = np.diff(s.times)
    ranges = s.ranges("before")
    held = [np.arange(lo, hi) for lo, hi in ranges]
    n = len(held)
    tg, moved, left = v6f.gathered(s.times, held, rest_ms, np.random.default_rng([SEED, 6, 8, *key]))
    out = {"gathered": (np.diff(tg), moved, left)}
    half = np.random.default_rng([SEED, 6, KEY_RECEIVING, *key, CONTROL["random_half"]]).choice(n, n // 2,
                                                                                                 replace=False)
    m = sum(int(is_rest(g[h], rest_ms).sum()) for k, h in enumerate(held) if k not in set(half.tolist()))
    for name, n_move in (("random_half", None), ("random_half_weak", math.ceil(m / 2))):
        rng = np.random.default_rng([SEED, 6, KEY_DRAW, *key, CONTROL["random_half"]])  # 3 uses 2's draw
        out[name] = move_rests(g, held, half, n_move, rest_ms, rng)
    block = np.arange(n // 3, (2 * n) // 3)
    out["middle_block"] = move_rests(g, held, block, None, rest_ms,
                                     np.random.default_rng([SEED, 6, KEY_DRAW, *key, CONTROL["middle_block"]]))
    return out


def q_stats(gaps: np.ndarray, ranges: list, m: dict, rest_ms: float) -> tuple[float, float]:
    _, q = v6i.pq_of_gaps(gaps, ranges, m, rest_ms)
    d = v6i.dispersion(q)
    return d["raw"], d["detrended"]


def person_a(s: v6.Series, key: list[int], m: dict, rest_ms: float) -> dict:
    """Per control and null: T of the control and the K values of the null (q only)."""
    g = np.diff(s.times)
    ranges = s.ranges("before")
    h = np.empty((K, 2))
    for k in range(K):
        h[k] = q_stats(np.random.default_rng([SEED, 6, v6i.KEY_H, *key, k]).permutation(g), ranges, m, rest_ms)
    out = {"tallies": {}}
    for name, (gc, moved, left) in controls(s, key, rest_ms).items():
        t = q_stats(gc, ranges, m, rest_ms)
        c = np.empty((K, 2))
        for k in range(K):
            c[k] = q_stats(v6i.circular(gc, np.random.default_rng([SEED, 6, KEY_C, *key, CONTROL[name], k])), ranges,
                           m, rest_ms)
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(c))):
            raise SystemExit("a dispersion that is not finite")
        out[name] = {"H_raw": (t[0], h[:, 0]), "H_detrended": (t[1], h[:, 1]), "C_raw": (t[0], c[:, 0]),
                     "C_detrended": (t[1], c[:, 1])}
        out["tallies"][name] = (moved, left)
    if not np.all(np.isfinite(h)):
        raise SystemExit("a dispersion that is not finite")
    return out


def per_person(pa: dict) -> dict:
    out = {}
    for name in CONTROL:
        for null in NULL:
            t, nv = pa[name][null]
            med = float(np.median(nv))
            out[(name, null)] = {"r": v6i.mid_rank(t, nv), "R": t / med if med > 0 else None,
                                 "degenerate": bool(np.all(nv == t))}
    return out


def count_a(pp: list[dict], tallies: list[dict], ds: str, task: str) -> dict:
    n = len(pp)
    out: dict = {"persons": n}
    for null, ni in NULL.items():
        idx = np.random.default_rng([SEED, 81, v6f.DATASET[ds], v6f.TASK[task], ni]).integers(0, n, (v6.N_BOOT, n))
        for name in CONTROL:
            rs = np.array([p[(name, null)]["r"] for p in pp])
            out.setdefault(name, {})[null] = v6i.r_cell(rs, [p[(name, null)]["R"] for p in pp],
                                                        np.array([p[(name, null)]["degenerate"] for p in pp]), idx)
    out["moved_and_left"] = {name: {"moved": v6d.five([t[name][0] for t in tallies]),
                                    "left": v6d.five([t[name][1] for t in tallies])} for name in CONTROL}
    return out


def check_gathered(pp: list[dict], rec_vi: dict, ds: str, task: str) -> None:
    """gathered against H_raw recomputed with v6i's key must equal v6i's record."""
    n = len(pp)
    idx = np.random.default_rng([SEED, 80, v6f.DATASET[ds], v6f.TASK[task], v6i.STAT["raw"]]).integers(
        0, n, (v6.N_BOOT, n))
    rs = np.array([p[("gathered", "H_raw")]["r"] for p in pp])
    mine = v6i.r_cell(rs, [p[("gathered", "H_raw")]["R"] for p in pp], np.zeros(n, dtype=bool), idx)
    ref = rec_vi["a"]["count"][f"{ds}_{task}"]["gathered"]["q"]
    for k in ("persons", "mean_r", "mean_r_ci", "share_r_at_or_above_0.975", "share_r_at_or_below_0.025", "median_R",
              "median_R_ci"):
        if mine[k] != ref[k]:
            raise SystemExit(f"{ds} {task}: gathered against H differs from v6i's ({k})")


def evaluate_a(cnt: dict) -> dict:
    names = [f"{ds}_{task}" for ds, task in v6i.MAIN_A]
    seen = {name: {nm: {null: cnt[nm][name][null]["mean_r_ci"][0] > 0.5 for null in NULL} for nm in names}
            for name in CONTROL}
    reading = {nm: {null: ("sensitive to a concentration in particular sentences of this strength"
                           if seen[RULE_CONTROL][nm][null] else "not evaluable") for null in NULL} for nm in names}
    pred = {"P1": all(seen["random_half"][nm][nl] for nm in names for nl in NULL),
            "P2": all(seen["gathered"][nm][nl] for nm in names for nl in ("H_detrended", "C_raw")),
            "P3": any(seen["middle_block"][nm]["H_raw"] and seen["middle_block"][nm]["C_raw"]
                      and not seen["middle_block"][nm]["H_detrended"] for nm in names)}
    return {"control_seen": seen, "reading_of_rest": reading, "predictions": pred}


def run_a(c6: dict, m: dict, r: dict, recs: dict, d_lwp: dict) -> dict:
    rest_ms = c6["units"]["rest_ms"]
    rec: dict = {"persons": {}, "count": {}}
    for ds, task in v6i.MAIN_A:
        name = f"{ds}_{task}"
        if ds == "kupa":
            chosen, series, place = v6i.kupa_persons_a(c6, recs)
        else:
            chosen, series, place = v6i.lwp_persons_a(ROOT / d_lwp["file"], recs)
        ids = hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest()
        if ids != recs["v6i"]["a"]["persons"][ds]["ids_sha256"]:
            raise SystemExit(f"{name}: the drawn persons differ from v6i's")
        pp, tallies = [], []
        for pid in chosen:
            s = series[pid]
            if v6i.eligible_a(s.ranges("before")) is not None:
                continue
            pa = person_a(s, [v6f.DATASET[ds], v6f.TASK[task], place[pid]], m, rest_ms)
            tallies.append(pa.pop("tallies"))
            pp.append(per_person(pa))
        if len(pp) != recs["v6i"]["a"]["count"][name]["persons"]:
            raise SystemExit(f"{name}: {len(pp)} persons, not v6i's")
        check_gathered(pp, recs["v6i"], ds, task)
        rec["persons"][ds] = {"n": len(pp), "drawn_ids_sha256": ids}
        rec["count"][name] = count_a(pp, tallies, ds, task)
        del series, pp
    rec["rule"] = evaluate_a(rec["count"])
    return rec


# ================================================================ part (b)


def scrambled_seq(t: v6.Tiles, rng: np.random.Generator) -> np.ndarray:
    """The (p, q) of the tiles in random order, each tile's pair kept together."""
    o = rng.permutation(t.n)
    return np.column_stack([t.p[o], t.q[o]])


def dtw_registration(reg: list[np.ndarray], ev: list[np.ndarray]) -> tuple[np.ndarray | None, list[int]]:
    """DTW with each number of both sets z-scored by the registration set's mean and standard deviation over all its
    tiles; a number constant (ptp 0) in either set dropped from both."""
    span = [np.ptp(np.concatenate(x), axis=0) for x in (reg, ev)]
    drop = [j for j in range(span[0].size) if span[0][j] == 0 or span[1][j] == 0]
    keep = [j for j in range(span[0].size) if j not in drop]
    if not keep:
        return None, drop
    allr = np.concatenate(reg)[:, keep]
    mu, sd = allr.mean(axis=0), allr.std(axis=0)
    rz = [(x[:, keep] - mu) / sd for x in reg]
    ez = [(x[:, keep] - mu) / sd for x in ev]
    return np.array([v6i.dtw(e, rz) for e in ez]), drop


def vec_distance(g: np.ndarray, e: np.ndarray, scale: str) -> tuple[np.ndarray | None, list[int]]:
    how = "main" if scale == "main" else "registration_scale"
    gs, es, drop = v6.standardise(g, e, how)
    if gs.shape[1] == 0:
        return None, drop
    return v6.distances(es, gs), drop


def key_b(ds: str, direction: int, stage: int, unit: str, scale: str) -> list[int]:
    return [SEED, 65, 1, v6.DATASET[ds], direction, stage, v6.UNIT[unit], SCALE[scale]]


def check_registration(reg: dict, ev: dict, rec_v6: dict, name: str, unit: str) -> None:
    """S1 and S2 under the registration scale must equal v6's record."""
    for st in ("S1", "S2"):
        d, _ = vec_distance(reg[st], ev[st], "registration")
        _, rr = v6.per_person(d)
        if not abs(float(rr.mean()) - rec_v6["count_2_person"][name][unit]["registration_scale"][st]["rank_auc"]) \
                < v6c.CHECK_TOL:
            raise SystemExit(f"{name} {unit} {st}: the registration scale differs from v6's")


def check_v6i(cell: dict, ref: dict, where: str) -> None:
    for k in ("top1", "rank_auc", "rank_auc_ci", "rank_n", "eer"):
        if cell[k] != ref[k]:
            raise SystemExit(f"{where}: {k} differs from v6i's")


def run_b(tiles: dict, rec_v6: dict, rec_vi: dict) -> dict:
    out: dict = {"count": {}, "checked_registration_s1_s2": {}}
    for name, (ds, a, b, direction) in v6c.DIRECTIONS.items():
        out["count"][name] = {}
        for unit in v6.UNIT:
            if ds == "aalto" and unit == "window_10s":  # no new stages (v6i); only v6's check of S1 and S2 runs here
                reg = v6i.stages_of(tiles[(ds, a)][unit], ds, a, unit, False)
                ev = v6i.stages_of(tiles[(ds, b)][unit], ds, b, unit, False)
                check_registration(reg, ev, rec_v6, name, unit)
                out["checked_registration_s1_s2"][f"{name} {unit}"] = "equal to v6's record"
                continue
            reg = v6i.stages_of(tiles[(ds, a)][unit], ds, a, unit, True)
            ev = v6i.stages_of(tiles[(ds, b)][unit], ds, b, unit, True)
            ref = rec_vi["b"]["count"][name][unit]
            cell, rauc, idx = {"main": {}, "registration": {}}, {}, {}
            # checks: v6i's main stages with v6i's keys, v6's S1 and S2 under the registration scale
            for st in CHECK_V6I:
                if st == "DTW_pq" and unit != "sentence":
                    continue
                if st == "DTW_pq":
                    d, _ = v6i.dtw_matrix([np.column_stack([t.p, t.q]) for t in tiles[(ds, a)][unit]],
                                          [np.column_stack([t.p, t.q]) for t in tiles[(ds, b)][unit]])
                else:
                    d, _ = vec_distance(reg[st], ev[st], "main")
                if d is None:
                    raise SystemExit(f"{name} {unit} {st}: constant under the main scale, unlike v6i")
                res, _, _ = v6i.full_d(d, [SEED, 64, 1, v6.DATASET[ds], direction, v6i.STAGE_B[st], v6.UNIT[unit]])
                check_v6i(res, ref[st], f"{name} {unit} {st}")
                # reported with v6j's own key (the conditions' match), on the same distances
                res, rr, ii = v6i.full_d(d, key_b(ds, direction, v6i.STAGE_B[st], unit, "main"))
                cell["main"][st], rauc[("main", st)], idx[("main", st)] = res, rr, ii
            check_registration(reg, ev, rec_v6, name, unit)
            out["checked_registration_s1_s2"][f"{name} {unit}"] = "equal to v6's record"
            d, _ = vec_distance(reg["S1"], ev["S1"], "main")
            _, rauc[("main", "S1")] = v6.per_person(d)
            # new: registration scale of the vectors
            for st in REG_VECTOR:
                d, drop = vec_distance(reg[st], ev[st], "registration")
                if d is None:
                    cell["registration"][st] = {"constant": True, "dropped_features": drop}
                    continue
                res, rr, ii = v6i.full_d(d, key_b(ds, direction, v6i.STAGE_B[st], unit, "registration"))
                res["dropped_features"] = drop
                cell["registration"][st], rauc[("registration", st)], idx[("registration", st)] = res, rr, ii
            if unit == "sentence":
                seqs = {p: [np.column_stack([t.p, t.q]) for t in tiles[(ds, p)][unit]] for p in (a, b)}
                scr = {p: [[scrambled_seq(t, np.random.default_rng([SEED, 65, 3, v6.DATASET[ds], v6i.PART[p],
                                                                    v6.UNIT[unit], place, s]))
                            for place, t in enumerate(tiles[(ds, p)][unit])] for s in range(S)] for p in (a, b)}
                for scale, fn in (("main", lambda r_, e_: v6i.dtw_matrix(r_, e_)),
                                  ("registration", dtw_registration)):
                    if scale == "registration":
                        d, drop = fn(seqs[a], seqs[b])
                        if d is None:
                            raise SystemExit(f"{name}: every number of DTW is constant under the registration scale")
                        res, rr, ii = v6i.full_d(d, key_b(ds, direction, v6i.STAGE_B["DTW_pq"], unit, scale))
                        res["dropped_features"] = drop
                        cell[scale]["DTW_pq"], rauc[(scale, "DTW_pq")], idx[(scale, "DTW_pq")] = res, rr, ii
                    draws, rs = [], []
                    for s in range(S):
                        d, drop = fn(scr[a][s], scr[b][s])
                        if d is None:
                            raise SystemExit(f"{name}: every number of a scrambled DTW is constant")
                        res, rr, ii = v6i.full_d(d, key_b(ds, direction, STAGE_SCR, unit, scale))
                        res["dropped_features"] = drop
                        draws.append(res)
                        rs.append(rr)
                    mean_rr = np.mean(rs, axis=0)
                    cell[scale]["DTW_pq_scrambled"] = {
                        "draws": draws, "rank_auc_mean_over_draws": float(mean_rr.mean()),
                        "rank_auc_range_over_draws": [min(x["rank_auc"] for x in draws),
                                                      max(x["rank_auc"] for x in draws)]}
                    rauc[(scale, "DTW_pq_scrambled")], idx[(scale, "DTW_pq_scrambled")] = mean_rr, ii
            cell["paired"] = paired(rauc, idx, unit)
            out["count"][name][unit] = cell
    out["predictions"] = predict_b(out["count"])
    return out


def paired(rauc: dict, idx: dict, unit: str) -> dict:
    """x - y on the same evaluation persons, with the resamples of y (the later-named)."""
    pairs = [(("main", "DTW_pq"), ("main", "DTW_pq_scrambled")),
             (("registration", "IMG5_pq"), ("registration", "S1")),
             (("registration", "DTW_pq"), ("registration", "S1")),
             (("main", "S1"), ("registration", "S1")),
             (("main", "IMG5_pq"), ("registration", "IMG5_pq")),
             (("main", "DTW_pq"), ("registration", "DTW_pq"))]
    out = {}
    for x, y in pairs:
        if unit != "sentence" and {x[1], y[1]} & {"DTW_pq", "DTW_pq_scrambled"}:
            continue  # DTW is counted in sentences only
        name = f"{x[0]} {x[1]} - {y[0]} {y[1]}"
        if not (x in rauc and y in rauc and y in idx):
            out[name] = {"constant": True}
            continue
        diff = rauc[x] - rauc[y]
        out[name] = {"rank_auc_difference": float(diff.mean()), "ci": v6.ci(diff, idx[y])}
    return out


def predict_b(cnt: dict) -> dict:
    c = cnt["kupa_copy_to_essay"]["sentence"]
    p4 = abs(c["main"]["DTW_pq_scrambled"]["rank_auc_mean_over_draws"] - c["main"]["DTW_pq"]["rank_auc"]) <= 0.03
    falls = []
    for d in ("kupa_copy_to_essay", "kupa_essay_to_copy"):
        x = cnt[d]["sentence"]["paired"]
        falls += [x[f"main {st} - registration {st}"]["rank_auc_difference"] for st in ("IMG5_pq", "DTW_pq")]
    return {"P4": p4, "P5": all(0.05 <= f <= 0.10 for f in falls)}


# ================================================================ run


def main_cells(rec: dict) -> dict:
    out = {}
    for ds, task in v6i.MAIN_A:
        nm = f"{ds}_{task}"
        for name in CONTROL:
            out[f"a | {nm} | {name}"] = {null: [round(rec["a"]["count"][nm][name][null]["mean_r"], 3),
                                                [round(x, 3) for x in rec["a"]["count"][nm][name][null]["mean_r_ci"]]]
                                         for null in NULL}
    c = rec["b"]["count"]["kupa_copy_to_essay"]["sentence"]
    out["b | kupa_copy_to_essay | sentence | main DTW_pq"] = c["main"]["DTW_pq"]["rank_auc"]
    out["b | kupa_copy_to_essay | sentence | main DTW_pq_scrambled"] = {
        k: c["main"]["DTW_pq_scrambled"][k] for k in ("rank_auc_mean_over_draws", "rank_auc_range_over_draws")}
    for st in ("S1", "IMG5_pq", "DTW_pq"):
        out[f"b | kupa_copy_to_essay | sentence | registration {st}"] = c["registration"][st]["rank_auc"]
    out["b | kupa_copy_to_essay | sentence | paired"] = c["paired"]
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
    c4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_bytes())
    if c6["generator"] != c["generator"] or c6["units"]["rest_ms"] != 2000.0:
        raise SystemExit("the generator or rest_ms of v6's conditions differs")
    d_lwp = json.loads((ROOT / "works/conditions/prototype_v6e.json").read_text())["data"]["lwp"]
    for f, h in [(d_lwp["file"], d_lwp["sha256"]), (c6["data"]["aalto"]["file"], c6["data"]["aalto"]["sha256"])] + [
            (x["file"], x["sha256"]) for x in c6["data"]["kupa"]["files"].values()]:
        if cm.sha256(ROOT / f) != h:
            raise SystemExit(f"{f} does not match its SHA-256")
    recs = {k: json.loads((ROOT / f"results/prototype_{k}/record.json").read_text())
            for k in ("v6", "v6b", "v6c", "v6e", "v6h", "v6i")}
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    m, r = c4["mapping"], c4["relief"]
    rec_b = run_b(v6i.tiles_b(c6, m, r, recs["v6"]), recs["v6"], recs["v6i"])  # first: its checks are cheaper
    rec = {"a": run_a(c6, m, r, recs, d_lwp), "b": rec_b}
    rec = {"main_cells": main_cells(rec), **rec}
    v6d.no_per_person(rec)
    v6i.no_long_list(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "data_sha256": {"kupa_copy": c6["data"]["kupa"]["files"]["copy"]["sha256"],
                                         "kupa_essay": c6["data"]["kupa"]["files"]["essay"]["sha256"],
                                         "aalto": c6["data"]["aalto"]["sha256"], "lwp": d_lwp["sha256"]},
                         "choices_in_code": {
                             **v6i.CHOICES,
                             "paired_resamples": "the resamples of the later-named stage (y in x - y)",
                             "gathered_gaps": "gathered's gaps pass through v6f's cumsum and differences, so they equal "
                                              "text's multiset within rounding only (as v6i)",
                             "P5": "read in both KUPA directions (copy -> essay and essay -> copy), sentence unit",
                             "main_keys": "the main stages are checked against v6i with v6i's keys, then counted "
                                          "again on the same distances with v6j's keys for the record",
                             "middle_block_key": "the block is deterministic, so the receiving key [.., 32, .., 4] "
                                                 "named in the conditions is not drawn",
                             "weak_m": "m counts the rests held by giving tiles of random_half before any move",
                             "scrambled_bootstrap": "the five draws share one bootstrap key (stage 12)"},
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6j-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({"a": rec["a"]["rule"], "b": rec["b"]["predictions"]}))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
