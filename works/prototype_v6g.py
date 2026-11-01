"""Prototype v6g: the tile sequence (IMG) against its own sampling variation: each sentence tile re-made from its own
gaps drawn with replacement (20 draws), with a redrawn control for the symmetry of the resampler, on v6f's persons and
rows (decisions 1 and 2 of .steering/20261101-v6g-tile-yardstick; decision 4 of .steering/20261101-v6f-image-level).

Usage:  uv run python works/prototype_v6g.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6g.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and the LWP logs (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6g/record.json (written to a temporary directory beside it and renamed when complete). No
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

COND = ROOT / "works" / "conditions" / "prototype_v6g.json"
OUT_DIR = ROOT / "results" / "prototype_v6g"
CODE = ["works/prototype_v6g.py", "works/prototype_v6f.py", "works/prototype_v6e.py", "works/prototype_v6d.py",
        "works/prototype_v6b.py", "works/prototype_v6.py", "works/prototype_v5.py", "works/prototype_v4.py",
        "works/_common.py", "src/keystroke.py", "uv.lock"]
SEED = v6.SEED
UNIT = "sentence"
B = 20
STAGE = {"IMG_pq": 7, "IMG_df": 8, "S1": 1, "S2": 2}
IMG = ("IMG_pq", "IMG_df")
ROWS = ["inside/before", "shuffled/before", "shuffled_in_halves/before", "gathered/before", "rests_x1.5/before",
        "rests_x2/before", "rests_x3/before", "redrawn"]
ROW_INDEX = {row: i + 1 for i, row in enumerate(ROWS)}
SENS_OFFSET = 20
SHUFFLES = ("shuffled/before", "shuffled_in_halves/before")
MAIN = (("kupa", "essay"), ("lwp", "lwp"))
PLAN = (("kupa", "essay"), ("kupa", "copy"), ("lwp", "lwp"))
KEY_BASE, KEY_REDRAW = {False: 9, True: 11}, {False: 10, True: 12}  # index 2 of the draw's key (boundary kept: True)


# ---------------------------------------------------------------- resampling


def resampled(times: np.ndarray, held: list[np.ndarray], rng: np.random.Generator,
              keep_boundary: bool = False) -> np.ndarray:
    """Each tile's gaps drawn uniformly with replacement from its own gaps and put in its indices in the order drawn,
    tiles in order; with keep_boundary the first gap of every tile but the first (the gap before its sentence under
    the counting before) stays and only the others are drawn. New times from 0."""
    t = ks.check_times(times)
    gaps = np.diff(t)
    new = gaps.copy()
    for k, h in enumerate(held):
        idx = h[1:] if keep_boundary and k > 0 else h
        if idx.size == 0:
            continue
        new[idx] = rng.choice(gaps[idx], size=idx.size, replace=True)
    return np.concatenate([[0.0], np.cumsum(new)])


# ---------------------------------------------------------------- observables and counts


def observables(t: v6.Tiles) -> dict:
    """IMG tile arrays and the S1, S2 vectors over all tiles of one row."""
    v = t.all()
    return {"IMG_pq": v6f.features(t, "IMG_pq"), "IMG_df": v6f.features(t, "IMG_df"), "S1": v["S1"], "S2": v["S2"]}


def scale(text: list, st: str, where: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-number standard deviation over the text of the cell's persons (all tiles for IMG, the persons' vectors for
    S1 and S2), and the numbers kept."""
    if st in IMG:
        return v6f.img_scale(text, where)
    allv = np.array(text)
    if not np.all(np.isfinite(allv)):
        raise SystemExit(f"a non-finite value for {where}")
    sd = allv.std(axis=0)
    keep = sd > 0
    if not keep.any():
        raise SystemExit(f"every feature is constant for {where}")
    return sd, keep


def dist(x: np.ndarray, y: np.ndarray, st: str) -> float:
    return v6f.rms(x, y) if st in IMG else float(np.linalg.norm(x - y))


def own_terms(text: list, base: list[list], change: list, st: str, where: str):
    """Per person: own = mean over draws of 1 (change > base_b), 1/2 (equal), 0; the per-draw indicators; the median
    base over draws and the change (per draw for redrawn). base[i][b]; change[i] is one array or a list of B arrays."""
    sd, keep = scale(text, st, where)

    def z(x):
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise SystemExit(f"a non-finite value for {where}")
        return x[..., keep] / sd[keep]

    n = len(text)
    ind = np.zeros((n, B))
    med_base, med_change = np.zeros(n), np.zeros(n)
    zero_base = 0
    for i in range(n):
        zt = z(text[i])
        bs = np.array([dist(z(base[i][b]), zt, st) for b in range(B)])
        if isinstance(change[i], list):
            cs = np.array([dist(z(change[i][b]), zt, st) for b in range(B)])
        else:
            cs = np.full(B, dist(z(change[i]), zt, st))
        ind[i] = v6f.win(cs, bs)
        med_base[i], med_change[i] = np.median(bs), np.median(cs)
        zero_base += int((bs == 0).any())
    return ind, med_base, med_change, zero_base, [int(j) for j in np.flatnonzero(~keep)]


def summary(ind: np.ndarray, med_base: np.ndarray, med_change: np.ndarray, zero_base: int, dropped: list[int],
            key: list[int]) -> dict:
    own = ind.mean(axis=1)
    n = own.size
    if n < 2:
        raise SystemExit(f"fewer than 2 persons for {key}")
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    c = v6.ci(own, idx)
    per_draw = ind.mean(axis=0)
    pos = med_base > 0
    return {"persons": n, "dropped_features": dropped, "share_vs_own": float(own.mean()), "share_vs_own_ci": c,
            "reading": v6.reading(*c), "share_by_single_draw_min": float(per_draw.min()),
            "share_by_single_draw_max": float(per_draw.max()),
            "median_base": float(np.median(med_base)), "median_change": float(np.median(med_change)),
            "median_change_over_base": float(np.median(med_change[pos] / med_base[pos])) if pos.any() else None,
            "persons_with_a_zero_base": zero_base}


def key_of(ds: str, task: str, st: str, row: str, sens: bool) -> list[int]:
    return [SEED, 71, v6f.DATASET[ds], v6f.TASK[task], STAGE[st], ROW_INDEX[row] + (SENS_OFFSET if sens else 0)]


# ---------------------------------------------------------------- one dataset-task


def build(ds: str, task: str, chosen: list[str], series: dict, place: dict, m: dict, r: dict, rest_ms: float,
          sens: bool) -> dict:
    """Per count person (v6f's in the sentence unit): the observables of text, of every row, of the 20 base draws and
    of the 20 redrawn draws (and, if sens, of the draws with the boundary gap kept)."""
    out = {"text": [], "rows": {row: [] for row in ROWS if row != "redrawn"},
           "base": {False: [], True: []}, "redrawn": {False: [], True: []}, "dq0": [], "left_out": {
               "fewer_than_8": 0, "part_without_gap": 0}}
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
        rw, _ = v6.make_rows("kupa", s, UNIT, tt, key, m, r, rest_ms)
        v6.check_s0(rw, UNIT)
        tg, _, _ = v6f.gathered(s.times, tt.held, rest_ms, np.random.default_rng([SEED, 6, 8, *key]))
        rw["gathered/before"] = v6.Tiles(s.with_times(tg), UNIT, "before", m, r, rest_ms)
        for row, f in v6f.SCALES.items():
            rw[row] = v6.Tiles(s.with_times(v6f.scaled(s.times, f, rest_ms)[0]), UNIT, "before", m, r, rest_ms)

        def tiles_of(times):
            t = v6.Tiles(s.with_times(times), UNIT, "before", m, r, rest_ms)
            if t.n != tt.n:
                raise SystemExit(f"{ds} {task}: a resampled or operated row has {t.n} tiles, not {tt.n}")
            return t

        out["text"].append(observables(tt))
        for row in out["rows"]:
            if rw[row].n != tt.n:
                raise SystemExit(f"{ds} {task} {row}: {rw[row].n} tiles, not {tt.n}")
            out["rows"][row].append(observables(rw[row]))
        for kb in ((False, True) if sens else (False,)):
            bs, rd = [], []
            for b in range(B):
                tb = tiles_of(resampled(s.times, tt.held, np.random.default_rng([SEED, 6, KEY_BASE[kb], *key, b]), kb))
                tr = tiles_of(resampled(s.times, tt.held, np.random.default_rng([SEED, 6, KEY_REDRAW[kb], *key, b]),
                                        kb))
                if b == 0 and not kb:
                    out["dq0"].append(float(np.median(np.abs(tb.q - tt.q))))
                bs.append(observables(tb))
                rd.append(observables(tr))
            out["base"][kb].append(bs)
            out["redrawn"][kb].append(rd)
    return out


def count(b: dict, ds: str, task: str, sens: bool) -> dict:
    stages = IMG if sens else tuple(STAGE)
    res: dict = {}
    for row in ROWS:
        res[row] = {}
        for st in stages:
            text = [x[st] for x in b["text"]]
            base = [[d[st] for d in person] for person in b["base"][sens]]
            if row == "redrawn":
                change = [[d[st] for d in person] for person in b["redrawn"][sens]]
            else:
                change = [x[st] for x in b["rows"][row]]
            where = f"{ds} {task} {row} {st}{' boundary kept' if sens else ''}"
            res[row][st] = summary(*own_terms(text, base, change, st, where), key=key_of(ds, task, st, row, sens))
    return res


# ---------------------------------------------------------------- rule and predictions


def evaluate(cnt: dict) -> dict:
    names = [f"{ds}_{task}" for ds, task in MAIN]

    def ci(nm, row, st):
        return cnt[nm][row][st]["share_vs_own_ci"]

    def share(nm, row, st):
        return cnt[nm][row][st]["share_vs_own"]

    a = [(nm, "gathered/before", st) for nm in names for st in IMG]
    sym = [(nm, "redrawn", st) for nm in names for st in IMG]
    c = [(nm, row, st) for nm in names for row in SHUFFLES for st in IMG]
    if (len(a), len(sym), len(c)) != (4, 4, 8):
        raise SystemExit("the cells of the rule differ from the conditions")
    a_held = [x for x in a if ci(*x)[0] > 0.60]
    pc = "met" if len(a_held) == 4 else "not met"
    if any(not 0.45 <= share(*x) <= 0.55 for x in sym):
        symmetry = "failed"
    elif all(0.40 <= ci(*x)[0] and ci(*x)[1] <= 0.60 for x in sym):
        symmetry = "passed"
    else:
        symmetry = "undetermined"
    above = [x for x in c if ci(*x)[0] > 0.60]
    below = [x for x in c if ci(*x)[1] < 0.60]
    if pc != "met" or symmetry != "passed":
        verdict = "not evaluable"
    elif len(above) == 8:
        verdict = "shows the shuffles beyond its sampling variation"
    elif len(below) == 8:
        verdict = "does not show the shuffles in most persons beyond its sampling variation"
    else:
        verdict = "mixed"
    scaled_strong = [(nm, row, st) for nm in names for row in ("rests_x2/before", "rests_x3/before") for st in IMG]
    scaled_weak = [(nm, "rests_x1.5/before", st) for nm in names for st in IMG]
    pred = {"P1": len(a_held) == 4,
            "P2": all(ci(*x)[0] > 0.60 for x in scaled_strong) and any(not ci(*x)[0] > 0.60 for x in scaled_weak),
            "P3": all(0.45 <= share(*x) <= 0.55 and 0.40 <= ci(*x)[0] and ci(*x)[1] <= 0.60 for x in sym),
            "P4": len(above) == 8,
            "P5": all(cnt[nm]["inside/before"][st]["share_vs_own"] == 0.0 for nm in cnt for st in cnt[nm]["inside/before"])}
    return {"a_held": len(a_held), "a_of": 4, "positive_control": pc, "symmetry_check": symmetry,
            "shuffles_above": len(above), "shuffles_below": len(below), "shuffles_of": 8, "tile_sequence": verdict,
            "cells_not_above": {" | ".join(x): "not above" for x in c if x not in above},
            "a_cells_not_held": {" | ".join(x): "not held" for x in a if x not in a_held},
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
    rec: dict = {"persons": {}, "count": {}, "sensitivity_boundary_kept": {}, "left_out": {}, "tallies": {}}
    for ds in ("kupa", "lwp"):
        if ds == "kupa":
            chosen, series, place = v6f.kupa_persons(c6, recs["v6"], recs["v6b"])
        else:
            chosen, series, place = v6f.lwp_persons(ROOT / d_lwp["file"], recs["v6e"])
        rec["persons"][ds] = {"n": len(chosen), "ids_sha256": hashlib.sha256("\n".join(sorted(chosen)).encode())
                              .hexdigest()}
        for d2, task in PLAN:
            if d2 != ds:
                continue
            name = f"{ds}_{task}"
            sens = (ds, task) in MAIN
            b = build(ds, task, chosen, series, place, m, r, rest_ms, sens)
            n_f = recs["v6f"]["count"][name][UNIT]["oddeven"]["persons"]
            if len(b["text"]) != n_f:
                raise SystemExit(f"{name}: {len(b['text'])} count persons, not v6f's {n_f}")
            rec["count"][name] = count(b, ds, task, False)
            if sens:
                rec["sensitivity_boundary_kept"][name] = count(b, ds, task, True)
            rec["left_out"][name] = b["left_out"]
            zero = rec["count"][name]["redrawn"]["IMG_pq"]["persons_with_a_zero_base"]
            rec["tallies"][name] = {"median_abs_dq_draw0": v6d.five(b["dq0"]), "persons_with_a_zero_base_IMG_pq": zero}
            del b
        del series
    rec["rule"] = evaluate(rec["count"])
    return {"main_cells": main_cells(rec), **rec}


def main_cells(rec: dict) -> dict:
    out: dict = {}
    for ds, task in MAIN:
        cell = rec["count"][f"{ds}_{task}"]
        for st in (*IMG, "S1", "S2"):
            out[f"{ds}_{task} | {st}"] = {row: {k: cell[row][st][k] for k in ("share_vs_own", "share_vs_own_ci")}
                                          for row in ["redrawn", "gathered/before", *v6f.SCALES, "inside/before",
                                                      *SHUFFLES]}
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
    recs = {k: json.loads((ROOT / f"results/prototype_{k}/record.json").read_text()) for k in ("v6", "v6b", "v6e", "v6f")}
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], "results/prototype_v6/record.json", *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    rec = run(c6, c4["mapping"], c4["relief"], recs)
    v6d.no_per_person(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6g-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    rule = rec["rule"]
    print(json.dumps({k: rule[k] for k in ("a_held", "positive_control", "symmetry_check", "shuffles_above",
                                           "shuffles_below", "tile_sequence", "predictions")}))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps({row: round(x["share_vs_own"], 3) for row, x in v.items()}))


if __name__ == "__main__":
    main()
