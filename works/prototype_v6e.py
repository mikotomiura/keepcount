"""Prototype v6e: v6b's count 1 and v6d's split of LOC on the essays of the Kaggle competition Linking Writing
Processes to Writing Quality (LWP), with the predictions and the confirmation rule fixed in the conditions (decisions
1-6 of .steering/20261101-second-dataset).

Usage:  uv run python works/prototype_v6e.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6e.json (committed before this script is first run)
Input:  external/kaggle-lwp/data/train_logs.csv (not in git, checked by SHA-256; not redistributed)
Output: results/prototype_v6e/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value or essay id is written.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
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
import prototype_v6 as v6  # noqa: E402
import prototype_v6b as v6b  # noqa: E402
import prototype_v6d as v6d  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6e.json"
OUT_DIR = ROOT / "results" / "prototype_v6e"
CODE = ["works/prototype_v6e.py", "works/prototype_v6d.py", "works/prototype_v6b.py", "works/prototype_v6.py",
        "works/prototype_v5.py", "works/prototype_v4.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
DATASET = 2
STAGE = {"S0": 0, "S1": 1, "S2": 2, "SEQ": 3, "LOC": 4, "R_pos": 5, "R_lag": 6}
LOC_LIKE = ("LOC", "R_pos", "R_lag")
NAMED = ("Enter", "Backspace", "Delete")
EDIT_ACTIVITIES = ("Input", "Remove/Cut")
MAIN_ROWS = ("inside/before", "shuffled/before", "shuffled_in_halves/before")


# ---------------------------------------------------------------- reading


def lwp_key(down_event: str) -> str | None:
    """The key of a counted LWP press (conditions data.lwp.counted_press), or None."""
    if down_event == "Space":
        return " "
    if down_event in NAMED or len(down_event) == 1:
        return down_event
    return None


def read_lwp(path: Path, activities: tuple[str, ...] | None) -> tuple[dict[str, dict], list[str], dict]:
    """Per id: the counted presses (v6.clean) and their v6b kinds; every id; tallies. Rows must come grouped by id and
    in increasing event_id (checked)."""
    raw: dict[str, tuple[list[float], list[str]]] = {}
    ids: list[str] = []
    tally = {"events": 0, "decreasing_down_time": 0, "not_counted": 0, "other_activity": 0}
    last_id, last_ev, last_t = None, -1, None
    done: set[str] = set()
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tally["events"] += 1
            pid, ev, t = row["id"], int(row["event_id"]), float(row["down_time"])
            if not math.isfinite(t):
                raise SystemExit("a down_time is not a finite number")
            if pid != last_id:
                if pid in done:
                    raise SystemExit("rows of an id are not contiguous")
                if last_id is not None:
                    done.add(last_id)
                ids.append(pid)
                last_id, last_ev, last_t = pid, -1, None
            if ev <= last_ev:
                raise SystemExit("event_id does not increase within an id")
            if last_t is not None and t < last_t:
                tally["decreasing_down_time"] += 1
            last_ev, last_t = ev, t
            if activities is not None and row["activity"] not in activities:
                tally["other_activity"] += 1
                continue
            key = lwp_key(row["down_event"])
            kind = v6b.kind_b(key) if key is not None else None
            if kind is None:
                tally["not_counted"] += 1
                continue
            ts, ks_ = raw.setdefault(pid, ([], []))
            ts.append(t)
            ks_.append(kind)
    out, dropped = {}, 0
    for pid, (ts, kinds) in raw.items():
        t, k, d = v6.clean(ts, kinds)
        out[pid] = {"times": t, "kinds": k}
        dropped += d
    tally["dropped_equal_times"] = dropped
    return out, ids, tally


def choose(lwp: dict, ids: list[str], allow_fewer: bool = False) -> tuple[list[str], dict, int]:
    """The first 200 eligible ids in v6's draw order with d = 2, their series and kinds, and the number looked at
    (fewer only if allowed: the sensitivity)."""
    chosen, series, looked = [], {}, 0
    for pid in v6.draw_order(sorted(ids), 2):
        looked += 1
        x = lwp.get(pid)
        if x is None or x["times"].size < 2:
            continue
        k6 = v6b.v6_kinds(x["kinds"])
        st = v6.kupa_starts(k6)
        s = v6.Series(x["times"], st, v6.kupa_word_starts(k6))
        if len(st) < 5 or int(np.floor(s.times[-1] / v6.WINDOW_MS)) < 5 or not v6.every_sentence_holds_a_gap(s):
            continue
        chosen.append(pid)
        series[pid] = (s, x["kinds"])
        if len(chosen) == v6.N:
            break
    if len(chosen) < v6.N and not allow_fewer:
        raise SystemExit(f"only {len(chosen)} eligible ids")
    return chosen, series, looked


# ---------------------------------------------------------------- vectors and count


def vectors(t: v6.Tiles, idx: np.ndarray, lab: np.ndarray, how: str) -> dict:
    v = v6b.vectors(t, idx, lab, how)
    v.update({k: x for k, x in v6d.split(v["LOC"]).items() if k != "LOC"})
    return v


def count(halves: dict, rows: list[str], unit: str, how: str, stages: tuple[str, ...], prefix: int,
          main_rows: list[str] | None = None) -> dict:
    """v6b's one() per row and stage; LOC-like stages on the persons with LOC for 'text' and the row, and S1, S2 and
    (halves) SEQ again on those persons."""
    out: dict = {}
    n_all = len(halves[rows[0]])
    for ri, row in enumerate(rows[1:], start=1):
        if main_rows is not None:
            ri = main_rows.index(row)  # the row index of the main counts, for the same keys
        res: dict = {}
        loc_ok = [i for i in range(n_all) if v6b.has(halves[rows[0]][i], "LOC") and v6b.has(halves[row][i], "LOC")]
        for st in stages:
            if st == "SEQ" and how != "halves":
                continue
            key = [20261031, prefix, v6.UNIT[unit], STAGE[st], ri, v6b.BASELINE[how]]
            if st in LOC_LIKE:
                if row == "text/none":
                    if loc_ok:
                        raise SystemExit(f"{unit} text/none has LOC for some persons")
                    res[st] = {"defined": False, "persons": 0}
                    continue
                ok = loc_ok
            else:
                ok = list(range(n_all))
                if not all(v6b.has(halves[rows[0]][i], st) and v6b.has(halves[row][i], st) for i in ok):
                    raise SystemExit(f"{unit} {row} {st}: a vector is missing")
            res[st] = v6b.one([halves[rows[0]][i] for i in ok], [halves[row][i] for i in ok], st, key)
            res[st]["left_out_no_vector"] = n_all - len(ok)
        if row != "text/none" and any(st in LOC_LIKE for st in stages):
            same = {}
            for s2 in ("S1", "S2") + (("SEQ",) if how == "halves" else ()):
                if s2 not in stages:
                    continue
                k2 = [20261031, prefix, v6.UNIT[unit], STAGE[s2], ri, v6b.BASELINE[how]]
                same[s2] = v6b.one([halves[rows[0]][i] for i in loc_ok], [halves[row][i] for i in loc_ok], s2, k2)
            res["same_persons_as_LOC"] = same
        out[row] = res
    out["persons"] = n_all
    return out


def build(series: dict, chosen: list[str], place: dict, m: dict, r: dict, rest_ms: float,
          rows_of: dict[str, list[str]]) -> tuple[dict, dict]:
    """Per unit: the halves of every row for each baseline, and the lengths."""
    hv: dict = {}
    lengths: dict = {}
    for unit, rows in rows_of.items():
        hv[unit] = {how: {row: [] for row in rows} for how in v6b.BASELINE}
        tiles_all, tiles_count = [], []
        halves_info = {how: {h: {"tiles": [], "gaps": [], "pairs": []} for h in "AB"} for how in v6b.BASELINE}
        left = {"fewer_than_8": 0, "part_without_gap": 0}
        for pid in chosen:
            s, kinds = series[pid]
            tt = v6.Tiles(s, unit, "before", m, r, rest_ms)
            tiles_all.append(tt.n)
            if tt.n < v6.MIN_TILES_COUNT1:
                left["fewer_than_8"] += 1
                continue
            if any(sum(tt.held[i].size for i in part) == 0 for how in v6b.BASELINE for part in v6b.parts(tt.n, how)):
                left["part_without_gap"] += 1
                continue
            tiles_count.append(tt.n)
            # random keys [dataset 2, task 0 (the one essay), place]
            rw, _ = v6.make_rows("kupa", s, unit, tt, [DATASET, 0, place[pid]], m, r, rest_ms)
            rw = {row: t for row, t in rw.items() if row in rows}
            v6.check_s0(rw, unit)
            lab = v6b.labels(s, kinds)
            for how in v6b.BASELINE:
                A, B = v6b.parts(tt.n, how)
                for h, idx in (("A", A), ("B", B)):
                    nt, ng, npair = v6d.half_counts(tt, idx)
                    for k, v in (("tiles", nt), ("gaps", ng), ("pairs", npair)):
                        halves_info[how][h][k].append(v)
                for row, t in rw.items():
                    if t.n < tt.n:
                        raise SystemExit(f"{unit} {row}: fewer tiles than the first row")
                    hv[unit][how][row].append((vectors(t, A, lab, how), vectors(t, B, lab, how)))
        lengths[unit] = {"tiles": v6d.five(tiles_all), "tiles_count_persons": v6d.five(tiles_count), "left_out": left,
                         "halves": {how: {h: {k: v6d.five(v) for k, v in d.items()} for h, d in hi.items()}
                                    for how, hi in halves_info.items()}}
    return hv, lengths


# ---------------------------------------------------------------- predictions and rule


def evaluate(cnt: dict) -> dict:
    """The predictions and the confirmation rule of the conditions, read mechanically from the counts."""
    rows_img = {"sentence": ["text/after", "text/none", "inside/before", "shuffled/before", "shuffled_in_halves/before"],
                "window_10s": ["inside", "shuffled", "shuffled_in_halves"]}
    shuf = {"sentence": ["shuffled/before", "shuffled_in_halves/before"], "window_10s": ["shuffled", "shuffled_in_halves"]}
    inside = {"sentence": "inside/before", "window_10s": "inside"}

    def cells(rows: dict, stages_by_how: dict) -> list[tuple]:
        return [(u, how, row, st) for u, rs in rows.items() for how, sts in stages_by_how.items() for row in rs
                for st in sts]

    img = {"oddeven": ("S1", "S2"), "halves": ("S1", "S2", "SEQ")}
    loc = {"oddeven": ("LOC",), "halves": ("LOC",)}

    def ci(c):
        u, how, row, st = c
        return cnt[u][how][row][st]["share_vs_own_ci"]

    p1 = cells(rows_img, img)
    p2 = cells(shuf, img)
    p3 = cells(shuf, loc)
    p4 = cells(shuf, {"oddeven": ("R_pos", "R_lag"), "halves": ("R_pos", "R_lag")})
    p5_in = [(u, how, inside[u], "LOC") for u in inside for how in v6b.BASELINE]
    p5_after = [("sentence", how, "text/after", st) for how in v6b.BASELINE
                for st in img[how] + ("S0", "LOC", "R_pos", "R_lag")]
    if (len(p1), len(p2), len(p3), len(p5_after)) != (40, 20, 8, 13):
        raise SystemExit("the cells of the rule differ from the conditions")
    held = {
        "P1": sum(ci(c)[1] < 0.60 for c in p1),
        "P1_same_persons": sum((cnt[c[0]][c[1]][c[2]]["same_persons_as_LOC"][c[3]]["share_vs_own_ci"][1] < 0.60)
                               for c in p1 if c[2] != "text/none"),
        "P2": sum(ci(c)[1] < 0.5 for c in p2),
        "P3": sum(ci(c)[0] > 0.60 for c in p3),
        "P4_R_pos": sum(ci(c)[0] > 0.60 for c in p4 if c[3] == "R_pos"),
        "P4_R_lag": sum(ci(c)[0] > 0.60 for c in p4 if c[3] == "R_lag"),
        "P5_inside_not_above": sum(not ci(c)[0] > 0.60 for c in p5_in),
        "P5_after_below": sum(ci(c)[1] < 0.60 for c in p5_after),
    }
    of = {"P1": 40, "P1_same_persons": len([c for c in p1 if c[2] != "text/none"]), "P2": 20, "P3": 8,
          "P4_R_pos": 4 * 2, "P4_R_lag": 4 * 2, "P5_inside_not_above": 4, "P5_after_below": len(p5_after)}
    if held["P1"] == 40 and held["P3"] == 8:
        finding = "confirmed"
    elif held["P1"] == 40 and held["P3"] >= 6:
        finding = "partly confirmed"
    else:
        finding = "not confirmed"
    stronger = finding == "confirmed" and held["P2"] >= 14
    failed = {"P1": [c for c in p1 if not ci(c)[1] < 0.60], "P2": [c for c in p2 if not ci(c)[1] < 0.5],
              "P3": [c for c in p3 if not ci(c)[0] > 0.60]}
    return {"held": held, "of": of, "finding_1": finding,
            "stronger_form": "confirmed" if stronger else "not confirmed",
            "cells_not_holding": {k: {" | ".join(c): "not held" for c in v} for k, v in failed.items()}}


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
    raw6 = cm.check_file(ROOT / c["v6_conditions"]["file"], c["v6_conditions"]["sha256"], "v6's conditions")
    c6 = json.loads(raw6)
    m4 = c6["mapping"]["v4_conditions"]
    if c["v4_conditions"]["file"] != m4["file"]:
        raise SystemExit("v4's conditions differ between v6 and v6e")
    raw4 = cm.check_file(ROOT / m4["file"], m4["sha256"], "v4's conditions")
    c4 = json.loads(raw4)
    if c6["generator"] != c["generator"]:
        raise SystemExit("the generator of v6's conditions differs from v6e's")
    rest_ms = c6["units"]["rest_ms"]
    if rest_ms != 2000.0 or c["participants"]["n"] != v6.N:
        raise SystemExit("rest_ms or n differs")
    d = c["data"]["lwp"]
    paths = [COND.relative_to(ROOT).as_posix(), c["v6_conditions"]["file"], m4["file"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    if cm.sha256(ROOT / d["file"]) != d["sha256"]:
        raise SystemExit(f"{d['file']} does not match the SHA-256 of the conditions")
    m, r = c4["mapping"], c4["relief"]
    rows_of = {"sentence": v6.ROWS["kupa"]["sentence"], "window_10s": v6.ROWS["kupa"]["window_10s"]}

    lwp, ids, tally = read_lwp(ROOT / d["file"], None)
    chosen, series, looked = choose(lwp, ids)
    place = {pid: i for i, pid in enumerate(sorted(ids))}
    del lwp
    hv, lengths = build(series, chosen, place, m, r, rest_ms, rows_of)
    stages = tuple(STAGE)
    cnt = {u: {how: count(hv[u][how], rows_of[u], u, how, stages, 69) for how in v6b.BASELINE} for u in rows_of}
    rec: dict = {
        "persons": {"ids_in_file": len(ids), "ids_looked_at": looked,
                    "ids_sha256": hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest(), "tally": tally,
                    "presses": v6d.five([series[p][0].times.size for p in chosen]),
                    "sentences": v6d.five([len(series[p][0].starts) for p in chosen]),
                    "windows_10s": v6d.five([int(np.floor(series[p][0].times[-1] / v6.WINDOW_MS))
                                             for p in chosen])},
        "lengths": lengths, "count": cnt, "rule": evaluate(cnt)}

    # sensitivity: only Input and Remove/Cut events, main cells
    lwp2, ids2, tally2 = read_lwp(ROOT / d["file"], EDIT_ACTIVITIES)
    chosen2, series2, looked2 = choose(lwp2, ids2, allow_fewer=True)
    del lwp2
    rows2 = {"sentence": [rows_of["sentence"][0], *MAIN_ROWS]}
    hv2, lengths2 = build(series2, chosen2, place, m, r, rest_ms, rows2)
    st2 = ("S1", "S2", "LOC", "R_pos", "R_lag")
    rec["sensitivity_input_and_remove_only"] = {
        "persons": len(chosen2), "lengths": lengths2, "ids_looked_at": looked2, "ids_sha256": hashlib.sha256("\n".join(sorted(chosen2)).encode()).hexdigest(),
        "overlap_with_main": len(set(chosen) & set(chosen2)), "tally": tally2,
        "keys": "as the main counts: [20261031, 69, unit, stage, row index in the main rows, baseline]",
        "count": {how: count(hv2["sentence"][how], rows2["sentence"], "sentence", how, st2, 69,
                             main_rows=rows_of["sentence"]) for how in v6b.BASELINE}}
    v6d.no_per_person(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "v6_conditions_sha256": hashlib.sha256(raw6).hexdigest(),
                         "v4_conditions_sha256": hashlib.sha256(raw4).hexdigest(),
                         "data_sha256": d["sha256"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6e-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    main_cell = rec["count"]["sentence"]["oddeven"]
    print(json.dumps({"rule": rec["rule"],
                      "main": {row: {st: round(main_cell[row][st]["share_vs_own"], 3)
                                     for st in ("S1", "S2", "LOC", "R_pos", "R_lag")} for row in MAIN_ROWS}}))


if __name__ == "__main__":
    main()
