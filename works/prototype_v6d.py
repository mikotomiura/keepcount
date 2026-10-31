"""Prototype v6d: v6b's reference observable of the place of the pauses (LOC) split into its position part (R_pos)
and its order part (R_lag), and the lengths of the records of v6 and v6b (decisions 10 and 11 of
.steering/20261031-paper-a-outline).

Usage:  uv run python works/prototype_v6d.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6d.json (committed before this script is first run), v6b's and v6's
            conditions and records
Input:  KUPA-KEYS TASK-1 and TASK-2 and Aalto 136M (external/; not in git, checked by SHA-256)
Output: results/prototype_v6d/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, per-person vector, per-person value or person id is written.
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
import prototype_v6 as v6  # noqa: E402
import prototype_v6b as v6b  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6d.json"
OUT_DIR = ROOT / "results" / "prototype_v6d"
CODE = ["works/prototype_v6d.py", "works/prototype_v6b.py", "works/prototype_v6.py", "works/prototype_v5.py",
        "works/prototype_v4.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
PART = {"R_pos": 0, "R_lag": 1}
N_POS = 6
CHECK_TOL = 1e-12


# ---------------------------------------------------------------- split and count


def split(v: np.ndarray | None) -> dict:
    """LOC (7 numbers) and its parts: the six numbers of position and the lag-1 alone."""
    if v is None:
        return {"LOC": None, "R_pos": None, "R_lag": None}
    if v.size != N_POS + 1:
        raise SystemExit(f"LOC has {v.size} numbers, not {N_POS + 1}")
    return {"LOC": v, "R_pos": v[:N_POS], "R_lag": v[N_POS:]}


def count(halves: dict, rows: list[str], task: str, unit: str, how: str, v6b_cell: dict) -> dict:
    """v6b's count 1 on LOC (checked against v6b), R_pos and R_lag, on the persons of LOC of each row."""
    out: dict = {}
    n_all = len(halves[rows[0]])
    for ri, row in enumerate(rows[1:], start=1):
        ok = [i for i in range(n_all) if v6b.has(halves[rows[0]][i], "LOC") and v6b.has(halves[row][i], "LOC")]
        if row == "text/none":
            if ok:
                raise SystemExit(f"{task} {unit} text/none has LOC for some persons")
            out[row] = {"defined": False, "persons": 0}
            continue
        first, sel = [halves[rows[0]][i] for i in ok], [halves[row][i] for i in ok]
        key = [20261031, 66, v6.TASK[task], v6.UNIT[unit], v6b.STAGE["LOC"], ri, v6b.BASELINE[how]]
        loc = v6b.one(first, sel, "LOC", key)
        ref = v6b_cell[row]["LOC"]
        for k in ("share_vs_own", "median_change_over_base"):
            if not abs(loc[k] - ref[k]) < CHECK_TOL:
                raise SystemExit(f"{task} {unit} {how} {row} LOC {k}: {loc[k]} differs from v6b's {ref[k]}")
        res = {"LOC": loc}
        for part, code in PART.items():
            res[part] = v6b.one(first, sel, part,
                                [20261031, 67, v6.TASK[task], v6.UNIT[unit], code, ri, v6b.BASELINE[how]])
        out[row] = res
    out["persons"] = n_all
    return out


def five(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        raise SystemExit("no value to summarise")
    q = np.percentile(a, [0, 25, 50, 75, 100], method="linear")
    return {"persons": int(a.size), "min": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "max": float(q[4])}


# ---------------------------------------------------------------- v6b's persons (as v6b.run)


def run_v6b(c6: dict, m: dict, r: dict, rec6: dict, rec6b: dict) -> tuple[dict, dict]:
    rest_ms = c6["units"]["rest_ms"]
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
    del kupa
    res: dict = {}
    lengths: dict = {}
    for task in ("essay", "copy"):
        lengths[task] = {"presses": five([series[p][task][0].times.size for p in chosen]),
                         "sentences": five([len(series[p][task][0].starts) for p in chosen])}
        for unit in v6.UNIT:
            rows = v6.ROWS["kupa"][unit]
            hv: dict = {how: {row: [] for row in rows} for how in v6b.BASELINE}
            tiles_all, tiles_count, half_sizes = [], [], {how: [] for how in v6b.BASELINE}
            for pid in chosen:
                s, kinds = series[pid][task]
                tt = v6.Tiles(s, unit, "before", m, r, rest_ms)
                tiles_all.append(tt.n)
                if tt.n < v6.MIN_TILES_COUNT1:
                    continue
                if any(sum(tt.held[i].size for i in part) == 0 for how in v6b.BASELINE
                       for part in v6b.parts(tt.n, how)):
                    continue
                tiles_count.append(tt.n)
                rw, _ = v6.make_rows("kupa", s, unit, tt, [0, v6.TASK[task], place[pid]], m, r, rest_ms)
                v6.check_s0(rw, unit)
                lab = v6b.labels(s, kinds)
                for how in v6b.BASELINE:
                    A, B = v6b.parts(tt.n, how)
                    half_sizes[how].append(min(A.size, B.size))
                    for row, t in rw.items():
                        if t.n < tt.n:
                            raise SystemExit(f"{task} {unit} {row}: fewer tiles than the first row")
                        hv[how][row].append((split(v6b.loc(t, A, lab)), split(v6b.loc(t, B, lab))))
            cell = f"{task}_{unit}"
            res[cell] = {how: count(hv[how], rows, task, unit, how, rec6b["count"][cell][how])
                         for how in v6b.BASELINE}
            lengths[task][f"tiles_{unit}"] = five(tiles_all)
            lengths[task][f"tiles_{unit}_count_persons"] = five(tiles_count)
            lengths[task][f"smaller_half_{unit}"] = {how: five(v) for how, v in half_sizes.items()}
    return res, lengths


# ---------------------------------------------------------------- v6's persons (as v6.run)


def lengths_v6(c6: dict, m: dict, r: dict, rec6: dict) -> dict:
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
    if hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest() != rec6["datasets"]["kupa"]["ids_sha256"]:
        raise SystemExit("KUPA: the persons differ from v6's")
    del kupa
    out: dict = {"kupa": {}, "aalto": {}}
    for task in ("copy", "essay"):
        ss = [series[p][task] for p in chosen]
        out["kupa"][task] = {"presses": five([s.times.size for s in ss]), "sentences": five([len(s.starts) for s in ss]),
                             "windows_10s": five([v6.Tiles(s, "window_10s", "before", m, r, rest_ms).n for s in ss])}
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
            rec6["datasets"]["aalto"]["ids_sha256"]:
        raise SystemExit("Aalto: the persons differ from v6's")
    for part, sl in (("first7", slice(0, 7)), ("last8", slice(7, 15))):
        sps = [v6.aalto_series(a_series[p][sl]) for p in a_chosen]
        out["aalto"][part] = {"presses": five([sp.times.size for sp in sps]),
                              "sentences": five([len(a_series[p][sl]) for p in a_chosen]),
                              "windows_10s": five([v6.Tiles(sp, "window_10s", "before", m, r, rest_ms).n for sp in sps])}
    return out


def no_per_person(x, path: str = "") -> None:
    """The record holds no list as long as a set of persons."""
    if isinstance(x, dict):
        for k, v in x.items():
            no_per_person(v, f"{path}/{k}")
    elif isinstance(x, (list, tuple)):
        if len(x) >= 25:
            raise SystemExit(f"{path}: a list of {len(x)} values in the record")
        for v in x:
            no_per_person(v, path)


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
    rawb = cm.check_file(ROOT / c["v6b"]["conditions"]["file"], c["v6b"]["conditions"]["sha256"], "v6b's conditions")
    rawrb = cm.check_file(ROOT / c["v6b"]["record"]["file"], c["v6b"]["record"]["sha256"], "v6b's record")
    c6b, rec6b = json.loads(rawb), json.loads(rawrb)
    f6 = c6b["data"]["v6_conditions"]
    raw6 = cm.check_file(ROOT / f6["file"], f6["sha256"], "v6's conditions")
    c6 = json.loads(raw6)
    rawr6 = (ROOT / "results/prototype_v6/record.json").read_bytes()
    rec6 = json.loads(rawr6)
    for rec in (rec6b, rec6):
        for p, h in rec["provenance"]["code_sha256"].items():
            if cm.sha256(ROOT / p) != h:
                raise SystemExit(f"{p} differs from the code that made a record v6d reuses")
    if c6["generator"] != c["generator"]:
        raise SystemExit("the generator of v6's conditions differs from v6d's")
    m4 = c6["mapping"]["v4_conditions"]
    raw4 = cm.check_file(ROOT / m4["file"], m4["sha256"], "v4's conditions")
    c4 = json.loads(raw4)
    paths = [COND.relative_to(ROOT).as_posix(), c["v6b"]["conditions"]["file"], c["v6b"]["record"]["file"],
             f6["file"], "results/prototype_v6/record.json", m4["file"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    data = {"kupa_copy": c6["data"]["kupa"]["files"]["copy"], "kupa_essay": c6["data"]["kupa"]["files"]["essay"],
            "aalto": {"file": c6["data"]["aalto"]["file"], "sha256": c6["data"]["aalto"]["sha256"]}}
    for x in data.values():
        if cm.sha256(ROOT / x["file"]) != x["sha256"]:
            raise SystemExit(f"{x['file']} does not match the SHA-256 of v6's conditions")
    count_res, len_b = run_v6b(c6, c4["mapping"], c4["relief"], rec6, rec6b)
    rec = {"count": count_res, "lengths": {"v6b_kupa": len_b, "v6": lengths_v6(c6, c4["mapping"], c4["relief"], rec6)}}
    no_per_person(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "v6b_conditions_sha256": hashlib.sha256(rawb).hexdigest(),
                         "v6b_record_sha256": hashlib.sha256(rawrb).hexdigest(),
                         "v6_conditions_sha256": hashlib.sha256(raw6).hexdigest(),
                         "v6_record_sha256": hashlib.sha256(rawr6).hexdigest(),
                         "v4_conditions_sha256": hashlib.sha256(raw4).hexdigest(),
                         "data_sha256": {n: x["sha256"] for n, x in data.items()},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "versions": cm.versions("numpy", "pillow", "scipy"),
                         "check_against_v6b": f"LOC share_vs_own and median_change_over_base of every cell, baseline "
                                              f"and row within {CHECK_TOL}"}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6d-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    main_cell = rec["count"]["essay_sentence"]["oddeven"]
    print(json.dumps({row: {st: round(main_cell[row][st]["share_vs_own"], 3) for st in ("LOC", "R_pos", "R_lag")}
                      for row in ("inside/before", "shuffled/before", "shuffled_in_halves/before")}))


if __name__ == "__main__":
    main()
