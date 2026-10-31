"""Prototype v6b: the operations of v6 against the person's own variation with an odd/even baseline, the order of the
tiles in the image (SEQ) and the place of the pauses at the stage of times (LOC), on KUPA-KEYS persons not used in v6
(decisions 1-3 of .steering/20261031-v6b-own-variation).

Usage:  uv run python works/prototype_v6b.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6b.json (committed before this script is first run) and v6's conditions
Input:  KUPA-KEYS TASK-1 and TASK-2 (external/kupa-keys; not in git, checked by SHA-256)
Output: results/prototype_v6b/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, per-person vector or person id is written.
"""

from __future__ import annotations

import csv
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
import prototype_v6 as v6  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v6b.json"
OUT_DIR = ROOT / "results" / "prototype_v6b"
CODE = ["works/prototype_v6b.py", "works/prototype_v6.py", "works/prototype_v5.py", "works/prototype_v4.py",
        "works/_common.py", "src/keystroke.py", "uv.lock"]
V6_LOOKED = 217
STAGE = {"S0": 0, "S1": 1, "S2": 2, "SEQ": 3, "LOC": 4}
BASELINE = {"oddeven": 0, "halves": 1}
SUB = ",;:"


# ---------------------------------------------------------------- reading


def kind_b(key: str) -> str | None:
    """v6's kind, with ',', ';' and ':' as 'sub' (v6 counts them as 'other'; the sentence and word starts do not use
    either)."""
    k = v6.kupa_kind(key)
    return "sub" if k == "other" and key in SUB else k


def read_kupa_b(path: Path) -> tuple[dict[str, dict], set[str]]:
    """As v6.read_kupa, with the kinds of kind_b."""
    raw: dict[str, tuple[list[float], list[str]]] = {}
    seen: set[str] = set()
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add(row["id"])
            if row["type"] != "down" or row["is_repeat"] == "True":
                continue
            kind = kind_b(row["key"])
            if kind is None:
                continue
            ts, ks_ = raw.setdefault(row["id"], ([], []))
            ts.append(float(row["time"]))
            ks_.append(kind)
    out = {}
    for pid, (ts, kinds) in raw.items():
        t, k, dropped = v6.clean(ts, kinds)
        out[pid] = {"times": t, "kinds": k, "dropped": dropped}
    return out, seen


def v6_kinds(kinds: list[str]) -> list[str]:
    return ["other" if k == "sub" else k for k in kinds]


# ---------------------------------------------------------------- labels and features


def labels(s: v6.Series, kinds: list[str]) -> np.ndarray:
    """The label of gap j, by press j + 1: sentence, subsentence, word, within or other."""
    st, ws = set(s.starts), set(s.word_starts)
    n = s.times.size
    lab = np.full(n - 1, "other", dtype=object)
    after_sub = kinds[0] == "sub"
    for p in range(1, n):
        k, prev = kinds[p], kinds[p - 1]
        if k == "alnum":
            if p in st:
                lab[p - 1] = "sentence"
            elif after_sub:
                lab[p - 1] = "subsentence"
            elif p in ws:
                lab[p - 1] = "word"
            elif prev == "alnum":
                lab[p - 1] = "within"
            after_sub = False
        elif k == "sub":
            after_sub = True
        elif k == "end":
            after_sub = False
    return lab


def lag1(x: np.ndarray) -> float:
    """Pearson lag-1 correlation; 0 when fewer than 3 values or either side is constant."""
    x = np.asarray(x, dtype=float)
    if x.size < 3 or np.ptp(x[:-1]) == 0 or np.ptp(x[1:]) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def seq(t: v6.Tiles, idx: np.ndarray) -> np.ndarray:
    p, q = t.p[idx], t.q[idx]
    return np.array([lag1(p), lag1(q), float(np.mean(np.abs(np.diff(p)))) if p.size > 1 else 0.0,
                     float(np.mean(np.abs(np.diff(q)))) if q.size > 1 else 0.0])


def loc(t: v6.Tiles, idx: np.ndarray, lab: np.ndarray) -> np.ndarray | None:
    """LOC of the gaps held by the tiles idx; None if a label has no gap."""
    held = np.unique(np.concatenate([t.held[i] for i in idx]))
    g, L = t.gaps[held], lab[held]
    meds = {}
    for k in ("sentence", "subsentence", "word", "within"):
        x = g[L == k]
        if x.size == 0:
            return None
        meds[k] = float(np.median(np.log(x)))
    rest = g >= 2000.0 - 1e-6
    share = float(g[rest & np.isin(L, ["sentence", "subsentence"])].sum() / g[rest].sum()) if rest.any() else 0.0
    hs = set(held.tolist())
    pairs = [(j, j + 1) for j in held if j + 1 in hs]
    lg = np.log(t.gaps)
    if len(pairs) >= 3:
        a = np.array([lg[i] for i, _ in pairs])
        b = np.array([lg[j] for _, j in pairs])
        ac = 0.0 if np.ptp(a) == 0 or np.ptp(b) == 0 else float(np.corrcoef(a, b)[0, 1])
    else:
        ac = 0.0
    return np.array([meds["sentence"], meds["subsentence"], meds["word"], meds["within"],
                     meds["sentence"] - meds["within"], share, ac])


def parts(n: int, how: str) -> tuple[np.ndarray, np.ndarray]:
    if how == "oddeven":
        return np.arange(0, n, 2), np.arange(1, n, 2)
    h = n // 2
    return np.arange(h), np.arange(h, n)


def vectors(t: v6.Tiles, idx: np.ndarray, lab: np.ndarray, how: str) -> dict:
    v = t.vectors(idx)
    out = {"S0": v["S0"], "S1": v["S1"], "S2": v["S2"], "LOC": loc(t, idx, lab)}
    if how == "halves":
        out["SEQ"] = seq(t, idx)
    return out


# ---------------------------------------------------------------- count


def has(v: tuple, st: str) -> bool:
    return v[0][st] is not None and v[1][st] is not None


def one(first: list, sel: list, st: str, rng_key: list[int]) -> dict:
    """v6's count 1 for one stage and row on given persons: first and sel are their (A, B) vectors of 'text' and R."""
    n = len(first)
    if n < 2:
        raise SystemExit(f"fewer than 2 persons for {rng_key}")
    t_all = np.array([v[h][st] for v in first for h in (0, 1)])
    sd = t_all.std(axis=0)
    keep = sd > 0
    if not keep.any():
        raise SystemExit(f"every feature is constant for {rng_key}")

    def z(x):
        return x[keep] / sd[keep]

    tA = np.array([z(v[0][st]) for v in first])
    tB = np.array([z(v[1][st]) for v in first])
    rA = np.array([z(v[0][st]) for v in sel])
    rB = np.array([z(v[1][st]) for v in sel])
    for a in (tA, tB, rA, rB):
        if not np.all(np.isfinite(a)):
            raise SystemExit(f"a non-finite value for {rng_key}")
    base = np.linalg.norm(tA - tB, axis=1)
    dAB = v6.distances(tA, tB)
    between = np.array([np.median(np.delete(dAB[i], i)) for i in range(n)])
    change = (np.linalg.norm(rA - tA, axis=1) + np.linalg.norm(rB - tB, axis=1)) / 2
    own = np.where(change > base, 1.0, np.where(change == base, 0.5, 0.0))
    other = np.where(change > between, 1.0, np.where(change == between, 0.5, 0.0))
    idx = np.random.default_rng(rng_key).integers(0, n, (v6.N_BOOT, n))
    c_own = v6.ci(own, idx)
    pos = base > 0
    return {"persons": n, "dropped_features": [int(j) for j in np.flatnonzero(~keep)],
            "share_vs_own": float(own.mean()), "share_vs_own_ci": c_own, "reading": v6.reading(*c_own),
            "share_vs_another": float(other.mean()), "share_vs_another_ci": v6.ci(other, idx),
            "median_base": float(np.median(base)), "median_change": float(np.median(change)),
            "median_between": float(np.median(between)),
            "median_change_over_base": float(np.median(change[pos] / base[pos])) if pos.any() else None,
            "base_zero": int((~pos).sum())}


def count(halves: dict, rows: list[str], task: str, unit: str, how: str) -> dict:
    """v6's count 1 per row and stage; for LOC the persons are those with LOC for 'text' and the row (decision 4),
    and S1 and S2 are counted again on those persons (same-person sensitivity)."""
    out: dict = {}
    stages = [s for s in STAGE if s != "SEQ" or how == "halves"]
    n_all = len(halves[rows[0]])
    for ri, row in enumerate(rows[1:], start=1):
        res: dict = {}
        for st in stages:
            key = [20261031, 66, v6.TASK[task], v6.UNIT[unit], STAGE[st], ri, BASELINE[how]]
            ok = [i for i in range(n_all) if has(halves[rows[0]][i], st) and has(halves[row][i], st)]
            if st == "LOC" and not ok:
                res[st] = {"defined": False, "persons": 0}
                continue
            if st != "LOC" and len(ok) != n_all:
                raise SystemExit(f"{task} {unit} {row} {st}: a vector is missing")
            res[st] = one([halves[rows[0]][i] for i in ok], [halves[row][i] for i in ok], st, key)
            res[st]["left_out_no_vector"] = n_all - len(ok)
            if st == "LOC":
                same = {}
                for s2 in ("S1", "S2"):
                    k2 = [20261031, 66, v6.TASK[task], v6.UNIT[unit], STAGE[s2], ri, BASELINE[how]]
                    same[s2] = one([halves[rows[0]][i] for i in ok], [halves[row][i] for i in ok], s2, k2)
                res["same_persons_as_LOC"] = same
        out[row] = res
    out["persons"] = n_all
    return out


# ---------------------------------------------------------------- run


def run(c6: dict, m: dict, r: dict) -> dict:
    rest_ms = c6["units"]["rest_ms"]
    kp = c6["data"]["kupa"]["files"]
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = read_kupa_b(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    place = {pid: i for i, pid in enumerate(ids_sorted)}
    order = v6.draw_order(ids_sorted, 0)

    def eligible(pid):
        ser = {}
        for task in ("copy", "essay"):
            x = kupa[task].get(pid)
            if x is None or x["times"].size < 2:
                return None
            k6 = v6_kinds(x["kinds"])
            st = v6.kupa_starts(k6)
            s = v6.Series(x["times"], st, v6.kupa_word_starts(k6))
            if len(st) < 5 or int(np.floor(s.times[-1] / v6.WINDOW_MS)) < 5 or not v6.every_sentence_holds_a_gap(s):
                return None
            ser[task] = (s, x["kinds"])
        return ser

    v6_ids = [pid for pid in order[:V6_LOOKED] if eligible(pid) is not None]
    rec6 = json.loads((ROOT / "results/prototype_v6/record.json").read_text())
    if len(v6_ids) != v6.N or hashlib.sha256("\n".join(sorted(v6_ids)).encode()).hexdigest() != \
            rec6["datasets"]["kupa"]["ids_sha256"]:
        raise SystemExit("the first 217 ids of the permutation do not give v6's persons")
    chosen, series, looked = [], {}, V6_LOOKED
    for pid in order[V6_LOOKED:]:
        looked += 1
        ser = eligible(pid)
        if ser is not None:
            chosen.append(pid)
            series[pid] = ser
            if len(chosen) == v6.N:
                break
    if len(chosen) < v6.N or set(chosen) & set(v6_ids):
        raise SystemExit("not 200 new eligible persons")
    rec: dict = {"persons": {"ids_looked_at": looked, "ids_sha256": hashlib.sha256("\n".join(sorted(chosen)).encode())
                             .hexdigest(), "overlap_with_v6": 0}}
    del kupa
    res: dict = {}
    for task in ("essay", "copy"):
        for unit in v6.UNIT:
            rows = v6.ROWS["kupa"][unit]
            hv: dict = {how: {row: [] for row in rows} for how in BASELINE}
            left = {"fewer_than_8": 0, "part_without_gap": 0}
            for pid in chosen:
                s, kinds = series[pid][task]
                tt = v6.Tiles(s, unit, "before", m, r, rest_ms)
                if tt.n < v6.MIN_TILES_COUNT1:
                    left["fewer_than_8"] += 1
                    continue
                if any(sum(tt.held[i].size for i in part) == 0 for how in BASELINE for part in parts(tt.n, how)):
                    left["part_without_gap"] += 1
                    continue
                rw, _ = v6.make_rows("kupa", s, unit, tt, [0, v6.TASK[task], place[pid]], m, r, rest_ms)
                v6.check_s0(rw, unit)
                lab = labels(s, kinds)
                for how in BASELINE:
                    A, B = parts(tt.n, how)
                    for row, t in rw.items():
                        if t.n < tt.n:
                            raise SystemExit(f"{task} {unit} {row}: fewer tiles than the first row")
                        hv[how][row].append((vectors(t, A, lab, how), vectors(t, B, lab, how)))
            res[f"{task}_{unit}"] = {"left_out": left,
                                     **{how: count(hv[how], rows, task, unit, how) for how in BASELINE}}
    rec["count"] = res
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
    f6 = c["data"]["v6_conditions"]
    raw6 = cm.check_file(ROOT / f6["file"], f6["sha256"], "v6's conditions")
    c6 = json.loads(raw6)
    m4 = c6["mapping"]["v4_conditions"]
    c4 = json.loads(cm.check_file(ROOT / m4["file"], m4["sha256"], "v4's conditions"))
    paths = [COND.relative_to(ROOT).as_posix(), f6["file"], m4["file"], "results/prototype_v6/record.json", *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    for task, x in c6["data"]["kupa"]["files"].items():
        if cm.sha256(ROOT / x["file"]) != x["sha256"]:
            raise SystemExit(f"{x['file']} does not match the SHA-256 of v6's conditions")
    rec = run(c6, c4["mapping"], c4["relief"])
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "v6_conditions_sha256": hashlib.sha256(raw6).hexdigest(),
                         "data_sha256": {t: x["sha256"] for t, x in c6["data"]["kupa"]["files"].items()},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6b-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    main_cell = rec["count"]["essay_sentence"]["oddeven"]
    print(json.dumps({row: {st: (round(v["share_vs_own"], 3), v["reading"]["size"]) for st, v in main_cell[row].items()}
                      for row in ("inside/before", "shuffled/before", "shuffled_in_halves/before")}))


if __name__ == "__main__":
    main()
