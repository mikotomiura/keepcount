"""Prototype v6c: v6's identification (count 2) with the measures of keystroke biometrics (Rank-n rates, EER) and its
dependence on the number of persons (decision 8 of .steering/20261031-paper-a-outline).

Usage:  uv run python works/prototype_v6c.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6c.json (committed before this script is first run), v6's conditions and record
Input:  KUPA-KEYS TASK-1 and TASK-2 and Aalto 136M (external/; not in git, checked by SHA-256)
Output: results/prototype_v6c/record.json (written to a temporary directory beside it and renamed when complete). No
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

COND = ROOT / "works" / "conditions" / "prototype_v6c.json"
OUT_DIR = ROOT / "results" / "prototype_v6c"
CODE = ["works/prototype_v6c.py", "works/prototype_v6.py", "works/prototype_v5.py", "works/prototype_v4.py",
        "works/_common.py", "src/keystroke.py", "uv.lock"]
RANKS = (1, 5, 10, 20)
POPULATION = (25, 50, 100, 200)
N_SUBSETS = 200
CHECK_TOL = 1e-12
DIRECTIONS = {"kupa_copy_to_essay": ("kupa", "copy", "essay", 0), "kupa_essay_to_copy": ("kupa", "essay", "copy", 1),
              "aalto_first7_to_last8": ("aalto", "A", "B", 0)}


# ---------------------------------------------------------------- counts


def rank_rates(d: np.ndarray, ranks: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Per evaluation person, the rate of identification within rank n (ties broken at random in expectation):
    r = 1 + impostors closer than the genuine, t = impostors as close; rate = clip((n - r + 1) / (t + 1), 0, 1)."""
    n = d.shape[0]
    gen = np.diag(d)
    imp = d[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    r = 1 + (imp < gen[:, None]).sum(axis=1)
    t = (imp == gen[:, None]).sum(axis=1)
    return {k: np.clip((k - r + 1) / (t + 1), 0.0, 1.0) for k in ranks}


def eer(d: np.ndarray) -> dict:
    """Pooled equal error rate: at each distinct distance as threshold, FAR = impostors at or below it, FRR =
    genuines above it; EER = (FAR + FRR) / 2 at the first threshold where |FAR - FRR| is smallest."""
    n = d.shape[0]
    gen = np.sort(np.diag(d))
    imp = np.sort(d[~np.eye(n, dtype=bool)])
    thr = np.unique(np.concatenate([gen, imp]))
    far = np.searchsorted(imp, thr, side="right") / imp.size
    frr = 1.0 - np.searchsorted(gen, thr, side="right") / gen.size
    i = int(np.argmin(np.abs(far - frr)))
    return {"eer": float((far[i] + frr[i]) / 2), "far": float(far[i]), "frr": float(frr[i])}


def full(g: np.ndarray, e: np.ndarray, key: list[int]) -> dict:
    """The full set: Rank-n rates with bootstrap intervals, rank_auc and top1 (for the check against v6), EER."""
    gs, es, drop = v6.standardise(g, e, "main")
    d = v6.distances(es, gs)
    top1, rauc = v6.per_person(d)
    rates = rank_rates(d, RANKS)
    n = d.shape[0]
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    return {"top1": float(top1.mean()), "rank_auc": float(rauc.mean()),
            "rank_n": {str(k): {"rate": float(v.mean()), "ci": v6.ci(v, idx), "chance": k / n} for k, v in rates.items()},
            "eer": eer(d), "dropped_features": drop}


def subset(g: np.ndarray, e: np.ndarray, sel: np.ndarray) -> tuple[float, float, float]:
    gs, es, _ = v6.standardise(g[sel], e[sel], "main")
    d = v6.distances(es, gs)
    top1, rauc = v6.per_person(d)
    return float(top1.mean()), float(rank_rates(d, (5,))[5].mean()), float(rauc.mean())


def population(g: dict, e: dict, ds: str, direction: int) -> dict:
    """g[unit][stage], e[unit][stage]: (200, k). The same subsets for every stage and unit of a direction."""
    n = next(iter(next(iter(g.values())).values())).shape[0]
    subsets = {}
    for size in POPULATION:
        if size == n:
            subsets[size] = [np.arange(n)]
        else:
            rng = np.random.default_rng([v6.SEED, 63, 2, v6.DATASET[ds], direction, size])
            subsets[size] = [rng.choice(n, size, replace=False) for _ in range(N_SUBSETS)]
    out: dict = {}
    for unit in v6.UNIT:
        out[unit] = {}
        for st in v6.STAGE:
            out[unit][st] = {}
            for size, sels in subsets.items():
                vals = np.array([subset(g[unit][st], e[unit][st], s) for s in sels])
                out[unit][st][str(size)] = {
                    name: {"mean": float(vals[:, j].mean()),
                           **({"p2.5": float(np.percentile(vals[:, j], 2.5)),
                               "p97.5": float(np.percentile(vals[:, j], 97.5))} if len(sels) > 1 else {})}
                    for j, name in enumerate(("top1", "rank5", "rank_auc"))}
                out[unit][st][str(size)]["chance"] = {"top1": 1 / size, "rank5": 5 / size}
    return out


def check_v6(res: dict, v6rec: dict, name: str) -> None:
    for unit, su in res.items():
        for st, r_ in su.items():
            ref = v6rec["count_2_person"][name][unit]["main"][st]
            for k in ("top1", "rank_auc"):
                if abs(r_[k] - ref[k]) >= CHECK_TOL:
                    raise SystemExit(f"{name} {unit} {st} {k}: {r_[k]} differs from v6's {ref[k]}")


# ---------------------------------------------------------------- vectors (as v6.run)


def vectors(c6: dict, m: dict, r: dict, v6rec: dict) -> dict:
    """(dataset, part) -> unit -> stage -> (200, k), built as v6.run builds count 2's vectors."""
    rest_ms = c6["units"]["rest_ms"]
    kp = c6["data"]["kupa"]["files"]
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6.read_kupa(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    chosen, series = [], {}
    for pid in v6.draw_order(ids_sorted, 0):
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
    vec: dict = {}
    for task in ("copy", "essay"):
        for pid in chosen:
            for unit in v6.UNIT:
                counting = v6.ROWS["kupa"][unit][0].partition("/")[2] or "before"
                tt = v6.Tiles(series[pid][task], unit, counting, m, r, rest_ms)
                for st, v in tt.all().items():
                    vec.setdefault(("kupa", task), {}).setdefault(unit, {}).setdefault(st, []).append(v)
    for pid in a_chosen:
        for part, sl in (("A", slice(0, 7)), ("B", slice(7, 15))):
            sp = v6.aalto_series(a_series[pid][sl])
            for unit in v6.UNIT:
                tp = v6.Tiles(sp, unit, "before", m, r, rest_ms)
                for st, v in tp.all().items():
                    vec.setdefault(("aalto", part), {}).setdefault(unit, {}).setdefault(st, []).append(v)
    return {k: {u: {st: np.array(v) for st, v in uv.items()} for u, uv in d.items()} for k, d in vec.items()}


def run(c6: dict, m: dict, r: dict, v6rec: dict) -> dict:
    arr = vectors(c6, m, r, v6rec)
    rec: dict = {"full": {}, "population": {}}
    for name, (ds, a, b, direction) in DIRECTIONS.items():
        g, e = arr[(ds, a)], arr[(ds, b)]
        res = {unit: {st: full(g[unit][st], e[unit][st],
                                [v6.SEED, 63, 1, v6.DATASET[ds], direction, v6.STAGE[st], v6.UNIT[unit]])
                      for st in v6.STAGE} for unit in v6.UNIT}
        check_v6(res, v6rec, name)
        rec["full"][name] = res
        rec["population"][name] = population(g, e, ds, direction)
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
    raw6 = cm.check_file(ROOT / c["v6"]["conditions"]["file"], c["v6"]["conditions"]["sha256"], "v6's conditions")
    rawr = cm.check_file(ROOT / c["v6"]["record"]["file"], c["v6"]["record"]["sha256"], "v6's record")
    c6, v6rec = json.loads(raw6), json.loads(rawr)
    if c6["generator"] != c["generator"]:
        raise SystemExit("the generator of v6's conditions differs from v6c's")
    m4 = c6["mapping"]["v4_conditions"]
    raw4 = cm.check_file(ROOT / m4["file"], m4["sha256"], "v4's conditions")
    c4 = json.loads(raw4)
    paths = [COND.relative_to(ROOT).as_posix(), c["v6"]["conditions"]["file"], c["v6"]["record"]["file"], m4["file"],
             *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    data = {"kupa_copy": c6["data"]["kupa"]["files"]["copy"], "kupa_essay": c6["data"]["kupa"]["files"]["essay"],
            "aalto": {"file": c6["data"]["aalto"]["file"], "sha256": c6["data"]["aalto"]["sha256"]}}
    for x in data.values():
        if cm.sha256(ROOT / x["file"]) != x["sha256"]:
            raise SystemExit(f"{x['file']} does not match the SHA-256 of v6's conditions")
    rec = run(c6, c4["mapping"], c4["relief"], v6rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "v6_conditions_sha256": hashlib.sha256(raw6).hexdigest(),
                         "v6_record_sha256": hashlib.sha256(rawr).hexdigest(),
                         "v4_conditions_sha256": hashlib.sha256(raw4).hexdigest(),
                         "data_sha256": {n: x["sha256"] for n, x in data.items()},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "atact_library_sha256": cm.library_sha256(),
                         "versions": cm.versions("numpy", "pillow", "scipy"),
                         "bit_generator": type(np.random.default_rng(0).bit_generator).__name__,
                         "check_against_v6": f"top1 and rank_auc of every direction, unit and stage within {CHECK_TOL}"}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6c-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    main_cell = rec["full"]["kupa_copy_to_essay"]["sentence"]
    print(json.dumps({st: {"rank_n": {k: round(v["rate"], 3) for k, v in x["rank_n"].items()},
                           "eer": round(x["eer"]["eer"], 3)} for st, x in main_cell.items()}))


if __name__ == "__main__":
    main()
