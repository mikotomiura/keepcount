"""Prototype v6s: the share of persons whose per-person verdict of count 1 is the same under the two fixed splits
(odd/even and halves), and only that (decision 40 of .steering/20261108-paper-a-jowr-reframe; decision 1 of
.steering/20261109-v6s-split-agreement-po). v6r's persons, stages, rows and verdicts (imported); on the common persons the
agreement p_o with its interval is written, unless the disagreement count would narrow the feasible sums of a
one-split-only set of 1-4 persons given v6n's rule-cell shares. No share over the common persons, table or chance level.

Usage:  uv run python works/prototype_v6s.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6s.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Output: results/prototype_v6s/record.json. No press time, gap, key, text, per-person vector, value, contribution,
        indicator, change or id is written.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import prototype_v6r as r6  # noqa: E402

q6, p6, o6, n6, m6, k6 = r6.q6, r6.p6, r6.o6, r6.n6, r6.m6, r6.k6
v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6s.json"
REC_N, REC_P, REC_Q = r6.REC_N, r6.REC_P, r6.REC_Q
REC_R = ROOT / "results" / "prototype_v6r" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6s"
CODE = ["works/prototype_v6s.py", *r6.CODE]
SEED = k6.SEED
TAG = 102
I2 = r6.I2
SPLITS = r6.SPLITS
STAGES_R = r6.STAGES_R
ROWS_R = r6.ROWS_R
PRIMARY_ROWS = r6.PRIMARY_ROWS
SHUFFLES = r6.SHUFFLES
READ_FIRST = r6.READ_FIRST
MIN_SUMMARY = r6.MIN_SUMMARY
PROTECT = MIN_SUMMARY - 1  # one-split-only sets of 1-4 persons are protected
TOL = r6.TOL
SUM_TOL = 1e-9


def skey(ds: str, task: str, st: str, row: str) -> list[int]:
    return [SEED, TAG, k6.DATASET[ds], k6.TASK[task], r6.STAGE_IX[st], n6.ROW_IX[row], r6.PAIR, r6.CELL, 0]


# ================================================================ what d tells about a one-split-only sum


def dset(d: int, x: int, y: int) -> bool:
    """Can d disagreeing persons (pairs of contributions in half units (i, j), i != j, i, j in {0, 1, 2}) have sums x
    (odd/even) and y (halves)? d = 0: only (0, 0); d = 1: the six pairs; d >= 2: 0 <= x, y <= 2d and d <= x + y <= 3d
    (checked against enumeration in the tests)."""
    if d == 0:
        return x == 0 and y == 0
    if d == 1:
        return (x, y) in {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
    return 0 <= x <= 2 * d and 0 <= y <= 2 * d and d <= x + y <= 3 * d


def joint(x2: int, y2: int, n_c: int, d: int) -> bool:
    """Can n_c common persons, d of them disagreeing, have sums x2 (odd/even) and y2 (halves) in half units? The
    agreeing persons add the same s to both sums, any integer in [0, 2 (n_c - d)]; the rest is dset (solved for s as an
    interval for d >= 2; checked against enumeration in the tests)."""
    if not 0 <= d <= n_c:
        return False
    hi_s = 2 * (n_c - d)
    if d == 0:
        return x2 == y2 and 0 <= x2 <= hi_s
    if d == 1:
        return any(0 <= x2 - a <= hi_s and x2 - a == y2 - b
                   for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))
    lo = max(0, x2 - 2 * d, y2 - 2 * d, -((3 * d - x2 - y2) // 2))
    hi = min(hi_s, x2, y2, (x2 + y2 - d) // 2)
    return lo <= hi


def pairs(h2: dict, m: dict, n_c: int, d: int | None) -> set[tuple[int, int]]:
    """Twice the feasible (S_oe, S_h) given the full-rule-set sums h2 (twice), the one-split-only counts m, the common
    persons n_c and, if d is not None, the disagreement count (exact joint feasibility; without d, X and Y only in
    [0, n_c])."""
    out = set()
    for s in range(2 * m["oddeven"] + 1):
        for t in range(2 * m["halves"] + 1):
            x2, y2 = h2["oddeven"] - s, h2["halves"] - t
            if 0 <= x2 <= 2 * n_c and 0 <= y2 <= 2 * n_c and (d is None or joint(x2, y2, n_c, d)):
                out.add((s, t))
    return out


def feasible(h2: dict, m: dict, n_c: int, d: int | None, split: str) -> set[int]:
    """Twice the feasible values of S_split (the projection of pairs)."""
    k = 0 if split == "oddeven" else 1
    return {p[k] for p in pairs(h2, m, n_c, d)}


def view(ps: set, m: dict) -> tuple:
    """What a reader learns about the protected one-split-only sums: the projection on each protected split, and the
    pairs when both are protected (code-reviewer M1)."""
    prot = [k for k, sp in enumerate(SPLITS) if 1 <= m[sp] <= PROTECT]
    if len(prot) == 2:
        return (frozenset(ps),)
    return tuple(frozenset(p[k] for p in ps) for k in prot)


def narrows_d(h2: dict, m: dict, n_c: int, d: int) -> bool:
    return view(pairs(h2, m, n_c, d), m) != view(pairs(h2, m, n_c, None), m)


def withhold(h2: dict, m: dict, n_c: int, d: int, true2: dict) -> bool:
    """Withhold the cell? Decided from published quantities so that withholding itself tells nothing (code-reviewer
    H1): W = the d in [0, n_c] that narrow the view; if the view over the union of W's feasible pairs is narrower than
    without d, the cell is withheld whatever d is; else it is withheld exactly when d is in W. The true sums are checked
    feasible with and without d."""
    true_pair = (true2["oddeven"], true2["halves"])
    without = pairs(h2, m, n_c, None)
    if true_pair not in pairs(h2, m, n_c, d) or true_pair not in without:
        raise SystemExit("the true one-split-only sum is not feasible")
    if not any(1 <= m[sp] <= PROTECT for sp in SPLITS):
        return False
    w = [dd for dd in range(n_c + 1) if narrows_d(h2, m, n_c, dd)]
    if not w:
        return False
    union = set().union(*(pairs(h2, m, n_c, dd) for dd in w))
    if view(union, m) != view(without, m):
        return True
    return d in w


def half2(x: float, where: str) -> int:
    """Twice a sum of contributions (0, 1/2, 1) as an integer."""
    v = 2 * x
    if not abs(v - round(v)) <= SUM_TOL:
        raise SystemExit(f"{where}: a sum of contributions is not a multiple of 1/2")
    return int(round(v))


# ================================================================ one cell


def agreement_po(a: np.ndarray, b: np.ndarray, key: list[int]) -> dict:
    n = len(a)
    if len(b) != n:
        raise SystemExit("the two splits' verdicts are not over the same persons")
    if not (np.all(np.isin(a, r6.CATS)) and np.all(np.isin(b, r6.CATS))):
        raise SystemExit("a contribution not 0, 1/2 or 1")
    agree = int(np.sum(a == b))
    eq = np.zeros(n)
    eq[:agree] = 1.0  # the indicator in a canonical order, so the interval depends on the counts alone
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    ci = v6.ci(eq, idx)
    if not p6.finite_ci(ci):
        raise SystemExit("a non-finite interval")
    po = agree / n
    if not abs(po - float(eq.mean())) <= TOL:
        raise SystemExit("p_o differs from the indicator")
    return {"persons_agree": agree, "persons_disagree": n - agree, "p_o": po, "p_o_ci": ci, "one_minus_p_o": 1.0 - po,
            "one_minus_p_o_ci": [1.0 - ci[1], 1.0 - ci[0]]}


def full_sum(v: dict, rn: dict, rq: dict, nm: str, how: str, row: str, st: str) -> float:
    """The sum of the split's contributions over its own rule persons, checked against share x persons of v6n (I[2]:
    v6q)."""
    own = v["own"]
    if st == I2:
        old = rq["cells"][nm][how][row]["rule"][I2]
    else:
        src = "I" if st in ("R_loc", "R_mean") else st
        part = "stage" if st == src else st
        old = rn["rule_persons"][nm][how][row][src][part]
    if not abs(float(own.sum()) - old["share_vs_own"] * len(own)) <= SUM_TOL:
        raise SystemExit(f"{nm} {how} {row} {st}: the sum differs from v6n's share x persons")
    return float(own.sum())


def set_cells(hv: dict, ds: str, task: str, rn: dict, rp: dict, rq: dict, rr: dict) -> dict:
    nm = f"{ds}_{task}"
    if len(hv["oddeven"]["text/before"]) != len(hv["halves"]["text/before"]):
        raise SystemExit(f"{nm}: the two splits do not index the same persons")
    hq = q6.with_i2(hv)
    q6.check_i2(hq)
    for how in SPLITS:
        r6.same_row(rp, rq, nm, how)
    out: dict = {}
    for row in ROWS_R:
        vs = {how: r6.verdicts(hv, hq, ds, task, how, row, rn, rq) for how in SPLITS}
        out[row] = {}
        for st in STAGES_R:
            a, b = vs["oddeven"]["stages"][st], vs["halves"]["stages"][st]
            common = sorted(set(a["persons"]) & set(b["persons"]))
            only = {"oddeven": sorted(set(a["persons"]) - set(b["persons"])),
                    "halves": sorted(set(b["persons"]) - set(a["persons"]))}
            e: dict = {"persons_common": len(common),
                       "persons_rule": {"oddeven": len(a["persons"]), "halves": len(b["persons"])},
                       "persons_one_split_only": {h: len(v) for h, v in only.items()}}
            old = rr["cells"][nm][row][st]
            if {k: old[k] for k in e} != e:
                raise SystemExit(f"{nm} {row} {st}: the common persons differ from v6r's record")
            if r6.is_fixed(st, row):
                out[row][st] = {**e, "fixed_by_construction": True}
                continue
            if a.get("no_count") or b.get("no_count"):
                out[row][st] = {**e, "no_count": True}
                continue
            if len(common) < MIN_SUMMARY:
                out[row][st] = {**e, "withheld": "fewer than 5"}
                continue
            pa = {p: k for k, p in enumerate(a["persons"])}
            pb = {p: k for k, p in enumerate(b["persons"])}
            ca = a["own"][[pa[p] for p in common]]
            cb = b["own"][[pb[p] for p in common]]
            res = agreement_po(ca, cb, skey(ds, task, st, row))
            where = f"{nm} {row} {st}"
            h2 = {how: half2(full_sum(vs[how]["stages"][st], rn, rq, nm, how, row, st), where) for how in SPLITS}
            true2 = {"oddeven": half2(float(a["own"][[pa[p] for p in only["oddeven"]]].sum()), where),
                     "halves": half2(float(b["own"][[pb[p] for p in only["halves"]]].sum()), where)}
            m = e["persons_one_split_only"]
            if withhold(h2, m, len(common), res["persons_disagree"], true2):
                out[row][st] = {**e, "withheld": "narrows a one-split-only sum"}
                continue
            out[row][st] = {**e, **res, "class_own_rule_persons": {"oddeven": a["class"], "halves": b["class"]},
                            "state": {"oddeven": a["state"], "halves": b["state"]}}
    return out


# ================================================================ predictions, summaries and main cells


NE = "not evaluable"


def written(e: dict) -> bool:
    return "p_o" in e


def predictions(rec: dict) -> dict:
    cells = rec["cells"]
    lw = cells["lwp_lwp"]
    out: dict = {}
    rl = [lw[row]["R_loc"] for row in ("inside_75", "inside_100")]
    out["P1"] = NE if not all(map(written, rl)) else (
        ("held" if all(4 * e["persons_agree"] >= 3 * e["persons_common"] for e in rl) else "did not hold")
        + " (" + ", ".join(f"{e['p_o']:.3f}" for e in rl) + ")")
    prim = [cells[nm][row][st] for nm in cells for row in PRIMARY_ROWS for st in STAGES_R]
    prim = [e for e in prim if written(e)]
    if not prim:
        out["P2"] = NE
    else:
        k = sum(5 * e["persons_disagree"] >= e["persons_common"] for e in prim)
        out["P2"] = ("held" if 2 * k >= len(prim) else "did not hold") + f" ({k} of {len(prim)})"
    ii = [lw[row]["I"] for row in ("inside_75", "inside_100")]
    out["P3"] = NE if not all(map(written, ii)) else (
        ("held" if all(5 * e["persons_agree"] < 4 * e["persons_common"] for e in ii) else "did not hold")
        + " (" + ", ".join(f"{e['p_o']:.3f}" for e in ii) + ")")
    return out


def r3(x: float) -> float:
    return round(x, 3)


def short(e: dict) -> dict | str:
    if not written(e):
        why = ("fixed" if e.get("fixed_by_construction") else "no count" if e.get("no_count") else e.get("withheld"))
        return f"not written ({why}; common {e['persons_common']})"
    return {"common": e["persons_common"], "1-p_o": r3(e["one_minus_p_o"]),
            "1-p_o_ci": [r3(x) for x in e["one_minus_p_o_ci"]], "class": e["class_own_rule_persons"],
            "state": e["state"]}


def summary(rec: dict) -> dict:
    out: dict = {}
    for name, rows in (("primary", PRIMARY_ROWS), ("shuffles", SHUFFLES)):
        es = [rec["cells"][nm][row][st] for nm in rec["cells"] for row in rows for st in STAGES_R]
        xs = [e["one_minus_p_o"] for e in es if written(e)]
        out[name] = {"written": len(xs), "fixed": sum(bool(e.get("fixed_by_construction")) for e in es),
                     "no_count": sum(bool(e.get("no_count")) for e in es),
                     "withheld_narrows": sum(e.get("withheld") == "narrows a one-split-only sum" for e in es),
                     "withheld_fewer_than_5": sum(e.get("withheld") == "fewer than 5" for e in es),
                     "one_minus_p_o": r6.spread(xs)}
    return out


def main_cells_out(rec: dict) -> dict:
    return {"read_first": {f"{nm} {row} {st}": short(rec["cells"][nm][row][st]) for nm, row, st in READ_FIRST},
            "primary": {f"{nm} {row}": {st: short(rec["cells"][nm][row][st]) for st in STAGES_R}
                        for nm in rec["cells"] for row in PRIMARY_ROWS},
            "shuffles": {f"{nm} {row}": {st: short(rec["cells"][nm][row][st]) for st in STAGES_R}
                         for nm in rec["cells"] for row in SHUFFLES},
            "summary": rec["summary"], "predictions": rec["predictions"]}


# ================================================================ main


def count_all(n_series: dict, n_ids: list, kplace: dict, ln_series: dict, ln_ids: list, lplace: dict, m: dict,
              rn: dict, rp: dict, rq: dict, rr: dict) -> dict:
    rec: dict = {"cells": {}, "tallies": {}}
    for ds, task, series, ids, place in (("kupa", "essay", n_series, n_ids, kplace),
                                         ("lwp", "lwp", ln_series, ln_ids, lplace)):
        nm = f"{ds}_{task}"
        b = n6.build(series, ids, ds, task, place, m, o6.BUILD_ROWS)
        if len(b["persons"]) != rn["tallies"][nm]["count_persons"]:
            raise SystemExit(f"{nm}: the count persons differ from v6n's")
        hv = n6.hv_of(b, o6.BUILD_ROWS)
        rec["cells"][nm] = set_cells(hv, ds, task, rn, rp, rq, rr)
        rec["tallies"][nm] = {"count_persons": len(b["persons"])}
        del hv, b
    rec["summary"] = summary(rec)
    rec["predictions"] = predictions(rec)
    rec["checks"] = {
        "reuse": "v6r's verdicts (v6n's rule cells recomputed for every stage, split and row; I[2] against v6q; the "
                 "persons with I[2] and R are I's rule persons; inside/before and inside_100 the same in v6p and v6q); "
                 "the common persons' counts equal v6r's record; the full-rule-set sums equal v6n's share x persons",
        "construction": "contributions 0, 1/2 or 1; agree + disagree = common; p_o from the indicator; the true "
                        "one-split-only sums feasible with and without d",
        "sets": "p_o written only where d does not narrow the feasible sums of a one-split-only set of 1-4 persons"}
    return rec


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    for p, h in c["earlier"].items():
        cm.check_file(ROOT / p, h, p)
    ck = json.loads((ROOT / "works/conditions/prototype_v6k.json").read_bytes())
    cm.check_generator(ck["generator"])
    from importlib import metadata
    for pkg in ("numpy", "pillow", "scipy"):
        if metadata.version(pkg) != c[pkg]:
            raise SystemExit(f"{pkg} is {metadata.version(pkg)}, not {c[pkg]}")
    c6 = json.loads((ROOT / "works/conditions/prototype_v6.json").read_bytes())
    c4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_bytes())
    cm.check_file(ROOT / "works/conditions/prototype_v4.json", ck["earlier"]["works/conditions/prototype_v4.json"],
                  "v4's conditions")
    d_lwp = json.loads((ROOT / "works/conditions/prototype_v6e.json").read_text())["data"]["lwp"]
    kp = c6["data"]["kupa"]["files"]
    for f, h in [(d_lwp["file"], d_lwp["sha256"])] + [(x["file"], x["sha256"]) for x in kp.values()]:
        if cm.sha256(ROOT / f) != h:
            raise SystemExit(f"{f} does not match its SHA-256")
    rn, rp, rq, rr = (json.loads(p.read_text()) for p in (REC_N, REC_P, REC_Q, REC_R))
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    m = c4["mapping"]
    rec: dict = {"persons": {}}
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6b.read_kupa_b(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    kplace = {pid: i for i, pid in enumerate(ids_sorted)}
    order = v6.draw_order(ids_sorted, 0)
    elig = lambda pid: k6.kupa_eligible(kupa, pid)  # noqa: E731
    n_ids, n_series = m6.draw_checked(order, elig, rn["persons"]["kupa"]["ids_looked_at_from_656"], m6.SKIP["kupa"],
                                      rn["persons"]["kupa"]["ids_sha256"], "v6n's KUPA persons")
    rec["persons"]["kupa"] = rn["persons"]["kupa"]
    del kupa
    lwp, lids, _ = v6e.read_lwp(ROOT / d_lwp["file"], None)
    lplace = {pid: i for i, pid in enumerate(sorted(lids))}
    lorder = v6.draw_order(sorted(lids), 2)
    lelig = lambda pid: k6.lwp_eligible(lwp, pid)  # noqa: E731
    ln_ids, ln_series = m6.draw_checked(lorder, lelig, rn["persons"]["lwp"]["ids_looked_at_from_411"], m6.SKIP["lwp"],
                                        rn["persons"]["lwp"]["ids_sha256"], "v6n's LWP persons")
    rec["persons"]["lwp"] = rn["persons"]["lwp"]
    del lwp
    rec.update(n6.jsonish(count_all(n_series, n_ids, kplace, ln_series, ln_ids, lplace, m, rn, rp, rq, rr)))
    rec = {"main_cells": main_cells_out(rec), **rec}
    v6d.no_per_person(rec)
    v6i.no_long_list(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "data_sha256": {"kupa_copy": kp["copy"]["sha256"], "kupa_essay": kp["essay"]["sha256"],
                                         "lwp": d_lwp["sha256"]},
                         "choices_in_code": CHOICES, "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6s-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    print(json.dumps(rec["summary"]))
    for k, v in rec["main_cells"]["read_first"].items():
        print(k, json.dumps(v))


CHOICES = {
    "withhold_from_published": "decision 2: whether a cell is withheld is decided over all d in [0, n_c] from published "
                               "quantities (code-reviewer H1), and both one-split-only sums are compared as pairs when "
                               "both are protected (M1)",
    "joint_feasibility": "decision 2 (before the run): the conditions check X - Y in D_d; the code checks the exact joint "
                         "feasibility of X, Y and d on the common persons (stricter; Codex's counterexample)",
    "canonical_indicator": "decision 2: p_o's interval is v6.ci of the agreement indicator sorted (agreeing first), so "
                           "the interval depends on n and the agreement count alone",
    "half_units": "sums of contributions are handled as twice their value (integers); D_d by enumeration",
    "narrows_order": "the narrowing check runs after p_o is computed (p_o is not written if withheld)",
    "beside": "class and state per split are v6r's verdicts' (v6n's rule cell of the stage; I[2], R_loc, R_mean: I's)",
}


if __name__ == "__main__":
    main()
