"""Prototype v6r: how much does the per-person verdict of count 1 depend on which fixed two-part split builds it,
odd/even sentences or halves? (R3 m1 of the round-8 pre-review; decision 37 of .steering/20261108-paper-a-jowr-reframe;
decision 1 of .steering/20261109-v6r-split-agreement). On v6n's persons, per dataset, row and stage, the persons who are
rule persons under both splits (the common persons) have their two verdicts (1, 1/2 or 0; each split's own cell scale,
exactly the verdicts inside v6n's shares) cross-tabulated: agreement p_o with its interval, the 3 x 3 table, the chance
agreement p_e from the two marginal distributions and Cohen's kappa. A common set that would make a new set of 1-4 persons
linearly derivable from the ledgers of v6n-v6q is withheld. All descriptions.

Usage:  uv run python works/prototype_v6r.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6r.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6r/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, value, contribution, change or id is written.
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
import prototype_v6q as q6  # noqa: E402

p6 = q6.p6
o6, n6, m6, k6 = p6.o6, p6.n6, p6.m6, p6.k6
v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6r.json"
REC_N = ROOT / "results" / "prototype_v6n" / "record.json"
REC_O = ROOT / "results" / "prototype_v6o" / "record.json"
REC_P = ROOT / "results" / "prototype_v6p" / "record.json"
REC_Q = ROOT / "results" / "prototype_v6q" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6r"
CODE = ["works/prototype_v6r.py", *q6.CODE]
SEED = k6.SEED
TAG = 101
PAIR = 7  # the pair of splits in the keys (v6b's baselines are 0 and 1)
CELL = 9  # a whole cell
I2 = q6.I2
SPLITS = ("oddeven", "halves")
STAGES_R = ("U@2000", "B@2000", "Bsd@2000", "I", I2, "R_loc", "R_mean")  # the order of admission to the ledgers
STAGE_IX = {"U@2000": 2, "B@2000": 3, "Bsd@2000": 4, "I": 5, "R_loc": 8, "R_mean": 12, I2: q6.I2_IX}
ROWS_R = ("inside_75", "inside_100", "cross_100", "shuffled/before", "shuffled_in_halves/before")
PRIMARY_ROWS = p6.PRIMARY_ROWS
SHUFFLES = q6.SHUFFLES
READ_FIRST = tuple(("lwp_lwp", row, st) for row in ("inside_75", "inside_100") for st in ("I", I2, "R_loc"))
CATS = (0.0, 0.5, 1.0)
LAB = ("0", "1/2", "1")
MIN_SUMMARY = p6.MIN_SUMMARY
TOL = p6.TOL
ID_TOL = 1e-9


def rkey(ds: str, task: str, st: str, row: str) -> list[int]:
    return [SEED, TAG, k6.DATASET[ds], k6.TASK[task], STAGE_IX[st], n6.ROW_IX[row], PAIR, CELL, 0]


def is_fixed(st: str, row: str) -> bool:
    """Fixed by construction under the row in either split (v6p's fixed; I[2], R_loc and R_mean are never fixed)."""
    return st in p6.STAGES and any(p6.fixed(st, row, how) for how in SPLITS)


# ================================================================ the agreement of two verdicts


def table_of(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The 3 x 3 counts: rows odd/even's category, columns halves' (0, 1/2, 1)."""
    if not (np.all(np.isin(a, CATS)) and np.all(np.isin(b, CATS))):
        raise SystemExit("a contribution not 0, 1/2 or 1")
    return np.array([[int(np.sum((a == x) & (b == y))) for y in CATS] for x in CATS])


def chance(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """p_e from the category proportions (last axis the three categories)."""
    return np.sum(pa * pb, axis=-1)


def agreement(a: np.ndarray, b: np.ndarray, key: list[int]) -> dict:
    """Agreement of the two splits' verdicts on the common persons (the conditions' cells.agreement)."""
    n = len(a)
    if len(b) != n:
        raise SystemExit("the two splits' verdicts are not over the same persons")
    t = table_of(a, b)
    if int(t.sum()) != n:
        raise SystemExit("the table does not add to the common persons")
    po = float(np.trace(t)) / n
    pa, pb = t.sum(axis=1) / n, t.sum(axis=0) / n
    pe = float(chance(pa, pb))
    sa, sb = float(a.mean()), float(b.mean())
    if not (all(abs(pa[k] - float(np.mean(a == x))) <= TOL and abs(pb[k] - float(np.mean(b == x))) <= TOL
                for k, x in enumerate(CATS)) and abs(po - float(np.mean(a == b))) <= TOL and abs(sa - float(pa @ np.array(CATS))) <= TOL
            and abs(sb - float(pb @ np.array(CATS))) <= TOL):
        raise SystemExit("p_o or a share differs from the table")
    lhs = (sa - sb) * n
    rhs = (t[2, 0] - t[0, 2]) + 0.5 * ((t[2, 1] + t[1, 0]) - (t[1, 2] + t[0, 1]))
    if not abs(lhs - rhs) <= ID_TOL:
        raise SystemExit("the identity of the shares and the disagreements does not hold")
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    eq = (a == b).astype(float)
    po_ci = v6.ci(eq, idx)
    A, B = a[idx], b[idx]
    po_b = (A == B).mean(axis=1)
    if not np.array_equal(po_b, eq[idx].mean(axis=1)):
        raise SystemExit("the resampled agreement differs from v6.ci's")
    pe_b = chance(np.stack([(A == x).mean(axis=1) for x in CATS], axis=-1),
                  np.stack([(B == x).mean(axis=1) for x in CATS], axis=-1))
    out: dict = {
        "table": {f"{LAB[i]}->{LAB[j]}": int(t[i, j]) for i in range(3) for j in range(3)},
        "1->0": int(t[2, 0]), "0->1": int(t[0, 2]),
        "with_1/2": int(t[2, 1] + t[1, 0] + t[1, 2] + t[0, 1]),
        "share": {"oddeven": sa, "halves": sb},
        "proportions": {"oddeven": [float(x) for x in pa], "halves": [float(x) for x in pb]},
        "persons_agree": int(np.trace(t)), "persons_disagree": n - int(np.trace(t)),
        "p_o": po, "p_o_ci": po_ci, "one_minus_p_o": 1.0 - po, "p_e": pe}
    if not p6.finite_ci(po_ci):
        raise SystemExit("a non-finite interval")
    if pe < 1.0:
        out["kappa"] = (po - pe) / (1.0 - pe)
        undefined = int(np.sum(pe_b >= 1.0))
        if undefined == 0:
            kb = (po_b - pe_b) / (1.0 - pe_b)
            out["kappa_ci"] = [float(np.percentile(kb, 2.5)), float(np.percentile(kb, 97.5))]
            if not p6.finite_ci(out["kappa_ci"]):
                raise SystemExit("a non-finite interval")
        else:
            out["kappa_resamples_p_e_1"] = undefined
    else:
        out["kappa_not_written"] = "p_e = 1"
    return out


# ================================================================ one split: verdicts and ledgers


def state_of(old_n: dict) -> str:
    """The gate state of a v6n rule cell from its R_loc's and R_mean's classes (decision 25)."""
    return q6.cell_state(o6.size_class(old_n["R_loc"]["share_vs_own_ci"]),
                         o6.size_class(old_n["R_mean"]["share_vs_own_ci"]))


def verdicts(hv: dict, hq: dict, ds: str, task: str, how: str, row: str, rn: dict, rq: dict) -> dict:
    """Per stage of STAGES_R: the persons and contributions of the split (v6n's rule cells recomputed and checked by
    v6p's rule_keep; I[2] on I's rule persons checked against v6q's record), and the rules for the ledgers."""
    nm = f"{ds}_{task}"
    rules = {st: p6.rule_keep(hv, ds, task, how, row, st, rn) for st in p6.STAGES}
    out: dict = {"rules": rules, "stages": {}}
    for st in STAGES_R:
        src = "I" if st in (I2, "R_loc", "R_mean") else st
        rule = rules[src]
        if rule.get("no_count"):
            out["stages"][st] = {"persons": rule["persons"], "no_count": True}
            continue
        if st == I2:
            where = f"{nm} {how} {row} I[2]"
            if p6.persons_with(hq, how, row, (I2, "R")) != rule["persons"]:
                raise SystemExit(f"{where}: the persons with I[2] and R differ from I's rule persons")
            first = [hq[how]["text/before"][i] for i in rule["persons"]]
            sel = [hq[how][row][i] for i in rule["persons"]]
            own = q6.contrib(first, sel, I2, where)
            old = rq["cells"][nm][how][row]["rule"]
            if "withheld" in old or old.get("no_count") or old.get("persons") != len(rule["persons"]):
                raise SystemExit(f"{where}: v6q did not write I[2] on I's rule persons")
            idx = np.random.default_rng(q6.qkey(ds, task, row, how, q6.CELL, q6.LAST["rule"])).integers(
                0, len(own), (v6.N_BOOT, len(own)))
            if not (abs(float(own.mean()) - old[I2]["share_vs_own"]) <= TOL and v6.ci(own, idx) == old[I2]["share_vs_own_ci"]):
                raise SystemExit(f"{where}: I[2]'s contributions do not give v6q's share and interval")
        else:
            own = rule["stage"] if st == src else rule[st]
        old_n = rn["rule_persons"][nm][how][row][src]
        part = "stage" if st in (src, I2) else st
        cls = (o6.size_class(rq["cells"][nm][how][row]["rule"][I2]["share_vs_own_ci"]) if st == I2
               else o6.size_class(old_n[part]["share_vs_own_ci"]))
        out["stages"][st] = {"persons": rule["persons"], "own": own, "class": cls,
                             "state": state_of(old_n)}
    return out


def ledgers(hv: dict, how: str, row: str, gate0: list[int], o_fams: list[np.ndarray], lab_how: dict,
            written_o: dict, rules: dict, nm: str, rn: dict, rp: dict, rq: dict, n: int) -> dict:
    """The ledgers of the split and row as v6p and v6q left them: the pooled ledger and each stage's (v6p's base sets
    and written families; v6p's plan rebuilt and checked against its record), and I[2]'s own ledger (v6q)."""
    base = p6.base_sets(hv, how, row, gate0, o_fams, rn["count_1"][nm][how], rules, n)
    plan, _ = p6.plan_row(base, how, row, gate0, lab_how, written_o, rules, n)
    if plan != rp["sets"][nm]["written"][how][row]:
        raise SystemExit(f"{nm} {how} {row}: v6p's plan rebuilt differs from its record")
    pos = {p: k for k, p in enumerate(gate0)}
    pool = p6.pooled(base)
    stage: dict = {}
    for st in p6.STAGES:
        led = p6.stage_ledger(base, st)
        for c in p6.COVARIATES:
            if plan[st].get(c) == "written":
                fam = p6.family(rules[st]["persons"], lab_how[c][2][[pos[p] for p in rules[st]["persons"]]], n)
                led += fam
                pool += fam
        stage[st] = led
    own: list[np.ndarray] = []
    ri = rules["I"]
    qc = rq["cells"][nm][how][row]
    if not ri.get("no_count") and "withheld" not in qc["rule"] and not qc["rule"].get("no_count"):
        own.append(p6.mask_of(ri["persons"], n))
        for c in p6.COVARIATES:
            if rq["groups"][c][nm][how][row].get("written"):
                own += p6.family(ri["persons"], p6.group_of_stage(ri, gate0, lab_how, c), n)
    if "withheld" not in qc["gate_0"] and not qc["gate_0"].get("no_count"):
        own.append(p6.mask_of(p6.persons_with(hv, how, row, ("I",)), n))
    return {"pool": pool, "stage": stage, I2: own}


def ledger_of(led: dict, st: str) -> list[list[np.ndarray]]:
    """The ledgers a common set of the stage must keep clean: the pooled and the stage's (R_loc and R_mean: I's), or
    I[2]'s own."""
    if st == I2:
        return [led[I2]]
    return [led["pool"], led["stage"]["I" if st in ("R_loc", "R_mean") else st]]


def new_small(sets: list[np.ndarray], extra: list[np.ndarray]) -> bool:
    """Would adding extra make a set of 1-4 persons derivable that was not before (v6p's small_sets)?"""
    before = p6.small_sets(sets) if sets else set()
    return bool(p6.small_sets(sets + extra) - before)


def admit_common(leds: dict, st: str, mask: np.ndarray) -> bool:
    """Admit the common set under both splits, or under neither."""
    for how in SPLITS:
        if any(new_small(s, [mask]) for s in ledger_of(leds[how], st)):
            return False
    for how in SPLITS:
        for s in ledger_of(leds[how], st):
            s.append(mask)
    return True


def same_row(rp: dict, rq: dict, nm: str, how: str) -> None:
    """inside/before is the same row as inside_100: v6p's plan and v6q's written sets must be the same in both, so
    that the ledger rebuilt for inside_100 holds every set written for this quantity."""
    a, b = "inside/before", "inside_100"
    if rp["sets"][nm]["written"][how][a] != rp["sets"][nm]["written"][how][b]:
        raise SystemExit(f"{nm} {how}: v6p's plans for inside/before and inside_100 differ")

    def mark(e: dict) -> str:
        return "withheld" if "withheld" in e else "no_count" if e.get("no_count") else "written"

    for part in ("rule", "gate_0"):
        if mark(rq["cells"][nm][how][a][part]) != mark(rq["cells"][nm][how][b][part]):
            raise SystemExit(f"{nm} {how}: v6q's {part} differs between inside/before and inside_100")
    for c in p6.COVARIATES:
        if bool(rq["groups"][c][nm][how][a].get("written")) != bool(rq["groups"][c][nm][how][b].get("written")):
            raise SystemExit(f"{nm} {how} {c}: v6q's families differ between inside/before and inside_100")


# ================================================================ one set


def set_cells(hv: dict, cov: list[dict], ds: str, task: str, rn: dict, ro: dict, rp: dict, rq: dict,
              kupa_cuts: dict | None) -> tuple[dict, dict]:
    nm = f"{ds}_{task}"
    if len(hv["oddeven"]["text/before"]) != len(hv["halves"]["text/before"]):
        raise SystemExit(f"{nm}: the two splits do not index the same persons")
    n = len(hv["oddeven"]["text/before"])
    hq = q6.with_i2(hv)
    q6.check_i2(hq)
    keep = o6.set_keep(hv, cov, ds, task, rn)
    lab = o6.labels_of(keep)
    p6.check_v6o_groups(lab, ro, nm)
    blab = p6.band_labels(keep, ro, kupa_cuts) if kupa_cuts is not None else None
    setup = {}
    for how in SPLITS:
        gate0 = o6.cell_persons(hv, how, p6.ROWS_O[0], "R")
        written_o = {c: bool(ro["written"][how][nm][c]) for c in p6.COVARIATES}
        g0 = np.array(gate0)
        o_fams = [p6.mask_of(g0[lab[how][c][2] == j], n) for c in p6.COVARIATES if written_o[c] for j in range(3)]
        if blab is not None:
            o_fams += [p6.mask_of(g0[blab[how][c] == j], n) for c in p6.COVARIATES if ro["written"][how]["bands"][c]
                       for j in range(3)]
        setup[how] = (gate0, written_o, o_fams)
    for how in SPLITS:
        same_row(rp, rq, nm, how)
    out: dict = {}
    for row in ROWS_R:
        vs, leds, before = {}, {}, {}
        for how in SPLITS:
            gate0, written_o, o_fams = setup[how]
            vs[how] = verdicts(hv, hq, ds, task, how, row, rn, rq)
            leds[how] = ledgers(hv, how, row, gate0, o_fams, lab[how], written_o, vs[how]["rules"], nm, rn, rp, rq, n)
            before[how] = {"pool": p6.small_sets(leds[how]["pool"]),
                           **{st: p6.small_sets(s) for st, s in leds[how]["stage"].items()},
                           I2: p6.small_sets(leds[how][I2]) if leds[how][I2] else set()}
        out[row] = {}
        for st in STAGES_R:
            a, b = vs["oddeven"]["stages"][st], vs["halves"]["stages"][st]
            common = sorted(set(a["persons"]) & set(b["persons"]))
            e: dict = {"persons_common": len(common),
                       "persons_rule": {"oddeven": len(a["persons"]), "halves": len(b["persons"])},
                       "persons_one_split_only": {"oddeven": len(set(a["persons"]) - set(b["persons"])),
                                                  "halves": len(set(b["persons"]) - set(a["persons"]))}}
            if st == I2 and common != out[row]["I"].get("_common"):
                raise SystemExit(f"{nm} {row}: I[2]'s common persons differ from I's")
            if st == "I":
                e["_common"] = common
            if is_fixed(st, row):
                out[row][st] = {**e, "fixed_by_construction": True}
                continue
            if a.get("no_count") or b.get("no_count"):
                out[row][st] = {**e, "no_count": True}
                continue
            if len(common) < MIN_SUMMARY:
                out[row][st] = {**e, "withheld": "fewer than 5"}
                continue
            if not admit_common(leds, st, p6.mask_of(common, n)):
                out[row][st] = {**e, "withheld": "derivable"}
                continue
            pa = {p: k for k, p in enumerate(a["persons"])}
            pb = {p: k for k, p in enumerate(b["persons"])}
            ca = a["own"][[pa[p] for p in common]]
            cb = b["own"][[pb[p] for p in common]]
            out[row][st] = {**e, **agreement(ca, cb, rkey(ds, task, st, row)),
                            "class_own_rule_persons": {"oddeven": a["class"], "halves": b["class"]},
                            "state": {"oddeven": a["state"], "halves": b["state"]}}
        for st in STAGES_R:
            out[row][st].pop("_common", None)
        for how in SPLITS:
            l = leds[how]
            if p6.small_sets(l["pool"]) - before[how]["pool"]:
                raise SystemExit(f"{nm} {how} {row}: a new small set is derivable from the pooled ledger")
            for st, s in l["stage"].items():
                if p6.small_sets(s) - before[how][st]:
                    raise SystemExit(f"{nm} {how} {row} {st}: a new small set is derivable from the stage's ledger")
            if l[I2] and p6.small_sets(l[I2]) - before[how][I2]:
                raise SystemExit(f"{nm} {how} {row}: a new small set is derivable from I[2]'s own ledger")
    cuts = {how: {c: lab[how][c][:2] for c in p6.COVARIATES} for how in SPLITS}
    return out, cuts


# ================================================================ predictions, summaries and main cells


NE = "not evaluable"


def written(e: dict) -> bool:
    return "p_o" in e


def predictions(rec: dict) -> dict:
    cells = rec["cells"]
    out: dict = {}
    lw = cells["lwp_lwp"]
    rl = [lw[row]["R_loc"] for row in ("inside_75", "inside_100")]
    out["P1"] = NE if not all(map(written, rl)) else (
        ("held" if all(4 * e["persons_agree"] >= 3 * e["persons_common"] for e in rl) else "did not hold")
        + " (" + ", ".join(f"{e['p_o']:.3f}" for e in rl) + ")")
    prim = [cells[nm][row][st] for nm in cells for row in PRIMARY_ROWS for st in STAGES_R]
    prim = [e for e in prim if written(e) and "kappa" in e]
    if not prim:
        out["P2"] = NE
    else:
        k = sum(5 * e["persons_disagree"] >= e["persons_common"] for e in prim)
        out["P2"] = ("held" if 2 * k >= len(prim) else "did not hold") + f" ({k} of {len(prim)})"
    ii = [lw[row]["I"] for row in ("inside_75", "inside_100")]
    out["P3"] = NE if not all(map(written, ii)) else (
        ("held" if all(5 * e["persons_agree"] < 4 * e["persons_common"] for e in ii) else "did not hold")
        + " (" + ", ".join(f"{e['p_o']:.3f}" for e in ii) + ")")
    a, b = lw["inside_100"][I2], lw["inside_100"]["I"]
    out["P4"] = NE if not ("kappa" in a and "kappa" in b) else (
        ("held" if a["kappa"] < b["kappa"] else "did not hold") + f" (I[2] {a['kappa']:.3f}, I {b['kappa']:.3f})")
    return out


def r3(x: float) -> float:
    return round(x, 3)


def short(e: dict) -> dict | str:
    if not written(e):
        why = ("fixed" if e.get("fixed_by_construction") else "no count" if e.get("no_count") else e.get("withheld"))
        return (f"not written ({why}; common {e['persons_common']}, rule {e['persons_rule']['oddeven']}/"
                f"{e['persons_rule']['halves']})")
    k = (f"{r3(e['kappa'])} [{r3(e['kappa_ci'][0])}, {r3(e['kappa_ci'][1])}]" if "kappa_ci" in e
         else f"{r3(e['kappa'])} (no interval: p_e = 1 in {e['kappa_resamples_p_e_1']} resamples)" if "kappa" in e
         else f"not written ({e['kappa_not_written']})")
    return {"common": e["persons_common"], "rule": e["persons_rule"], "one_split_only": e["persons_one_split_only"],
            "1-p_o": r3(e["one_minus_p_o"]), "p_o_ci": [r3(x) for x in e["p_o_ci"]], "p_e": r3(e["p_e"]),
            "kappa": k, "share": {h: r3(v) for h, v in e["share"].items()}, "table": e["table"],
            "class": e["class_own_rule_persons"], "state": e["state"]}


def spread(xs: list[float]) -> dict:
    return {"cells": len(xs)} if not xs else {"cells": len(xs), "min": r3(min(xs)), "median": r3(float(np.median(xs))),
                                              "max": r3(max(xs))}


def summary(rec: dict) -> dict:
    out: dict = {}
    for name, rows in (("primary", PRIMARY_ROWS), ("shuffles", SHUFFLES)):
        es = [rec["cells"][nm][row][st] for nm in rec["cells"] for row in rows for st in STAGES_R]
        out[name] = {"written": sum(map(written, es)),
                     "fixed": sum(bool(e.get("fixed_by_construction")) for e in es),
                     "no_count": sum(bool(e.get("no_count")) for e in es),
                     "withheld_derivable": sum(e.get("withheld") == "derivable" for e in es),
                     "withheld_fewer_than_5": sum(e.get("withheld") == "fewer than 5" for e in es),
                     "one_minus_p_o": spread([e["one_minus_p_o"] for e in es if written(e)]),
                     "kappa": spread([e["kappa"] for e in es if "kappa" in e])}
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
              rn: dict, ro: dict, rp: dict, rq: dict) -> dict:
    rec: dict = {"cells": {}, "tallies": {}}
    kupa_cuts = None
    for ds, task, series, ids, place in (("kupa", "essay", n_series, n_ids, kplace),
                                         ("lwp", "lwp", ln_series, ln_ids, lplace)):
        nm = f"{ds}_{task}"
        b = n6.build(series, ids, ds, task, place, m, o6.BUILD_ROWS)
        if len(b["persons"]) != rn["tallies"][nm]["count_persons"]:
            raise SystemExit(f"{nm}: the count persons differ from v6n's")
        cov = []
        for pid in b["persons"]:
            s, kinds = series[pid][task] if ds == "kupa" else series[pid]
            cov.append(o6.covariates_of(s, kinds))
        hv = n6.hv_of(b, o6.BUILD_ROWS)
        rec["cells"][nm], cuts = set_cells(hv, cov, ds, task, rn, ro, rp, rq, kupa_cuts)
        if ds == "kupa":
            kupa_cuts = cuts
        rec["tallies"][nm] = {"count_persons": len(b["persons"])}
        del hv, b
    rec["summary"] = summary(rec)
    rec["predictions"] = predictions(rec)
    rec["checks"] = {
        "reuse": "v6n's persons and count persons; per set, split and row, v6n's rule cell of every stage (stage, R_loc, "
                 "R_mean: persons, share, interval; paired difference) recomputed by v6p's rule_keep; I[2]'s "
                 "contributions give v6q's share and interval; v6o's cuts and groups; v6p's plan rebuilt equal to its "
                 "record",
        "construction": "the two splits index the same count persons; contributions 0, 1/2 or 1; the table adds to "
                        "the common persons; p_o and the shares recomputed from the table; the identity of the shares "
                        "and the disagreements (1e-9); I[2]'s common persons equal I's",
        "sets": "a cell is written only if its common set is admitted under both splits (no new set of 1-4 persons "
                "derivable from the pooled ledger and the stage's, or I[2]'s own ledger, as v6p and v6q left them); "
                "the whole ledgers checked again per split and row"}
    return rec


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    if not OUT_DIR.parent.is_dir():
        raise SystemExit(f"{OUT_DIR.parent} does not exist")
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
    rn = json.loads(REC_N.read_text())
    ro = json.loads(REC_O.read_text())
    rp = json.loads(REC_P.read_text())
    rq = json.loads(REC_Q.read_text())
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

    rec.update(n6.jsonish(count_all(n_series, n_ids, kplace, ln_series, ln_ids, lplace, m, rn, ro, rp, rq)))
    rec = {"main_cells": main_cells_out(rec), **rec}
    v6d.no_per_person(rec)
    v6i.no_long_list(rec)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(), "earlier_sha256": c["earlier"],
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "data_sha256": {"kupa_copy": kp["copy"]["sha256"], "kupa_essay": kp["essay"]["sha256"],
                                         "lwp": d_lwp["sha256"]},
                         "choices_in_code": CHOICES,
                         "versions": cm.versions("numpy", "pillow", "scipy")}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6r-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    print(json.dumps(rec["summary"]))
    for k, v in rec["main_cells"]["read_first"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "refs_persons": "R_loc's and R_mean's verdicts are those on I's rule persons (v6p's rule_keep of I), and their "
                    "common set is admitted against the pooled ledger and I's stage ledger",
    "i2_ledger": "I[2]'s own ledger is rebuilt from v6q's record: I's rule persons, I's families that v6q wrote, and "
                 "the gate 0 persons, each where v6q wrote them; a common set of I[2] is checked against it only",
    "admission_order": "per row the stages are admitted in the order U, B, Bsd, I, I[2], R_loc, R_mean; a fixed, "
                       "no-count or fewer-than-5 cell adds nothing to the ledgers",
    "both_or_neither": "a common set is added to the ledgers of both splits or of neither",
    "key": "the agreement's resamples (p_o's interval and kappa's) use rkey; p_e is recomputed in each resample",
    "kappa_interval": "kappa's interval is written only if p_e < 1 in all 2000 resamples; else the number with p_e = 1",
    "beside": "class_own_rule_persons is v6n's class of the stage on each split's own rule persons (I[2]: v6q's; R_loc "
              "and R_mean: on I's rule persons) and state is the gate state of the stage's own rule cell per split (I[2], "
              "R_loc and R_mean: I's); not recomputed on the common persons",
    "integer_thresholds": "the predictions compare counts (persons agreeing or disagreeing against the common persons) "
                          "instead of floating shares, so that 1 - p_o = 0.2 is not lost to rounding",
    "same_index": "the two splits' vectors come from one n6.build of the count persons, so they index the same persons "
                  "by construction; the run checks the lengths",
    "v6q_withheld_stops": "if v6q withheld or did not count I[2] on I's rule persons in a counted row, the run stops "
                          "(I[2]'s verdicts would not be v6q's); intended",
    "predictions_not_evaluable": "a prediction is 'not evaluable' if a cell it needs is not written",
}


if __name__ == "__main__":
    main()
