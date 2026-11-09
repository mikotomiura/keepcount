"""Prototype v6q: does I's negative result under inside remain when the own distance is built from I's word-start
mean I[2] alone? (R1 M1 and Codex 2 of the round-7 pre-review; decision 33 of .steering/20261108-paper-a-jowr-reframe;
decision 1 of .steering/20261109-v6q-i2-only-distance). On v6n's persons, per dataset, baseline and row, the share of
persons whose row lies beyond their own parts is counted with I[2] alone on I's rule persons (v6n's rule cell of I) and,
beside, on I's count 1 persons; it is read through the gate R_loc (decision 25) and split by v6p's groups of sentence
starts per part. I (three numbers) is written beside, checked equal to v6n's and v6p's records. All descriptions.

Usage:  uv run python works/prototype_v6q.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6q.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6q/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value, n_s, change or id is written.
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
import prototype_v6p as p6  # noqa: E402

o6, n6, m6, k6 = p6.o6, p6.n6, p6.m6, p6.k6
v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6q.json"
REC_N = ROOT / "results" / "prototype_v6n" / "record.json"
REC_O = ROOT / "results" / "prototype_v6o" / "record.json"
REC_P = ROOT / "results" / "prototype_v6p" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6q"
CODE = ["works/prototype_v6q.py", *p6.CODE]
SEED = k6.SEED
TAG = 100
I2 = "I[2]"
I2_IX = 22  # stage index of I[2] in the keys (v6n's STAGE goes to 12)
CELL = 9  # group index of a whole cell in the keys
LAST = {"rule": 0, "gate_0": 1}
ROWS_O = p6.ROWS_O
COUNTED_ROWS = p6.COUNTED_ROWS
COVARIATES = p6.COVARIATES
GROUPS = o6.GROUPS
MIN_SUMMARY = p6.MIN_SUMMARY
PRIMARY_ROWS = p6.PRIMARY_ROWS
GATE_CELLS = (("lwp_lwp", "oddeven", "inside_100"), ("lwp_lwp", "halves", "inside_75"),
              ("lwp_lwp", "halves", "inside_100"))  # named in the conditions (by v6n's rule cells)
SHUFFLES = ("shuffled/before", "shuffled_in_halves/before")
TOL = p6.TOL
FIX_TOL = 1e-9  # seconds; I[0] and I[1] under inside
STATES = p6.STATES
CELL_STATES = ("gate", "R_mean only", "not read")
CLASSES = p6.CLASSES


def is_inside(row: str) -> bool:
    return row == "inside/before" or row.startswith("inside_")


def qkey(ds: str, task: str, row: str, how: str, group: int, last: int) -> list[int]:
    return [SEED, TAG, k6.DATASET[ds], k6.TASK[task], I2_IX, n6.ROW_IX[row], v6b.BASELINE[how], group, last]


def cell_state(loc: str, mean: str) -> str:
    """The gate rule (decision 25) on a whole cell: R_loc's and R_mean's classes on the rule persons."""
    return "gate" if loc == "above" else ("R_mean only" if mean == "above" else "not read")


# ================================================================ I[2] beside I


def with_i2(hv: dict) -> dict:
    """The same vectors with I[2] (I's third number, as a vector of one) beside; None where I is None."""
    out: dict = {}
    for how, rows in hv.items():
        out[how] = {}
        for row, pairs in rows.items():
            new = []
            for pair in pairs:
                two = []
                for h in (0, 1):
                    d = dict(pair[h])
                    i = d.get("I")
                    d[I2] = None if i is None else np.array([float(np.asarray(i, dtype=float)[2])])
                    two.append(d)
                new.append(tuple(two))
            out[how][row] = new
    return out


def check_i2(hv: dict) -> None:
    """I[2] equals I's third number for every person, part and row, and exists exactly when I does."""
    for how, rows in hv.items():
        for row, pairs in rows.items():
            for pair in pairs:
                for h in (0, 1):
                    i, j = pair[h].get("I"), pair[h].get(I2)
                    if (i is None) != (j is None):
                        raise SystemExit(f"{how} {row}: I[2] exists without I or I without I[2]")
                    if i is not None and not (np.shape(j) == (1,) and float(j[0]) == float(np.asarray(i)[2])):
                        raise SystemExit(f"{how} {row}: I[2] differs from I's third number")


def check_inside_fixed(first: list, sel: list, where: str) -> None:
    """Under inside, I[0] and I[1] of each part do not change (|change| <= FIX_TOL seconds)."""
    for t, r in zip(first, sel):
        for h in (0, 1):
            d = np.abs(np.asarray(r[h]["I"], dtype=float)[:2] - np.asarray(t[h]["I"], dtype=float)[:2])
            if not np.all(d <= FIX_TOL):
                raise SystemExit(f"{where}: I[0] or I[1] changed under inside")


def contrib(first: list, sel: list, key: str, where: str) -> np.ndarray:
    own, base, change = o6.contrib_base_change(first, sel, key)
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(change)) and np.all(np.isin(own, (0.0, 0.5, 1.0)))):
        raise SystemExit(f"{where} {key}: a non-finite distance or a contribution not 0, 1/2 or 1")
    return own


def i2_cell(first: list, sel: list, rng_key: list[int], where: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """I[2]'s count 1 on given persons (v6b.one), with its contributions and resamples (the same key)."""
    res = v6b.one(first, sel, I2, rng_key)
    own = contrib(first, sel, I2, where)
    if not abs(float(own.mean()) - res["share_vs_own"]) <= TOL:
        raise SystemExit(f"{where}: I[2]'s contributions do not give its share")
    idx = np.random.default_rng(rng_key).integers(0, len(own), (v6.N_BOOT, len(own)))
    ci = res["share_vs_own_ci"]
    if ci != v6.ci(own, idx) or not p6.finite_ci(ci):
        raise SystemExit(f"{where}: I[2]'s interval")
    return {"persons": len(own), "share_vs_own": res["share_vs_own"], "share_vs_own_ci": ci,
            "class": o6.size_class(ci)}, own, idx


def diff(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict:
    d = a - b
    ci = v6.ci(d, idx)
    if not p6.finite_ci(ci):
        raise SystemExit("a non-finite interval")
    return {"difference": float(d.mean()), "ci": ci}


def old_part(old: dict) -> dict:
    return {"share_vs_own": old["share_vs_own"], "share_vs_own_ci": old["share_vs_own_ci"],
            "class": o6.size_class(old["share_vs_own_ci"])}


# ================================================================ the cells of one row


def rule_i2(hv: dict, ds: str, task: str, how: str, row: str, rn: dict) -> tuple[dict, dict | None]:
    """I's rule cell (v6p's rule_keep: recomputed and checked against v6n) and I[2] on its persons. Returns the
    written cell and the kept contributions (None if v6n wrote no share)."""
    nm = f"{ds}_{task}"
    where = f"{nm} {how} {row} rule"
    rule = p6.rule_keep(hv, ds, task, how, row, "I", rn)
    first = [hv[how]["text/before"][i] for i in rule["persons"]]
    sel = [hv[how][row][i] for i in rule["persons"]]
    if is_inside(row):
        check_inside_fixed(first, sel, where)
    if rule.get("no_count"):
        return {"no_count": True, "persons": len(rule["persons"])}, None
    old = rn["rule_persons"][nm][how][row]["I"]
    e, own2, idx = i2_cell(first, sel, qkey(ds, task, row, how, CELL, LAST["rule"]), where)
    parts = {p: old_part(old[p]) for p in ("stage", "R_loc", "R_mean")}
    if rule["cell_R_loc_class"] != parts["R_loc"]["class"]:
        raise SystemExit(f"{where}: R_loc's class differs from v6p's rule_keep")
    out = {"persons": len(rule["persons"]), I2: e, "I": parts["stage"], "R_loc": parts["R_loc"],
           "R_mean": parts["R_mean"], "state": cell_state(parts["R_loc"]["class"], parts["R_mean"]["class"]),
           "I[2]_minus_R_loc": diff(own2, rule["R_loc"], idx), "I[2]_minus_I": diff(own2, rule["stage"], idx)}
    if is_inside(row):
        out["persons_I[2]_below_I"] = int(np.sum(own2 < rule["stage"]))
    return out, {**rule, I2: own2}


def gate0_i2(hv: dict, ds: str, task: str, how: str, row: str, rn: dict) -> tuple[dict, list[int] | None]:
    """I's count 1 cell (gate 0; recomputed and checked against v6n) and I[2] on its persons."""
    nm = f"{ds}_{task}"
    where = f"{nm} {how} {row} gate 0"
    ok = p6.persons_with(hv, how, row, ("I",))
    first = [hv[how]["text/before"][i] for i in ok]
    sel = [hv[how][row][i] for i in ok]
    if is_inside(row):
        check_inside_fixed(first, sel, where)
    c = n6.one_cell(hv, row, how, "I", n6.key1(ds, task, "I", row, how, 0))
    old = rn["count_1"][nm][how][row]["I"]
    if "share_vs_own_ci" not in c:
        if "share_vs_own_ci" in old or old.get("persons") != c.get("persons") or c.get("persons") != len(ok):
            raise SystemExit(f"{where}: I's count 1 cell differs from v6n's record")
        return {"no_count": True, "persons": len(ok)}, None
    if not p6.same(c, old, ("persons", "share_vs_own", "share_vs_own_ci")) or c["persons"] != len(ok):
        raise SystemExit(f"{where}: I's count 1 cell differs from v6n's record")
    e, own2, _ = i2_cell(first, sel, qkey(ds, task, row, how, CELL, LAST["gate_0"]), where)
    out = {"persons": len(ok), I2: e, "I": old_part(c)}
    if is_inside(row):
        out["persons_I[2]_below_I"] = int(np.sum(own2 < contrib(first, sel, "I", where)))
    return out, ok


def same_json(a, b) -> bool:
    return json.dumps(n6.jsonish(a), sort_keys=True) == json.dumps(n6.jsonish(b), sort_keys=True)


def i2_groups(rule: dict, g: np.ndarray, ds: str, task: str, row: str, how: str, c: str, i_groups: dict) -> dict:
    """I[2]'s groups beside I's (i_groups: v6p's stage_groups for I, checked against v6p's record): per group I[2]'s
    share, interval and class, R_loc's and R_mean's and the state as I's (same persons and contributions), I[2] - R_loc
    and I[2] - I over I[2]'s resamples of the group; high - low of I[2] and rising."""
    off = COVARIATES[c]
    out: dict = {"persons": i_groups["persons"], "written": True, "groups": {}}
    means = {}
    own2 = rule[I2]
    for j, gn in enumerate(GROUPS):
        sel = g == j
        if int(sel.sum()) != i_groups["groups"][gn]["persons"] or int(sel.sum()) < MIN_SUMMARY:
            raise SystemExit("the group's persons differ from I's or are fewer than 5")
        own = own2[sel]
        idx, m = o6.boot_means(own, qkey(ds, task, row, how, off + j, LAST["rule"]))
        ci = v6.ci(own, idx)
        if ci != o6.pct(m) or not p6.finite_ci(ci):
            raise SystemExit("the bootstrap interval differs from v6.ci")
        means[gn] = m
        ig = i_groups["groups"][gn]
        out["groups"][gn] = {
            "persons": int(sel.sum()), "share_vs_own": float(own.mean()), "share_vs_own_ci": ci,
            "class": o6.size_class(ci), "I": {k: ig[k] for k in ("share_vs_own", "share_vs_own_ci", "class")},
            "R_loc": ig["R_loc"], "R_mean": ig["R_mean"], "state": ig["state"],
            "I[2]_minus_R_loc": diff(own, rule["R_loc"][sel], idx), "I[2]_minus_I": diff(own, rule["stage"][sel], idx)}
    w = sum(out["groups"][gn]["share_vs_own"] * out["groups"][gn]["persons"] for gn in GROUPS) / len(own2)
    if not abs(w - float(own2.mean())) <= TOL:
        raise SystemExit("the groups' shares of I[2] do not give the cell's share")
    out["high_minus_low"] = {"difference": out["groups"]["high"]["share_vs_own"] - out["groups"]["low"]["share_vs_own"],
                             "ci": o6.pct(means["high"] - means["low"])}
    if not p6.finite_ci(out["high_minus_low"]["ci"]):
        raise SystemExit("a non-finite interval")
    out["I_high_minus_low"] = i_groups["high_minus_low"]
    sh = [out["groups"][gn]["share_vs_own"] for gn in GROUPS]
    out["rising"] = bool(sh[0] <= sh[1] <= sh[2])
    return out


def same_ledger(a: list[np.ndarray], b: list[np.ndarray], where: str) -> None:
    """inside/before and inside_100 are the same row: I[2] must be written over the same sets in both, so that the
    two ledgers together add nothing to either."""
    if len(a) != len(b) or not all(np.array_equal(x, y) for x, y in zip(a, b)):
        raise SystemExit(f"{where}: inside/before and inside_100 write I[2] over different sets")


def own_ledger_ok(sets: list[np.ndarray]) -> None:
    """No set of 1-4 persons is linearly derivable from the sets over which I[2]'s share is written (v6p's
    small_sets)."""
    if p6.small_sets(sets):
        raise SystemExit("a set of 1-4 persons is derivable from I[2]'s own ledger")


# ================================================================ one set


def admit(ledger: list[np.ndarray], sets: list[np.ndarray]) -> bool:
    """Add sets to I[2]'s own ledger if no set of 1-4 persons becomes derivable (decision 2); else leave it."""
    if p6.small_sets(ledger + sets):
        return False
    ledger += sets
    return True


def family_out(hv: dict, rule: dict | None, rule_out: dict, gate0: list[int], lab: dict, ds: str, task: str, how: str,
               row: str, c: str, rp: dict, ro: dict, ledger: list[np.ndarray], n: int) -> dict:
    """One family of I[2]'s groups: written where v6p wrote I's and I[2]'s own ledger stays free of small sets."""
    nm = f"{ds}_{task}"
    prec = rp["groups"][c][nm][how]["I"][row]
    if rule is None:
        if not prec.get("no_count"):
            raise SystemExit(f"{nm} {how} {row} {c}: v6p counted I where v6n did not")
        return {"no_count": True, "persons": rule_out["persons"], "written": False}
    g = p6.group_of_stage(rule, gate0, lab, c)
    sizes = dict(zip(GROUPS, [int(np.sum(g == j)) for j in range(3)]))
    if not prec.get("written"):
        if prec.get("persons") != sizes or prec.get("plan") in (None, "written"):
            raise SystemExit(f"{nm} {how} {row} {c}: I's withheld family differs from v6p's record")
        return {"persons": sizes, "written": False, "plan": prec["plan"]}
    same0 = rule["persons"] == gate0
    if same0:
        p6.check_gate_v6o(rule, g, ds, task, "I", row, how, c,
                          {ref: ro["groups"][c][nm][how]["cells"][ref][row] for ref in p6.REFS})
    ig = p6.stage_groups(rule, g, ds, task, "I", row, how, c, "written", same0)
    ig["cell_R_loc_class"] = rule["cell_R_loc_class"]
    if not same_json(ig, prec):
        raise SystemExit(f"{nm} {how} {row} {c}: I's groups differ from v6p's record")
    if "withheld" in rule_out or not admit(ledger, p6.family(rule["persons"], g, n)):
        return {"persons": sizes, "written": False, "plan": "derivable (I[2])"}
    return {**i2_groups(rule, g, ds, task, row, how, c, ig), "plan": "written",
            "cell_R_loc_class": rule["cell_R_loc_class"]}




def set_cells(hv: dict, cov: list[dict], ds: str, task: str, rn: dict, ro: dict, rp: dict) -> dict:
    nm = f"{ds}_{task}"
    keep = o6.set_keep(hv, cov, ds, task, rn)
    lab = o6.labels_of(keep)
    p6.check_v6o_groups(lab, ro, nm)
    n = len(hv["oddeven"]["text/before"])
    cells: dict = {}
    groups: dict = {c: {} for c in COVARIATES}
    for how in v6b.BASELINE:
        gate0 = o6.cell_persons(hv, how, ROWS_O[0], "R")
        cells[how] = {}
        for c in COVARIATES:
            groups[c][how] = {}
        ledgers: dict = {}
        for row in ROWS_O:
            rule_out, rule = rule_i2(hv, ds, task, how, row, rn)
            g0_out, g0 = gate0_i2(hv, ds, task, how, row, rn)
            ledger: list[np.ndarray] = []
            if rule is not None and not admit(ledger, [p6.mask_of(rule["persons"], n)]):
                rule_out = {"persons": rule_out["persons"], "withheld": "derivable (I[2])"}
            for c in COVARIATES:
                groups[c][how][row] = family_out(hv, rule, rule_out, gate0, lab[how], ds, task, how, row, c, rp, ro,
                                                 ledger, n)
            if g0 is not None and not admit(ledger, [p6.mask_of(g0, n)]):
                g0_out = {"persons": g0_out["persons"], "withheld": "derivable (I[2])"}
            if ledger:
                own_ledger_ok(ledger)
            ledgers[row] = ledger
            cells[how][row] = {"rule": rule_out, "gate_0": g0_out}
        same_ledger(ledgers["inside/before"], ledgers["inside_100"], f"{nm} {how}")
    return {"cells": cells, "groups": groups}


# ================================================================ predictions, counts and main cells


NE = "not evaluable"


def rule_of(rec: dict, nm: str, how: str, row: str) -> dict:
    return rec["cells"][nm][how][row]["rule"]


def unread(e: dict) -> bool:
    """A cell with no share written: v6n wrote none (no_count) or I[2]'s own ledger withheld it."""
    return bool(e.get("no_count")) or "withheld" in e


def predictions(rec: dict) -> dict:
    out: dict = {}
    gate = [rule_of(rec, nm, how, row) for nm, how, row in GATE_CELLS]
    for (nm, how, row), e in zip(GATE_CELLS, gate):
        if not unread(e) and e["state"] != "gate":
            raise SystemExit(f"{nm} {how} {row}: named as a gate cell but its state is {e['state']}")
    if any(unread(e) for e in gate):
        out["P1"] = out["P2"] = NE
    else:
        states = "; ".join(f"{nm} {how} {row}: {e['state']}, I[2] {e[I2]['class']}"
                           for (nm, how, row), e in zip(GATE_CELLS, gate))
        n_above = sum(e[I2]["class"] == "above" for e in gate)
        out["P1"] = ("held" if n_above >= 2 else "did not hold") + f" ({n_above} of 3 above; {states})"
        n_below = sum(e["I[2]_minus_R_loc"]["ci"][1] < 0 for e in gate)
        out["P2"] = ("held" if n_below == 3 else "did not hold") + f" ({n_below} of 3 upper ends below 0)"
    ks = [rule_of(rec, "kupa_essay", how, row) for how in v6b.BASELINE for row in PRIMARY_ROWS]
    if any(unread(e) for e in ks):
        out["P3"] = NE
    else:
        cls = [e[I2]["class"] for e in ks]
        out["P3"] = ("held" if "above" not in cls else "did not hold") + " (" + ", ".join(cls) + ")"
    sh = [rule_of(rec, nm, how, row) for nm in rec["cells"] for how in v6b.BASELINE for row in SHUFFLES]
    if any(unread(e) for e in sh):
        out["P4"] = NE
    else:
        n_lower = sum(e[I2]["share_vs_own"] < e["I"]["share_vs_own"] for e in sh)
        out["P4"] = ("held" if n_lower >= 6 else "did not hold") + f" ({n_lower} of {len(sh)})"
    return out


def counts(rec: dict) -> dict:
    """Over the rule cells of all rows except inside/before: cells by state and, within a state, by I[2]'s and by I's
    class; the same over groups (written families) by covariate, with families withheld and high - low's side."""
    out: dict = {"cells": {s: {"I[2]": dict.fromkeys(CLASSES, 0), "I": dict.fromkeys(CLASSES, 0)}
                           for s in CELL_STATES}, "cells_no_count": 0, "cells_withheld": 0,
           "gate_0_withheld": 0, "gate_0_written": 0}
    for nm in rec["cells"]:
        for how in v6b.BASELINE:
            for row in COUNTED_ROWS:
                g0 = rec["cells"][nm][how][row]["gate_0"]
                if "withheld" in g0:
                    out["gate_0_withheld"] += 1
                elif not g0.get("no_count"):
                    out["gate_0_written"] += 1
                e = rule_of(rec, nm, how, row)
                if unread(e):
                    out["cells_no_count" if e.get("no_count") else "cells_withheld"] += 1
                    continue
                out["cells"][e["state"]][I2][e[I2]["class"]] += 1
                out["cells"][e["state"]]["I"][e["I"]["class"]] += 1
    for c in COVARIATES:
        o = {"states": {s: {"I[2]": dict.fromkeys(CLASSES, 0), "I": dict.fromkeys(CLASSES, 0)} for s in STATES},
             "families_withheld_v6p": 0, "families_withheld_I2": 0, "no_count": 0, "cells_written": 0,
             "high_minus_low": {"above 0": 0, "includes 0": 0, "below 0": 0}}
        for nm in rec["groups"][c]:
            for how in v6b.BASELINE:
                for row in COUNTED_ROWS:
                    cell = rec["groups"][c][nm][how][row]
                    if cell.get("no_count"):
                        o["no_count"] += 1
                        continue
                    if not cell["written"]:
                        o["families_withheld_I2" if cell.get("plan") == "derivable (I[2])"
                          else "families_withheld_v6p"] += 1
                        continue
                    o["cells_written"] += 1
                    o["high_minus_low"][o6.interval_side(cell["high_minus_low"]["ci"])] += 1
                    for gn in GROUPS:
                        e = cell["groups"][gn]
                        o["states"][e["state"]][I2][e["class"]] += 1
                        o["states"][e["state"]]["I"][e["I"]["class"]] += 1
        out[f"groups_{c}"] = o
    return out


def r3(x: float) -> float:
    return round(x, 3)


def short_cell(e: dict) -> dict:
    if unread(e):
        return {k: e[k] for k in ("no_count", "withheld", "persons") if k in e}
    out = {"persons": e["persons"], "state": e["state"], "I[2]": [r3(e[I2]["share_vs_own"]), e[I2]["class"]],
           "I": [r3(e["I"]["share_vs_own"]), e["I"]["class"]],
           "R_loc": [r3(e["R_loc"]["share_vs_own"]), e["R_loc"]["class"]],
           "I[2]_minus_R_loc": [r3(e["I[2]_minus_R_loc"]["difference"]), *map(r3, e["I[2]_minus_R_loc"]["ci"])],
           "I[2]_minus_I": [r3(e["I[2]_minus_I"]["difference"]), *map(r3, e["I[2]_minus_I"]["ci"])],
           "I[2]_ci": [r3(v) for v in e[I2]["share_vs_own_ci"]]}
    if "persons_I[2]_below_I" in e:
        out["persons_I[2]_below_I"] = e["persons_I[2]_below_I"]
    return out


def main_cells_out(rec: dict) -> dict:
    out: dict = {"gate_cells": {f"{nm} {how} {row}": short_cell(rule_of(rec, nm, how, row))
                                for nm, how, row in GATE_CELLS},
                 "primary": {}, "primary_groups_n_s": {}, "table": {}, "counts": rec["counts"]}
    for nm in rec["cells"]:
        for how in v6b.BASELINE:
            for row in PRIMARY_ROWS:
                out["primary"][f"{nm} {how} {row}"] = short_cell(rule_of(rec, nm, how, row))
                cell = rec["groups"]["n_s"][nm][how][row]
                key = f"{nm} {how} {row}"
                if not cell.get("written"):
                    out["primary_groups_n_s"][key] = {"written": False, "persons": cell["persons"],
                                                      "plan": cell.get("plan", "no count")}
                    continue
                out["primary_groups_n_s"][key] = {
                    "groups": {gn: {"I[2]": [r3(e["share_vs_own"]), e["class"]],
                                    "I": [r3(e["I"]["share_vs_own"]), e["I"]["class"]], "state": e["state"]}
                               for gn, e in cell["groups"].items()},
                    "high_minus_low": [r3(cell["high_minus_low"]["difference"]),
                                       *map(r3, cell["high_minus_low"]["ci"])],
                    "side": o6.interval_side(cell["high_minus_low"]["ci"])}
            t = {}
            for row in ROWS_O:
                e = rec["cells"][nm][how][row]
                r, g0 = e["rule"], e["gate_0"]
                t[row] = ("not written" if unread(r) else
                          f"{r['state']} | I[2] {r3(r[I2]['share_vs_own'])} {r[I2]['class']} | "
                          f"I {r3(r['I']['share_vs_own'])} {r['I']['class']}") + \
                         (" | gate 0 withheld" if "withheld" in g0 else "" if unread(g0) else
                          f" | gate 0 I[2] {r3(g0[I2]['share_vs_own'])} {g0[I2]['class']}")
            out["table"][f"{nm} {how}"] = t
    return out


# ================================================================ main


def count_all(n_series: dict, n_ids: list, kplace: dict, ln_series: dict, ln_ids: list, lplace: dict, m: dict,
              rn: dict, ro: dict, rp: dict) -> dict:
    rec: dict = {"cells": {}, "groups": {c: {} for c in COVARIATES}, "tallies": {}}
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
        hv = with_i2(n6.hv_of(b, o6.BUILD_ROWS))
        check_i2(hv)
        res = set_cells(hv, cov, ds, task, rn, ro, rp)
        rec["cells"][nm] = res["cells"]
        for c in COVARIATES:
            rec["groups"][c][nm] = res["groups"][c]
        rec["tallies"][nm] = {"count_persons": len(b["persons"])}
        del hv, b
    rec["counts"] = counts(rec)
    rec["checks"] = {
        "reuse": "v6n's persons and count persons; for both sets, baselines and the eleven rows, v6n's rule cell of I "
                 "(I, R_loc, R_mean: persons, share and interval; paired_minus_R_loc) and count 1 cell of I (persons, "
                 "share, interval); v6o's cuts and persons per group; v6p's plan for I's families and, where written, "
                 "v6p's groups of I (all fields), with R_loc and R_mean against v6o's groups",
        "construction": "I[2] is I's third number and exists exactly when I does; under the inside rows I[0] and I[1] "
                        "of each part do not change (1e-9 s) for the rule and gate 0 persons; contributions are 0, 1/2 "
                        "or 1 and the distances finite; the groups' shares of I[2] give the cell's (1e-12)",
        "sets": "every written set is I's rule persons, I's count 1 persons or one of v6p's written families of I; "
                "I[2]'s own ledger per row and baseline makes no set of 1-4 persons linearly derivable: what would have "
                "made one derivable was withheld (decision 2, CHOICES.own_ledger_withhold) and the whole ledger "
                "checked again; inside/before and inside_100 write over the same sets"}
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

    rec.update(n6.jsonish(count_all(n_series, n_ids, kplace, ln_series, ln_ids, lplace, m, rn, ro, rp)))
    rec["predictions"] = predictions(rec)
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
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6q-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    below = sum(e[part].get("persons_I[2]_below_I", 0) for nm in rec["cells"] for how in rec["cells"][nm]
                for e in rec["cells"][nm][how].values() for part in ("rule", "gate_0"))
    print("persons with I[2] below I under inside (expected 0):", below)
    for k, v in rec["main_cells"]["gate_cells"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "i2_vector": "I[2] is stored as a vector of one number beside I in each part's vectors, so v6b.one and v6o's "
                 "contributions apply unchanged",
    "cell_resamples": "I[2]'s cell interval and the paired differences use the resamples of I[2]'s key (v6b.one draws "
                      "them from the same key; checked equal)",
    "group_state": "a group's R_loc, R_mean and state are I's group's from v6p's stage_groups (same persons and "
                   "contributions; checked against v6p's record)",
    "gate_0": "the gate 0 cell is not read through the gate and has no groups",
    "inside_fixed": "the check of I[0] and I[1] runs on the rule persons and on the gate 0 persons of every inside row",
    "below_I": "persons_I[2]_below_I counts persons whose I[2] contribution is below I's under the inside rows "
               "(expected 0 by construction); written, not a stop",
    "own_ledger": "I[2]'s own ledger per row and baseline: the rule persons, the gate 0 persons and every written "
                  "family of both covariates",
    "counts": "over the ten rows other than inside/before; the table lists all eleven rows",
    "own_ledger_withhold": "decision 2 (before the run, no values seen): the conditions say the run stops if I[2]'s own "
                           "ledger makes a set of 1-4 persons derivable; the code instead adds, per row and baseline, "
                           "the rule cell, the n_s family, the n_s_min family and the gate 0 cell in that order, and "
                           "writes the persons only (with 'derivable (I[2])') for anything that would make such a set "
                           "derivable; the whole ledger is then checked again (stop if not clean). Withheld cells and "
                           "families are counted apart from those v6p withheld",
    "predictions_not_evaluable": "a prediction is 'not evaluable' if any of its cells is not written (no count or "
                                 "withheld)",
    "gate_cells_state": "the three gate cells named in the conditions must have the state 'gate' (the run stops "
                        "otherwise)",
    "below_I_total": "the persons with I[2]'s contribution below I's under inside are counted on the rule and gate 0 "
                     "cells and their total is printed after the run",
}


if __name__ == "__main__":
    main()
