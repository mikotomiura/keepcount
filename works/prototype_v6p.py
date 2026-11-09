"""Prototype v6p: does the rule stages' per-person reading also depend on the number of sentence starts per part?
(decision 29 of .steering/20261108-paper-a-jowr-reframe; decision 1 of .steering/20261109-v6p-rule-stages-by-sentence-count).
On v6n's persons, per dataset and baseline, the rule stages' persons (v6n's rule cells) are split by v6o's groups of
n_s (and n_s_min beside it), and per group the share of persons whose row exceeds their own parts is reported for each
rule stage beside R_loc and R_mean on the same persons; per group the gate rule (decision 25) says how the stage is read.
All descriptions.

Usage:  uv run python works/prototype_v6p.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6p.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6p/record.json (written to a temporary directory beside it and renamed when complete). No
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
import prototype_v6o as o6  # noqa: E402

n6, m6, k6 = o6.n6, o6.m6, o6.k6
v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6p.json"
REC_N = ROOT / "results" / "prototype_v6n" / "record.json"
REC_O = ROOT / "results" / "prototype_v6o" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6p"
CODE = ["works/prototype_v6p.py", *o6.CODE]
SEED = k6.SEED
TAG = 99
STAGES = (*n6.RULE_STAGES, *n6.BESIDE)  # U@2000, B@2000, Bsd@2000, I, U_med, B_complete@2000, Bsd_complete@2000
REFS = o6.REFS  # R_loc, R_mean
ROWS_O = o6.ROWS_O
COUNTED_ROWS = tuple(r for r in ROWS_O if r != "inside/before")  # inside/before is inside_100
COVARIATES = o6.COVARIATES
GROUPS = o6.GROUPS
MIN_SUMMARY = o6.MIN_SUMMARY
PRIMARY_STAGES = ("I", "Bsd@2000")
PRIMARY_ROWS = o6.PRIMARY_ROWS
TOL = o6.TOL
STATES = ("gate", "gate in group only", "R_mean only", "not read")
CLASSES = ("above", "undecided", "below")


def stage_ix(st: str) -> int:
    return n6.STAGE[k6.base_of(st)[0]]


def fixed(st: str, row: str, how: str) -> bool:
    """Fixed by construction: U, U_med and B under the inside rows; B under shuffled_in_halves with halves."""
    inside = row == "inside/before" or row.startswith("inside_")
    return (st in n6.INSIDE_FIXED and inside) or (st == "B@2000" and row == "shuffled_in_halves/before"
                                                  and how == "halves")


def skey(ds: str, task: str, st: str, row: str, how: str, group: int, last: int = 0) -> list[int]:
    return [SEED, TAG, k6.DATASET[ds], k6.TASK[task], stage_ix(st), n6.ROW_IX[row], v6b.BASELINE[how], group, last]


def persons_with(hv: dict, how: str, row: str, need: tuple[str, ...]) -> list[int]:
    first, sel = hv[how]["text/before"], hv[how][row]
    return [i for i in range(len(first)) if all(v6b.has(first[i], k) and v6b.has(sel[i], k) for k in need)]


def gate_state(cell_loc: str, loc: str, mean: str) -> str:
    """The gate rule applied to a group (decision 25; the table of the conditions' per_group.reading_by_group)."""
    if loc == "above":
        return "gate" if cell_loc == "above" else "gate in group only"
    return "R_mean only" if mean == "above" else "not read"


# ================================================================ one rule cell, checked against v6n


def same(a: dict, b: dict, fields: tuple[str, ...]) -> bool:
    """Every field present (not None) in both and equal."""
    return all(a.get(f) is not None and b.get(f) is not None and a[f] == b[f] for f in fields)


def finite_ci(ci) -> bool:
    return isinstance(ci, list) and len(ci) == 2 and all(np.isfinite(ci))


def rule_keep(hv: dict, ds: str, task: str, how: str, row: str, st: str, rn: dict) -> dict:
    """v6n's rule cell recomputed and checked against v6n's record, and the per-person contributions of the stage,
    R_loc and R_mean on its stage persons (the full rule cell's scale)."""
    nm = f"{ds}_{task}"
    where = f"{nm} {how} {row} {st}"
    try:
        old = rn["rule_persons"][nm][how][row][st]
    except KeyError:
        raise SystemExit(f"{where}: not in v6n's rule cells") from None
    ok = persons_with(hv, how, row, (st, "R"))
    for ref in REFS:
        if persons_with(hv, how, row, (ref, st)) != ok:
            raise SystemExit(f"{where}: the persons of {ref} differ from the stage's")
    c = n6.rule_cell(hv, ds, task, row, how, st)
    counted = ["share_vs_own_ci" in c[part] for part in ("stage", *REFS)]
    if not any(counted):
        for part in ("stage", *REFS):
            o = old.get(part, {})
            if "share_vs_own_ci" in o or o.get("persons") != c[part].get("persons") or c[part].get("persons") != len(ok):
                raise SystemExit(f"{where}: the rule cell's {part} differs from v6n's record")
        return {"persons": ok, "no_count": True}  # too few persons in the rule cell (v6n wrote no share)
    if not all(counted):
        raise SystemExit(f"{where}: the stage and the references are not all counted")
    for part in ("stage", *REFS):
        if not same(c[part], old.get(part, {}), ("persons", "share_vs_own", "share_vs_own_ci")):
            raise SystemExit(f"{where}: the rule cell's {part} differs from v6n's record")
        if not finite_ci(c[part]["share_vs_own_ci"]):
            raise SystemExit(f"{where}: a non-finite interval")
    if not same(c.get("paired_minus_R_loc", {}), old.get("paired_minus_R_loc", {}), ("difference", "ci")):
        raise SystemExit(f"{where}: the rule cell's paired difference differs from v6n's record")
    first = [hv[how]["text/before"][i] for i in ok]
    sel = [hv[how][row][i] for i in ok]
    out: dict = {"persons": ok, "cell_R_loc_class": o6.size_class(c["R_loc"]["share_vs_own_ci"])}
    for part, key in (("stage", st), *((r, r) for r in REFS)):
        own, base, change = o6.contrib_base_change(first, sel, key)
        if not (np.all(np.isfinite(base)) and np.all(np.isfinite(change)) and np.all(np.isin(own, (0.0, 0.5, 1.0)))):
            raise SystemExit(f"{where} {part}: a non-finite distance or a contribution not 0, 1/2 or 1")
        if len(ok) != c[part]["persons"] or not abs(float(own.mean()) - c[part]["share_vs_own"]) <= TOL:
            raise SystemExit(f"{where} {part}: the contributions do not give v6n's share")
        out[part] = own
    return out


# ================================================================ what may be written


def mask_of(ix, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[list(ix)] = True
    return m


def base_sets(hv: dict, how: str, row: str, gate0: list[int], o_fams: list[np.ndarray], rn_c1: dict, rules: dict,
              n: int) -> dict:
    """The sets over which v6n and v6o wrote a share of a rule stage, R_loc or R_mean for this row and baseline (boolean
    arrays over the count persons): count 1 cells (gate 0) per key, rule cells' stage persons per stage, and v6o's
    written group families and LWP's band families (o_fams, over v6o's cell persons)."""
    c1: dict = {}
    for key in (*STAGES, *REFS):
        ix = persons_with(hv, how, row, (key,))
        old = rn_c1[row][key]
        if len(ix) != old.get("persons"):
            raise SystemExit(f"{how} {row} {key}: the count 1 persons differ from v6n's")
        if "share_vs_own" in old:
            c1[key] = mask_of(ix, n)
    rule = {st: mask_of(rules[st]["persons"], n) for st in STAGES if not rules[st].get("no_count")}
    return {"count_1": c1, "rule": rule, "v6o": list(o_fams)}


def pooled(base: dict) -> list[np.ndarray]:
    return [*base["count_1"].values(), *base["rule"].values(), *base["v6o"]]


def stage_ledger(base: dict, st: str) -> list[np.ndarray]:
    """The conditions' ledger for one stage: its count 1 cell and rule cell, R_loc's and R_mean's count 1 cells, and
    v6o's families."""
    return [*(base["count_1"][k] for k in (st, *REFS) if k in base["count_1"]),
            *([base["rule"][st]] if st in base["rule"] else []), *base["v6o"]]


def family(ok: list[int], g: np.ndarray, n: int) -> list[np.ndarray]:
    return [mask_of([p for p, gj in zip(ok, g) if gj == j], n) for j in range(3)]


def plan_row(base: dict, how: str, row: str, gate0: list[int], lab: dict, written_o: dict, rules: dict,
             n: int) -> tuple[dict, dict]:
    """Per stage (not fixed, counted) and covariate, in the order of STAGES and COVARIATES, the family's reason:
    'written', 'v6o withheld' (the stage persons are v6o's and v6o withheld that family), 'group under 5', or
    'derivable' (a set of 1-4 persons would become derivable that was not before, either in the pooled ledger of the
    row or in the stage's own ledger: decision 2). Returns the plan and tallies of the small sets derivable before."""
    pool = pooled(base)
    pool_before = small_sets(pool)
    pos = {p: k for k, p in enumerate(gate0)}
    out: dict = {}
    tally = {"pooled_before": len(pool_before), "stage_before": {}, "written_where_strict_would_withhold": 0}
    for st in STAGES:
        out[st] = {}
        if fixed(st, row, how) or rules[st].get("no_count"):
            continue
        ok = rules[st]["persons"]
        led = stage_ledger(base, st)
        led_before = small_sets(led)
        tally["stage_before"][st] = len(led_before)
        for c in COVARIATES:
            fam = family(ok, lab[c][2][[pos[p] for p in ok]], n)
            if ok == gate0 and not written_o[c]:
                why = "v6o withheld"
            elif min(int(f.sum()) for f in fam) < MIN_SUMMARY:
                why = "group under 5"
            elif (small_sets(pool + fam) - pool_before) or (small_sets(led + fam) - led_before):
                why = "derivable"
            else:
                why = "written"
            out[st][c] = why
            if why == "written":
                if small_sets(led + fam):
                    tally["written_where_strict_would_withhold"] += 1
                pool += fam
                led += fam
    return out, tally


def check_plan(base: dict, plan: dict, gate0: list[int], lab: dict, rules: dict, n: int) -> None:
    """From the plan alone, rebuild the ledgers with the written families and check that no small set is new."""
    pos = {p: k for k, p in enumerate(gate0)}
    pool, pool_before = pooled(base), small_sets(pooled(base))
    for st in STAGES:
        fams = [family(rules[st]["persons"], lab[c][2][[pos[p] for p in rules[st]["persons"]]], n)
                for c in COVARIATES if plan[st].get(c) == "written"]
        led = stage_ledger(base, st)
        before = small_sets(led)
        for f in fams:
            led += f
            pool += f
        if small_sets(led) - before:
            raise SystemExit(f"{st}: a new small set is derivable from the stage's ledger")
    if small_sets(pool) - pool_before:
        raise SystemExit("a new small set is derivable from the whole ledger")


def small_sets(sets: list[np.ndarray]) -> set[frozenset]:
    """The sets of 1 to MIN_SUMMARY - 1 persons whose sum is a linear combination of the sums over the given sets
    (v6o's small_derivable, returning the persons of each such union of small atoms instead of a bool)."""
    from itertools import combinations
    M = np.array(sets, dtype=bool)
    if M.ndim != 2 or M.shape[0] == 0:
        raise SystemExit("no sets to check")
    sig, inv = np.unique(M.T, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    sizes = np.bincount(inv, minlength=sig.shape[0])
    A = sig.T.astype(float)
    r = np.linalg.matrix_rank(A)
    small = [j for j in range(len(sizes)) if sizes[j] < MIN_SUMMARY]
    out: set[frozenset] = set()
    for k in range(1, MIN_SUMMARY):
        for comb in combinations(small, k):
            if not 0 < sizes[list(comb)].sum() < MIN_SUMMARY:
                continue
            v = np.zeros(A.shape[1])
            v[list(comb)] = 1.0
            if np.linalg.matrix_rank(np.vstack([A, v])) == r:
                out.add(frozenset(int(p) for p in np.flatnonzero(np.isin(inv, comb))))
    if bool(out) != o6.small_derivable(sets):
        raise SystemExit("small_sets disagrees with v6o's small_derivable")
    return out


# ================================================================ the groups of one stage


def group_of_stage(rule: dict, gate0: list[int], lab: dict, c: str) -> np.ndarray:
    pos = {p: k for k, p in enumerate(gate0)}
    if any(p not in pos for p in rule["persons"]):
        raise SystemExit("a stage person is not among v6o's cell persons")
    return lab[c][2][[pos[p] for p in rule["persons"]]]


def ref_key(ds: str, task: str, st: str, ref: str, row: str, how: str, group: int, same0: bool) -> list[int]:
    return o6.gkey(ds, task, ref, row, how, group) if same0 else skey(ds, task, st, row, how, group, 1 + n6.STAGE[ref])


def check_gate_v6o(rule: dict, g: np.ndarray, ds: str, task: str, st: str, row: str, how: str, c: str,
                   o_cells: dict) -> None:
    """Where the stage persons are v6o's cell persons and v6o wrote the family: per group, R_loc's and R_mean's share,
    interval and class (v6o's key) equal v6o's record. Run for every such cell, written by v6p or not."""
    off = COVARIATES[c]
    for ref in REFS:
        oc = o_cells[ref]
        if not oc.get("written"):
            raise SystemExit(f"{ds} {task} {how} {row} {c} {ref}: v6o's family is not written")
        for j, gn in enumerate(GROUPS):
            r_own = rule[ref][g == j]
            ridx, _ = o6.boot_means(r_own, o6.gkey(ds, task, ref, row, how, off + j))
            rci = v6.ci(r_own, ridx)
            og = oc["groups"][gn]
            if not (og["persons"] == len(r_own) and og["share_vs_own"] == float(r_own.mean())
                    and og["share_vs_own_ci"] == rci and og["class"] == o6.size_class(rci)):
                raise SystemExit(f"{ds} {task} {how} {row} {st} {gn}: {ref} differs from v6o's group")


def stage_groups(rule: dict, g: np.ndarray, ds: str, task: str, st: str, row: str, how: str, c: str, why: str,
                 same0: bool) -> dict:
    """The groups of one stage cell: per group the stage's share, interval and class, R_loc's and R_mean's on the same
    persons, the paired differences and the reading by group; high - low and rising of the stage."""
    off = COVARIATES[c]
    sizes = [int(np.sum(g == j)) for j in range(3)]
    out: dict = {"persons": dict(zip(GROUPS, sizes)), "written": why == "written", "plan": why}
    if why != "written":
        return out
    if min(sizes) < MIN_SUMMARY:
        raise SystemExit("a group under 5 persons would be written")
    out["groups"] = {}
    means = {}
    for j, gn in enumerate(GROUPS):
        sel = g == j
        own = rule["stage"][sel]
        idx, m = o6.boot_means(own, skey(ds, task, st, row, how, off + j))
        ci = v6.ci(own, idx)
        if ci != o6.pct(m):
            raise SystemExit("the bootstrap interval differs from v6.ci")
        means[gn] = m
        e: dict = {"persons": int(sel.sum()), "share_vs_own": float(own.mean()), "share_vs_own_ci": ci,
                   "class": o6.size_class(ci)}
        for ref in REFS:
            r_own = rule[ref][sel]
            ridx, _ = o6.boot_means(r_own, ref_key(ds, task, st, ref, row, how, off + j, same0))
            rci = v6.ci(r_own, ridx)
            e[ref] = {"share_vs_own": float(r_own.mean()), "share_vs_own_ci": rci, "class": o6.size_class(rci)}
            d = own - r_own
            e[f"stage_minus_{ref}"] = {"difference": float(d.mean()), "ci": v6.ci(d, idx)}
        if not all(finite_ci(x) for x in (ci, e["R_loc"]["share_vs_own_ci"], e["R_mean"]["share_vs_own_ci"],
                                          e["stage_minus_R_loc"]["ci"], e["stage_minus_R_mean"]["ci"])):
            raise SystemExit("a non-finite interval")
        e["state"] = gate_state(rule["cell_R_loc_class"], e["R_loc"]["class"], e["R_mean"]["class"])
        out["groups"][gn] = e
    for part in ("stage", *REFS):
        sh = (out["groups"][gn]["share_vs_own"] if part == "stage" else out["groups"][gn][part]["share_vs_own"]
              for gn in GROUPS)
        w = sum(s * out["groups"][gn]["persons"] for s, gn in zip(sh, GROUPS)) / len(rule["persons"])
        if not abs(w - float(rule[part].mean())) <= TOL:
            raise SystemExit(f"the groups' shares do not give the rule cell's share ({part})")
    d = means["high"] - means["low"]
    out["high_minus_low"] = {"difference": out["groups"]["high"]["share_vs_own"] - out["groups"]["low"]["share_vs_own"],
                             "ci": o6.pct(d)}
    sh = [out["groups"][gn]["share_vs_own"] for gn in GROUPS]
    out["rising"] = bool(sh[0] <= sh[1] <= sh[2])
    return out


# ================================================================ one set


def check_v6o_groups(lab: dict, ro: dict, nm: str) -> None:
    for how in v6b.BASELINE:
        for c in COVARIATES:
            o = ro["groups"][c][nm][how]
            c1, c2, g = lab[how][c]
            sizes = dict(zip(GROUPS, [int(np.sum(g == j)) for j in range(3)]))
            if [c1, c2] != o["cuts"] or sizes != o["persons"]:
                raise SystemExit(f"{nm} {how} {c}: the cuts or groups differ from v6o's record")
            if bool(ro["written"][how][nm][c]) != bool(o["written"]):
                raise SystemExit(f"{nm} {how} {c}: v6o's written plan and its groups disagree")


def band_labels(keep: dict, ro: dict, kupa_cuts: dict) -> dict:
    """LWP's persons labelled by KUPA essay's cuts (v6o's bands), checked against v6o's record."""
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for c in COVARIATES:
            o = ro["bands"][c][how]
            if list(kupa_cuts[how][c]) != o["cuts_from_kupa_essay"]:
                raise SystemExit(f"{how} {c}: KUPA essay's cuts differ from v6o's bands")
            g = o6.group_of(keep[how]["x"][c], *kupa_cuts[how][c])
            if dict(zip(GROUPS, [int(np.sum(g == j)) for j in range(3)])) != o["persons"]["lwp"]:
                raise SystemExit(f"{how} {c}: LWP's band persons differ from v6o's")
            out[how][c] = g
    return out


def set_cells(hv: dict, cov: list[dict], ds: str, task: str, rn: dict, ro: dict, kupa_cuts: dict | None) -> dict:
    """One set. kupa_cuts None for KUPA essay; for LWP, KUPA essay's cuts per baseline and covariate (v6o's bands)."""
    nm = f"{ds}_{task}"
    keep = o6.set_keep(hv, cov, ds, task, rn)
    lab = o6.labels_of(keep)
    check_v6o_groups(lab, ro, nm)
    blab = band_labels(keep, ro, kupa_cuts) if kupa_cuts is not None else None
    n = len(hv["oddeven"]["text/before"])
    out: dict = {c: {} for c in COVARIATES}
    persons_differ: dict = {}
    plans: dict = {}
    tallies: dict = {}
    for how in v6b.BASELINE:
        gate0 = o6.cell_persons(hv, how, ROWS_O[0], "R")
        written_o = {c: bool(ro["written"][how][nm][c]) for c in COVARIATES}
        g0 = np.array(gate0)
        o_fams = [mask_of(g0[lab[how][c][2] == j], n) for c in COVARIATES if written_o[c] for j in range(3)]
        if blab is not None:
            o_fams += [mask_of(g0[blab[how][c] == j], n) for c in COVARIATES if ro["written"][how]["bands"][c]
                       for j in range(3)]
        for c in COVARIATES:
            out[c][how] = {}
        plans[how], tallies[how] = {}, {}
        for row in ROWS_O:
            rules = {st: rule_keep(hv, ds, task, how, row, st, rn) for st in STAGES}
            base = base_sets(hv, how, row, gate0, o_fams, rn["count_1"][nm][how], rules, n)
            plan, tally = plan_row(base, how, row, gate0, lab[how], written_o, rules, n)
            check_plan(base, plan, gate0, lab[how], rules, n)
            plans[how][row], tallies[how][row] = plan, tally
            for st in STAGES:
                rule = rules[st]
                same0 = rule["persons"] == gate0
                if not same0:
                    persons_differ.setdefault(f"{how} {st}", {})[row] = len(rule["persons"])
                for c in COVARIATES:
                    cell = out[c][how].setdefault(st, {})
                    if rule.get("no_count"):
                        cell[row] = {"no_count": True, "persons": len(rule["persons"]), "written": False}
                        continue
                    g = group_of_stage(rule, gate0, lab[how], c)
                    if same0 and written_o[c]:
                        check_gate_v6o(rule, g, ds, task, st, row, how, c,
                                       {ref: ro["groups"][c][nm][how]["cells"][ref][row] for ref in REFS})
                    if fixed(st, row, how):
                        cell[row] = {"fixed_by_construction": True, "persons": len(rule["persons"])}
                        continue
                    cell[row] = stage_groups(rule, g, ds, task, st, row, how, c, plan[st][c], same0)
                    cell[row]["cell_R_loc_class"] = rule["cell_R_loc_class"]
    cuts = {how: {c: lab[how][c][:2] for c in COVARIATES} for how in v6b.BASELINE}
    return {"groups": out, "stage_persons_not_v6o": persons_differ, "written": plans, "ledger_tallies": tallies,
            "cuts": cuts}


# ================================================================ predictions, counts and main cells


NE = "not evaluable"


def grp(rec: dict, nm: str, how: str, st: str, row: str, c: str = "n_s") -> dict:
    return rec["groups"][c][nm][how][st][row]


def predictions(rec: dict) -> dict:
    out: dict = {}
    for p, nm, hows, want in (("P1", "lwp_lwp", tuple(v6b.BASELINE), "gate"),
                              ("P4", "kupa_essay", ("oddeven",), "gate in group only")):
        res, cls = [], []
        for how in hows:
            cell = grp(rec, nm, how, "I", "inside_100")
            if not cell.get("written"):
                res.append(None)
                continue
            h = cell["groups"]["high"]
            cls.append(f"{how} {h['state']} {h['class']}")
            res.append(h["state"] == want and h["class"] != "above")
        out[p] = NE if None in res else ("held" if all(res) else "did not hold") + " (" + "; ".join(cls) + ")"
    n_read, n_below, unw = 0, 0, 0
    for nm in rec["groups"]["n_s"]:
        for how in v6b.BASELINE:
            for st in PRIMARY_STAGES:
                for row in PRIMARY_ROWS:
                    cell = grp(rec, nm, how, st, row)
                    if not cell.get("written"):
                        unw += 1
                        continue
                    for gn in GROUPS:
                        e = cell["groups"][gn]
                        if e["state"] in ("gate", "gate in group only"):
                            n_read += 1
                            n_below += e["stage_minus_R_loc"]["ci"][1] < 0
    p2 = ("held" if n_below == n_read else "did not hold") + f" ({n_below} of {n_read} groups read)"
    out["P2"] = (NE + " (no group read)" if n_read == 0 else
                 f"{NE} ({unw} of 24 cells withheld); over the written: {p2}" if unw else p2)
    inc, tot = 0, 0
    for nm in rec["groups"]["n_s"]:
        for how in v6b.BASELINE:
            for row in PRIMARY_ROWS:
                cell = grp(rec, nm, how, "I", row)
                if cell.get("written"):
                    tot += 1
                    inc += o6.interval_side(cell["high_minus_low"]["ci"]) == "includes 0"
    p3 = ("held" if inc >= 8 else "did not hold") + f" ({inc} of {tot})"
    out["P3"] = p3 if tot == 12 else f"{NE} ({12 - tot} of 12 cells withheld); over the written: {inc} of {tot}"
    return out


def counts(rec: dict, c: str) -> dict:
    """Over all cells except inside/before and the fixed ones: groups by state and, within a state, by the stage's
    class; families withheld; cells whose high - low lies above, includes or lies below 0."""
    out: dict = {}
    for st in STAGES:
        o = {"states": {s: dict.fromkeys(CLASSES, 0) for s in STATES}, "families_withheld": 0, "cells_written": 0,
             "high_minus_low": {"above 0": 0, "includes 0": 0, "below 0": 0}}
        for nm in rec["groups"][c]:
            for how in v6b.BASELINE:
                for row in COUNTED_ROWS:
                    cell = rec["groups"][c][nm][how][st][row]
                    if cell.get("fixed_by_construction"):
                        continue
                    if cell.get("no_count"):
                        o["no_count"] = o.get("no_count", 0) + 1
                        continue
                    if not cell["written"]:
                        o["families_withheld"] += 1
                        continue
                    o["cells_written"] += 1
                    o["high_minus_low"][o6.interval_side(cell["high_minus_low"]["ci"])] += 1
                    for gn in GROUPS:
                        e = cell["groups"][gn]
                        o["states"][e["state"]][e["class"]] += 1
        out[st] = o
    return out


def main_cells_out(rec: dict) -> dict:
    out: dict = {"stage_persons_not_v6o": {nm: v["stage_persons_not_v6o"] for nm, v in rec["sets"].items()},
                 "primary": {}, "counts_n_s": rec["counts"]["n_s"]}
    for nm in rec["groups"]["n_s"]:
        for how in v6b.BASELINE:
            for st in PRIMARY_STAGES:
                for row in PRIMARY_ROWS:
                    cell = grp(rec, nm, how, st, row)
                    key = f"{nm} {how} {st} {row}"
                    if not cell.get("written"):
                        out["primary"][key] = {"written": False, "persons": cell["persons"]}
                        continue
                    out["primary"][key] = {
                        "cell_R_loc_class": cell["cell_R_loc_class"],
                        "groups": {gn: {"share": round(e["share_vs_own"], 3), "class": e["class"], "state": e["state"],
                                        "R_loc": round(e["R_loc"]["share_vs_own"], 3),
                                        "stage_minus_R_loc": [round(v, 3) for v in e["stage_minus_R_loc"]["ci"]]}
                                   for gn, e in cell["groups"].items()},
                        "high_minus_low": round(cell["high_minus_low"]["difference"], 3),
                        "ci": [round(v, 3) for v in cell["high_minus_low"]["ci"]],
                        "side": o6.interval_side(cell["high_minus_low"]["ci"])}
    out["table_n_s"] = {}
    for nm in rec["groups"]["n_s"]:
        for how in v6b.BASELINE:
            for st in STAGES:
                t = {}
                for row in COUNTED_ROWS:
                    cell = grp(rec, nm, how, st, row)
                    if cell.get("fixed_by_construction"):
                        t[row] = "fixed"
                    elif cell.get("no_count"):
                        t[row] = "no count"
                    elif not cell["written"]:
                        t[row] = "withheld: " + cell["plan"]
                    else:
                        t[row] = " | ".join(f"{gn} {cell['groups'][gn]['state']} {cell['groups'][gn]['class']}"
                                            for gn in GROUPS)
                out["table_n_s"][f"{nm} {how} {st}"] = t
    return out


# ================================================================ main


def count_all(n_series: dict, n_ids: list, kplace: dict, ln_series: dict, ln_ids: list, lplace: dict, m: dict,
              rn: dict, ro: dict) -> dict:
    rec: dict = {"groups": {c: {} for c in COVARIATES}, "sets": {}, "tallies": {}}
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
        res = set_cells(hv, cov, ds, task, rn, ro, kupa_cuts)
        if ds == "kupa":
            kupa_cuts = res["cuts"]
        for c in COVARIATES:
            rec["groups"][c][nm] = res["groups"][c]
        rec["sets"][nm] = {"stage_persons_not_v6o": res["stage_persons_not_v6o"], "written": res["written"],
                           "ledger_tallies": res["ledger_tallies"]}
        rec["tallies"][nm] = {"count_persons": len(b["persons"])}
        del hv, b
    rec["counts"] = {c: counts(rec, c) for c in COVARIATES}
    rec["checks"] = {
        "reuse": "v6n's persons and count persons; v6o's cuts and persons per group; for both sets, baselines, the "
                 "eleven rows and the seven stages, v6n's rule cell (stage, R_loc, R_mean: persons, share and "
                 "interval; paired_minus_R_loc); where the stage persons are v6o's cell persons, the groups' R_loc "
                 "equals v6o's (persons, share, interval, class)",
        "groups": "the stage persons are among v6o's cell persons; the groups partition them; the groups' shares "
                  "weighted by persons give the rule cell's share for the stage, R_loc and R_mean (1e-12); no set of "
                  "1-4 persons becomes linearly derivable from the ledger of each row and baseline that was not "
                  "derivable from v6n's and v6o's records pooled (decision 2; checked during and after the plan)"}
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

    rec.update(count_all(n_series, n_ids, kplace, ln_series, ln_ids, lplace, m, rn, ro))
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
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6p-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    for k, v in rec["main_cells"]["primary"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "ledger_order": "per set, row and baseline, the stages in the order U@2000, B@2000, Bsd@2000, I, U_med, "
                    "B_complete@2000, Bsd_complete@2000 and within a stage n_s before n_s_min; a later family is "
                    "withheld if an earlier one already makes it derivable",
    "ledger_count_1": "v6n's count 1 cells enter the ledger when they have a share (two persons or more)",
    "gate_check": "where the stage persons are v6o's cell persons and v6o wrote the family, the group's R_loc and "
                  "R_mean share, interval and class must equal v6o's (the run stops otherwise)",
    "paired_resamples": "stage - R_loc and stage - R_mean use the stage's resamples of the group (the same indices for "
                        "both terms)",
    "fixed": "fixed cells are written with their persons only, under both covariates",
    "no_count": "a rule cell where v6n wrote no share (fewer than 2 persons) is written with its persons only and "
                "counted apart (no_count); it adds no set to the ledger",
    "counts": "over the ten rows other than inside/before and over the cells not fixed",
    "derivable_before": "decision 2 (before the run): pooled over the stage, R_loc and R_mean, v6n's records already "
                        "make some sets of 1-4 persons derivable where two of its cells' persons differ by 1-4 (e.g. a "
                        "count 1 cell and a rule cell); v6p withholds a family if it makes a new such set derivable, "
                        "both in the pooled ledger of the row (every stage, R_loc and R_mean) and in the stage's own "
                        "ledger (its count 1 and rule cells, R_loc's and R_mean's count 1 cells, v6o's families), and "
                        "writes per row and baseline the numbers of such sets before and of families written where "
                        "the strict rule (no small set at all in the stage's ledger) would have withheld them",
    "ledger_bands": "v6o's LWP band families (KUPA essay's cuts on LWP's cell persons, where v6o wrote them) are in "
                    "the ledger of LWP's rows; KUPA essay is run first and its cuts are checked against v6o's bands",
    "v6o_withheld": "where the stage persons are v6o's cell persons and v6o withheld a family, v6p withholds it too",
    "gate_check_all": "the check of R_loc and R_mean against v6o's groups runs on every counted cell whose persons "
                      "are v6o's and whose family v6o wrote, written by v6p or not, fixed or not",
    "plan_reasons": "the plan writes per family 'written', 'v6o withheld', 'group under 5' or 'derivable'",
    "final_check": "after the plan, the ledgers are rebuilt from the plan alone and checked again",
    "contributions": "the per-person contributions are checked to be 0, 1/2 or 1 and the distances finite",
}


if __name__ == "__main__":
    main()
