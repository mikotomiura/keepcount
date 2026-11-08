"""Prototype v6o: does the positive control's reach depend on the number of sentence starts per part? (decision 26 of
.steering/20261108-paper-a-jowr-reframe; decision 1 of .steering/20261109-v6o-sentence-count). On v6n's persons, per
dataset and baseline, persons are split into three groups by their number of sentence starts per part (n_s, and the
smaller part's n_s_min beside it), and the share of persons whose row exceeds their own parts is reported per group for
R_loc and R_mean on v6n's dose rows and v6k's main rows; KUPA essay and LWP are compared within KUPA essay's bands.
All descriptions.

Usage:  uv run python works/prototype_v6o.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6o.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6o/record.json (written to a temporary directory beside it and renamed when complete). No
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
import prototype_v6k as k6  # noqa: E402
import prototype_v6m as m6  # noqa: E402
import prototype_v6n as n6  # noqa: E402

v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6o.json"
REC_N = ROOT / "results" / "prototype_v6n" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6o"
CODE = ["works/prototype_v6o.py", *n6.CODE]
SEED = k6.SEED
TAG = 96
REFS = ("R_loc", "R_mean")
ROWS_O = (*k6.MAIN_ROWS, *n6.DOSE_ROWS)
BUILD_ROWS = ["text/before", *ROWS_O]
SETS = (("kupa", "essay"), ("lwp", "lwp"))
COVARIATES = {"n_s": 0, "n_s_min": 10}  # group index offset in the keys; bands add 3
GROUPS = ("low", "mid", "high")
MIN_SUMMARY = 5
PRIMARY_ROWS = ("inside_75", "inside_100", "cross_100")
TOL = 1e-12


# ================================================================ the covariate


def sentence_counts(s: v6.Series, kinds: list[str], row: str, how: str) -> tuple[int, int]:
    """The numbers of gaps labelled 'sentence' that the two parts hold under the row's counting (the gaps R uses)."""
    lab = v6b.labels(s, kinds)
    ranges = s.ranges(m6.COUNTING.get(row, "before"))
    held = [np.arange(lo, hi) for lo, hi in ranges]
    n = len(ranges)
    out = []
    for idx in v6b.parts(n, how):
        h = np.unique(np.concatenate([held[i] for i in idx])) if len(idx) else np.array([], dtype=int)
        out.append(int(np.sum(lab[h] == "sentence")))
    return out[0], out[1]


def covariates_of(s: v6.Series, kinds: list[str]) -> dict:
    """Per baseline: n_s (mean of the two parts' counts) and n_s_min (the smaller), checked equal under every row."""
    out = {}
    for how in v6b.BASELINE:
        a, b = sentence_counts(s, kinds, "text/before", how)
        for row in ROWS_O:
            if sentence_counts(s, kinds, row, how) != (a, b):
                raise SystemExit(f"{row} {how}: the parts' sentence starts differ from the text's")
        out[how] = {"n_s": (a + b) / 2, "n_s_min": float(min(a, b))}
    return out


# ================================================================ per-person contributions with base and change


def contrib_base_change(first: list, sel: list, st: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v6b.one's per-person contributions (1, 1/2, 0), own distance (base) and change, with the cell's scale."""
    t_all = np.array([v[h][st] for v in first for h in (0, 1)])
    sd = t_all.std(axis=0)
    keep = sd > 0

    def z(x):
        return np.asarray(x, dtype=float)[keep] / sd[keep]

    base = np.array([np.linalg.norm(z(v[0][st]) - z(v[1][st])) for v in first])
    change = np.array([(np.linalg.norm(z(r[0][st]) - z(t[0][st])) + np.linalg.norm(z(r[1][st]) - z(t[1][st]))) / 2
                       for t, r in zip(first, sel)])
    own = np.where(change > base, 1.0, np.where(change == base, 0.5, 0.0))
    if not np.array_equal(own, m6.own_contrib(first, sel, st)):
        raise SystemExit(f"{st}: the contributions differ from v6m's own_contrib")
    return own, base, change


def cell_persons(hv: dict, how: str, row: str, ref: str) -> list[int]:
    first, sel = hv[how]["text/before"], hv[how][row]
    return [i for i in range(len(first)) if v6b.has(first[i], ref) and v6b.has(sel[i], ref)]


# ================================================================ groups


def cuts_of(x: np.ndarray) -> tuple[float, float]:
    c1, c2 = np.quantile(x, [1 / 3, 2 / 3])
    return float(c1), float(c2)


def group_of(x: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """0 low (x <= c1), 1 mid (c1 < x <= c2), 2 high (x > c2)."""
    return np.where(x <= c1, 0, np.where(x <= c2, 1, 2))


def boot_means(v: np.ndarray, key: list[int]) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng(key).integers(0, len(v), (v6.N_BOOT, len(v)))
    return idx, v[idx].mean(axis=1)


def pct(m: np.ndarray) -> list[float]:
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def size_class(ci: list[float]) -> str:
    lo, hi = ci
    return "above" if lo > k6.SIZE else ("below" if hi < k6.SIZE else "undecided")


def gkey(ds: str, task: str, ref: str, row: str, how: str, group: int) -> list[int]:
    return [SEED, TAG, k6.DATASET[ds], k6.TASK[task], n6.STAGE[ref], n6.ROW_IX[row], v6b.BASELINE[how], group, 0]


def group_stats(own: np.ndarray, base: np.ndarray, change: np.ndarray, key: list[int]) -> tuple[dict, np.ndarray]:
    """One group's share, interval and class, with the medians of change and change/base (the own distance's median
    is written once per group in base_by_group)."""
    idx, m = boot_means(own, key)
    ci = v6.ci(own, idx)
    if ci != pct(m):
        raise SystemExit("the bootstrap interval differs from v6.ci")
    pos = base > 0
    return ({"persons": int(len(own)), "share_vs_own": float(own.mean()), "share_vs_own_ci": ci,
             "class": size_class(ci), "median_change": float(np.median(change)),
             "median_change_over_base": float(np.median(change[pos] / base[pos])) if pos.sum() >= MIN_SUMMARY
             else None}, m)


def by_group(own: np.ndarray, base: np.ndarray, change: np.ndarray, g: np.ndarray, ds: str, task: str, ref: str,
             row: str, how: str, offset: int, write: bool = True) -> dict:
    """The groups of one cell; nothing but the persons per group if any group has fewer than MIN_SUMMARY persons or
    the family is withheld (write False: a set of 1-4 persons could be derived from what is written)."""
    sizes = [int(np.sum(g == j)) for j in range(3)]
    if sum(sizes) != len(own):
        raise SystemExit("the groups do not partition the cell persons")
    if min(sizes) < MIN_SUMMARY or not write:
        return {"persons": dict(zip(GROUPS, sizes)), "written": False}
    out: dict = {"persons": dict(zip(GROUPS, sizes)), "written": True, "groups": {}}
    means = {}
    for j, nm in enumerate(GROUPS):
        sel = g == j
        out["groups"][nm], means[nm] = group_stats(own[sel], base[sel], change[sel],
                                                   gkey(ds, task, ref, row, how, offset + j))
    w = sum(out["groups"][nm]["share_vs_own"] * out["groups"][nm]["persons"] for nm in GROUPS) / len(own)
    if abs(w - float(own.mean())) > TOL:
        raise SystemExit("the groups' shares do not give the cell's share")
    d = means["high"] - means["low"]
    out["high_minus_low"] = {"difference": out["groups"]["high"]["share_vs_own"] - out["groups"]["low"]["share_vs_own"],
                             "ci": pct(d)}
    sh = [out["groups"][nm]["share_vs_own"] for nm in GROUPS]
    out["rising"] = bool(sh[0] <= sh[1] <= sh[2])
    return out


def interval_side(ci: list[float]) -> str:
    return "above 0" if ci[0] > 0 else ("below 0" if ci[1] < 0 else "includes 0")


# ================================================================ what may be written


def small_derivable(sets: list[np.ndarray]) -> bool:
    """True if a set of 1 to MIN_SUMMARY - 1 persons has a sum that is a linear combination of the sums over the
    given sets (each a boolean array over the same persons): from the shares and persons written, its mean could be
    derived. The persons are grouped into atoms by their membership; a candidate is a union of small atoms."""
    M = np.array(sets, dtype=bool)
    if M.ndim != 2 or M.shape[0] == 0:
        raise SystemExit("no sets to check")
    sig, inv = np.unique(M.T, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    sizes = np.bincount(inv, minlength=sig.shape[0])
    A = sig.T.astype(float)  # rows: sets; columns: atoms
    r = np.linalg.matrix_rank(A)
    small = [j for j in range(len(sizes)) if sizes[j] < MIN_SUMMARY]
    from itertools import combinations
    for k in range(1, MIN_SUMMARY):
        for comb in combinations(small, k):
            if not 0 < sizes[list(comb)].sum() < MIN_SUMMARY:
                continue
            v = np.zeros(A.shape[1])
            v[list(comb)] = 1.0
            if np.linalg.matrix_rank(np.vstack([A, v])) == r:
                return True
    return False


def plan_written(gk: dict, gl: dict, bl: dict) -> dict:
    """Per baseline (given the group labels of KUPA essay gk and LWP gl per covariate, and LWP's band labels bl per
    covariate), which families are written: each set's groups per covariate and the bands per covariate. Families are
    added in the order n_s groups, n_s bands, n_s_min groups, n_s_min bands, and one is withheld if it has a group or
    band under MIN_SUMMARY persons or if, with the families already written and all persons, a set of 1-4 persons
    could be derived. KUPA essay's bands are its groups, so a band family needs KUPA's group family."""
    def parts(lab):
        return [lab == j for j in range(3)]

    def ok_size(lab):
        return min(int(np.sum(lab == j)) for j in range(3)) >= MIN_SUMMARY

    written = {"kupa_essay": {}, "lwp_lwp": {}, "bands": {}}
    sets = {"kupa_essay": [np.ones(len(gk["n_s"]), dtype=bool)], "lwp_lwp": [np.ones(len(gl["n_s"]), dtype=bool)]}
    for c in COVARIATES:
        for nm, lab in (("kupa_essay", gk[c]), ("lwp_lwp", gl[c])):
            w = ok_size(lab) and not small_derivable(sets[nm] + parts(lab))
            written[nm][c] = bool(w)
            if w:
                sets[nm] += parts(lab)
        wb = (written["kupa_essay"][c] and ok_size(bl[c]) and ok_size(gk[c])
              and not small_derivable(sets["lwp_lwp"] + parts(bl[c])))
        written["bands"][c] = bool(wb)
        if wb:
            sets["lwp_lwp"] += parts(bl[c])
    if small_derivable(sets["kupa_essay"]) or small_derivable(sets["lwp_lwp"]):
        raise SystemExit("a set of fewer than 5 persons is derivable from the written sets")
    return written


# ================================================================ one set


def set_keep(hv: dict, cov: list[dict], ds: str, task: str, rn: dict) -> dict:
    """Per baseline, reference and row: the cell's contributions, own distances, changes and full-cell bootstrap
    means, after the check against v6n's record; and per baseline the covariates of the cell persons."""
    nm = f"{ds}_{task}"
    keep: dict = {}
    for how in v6b.BASELINE:
        persons = None
        keep[how] = {}
        base0: dict = {}
        for ref in (*REFS, "R"):
            if ref != "R":
                keep[how][ref] = {}
            for row in ROWS_O:
                ok = cell_persons(hv, how, row, ref)
                if persons is None:
                    persons = ok
                elif ok != persons:
                    raise SystemExit(f"{nm} {how} {row} {ref}: the persons with R differ between rows or references")
                if ref == "R":
                    continue
                try:
                    old = rn["count_1"][nm][how][row][ref]
                except KeyError:
                    raise SystemExit(f"{nm} {how} {row} {ref}: not in v6n's record") from None
                cell = n6.one_cell(hv, row, how, ref, n6.key1(ds, task, ref, row, how, 0))
                if (cell["persons"] != old.get("persons") or cell["share_vs_own"] != old.get("share_vs_own")
                        or cell["share_vs_own_ci"] != old.get("share_vs_own_ci")):
                    raise SystemExit(f"{nm} {how} {row} {ref}: the cell differs from v6n's record")
                first = [hv[how]["text/before"][i] for i in ok]
                sel = [hv[how][row][i] for i in ok]
                own, base, change = contrib_base_change(first, sel, ref)
                if not (np.all(np.isfinite(base)) and np.all(np.isfinite(change))):
                    raise SystemExit(f"{nm} {how} {row} {ref}: a non-finite distance")
                # the effective guard: the contributions give v6b.one's share (own_contrib is the same formula)
                if abs(float(own.mean()) - cell["share_vs_own"]) > TOL or len(ok) != cell["persons"]:
                    raise SystemExit(f"{nm} {how} {row} {ref}: the contributions do not give v6n's share")
                if ref in base0 and not np.array_equal(base0[ref], base):
                    raise SystemExit(f"{nm} {how} {ref}: the own distance depends on the row")
                base0[ref] = base
                _, full = boot_means(own, n6.key1(ds, task, ref, row, how, 0))
                keep[how][ref][row] = {"own": own, "base": base, "change": change, "full_means": full}
        keep[how]["x"] = {c: np.array([cov[i][how][c] for i in persons]) for c in COVARIATES}
        for c in COVARIATES:
            if not np.all(np.isfinite(keep[how]["x"][c])):
                raise SystemExit(f"{nm} {how}: a non-finite covariate")
    return keep


def labels_of(keep: dict) -> dict:
    """Per baseline and covariate: the cuts and group labels of the set's cell persons."""
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for c in COVARIATES:
            x = keep[how]["x"][c]
            c1, c2 = cuts_of(x)
            out[how][c] = (c1, c2, group_of(x, c1, c2))
    return out


def set_groups(keep: dict, lab: dict, written: dict, ds: str, task: str) -> dict:
    """Per covariate and baseline: the cuts, persons per group, the covariate's median per group and the cells."""
    nm = f"{ds}_{task}"
    out: dict = {c: {} for c in COVARIATES}
    for how in v6b.BASELINE:
        for c, off in COVARIATES.items():
            x = keep[how]["x"][c]
            c1, c2, g = lab[how][c]
            w = written[how][nm][c]
            sizes = [int(np.sum(g == j)) for j in range(3)]
            o: dict = {"cuts": [c1, c2], "tied_at_cut": [int(np.sum(x == c1)), int(np.sum(x == c2))],
                       "persons": dict(zip(GROUPS, sizes)), "written": w, "covariate_median_by_group": None,
                       "cells": {}, "base_by_group": {}}
            if w:
                o["covariate_median_by_group"] = {gn: float(np.median(x[g == j])) for j, gn in enumerate(GROUPS)}
            for ref in REFS:
                o["cells"][ref] = {row: by_group(keep[how][ref][row]["own"], keep[how][ref][row]["base"],
                                                 keep[how][ref][row]["change"], g, ds, task, ref, row, how, off,
                                                 write=w) for row in ROWS_O}
                base = keep[how][ref][ROWS_O[0]]["base"]
                o["base_by_group"][ref] = ({gn: float(np.median(base[g == j])) for j, gn in enumerate(GROUPS)}
                                           if w else None)
            out[c][how] = o
    return out


# ================================================================ matched bands


def band_labels(kk: dict, lk: dict) -> dict:
    """Per baseline and covariate: KUPA essay's cuts, and LWP's persons labelled by them."""
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for c in COVARIATES:
            c1, c2 = cuts_of(kk[how]["x"][c])
            out[how][c] = (c1, c2, group_of(kk[how]["x"][c], c1, c2), group_of(lk[how]["x"][c], c1, c2))
    return out


def bands(kk: dict, lk: dict, blab: dict, written: dict) -> dict:
    """Per covariate, baseline, reference and row: per band both shares and LWP - KUPA, written only where the plan
    writes the band family (every band >= MIN_SUMMARY persons in both sets and nothing small derivable); beside,
    the whole difference."""
    out: dict = {c: {} for c in COVARIATES}
    for c, off in COVARIATES.items():
        for how in v6b.BASELINE:
            xk, xl = kk[how]["x"][c], lk[how]["x"][c]
            c1, c2, gk, gl = blab[how][c]
            nk = [int(np.sum(gk == j)) for j in range(3)]
            nl = [int(np.sum(gl == j)) for j in range(3)]
            o: dict = {"cuts_from_kupa_essay": [c1, c2], "persons": {"kupa_essay": dict(zip(GROUPS, nk)),
                                                                     "lwp": dict(zip(GROUPS, nl))}}
            o["written"] = bool(written[how]["bands"][c])
            if o["written"] and min(nk + nl) < MIN_SUMMARY:
                raise SystemExit("a band under 5 persons would be written")
            if o["written"]:
                o["median_covariate"] = {gn: {"kupa_essay": float(np.median(xk[gk == j])),
                                              "lwp": float(np.median(xl[gl == j]))} for j, gn in enumerate(GROUPS)}
            o["cells"] = {}
            for ref in REFS:
                o["cells"][ref] = {}
                for row in ROWS_O:
                    a, b = kk[how][ref][row], lk[how][ref][row]
                    d = b["full_means"] - a["full_means"]
                    cell: dict = {"whole": {"difference": float(b["own"].mean() - a["own"].mean()), "ci": pct(d)}}
                    if o["written"]:
                        cell["bands"] = {}
                        for j, gn in enumerate(GROUPS):
                            ok_, ol_ = a["own"][gk == j], b["own"][gl == j]
                            _, mk = boot_means(ok_, gkey("kupa", "essay", ref, row, how, off + j))
                            _, ml = boot_means(ol_, gkey("lwp", "lwp", ref, row, how, off + 3 + j))
                            cell["bands"][gn] = {"kupa_essay": float(ok_.mean()), "lwp": float(ol_.mean()),
                                                 "lwp_minus_kupa": float(ol_.mean() - ok_.mean()), "ci": pct(ml - mk)}
                    o["cells"][ref][row] = cell
            out[c][how] = o
    return out


def plan_all(kk: dict, lk: dict) -> tuple[dict, dict, dict, dict]:
    labk, labl, blab = labels_of(kk), labels_of(lk), band_labels(kk, lk)
    written = {}
    for how in v6b.BASELINE:
        for c in COVARIATES:
            if not np.array_equal(blab[how][c][2], labk[how][c][2]):
                raise SystemExit("KUPA essay's bands are not its groups")
        written[how] = plan_written({c: labk[how][c][2] for c in COVARIATES}, {c: labl[how][c][2] for c in COVARIATES},
                                    {c: blab[how][c][3] for c in COVARIATES})
    return labk, labl, blab, written


# ================================================================ predictions and main cells


NE = "not evaluable"


def hl_side(rec: dict, nm: str, how: str, ref: str, row: str, cov: str = "n_s") -> str | None:
    cell = rec["groups"][cov][nm][how]["cells"][ref][row]
    return interval_side(cell["high_minus_low"]["ci"]) if cell["written"] else None


def predictions(rec: dict) -> dict:
    out = {}
    for p, nm, row in (("P1", "kupa_essay", "inside_100"), ("P2", "lwp_lwp", "inside_75")):
        s = [hl_side(rec, nm, how, "R_loc", row) for how in v6b.BASELINE]
        out[p] = NE if None in s else ("held" if all(x == "above 0" for x in s) else "did not hold")
    lower, total = 0, 0
    for ref in REFS:
        for nm in rec["groups"]["n_s"]:
            for how in v6b.BASELINE:
                b = rec["groups"]["n_s"][nm][how]["base_by_group"][ref]
                if b is None:
                    continue
                total += 1
                lower += b["high"] < b["low"]
    out["P3"] = NE if total < 8 else ("held" if lower >= 6 else "did not hold") + f" ({lower} of {total})"
    res = []
    for how in v6b.BASELINE:
        o = rec["bands"]["n_s"][how]
        if not o["written"]:
            res.append(None)
            continue
        cb = o["cells"]["R_loc"]["inside_100"]["bands"]
        res.append(all(interval_side(cb[gn]["ci"]) == "includes 0" for gn in GROUPS))
    out["P4"] = NE if all(r is None for r in res) else ("held" if any(r for r in res if r is not None)
                                                        else "did not hold")
    return out


def main_cells_out(rec: dict) -> dict:
    out: dict = {"groups_n_s": {}, "primary": {}}
    for nm, by in rec["groups"]["n_s"].items():
        for how, o in by.items():
            out["groups_n_s"][f"{nm} {how}"] = {"cuts": o["cuts"], "persons": o["persons"], "written": o["written"],
                                               "covariate_median_by_group": o["covariate_median_by_group"],
                                               "base_by_group": o["base_by_group"]}
            for ref in REFS:
                for row in PRIMARY_ROWS:
                    c = o["cells"][ref][row]
                    key = f"{nm} {how} {ref} {row}"
                    if not c["written"]:
                        out["primary"][key] = {"written": False, "persons": c["persons"]}
                        continue
                    out["primary"][key] = {
                        "shares": {g: round(c["groups"][g]["share_vs_own"], 3) for g in GROUPS},
                        "classes": {g: c["groups"][g]["class"] for g in GROUPS},
                        "high_minus_low": round(c["high_minus_low"]["difference"], 3),
                        "ci": [round(v, 3) for v in c["high_minus_low"]["ci"]],
                        "side": interval_side(c["high_minus_low"]["ci"])}
    out["bands_inside_100_R_loc"] = {}
    for how, o in rec["bands"]["n_s"].items():
        cell = o["cells"]["R_loc"]["inside_100"]
        out["bands_inside_100_R_loc"][how] = {"whole": cell["whole"], "written": o["written"],
                                              "bands": cell.get("bands"), "persons": o["persons"]}
    return out


# ================================================================ main


def count_all(n_series: dict, n_ids: list, kplace: dict, ln_series: dict, ln_ids: list, lplace: dict, m: dict,
              rn: dict) -> dict:
    rec: dict = {"groups": {c: {} for c in COVARIATES}, "tallies": {}}
    keeps = {}
    for ds, task, series, ids, place in (("kupa", "essay", n_series, n_ids, kplace),
                                         ("lwp", "lwp", ln_series, ln_ids, lplace)):
        nm = f"{ds}_{task}"
        b = n6.build(series, ids, ds, task, place, m, BUILD_ROWS)
        if len(b["persons"]) != rn["tallies"][nm]["count_persons"]:
            raise SystemExit(f"{nm}: the count persons differ from v6n's")
        cov = []
        for pid in b["persons"]:
            s, kinds = series[pid][task] if ds == "kupa" else series[pid]
            cov.append(covariates_of(s, kinds))
        hv = n6.hv_of(b, BUILD_ROWS)
        keeps[nm] = set_keep(hv, cov, ds, task, rn)
        rec["tallies"][nm] = {"count_persons": len(b["persons"])}
        del hv, b
    kk, lk = keeps["kupa_essay"], keeps["lwp_lwp"]
    labk, labl, blab, written = plan_all(kk, lk)
    rec["written"] = written
    for nm, keep, lab, (ds, task) in (("kupa_essay", kk, labk, SETS[0]), ("lwp_lwp", lk, labl, SETS[1])):
        cells = set_groups(keep, lab, written, ds, task)
        for c in COVARIATES:
            rec["groups"][c][nm] = cells[c]
    rec["bands"] = bands(kk, lk, blab, written)
    rec["checks"] = {
        "reuse": "v6n's persons and count persons; for KUPA essay and LWP, both baselines, the eleven rows and R_loc "
                 "and R_mean, persons, share_vs_own and its interval equal v6n's record (gate 0)",
        "groups": "the groups partition the cell persons; their shares weighted by persons give the cell's share "
                  "(1e-12); the persons with R, R_loc and R_mean are the same set in every row; the parts' "
                  "sentence starts are the same under every row; the own distance is the same in every row (per "
                  "person); no set of 1-4 persons is derivable from the sets written (per set and baseline)"}
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

    rec.update(count_all(n_series, n_ids, kplace, ln_series, ln_ids, lplace, m, rn))
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
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6o-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    for k, v in rec["main_cells"]["primary"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "written_plan": "per set and baseline, families are added in the order n_s groups, n_s bands, n_s_min groups, "
                    "n_s_min bands; a family is withheld (persons only) if it has a group under 5 persons or if, with "
                    "all persons and the families already written, the sum over a set of 1-4 persons is a linear "
                    "combination of the sums written (checked on atoms of membership). v6n's rule-person cells are "
                    "not included (written by v6n)",
    "covariate_summary": "per group only the median of the covariate (min and max would be one person's value); "
                         "written only when the family is written",
    "own_distance": "the own distance per person is checked equal over rows; its median per group is written once "
                    "(base_by_group), not in every cell",
    "main_cells": "the groups by n_s, the 24 primary cells and the bands of R_loc at inside_100; the rest is in the "
                  "body",
    "rows_built": "text/before, v6k's three main rows and v6n's eight dose rows (v6n's other rows are not built)",
    "covariate_check": "the parts' sentence starts are recomputed from each row's counting and the text's labels; rows "
                       "change only press times, so the check holds by construction and guards against regressions",
    "cell_persons": "per set and baseline, the persons with the reference in the text and the row; the run stops if "
                    "they differ between rows or between R_loc and R_mean",
    "ratio_median": "the median of change/base is written only over at least 5 persons with base > 0",
    "whole_difference": "LWP - KUPA over all persons uses each set's v6n key (gate 0) for its resamples",
    "P3": "counts the written (reference, set, baseline) combinations by n_s; not evaluable if fewer than 8",
    "P4": "held if, under at least one baseline whose bands are written, every band's interval includes 0",
}


if __name__ == "__main__":
    main()
