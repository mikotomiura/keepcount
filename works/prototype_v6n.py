"""Prototype v6n: the sensitivity of the reference on v6m's persons (decision 23 of .steering/20261108-paper-a-jowr-reframe;
decisions 1-4 of .steering/20261108-v6n-reference-sensitivity). Part 1: R_mean (R_loc's four medians replaced by means)
on v6m's dose rows with a 75% step added, beside R_loc. Part 2: the direction at the level of the group at every step,
beside count 1. Part 3: the split-half reliability on v6m's persons, beside v6l's on v6k's. All descriptions.

Usage:  uv run python works/prototype_v6n.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6n.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6n/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value, per-person change or id is written.
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
import keystroke as ks  # noqa: E402
import prototype_v6k as k6  # noqa: E402
import prototype_v6l as l6  # noqa: E402
import prototype_v6m as m6  # noqa: E402

v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6n.json"
REC_M = ROOT / "results" / "prototype_v6m" / "record.json"
REC_L = ROOT / "results" / "prototype_v6l" / "record.json"
REC_K = ROOT / "results" / "prototype_v6k" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6n"
CODE = ["works/prototype_v6n.py", *m6.CODE]
SEED = k6.SEED
REST_MS = k6.REST_MS

STAGE = {**m6.STAGE, "R_mean": 12}
STEPS = (("25", 0.25), ("50", 0.50), ("75", 0.75), ("100", 1.00))
FAMILIES = m6.FAMILIES
DOSE_ROWS = [f"{f}_{s}" for f in FAMILIES for s, _ in STEPS]
NEW_ROWS = ("inside_75", "cross_75")
ROWS = [*k6.ROWS, *DOSE_ROWS]
ROW_IX = {**m6.ROW_IX, "inside_75": 14, "cross_75": 15}
MAIN = k6.MAIN
MAIN_ROWS = k6.MAIN_ROWS
RULE_STAGES = k6.RULE_STAGES  # U@2000, B@2000, Bsd@2000, I
BESIDE = m6.BESIDE  # U_med, B_complete@2000, Bsd_complete@2000
REFS = ("R_loc", "R_mean")
GATES = (*l6.GATES, "R_mean")  # counted on each rule stage's persons: R, R_loc, R_lag, R_mean
KEYS = [*m6.KEYS, "R_mean"]
SIGNED_STAGES = (*m6.SIGNED_STAGES, "R_mean")
SIGNED_ROWS = tuple(DOSE_ROWS)
INSIDE_FIXED = m6.INSIDE_FIXED  # U@2000, U_med, B@2000
WHOLE_FIXED = {"B@2000": (0,), "I": (0,)}  # under every operated row (the whole record is one run; the multiset kept)
WHOLE_FIXED_INSIDE = {"U@2000": (0, 1, 2, 3)}  # under inside_p (each sentence keeps its multiset)
REL_STAGES = (*l6.REL_STAGES, "R_mean")
REL_DIMS = {**l6.REL_DIMS, "R_mean": 6}
REL_KEY = 98
P7_TOL = 0.15
CONSTR_TOL = k6.CONSTR_TOL
MIN_SUMMARY = l6.MIN_SUMMARY
LABS = ("sentence", "subsentence", "word", "within")
sk, base_of, size_class = k6.sk, k6.base_of, k6.size_class


# ================================================================ R_mean and the part vectors


def r_mean_of(gaps: np.ndarray, lab: np.ndarray, held: np.ndarray) -> np.ndarray | None:
    """R_loc with means for medians: the mean ln gap at sentence, subsentence, word and within-word starts, sentence
    minus within, and the share of long time at sentence and subsentence starts (v6b.loc's sixth number)."""
    g, L = gaps[held], lab[held]
    means = []
    for k in LABS:
        x = g[L == k]
        if x.size == 0:
            return None
        means.append(float(np.mean(np.log(x))))
    rest = g >= 2000.0 - 1e-6
    share = float(g[rest & np.isin(L, ["sentence", "subsentence"])].sum() / g[rest].sum()) if rest.any() else 0.0
    return np.array([*means, means[0] - means[3], share])


def add_r_mean(out: dict, rec: k6.Rec, idx: np.ndarray) -> None:
    """R_mean beside v6m's vectors; it exists exactly when R_loc does and its sixth number is R_loc's (checked)."""
    held = np.unique(np.concatenate([rec.held[i] for i in idx])) if len(idx) else np.array([], dtype=int)
    rm = r_mean_of(rec.gaps, rec.lab, held)
    loc = out["R_loc"]
    if (rm is None) != (loc is None):
        raise SystemExit("R_mean and R_loc do not exist for the same part")
    if rm is not None and rm[5] != loc[5]:
        raise SystemExit("R_mean's sixth number differs from R_loc's")
    out["R_mean"] = rm


def part_vectors_n(rec: k6.Rec, idx: np.ndarray) -> tuple[dict, dict]:
    out, tally = m6.part_vectors_m(rec, idx)
    add_r_mean(out, rec, idx)
    return out, tally


# ================================================================ the dose rows (v6m's plans, four steps)


def dose_rows(s: v6.Series, key: list[int]) -> tuple[dict, dict]:
    """The press times of the eight dose rows and the tallies of what moved (v6m's, with the positions changed)."""
    g = np.diff(ks.check_times(s.times))
    total = float(g.sum())
    n = len(s.ranges("before"))
    sent = m6.sentence_of_gap(s.starts, s.times.size)
    part_of_sent = {how: np.zeros(n, dtype=int) for how in v6b.BASELINE}
    for how in v6b.BASELINE:
        part_of_sent[how][v6b.parts(n, how)[1]] = 1
    out, tally = {}, {}
    for fam, plan in (("inside", m6.inside_plan(s, key)), ("cross", m6.cross_plan(s, key))):
        used = [x for a, b in plan for x in (a, b) if b is not None]
        if len(used) != len(set(used)):
            raise SystemExit(f"{fam}: a gap position is in two swaps of the plan")
        m = len(plan)
        sizes = [m6.n_at(p, m) for _, p in STEPS]
        mades = [[(a, b) for a, b in plan[:k] if b is not None] for k in sizes]
        for i, ((st, _), k) in enumerate(zip(STEPS, sizes)):
            row = f"{fam}_{st}"
            sw = plan[:k]
            made = mades[i]
            gr = m6.apply_swaps(g, sw)
            out[row] = m6.times_of(gr)
            if fam == "cross":
                moved = float(sum(g[a] for a, _ in made))
            else:
                moved = float(sum(abs(g[a] - g[b]) for a, b in made))
            tally[row] = {"m": m, "swaps": len(made), "share_swapped": (len(made) / m) if m else None,
                          "time_moved_share": moved / total, "left_without_partner": len(sw) - len(made),
                          "m_zero": m == 0,
                          "same_as_next_step": i + 1 < len(sizes) and made == mades[i + 1],
                          "made": made,
                          "positions_changed": int(np.sum(np.abs(gr - g) > ks.TOL_MS)),
                          "crossing_parts": {how: int(sum(part_of_sent[how][sent[a]] != part_of_sent[how][sent[b]]
                                                          for a, b in made)) for how in v6b.BASELINE}}
    return out, tally


def check_rows(text_gaps: np.ndarray, rows_times: dict, inside_times: np.ndarray | None,
               tally: dict | None = None) -> None:
    """Each row a permutation of the text's gaps; the four steps nested (swaps a prefix of the next step's, changed
    positions nested); inside_100 equal to v6k's inside/before."""
    m6.check_permutation(text_gaps, rows_times)
    changed = {row: set(np.flatnonzero(np.abs(np.diff(t) - text_gaps) > ks.TOL_MS).tolist())
               for row, t in rows_times.items()}
    for fam in FAMILIES:
        ch = [changed[f"{fam}_{s}"] for s, _ in STEPS]
        if not all(ch[i] <= ch[i + 1] for i in range(len(ch) - 1)):
            raise SystemExit(f"{fam}: the doses are not nested")
        if tally is not None:
            ms = [tally[f"{fam}_{s}"]["made"] for s, _ in STEPS]
            if not all(ms[i + 1][:len(ms[i])] == ms[i] for i in range(len(ms) - 1)):
                raise SystemExit(f"{fam}: the swaps of the doses are not nested")
    if inside_times is not None and not np.array_equal(rows_times["inside_100"], inside_times):
        raise SystemExit("inside_100 differs from v6k's inside/before")


# ================================================================ persons and rows


def person_rows(s: v6.Series, kinds: list[str], key: list[int], m: dict, rows: list[str]
                ) -> tuple[dict, dict, dict, dict, k6.Rec]:
    """v6m's person_rows with R_mean and the 75% steps; also the text's Rec (for the reliability)."""
    lab = v6b.labels(s, kinds)
    n = len(s.ranges("before"))
    dose_t, dose_tally = dose_rows(s, key) if any(r in DOSE_ROWS for r in rows) else ({}, {})
    out, whole, tallies, times, text_rec = {}, {}, {}, {}, None
    for row in rows:
        t = dose_t[row] if row in DOSE_ROWS else k6.row_times(s, row, key)
        times[row] = t
        rec = k6.Rec(t, s.ranges(m6.COUNTING.get(row, "before")), lab, m)
        if rec.n != n:
            raise SystemExit(f"{row}: {rec.n} sentences, not {n}")
        if row == "text/before":
            text_rec = rec
        out[row], tallies[row] = {}, {}
        for how in v6b.BASELINE:
            A, B = v6b.parts(n, how)
            va, ta = part_vectors_n(rec, A)
            vb, tb = part_vectors_n(rec, B)
            out[row][how] = (va, vb)
            tallies[row][how] = (ta, tb)
        if not row.startswith("text/") or row == "text/before":
            whole[row] = part_vectors_n(rec, np.arange(n))[0]
    m6.check_permutation(np.diff(s.times), {r: t for r, t in times.items() if not r.startswith("text/")})
    if dose_t:
        check_rows(np.diff(s.times), dose_t, times.get("inside/before", k6.row_times(s, "inside/before", key)),
                   dose_tally)
    return out, whole, tallies, dose_tally, text_rec


def check_construction(vec: dict, tallies: dict, whole: dict) -> None:
    """v6m's identities with inside_75 (through the dose rows' names), and the whole-record identities."""
    keep = {(row, how) for row, how in k6.KEEPS_PART_MULTISET}
    keep |= {(f"inside_{s}", how) for s, _ in STEPS for how in v6b.BASELINE}
    for row, how in keep:
        if row not in vec:
            continue
        inside = row.startswith("inside")
        for h in (0, 1):
            a, b = vec["text/before"][how][h], vec[row][how][h]
            pairs = {"B@2000": (a["B@2000"], b["B@2000"]),
                     "I0": (tallies["text/before"][how][h]["I0"], tallies[row][how][h]["I0"])}
            if inside:
                pairs.update({k: (a[k], b[k]) for k in ("U@2000", "U_med", "S1")})
            same_or_stop(pairs, f"{row} {how}")
    text = whole.get("text/before")
    for row, w in whole.items():
        if row.startswith("text/"):
            continue
        fixed = dict(WHOLE_FIXED)
        if row.startswith("inside"):
            fixed.update(WHOLE_FIXED_INSIDE)
        pairs = {f"whole {st}[{j}]": (None if text[st] is None else text[st][j], None if w[st] is None else w[st][j])
                 for st, js in fixed.items() for j in js}
        same_or_stop(pairs, row)


def same_or_stop(pairs: dict, where: str) -> None:
    for nm, (x, y) in pairs.items():
        if x is None and y is None:
            continue
        if (x is None) != (y is None):
            raise SystemExit(f"{where} {nm}: a value exists on one side only")
        if not np.all(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)) <= CONSTR_TOL):
            raise SystemExit(f"{where} {nm}: changed where it cannot by construction")


def build(series: dict, ids: list, ds: str, task: str, place: dict, m: dict, rows: list[str]) -> dict:
    """Per count person (by id): the part vectors per row and baseline, the whole-record vectors, the tallies, and the
    text's Rec with its place (for the reliability)."""
    out: dict = {"persons": {}, "left_out": 0, "dose_tallies": {row: [] for row in DOSE_ROWS}, "sentences": [],
                 "texts": []}
    for pid in ids:
        s, kinds = series[pid][task] if ds == "kupa" else series[pid]
        if not k6.count1_ok(s):
            out["left_out"] += 1
            continue
        vec, whole, tallies, dose_tally, text_rec = person_rows(s, kinds, [k6.DATASET[ds], k6.TASK[task], place[pid]],
                                                                m, rows)
        check_construction(vec, tallies, whole)
        out["persons"][pid] = {"vec": vec, "whole": whole}
        out["sentences"].append(len(s.ranges("before")))
        out["texts"].append((text_rec, place[pid], vec["text/before"]))
        for row in dose_tally:
            out["dose_tallies"][row].append(dose_tally[row])
    return out


def hv_of(b: dict, rows: list[str]) -> dict:
    return {how: {row: [p["vec"][row][how] for p in b["persons"].values()] for row in rows} for how in v6b.BASELINE}


def dose_tally_summary(tallies: list[dict]) -> dict:
    out = m6.dose_tally_summary(tallies)
    if out:
        out["positions_changed"] = l6.five_or_none([t["positions_changed"] for t in tallies])
    return out


# ================================================================ count 1 and the rule persons


def key1(ds: str, task: str, key_name: str, row: str, how: str, gate: int) -> list[int]:
    st, _ = base_of(key_name)
    return [SEED, 95, k6.DATASET[ds], k6.TASK[task], STAGE[st], ROW_IX[row], v6b.BASELINE[how], 0, gate]


def gate_of(key_name: str) -> int:
    return 1 + STAGE[base_of(key_name)[0]]


def one_cell(hv: dict, row: str, how: str, key_name: str, rng_key: list[int], extra: str | None = None) -> dict:
    """v6m's one_cell; R_mean, like R, R_loc and R_lag, is not defined under text/none."""
    if base_of(key_name)[0] == "R_mean" and row == "text/none":
        sel = hv[how][row]
        if any(v6b.has(sel[i], key_name) for i in range(len(sel))):
            raise SystemExit(f"text/none has {key_name} for some persons")
        return {"defined": False, "persons": 0}
    return m6.one_cell(hv, row, how, key_name, rng_key, extra)


def count1(hv: dict, ds: str, task: str) -> dict:
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for row in ROWS[1:]:
            out[how][row] = {k: one_cell(hv, row, how, k, key1(ds, task, k, row, how, 0)) for k in KEYS}
    return out


def paired(hv: dict, row: str, how: str, a_key: str, b_key: str, need: tuple[str, ...], cell_a: dict, cell_b: dict,
           rng_key: list[int], where: str) -> dict | None:
    """The mean difference of the per-person contributions of a_key and b_key on the persons with every stage of
    need (text and row), over the resamples of rng_key; None if either cell has no count."""
    if "share_vs_own_ci" not in cell_a or "share_vs_own_ci" not in cell_b:
        return None
    first, sel = hv[how]["text/before"], hv[how][row]
    ok = [i for i in range(len(first)) if all(v6b.has(first[i], x) and v6b.has(sel[i], x) for x in need)]
    f, s = [first[i] for i in ok], [sel[i] for i in ok]
    a, b = m6.own_contrib(f, s, a_key), m6.own_contrib(f, s, b_key)
    if abs(float(a.mean()) - cell_a["share_vs_own"]) > 1e-12 or abs(float(b.mean()) - cell_b["share_vs_own"]) > 1e-12:
        raise SystemExit(f"{where}: the contributions do not give the shares")
    if len(ok) != cell_a["persons"] or len(ok) != cell_b["persons"]:
        raise SystemExit(f"{where}: the paired persons differ from the cells'")
    idx = np.random.default_rng(rng_key).integers(0, len(ok), (v6.N_BOOT, len(ok)))
    d = a - b
    return {"persons": len(ok), "difference": float(d.mean()), "ci": v6.ci(d, idx)}


def rule_cell(hv: dict, ds: str, task: str, row: str, how: str, k: str) -> dict:
    """v6m's rule cell (the stage on its rule persons with R, R_loc and R_lag, and the paired difference with R_loc),
    with R_mean on the same persons and the paired differences stage - R_mean and R_mean - R_loc."""
    gate = gate_of(k)
    kk = key1(ds, task, k, row, how, gate)
    c = {"stage": one_cell(hv, row, how, k, kk, extra="R")}
    for g in GATES:
        c[g] = one_cell(hv, row, how, g, key1(ds, task, g, row, how, gate), extra=k)
    where = f"{ds} {task} {row} {how} {k}"
    p = paired(hv, row, how, k, "R_loc", (k, "R"), c["stage"], c["R_loc"], kk, where)
    if p is not None:
        c["paired_minus_R_loc"] = {"difference": p["difference"], "ci": p["ci"]}
    p = paired(hv, row, how, k, "R_mean", (k, "R"), c["stage"], c["R_mean"], kk, where)
    if p is not None:
        c["paired_minus_R_mean"] = p
    p = paired(hv, row, how, "R_mean", "R_loc", (k, "R"), c["R_mean"], c["R_loc"], kk, where)
    if p is not None:
        c["paired_R_mean_minus_R_loc"] = p
    return c


def rule_persons(hv: dict, ds: str, task: str, rows: tuple[str, ...]) -> dict:
    return {how: {row: {k: rule_cell(hv, ds, task, row, how, k) for k in (*RULE_STAGES, *BESIDE)} for row in rows}
            for how in v6b.BASELINE}


def gate0_refs(hv: dict, c1: dict, ds: str, task: str) -> dict:
    """R_mean - R_loc on the persons with R (gate 0), over R_mean's resamples, in every operated row."""
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for row in ROWS[1:]:
            if row.startswith("text/"):
                continue
            a, b = c1[how][row]["R_mean"], c1[how][row]["R_loc"]
            if ("share_vs_own_ci" in a) != ("share_vs_own_ci" in b):
                raise SystemExit(f"{ds} {task} {how} {row}: R_mean and R_loc are not both counted")
            if "share_vs_own_ci" in a and "share_vs_own_ci" in b and a["persons"] != b["persons"]:
                raise SystemExit(f"{ds} {task} {how} {row}: R_mean and R_loc have different persons")
            out[how][row] = paired(hv, row, how, "R_mean", "R_loc", ("R_mean",), a, b,
                                   key1(ds, task, "R_mean", row, how, 0), f"{ds} {task} {row} {how} gate 0")
    return out


# ================================================================ part 1: the reference on the doses


def ladder(cells: list[dict], fixed: bool = False) -> dict:
    """Classes per step, shares, the first step above 0.60 and whether the share rises with the step."""
    cls = [size_class(c) for c in cells]
    shares = [c.get("share_vs_own") for c in cells]
    first = "none"
    for (s, _), x in zip(STEPS, cls):
        if x == "no count":
            first = "not evaluable"
            break
        if x == "above":
            first = s
            break
    rising = None if None in shares else bool(all(shares[i] <= shares[i + 1] for i in range(len(shares) - 1)))
    return {"classes": dict(zip([s for s, _ in STEPS], cls)),
            "shares": dict(zip([s for s, _ in STEPS], shares)),
            "first_step_above": "not read" if fixed else first, "rising": rising}


def medians(c: dict) -> dict:
    return {f: c.get(f) for f in ("persons", "median_change", "median_base", "median_change_over_base", "base_zero")}


def part1(c1: dict, g0: dict, rp: dict) -> dict:
    """Per set, baseline and family: R_loc and R_mean on the persons with R (gate 0) with their paired difference,
    and on each rule stage's persons; the rule stages' classes at every step with the paired differences."""
    out: dict = {"gate_0": {}, "rule_persons": {}}
    for ds, task in MAIN:
        nm = f"{ds}_{task}"
        for how in v6b.BASELINE:
            for fam in FAMILIES:
                rows = [f"{fam}_{s}" for s, _ in STEPS]
                where = f"{nm} | {how} | {fam}"
                e = {r: ladder([c1[nm][how][row][r] for row in rows]) for r in REFS}
                e["medians"] = {r: {s: medians(c1[nm][how][row][r]) for (s, _), row in zip(STEPS, rows)}
                                for r in REFS}
                e["paired_R_mean_minus_R_loc"] = {s: g0[nm][how][row] for (s, _), row in zip(STEPS, rows)}
                out["gate_0"][where] = e
                for k in (*RULE_STAGES, *BESIDE):
                    cells = [rp[nm][how][row][k] for row in rows]
                    fixed = fam == "inside" and k in INSIDE_FIXED
                    ent: dict = {"fixed_by_construction": fixed,
                                 "stage": ladder([c["stage"] for c in cells], fixed)}
                    for r in REFS:
                        ent[r] = ladder([c[r] for c in cells])
                    for pk in ("paired_minus_R_loc", "paired_minus_R_mean", "paired_R_mean_minus_R_loc"):
                        ent[pk] = {s: c.get(pk) for (s, _), c in zip(STEPS, cells)}
                    out["rule_persons"].setdefault(k, {})[where] = ent
    out["main_rows_rule_persons"] = {
        f"{ds}_{task} | {how} | {row} | {k}": {r: size_class(rp[f"{ds}_{task}"][how][row][k][r]) for r in
                                               ("stage", *REFS)}
        for ds, task in MAIN for how in v6b.BASELINE for row in MAIN_ROWS for k in RULE_STAGES}
    return out


# ================================================================ part 2: the group direction at every step


def signed_set(pairs: list[tuple[dict, dict]], ds: int, task: int, row_ix: int) -> dict:
    """v6m's signed_set over SIGNED_STAGES with R_mean (its stage index 12)."""
    out: dict = {}
    for st in SIGNED_STAGES:
        ok = [(a[st], b[st]) for a, b in pairs if a[st] is not None and b[st] is not None]
        out[st] = {}
        if not ok:
            continue
        d = np.array([np.asarray(b, dtype=float) - np.asarray(a, dtype=float) for a, b in ok])
        for f in range(d.shape[1]):
            out[st][str(f)] = m6.signed(d[:, f], [SEED, 97, ds, task, STAGE[base_of(st)[0]], row_ix, f])
    return out


def fixed_feature(row: str, st: str, f: str) -> bool:
    j = int(f)
    if j in WHOLE_FIXED.get(st, ()):
        return True
    return row.startswith("inside") and j in WHOLE_FIXED_INSIDE.get(st, ())


def part2(p4: dict, c1: dict, rp: dict) -> dict:
    """Per set, row and stage: the shares of signs per feature (fixed features marked, not read) beside the stage's
    count 1 shares (gate 0 and, for rule stages, rule persons) under both baselines."""
    out: dict = {}
    for ds, task in MAIN:
        nm = f"{ds}_{task}"
        for row in SIGNED_ROWS:
            for st in SIGNED_STAGES:
                feats = {}
                for f, v in p4[nm][row][st].items():
                    feats[f] = None if v is None else {
                        "persons": v["persons"], "share_positive": v["share_positive"],
                        "share_negative": v["share_negative"], "share_zero": v["share_zero"],
                        "fixed_by_construction": fixed_feature(row, st, f)}
                c1s = {how: {"gate_0": [c1[nm][how][row][st].get("share_vs_own"),
                                        size_class(c1[nm][how][row][st])]} for how in v6b.BASELINE}
                if st in (*RULE_STAGES, *BESIDE):
                    for how in v6b.BASELINE:
                        c = rp[nm][how][row][st]["stage"]
                        c1s[how]["rule_persons"] = [c.get("share_vs_own"), size_class(c)]
                out[f"{nm} | {row} | {st}"] = {"signs": feats, "count_1": c1s}
    return out


# ================================================================ part 3: reliability


class FastN(l6.Fast):
    """v6l's Fast with R_mean (checked equal to part_vectors_n on the fixed parts and the first splits)."""

    def __init__(self, rec: k6.Rec):
        super().__init__(rec)

    def vectors(self, idx: np.ndarray) -> dict:
        out = super().vectors(idx)
        idx = np.sort(np.asarray(idx))
        sel = np.zeros(self.rec.n, dtype=bool)
        sel[idx] = True
        mask = np.zeros(self.g.size, dtype=bool)
        ok = self.sid >= 0
        mask[ok] = sel[self.sid[ok]]
        rm = r_mean_of(self.g, self.rec.lab, np.flatnonzero(mask))
        if (rm is None) != (out["R"] is None) or (rm is not None and rm[5] != out["R"][5]):
            raise SystemExit("R_mean and R differ in existence or in the sixth number in a split")
        out["R_mean"] = rm
        return out


def flat(v: dict, st: str) -> np.ndarray:
    x = v[st]
    if x is None:
        return np.full(REL_DIMS[st], np.nan)
    x = np.asarray(x, dtype=float)
    if x.shape != (REL_DIMS[st],) or not np.all(np.isfinite(x)):
        raise SystemExit(f"{st}: a vector of the wrong shape or a non-finite value")
    return x


def fixed_parts(texts: list, stages: tuple[str, ...]) -> dict:
    """v6l's fixed-part statistics (odd/even and halves) per feature."""
    out: dict = {how: {} for how in v6b.BASELINE}
    for how in v6b.BASELINE:
        for st in stages:
            A = np.array([flat(vec[how][0], st) for _, _, vec in texts])
            B = np.array([flat(vec[how][1], st) for _, _, vec in texts])
            for j in range(REL_DIMS[st]):
                fp = l6.fixed_part_stats(A[:, j], B[:, j])
                if fp["persons"] < MIN_SUMMARY:
                    for f in ("mean_A_minus_B", "mean_abs_A_minus_B", "median_abs_A_minus_B", "sd_A", "sd_B"):
                        fp.pop(f, None)
                out[how][f"{st}[{j}]"] = fp
    return out


def reliability(texts: list, ds: str, task: str, n_splits: int = l6.N_SPLITS) -> dict:
    """v6l.reliability with R_mean and the key starting 98. texts: (Rec of the text, place, fixed-part vectors)."""
    fasts = [FastN(rec) for rec, _, _ in texts]
    for f, (rec, _, vec) in zip(fasts, texts):
        for how in v6b.BASELINE:
            for h, part in enumerate(v6b.parts(rec.n, how)):
                if not l6.same_vectors(f.vectors(part), vec[how][h], REL_STAGES):
                    raise SystemExit(f"{ds} {task} {how}: the splits' part vectors differ from the count's")
    rngs = [np.random.default_rng([SEED, REL_KEY, k6.DATASET[ds], k6.TASK[task], place]) for _, place, _ in texts]
    feats = [(st, j) for st in REL_STAGES for j in range(REL_DIMS[st])]
    r_all = np.full((n_splits, len(feats)), np.nan)
    why_all: dict = {}
    n_per: dict = {st: [] for st in REL_STAGES}
    for s in range(n_splits):
        A = {st: np.empty((len(texts), REL_DIMS[st])) for st in REL_STAGES}
        B = {st: np.empty((len(texts), REL_DIMS[st])) for st in REL_STAGES}
        for p, (f, (rec, _, _)) in enumerate(zip(fasts, texts)):
            perm = rngs[p].permutation(rec.n)
            h = rec.n // 2
            va, vb = f.vectors(perm[:h]), f.vectors(perm[h:])
            if s < 2:
                ref_a, _ = part_vectors_n(rec, np.sort(perm[:h]))
                ref_b, _ = part_vectors_n(rec, np.sort(perm[h:]))
                if not (l6.same_vectors(va, ref_a, REL_STAGES) and l6.same_vectors(vb, ref_b, REL_STAGES)):
                    raise SystemExit(f"{ds} {task} split {s}: the splits' part vectors differ from the count's")
            for st in REL_STAGES:
                A[st][p], B[st][p] = flat(va, st), flat(vb, st)
        for st in REL_STAGES:
            both = ~(np.isnan(A[st]).any(axis=1) | np.isnan(B[st]).any(axis=1))
            n_per[st].append(int(both.sum()))
        for fi, (st, j) in enumerate(feats):
            r, why, _ = l6.corr_split(A[st][:, j], B[st][:, j])
            if r is None:
                why_all.setdefault(f"{st}[{j}]", {}).setdefault(why, 0)
                why_all[f"{st}[{j}]"][why] += 1
            else:
                r_all[s, fi] = r
    fixed = fixed_parts(texts, REL_STAGES)
    out: dict = {"persons": len(texts), "splits": n_splits, "features": {}}
    for fi, (st, j) in enumerate(feats):
        r = r_all[:, fi]
        r = r[~np.isnan(r)]
        nm = f"{st}[{j}]"
        cell = {"splits_with_a_value": int(r.size), "splits_without": why_all.get(nm, {}),
                "persons_per_split": ({"min": int(min(n_per[st])), "median": float(np.median(n_per[st])),
                                       "max": int(max(n_per[st]))} if n_per[st] else None),
                "fixed_parts": {h: fixed[h][nm] for h in v6b.BASELINE}}
        if 0 < r.size < MIN_SUMMARY:
            cell["summary_not_written"] = "fewer than 5 splits with a value"
        elif r.size:
            rs = 2 * r / (1 + r)
            cell.update({"mean_r": float(r.mean()), "r_percentile_95": [float(np.percentile(r, 2.5)),
                                                                        float(np.percentile(r, 97.5))],
                         "mean_rSB": float(rs.mean()), "rSB_percentile_95": [float(np.percentile(rs, 2.5)),
                                                                             float(np.percentile(rs, 97.5))],
                         "low": bool(rs.mean() < l6.LOW_RSB)})
        out["features"][nm] = cell
    return out


def beside_v6l(rel: dict, rl: dict) -> dict:
    """Per set and feature: the new mean rSB, v6l's on v6k's persons, and new minus v6l."""
    out: dict = {}
    for nm, r in rel.items():
        out[nm] = {}
        for f, cell in r["features"].items():
            old = rl["reliability"].get(nm, {}).get("features", {}).get(f)
            new_m = cell.get("mean_rSB")
            old_m = None if old is None else old.get("mean_rSB")
            out[nm][f] = {"new_mean_rSB": new_m, "new_interval": cell.get("rSB_percentile_95"),
                          "v6l_mean_rSB": old_m, "v6l_interval": None if old is None else old.get("rSB_percentile_95"),
                          "new_minus_v6l": (new_m - old_m) if new_m is not None and old_m is not None else None}
    return out


# ================================================================ checks against v6m and v6l


def jsonish(x):
    return json.loads(json.dumps(x, allow_nan=False))


def check_v6m(rec: dict, rm: dict) -> None:
    """Every cell that v6m counted equals v6m's record (its rows, stages and fields)."""
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for row, cells in rm["count_1"][nm][how].items():
                for k, theirs in cells.items():
                    if jsonish(rec["count_1"][nm][how][row][k]) != theirs:
                        raise SystemExit(f"count 1 {nm} {how} {row} {k}: differs from v6m's")
            for row, cells in rm["rule_persons"][nm][how].items():
                for k, theirs in cells.items():
                    mine = jsonish(rec["rule_persons"][nm][how][row][k])
                    for part in ("stage", *l6.GATES, "paired_minus_R_loc"):
                        if (part in mine) != (part in theirs):
                            raise SystemExit(f"rule persons {nm} {how} {row} {k} {part}: present on one side only")
                    for part, val in theirs.items():
                        if mine.get(part) != val:
                            raise SystemExit(f"rule persons {nm} {how} {row} {k} {part}: differs from v6m's")
        for row, stages in rm["part_4"][nm].items():
            if row not in rec["part_4_all"][nm]:
                raise SystemExit(f"part 4 {nm} {row}: not counted")
            for st, feats in stages.items():
                if jsonish(rec["part_4_all"][nm][row][st]) != feats:
                    raise SystemExit(f"part 4 {nm} {row} {st}: differs from v6m's")
        t_mine, t_theirs = jsonish(rec["tallies"][nm]), rm["tallies"][nm]
        for f in ("count_persons", "left_out_fewer_than_8_or_part_without_gap", "sentences"):
            if t_mine[f] != t_theirs[f]:
                raise SystemExit(f"tallies {nm} {f}: differ from v6m's")
        for row, theirs in t_theirs["doses"].items():
            for f, val in theirs.items():
                if f == "persons_same_as_next_step" and row.endswith("_50"):
                    continue
                if f not in t_mine["doses"][row] or t_mine["doses"][row][f] != val:
                    raise SystemExit(f"tallies {nm} {row} {f}: differ from v6m's")
    if rec["tallies"]["kupa_copy"]["count_persons"] != rm["tallies"]["kupa_copy"]["count_persons"] or \
            jsonish(rec["tallies"]["kupa_copy"]["sentences"]) != rm["tallies"]["kupa_copy"]["sentences"]:
        raise SystemExit("tallies kupa_copy: the count persons differ from v6m's")
    if jsonish(m6.doses(rec["rule_persons"])) != rm["part_2"]:
        raise SystemExit("v6m's part 2 recomputed from these cells differs from v6m's")


def check_v6l_fixed(texts: list, nm: str, rl: dict) -> None:
    """On v6k's persons, the fixed-part statistics of every feature of v6l's stages equal v6l's record."""
    fp = jsonish(fixed_parts(texts, l6.REL_STAGES))
    for how in v6b.BASELINE:
        for f, mine in fp[how].items():
            if mine != rl["reliability"][nm]["features"][f]["fixed_parts"][how]:
                raise SystemExit(f"v6k's persons {nm} {how} {f}: the fixed-part reliability differs from v6l's")


# ================================================================ predictions and main cells


NE = "not evaluable"


def predictions(rec: dict) -> dict:
    g0 = rec["part_1"]["gate_0"]
    c1 = rec["count_1"]

    def cls(nm, how, row, k):
        return size_class(c1[nm][how][row][k])

    cr = [cls(f"{ds}_{task}", how, "cross_100", "R_mean") for ds, task in MAIN for how in v6b.BASELINE]
    p1 = NE if "no count" in cr else sum(x == "above" for x in cr) >= 2
    lw = [g0[f"lwp_lwp | {how} | inside"]["R_mean"] for how in v6b.BASELINE]
    p2 = NE if any("no count" in e["classes"].values() for e in lw) else all(
        e["first_step_above"] in ("25", "50", "75") for e in lw)
    ku = [cls("kupa_essay", how, "inside_100", "R_mean") for how in v6b.BASELINE]
    p3 = NE if "no count" in ku else all(x == "above" for x in ku)
    rs = [size_class(rec["rule_persons"][f"{ds}_{task}"][how][row][k]["stage"]) for ds, task in MAIN
          for how in v6b.BASELINE for row in NEW_ROWS for k in RULE_STAGES]
    p4 = NE if "no count" in rs else not any(x == "above" for x in rs)
    l75 = [cls("lwp_lwp", how, "inside_75", "R_loc") for how in v6b.BASELINE]
    p5 = NE if "no count" in l75 else all(x in ("below", "undecided") for x in l75)
    rel = rec["part_3"]["reliability"]
    p6v = [(rel[nm]["features"]["R_mean[0]"].get("mean_rSB"), rel[nm]["features"]["R[0]"].get("mean_rSB"))
           for nm in ("kupa_essay", "lwp_lwp")]
    p6 = NE if any(a is None or b is None for a, b in p6v) else all(a < b for a, b in p6v)
    bes = rec["part_3"]["beside_v6l"]
    diffs = [bes[nm][f"{st}[{j}]"]["new_minus_v6l"] for nm in bes for st in l6.REL_STAGES
             for j in range(l6.REL_DIMS[st])]
    p7 = NE if any(d is None for d in diffs) else all(abs(d) <= P7_TOL for d in diffs)
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6, "P7": p7}


def main_cells_out(rec: dict) -> dict:
    def sh(res):
        return [round(res["share_vs_own"], 3), [round(x, 3) for x in res["share_vs_own_ci"]]] \
            if "share_vs_own_ci" in res else res

    out: dict = {}
    for where, e in rec["part_1"]["gate_0"].items():
        nm, how, fam = where.split(" | ")
        rows = [f"{fam}_{s}" for s, _ in STEPS]
        c1 = rec["count_1"][nm][how]
        out[f"part 1 | {where}"] = {
            r: {"first_step_above": e[r]["first_step_above"],
                **{s: [e[r]["classes"][s], sh(c1[row][r])] for (s, _), row in zip(STEPS, rows)}} for r in REFS}
        out[f"part 1 | {where}"]["paired_R_mean_minus_R_loc"] = {
            s: (None if v is None else [round(v["difference"], 3), [round(x, 3) for x in v["ci"]]])
            for s, v in e["paired_R_mean_minus_R_loc"].items()}
    for k in RULE_STAGES:
        for ds, task in MAIN:
            for how in v6b.BASELINE:
                for row in NEW_ROWS:
                    c = rec["rule_persons"][f"{ds}_{task}"][how][row][k]["stage"]
                    fixed = row.startswith("inside") and k in INSIDE_FIXED
                    out[f"part 1 | 75 | {k} | {ds}_{task} | {how} | {row}"] = (
                        ["fixed by construction, not read", sh(c)] if fixed else [size_class(c), sh(c)])
    for ds, task in MAIN:
        nm = f"{ds}_{task}"
        for row in SIGNED_ROWS:
            for st, fs in (("R_loc", ("0", "4")), ("R_mean", ("0", "4")), ("I", ("0", "1", "2"))):
                e = rec["part_2"][f"{nm} | {row} | {st}"]
                out[f"part 2 | {nm} | {row} | {st}"] = {
                    "signs": {f: (None if e["signs"].get(f) is None else
                                  [round(e["signs"][f]["share_positive"], 3), round(e["signs"][f]["share_negative"], 3),
                                   "fixed" if e["signs"][f]["fixed_by_construction"] else ""]) for f in fs},
                    "count_1_gate_0": {how: e["count_1"][how]["gate_0"] for how in v6b.BASELINE}}
    for nm, feats in rec["part_3"]["beside_v6l"].items():
        for f, v in feats.items():
            if f.split("[")[0] in ("R", "R_mean", "B@2000", "I"):
                out[f"part 3 | {nm} | {f}"] = v
    return out


# ================================================================ run


def count_all(n_series: dict, n_ids: list, kplace: dict, ln_series: dict, ln_ids: list, lplace: dict, m: dict,
              rl: dict, n_splits: int = l6.N_SPLITS, rm: dict | None = None) -> dict:
    """Everything on v6m's persons; with rm (v6m's record), the reuse check runs before the reliability's splits."""
    rec: dict = {"checks": {}, "count_1": {}, "rule_persons": {}, "tallies": {}, "part_4_all": {}, "gate_0_refs": {}}
    texts: dict = {}
    for ds, task, series, ids, place, rows in (("kupa", "essay", n_series, n_ids, kplace, ROWS),
                                               ("kupa", "copy", n_series, n_ids, kplace, ["text/before"]),
                                               ("lwp", "lwp", ln_series, ln_ids, lplace, ROWS)):
        nm = f"{ds}_{task}"
        b = build(series, ids, ds, task, place, m, rows)
        texts[nm] = b["texts"]
        rec["tallies"][nm] = {"count_persons": len(b["persons"]), "left_out_fewer_than_8_or_part_without_gap":
                              b["left_out"], "sentences": l6.five_or_none(b["sentences"]),
                              "doses": {row: dose_tally_summary(t) for row, t in b["dose_tallies"].items()}}
        if task == "copy":
            continue
        hv = hv_of(b, rows)
        rec["count_1"][nm] = count1(hv, ds, task)
        rec["gate_0_refs"][nm] = gate0_refs(hv, rec["count_1"][nm], ds, task)
        rec["rule_persons"][nm] = rule_persons(hv, ds, task, (*MAIN_ROWS, *DOSE_ROWS))
        persons = list(b["persons"].values())
        rec["part_4_all"][nm] = {row: signed_set([(p["whole"]["text/before"], p["whole"][row]) for p in persons],
                                                 k6.DATASET[ds], k6.TASK[task], ROW_IX[row])
                                 for row in (*m6.SIGNED_ROWS[:4], *SIGNED_ROWS)}
        del hv, b
    rec["checks"]["rows"] = ("every row is a permutation of the text's gaps; no gap position is in two swaps of a plan; "
                             "the four steps are nested; inside_100 equals inside/before for every count person")
    rec["checks"]["construction"] = ("v6m's identities with inside_75, R_mean's existence and sixth number, and the "
                                     "whole-record identities held for every count person")
    rec["part_1"] = part1(rec["count_1"], rec["gate_0_refs"], rec["rule_persons"])
    rec["part_2"] = part2(rec["part_4_all"], rec["count_1"], rec["rule_persons"])
    if rm is not None:
        check_v6m(rec, rm)
        rec["checks"]["reuse"] = ("v6m's persons, count persons, every count 1 and rule-person cell, part 4 cell and "
                                  "dose tally (but same_as_next_step at 50), and part 2 recomputed, equal v6m's record")
    rel = {nm: reliability(t, *nm.split("_", 1), n_splits=n_splits) for nm, t in texts.items()}
    rec["part_3"] = {"reliability": rel, "beside_v6l": beside_v6l(rel, rl)}
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
    rk, rl, rm = (json.loads(p.read_text()) for p in (REC_K, REC_L, REC_M))
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    m = c4["mapping"]
    rec: dict = {"persons": {}, "checks": {}}

    # KUPA: v6k's persons (for the reliability check) and v6m's
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6b.read_kupa_b(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    kplace = {pid: i for i, pid in enumerate(ids_sorted)}
    order = v6.draw_order(ids_sorted, 0)
    elig = lambda pid: k6.kupa_eligible(kupa, pid)  # noqa: E731
    kk_ids, kk_series = m6.draw_checked(order, elig, rk["persons"]["kupa"]["ids_looked_at_from_437"], m6.KUPA_V6K[0],
                                        rk["persons"]["kupa"]["ids_sha256"], "v6k's KUPA persons")
    n_ids, n_series = m6.draw_checked(order, elig, rm["persons"]["kupa"]["ids_looked_at_from_656"], m6.SKIP["kupa"],
                                      rm["persons"]["kupa"]["ids_sha256"], "v6m's KUPA persons")
    rec["persons"]["kupa"] = {k: rm["persons"]["kupa"][k] for k in ("ids_looked_at_from_656", "ids_sha256")}
    del kupa

    # LWP
    lwp, lids, _ = v6e.read_lwp(ROOT / d_lwp["file"], None)
    lplace = {pid: i for i, pid in enumerate(sorted(lids))}
    lorder = v6.draw_order(sorted(lids), 2)
    lelig = lambda pid: k6.lwp_eligible(lwp, pid)  # noqa: E731
    lk_ids, lk_series = m6.draw_checked(lorder, lelig, rk["persons"]["lwp"]["ids_looked_at_from_204"], m6.LWP_V6K[0],
                                        rk["persons"]["lwp"]["ids_sha256"], "v6k's LWP persons")
    ln_ids, ln_series = m6.draw_checked(lorder, lelig, rm["persons"]["lwp"]["ids_looked_at_from_411"], m6.SKIP["lwp"],
                                        rm["persons"]["lwp"]["ids_sha256"], "v6m's LWP persons")
    rec["persons"]["lwp"] = {k: rm["persons"]["lwp"][k] for k in ("ids_looked_at_from_411", "ids_sha256")}
    del lwp

    # the reliability check on v6k's persons (fixed parts against v6l's record)
    for ds, task, series, ids, place in (("kupa", "essay", kk_series, kk_ids, kplace),
                                         ("kupa", "copy", kk_series, kk_ids, kplace),
                                         ("lwp", "lwp", lk_series, lk_ids, lplace)):
        nm = f"{ds}_{task}"
        b = build(series, ids, ds, task, place, m, ["text/before"])
        if len(b["persons"]) != rl["reliability"][nm]["persons"]:
            raise SystemExit(f"v6k's persons {nm}: the count persons differ from v6l's")
        check_v6l_fixed(b["texts"], nm, rl)
        del b
    rec["checks"]["reliability_reuse"] = ("on v6k's persons, the fixed-part statistics of every feature of v6l's "
                                          "stages equal v6l's record")
    del kk_series, lk_series

    new = count_all(n_series, n_ids, kplace, ln_series, ln_ids, lplace, m, rl, rm=rm)
    if "reuse" not in new["checks"]:
        raise SystemExit("the reuse check against v6m did not run")
    rec["checks"].update(new.pop("checks"))
    rec.update(new)
    p4 = rec.pop("part_4_all")
    rec["part_2_signed_dose_rows"] = {nm: {row: p4[nm][row] for row in SIGNED_ROWS} for nm in p4}
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
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6n-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "paired_indices": "every paired difference on rule persons uses the rule stage's bootstrap indices (its key with "
                      "the rule gate); R_mean - R_loc at gate 0 uses R_mean's gate 0 key",
    "gate_0_persons": "R_mean and R_loc at gate 0 are counted on all count persons with the stage in the text and the "
                      "row; the run stops if their persons differ",
    "r_mean_held": "R_mean uses the gaps held by the part's sentences, as v6b.loc (the union of the sentences' held "
                   "gaps)",
    "dose_tallies": "as v6m; positions_changed counts the gaps of the row that differ from the text's by more than v6's "
                    "tolerance; 'same as next step' is counted at 25, 50 and 75",
    "whole_record": "as v6m: the whole-record vector of a row is the part vector of all its sentences",
    "part_4_rows": "part 4 is counted for v6m's rows (to check against v6m) and the eight dose rows; only the dose rows "
                   "are written (v6m's other rows are in v6m's record)",
    "copy_rows": "KUPA copy is built with text/before only (the reliability uses its text)",
    "reliability_check": "the v6k persons' check compares the fixed parts only (v6l's random splits used key 93; here "
                         "98); the splits' vectors are checked against part_vectors_n on the fixed parts and the "
                         "first 2 splits, as v6l",
    "predictions": "P1, P2, P3 and P5 read the gate 0 cells (persons with R); P4 every rule stage's cell at 75 on "
                   "its rule persons, including the cells fixed by construction (as the conditions word it); P7 compares every feature of v6l's stages in the three sets; a "
                   "prediction whose input has no count is 'not evaluable'",
    "small_counts": "as v6m: a count over fewer than 5 persons keeps its share and interval but not its medians; a "
                    "fixed-part reliability over fewer than 5 persons keeps r and rSB but not the means and SDs",
    "first_step_unreadable": "a family whose steps before the first 'above' include one with no count has its first "
                             "step 'not evaluable'",
    "reuse_order": "the reuse check against v6m runs inside count_all before the reliability's 5000 splits",
    "signed_zero": "as v6m: a change counts as zero if |change| <= 1e-9 in the feature's units",
}


if __name__ == "__main__":
    main()
