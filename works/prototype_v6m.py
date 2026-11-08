"""Prototype v6m: v6k's rule recounted on persons not used before (part 0), the same KUPA person's copy against their
essay as a real difference with R_loc as the gate (part 1), doses of 25%, 50% and 100% of two operations (part 2),
U's two medians alone (part 3) and signed changes at the level of the group (part 4) (decisions 14-16 of
.steering/20261108-paper-a-jowr-reframe; decisions 1-5 of .steering/20261108-v6m-real-contrast-and-dose).

Usage:  uv run python works/prototype_v6m.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6m.json (committed before this script is first run), and the conditions and
            records named in it (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6m/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value, per-person change or id is written.
"""

from __future__ import annotations

import atexit
import datetime as dt
import hashlib
import json
import math
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

v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6m.json"
REC_L = ROOT / "results" / "prototype_v6l" / "record.json"
REC_K = ROOT / "results" / "prototype_v6k" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6m"
CODE = ["works/prototype_v6m.py", *l6.CODE]
SEED = k6.SEED
REST_MS = k6.REST_MS
TH2000 = {"2000": REST_MS}

STAGE = {**l6.STAGE, "U_med": 11}
STEPS = (("25", 0.25), ("50", 0.50), ("100", 1.00))
FAMILIES = ("inside", "cross")
DOSE_ROWS = [f"{f}_{s}" for f in FAMILIES for s, _ in STEPS]
ROWS = [*k6.ROWS, *DOSE_ROWS]
ROW_IX = {row: i for i, row in enumerate(ROWS)}
CONTRAST_ROW_IX = 13
COUNTING = k6.COUNTING
MAIN = k6.MAIN
MAIN_ROWS = k6.MAIN_ROWS
RULE_STAGES = k6.RULE_STAGES  # U@2000, B@2000, Bsd@2000, I
BESIDE = ("U_med", "B_complete@2000", "Bsd_complete@2000")
GATES = l6.GATES  # R, R_loc, R_lag
KEYS = ["S0", "S1", "U@2000", "U_med", "B@2000", "Bsd@2000", "I", "R", "R_loc", "R_lag", "B_complete@2000",
        "Bsd_complete@2000"]
SKIP = {"kupa": 656, "lwp": 411}
KUPA_V6K = (437, 656)
LWP_V6K = (204, 411)
BASES = {"copy": 0, "essay": 1}
CONTRAST_LABELLED = RULE_STAGES
CONTRAST_BESIDE = ("U_med", "B_complete@2000", "Bsd_complete@2000", "S0", "R", "R_loc", "R_lag")
SIGNED_STAGES = ("U@2000", "B@2000", "Bsd@2000", "I", "R_loc", "R_lag", "B_complete@2000")
SIGNED_ROWS = ("inside/before", "shuffled/before", "shuffled_in_halves/before", "random_half", *DOSE_ROWS)
INSIDE_FIXED = ("U@2000", "U_med", "B@2000")
SIGN_TOL = 1e-9
CONSTR_TOL = k6.CONSTR_TOL
MIN_SUMMARY = l6.MIN_SUMMARY
SIZE = k6.SIZE
sk, base_of, size_class, label = k6.sk, k6.base_of, k6.size_class, k6.label


# ================================================================ part vectors


def part_vectors_m(rec: k6.Rec, idx: np.ndarray) -> tuple[dict, dict]:
    """v6k's part vectors at 2 s, with v6l's R_loc, R_lag and B_complete@2000, and U_med (U's two medians)."""
    out, tally = k6.part_vectors(rec, idx, TH2000)
    l6.add_new(out, tally, rec, idx)
    u = out["U@2000"]
    out["U_med"] = None if u is None else u[[0, 2]].copy()
    return out, tally


# ================================================================ the dose plans


def sentence_of_gap(starts: list[int], n_presses: int) -> np.ndarray:
    """The sentence of each gap under 'before' (gap j precedes press j + 1)."""
    out = np.empty(n_presses - 1, dtype=int)
    for k, (a, b) in enumerate(ks._sentence_bounds(starts, n_presses)):
        out[max(a - 1, 0):b - 1] = k
    return out


def inside_plan(s: v6.Series, key: list[int]) -> list[tuple[int, int]]:
    """The swaps of rests_inside_at (its draws, in sentence order), put in the order of the conditions' permutation."""
    t = ks.check_times(s.times)
    ws = np.array(sorted(set(s.word_starts)), dtype=int)
    rng = np.random.default_rng([SEED, 6, 5, *key])
    cands = []
    for a, b in ks._sentence_bounds(s.starts, t.size)[1:]:
        inner = ws[(ws > a) & (ws < b)]
        if inner.size == 0:
            continue
        j = int(inner[int(rng.integers(inner.size))]) - 1
        cands.append((a - 1, j))
    if not cands:
        return []
    order = np.random.default_rng([SEED, 94, 1, *key]).permutation(len(cands))
    return [cands[int(i)] for i in order]


def cross_plan(s: v6.Series, key: list[int]) -> list[tuple[int, int | None]]:
    """The givers (long sentence-initial gaps) in the conditions' order, each with its partner or None."""
    t = ks.check_times(s.times)
    g = np.diff(t)
    long = g >= REST_MS - ks.TOL_MS
    bounds = ks._sentence_bounds(s.starts, t.size)
    sent = sentence_of_gap(s.starts, t.size)
    givers = [a - 1 for a, _ in bounds[1:] if long[a - 1]]
    if not givers:
        return []
    ws = np.array(sorted(set(s.word_starts)), dtype=int)
    pool = []
    for a, b in bounds:
        pool.extend(int(w) - 1 for w in ws[(ws > a) & (ws < b)] if not long[int(w) - 1])
    pool = np.array(sorted(pool), dtype=int)
    free = np.ones(pool.size, dtype=bool)
    order = np.random.default_rng([SEED, 94, 2, *key]).permutation(len(givers))
    rng = np.random.default_rng([SEED, 94, 3, *key])
    plan: list[tuple[int, int | None]] = []
    for i in order:
        gi = givers[int(i)]
        ok = np.flatnonzero(free & (sent[pool] != sent[gi]))
        if ok.size == 0:
            plan.append((gi, None))
            continue
        pick = int(ok[int(rng.integers(ok.size))])
        free[pick] = False
        plan.append((gi, int(pool[pick])))
    return plan


def n_at(p: float, m: int) -> int:
    return math.ceil(p * m)


def apply_swaps(gaps: np.ndarray, swaps) -> np.ndarray:
    g = np.asarray(gaps, dtype=float).copy()
    for a, b in swaps:
        if b is None:
            continue
        g[a], g[b] = g[b], g[a]
    return g


def times_of(gaps: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(gaps)])


def dose_rows(s: v6.Series, key: list[int]) -> tuple[dict, dict]:
    """The press times of the six dose rows and the tallies of what moved."""
    g = np.diff(ks.check_times(s.times))
    total = float(g.sum())
    n = len(s.ranges("before"))
    sent = sentence_of_gap(s.starts, s.times.size)
    part_of_sent = {how: np.zeros(n, dtype=int) for how in v6b.BASELINE}
    for how in v6b.BASELINE:
        part_of_sent[how][v6b.parts(n, how)[1]] = 1
    out, tally = {}, {}
    for fam, plan in (("inside", inside_plan(s, key)), ("cross", cross_plan(s, key))):
        m = len(plan)
        sizes = [n_at(p, m) for _, p in STEPS]
        mades = [[(a, b) for a, b in plan[:k] if b is not None] for k in sizes]
        for i, ((st, _), k) in enumerate(zip(STEPS, sizes)):
            row = f"{fam}_{st}"
            sw = plan[:k]
            made = mades[i]
            out[row] = times_of(apply_swaps(g, sw))
            if fam == "cross":
                moved = float(sum(g[a] for a, _ in made))
            else:
                moved = float(sum(abs(g[a] - g[b]) for a, b in made))
            tally[row] = {"m": m, "swaps": len(made), "share_swapped": (len(made) / m) if m else None,
                          "time_moved_share": moved / total, "left_without_partner": len(sw) - len(made),
                          "m_zero": m == 0,
                          "same_as_next_step": i + 1 < len(sizes) and made == mades[i + 1],
                          "made": made,
                          "crossing_parts": {how: int(sum(part_of_sent[how][sent[a]] != part_of_sent[how][sent[b]]
                                                          for a, b in made)) for how in v6b.BASELINE}}
    return out, tally


def check_permutation(text_gaps: np.ndarray, rows_times: dict) -> None:
    srt = np.sort(text_gaps)
    for row, t in rows_times.items():
        g = np.diff(t)
        if g.size != text_gaps.size or not np.all(np.abs(np.sort(g) - srt) <= ks.TOL_MS):
            raise SystemExit(f"{row}: not a permutation of the text's gaps")


def check_rows(text_gaps: np.ndarray, rows_times: dict, inside_times: np.ndarray | None,
               tally: dict | None = None) -> None:
    """Each row a permutation of the text's gaps; the doses nested (the swaps applied at each step a prefix of the
    next step's, and the changed positions nested); inside_100 equal to v6k's inside/before."""
    check_permutation(text_gaps, rows_times)
    changed = {row: set(np.flatnonzero(np.abs(np.diff(t) - text_gaps) > ks.TOL_MS).tolist())
               for row, t in rows_times.items()}
    for fam in FAMILIES:
        a, b, c = (changed[f"{fam}_{s}"] for s, _ in STEPS)
        if not (a <= b <= c):
            raise SystemExit(f"{fam}: the doses are not nested")
        if tally is not None:
            ms = [tally[f"{fam}_{s}"]["made"] for s, _ in STEPS]
            if not all(ms[i + 1][:len(ms[i])] == ms[i] for i in range(len(ms) - 1)):
                raise SystemExit(f"{fam}: the swaps of the doses are not nested")
    if inside_times is not None and not np.array_equal(rows_times["inside_100"], inside_times):
        raise SystemExit("inside_100 differs from v6k's inside/before")


# ================================================================ persons and rows


def person_rows(s: v6.Series, kinds: list[str], key: list[int], m: dict, rows: list[str]
                ) -> tuple[dict, dict, dict, dict]:
    """Per row and baseline the (A, B) vectors; per row the whole-record vector; per row and baseline the tallies;
    the dose tallies."""
    lab = v6b.labels(s, kinds)
    n = len(s.ranges("before"))
    dose_t, dose_tally = dose_rows(s, key) if any(r in DOSE_ROWS for r in rows) else ({}, {})
    out, whole, tallies, times = {}, {}, {}, {}
    for row in rows:
        t = dose_t[row] if row in DOSE_ROWS else k6.row_times(s, row, key)
        times[row] = t
        rec = k6.Rec(t, s.ranges(COUNTING.get(row, "before")), lab, m)
        if rec.n != n:
            raise SystemExit(f"{row}: {rec.n} sentences, not {n}")
        out[row], tallies[row] = {}, {}
        for how in v6b.BASELINE:
            A, B = v6b.parts(n, how)
            va, ta = part_vectors_m(rec, A)
            vb, tb = part_vectors_m(rec, B)
            out[row][how] = (va, vb)
            tallies[row][how] = (ta, tb)
        if not row.startswith("text/") or row == "text/before":
            whole[row] = part_vectors_m(rec, np.arange(n))[0]
    check_permutation(np.diff(s.times), {r: t for r, t in times.items() if not r.startswith("text/")})
    if dose_t:
        check_rows(np.diff(s.times), dose_t, times.get("inside/before", k6.row_times(s, "inside/before", key)),
                   dose_tally)
    return out, whole, tallies, dose_tally


def check_construction(vec: dict, tallies: dict) -> None:
    """At 2 s: B under the rows and baselines that keep each run's multiset (v6k's), U, U_med and S1 under inside and
    the inside doses, and I's first number under all of them."""
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
            for nm, (x, y) in pairs.items():
                if x is None and y is None:
                    continue
                if (x is None) != (y is None):
                    raise SystemExit(f"{row} {how} {nm}: a value exists on one side only")
                if not np.all(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)) <= CONSTR_TOL):
                    raise SystemExit(f"{row} {how} {nm}: changed where it cannot by construction")


def build(series: dict, ids: list, ds: str, task: str, place: dict, m: dict, rows: list[str]) -> dict:
    """Per count person (by id): the part vectors per row and baseline, the whole-record vectors, and the tallies."""
    out: dict = {"persons": {}, "left_out": 0, "dose_tallies": {row: [] for row in DOSE_ROWS}, "sentences": []}
    for pid in ids:
        s, kinds = series[pid][task] if ds == "kupa" else series[pid]
        if not k6.count1_ok(s):
            out["left_out"] += 1
            continue
        vec, whole, tallies, dose_tally = person_rows(s, kinds, [k6.DATASET[ds], k6.TASK[task], place[pid]], m, rows)
        check_construction(vec, tallies)
        out["persons"][pid] = {"vec": vec, "whole": whole}
        out["sentences"].append(len(s.ranges("before")))
        for row in dose_tally:
            out["dose_tallies"][row].append(dose_tally[row])
    return out


def hv_of(b: dict, rows: list[str]) -> dict:
    """hv[how][row] = list over the count persons (in id order of the draw) of (A, B)."""
    return {how: {row: [p["vec"][row][how] for p in b["persons"].values()] for row in rows} for how in v6b.BASELINE}


def dose_tally_summary(tallies: list[dict]) -> dict:
    if not tallies:
        return {}
    five = l6.five_or_none
    return {"persons": len(tallies), "m": five([t["m"] for t in tallies]),
            "swaps": five([t["swaps"] for t in tallies]),
            "share_swapped": five([t["share_swapped"] for t in tallies if t["share_swapped"] is not None]),
            "time_moved_share": five([t["time_moved_share"] for t in tallies]),
            "left_without_partner": int(sum(t["left_without_partner"] for t in tallies)),
            "persons_m_zero": int(sum(t["m_zero"] for t in tallies)),
            "persons_same_as_next_step": int(sum(t["same_as_next_step"] for t in tallies)),
            "crossing_parts": {how: five([t["crossing_parts"][how] for t in tallies]) for how in v6b.BASELINE}}


# ================================================================ count 1 and the rule persons


def key1(ds: str, task: str, key_name: str, row: str, how: str, gate: int) -> list[int]:
    st, _ = base_of(key_name)
    return [SEED, 95, k6.DATASET[ds], k6.TASK[task], STAGE[st], ROW_IX[row], v6b.BASELINE[how], 0, gate]


def gate_of(key_name: str) -> int:
    return 1 + STAGE[base_of(key_name)[0]]


def count1(hv: dict, ds: str, task: str) -> dict:
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for row in ROWS[1:]:
            out[how][row] = {k: one_cell(hv, row, how, k, key1(ds, task, k, row, how, 0)) for k in KEYS}
    return out


def own_contrib(first: list, sel: list, st: str) -> np.ndarray:
    """The per-person contributions of v6b.one (1, 1/2, 0 against the own two parts), with its scaling."""
    t_all = np.array([v[h][st] for v in first for h in (0, 1)])
    sd = t_all.std(axis=0)
    keep = sd > 0

    def z(x):
        return np.asarray(x, dtype=float)[keep] / sd[keep]

    base = np.array([np.linalg.norm(z(v[0][st]) - z(v[1][st])) for v in first])
    change = np.array([(np.linalg.norm(z(r[0][st]) - z(t[0][st])) + np.linalg.norm(z(r[1][st]) - z(t[1][st]))) / 2
                       for t, r in zip(first, sel)])
    return np.where(change > base, 1.0, np.where(change == base, 0.5, 0.0))


def rule_cell(hv: dict, ds: str, task: str, row: str, how: str, k: str) -> dict:
    """A stage on its rule persons (stage and R), with R, R_loc and R_lag there, and the paired difference of the
    stage's and R_loc's contributions over the stage's resamples."""
    gate = gate_of(k)
    c = {"stage": one_cell(hv, row, how, k, key1(ds, task, k, row, how, gate), extra="R")}
    for g in GATES:
        c[g] = one_cell(hv, row, how, g, key1(ds, task, g, row, how, gate), extra=k)
    if "share_vs_own_ci" in c["stage"] and "share_vs_own_ci" in c["R_loc"]:
        first, sel = hv[how]["text/before"], hv[how][row]
        ok = [i for i in range(len(first)) if all(v6b.has(first[i], x) and v6b.has(sel[i], x) for x in (k, "R"))]
        f, s = [first[i] for i in ok], [sel[i] for i in ok]
        a, b = own_contrib(f, s, k), own_contrib(f, s, "R_loc")
        if abs(float(a.mean()) - c["stage"]["share_vs_own"]) > 1e-12 or \
                abs(float(b.mean()) - c["R_loc"]["share_vs_own"]) > 1e-12:
            raise SystemExit(f"{ds} {task} {row} {how} {k}: the contributions do not give the shares")
        idx = np.random.default_rng(key1(ds, task, k, row, how, gate)).integers(0, len(ok), (v6.N_BOOT, len(ok)))
        d = a - b
        c["paired_minus_R_loc"] = {"difference": float(d.mean()), "ci": v6.ci(d, idx)}
    return c


def rule_persons(hv: dict, ds: str, task: str, rows: tuple[str, ...]) -> dict:
    return {how: {row: {k: rule_cell(hv, ds, task, row, how, k) for k in (*RULE_STAGES, *BESIDE)} for row in rows}
            for how in v6b.BASELINE}


# ================================================================ part 1: the real contrast


def guard(res: dict) -> dict:
    """No median over fewer than MIN_SUMMARY persons (a median of a few persons is close to a person's value)."""
    if "persons" not in res or "median_change" not in res:
        return res
    if res["persons"] < MIN_SUMMARY:
        for f in ("median_change", "median_base", "median_between", "median_change_over_base"):
            res.pop(f, None)
        res["medians_not_written"] = "fewer than 5 persons"
    elif res["persons"] - res.get("base_zero", 0) < MIN_SUMMARY:
        res["median_change_over_base"] = None
    return res


def one_cell(hv: dict, row: str, how: str, key_name: str, rng_key: list[int], extra: str | None = None) -> dict:
    return guard(l6.one_cell(hv, row, how, key_name, rng_key, extra))


def contrast_one(cA: np.ndarray, cB: np.ndarray, eA: np.ndarray, eB: np.ndarray, base: str, idx: np.ndarray,
                 scale: str = "pooled", all_pairs: bool = False) -> tuple[dict, np.ndarray]:
    """The share of persons whose essay-copy change exceeds the distance between the base task's two parts."""
    n = cA.shape[0]
    X = np.vstack([cA, cB, eA, eB])
    if not np.all(np.isfinite(X)):
        raise SystemExit("a non-finite value in the contrast")
    if scale == "pooled":
        sd = X.std(axis=0)
    else:
        vc = np.vstack([cA, cB]).var(axis=0)
        ve = np.vstack([eA, eB]).var(axis=0)
        sd = np.sqrt((vc + ve) / 2)
    keep = sd > 0
    if not keep.any():
        raise SystemExit("every feature of the contrast is constant")

    def z(x):
        return x[:, keep] / sd[keep]

    zcA, zcB, zeA, zeB = z(cA), z(cB), z(eA), z(eB)

    def nrm(x):
        return np.linalg.norm(x, axis=1)

    if all_pairs:
        change = (nrm(zeA - zcA) + nrm(zeA - zcB) + nrm(zeB - zcA) + nrm(zeB - zcB)) / 4
    else:
        change = (nrm(zeA - zcA) + nrm(zeB - zcB)) / 2
    b = nrm(zcA - zcB) if base == "copy" else nrm(zeA - zeB)
    own = np.where(change > b, 1.0, np.where(change == b, 0.5, 0.0))
    c_own = v6.ci(own, idx)
    pos = b > 0
    return (guard({"persons": n, "dropped_features": [int(j) for j in np.flatnonzero(~keep)],
             "share_vs_own": float(own.mean()), "share_vs_own_ci": c_own, "reading": v6.reading(*c_own),
             "median_change": float(np.median(change)), "median_base": float(np.median(b)),
             "median_change_over_base": float(np.median(change[pos] / b[pos])) if pos.any() else None,
             "base_zero": int((~pos).sum())}), own)


def contrast_arrays(pairs: list[tuple[dict, dict]], how: str, st: str) -> tuple[np.ndarray, ...]:
    """pairs: per person (copy vec, essay vec) where vec[how] = (A, B)."""
    return tuple(np.array([np.asarray(p[t][how][h][st], dtype=float) for p in pairs]) for t in (0, 1) for h in (0, 1))


def has4(p: tuple[dict, dict], how: str, keys: tuple[str, ...]) -> bool:
    return all(p[t][how][h][k] is not None for t in (0, 1) for h in (0, 1) for k in keys)


def key_c(st: str, how: str, base: str, gate: int) -> list[int]:
    return [SEED, 96, STAGE[base_of(st)[0]], v6b.BASELINE[how], BASES[base], gate]


def contrast_cell(pairs: list, st: str, how: str, base: str, gate: int, extra: str | None = None,
                  beside: bool = False) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    need = (st,) + ((extra,) if extra else ())
    ok = [p for p in pairs if has4(p, how, need)]
    if len(ok) < 2:
        return {"too_few_persons": True, "persons": len(ok)}, None, None
    cA, cB, eA, eB = contrast_arrays(ok, how, st)
    idx = np.random.default_rng(key_c(st, how, base, gate)).integers(0, len(ok), (v6.N_BOOT, len(ok)))
    res, own = contrast_one(cA, cB, eA, eB, base, idx)
    res["left_out_no_vector"] = len(pairs) - len(ok)
    if beside:
        res["within_scale"] = contrast_one(cA, cB, eA, eB, base, idx, scale="within")[0]
        res["all_pairs"] = contrast_one(cA, cB, eA, eB, base, idx, all_pairs=True)[0]
    return res, own, idx


def contrast_stage(pairs: list, k: str) -> dict:
    """Per cell (baseline | base): the stage on its own contrast persons (gate 0, with the beside scales), and, unless
    k is R_loc, on its contrast rule persons (k and R_loc) with R_loc, R and R_lag and the paired difference."""
    out: dict = {}
    g = gate_of(k)
    for how in v6b.BASELINE:
        for base in BASES:
            cell = {"own_persons": contrast_cell(pairs, k, how, base, 0, beside=True)[0]}
            if k != "R_loc":
                stage, a, idx = contrast_cell(pairs, k, how, base, g, extra="R_loc", beside=True)
                cell["stage"] = stage
                for gt in GATES:
                    cell[gt] = contrast_cell(pairs, gt, how, base, g, extra=k)[0]
                if a is not None:
                    ok = [p for p in pairs if has4(p, how, (k, "R_loc"))]
                    cA, cB, eA, eB = contrast_arrays(ok, how, "R_loc")
                    _, b = contrast_one(cA, cB, eA, eB, base, idx)
                    if abs(float(b.mean()) - cell["R_loc"]["share_vs_own"]) > 1e-12:
                        raise SystemExit(f"contrast {k} {how} {base}: R_loc's contributions do not give its share")
                    d = a - b
                    cell["paired_minus_R_loc"] = {"difference": float(d.mean()), "ci": v6.ci(d, idx)}
            out[f"{how} | {base}"] = cell
    return out


def contrast(pairs: list) -> dict:
    """Part 1: every stage on its own persons and on its contrast rule persons; labels only for the labelled ones."""
    return {"persons": len(pairs), "labelled": {k: contrast_stage(pairs, k) for k in CONTRAST_LABELLED},
            "beside": {k: contrast_stage(pairs, k) for k in CONTRAST_BESIDE}}


def contrast_rule(c: dict) -> dict:
    out = {}
    for k in CONTRAST_LABELLED:
        classes, status = [], {}
        for where, cell in c["labelled"][k].items():
            if size_class(cell["stage"]) == "no count" or size_class(cell["R_loc"]) == "no count":
                status[where] = "fewer than 2 rule persons"
                continue
            if size_class(cell["R_loc"]) != "above":
                status[where] = "control not met"
                continue
            cls = size_class(cell["stage"])
            status[where] = f"read: {cls}"
            classes.append(cls)
        by_base = {base: {c_: sum(v == f"read: {c_}" for w, v in status.items() if w.endswith(f"| {base}"))
                          for c_ in ("above", "below", "undecided")} for base in BASES}
        out[k] = {"label": label(classes), "cells_read": len(classes), "status": status, "classes_by_base": by_base,
                  "R_class": {w: size_class(cell["R"]) for w, cell in c["labelled"][k].items()},
                  "R_loc_class": {w: size_class(cell["R_loc"]) for w, cell in c["labelled"][k].items()}}
    return out


# ================================================================ part 2: the doses


def doses(rp: dict) -> dict:
    """rp[nm][how][row][k] over the dose rows: classes per step, the first step above, rising shares."""
    out: dict = {}
    for k in (*RULE_STAGES, *BESIDE):
        out[k] = {}
        for ds, task in MAIN:
            nm = f"{ds}_{task}"
            for how in v6b.BASELINE:
                for fam in FAMILIES:
                    rows = [f"{fam}_{s}" for s, _ in STEPS]
                    cells = [rp[nm][how][row][k] for row in rows]
                    fixed = fam == "inside" and k in INSIDE_FIXED
                    entry: dict = {"fixed_by_construction": fixed}
                    for part in ("stage", *GATES):
                        cls = [size_class(c[part]) for c in cells]
                        shares = [c[part].get("share_vs_own") for c in cells]
                        first = next((s for (s, _), x in zip(STEPS, cls) if x == "above"), "none")
                        rising = (None if None in shares else bool(shares[0] <= shares[1] <= shares[2]))
                        entry[part] = {"classes": dict(zip([s for s, _ in STEPS], cls)),
                                       "first_step_above": first if not (fixed and part == "stage") else "not read",
                                       "rising": rising}
                    entry["paired_minus_R_loc"] = {s: c.get("paired_minus_R_loc") for (s, _), c in zip(STEPS, cells)}
                    out[k][f"{nm} | {how} | {fam}"] = entry
    return out


# ================================================================ part 4: signed changes


def boot_median(d: np.ndarray, idx: np.ndarray) -> list[float]:
    m = np.median(d[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def signed(diffs: np.ndarray, rng_key: list[int]) -> dict | None:
    """diffs: persons x 1 values of one feature."""
    d = np.asarray(diffs, dtype=float)
    if not np.all(np.isfinite(d)):
        raise SystemExit(f"a non-finite signed change for {rng_key}")
    if d.size < MIN_SUMMARY:
        return None
    pos, neg = float(np.mean(d > SIGN_TOL)), float(np.mean(d < -SIGN_TOL))
    idx = np.random.default_rng(rng_key).integers(0, d.size, (v6.N_BOOT, d.size))
    return {"persons": int(d.size), "share_positive": pos, "share_negative": neg, "share_zero": 1.0 - pos - neg,
            "share_of_the_more_frequent_sign": max(pos, neg), "median": float(np.median(d)),
            "median_ci": boot_median(d, idx)}


def signed_set(pairs: list[tuple[dict, dict]], ds: int, task: int, row_ix: int) -> dict:
    """pairs: per person (from, to) whole-record vectors; the change to - from per stage and feature."""
    out: dict = {}
    for st in SIGNED_STAGES:
        ok = [(a[st], b[st]) for a, b in pairs if a[st] is not None and b[st] is not None]
        out[st] = {}
        if not ok:
            continue
        d = np.array([np.asarray(b, dtype=float) - np.asarray(a, dtype=float) for a, b in ok])
        for f in range(d.shape[1]):
            out[st][str(f)] = signed(d[:, f], [SEED, 97, ds, task, STAGE[base_of(st)[0]], row_ix, f])
    return out


def signed_rows(b: dict, ds: str, task: str) -> dict:
    persons = list(b["persons"].values())
    return {row: signed_set([(p["whole"]["text/before"], p["whole"][row]) for p in persons], k6.DATASET[ds],
                            k6.TASK[task], ROW_IX[row]) for row in SIGNED_ROWS}


# ================================================================ checks on v6k's persons


def check_v6k_persons(series: dict, ids: list, ds: str, task: str, place: dict, m: dict, rl: dict) -> dict:
    """The rule persons' counts (stage, R, R_loc) in the 12 main cells and the labels under gates R and R_loc equal
    v6l's record (v6l's keys)."""
    rows = ["text/before", *MAIN_ROWS]
    hv = {how: {row: [] for row in rows} for how in v6b.BASELINE}
    for pid in ids:
        s, kinds = series[pid][task] if ds == "kupa" else series[pid]
        if not k6.count1_ok(s):
            continue
        vec, _, _, _ = person_rows(s, kinds, [k6.DATASET[ds], k6.TASK[task], place[pid]], m, rows)
        for how in v6b.BASELINE:
            for row in rows:
                hv[how][row].append(vec[row][how])
    rp = l6.rule_persons(hv, ds, task)
    nm = f"{ds}_{task}"
    fields = ("persons", "share_vs_own", "share_vs_own_ci", "left_out_no_vector")
    for how in v6b.BASELINE:
        for row in MAIN_ROWS:
            for k in RULE_STAGES:
                for part in ("stage", "R", "R_loc"):
                    mine, theirs = rp[how][row][k][part], rl["rule_persons"][nm][how][row][k][part]
                    for f in fields:
                        if mine.get(f) != theirs.get(f):
                            raise SystemExit(f"v6k's persons {nm} {how} {row} {k} {part}: {f} differs from v6l's")
    return rp


def check_labels_v6l(rp_k: dict, rl: dict) -> None:
    for g in ("R", "R_loc"):
        part = l6.evaluate_rule(rp_k, g)
        for k in RULE_STAGES:
            if part[k]["label"] != rl["rule"]["part_1_by_gate"][g][k]["label"]:
                raise SystemExit(f"v6k's persons, gate {g}, {k}: the label differs from v6l's")


# ================================================================ predictions and main cells


NE = "not evaluable"


def predictions(rec: dict) -> dict:
    p0 = rec["part_0"]["by_gate"]["R"]
    lab = {k: p0[k]["label"] for k in RULE_STAGES}
    p1 = NE if NE in lab.values() else (
        lab["U@2000"] == lab["B@2000"] == "does not show them beyond it" and lab["I"] == "mixed directions"
        and lab["Bsd@2000"] in ("not beyond, partly undecided", "does not show them beyond it"))
    c = rec["part_1"]
    loc = [size_class(cell["R_loc"]) for cell in c["contrast"]["labelled"]["U@2000"].values()]
    p2 = NE if len(loc) != 4 or "no count" in loc else all(x == "above" for x in loc)
    shows = "shows these operations beyond the smallest difference"
    l3 = (c["rule"]["U@2000"]["label"], c["rule"]["I"]["label"])
    p3 = NE if NE in l3 else l3 == (shows, shows)
    b = [size_class(cell["stage"]) for cell in c["contrast"]["labelled"]["B@2000"].values()]
    p4 = NE if "no count" in b else sum(x == "above" for x in b) >= 2
    rp = rec["rule_persons"]
    lw = [size_class(rp["lwp_lwp"][how]["inside_100"]["U@2000"]["R_loc"]) for how in v6b.BASELINE]
    lw25 = [size_class(rp["lwp_lwp"][how]["inside_25"]["U@2000"]["R_loc"]) for how in v6b.BASELINE]
    p5 = NE if "no count" in lw + lw25 else (all(x == "above" for x in lw) and not any(x == "above" for x in lw25))
    cr = {nm: [size_class(rp[f"{ds}_{task}"][how]["cross_100"][k][part]) for ds, task in MAIN
               for how in v6b.BASELINE]
          for nm, k, part in (("U@2000", "U@2000", "stage"), ("I", "I", "stage"), ("R_loc", "U@2000", "R_loc"))}
    allc = cr["U@2000"] + cr["I"] + cr["R_loc"]
    p6 = NE if "no count" in allc else (all(x == "above" for x in cr["R_loc"]) and all(x == "above" for x in cr["I"])
                                        and sum(x == "above" for x in cr["U@2000"]) >= 2)
    um = [size_class(rec["count_1"][f"{ds}_{task}"][how][row]["U_med"]) for ds, task in MAIN for how in v6b.BASELINE
          for row in ("shuffled/before", "shuffled_in_halves/before")]
    p7 = NE if "no count" in um else not any(x == "above" for x in um)
    s = rec["part_4"]["kupa_essay_minus_copy"]["U@2000"].get("0")
    p8 = NE if s is None else (s["median"] > 0 and s["median_ci"][0] > 0)
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6, "P7": p7, "P8": p8}


def main_cells_out(rec: dict) -> dict:
    def sh(res):
        return [round(res["share_vs_own"], 3), [round(x, 3) for x in res["share_vs_own_ci"]]] \
            if "share_vs_own_ci" in res else res

    out: dict = {}
    for g in ("R", "R_loc"):
        for k in RULE_STAGES:
            out[f"part 0 | gate {g} | {k}"] = rec["part_0"]["by_gate"][g][k]["label"]
    for k in CONTRAST_LABELLED:
        out[f"part 1 | {k}"] = rec["part_1"]["rule"][k]["label"]
        for where, cell in rec["part_1"]["contrast"]["labelled"][k].items():
            out[f"part 1 | {k} | {where}"] = {"stage": sh(cell["stage"]), "R_loc": sh(cell["R_loc"])}
    for k in (*RULE_STAGES, "R_loc"):
        kk = "U@2000" if k == "R_loc" else k
        part = "R_loc" if k == "R_loc" else "stage"
        for ds, task in MAIN:
            for how in v6b.BASELINE:
                for fam in FAMILIES:
                    out[f"part 2 | {k} | {ds}_{task} | {how} | {fam}"] = {
                        s: [size_class(rec["rule_persons"][f"{ds}_{task}"][how][f"{fam}_{s}"][kk][part]),
                            sh(rec["rule_persons"][f"{ds}_{task}"][how][f"{fam}_{s}"][kk][part])] for s, _ in STEPS}
    return out


def beside_u_med(rp: dict, gate: str) -> dict:
    """Part 0's statuses of U_med in the 12 main cells under a gate (U's cells fixed by construction), no label."""
    status, read = {}, {}
    for ds, task in MAIN:
        nm = f"{ds}_{task}"
        for how in v6b.BASELINE:
            for row in MAIN_ROWS:
                c = rp[nm][how][row]["U_med"]
                where = f"{nm} | {how} | {row}"
                if (row, how) in k6.BY_CONSTRUCTION["U"]:
                    status[where] = "fixed by construction"
                elif "no count" in (size_class(c[gate]), size_class(c["stage"])):
                    status[where] = "fewer than 2 rule persons"
                elif size_class(c[gate]) != "above":
                    status[where] = "control not met"
                else:
                    read[where] = size_class(c["stage"])
                    status[where] = f"read: {read[where]}"
    return {"label": "beside the rule: no label", "cells_read": len(read), "status": status, "read": read}


def gate_cells_u_med(rp: dict) -> dict:
    out = {}
    for ds, task in MAIN:
        for how in v6b.BASELINE:
            for row in MAIN_ROWS:
                c = rp[f"{ds}_{task}"][how][row]["U_med"]
                cls = {g: size_class(c[g]) for g in GATES}
                out[f"{ds}_{task} | {how} | {row}"] = {**cls, "kind": l6.cell_kind(cls["R"], cls["R_loc"],
                                                                                   cls["R_lag"])}
    return out


def count_new(n_series: dict, n_ids: list, kplace: dict, ln_series: dict, ln_ids: list, lplace: dict, m: dict,
              rk: dict, rl: dict) -> dict:
    """Parts 0-4 on the new persons, and the predictions."""
    rec: dict = {"checks": {}}
    # the new persons
    rec["count_1"], rec["rule_persons"], rec["tallies"], rec["part_4"] = {}, {}, {}, {}
    builds = {}
    for ds, task, series, ids, place, rows in (("kupa", "essay", n_series, n_ids, kplace, ROWS),
                                               ("kupa", "copy", n_series, n_ids, kplace, ["text/before"]),
                                               ("lwp", "lwp", ln_series, ln_ids, lplace, ROWS)):
        nm = f"{ds}_{task}"
        b = build(series, ids, ds, task, place, m, rows)
        builds[nm] = b
        rec["tallies"][nm] = {"count_persons": len(b["persons"]), "left_out_fewer_than_8_or_part_without_gap":
                              b["left_out"], "sentences": l6.five_or_none(b["sentences"]),
                              "doses": {row: dose_tally_summary(t) for row, t in b["dose_tallies"].items()}}
        if task == "copy":
            continue
        hv = hv_of(b, rows)
        rec["count_1"][nm] = count1(hv, ds, task)
        rec["rule_persons"][nm] = rule_persons(hv, ds, task, (*MAIN_ROWS, *DOSE_ROWS))
        rec["part_4"][nm] = signed_rows(b, ds, task)
        del hv
    rec["checks"]["rows"] = ("every row is a permutation of the text's gaps; the doses are nested; inside_100 equals "
                             "inside/before for every count person")
    rec["checks"]["construction"] = "every identity held for every count person"

    # part 0
    by_gate = {g: l6.evaluate_rule(rec["rule_persons"], g) for g in ("R", "R_loc")}
    for g in ("R", "R_loc"):
        by_gate[g]["U_med"] = beside_u_med(rec["rule_persons"], g)
    rec["part_0"] = {"by_gate": by_gate, "labels_beside_earlier": {
        k: {"v6k_gate_R": rk["rule"]["part_1"][k]["label"], "v6l_gate_R_loc": rl["rule"]["part_1_by_gate"]["R_loc"][k][
            "label"], "new_gate_R": by_gate["R"][k]["label"], "new_gate_R_loc": by_gate["R_loc"][k]["label"]}
        for k in RULE_STAGES},
        "gate_cells": {**l6.gate_cells(rec["rule_persons"]), "U_med": gate_cells_u_med(rec["rule_persons"])}}

    # part 1
    ess, cop = builds["kupa_essay"]["persons"], builds["kupa_copy"]["persons"]
    both = [pid for pid in n_ids if pid in ess and pid in cop]
    pairs = [(cop[pid]["vec"]["text/before"], ess[pid]["vec"]["text/before"]) for pid in both]
    cc = contrast(pairs)
    rec["part_1"] = {"contrast_persons": len(both), "contrast": cc, "rule": contrast_rule(cc)}
    rec["part_4"]["kupa_essay_minus_copy"] = signed_set(
        [(cop[pid]["whole"]["text/before"], ess[pid]["whole"]["text/before"]) for pid in both],
        k6.DATASET["kupa"], k6.TASK["essay"], CONTRAST_ROW_IX)
    del builds, ess, cop, pairs

    # parts 2 and 3
    rec["part_2"] = doses(rec["rule_persons"])
    rec["part_3"] = {"U_med_vs_U_main_cells": {
        f"{ds}_{task} | {how} | {row}": {
            "U@2000": size_class(rec["rule_persons"][f"{ds}_{task}"][how][row]["U@2000"]["stage"]),
            "U_med": size_class(rec["rule_persons"][f"{ds}_{task}"][how][row]["U_med"]["stage"]),
            "fixed_by_construction": (row, how) in k6.BY_CONSTRUCTION["U"]}
        for ds, task in MAIN for how in v6b.BASELINE for row in MAIN_ROWS}}

    rec["predictions"] = predictions(rec)
    return rec


# ================================================================ run


def draw_checked(order: list, elig, looked_ref: int, start: int, ids_sha_ref: str, what: str) -> tuple[list, dict]:
    ids, series, looked = k6.draw(order, elig, start, None, v6.N)
    if k6.ids_sha(ids) != ids_sha_ref or looked != looked_ref:
        raise SystemExit(f"not {what}")
    return ids, series


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
    recs = {k: json.loads((ROOT / f"results/prototype_{k}/record.json").read_text()) for k in ("v6", "v6b", "v6e")}
    rk, rl = json.loads(REC_K.read_text()), json.loads(REC_L.read_text())
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    m = c4["mapping"]
    rec: dict = {"persons": {}, "checks": {}}

    # KUPA: the earlier persons, then the new ones
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6b.read_kupa_b(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    kplace = {pid: i for i, pid in enumerate(ids_sorted)}
    order = v6.draw_order(ids_sorted, 0)
    elig = lambda pid: k6.kupa_eligible(kupa, pid)  # noqa: E731
    v6_ids, _, _ = k6.draw(order, elig, 0, k6.V6_LOOKED, None)
    v6b_ids, _, _ = k6.draw(order, elig, k6.V6_LOOKED, KUPA_V6K[0], None)
    if k6.ids_sha(v6_ids) != recs["v6"]["datasets"]["kupa"]["ids_sha256"] \
            or k6.ids_sha(v6b_ids) != recs["v6b"]["persons"]["ids_sha256"]:
        raise SystemExit("positions 0-436 do not give v6's and v6b's persons")
    kk_ids, kk_series = draw_checked(order, elig, rk["persons"]["kupa"]["ids_looked_at_from_437"], KUPA_V6K[0],
                                     rk["persons"]["kupa"]["ids_sha256"], "v6k's KUPA persons")
    if KUPA_V6K[0] + rk["persons"]["kupa"]["ids_looked_at_from_437"] != SKIP["kupa"]:
        raise SystemExit("v6k's KUPA persons do not end at position 655")
    n_ids, n_series, n_looked = k6.draw(order, elig, SKIP["kupa"], None, v6.N)
    if len(n_ids) < v6.N or set(n_ids) & (set(v6_ids) | set(v6b_ids) | set(kk_ids)):
        raise SystemExit("not 200 new eligible KUPA persons")
    rec["persons"]["kupa"] = {"ids_looked_at_from_656": n_looked, "ids_in_the_order": len(order),
                              "ids_sha256": k6.ids_sha(n_ids), "overlap": 0}
    del kupa

    # LWP
    lwp, lids, tally = v6e.read_lwp(ROOT / d_lwp["file"], None)
    lplace = {pid: i for i, pid in enumerate(sorted(lids))}
    lorder = v6.draw_order(sorted(lids), 2)
    lelig = lambda pid: k6.lwp_eligible(lwp, pid)  # noqa: E731
    e_ids, _, _ = k6.draw(lorder, lelig, 0, LWP_V6K[0], None)
    if k6.ids_sha(e_ids) != recs["v6e"]["persons"]["ids_sha256"]:
        raise SystemExit("positions 0-203 do not give v6e's persons")
    lk_ids, lk_series = draw_checked(lorder, lelig, rk["persons"]["lwp"]["ids_looked_at_from_204"], LWP_V6K[0],
                                     rk["persons"]["lwp"]["ids_sha256"], "v6k's LWP persons")
    if LWP_V6K[0] + rk["persons"]["lwp"]["ids_looked_at_from_204"] != SKIP["lwp"]:
        raise SystemExit("v6k's LWP persons do not end at position 410")
    ln_ids, ln_series, ln_looked = k6.draw(lorder, lelig, SKIP["lwp"], None, v6.N)
    if len(ln_ids) < v6.N or set(ln_ids) & (set(e_ids) | set(lk_ids)):
        raise SystemExit("not 200 new eligible LWP persons")
    rec["persons"]["lwp"] = {"ids_looked_at_from_411": ln_looked, "ids_in_the_order": len(lorder),
                             "ids_sha256": k6.ids_sha(ln_ids), "overlap": 0, "tally_of_reading": tally}
    del lwp

    # reuse: v6k's persons against v6l's record
    rp_k = {}
    for ds, task, series, ids, place in (("kupa", "essay", kk_series, kk_ids, kplace),
                                         ("lwp", "lwp", lk_series, lk_ids, lplace)):
        rp_k[f"{ds}_{task}"] = check_v6k_persons(series, ids, ds, task, place, m, rl)
    check_labels_v6l(rp_k, rl)
    rec["checks"]["reuse"] = ("on v6k's persons, the rule persons' counts of U, B, Bsd and I with R and R_loc in the 12 "
                              "main cells, and the labels under gates R and R_loc, equal v6l's record")
    del kk_series, lk_series, rp_k

    new = count_new(n_series, n_ids, kplace, ln_series, ln_ids, lplace, m, rk, rl)
    rec["checks"].update(new.pop("checks"))
    rec.update(new)
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
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6m-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "paired_indices": "the paired difference of a stage and R_loc uses the stage's bootstrap indices (its key with "
                      "the rule gate) for both",
    "contrast_rule_persons": "a labelled stage's contrast cell, its R_loc, R and R_lag are counted on the persons with "
                             "the stage and R_loc in all four parts (R_loc exists exactly when R does)",
    "contrast_beside_persons": "the beside stages of part 1 are counted on their own persons with gate 0; the "
                               "within-scale and all-pairs counts use the cell's persons and indices",
    "dose_tallies": "the time moved by an inside swap is |difference of the two gaps|; a crossing swap is one whose two "
                    "gaps lie in sentences of different parts; 'same as next step' is counted at 25 and 50",
    "whole_record": "the whole-record vector of a row is the part vector of all its sentences (counting 'before'); "
                    "text/after and text/none have none",
    "copy_rows": "KUPA copy is built with text/before only (part 1 uses its text vectors)",
    "rule_rows": "rule persons are counted in the main rows and the dose rows for U@2000, B@2000, Bsd@2000, I and "
                 "U_med, B_complete@2000, Bsd_complete@2000",
    "part_0_labels": "part 0 uses v6l's evaluate_rule on the new persons' rule persons (its beside stages "
                     "Bsd_complete and B_complete are reported there without a label); U_med's statuses and cell kinds "
                     "are added beside under both gates, with U's cells fixed by construction",
    "small_counts": "a count over fewer than 5 persons keeps its share and interval but not its medians; the median "
                    "ratio needs 5 persons with a base above 0",
    "contrast_cells": "every part 1 stage is counted on its own contrast persons (gate 0, own_persons) and, except "
                      "R_loc, on its contrast rule persons with R_loc, R, R_lag and the paired difference",
    "gate_persons_in_predictions": "P2, P5 and P6 read R_loc on U@2000's (contrast) rule persons; on another stage's "
                                   "persons R_loc's class can differ (reported in the cells)",
    "signed_zero": "a person's change counts as zero if |change| <= 1e-9 in the feature's units",
    "predictions": "P2 reads R_loc on U@2000's contrast rule persons; P5 and P6 read R_loc on U@2000's rule persons; "
                   "P7 uses gate 0; a prediction whose input has no count is 'not evaluable'",
    "check_v6k": "the reuse check builds v6k's persons with the main rows only and compares with v6l's record, which "
                 "reproduces v6k's",
}


if __name__ == "__main__":
    main()
