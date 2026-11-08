"""Prototype v6l: on v6k's persons, (a1) the split-half reliability of each summary's features over persons by
permutation-based splits (Parsons et al. 2019); (a2) the positive control R split into its location part (R_loc) and its
order part (R_lag), with v6k's rule cells read again under each as the gate; (b1) the mean length of the P-bursts
bounded by long gaps at both ends (B_complete); (b2) the writer's threshold fitted after cutting inactive periods at the
start and the end of the record (decision 9 of .steering/20261108-paper-a-jowr-reframe; decisions 1-3 of
.steering/20261108-v6l-reliability-and-r-split).

Usage:  uv run python works/prototype_v6l.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v6l.json (committed before this script is first run), v6k's conditions and
            record, and the conditions and records named in v6k's (checked by SHA-256)
Input:  KUPA-KEYS TASK-1 and TASK-2 and LWP (external/; not in git, checked by SHA-256; LWP not redistributed)
Output: results/prototype_v6l/record.json (written to a temporary directory beside it and renamed when complete). No
        press time, gap, key, text, per-person vector, per-person value, per-person threshold or id is written.
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

v6, v6b, v6d, v6e, v6i = k6.v6, k6.v6b, k6.v6d, k6.v6e, k6.v6i

COND = ROOT / "works" / "conditions" / "prototype_v6l.json"
COND_K = ROOT / "works" / "conditions" / "prototype_v6k.json"
REC_K = ROOT / "results" / "prototype_v6k" / "record.json"
OUT_DIR = ROOT / "results" / "prototype_v6l"
CODE = ["works/prototype_v6l.py", *k6.CODE]
SEED = k6.SEED

STAGE = {**k6.STAGE, "R_loc": 8, "R_lag": 9, "B_complete": 10}
TRIM_MS = {"120": 120000.0, "30": 30000.0}
TRIM_TH = ("trim120", "trim120_fixed", "trim30", "trim30_fixed")
TH_IX = {"2000": 0, "trim120": 5, "trim120_fixed": 6, "trim30": 7, "trim30_fixed": 8}
TH_STAGES = k6.TH_STAGES  # U, B, Bsd
ROWS, ROW_IX, COUNTING = k6.ROWS, k6.ROW_IX, k6.COUNTING
MAIN, MAIN_ROWS = k6.MAIN, k6.MAIN_ROWS
RULE_STAGES = k6.RULE_STAGES  # U@2000, B@2000, Bsd@2000, I
BESIDE = ("Bsd_complete@2000", "B_complete@2000")
GATES = ("R", "R_loc", "R_lag")
R_STAGES = GATES
TRIM_WINDOW, TRIM_MIN_PRESSES = 10, 11
N_SPLITS, REL_MIN_PERSONS, LOW_RSB = 5000, 10, 0.50
MIN_SUMMARY = 5
REL_STAGES = ("S0", "U@2000", "B@2000", "Bsd@2000", "B_complete@2000", "Bsd_complete@2000", "I", "R")
REL_DIMS = {"S0": 7, "U@2000": 4, "B@2000": 1, "Bsd@2000": 1, "B_complete@2000": 1, "Bsd_complete@2000": 1, "I": 3,
            "R": 7}
REUSE_GATE0 = ("S0", "S1", "I", "R", "U@2000", "B@2000", "Bsd@2000", "Bsd_complete@2000")
SIZE = k6.SIZE
sk, base_of, size_class, label = k6.sk, k6.base_of, k6.size_class, k6.label


def all_keys() -> list[str]:
    return ["S0", "S1", "I", *R_STAGES, *[sk(st, th) for st in TH_STAGES for th in ("2000", *TRIM_TH)],
            "Bsd_complete@2000", "B_complete@2000"]


# ================================================================ the trim


def trim_range(gaps: np.ndarray, T: float) -> dict:
    """The conditions' trim: kept gap indices [a, b), the steps at each end, or consumed."""
    g = np.asarray(gaps, dtype=float)
    a, b = 0, g.size
    steps = {"start": 0, "end": 0}
    if b + 1 < TRIM_MIN_PRESSES:
        return {"consumed": True, "a": a, "b": b, "steps": steps}
    for end in ("start", "end"):
        while True:
            w = min(TRIM_WINDOW, b - a)
            lo = a if end == "start" else b - w
            long_at = np.flatnonzero(g[lo:lo + w] >= T - ks.TOL_MS)
            if long_at.size == 0:
                break
            if end == "start":
                a = lo + int(long_at[-1]) + 1
            else:
                b = lo + int(long_at[0])
            steps[end] += 1
            if b - a + 1 < TRIM_MIN_PRESSES:
                return {"consumed": True, "a": a, "b": b, "steps": steps}
    return {"consumed": False, "a": a, "b": b, "steps": steps}


def trimmed_fit(gaps: np.ndarray, word: np.ndarray, T: float, key: list[int]) -> tuple[dict | None, dict]:
    """The trim at T and v6k's mixture on the 'word' gaps kept (v6k's key); (fit or None, trim info)."""
    tr = trim_range(gaps, T)
    a, b = tr["a"], tr["b"]
    tr["word_dropped"] = int(word.sum() - word[a:b].sum())
    tr["presses_dropped"] = int(gaps.size - (b - a))
    tr["time_dropped_share"] = float((gaps.sum() - gaps[a:b].sum()) / gaps.sum())
    if tr["consumed"]:
        return None, tr
    return k6.fit_writer(np.log(gaps[a:b][word[a:b]]), key), tr


# ================================================================ part vectors


def add_new(out: dict, tally: dict, rec: k6.Rec, idx: np.ndarray) -> None:
    """R_loc, R_lag and B_complete@2000 beside v6k's part vectors; complete-burst tallies."""
    r = out["R"]
    out["R_loc"] = None if r is None else r[:6].copy()
    out["R_lag"] = None if r is None else r[6:7].copy()
    held = rec.held_of(idx)
    lengths, complete, _ = k6.bursts(rec.gaps, held, 2000.0)
    out["B_complete@2000"] = np.array([float(complete.mean())]) if complete.size >= 1 else None
    tally["complete_bursts"] = int(complete.size)
    tally["complete_share_bursts"] = complete.size / lengths.size
    tally["complete_share_presses"] = float(complete.sum() / lengths.sum())


def part_vectors_l(rec: k6.Rec, idx: np.ndarray, th_values: dict) -> tuple[dict, dict]:
    out, tally = k6.part_vectors(rec, idx, th_values)
    add_new(out, tally, rec, idx)
    return out, tally


class Fast:
    """The text record's part vectors at 2 s for any set of sentences (the reliability's splits): the same values as
    v6k's part_vectors (checked per person on the fixed parts and on the first splits)."""

    def __init__(self, rec: k6.Rec):
        self.rec = rec
        g = rec.gaps
        self.g, self.lg = g, np.log(g)
        self.sid = np.full(g.size, -1)
        for i, (lo, hi) in enumerate(rec.ranges):
            self.sid[lo:hi] = i
        lab = rec.lab
        self.is_lab = {k: lab == k for k in ("sentence", "subsentence", "word", "within")}
        self.ss = np.isin(lab, ["sentence", "subsentence"])
        self.rest = g >= 2000.0 - 1e-6
        self.over = g > k6.I_MS + ks.TOL_MS
        self.sp, self.ur = rec.u_tiles(2000.0)

    def vectors(self, idx: np.ndarray) -> dict:
        idx = np.sort(np.asarray(idx))
        sel = np.zeros(self.rec.n, dtype=bool)
        sel[idx] = True
        mask = np.zeros(self.g.size, dtype=bool)
        ok = self.sid >= 0
        mask[ok] = sel[self.sid[ok]]
        held = np.flatnonzero(mask)
        if held.size == 0:
            raise ValueError("the part holds no gap")
        g = self.g[held]
        out: dict = {"S0": np.quantile(np.log(g), v6.QUANTILES, method="linear")}
        meds = []
        for k in ("sentence", "subsentence", "word", "within"):
            x = g[self.is_lab[k][held]]
            if x.size == 0:
                meds = None
                break
            meds.append(float(np.median(np.log(x))))
        if meds is None:
            out["R"] = None
        else:
            rest = self.rest[held]
            share = float(g[rest & self.ss[held]].sum() / g[rest].sum()) if rest.any() else 0.0
            pm = mask[:-1] & mask[1:]
            a, b = self.lg[:-1][pm], self.lg[1:][pm]
            if a.size >= 3:
                ac = 0.0 if np.ptp(a) == 0 or np.ptp(b) == 0 else float(np.corrcoef(a, b)[0, 1])
            else:
                ac = 0.0
            out["R"] = np.array([*meds, meds[0] - meds[3], share, ac])
        lab_held = self.rec.lab[held]
        over = self.over[held]
        A, wi, wo = g[over], g[over & (lab_held == "within")], g[over & (lab_held == "word")]
        out["I"] = None if min(A.size, wi.size, wo.size) == 0 else np.array([A.mean(), wi.mean(), wo.mean()]) / 1000.0
        s = self.sp[idx]
        s = s[~np.isnan(s)]
        out["U@2000"] = None if s.size == 0 else np.concatenate([v6.med_iqr(s), v6.med_iqr(self.ur[idx])])
        lengths, complete, _ = k6.bursts(self.g, held, 2000.0)
        out["B@2000"] = np.array([float(lengths.mean())])
        out["Bsd@2000"] = np.array([float(lengths.std())])
        out["Bsd_complete@2000"] = np.array([float(complete.std())]) if complete.size >= 2 else None
        out["B_complete@2000"] = np.array([float(complete.mean())]) if complete.size >= 1 else None
        return out


def same_vectors(a: dict, b: dict, keys: tuple[str, ...]) -> bool:
    for k in keys:
        x, y = a[k], b[k]
        if (x is None) != (y is None):
            return False
        if x is not None and not np.array_equal(np.asarray(x, dtype=float), np.asarray(y, dtype=float)):
            return False
    return True


# ================================================================ persons and rows


def person_rows(s: v6.Series, kinds: list[str], key: list[int], m: dict, rows: list[str]
                ) -> tuple[dict, dict, dict]:
    """Per row and baseline the (A, B) vectors; the fits (untrimmed text, trimmed per T and row); the tallies."""
    lab = v6b.labels(s, kinds)
    word = lab == "word"
    n = len(s.ranges("before"))
    out, tallies = {}, {}
    fits: dict = {"text_untrimmed": None, "trim": {T: {} for T in TRIM_MS}}
    text: dict = {}
    for row in rows:
        t = k6.row_times(s, row, key)
        rec = k6.Rec(t, s.ranges(COUNTING.get(row, "before")), lab, m)
        if rec.n != n:
            raise SystemExit(f"{row}: {rec.n} sentences, not {n}")
        th_values: dict = {"2000": 2000.0}
        for T, T_ms in TRIM_MS.items():
            if row.startswith("text/"):
                if T not in text:
                    text[T] = trimmed_fit(rec.gaps, word, T_ms, [*key, ROW_IX["text/before"]])
                fit = text[T]
            else:
                fit = trimmed_fit(rec.gaps, word, T_ms, [*key, ROW_IX[row]])
            fits["trim"][T][row] = fit
            th_values[f"trim{T}"] = fit[0]["theta"] if fit[0] is not None else None
            th_values[f"trim{T}_fixed"] = text[T][0]["theta"] if text[T][0] is not None else None
        if row == "text/before":
            fits["text_untrimmed"] = k6.fit_writer(np.log(rec.gaps[word]), [*key, ROW_IX["text/before"]])
        out[row], tallies[row] = {}, {}
        for how in v6b.BASELINE:
            A, B = v6b.parts(n, how)
            va, ta = part_vectors_l(rec, A, th_values)
            vb, tb = part_vectors_l(rec, B, th_values)
            out[row][how] = (va, vb)
            tallies[row][how] = (ta, tb)
    return out, fits, tallies


def check_construction(vec: dict, tallies: dict) -> None:
    """v6k's identities at 2000 and at the text's trimmed thresholds; R_loc and R_lag are R's numbers."""
    for row in vec:
        for how in v6b.BASELINE:
            for h in (0, 1):
                v = vec[row][how][h]
                if v["R"] is None:
                    if v["R_loc"] is not None or v["R_lag"] is not None:
                        raise SystemExit(f"{row} {how}: R_loc or R_lag without R")
                elif not (np.array_equal(v["R_loc"], v["R"][:6]) and np.array_equal(v["R_lag"], v["R"][6:])):
                    raise SystemExit(f"{row} {how}: R_loc or R_lag differs from R's numbers")
    for row, how in k6.KEEPS_PART_MULTISET:
        if row not in vec:
            continue
        for h in (0, 1):
            a, b = vec["text/before"][how][h], vec[row][how][h]
            ths = ("2000", "trim120_fixed", "trim30_fixed")
            pairs = {sk("B", th): (a[sk("B", th)], b[sk("B", th)]) for th in ths}
            pairs["I0"] = (tallies["text/before"][how][h]["I0"], tallies[row][how][h]["I0"])
            if row == "inside/before":
                pairs.update({sk("U", th): (a[sk("U", th)], b[sk("U", th)]) for th in ths})
                pairs["S1"] = (a["S1"], b["S1"])
            for nm, (x, y) in pairs.items():
                if x is None and y is None:
                    continue
                if (x is None) != (y is None):
                    raise SystemExit(f"{row} {how} {nm}: a value exists on one side only")
                if np.max(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))) > k6.CONSTR_TOL:
                    raise SystemExit(f"{row} {how} {nm}: changed where it cannot by construction")


def build(series: dict, ids: list, ds: str, task: str, place: dict, m: dict) -> tuple[dict, dict, list]:
    """hv[how][row] over count persons, the diagnostics, and the text records (for the reliability)."""
    hv = {row: {how: [] for how in v6b.BASELINE} for row in ROWS}
    fits_text, fits_trim = [], {T: [] for T in TRIM_MS}
    tr_tally = {T: {"trimmed_start": 0, "trimmed_end": 0, "untrimmed": 0, "consumed": 0,
                    "no_theta_t_fewer_than_2_word_gaps": 0,
                    "steps_start": [], "steps_end": [], "presses_dropped": [], "time_dropped_share": [],
                    "word_dropped": []} for T in TRIM_MS}
    diff = {T: [] for T in TRIM_MS}
    dev = {T: {row: [] for row in ROWS if not row.startswith("text/")} for T in TRIM_MS}
    no_fit_row = {T: {row: 0 for row in ROWS} for T in TRIM_MS}
    comp = {how: {"bursts": [], "share_bursts": [], "share_presses": [], "edge_share_of_B": [],
                  "parts_without_a_complete_burst": 0} for how in v6b.BASELINE}
    left, texts = 0, []
    for pid in ids:
        s, kinds = series[pid][task] if ds == "kupa" else series[pid]
        if not k6.count1_ok(s):
            left += 1
            continue
        key = [k6.DATASET[ds], k6.TASK[task], place[pid]]
        vec, fits, tallies = person_rows(s, kinds, key, m, ROWS)
        check_construction(vec, tallies)
        for row in ROWS:
            for how in v6b.BASELINE:
                hv[row][how].append(vec[row][how])
        tw = fits["text_untrimmed"]
        if tw is not None:
            fits_text.append(tw)
        for T in TRIM_MS:
            fit, tr = fits["trim"][T]["text/before"]
            t = tr_tally[T]
            t["steps_start"].append(tr["steps"]["start"])
            t["steps_end"].append(tr["steps"]["end"])
            untrimmed = tr["steps"]["start"] == 0 and tr["steps"]["end"] == 0 and not tr["consumed"]
            t["consumed"] += int(tr["consumed"])
            t["trimmed_start"] += int(tr["steps"]["start"] > 0)
            t["trimmed_end"] += int(tr["steps"]["end"] > 0)
            t["untrimmed"] += int(untrimmed)
            for k in ("presses_dropped", "time_dropped_share", "word_dropped"):
                t[k].append(tr[k])
            if untrimmed and fit != tw:
                raise SystemExit(f"{ds} {task}: an untrimmed record does not give v6k's theta_w")
            if fit is None:
                t["no_theta_t_fewer_than_2_word_gaps"] += int(not tr["consumed"])
            else:
                fits_trim[T].append(fit)
                if tw is not None:
                    diff[T].append(math.log(fit["theta"]) - math.log(tw["theta"]))
                for row in dev[T]:
                    f = fits["trim"][T][row][0]
                    if f is not None:
                        dev[T][row].append(abs(math.log(f["theta"]) - math.log(fit["theta"])))
            for row in ROWS:
                if fits["trim"][T][row][0] is None:
                    no_fit_row[T][row] += 1
        for how in v6b.BASELINE:
            for t in tallies["text/before"][how]:
                comp[how]["bursts"].append(t["complete_bursts"])
                comp[how]["share_bursts"].append(t["complete_share_bursts"])
                comp[how]["share_presses"].append(t["complete_share_presses"])
                comp[how]["edge_share_of_B"].append(t["edge_share_2000"])
                comp[how]["parts_without_a_complete_burst"] += int(t["complete_bursts"] == 0)
        lab = v6b.labels(s, kinds)
        texts.append((k6.Rec(s.times, s.ranges("before"), lab, m), place[pid], vec["text/before"]))
    trim_out = {}
    for T in TRIM_MS:
        t = tr_tally[T]
        trim_out[T] = {**{k: t[k] for k in ("trimmed_start", "trimmed_end", "untrimmed", "consumed",
                                            "no_theta_t_fewer_than_2_word_gaps")},
                       **{k: five_or_none(t[k])
                          for k in ("steps_start", "steps_end", "presses_dropped", "time_dropped_share",
                                    "word_dropped")},
                       "theta_t_text": k6.writer_summary(fits_trim[T]) if len(fits_trim[T]) >= MIN_SUMMARY else None,
                       "ln_theta_t_minus_ln_theta_w_text": five_or_none(diff[T]),
                       "difference_exactly_0": int(sum(d == 0.0 for d in diff[T])),
                       "abs_ln_theta_t_row_minus_text": {row: five_or_none(v) for row, v in dev[T].items()},
                       "records_without_theta_t": no_fit_row[T]}
    diag = {"count_persons": len(texts), "left_out_fewer_than_8_or_part_without_gap": left,
            "theta_w_text_untrimmed": k6.writer_summary(fits_text), "trim": trim_out,
            "complete_bursts_2000_text_parts": {how: {k: (v if isinstance(v, int) else five_or_none(v))
                                                      for k, v in d.items()} for how, d in comp.items()}}
    return {how: {row: hv[row][how] for row in ROWS} for how in v6b.BASELINE}, diag, texts


def five_or_none(values: list) -> dict | None:
    """v6d.five over at least MIN_SUMMARY values, else None (a summary of a few records would show their values)."""
    return v6d.five(values) if len(values) >= MIN_SUMMARY else None


# ================================================================ count 1


def key1(ds: str, task: str, key_name: str, row: str, how: str, gate: int) -> list[int]:
    st, th = base_of(key_name)
    return [SEED, 90, k6.DATASET[ds], k6.TASK[task], STAGE[st], ROW_IX[row], v6b.BASELINE[how], TH_IX[th] if th else 0,
            gate]


def one_cell(hv: dict, row: str, how: str, key_name: str, rng_key: list[int], extra: str | None = None) -> dict:
    """v6k's one_cell; R_loc and R_lag, like R, are not defined under text/none."""
    if base_of(key_name)[0] in R_STAGES and row == "text/none":
        first, sel = hv[how]["text/before"], hv[how][row]
        if any(v6b.has(sel[i], key_name) for i in range(len(sel))):
            raise SystemExit(f"text/none has {key_name} for some persons")
        del first
        return {"defined": False, "persons": 0}
    return k6.one_cell(hv, row, how, key_name, rng_key, extra)


def count1(hv: dict, ds: str, task: str) -> dict:
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for row in ROWS[1:]:
            out[how][row] = {k: one_cell(hv, row, how, k, key1(ds, task, k, row, how, 0)) for k in all_keys()}
    return out


def rule_persons(hv: dict, ds: str, task: str) -> dict:
    """Each rule stage (and the two beside) with R, R_loc and R_lag on the persons with the stage and R."""
    out: dict = {}
    for how in v6b.BASELINE:
        out[how] = {}
        for row in MAIN_ROWS:
            cell = {}
            for k in (*RULE_STAGES, *BESIDE):
                gate = 1 + STAGE[base_of(k)[0]]
                c = {"stage": one_cell(hv, row, how, k, key1(ds, task, k, row, how, gate), extra="R")}
                for g in GATES:
                    c[g] = one_cell(hv, row, how, g, key1(ds, task, g, row, how, gate), extra=k)
                cell[k] = c
            out[how][row] = cell
    return out


def evaluate_rule(rp: dict, gate: str) -> dict:
    """v6k's part 1 with the given gate."""
    out: dict = {}
    for k in (*RULE_STAGES, *BESIDE):
        st = base_of(k)[0]
        cells = {"read": {}, "status": {}, "left_out_of_rule_persons": {}}
        classes = []
        for ds, task in MAIN:
            nm = f"{ds}_{task}"
            for how in v6b.BASELINE:
                for row in MAIN_ROWS:
                    c = rp[nm][how][row][k]
                    where = f"{nm} | {how} | {row}"
                    cells["left_out_of_rule_persons"][where] = c["stage"]["left_out_no_vector"]
                    if (row, how) in k6.BY_CONSTRUCTION.get(st, set()):
                        cells["status"][where] = "fixed by construction"
                        continue
                    if size_class(c[gate]) == "no count" or size_class(c["stage"]) == "no count":
                        cells["status"][where] = "fewer than 2 rule persons"
                        continue
                    if size_class(c[gate]) != "above":
                        cells["status"][where] = "control not met"
                        continue
                    cls = size_class(c["stage"])
                    cells["read"][where] = cls
                    cells["status"][where] = f"read: {cls}"
                    classes.append(cls)
        tally: dict = {}
        for where, cls in cells["read"].items():
            nm, _, row = where.split(" | ")
            tally.setdefault(f"{nm} | {row}", {"above": 0, "below": 0, "undecided": 0})[cls] += 1
        lab = label(classes) if k in RULE_STAGES else "beside the rule: no label"
        out[k] = {"label": lab, "cells_read": len(classes), "classes_by_dataset_and_row": tally, **cells}
    return out


def cell_kind(r: str, loc: str, lag: str) -> str:
    if r != "above":
        return "not seen"
    if loc == "above" and lag == "above":
        return "both"
    if loc == "above":
        return "location only"
    if lag == "above":
        return "order only"
    return "composite only"


def gate_cells(rp: dict) -> dict:
    """Per rule stage and main cell: the classes of R, R_loc and R_lag on that stage's rule persons, and the kind."""
    out: dict = {}
    for k in (*RULE_STAGES, *BESIDE):
        out[k] = {}
        for ds, task in MAIN:
            nm = f"{ds}_{task}"
            for how in v6b.BASELINE:
                for row in MAIN_ROWS:
                    c = rp[nm][how][row][k]
                    cls = {g: size_class(c[g]) for g in GATES}
                    out[k][f"{nm} | {how} | {row}"] = {**cls, "kind": cell_kind(cls["R"], cls["R_loc"], cls["R_lag"])}
    return out


def label_changes(parts: dict) -> dict:
    """Per stage: the label under each gate, and the cells whose status differs between gate R and gate R_loc."""
    out = {}
    for k in (*RULE_STAGES, *BESIDE):
        r, loc = parts["R"][k], parts["R_loc"][k]
        out[k] = {"labels": {g: parts[g][k]["label"] for g in GATES},
                  "changed_R_to_R_loc": r["label"] != loc["label"],
                  "cells_whose_status_differs_R_to_R_loc": {w: [r["status"][w], loc["status"][w]]
                                                            for w in r["status"] if r["status"][w] != loc["status"][w]}}
    return out


def trims_part2(cnt: dict) -> dict:
    out: dict = {}
    for st in TH_STAGES:
        out[st] = {}
        ref = {}
        for ds, task in MAIN:
            for how in v6b.BASELINE:
                for row in MAIN_ROWS:
                    ref[(ds, task, how, row)] = size_class(cnt[f"{ds}_{task}"][how][row][sk(st, "2000")])
        for th in TRIM_TH:
            classes = {f"{ds}_{task} | {how} | {row}": size_class(cnt[f"{ds}_{task}"][how][row][sk(st, th)])
                       for (ds, task, how, row) in ref}
            same = sum(classes[f"{ds}_{task} | {how} | {row}"] == r0 != "no count"
                       for (ds, task, how, row), r0 in ref.items())
            no_count = sum("no count" in (classes[f"{ds}_{task} | {how} | {row}"], r0)
                           for (ds, task, how, row), r0 in ref.items())
            out[st][th] = {"classes": classes, "same_as_2000": same, "cells_without_a_count": no_count,
                           "cells": len(ref)}
    return out


# ================================================================ reliability


def corr_split(A: np.ndarray, B: np.ndarray) -> tuple[float | None, str | None, int]:
    """Pearson r over persons with a value on both sides (rows with nan dropped); (r, reason without, persons)."""
    ok = ~(np.isnan(A) | np.isnan(B))
    n = int(ok.sum())
    if n < REL_MIN_PERSONS:
        return None, "fewer than 10 persons", n
    a, b = A[ok], B[ok]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, "constant", n
    r = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(r):
        raise SystemExit("a correlation is not finite")
    if r == -1.0:
        return None, "r = -1", n
    return r, None, n


def sb(r: float) -> float:
    return 2 * r / (1 + r)


def flat(v: dict, st: str) -> np.ndarray:
    x = v[st]
    if x is None:
        return np.full(REL_DIMS[st], np.nan)
    x = np.asarray(x, dtype=float)
    if x.shape != (REL_DIMS[st],) or not np.all(np.isfinite(x)):
        raise SystemExit(f"{st}: a vector of the wrong shape or a non-finite value")
    return x


def fixed_part_stats(A: np.ndarray, B: np.ndarray) -> dict:
    r, why, n = corr_split(A, B)
    ok = ~(np.isnan(A) | np.isnan(B))
    d = A[ok] - B[ok]
    out = {"r": r, "rSB": sb(r) if r is not None else None, "without": why, "persons": n}
    if ok.sum():
        out.update({"mean_A_minus_B": float(d.mean()), "mean_abs_A_minus_B": float(np.mean(np.abs(d))),
                    "median_abs_A_minus_B": float(np.median(np.abs(d))),
                    "sd_A": float(A[ok].std()), "sd_B": float(B[ok].std())})
    return out


def reliability(texts: list, ds: str, task: str, n_splits: int = N_SPLITS) -> dict:
    """texts: (Rec of the text, place, v6k's fixed-part vectors) per count person."""
    fasts = [Fast(rec) for rec, _, _ in texts]
    for f, (rec, _, vec) in zip(fasts, texts):
        for how in v6b.BASELINE:
            for h, part in enumerate(v6b.parts(rec.n, how)):
                if not same_vectors(f.vectors(part), vec[how][h], REL_STAGES):
                    raise SystemExit(f"{ds} {task} {how}: the splits' part vectors differ from v6k's")
    rngs = [np.random.default_rng([SEED, 93, k6.DATASET[ds], k6.TASK[task], place]) for _, place, _ in texts]
    feats = [(st, j) for st in REL_STAGES for j in range(REL_DIMS[st])]
    r_all = np.full((n_splits, len(feats)), np.nan)
    why_all: dict = {}
    n_per = {st: [] for st in REL_STAGES}
    for s in range(n_splits):
        A = {st: np.empty((len(texts), REL_DIMS[st])) for st in REL_STAGES}
        B = {st: np.empty((len(texts), REL_DIMS[st])) for st in REL_STAGES}
        for p, (f, (rec, _, _)) in enumerate(zip(fasts, texts)):
            perm = rngs[p].permutation(rec.n)
            h = rec.n // 2
            va, vb = f.vectors(perm[:h]), f.vectors(perm[h:])
            if s < 2:  # the splits' code against v6k's part vectors on the first splits too
                ref_a, _ = part_vectors_l(rec, np.sort(perm[:h]), {"2000": 2000.0})
                ref_b, _ = part_vectors_l(rec, np.sort(perm[h:]), {"2000": 2000.0})
                if not (same_vectors(va, ref_a, REL_STAGES) and same_vectors(vb, ref_b, REL_STAGES)):
                    raise SystemExit(f"{ds} {task} split {s}: the splits' part vectors differ from v6k's")
            for st in REL_STAGES:
                A[st][p], B[st][p] = flat(va, st), flat(vb, st)
        for st in REL_STAGES:
            both = ~(np.isnan(A[st]).any(axis=1) | np.isnan(B[st]).any(axis=1))
            n_per[st].append(int(both.sum()))
        for fi, (st, j) in enumerate(feats):
            r, why, _ = corr_split(A[st][:, j], B[st][:, j])
            if r is None:
                why_all.setdefault(f"{st}[{j}]", {}).setdefault(why, 0)
                why_all[f"{st}[{j}]"][why] += 1
            else:
                r_all[s, fi] = r
    fixed: dict = {how: {} for how in v6b.BASELINE}
    for how in v6b.BASELINE:
        A = {st: np.array([flat(vec[how][0], st) for _, _, vec in texts]) for st in REL_STAGES}
        B = {st: np.array([flat(vec[how][1], st) for _, _, vec in texts]) for st in REL_STAGES}
        for st, j in feats:
            fixed[how][f"{st}[{j}]"] = fixed_part_stats(A[st][:, j], B[st][:, j])
    out: dict = {"persons": len(texts), "splits": n_splits, "features": {}}
    for fi, (st, j) in enumerate(feats):
        r = r_all[:, fi]
        r = r[~np.isnan(r)]
        nm = f"{st}[{j}]"
        cell = {"splits_with_a_value": int(r.size), "splits_without": why_all.get(nm, {}),
                "persons_per_split": {"min": int(min(n_per[st])), "median": float(np.median(n_per[st])),
                                      "max": int(max(n_per[st]))},
                "fixed_parts": {h: fixed[h][nm] for h in v6b.BASELINE}}
        if r.size:
            rs = 2 * r / (1 + r)
            cell.update({"mean_r": float(r.mean()), "r_percentile_95": [float(np.percentile(r, 2.5)),
                                                                        float(np.percentile(r, 97.5))],
                         "mean_rSB": float(rs.mean()), "rSB_percentile_95": [float(np.percentile(rs, 2.5)),
                                                                             float(np.percentile(rs, 97.5))],
                         "low": bool(rs.mean() < LOW_RSB)})
        out["features"][nm] = cell
    return out


# ================================================================ checks against v6k


def check_against_v6k(rk: dict, nm: str, c1: dict, rp: dict | None, diag: dict) -> None:
    if diag["count_persons"] != rk["writer"][nm]["count_persons"]:
        raise SystemExit(f"{nm}: count persons differ from v6k's")
    if diag["theta_w_text_untrimmed"] != rk["writer"][nm]["theta_w_text"]:
        raise SystemExit(f"{nm}: the untrimmed text fits differ from v6k's")
    fields = ("persons", "share_vs_own", "share_vs_own_ci", "left_out_no_vector")
    for how in v6b.BASELINE:
        for row in ROWS[1:]:
            for k in REUSE_GATE0:
                mine, theirs = c1[how][row][k], rk["count_1"][nm][how][row][k]
                if mine.get("defined", True) is False or theirs.get("defined", True) is False:
                    if mine != theirs:
                        raise SystemExit(f"{nm} {how} {row} {k}: differs from v6k's (defined)")
                    continue
                for f in fields:
                    if mine.get(f) != theirs.get(f):
                        raise SystemExit(f"{nm} {how} {row} {k}: {f} differs from v6k's")
        if rp is None:
            continue
        for row in MAIN_ROWS:
            for k in (*RULE_STAGES, "Bsd_complete@2000"):
                for part in ("stage", "R"):
                    mine, theirs = rp[how][row][k][part], rk["rule_persons"][nm][how][row][k][part]
                    for f in fields:
                        if mine.get(f) != theirs.get(f):
                            raise SystemExit(f"{nm} {how} {row} {k} {part}: {f} differs from v6k's rule persons")


def check_rule_against_v6k(part_r: dict, rk: dict) -> None:
    for k in (*RULE_STAGES, "Bsd_complete@2000"):
        mine, theirs = part_r[k], rk["rule"]["part_1"][k]
        for f in ("label", "cells_read", "classes_by_dataset_and_row", "status", "left_out_of_rule_persons"):
            if mine[f] != theirs[f]:
                raise SystemExit(f"gate R {k}: {f} differs from v6k's part 1")


# ================================================================ predictions and main cells


NE = "not evaluable"


def predictions(rec: dict) -> dict:
    rp = rec["rule_persons"]
    shuffles = ("shuffled/before", "shuffled_in_halves/before")

    def cells(stage, gate, rows):
        return [rp[f"{ds}_{task}"][how][row][stage][gate] for ds, task in MAIN for how in v6b.BASELINE for row in rows]

    loc = [size_class(c) for c in cells("U@2000", "R_loc", shuffles)]
    p1 = NE if "no count" in loc else all(x == "above" for x in loc)
    lag = [size_class(c) for c in cells("U@2000", "R_lag", shuffles)]
    p2 = NE if "no count" in lag else sum(x == "above" for x in lag) >= 6
    lc = rec["rule"]["label_changes"]
    p3 = all(not lc[k]["changed_R_to_R_loc"] for k in RULE_STAGES)
    rel = rec["reliability"]

    def mean_rsb(nm, f):
        return rel[nm]["features"][f].get("mean_rSB")

    v4 = [mean_rsb("kupa_essay", f"S0[{j}]") for j in range(7)]
    p4 = NE if None in v4 else all(v >= 0.80 for v in v4)
    v5 = [mean_rsb(nm, f"U@2000[{j}]") for nm in ("kupa_essay", "lwp_lwp") for j in range(4)]
    p5 = NE if None in v5 else all(v >= 0.50 for v in v5)
    read = rec["rule"]["part_1_by_gate"]["R"]["B_complete@2000"]["read"]
    read_sh = [cls for where, cls in read.items() if where.split(" | ")[2] in shuffles]
    p6 = NE if not read_sh else all(cls != "above" for cls in read_sh)
    w = rec["writer"]
    d7 = [w[nm]["trim"]["120"]["ln_theta_t_minus_ln_theta_w_text"] for nm in w]
    p7 = NE if None in d7 else all(abs(d["median"]) <= 0.1 for d in d7)
    t8 = [(w[nm]["trim"]["30"]["theta_t_text"], w[nm]["theta_w_text_untrimmed"]) for nm in w]
    p8 = NE if any(a is None or b is None for a, b in t8) else all(
        2000.0 < a["theta_ms"]["median"] < b["theta_ms"]["median"] for a, b in t8)
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6, "P7": p7, "P8": p8}


def main_cells_out(rec: dict) -> dict:
    out: dict = {}
    for k, v in rec["rule"]["label_changes"].items():
        out[f"rule | {k} | labels"] = v["labels"]
    kinds: dict = {}
    for k, cells in rec["rule"]["gate_cells"].items():
        for where, c in cells.items():
            kinds.setdefault(k, {}).setdefault(c["kind"], 0)
            kinds[k][c["kind"]] += 1
    out["rule | cell kinds"] = kinds
    for nm in ("kupa_essay", "lwp_lwp"):
        for f, c in rec["reliability"][nm]["features"].items():
            if "mean_rSB" in c:
                out[f"reliability | {nm} | {f}"] = [round(c["mean_rSB"], 3), [round(x, 3) for x in c["rSB_percentile_95"]]]
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for row in MAIN_ROWS:
                c = rec["rule_persons"][nm][how][row]["B_complete@2000"]["stage"]
                out[f"B_complete | {nm} | {how} | {row}"] = (
                    [round(c["share_vs_own"], 3), [round(x, 3) for x in c["share_vs_own_ci"]]]
                    if "share_vs_own_ci" in c else c)
    for nm, w in rec["writer"].items():
        for T in TRIM_MS:
            t = w["trim"][T]
            out[f"trim {T} s | {nm}"] = {"trimmed_start": t["trimmed_start"], "trimmed_end": t["trimmed_end"],
                                         "untrimmed": t["untrimmed"], "consumed": t["consumed"],
                                         "theta_t_median": t["theta_t_text"]["theta_ms"]["median"] if t["theta_t_text"]
                                         else None,
                                         "theta_w_median": w["theta_w_text_untrimmed"]["theta_ms"]["median"],
                                         "ln_diff_median": (t["ln_theta_t_minus_ln_theta_w_text"] or {}).get("median")}
    return out


# ================================================================ run


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    if not OUT_DIR.parent.is_dir():
        raise SystemExit(f"{OUT_DIR.parent} does not exist")
    raw, c = cm.load_conditions(COND)
    for p, h in c["earlier"].items():
        cm.check_file(ROOT / p, h, p)
    ck = json.loads(COND_K.read_bytes())
    cm.check_generator(ck["generator"])
    from importlib import metadata
    for pkg in ("numpy", "pillow", "scipy"):
        if metadata.version(pkg) != c[pkg]:
            raise SystemExit(f"{pkg} is {metadata.version(pkg)}, not {c[pkg]}")
    for p, h in ck["earlier"].items():
        cm.check_file(ROOT / p, h, p)
    c6 = json.loads((ROOT / "works/conditions/prototype_v6.json").read_bytes())
    c4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_bytes())
    d_lwp = json.loads((ROOT / "works/conditions/prototype_v6e.json").read_text())["data"]["lwp"]
    kp = c6["data"]["kupa"]["files"]
    for f, h in [(d_lwp["file"], d_lwp["sha256"])] + [(x["file"], x["sha256"]) for x in kp.values()]:
        if cm.sha256(ROOT / f) != h:
            raise SystemExit(f"{f} does not match its SHA-256")
    recs = {k: json.loads((ROOT / f"results/prototype_{k}/record.json").read_text()) for k in ("v6", "v6b", "v6e")}
    rk = json.loads(REC_K.read_text())
    paths = [COND.relative_to(ROOT).as_posix(), *c["earlier"], *ck["earlier"], *CODE]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    m = c4["mapping"]
    rec: dict = {"persons": {}, "checks": {}}

    # v6k's persons (v6k's draw)
    kupa, seen = {}, {}
    for task in ("copy", "essay"):
        kupa[task], seen[task] = v6b.read_kupa_b(ROOT / kp[task]["file"])
    ids_sorted = sorted(seen["essay"])
    kplace = {pid: i for i, pid in enumerate(ids_sorted)}
    order = v6.draw_order(ids_sorted, 0)
    elig = lambda pid: k6.kupa_eligible(kupa, pid)  # noqa: E731
    v6_ids, _, _ = k6.draw(order, elig, 0, k6.V6_LOOKED, None)
    v6b_ids, _, _ = k6.draw(order, elig, k6.V6_LOOKED, k6.SKIP["kupa"], None)
    if k6.ids_sha(v6_ids) != recs["v6"]["datasets"]["kupa"]["ids_sha256"] \
            or k6.ids_sha(v6b_ids) != recs["v6b"]["persons"]["ids_sha256"]:
        raise SystemExit("positions 0-436 do not give v6's and v6b's persons")
    k_ids, k_series, k_looked = k6.draw(order, elig, k6.SKIP["kupa"], None, v6.N)
    if k6.ids_sha(k_ids) != rk["persons"]["kupa"]["ids_sha256"] or k_looked != rk["persons"]["kupa"][
            "ids_looked_at_from_437"]:
        raise SystemExit("not v6k's KUPA persons")
    rec["persons"]["kupa"] = {"ids_sha256": k6.ids_sha(k_ids), "same_as_v6k": True}
    del kupa
    lwp, lids, _ = v6e.read_lwp(ROOT / d_lwp["file"], None)
    lplace = {pid: i for i, pid in enumerate(sorted(lids))}
    lorder = v6.draw_order(sorted(lids), 2)
    lelig = lambda pid: k6.lwp_eligible(lwp, pid)  # noqa: E731
    e_ids, _, _ = k6.draw(lorder, lelig, 0, k6.SKIP["lwp"], None)
    if k6.ids_sha(e_ids) != recs["v6e"]["persons"]["ids_sha256"]:
        raise SystemExit("positions 0-203 do not give v6e's persons")
    l_ids, l_series, l_looked = k6.draw(lorder, lelig, k6.SKIP["lwp"], None, v6.N)
    if k6.ids_sha(l_ids) != rk["persons"]["lwp"]["ids_sha256"] or l_looked != rk["persons"]["lwp"][
            "ids_looked_at_from_204"]:
        raise SystemExit("not v6k's LWP persons")
    rec["persons"]["lwp"] = {"ids_sha256": k6.ids_sha(l_ids), "same_as_v6k": True}
    del lwp

    rec["writer"], rec["count_1"], rec["rule_persons"], rec["reliability"] = {}, {}, {}, {}
    for ds, task, series, ids, place in (("kupa", "essay", k_series, k_ids, kplace),
                                         ("kupa", "copy", k_series, k_ids, kplace),
                                         ("lwp", "lwp", l_series, l_ids, lplace)):
        nm = f"{ds}_{task}"
        hv, rec["writer"][nm], texts = build(series, ids, ds, task, place, m)
        rec["count_1"][nm] = count1(hv, ds, task)
        rp = rule_persons(hv, ds, task) if (ds, task) in MAIN else None
        check_against_v6k(rk, nm, rec["count_1"][nm], rp, rec["writer"][nm])
        if rp is not None:
            rec["rule_persons"][nm] = rp
        del hv
        rec["reliability"][nm] = reliability(texts, ds, task)
        del texts
    rec["checks"]["reuse"] = ("persons, count persons, untrimmed text fits, gate 0 counts of S0, S1, I, R, U, B, Bsd "
                              "and Bsd_complete at 2 s in every row and baseline, and the rule persons' counts equal "
                              "v6k's record")
    rec["checks"]["parts"] = "the splits' part vectors equal v6k's on the fixed parts and on the first 2 splits"
    rec["checks"]["construction"] = "every identity held for every count person"
    parts = {g: evaluate_rule(rec["rule_persons"], g) for g in GATES}
    check_rule_against_v6k(parts["R"], rk)
    rec["checks"]["gate_R"] = "gate R gives v6k's part 1 labels, classes and statuses"
    rec["rule"] = {"part_1_by_gate": parts, "label_changes": label_changes(parts),
                   "gate_cells": gate_cells(rec["rule_persons"]), "part_2_trim": trims_part2(rec["count_1"])}
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
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v6l-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rec["predictions"]))
    for k, v in rec["main_cells"].items():
        print(k, json.dumps(v))


CHOICES = {  # decisions of the code where the conditions are silent
    "fast_splits": "the random splits use a vectorised copy of v6k's part vectors at 2 s (class Fast); it is checked "
                   "equal (bitwise) to v6k's part_vectors for every person on odd/even, halves and the first 2 splits",
    "trim_keys": "the trimmed fits use v6k's key for the row ([dataset, task, place, row index]); the conditions' "
                 "decision 3 made them v6k's",
    "trim_text_rows": "text/after and text/none reuse the text's trimmed fit, as v6k's writer",
    "trim_short_record": "a record with fewer than 11 presses before any step is consumed",
    "theta_difference": "ln theta_t - ln theta_w over text records with both; exactly 0 is counted",
    "rel_persons_per_split": "the persons per split are those with the whole stage vector in both parts; a feature's r "
                             "uses the same persons (a stage's vector is missing as a whole or present)",
    "rel_fixed_parts": "the fixed parts' r and rSB use the same rules (10 persons, constant, r = -1) and add the mean "
                       "and median absolute A - B and the standard deviations over persons",
    "rule_beside": "B_complete@2000 and Bsd_complete@2000 get the gate classes and kinds but no label",
    "gate_cells_persons": "each stage's cell kinds are read on that stage's rule persons (R, R_loc and R_lag counted "
                          "on them with the stage's gate)",
    "predictions": "P1 and P2 on the rule persons of U@2000; P6 on the shuffle cells that gate R reads for "
                   "B_complete@2000 (its rule persons); P7 and P8 over the three sets; a prediction whose input is "
                   "missing is 'not evaluable'",
    "rel_fixed_parts_sd": "the standard deviations over persons use ddof 0; 'mean and median absolute difference' is "
                          "reported as the mean absolute difference, the median absolute difference and, beside, the "
                          "signed mean",
    "trim_consumed_tallies": "presses and gaps dropped are counted for consumed records too (up to the step that "
                             "consumed them); trimmed_start and trimmed_end count every record with a step there, "
                             "consumed or not",
    "untrimmed_check": "an untrimmed text record must give exactly v6k's untrimmed fit, or the run stops",
    "min_summary": "a five-number summary over fewer than 5 records is not written (None), so that it cannot show "
                   "a person's value",
    "edge_share_beside": "the edge share of B at 2 s is summarised over the text parts beside the complete bursts",
}


if __name__ == "__main__":
    main()
