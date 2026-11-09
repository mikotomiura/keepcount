import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6q as q6  # noqa: E402
from test_prototype_v6o import fake_sets  # noqa: E402

p6, o6, n6, m6, k6 = q6.p6, q6.o6, q6.n6, q6.m6, q6.k6
v6, v6b = k6.v6, k6.v6b
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
RL = json.loads((ROOT / "results/prototype_v6l/record.json").read_text())
COND = json.loads((ROOT / "works/conditions/prototype_v6q.json").read_text())
I2 = q6.I2


# ---------------------------------------------------------------- constants and small functions


def test_constants_follow_the_conditions():
    assert q6.TAG == 100 and q6.I2_IX == 22 and q6.CELL == 9 and q6.LAST == {"rule": 0, "gate_0": 1}
    assert "[20261031, 100, dataset, task, 22, row index, baseline, group, last]" in COND["cells"]["keys"]
    assert "group 9 for a whole cell" in COND["cells"]["keys"] and "10-12 by n_s_min" in COND["cells"]["keys"]
    assert "last 0 on the rule persons and 1 on the gate 0 persons" in COND["cells"]["keys"]
    assert max(n6.STAGE.values()) < q6.I2_IX
    assert q6.qkey("lwp", "lwp", "cross_75", "halves", 12, 0) == [20261031, 100, 2, 0, 22, 15, 1, 12, 0]
    assert q6.qkey("kupa", "essay", "inside/before", "oddeven", 9, 1) == [20261031, 100, 0, 1, 22, 3, 0, 9, 1]
    assert "LWP odd/even inside_100, LWP halves inside_75 and inside_100" in COND["primary"]
    assert q6.GATE_CELLS == (("lwp_lwp", "oddeven", "inside_100"), ("lwp_lwp", "halves", "inside_75"),
                             ("lwp_lwp", "halves", "inside_100"))
    assert q6.PRIMARY_ROWS == ("inside_75", "inside_100", "cross_100") and len(q6.COUNTED_ROWS) == 10
    assert q6.SHUFFLES == ("shuffled/before", "shuffled_in_halves/before")
    assert q6.FIX_TOL == 1e-9 and "1e-9" in COND["fixed_by_construction"]


def test_is_inside():
    assert all(q6.is_inside(r) for r in ("inside/before", "inside_25", "inside_50", "inside_75", "inside_100"))
    assert not any(q6.is_inside(r) for r in ("cross_100", "shuffled/before", "shuffled_in_halves/before"))


def test_cell_state_table():
    assert q6.cell_state("above", "below") == "gate" and q6.cell_state("above", "above") == "gate"
    assert q6.cell_state("undecided", "above") == "R_mean only" and q6.cell_state("below", "above") == "R_mean only"
    assert q6.cell_state("undecided", "undecided") == "not read" and q6.cell_state("below", "below") == "not read"


def pair(i_a, i_b, **extra):
    a = {"I": None if i_a is None else np.array(i_a, dtype=float), **extra}
    b = {"I": None if i_b is None else np.array(i_b, dtype=float), **extra}
    return (a, b)


def test_with_i2_and_check_i2():
    hv = {"oddeven": {"text/before": [pair([1, 2, 3.5], [4, 5, 6.25]), pair(None, [1, 1, 1])]}}
    out = q6.with_i2(hv)
    assert out["oddeven"]["text/before"][0][0][I2].tolist() == [3.5]
    assert out["oddeven"]["text/before"][0][1][I2].tolist() == [6.25]
    assert out["oddeven"]["text/before"][1][0][I2] is None and out["oddeven"]["text/before"][1][1][I2].tolist() == [1]
    assert "I[2]" not in hv["oddeven"]["text/before"][0][0]  # the input is not changed
    q6.check_i2(out)
    bad = copy.deepcopy(out)
    bad["oddeven"]["text/before"][0][1][I2] = np.array([6.0])
    with pytest.raises(SystemExit, match="third number"):
        q6.check_i2(bad)
    bad = copy.deepcopy(out)
    bad["oddeven"]["text/before"][1][0][I2] = np.array([1.0])
    with pytest.raises(SystemExit, match="exists"):
        q6.check_i2(bad)
    bad = copy.deepcopy(out)
    bad["oddeven"]["text/before"][0][0][I2] = np.array([3.5, 0.0])
    with pytest.raises(SystemExit, match="third number"):
        q6.check_i2(bad)


def test_check_inside_fixed():
    first = [pair([1, 2, 3], [4, 5, 6])]
    q6.check_inside_fixed(first, [pair([1 + 1e-12, 2, 9], [4, 5 - 1e-12, 1])], "w")
    with pytest.raises(SystemExit, match="changed under inside"):
        q6.check_inside_fixed(first, [pair([1 + 1e-6, 2, 3], [4, 5, 6])], "w")
    with pytest.raises(SystemExit, match="changed under inside"):
        q6.check_inside_fixed(first, [pair([1, 2, 3], [4, 5.01, 6])], "w")


def i2_people(t_a, t_b, r_a, r_b, others=None):
    first, sel = [], []
    for k, (ta, tb, ra, rb) in enumerate(zip(t_a, t_b, r_a, r_b)):
        o = others[k] if others else (1.0, 2.0)
        first.append(({"I": np.array([*o, ta]), I2: np.array([ta])}, {"I": np.array([*o, tb]), I2: np.array([tb])}))
        sel.append(({"I": np.array([*o, ra]), I2: np.array([ra])}, {"I": np.array([*o, rb]), I2: np.array([rb])}))
    return first, sel


def test_i2_cell_by_hand():
    t_a = [1.0, 2.0, 3.0, 0.5, 1.5, 2.5]
    t_b = [1.4, 2.0, 2.0, 0.9, 1.0, 2.6]
    r_a = [1.9, 2.3, 3.2, 0.5, 1.6, 2.5]
    r_b = [1.2, 2.4, 2.6, 1.7, 1.1, 2.7]
    first, sel = i2_people(t_a, t_b, r_a, r_b)
    base = np.abs(np.array(t_a) - np.array(t_b))
    change = (np.abs(np.array(r_a) - np.array(t_a)) + np.abs(np.array(r_b) - np.array(t_b))) / 2
    want = np.where(change > base, 1.0, np.where(change == base, 0.5, 0.0))
    key = [1, 2, 3]
    e, own, idx = q6.i2_cell(first, sel, key, "w")
    assert own.tolist() == want.tolist() and e["share_vs_own"] == pytest.approx(want.mean(), abs=1e-15)
    assert e["persons"] == 6
    idx2 = np.random.default_rng(key).integers(0, 6, (v6.N_BOOT, 6))
    assert np.array_equal(idx, idx2) and e["share_vs_own_ci"] == v6.ci(want, idx2)
    assert e["class"] == o6.size_class(e["share_vs_own_ci"])
    # the scale does not matter for one number: multiplying every value changes nothing
    f3, s3 = i2_people([3 * x for x in t_a], [3 * x for x in t_b], [3 * x for x in r_a], [3 * x for x in r_b])
    assert q6.i2_cell(f3, s3, key, "w")[1].tolist() == own.tolist()


def test_i2_at_least_i_when_others_fixed():
    rng = np.random.default_rng(5)
    t_a, t_b = rng.normal(size=12), rng.normal(size=12)
    r_a, r_b = t_a + rng.normal(size=12), t_b + rng.normal(size=12)
    others = [(float(x), float(y)) for x, y in rng.normal(size=(12, 2))]
    first, sel = i2_people(t_a, t_b, r_a, r_b, others)
    # I[0] and I[1] differ between the parts: shift the B part's
    for k in range(12):
        for d in (first[k][1], sel[k][1]):
            d["I"] = d["I"] + np.array([0.7, -0.4, 0.0])
    own2 = q6.contrib(first, sel, I2, "w")
    own = q6.contrib(first, sel, "I", "w")
    assert np.all(own2 >= own) and own2.sum() > own.sum()


def test_diff_and_old_part():
    a, b = np.array([1.0, 0.5, 0.0, 1.0]), np.array([0.0, 0.5, 0.0, 0.5])
    idx = np.random.default_rng(3).integers(0, 4, (v6.N_BOOT, 4))
    d = q6.diff(a, b, idx)
    assert d["difference"] == pytest.approx(0.375) and d["ci"] == v6.ci(a - b, idx)
    old = {"share_vs_own": 0.7, "share_vs_own_ci": [0.61, 0.8], "persons": 9}
    assert q6.old_part(old) == {"share_vs_own": 0.7, "share_vs_own_ci": [0.61, 0.8], "class": "above"}
    assert q6.old_part({"share_vs_own": 0.7, "share_vs_own_ci": [0.6, 0.8]})["class"] == "undecided"


def test_admit_withholds_a_new_small_set():
    n = 12
    led = [p6.mask_of(range(10), n)]
    assert not q6.admit(led, [p6.mask_of(range(11), n)])  # one person apart
    assert len(led) == 1
    assert q6.admit(led, [p6.mask_of(range(5), n), p6.mask_of(range(5, 10), n)])
    assert len(led) == 3
    assert not q6.admit(led, [p6.mask_of(range(3), n)])
    assert q6.admit([], [p6.mask_of(range(12), n)])
    assert not q6.admit([], [p6.mask_of(range(4), n)])


def test_own_ledger_ok():
    n = 12
    q6.own_ledger_ok([p6.mask_of(range(12), n), p6.mask_of(range(6), n)])
    with pytest.raises(SystemExit, match="own ledger"):
        q6.own_ledger_ok([p6.mask_of(range(12), n), p6.mask_of(range(8), n), p6.mask_of(range(10), n)])


# ---------------------------------------------------------------- the fake run


@pytest.fixture(scope="module")
def run():
    ks_, ls_ = fake_sets()
    kid, lid = sorted(ks_), sorted(ls_)
    kp, lp = {p: i for i, p in enumerate(kid)}, {p: i for i, p in enumerate(lid)}
    rn = n6.jsonish(n6.count_all(ks_, kid, kp, ls_, lid, lp, M, RL, n_splits=3))
    ro = n6.jsonish(o6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn))
    rp = n6.jsonish(p6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro))
    rec = n6.jsonish(q6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro, rp))
    return rec, rn, ro, rp, (ks_, kid, kp, ls_, lid, lp)


def all_cells(rec):
    for nm in rec["cells"]:
        for how in v6b.BASELINE:
            for row in q6.ROWS_O:
                yield nm, how, row, rec["cells"][nm][how][row]


def test_structure_no_per_person(run):
    rec = run[0]
    out = {"main_cells": q6.main_cells_out({**rec, "predictions": q6.predictions(rec)}), **rec,
           "predictions": q6.predictions(rec)}
    k6.v6d.no_per_person(out)
    k6.v6i.no_long_list(out)
    assert set(rec["cells"]) == {"kupa_essay", "lwp_lwp"}
    assert set(out["main_cells"]["gate_cells"]) == {f"{a} {b} {c}" for a, b, c in q6.GATE_CELLS}
    assert len(out["main_cells"]["primary"]) == 12 and len(out["main_cells"]["primary_groups_n_s"]) == 12
    for nm, how, row, e in all_cells(rec):
        assert set(e) == {"rule", "gate_0"}


def test_rule_and_gate0_i_equal_v6n(run):
    rec, rn = run[0], run[1]
    n_rule = n_g0 = 0
    for nm, how, row, e in all_cells(rec):
        r, g = e["rule"], e["gate_0"]
        old = rn["rule_persons"][nm][how][row]["I"]
        if not q6.unread(r):
            n_rule += 1
            for part, key in (("stage", "I"), ("R_loc", "R_loc"), ("R_mean", "R_mean")):
                assert r[key]["share_vs_own"] == old[part]["share_vs_own"]
                assert r[key]["share_vs_own_ci"] == old[part]["share_vs_own_ci"]
            assert r["persons"] == old["stage"]["persons"] == r[I2]["persons"]
            assert r["state"] == q6.cell_state(r["R_loc"]["class"], r["R_mean"]["class"])
        if not q6.unread(g):
            n_g0 += 1
            o = rn["count_1"][nm][how][row]["I"]
            assert g["I"]["share_vs_own"] == o["share_vs_own"] and g["persons"] == o["persons"] == g[I2]["persons"]
    assert n_rule > 20 and n_g0 > 0


def test_inside_gain_is_never_negative(run):
    rec = run[0]
    seen = 0
    for nm, how, row, e in all_cells(rec):
        r = e["rule"]
        if q6.unread(r):
            continue
        if q6.is_inside(row):
            seen += 1
            assert r["persons_I[2]_below_I"] == 0
            assert r["I[2]_minus_I"]["difference"] >= 0
            assert r[I2]["share_vs_own"] >= r["I"]["share_vs_own"]
        else:
            assert "persons_I[2]_below_I" not in r
    assert seen >= 10


def test_groups_follow_v6p(run):
    rec, rp = run[0], run[3]
    n_written = 0
    for c in q6.COVARIATES:
        for nm in rec["groups"][c]:
            for how in v6b.BASELINE:
                for row in q6.ROWS_O:
                    cell = rec["groups"][c][nm][how][row]
                    prec = rp["groups"][c][nm][how]["I"][row]
                    r = rec["cells"][nm][how][row]["rule"]
                    if not prec.get("written"):
                        assert not cell["written"]
                        continue
                    if not cell["written"]:
                        assert cell["plan"] == "derivable (I[2])"
                        continue
                    n_written += 1
                    w = sum(cell["groups"][g]["share_vs_own"] * cell["groups"][g]["persons"] for g in q6.GROUPS)
                    assert w / r["persons"] == pytest.approx(r[I2]["share_vs_own"], abs=1e-12)
                    for g in q6.GROUPS:
                        e, pe = cell["groups"][g], prec["groups"][g]
                        assert e["I"]["share_vs_own"] == pe["share_vs_own"] and e["state"] == pe["state"]
                        assert e["R_loc"] == pe["R_loc"] and e["persons"] == pe["persons"]
                        assert e["class"] == o6.size_class(e["share_vs_own_ci"])
                    assert cell["I_high_minus_low"] == prec["high_minus_low"]
                    sh = [cell["groups"][g]["share_vs_own"] for g in q6.GROUPS]
                    assert cell["rising"] == (sh[0] <= sh[1] <= sh[2])
                    assert cell["high_minus_low"]["difference"] == pytest.approx(sh[2] - sh[0], abs=1e-15)
    assert n_written > 0


def test_group_intervals_use_i2_keys(run):
    """Recompute one written group's interval and paired difference with I[2]'s key, from the per-person values of a
    fresh run of the same fake persons."""
    rec, rn, ro, rp, data = run
    ks_, kid, kp, ls_, lid, lp = data
    done = 0
    for ds, task, series, ids, place in (("kupa", "essay", ks_, kid, kp), ("lwp", "lwp", ls_, lid, lp)):
        nm = f"{ds}_{task}"
        b = n6.build(series, ids, ds, task, place, M, o6.BUILD_ROWS)
        hv = q6.with_i2(n6.hv_of(b, o6.BUILD_ROWS))
        cov = [o6.covariates_of(*(series[pid][task] if ds == "kupa" else series[pid])) for pid in b["persons"]]
        lab = o6.labels_of(o6.set_keep(hv, cov, ds, task, rn))
        for c, off in q6.COVARIATES.items():
            for how in v6b.BASELINE:
                gate0 = o6.cell_persons(hv, how, q6.ROWS_O[0], "R")
                for row in q6.ROWS_O:
                    cell = rec["groups"][c][nm][how][row]
                    if not cell.get("written"):
                        continue
                    _, rule = q6.rule_i2(hv, ds, task, how, row, rn)
                    g = p6.group_of_stage(rule, gate0, lab[how], c)
                    for j, gn in enumerate(q6.GROUPS):
                        own = rule[I2][g == j]
                        idx = np.random.default_rng(q6.qkey(ds, task, row, how, off + j, 0)).integers(
                            0, len(own), (v6.N_BOOT, len(own)))
                        assert cell["groups"][gn]["share_vs_own_ci"] == v6.ci(own, idx)
                        assert cell["groups"][gn]["I[2]_minus_I"]["ci"] == v6.ci(own - rule["stage"][g == j], idx)
                        assert cell["groups"][gn]["I[2]_minus_R_loc"]["ci"] == v6.ci(own - rule["R_loc"][g == j], idx)
                        done += 1
    assert done > 0


def test_cell_interval_uses_i2_key(run):
    rec, rn, ro, rp, data = run
    ks_, kid, kp, ls_, lid, lp = data
    b = n6.build(ls_, lid, "lwp", "lwp", lp, M, o6.BUILD_ROWS)
    hv = q6.with_i2(n6.hv_of(b, o6.BUILD_ROWS))
    for how in v6b.BASELINE:
        for row in ("cross_100", "inside_75"):
            e = rec["cells"]["lwp_lwp"][how][row]["rule"]
            if q6.unread(e):
                continue
            _, rule = q6.rule_i2(hv, "lwp", "lwp", how, row, rn)
            idx = np.random.default_rng(q6.qkey("lwp", "lwp", row, how, 9, 0)).integers(
                0, len(rule[I2]), (v6.N_BOOT, len(rule[I2])))
            assert e[I2]["share_vs_own_ci"] == v6.ci(rule[I2], idx)
            assert e["I[2]_minus_I"]["ci"] == v6.ci(rule[I2] - rule["stage"], idx)
            assert e["I[2]_minus_R_loc"]["ci"] == v6.ci(rule[I2] - rule["R_loc"], idx)


def rerun(run, rn=None, rp=None, ro=None):
    rec, rn0, ro0, rp0, (ks_, kid, kp, ls_, lid, lp) = run
    return q6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn or rn0, ro or ro0, rp or rp0)


def test_stops_on_a_changed_v6n_count_1_cell(run):
    rn = copy.deepcopy(run[1])
    c = rn["count_1"]["lwp_lwp"]["halves"]["cross_50"]["I"]
    c["share_vs_own"] += 1e-9
    with pytest.raises(SystemExit, match="count 1 cell differs"):
        rerun(run, rn=rn)


def test_stops_on_a_changed_v6n_rule_cell(run):
    rn = copy.deepcopy(run[1])
    rn["rule_persons"]["kupa_essay"]["oddeven"]["shuffled/before"]["I"]["stage"]["share_vs_own"] += 1e-9
    with pytest.raises(SystemExit, match="differs from v6n's record"):
        rerun(run, rn=rn)


def first_written(rp, c="n_s"):
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for row in q6.ROWS_O:
                if rp["groups"][c][nm][how]["I"][row].get("written"):
                    return nm, how, row
    raise AssertionError("no written family")


def test_stops_on_changed_v6p_groups(run):
    rp = copy.deepcopy(run[3])
    nm, how, row = first_written(rp)
    rp["groups"]["n_s"][nm][how]["I"][row]["groups"]["mid"]["share_vs_own"] += 1e-9
    with pytest.raises(SystemExit, match="differ from v6p's record"):
        rerun(run, rp=rp)
    rp = copy.deepcopy(run[3])
    rp["groups"]["n_s"][nm][how]["I"][row]["high_minus_low"]["ci"][0] -= 1e-9
    with pytest.raises(SystemExit, match="differ from v6p's record"):
        rerun(run, rp=rp)


def test_stops_on_a_changed_v6p_plan(run):
    rp = copy.deepcopy(run[3])
    nm, how, row = first_written(rp)
    cell = rp["groups"]["n_s"][nm][how]["I"][row]
    rp["groups"]["n_s"][nm][how]["I"][row] = {"persons": cell["persons"], "written": False, "plan": "written"}
    with pytest.raises(SystemExit, match="withheld family differs"):
        rerun(run, rp=rp)


def test_withheld_v6p_family_carries_over(run):
    rec, rp = run[0], run[3]
    seen = 0
    for c in q6.COVARIATES:
        for nm in rec["groups"][c]:
            for how in v6b.BASELINE:
                for row in q6.ROWS_O:
                    prec = rp["groups"][c][nm][how]["I"][row]
                    if prec.get("written") is False and not prec.get("no_count"):
                        cell = rec["groups"][c][nm][how][row]
                        assert cell == {"persons": prec["persons"], "written": False, "plan": prec["plan"]}
                        seen += 1
    assert seen > 0


def test_stops_on_v6o_gate(run):
    ro = copy.deepcopy(run[2])
    nm, how, row = first_written(run[3])
    ro["groups"]["n_s"][nm][how]["cells"]["R_loc"][row]["groups"]["low"]["share_vs_own"] += 1e-9
    with pytest.raises(SystemExit):
        rerun(run, ro=ro)


def test_withholding_is_recorded(run, monkeypatch):
    """With an admit that refuses everything, every cell and family is written with its persons only."""
    monkeypatch.setattr(q6, "admit", lambda ledger, sets: False)
    rec = n6.jsonish(rerun(run))
    for nm, how, row, e in all_cells(rec):
        for part in ("rule", "gate_0"):
            if not e[part].get("no_count"):
                assert set(e[part]) == {"persons", "withheld"}
    for c in q6.COVARIATES:
        for nm in rec["groups"][c]:
            for how in v6b.BASELINE:
                for row in q6.ROWS_O:
                    assert not rec["groups"][c][nm][how][row]["written"]


def test_withholding_rule_withholds_its_families(run, monkeypatch):
    calls = []

    def first_no(ledger, sets):
        calls.append(len(ledger))
        if not ledger:
            return False
        ledger += sets
        return True
    monkeypatch.setattr(q6, "admit", first_no)
    rec = n6.jsonish(rerun(run))
    for c in q6.COVARIATES:
        for nm in rec["groups"][c]:
            for how in v6b.BASELINE:
                for row in q6.ROWS_O:
                    cell = rec["groups"][c][nm][how][row]
                    assert not cell["written"]


def test_final_ledger_check_is_called(run, monkeypatch):
    seen = []
    monkeypatch.setattr(q6, "own_ledger_ok", lambda sets: seen.append(len(sets)))
    rerun(run)
    assert len(seen) == 2 * 2 * len(q6.ROWS_O) and max(seen) >= 4


def test_ledger_order_rule_ns_nsmin_gate0(run, monkeypatch):
    """With an admit that admits everything, every row and baseline adds rule, n_s, n_s_min, gate 0 in that order."""
    calls = []
    real_family = p6.family

    def admit_all(ledger, sets):
        calls.append(tuple(int(x.sum()) for x in sets))
        ledger += sets
        return True
    tags = []

    def fam(persons, g, n):
        tags.append(int(np.sum(g == 0)))
        return real_family(persons, g, n)
    monkeypatch.setattr(q6, "admit", admit_all)
    monkeypatch.setattr(q6, "own_ledger_ok", lambda sets: None)
    monkeypatch.setattr(q6, "same_ledger", lambda a, b, w: None)
    rec = n6.jsonish(rerun(run))
    k = 0
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for row in q6.ROWS_O:
                e = rec["cells"][nm][how][row]
                want = [1]
                for c in q6.COVARIATES:
                    if rec["groups"][c][nm][how][row]["written"]:
                        want.append(3)
                want.append(1)
                got = [len(t) for t in calls[k:k + len(want)]]
                assert got == want
                assert calls[k][0] == e["rule"]["persons"] and calls[k + len(want) - 1][0] == e["gate_0"]["persons"]
                for j, c in enumerate([c for c in q6.COVARIATES if rec["groups"][c][nm][how][row]["written"]]):
                    assert calls[k + 1 + j] == tuple(rec["groups"][c][nm][how][row]["persons"][g] for g in q6.GROUPS)
                k += len(want)
    assert k == len(calls)


# ---------------------------------------------------------------- predictions and counts by hand


def cell(i2_class, i_share=0.4, i2_share=0.5, upper=-0.1, state="gate"):
    return {"persons": 50, "state": state, I2: {"share_vs_own": i2_share, "class": i2_class},
            "I": {"share_vs_own": i_share, "class": "below"}, "I[2]_minus_R_loc": {"difference": -0.2,
                                                                                  "ci": [-0.3, upper]}}


def pred_rec(gate_classes, kupa_classes, shuffle_lower: int, upper=-0.1):
    cells: dict = {"kupa_essay": {}, "lwp_lwp": {}}
    for nm in cells:
        for how in v6b.BASELINE:
            cells[nm][how] = {row: {"rule": cell("below", upper=upper), "gate_0": {"persons": 1, "no_count": True}}
                              for row in q6.ROWS_O}
    for (nm, how, row), cl in zip(q6.GATE_CELLS, gate_classes):
        cells[nm][how][row]["rule"] = cell(cl, upper=upper)
    for (how, row), cl in zip([(h, r) for h in v6b.BASELINE for r in q6.PRIMARY_ROWS], kupa_classes):
        cells["kupa_essay"][how][row]["rule"] = cell(cl)
    sh = [(nm, how, row) for nm in cells for how in v6b.BASELINE for row in q6.SHUFFLES]
    for k, (nm, how, row) in enumerate(sh):
        cells[nm][how][row]["rule"] = cell("above", i_share=0.8, i2_share=0.7 if k < shuffle_lower else 0.9)
    return {"cells": cells}


def test_predictions_held():
    p = q6.predictions(pred_rec(["above", "above", "below"], ["below"] * 6, 6))
    assert p["P1"].startswith("held (2 of 3") and p["P2"].startswith("held (3 of 3")
    assert p["P3"].startswith("held") and p["P4"] == "held (6 of 8)"


def test_predictions_did_not_hold():
    p = q6.predictions(pred_rec(["above", "undecided", "below"], ["below"] * 5 + ["above"], 5, upper=0.0))
    assert p["P1"].startswith("did not hold (1 of 3") and p["P2"].startswith("did not hold (0 of 3")
    assert p["P3"].startswith("did not hold") and p["P4"] == "did not hold (5 of 8)"


def test_predictions_not_evaluable():
    r = pred_rec(["above"] * 3, ["below"] * 6, 8)
    r["cells"]["lwp_lwp"]["halves"]["inside_75"]["rule"] = {"persons": 50, "withheld": "derivable (I[2])"}
    r["cells"]["kupa_essay"]["oddeven"]["cross_100"]["rule"] = {"persons": 1, "no_count": True}
    r["cells"]["lwp_lwp"]["oddeven"]["shuffled/before"]["rule"] = {"persons": 50, "withheld": "derivable (I[2])"}
    p = q6.predictions(r)
    assert p["P1"] == p["P2"] == p["P3"] == p["P4"] == q6.NE


def test_predictions_ties_on_the_shuffles():
    r = pred_rec(["above"] * 3, ["below"] * 6, 0)
    for nm in r["cells"]:
        for how in v6b.BASELINE:
            for row in q6.SHUFFLES:
                r["cells"][nm][how][row]["rule"][I2]["share_vs_own"] = 0.8  # equal to I's: not below
    assert q6.predictions(r)["P4"] == "did not hold (0 of 8)"


def test_counts_by_hand():
    r = pred_rec(["above", "undecided", "below"], ["below"] * 6, 6)
    r["cells"]["kupa_essay"]["oddeven"]["cross_25"]["rule"] = {"persons": 1, "no_count": True}
    r["cells"]["kupa_essay"]["halves"]["cross_25"]["rule"] = {"persons": 30, "withheld": "derivable (I[2])"}
    r["cells"]["kupa_essay"]["halves"]["cross_50"]["rule"]["state"] = "R_mean only"
    r["cells"]["lwp_lwp"]["halves"]["inside/before"]["rule"] = cell("above", state="not read")  # not counted
    r["groups"] = {c: {nm: {how: {row: {"written": False, "persons": {}} for row in q6.ROWS_O}
                            for how in v6b.BASELINE} for nm in r["cells"]} for c in q6.COVARIATES}
    g = {gn: {"class": cl, "state": st, "I": {"class": "below"}} for gn, cl, st in
         (("low", "below", "gate"), ("mid", "above", "gate in group only"), ("high", "undecided", "not read"))}
    r["groups"]["n_s"]["lwp_lwp"]["oddeven"]["cross_100"] = {"written": True, "groups": g,
                                                             "high_minus_low": {"ci": [0.01, 0.2]}}
    r["groups"]["n_s"]["lwp_lwp"]["oddeven"]["inside/before"] = {"written": True, "groups": g,
                                                                 "high_minus_low": {"ci": [0.01, 0.2]}}
    r["groups"]["n_s_min"]["kupa_essay"]["halves"]["cross_75"] = {"no_count": True, "written": False}
    r["cells"]["lwp_lwp"]["oddeven"]["cross_25"]["gate_0"] = {"persons": 60, "withheld": "derivable (I[2])"}
    r["cells"]["lwp_lwp"]["oddeven"]["cross_50"]["gate_0"] = cell("below")
    r["cells"]["lwp_lwp"]["oddeven"]["inside/before"]["gate_0"] = cell("below")  # not counted
    r["groups"]["n_s"]["kupa_essay"]["oddeven"]["cross_75"] = {"written": False, "persons": {},
                                                               "plan": "derivable (I[2])"}
    o = q6.counts(r)
    assert o["gate_0_withheld"] == 1 and o["gate_0_written"] == 1
    assert o["groups_n_s"]["families_withheld_I2"] == 1
    assert o["cells_no_count"] == 1 and o["cells_withheld"] == 1
    total = sum(sum(v.values()) for s in o["cells"].values() for k, v in s.items() if k == I2)
    assert total == 2 * 2 * 10 - 2
    assert o["cells"]["R_mean only"][I2]["below"] == 1
    # one gate cell above and the 8 shuffle cells above; one undecided
    assert o["cells"]["gate"][I2]["above"] == 9 and o["cells"]["gate"][I2]["undecided"] == 1
    assert o["cells"]["not read"][I2] == dict.fromkeys(q6.CLASSES, 0)
    gn = o["groups_n_s"]
    assert gn["cells_written"] == 1 and gn["families_withheld_v6p"] == 2 * 2 * 10 - 2
    assert gn["high_minus_low"]["above 0"] == 1
    assert gn["states"]["gate"][I2]["below"] == 1 and gn["states"]["gate in group only"][I2]["above"] == 1
    assert gn["states"]["not read"][I2]["undecided"] == 1 and gn["states"]["not read"]["I"]["below"] == 1
    assert o["groups_n_s_min"]["no_count"] == 1


def test_short_cell_and_table(run):
    rec = run[0]
    mc = q6.main_cells_out({**rec, "predictions": {}})
    for k, v in mc["primary"].items():
        nm, how, row = k.split(" ")
        e = rec["cells"][nm][how][row]["rule"]
        if not q6.unread(e):
            assert v["I[2]"] == [round(e[I2]["share_vs_own"], 3), e[I2]["class"]]
            assert v["state"] == e["state"]
    assert set(mc["table"]) == {f"{nm} {how}" for nm in rec["cells"] for how in v6b.BASELINE}
    assert all(set(t) == set(q6.ROWS_O) for t in mc["table"].values())
    assert q6.short_cell({"persons": 3, "no_count": True}) == {"no_count": True, "persons": 3}
    assert q6.short_cell({"persons": 30, "withheld": "derivable (I[2])"}) == {"withheld": "derivable (I[2])",
                                                                             "persons": 30}


def test_gate0_written_and_withheld_in_the_fake_run(run):
    """Decision 2 end to end: where the gate 0 persons exceed the rule persons by 1-4, gate 0 is withheld and the rule
    cell is written; elsewhere both can be written."""
    rec = run[0]
    seen_withheld = seen_written = 0
    for nm, how, row, e in all_cells(rec):
        r, g = e["rule"], e["gate_0"]
        if "withheld" in g:
            assert g["withheld"] == "derivable (I[2])"
            if not q6.unread(r):
                seen_withheld += 1
                assert 1 <= g["persons"] - r["persons"] <= 4
        elif not g.get("no_count"):
            seen_written += 1
    assert seen_withheld > 0 and seen_written > 0


def test_gate0_interval_uses_the_gate0_key(run):
    rec, rn, ro, rp, data = run
    ks_, kid, kp, ls_, lid, lp = data
    done = 0
    for ds, task, series, ids, place in (("kupa", "essay", ks_, kid, kp), ("lwp", "lwp", ls_, lid, lp)):
        b = n6.build(series, ids, ds, task, place, M, o6.BUILD_ROWS)
        hv = q6.with_i2(n6.hv_of(b, o6.BUILD_ROWS))
        for how in v6b.BASELINE:
            for row in q6.ROWS_O:
                g = rec["cells"][f"{ds}_{task}"][how][row]["gate_0"]
                if q6.unread(g):
                    continue
                ok = p6.persons_with(hv, how, row, ("I",))
                first = [hv[how]["text/before"][i] for i in ok]
                sel = [hv[how][row][i] for i in ok]
                own = q6.contrib(first, sel, I2, "t")
                idx = np.random.default_rng(q6.qkey(ds, task, row, how, 9, 1)).integers(0, len(ok), (v6.N_BOOT, len(ok)))
                assert g[I2]["share_vs_own"] == pytest.approx(own.mean(), abs=1e-15)
                assert g[I2]["share_vs_own_ci"] == v6.ci(own, idx) and g["persons"] == len(ok)
                if q6.is_inside(row):
                    assert g["persons_I[2]_below_I"] == 0
                done += 1
    assert done > 0


@pytest.mark.parametrize("who", ["rule", "gate_0"])
def test_inside_change_of_i1_stops_the_run(run, monkeypatch, who):
    """Perturb I[1] under inside_75 for one person who is (rule) a rule person or (gate_0) a gate 0 person without R."""
    rec, rn = run[0], run[1]
    real = q6.with_i2

    def perturbed(hv):
        out = real(hv)
        first = out["oddeven"]["text/before"]
        target = None
        for i, (a, b_) in enumerate(first):
            has_r = v6b.has(first[i], "R") and v6b.has(out["oddeven"]["inside_75"][i], "R")
            if a.get("I") is not None and b_.get("I") is not None and (has_r if who == "rule" else not has_r):
                target = i
                break
        if target is None:
            return out
        pair = out["oddeven"]["inside_75"][target]
        d = dict(pair[1])
        d["I"] = np.asarray(d["I"], dtype=float) + np.array([0.0, 1e-6, 0.0])
        out["oddeven"]["inside_75"][target] = (pair[0], d)
        return out
    monkeypatch.setattr(q6, "with_i2", perturbed)
    with pytest.raises(SystemExit, match="changed under inside"):
        rerun(run)


def test_rule_withheld_withholds_its_families(run, monkeypatch):
    """Refuse only the rule cell (the first single set on an empty ledger); admit everything else."""
    def refuse_rule(ledger, sets):
        if not ledger and len(sets) == 1 and not getattr(refuse_rule, "after", False):
            refuse_rule.after = True
            return False
        ledger += sets
        return True

    def per_row(*a, **k):
        refuse_rule.after = False
        return real_rule(*a, **k)
    real_rule = q6.rule_i2
    monkeypatch.setattr(q6, "rule_i2", per_row)
    monkeypatch.setattr(q6, "admit", refuse_rule)
    monkeypatch.setattr(q6, "own_ledger_ok", lambda sets: None)
    rec = n6.jsonish(rerun(run))
    n = 0
    for c in q6.COVARIATES:
        for nm in rec["groups"][c]:
            for how in v6b.BASELINE:
                for row in q6.ROWS_O:
                    cell = rec["groups"][c][nm][how][row]
                    assert not cell["written"]
                    n += cell.get("plan") == "derivable (I[2])"
    assert n > 0


def test_same_ledger():
    m = [p6.mask_of(range(6), 10)]
    q6.same_ledger(m, [p6.mask_of(range(6), 10)], "w")
    with pytest.raises(SystemExit, match="different sets"):
        q6.same_ledger(m, [p6.mask_of(range(7), 10)], "w")
    with pytest.raises(SystemExit, match="different sets"):
        q6.same_ledger(m, m + m, "w")


def test_predictions_stop_on_a_non_gate_state():
    r = pred_rec(["above"] * 3, ["below"] * 6, 6)
    r["cells"]["lwp_lwp"]["halves"]["inside_75"]["rule"]["state"] = "R_mean only"
    with pytest.raises(SystemExit, match="named as a gate cell"):
        q6.predictions(r)


# ---------------------------------------------------------------- guards exercised directly (mutation survivors)


def test_contrib_stops_on_a_bad_contribution(monkeypatch):
    first, sel = i2_people([1.0, 2.0, 3.0], [1.5, 2.5, 2.0], [1.2, 2.9, 3.0], [1.0, 2.5, 2.4])
    monkeypatch.setattr(o6, "contrib_base_change", lambda f, s, k: (np.array([0.3, 1.0, 0.0]), np.ones(3), np.ones(3)))
    with pytest.raises(SystemExit, match="not 0, 1/2 or 1"):
        q6.contrib(first, sel, I2, "w")


def test_i2_cell_stops_when_v6b_one_disagrees(monkeypatch):
    first, sel = i2_people([1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 2.0, 3.0], [1.2, 2.9, 3.0, 4.4], [1.0, 2.5, 2.4, 3.1])
    real = v6b.one
    monkeypatch.setattr(v6b, "one", lambda *a: {**real(*a), "share_vs_own": real(*a)["share_vs_own"] + 0.01})
    with pytest.raises(SystemExit, match="do not give its share"):
        q6.i2_cell(first, sel, [1, 2], "w")
    monkeypatch.setattr(v6b, "one", lambda *a: {**real(*a), "share_vs_own_ci": [0.0, 1.0]})
    with pytest.raises(SystemExit, match="interval"):
        q6.i2_cell(first, sel, [1, 2], "w")


def fresh_hv(run, ds="lwp"):
    rec, rn, ro, rp, (ks_, kid, kp, ls_, lid, lp) = run
    series, ids, place, task = (ls_, lid, lp, "lwp") if ds == "lwp" else (ks_, kid, kp, "essay")
    b = n6.build(series, ids, ds, task, place, M, o6.BUILD_ROWS)
    return q6.with_i2(n6.hv_of(b, o6.BUILD_ROWS)), task


def test_rule_i2_checks_inside_alone(run):
    hv, task = fresh_hv(run)
    rn = run[1]
    rule = p6.rule_keep(hv, "lwp", task, "oddeven", "inside_50", "I", rn)
    i = rule["persons"][0]
    pair = hv["oddeven"]["inside_50"][i]
    d = dict(pair[0])
    d["I"] = np.asarray(d["I"], dtype=float) + np.array([1e-6, 0.0, 0.0])
    hv["oddeven"]["inside_50"][i] = (d, pair[1])
    with pytest.raises(SystemExit, match="changed under inside"):
        q6.rule_i2(hv, "lwp", task, "oddeven", "inside_50", rn)


def test_rule_i2_stops_on_a_class_mismatch(run, monkeypatch):
    hv, task = fresh_hv(run)
    real = p6.rule_keep

    def wrong(*a):
        r = real(*a)
        r["cell_R_loc_class"] = {"above": "below", "below": "above", "undecided": "above"}[r["cell_R_loc_class"]]
        return r
    monkeypatch.setattr(p6, "rule_keep", wrong)
    with pytest.raises(SystemExit, match="R_loc's class differs"):
        q6.rule_i2(hv, "lwp", task, "halves", "cross_100", run[1])


def test_gate0_no_count_must_match_v6n(run, monkeypatch):
    hv, task = fresh_hv(run)
    monkeypatch.setattr(n6, "one_cell", lambda *a, **k: {"too_few_persons": True, "persons": 1})
    with pytest.raises(SystemExit, match="count 1 cell differs"):
        q6.gate0_i2(hv, "lwp", task, "halves", "cross_100", run[1])


def test_gate0_counts_persons_below_i(run, monkeypatch):
    hv, task = fresh_hv(run)
    real = q6.contrib
    monkeypatch.setattr(q6, "contrib", lambda f, s, k, w: np.full(len(f), 2.0) if k == "I" else real(f, s, k, w))
    out, ok = q6.gate0_i2(hv, "lwp", task, "oddeven", "inside_100", run[1])
    assert out["persons_I[2]_below_I"] == len(ok) > 0


def hand_groups(n_per=6):
    rng = np.random.default_rng(11)
    n = 3 * n_per
    # continuous trial values (not 0, 1/2, 1), so that groups with the same values cannot share an interval by chance
    rule = {I2: rng.random(n), "stage": rng.random(n), "R_loc": rng.random(n), "persons": list(range(n))}
    g = np.repeat([0, 1, 2], n_per)
    ig = {"persons": {"low": n_per, "mid": n_per, "high": n_per}, "high_minus_low": {"difference": 0.1, "ci": [0, .2]},
          "groups": {gn: {"persons": n_per, "share_vs_own": 0.5, "share_vs_own_ci": [0.3, 0.7], "class": "undecided",
                          "R_loc": {"class": "above"}, "R_mean": {"class": "above"}, "state": "gate"}
                     for gn in q6.GROUPS}}
    return rule, g, ig


def test_i2_groups_by_hand_with_n_s_min_keys():
    rule, g, ig = hand_groups()
    out = q6.i2_groups(rule, g, "kupa", "essay", "cross_50", "halves", "n_s_min", ig)
    means = {}
    for j, gn in enumerate(q6.GROUPS):
        own = rule[I2][g == j]
        idx = np.random.default_rng(q6.qkey("kupa", "essay", "cross_50", "halves", 10 + j, 0)).integers(
            0, len(own), (v6.N_BOOT, len(own)))
        means[gn] = own[idx].mean(axis=1)
        e = out["groups"][gn]
        assert e["share_vs_own_ci"] == v6.ci(own, idx) and e["share_vs_own"] == pytest.approx(own.mean())
        assert e["I[2]_minus_I"]["ci"] == v6.ci(own - rule["stage"][g == j], idx)
    assert out["high_minus_low"]["ci"] == o6.pct(means["high"] - means["low"])
    assert out["I_high_minus_low"] == ig["high_minus_low"]


def test_i2_groups_stops_on_persons_mismatch():
    rule, g, ig = hand_groups()
    ig["groups"]["mid"]["persons"] = 5
    with pytest.raises(SystemExit, match="differ from I's"):
        q6.i2_groups(rule, g, "kupa", "essay", "cross_50", "halves", "n_s", ig)
    rule, g, ig = hand_groups(n_per=4)
    with pytest.raises(SystemExit, match="fewer than 5"):
        q6.i2_groups(rule, g, "kupa", "essay", "cross_50", "halves", "n_s", ig)


def test_family_out_stops_when_v6p_counted_where_v6n_did_not():
    rp = {"groups": {"n_s": {"kupa_essay": {"oddeven": {"I": {"cross_25": {"written": False, "plan": "derivable",
                                                                            "persons": {}}}}}}}}
    with pytest.raises(SystemExit, match="v6p counted I where v6n did not"):
        q6.family_out({}, None, {"no_count": True, "persons": 1}, [], {}, "kupa", "essay", "oddeven", "cross_25", "n_s",
                      rp, {}, [], 10)
    rp["groups"]["n_s"]["kupa_essay"]["oddeven"]["I"]["cross_25"] = {"no_count": True, "written": False}
    out = q6.family_out({}, None, {"no_count": True, "persons": 1}, [], {}, "kupa", "essay", "oddeven", "cross_25",
                        "n_s", rp, {}, [], 10)
    assert out == {"no_count": True, "persons": 1, "written": False}


def test_same_ledger_is_called_per_set_and_baseline(run, monkeypatch):
    seen = []
    monkeypatch.setattr(q6, "same_ledger", lambda a, b, w: seen.append((len(a), len(b), w)))
    rerun(run)
    assert len(seen) == 4 and all(a == b for a, b, _ in seen)
