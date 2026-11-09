import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6p as p6  # noqa: E402
from test_prototype_v6o import fake_sets  # noqa: E402

o6, n6, m6, k6 = p6.o6, p6.n6, p6.m6, p6.k6
v6, v6b = k6.v6, k6.v6b
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
RL = json.loads((ROOT / "results/prototype_v6l/record.json").read_text())
COND = json.loads((ROOT / "works/conditions/prototype_v6p.json").read_text())


# ---------------------------------------------------------------- constants and small functions


def test_constants_follow_the_conditions():
    assert p6.TAG == 99 and "99" in COND["per_group"]["stage"]
    assert p6.STAGES == ("U@2000", "B@2000", "Bsd@2000", "I", "U_med", "B_complete@2000", "Bsd_complete@2000")
    assert [p6.stage_ix(s) for s in p6.STAGES] == [2, 3, 4, 5, 11, 10, 7]
    assert "U 2, B 3, Bsd 4, I 5, Bsd_complete 7, B_complete 10, U_med 11; R_loc 8, R_mean 12" in COND["implementation"]
    assert "R_loc 9, R_mean 13" in COND["per_group"]["gate"]
    assert 1 + n6.STAGE["R_loc"] == 9 and 1 + n6.STAGE["R_mean"] == 13
    assert p6.skey("lwp", "lwp", "I", "cross_75", "halves", 12) == [20261031, 99, 2, 0, 5, 15, 1, 12, 0]
    assert p6.skey("kupa", "essay", "U_med", "inside/before", "oddeven", 0, 9) == [20261031, 99, 0, 1, 11, 3, 0, 0, 9]
    assert p6.ROWS_O == o6.ROWS_O and "inside/before" not in p6.COUNTED_ROWS and len(p6.COUNTED_ROWS) == 10
    assert p6.PRIMARY_STAGES == ("I", "Bsd@2000") and p6.PRIMARY_ROWS == ("inside_75", "inside_100", "cross_100")
    assert p6.COVARIATES == {"n_s": 0, "n_s_min": 10} and "10, 11, 12 by n_s_min" in COND["per_group"]["stage"]
    assert set(p6.STATES) == {"gate", "gate in group only", "R_mean only", "not read"}


def test_fixed_cells():
    for st in ("U@2000", "U_med", "B@2000"):
        for how in v6b.BASELINE:
            for row in ("inside/before", "inside_25", "inside_50", "inside_75", "inside_100"):
                assert p6.fixed(st, row, how)
            assert not p6.fixed(st, "cross_100", how) and not p6.fixed(st, "shuffled/before", how)
    assert p6.fixed("B@2000", "shuffled_in_halves/before", "halves")
    assert not p6.fixed("B@2000", "shuffled_in_halves/before", "oddeven")
    assert not p6.fixed("U@2000", "shuffled_in_halves/before", "halves")
    for st in ("Bsd@2000", "I", "B_complete@2000", "Bsd_complete@2000"):
        assert not any(p6.fixed(st, row, how) for row in p6.ROWS_O for how in v6b.BASELINE)


def test_gate_state_table():
    assert p6.gate_state("above", "above", "below") == "gate"
    assert p6.gate_state("undecided", "above", "below") == "gate in group only"
    assert p6.gate_state("below", "above", "above") == "gate in group only"
    for cell in ("above", "undecided", "below"):
        assert p6.gate_state(cell, "undecided", "above") == "R_mean only"
        assert p6.gate_state(cell, "below", "above") == "R_mean only"
        assert p6.gate_state(cell, "undecided", "undecided") == "not read"
        assert p6.gate_state(cell, "below", "below") == "not read"


def test_size_class_at_the_line():
    assert o6.size_class([0.60, 0.9]) == "undecided" and o6.size_class([0.3, 0.60]) == "undecided"
    assert o6.size_class([0.6000001, 0.9]) == "above" and o6.size_class([0.3, 0.5999999]) == "below"


# ---------------------------------------------------------------- small sets


def test_small_sets_agrees_with_v6o_and_names_persons():
    n = 20
    full = np.ones(n, bool)
    a = np.zeros(n, bool); a[:10] = True
    b = np.zeros(n, bool); b[:12] = True
    assert p6.small_sets([full, a, ~a]) == set()
    assert p6.small_sets([full, a, b]) == {frozenset({10, 11})}
    d = np.zeros(n, bool); d[16:] = True
    assert frozenset({16, 17, 18, 19}) in p6.small_sets([full, d])
    with pytest.raises(SystemExit):
        p6.small_sets([])


def test_small_sets_stops_if_v6o_disagrees(monkeypatch):
    n = 20
    full = np.ones(n, bool)
    a = np.zeros(n, bool); a[:10] = True
    monkeypatch.setattr(o6, "small_derivable", lambda sets: True)
    with pytest.raises(SystemExit, match="disagrees"):
        p6.small_sets([full, a, ~a])


def vec(keys):
    return ({k: (np.ones(2) if k in keys else None) for k in (*p6.STAGES, *p6.REFS, "R")},
            {k: (np.ones(2) if k in keys else None) for k in (*p6.STAGES, *p6.REFS, "R")})


ALL = (*p6.STAGES, *p6.REFS, "R")


def plan_case(missing: dict, written_o=True, n=21, row="cross_100", how="oddeven", extra_c1: dict | None = None,
              bands: np.ndarray | None = None, drop_c1: tuple = ()):
    """A fake row: every person has every key but those listed in missing {key: [persons]}; three groups of 7."""
    hv = {how: {"text/before": [], row: []}}
    for i in range(n):
        keys = {k for k in ALL if i not in missing.get(k, [])}
        hv[how]["text/before"].append(vec(keys))
        hv[how][row].append(vec(keys))
    gate0 = [i for i in range(n) if i not in missing.get("R", [])]
    g = np.repeat([0, 1, 2], 7)[gate0]
    lab = {c: (0.0, 1.0, g) for c in p6.COVARIATES}
    rn_c1 = {row: {k: ({"persons": len(p6.persons_with(hv, how, row, (k,)))} if k in drop_c1 else
                       {"persons": len(p6.persons_with(hv, how, row, (k,))), "share_vs_own": 0.5}) for k in ALL}}
    if extra_c1:
        rn_c1[row].update(extra_c1)
    rules = {st: {"persons": p6.persons_with(hv, how, row, (st, "R"))} for st in p6.STAGES}
    wo = {c: written_o for c in p6.COVARIATES}
    g0 = np.array(gate0)
    o_fams = [p6.mask_of(g0[g == j], n) for c in p6.COVARIATES if written_o for j in range(3)]
    if bands is not None:
        o_fams += [p6.mask_of(g0[bands == j], n) for j in range(3)]
    base = p6.base_sets(hv, how, row, gate0, o_fams, rn_c1, rules, n)
    plan, tally = p6.plan_row(base, how, row, gate0, lab, wo, rules, n)
    p6.check_plan(base, plan, gate0, lab, rules, n)
    return plan, tally


W = {"n_s": "written", "n_s_min": "written"}


def test_plan_row_writes_when_persons_agree():
    plan, tally = plan_case({})
    assert tally["pooled_before"] == 0 and tally["written_where_strict_would_withhold"] == 0
    assert all(plan[st] == W for st in p6.STAGES)


def test_plan_row_withholds_a_stage_two_persons_short():
    # B_complete misses person 0 (low group) and 7 (mid group). Before, {0, 7} is derivable (count 1's R_loc over 21
    # minus the rule cell over 19); the groups would make {0} and {7} derivable apart: new sets, withheld
    plan, tally = plan_case({"B_complete@2000": [0, 7]})
    assert plan["B_complete@2000"] == {"n_s": "derivable", "n_s_min": "derivable"}
    assert plan["I"] == W
    assert tally["pooled_before"] == 1 and tally["stage_before"]["B_complete@2000"] == 1


def test_plan_row_writes_when_the_small_set_is_only_the_old_one():
    plan, tally = plan_case({"B_complete@2000": [0, 1]})
    assert plan["B_complete@2000"] == W and tally["pooled_before"] == 1
    assert tally["written_where_strict_would_withhold"] == 2


def test_plan_row_group_under_5():
    plan, _ = plan_case({"Bsd_complete@2000": [14, 15, 16, 17, 18]})
    assert plan["Bsd_complete@2000"]["n_s"] == "group under 5"


def test_plan_row_v6o_withheld_carries_over():
    plan, _ = plan_case({}, written_o=False)
    assert all(plan[st] == {"n_s": "v6o withheld", "n_s_min": "v6o withheld"} for st in p6.STAGES)


def test_plan_row_earlier_stages_count_in_the_ledger():
    # without v6o's families, a stage whose persons are not v6o's is planned; if every stage misses {0, 7} and v6o
    # wrote nothing, no family splits them: written
    miss = {st: [0, 7] for st in p6.STAGES}
    plan, tally = plan_case(miss, written_o=False)
    assert all(plan[st] == W for st in p6.STAGES) and tally["pooled_before"] == 1


def test_plan_row_only_v6os_family_splits():
    # every stage misses {0, 7}; only v6o's families (over the 21 persons with R) split them: withheld
    miss = {st: [0, 7] for st in p6.STAGES}
    plan, _ = plan_case(miss, written_o=True)
    assert all(plan[st] == {"n_s": "derivable", "n_s_min": "derivable"} for st in p6.STAGES)


def test_plan_row_only_count_1_splits():
    # every stage misses {0, 7} in its rule cell; v6o wrote nothing; but the stages' count 1 cells have all 21: the
    # count 1 sets alone make {0, 7} derivable before, and the stage's groups (low = v6o's low minus 0) cannot be
    # compared with any 21-person group: written. Dropping I's count 1 share leaves it out of the ledger
    miss = {st: [0, 7] for st in p6.STAGES}
    miss["R"] = []
    plan, tally = plan_case(miss, written_o=False)
    assert tally["pooled_before"] >= 1
    plan2, tally2 = plan_case(miss, written_o=False, drop_c1=tuple(p6.STAGES) + p6.REFS)
    assert tally2["pooled_before"] == 0  # without the count 1 cells nothing is derivable before


def test_plan_row_bands_count_in_the_ledger():
    # B_complete misses person 0; v6o wrote no groups but LWP's bands cut low at 6 persons {0..5}: with the stage's
    # low group {1..6} and the band, {0} and {6} become derivable: withheld
    bands = np.array([0] * 6 + [1] * 8 + [2] * 7)
    plan, _ = plan_case({"B_complete@2000": [0]}, written_o=False, bands=bands)
    assert plan["B_complete@2000"]["n_s"] == "derivable"
    plan, _ = plan_case({"B_complete@2000": [0]}, written_o=False)
    assert plan["B_complete@2000"]["n_s"] == "written"


def test_plan_row_fixed_cells_have_no_plan():
    plan, _ = plan_case({}, row="inside_100")
    for st in ("U@2000", "U_med", "B@2000"):
        assert plan[st] == {}
    assert plan["I"] == W


def test_plan_row_stops_on_count_1_persons():
    with pytest.raises(SystemExit, match="count 1 persons"):
        plan_case({}, extra_c1={"I": {"persons": 20, "share_vs_own": 0.5}})


def test_check_plan_stops_on_a_wrong_plan():
    n, row, how = 21, "cross_100", "oddeven"
    plan, _ = plan_case({"B_complete@2000": [0, 7]})
    # rebuild the same case and force the withheld family to 'written'
    hv = {how: {"text/before": [], row: []}}
    for i in range(n):
        keys = {k for k in ALL if not (k == "B_complete@2000" and i in (0, 7))}
        hv[how]["text/before"].append(vec(keys))
        hv[how][row].append(vec(keys))
    gate0 = list(range(n))
    g = np.repeat([0, 1, 2], 7)
    lab = {c: (0.0, 1.0, g) for c in p6.COVARIATES}
    rn_c1 = {row: {k: {"persons": len(p6.persons_with(hv, how, row, (k,))), "share_vs_own": 0.5} for k in ALL}}
    rules = {st: {"persons": p6.persons_with(hv, how, row, (st, "R"))} for st in p6.STAGES}
    o_fams = [p6.mask_of(np.array(gate0)[g == j], n) for c in p6.COVARIATES for j in range(3)]
    base = p6.base_sets(hv, how, row, gate0, o_fams, rn_c1, rules, n)
    bad = copy.deepcopy(plan)
    bad["B_complete@2000"]["n_s"] = "written"
    with pytest.raises(SystemExit, match="new small set"):
        p6.check_plan(base, bad, gate0, lab, rules, n)


# ---------------------------------------------------------------- the whole count on fake persons


@pytest.fixture(scope="module")
def run():
    ks_, ls_ = fake_sets()
    kid, lid = sorted(ks_), sorted(ls_)
    kp, lp = {p: i for i, p in enumerate(kid)}, {p: i for i, p in enumerate(lid)}
    rn = n6.jsonish(n6.count_all(ks_, kid, kp, ls_, lid, lp, M, RL, n_splits=3))
    ro = n6.jsonish(o6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn))
    rec = n6.jsonish(p6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro))
    return rec, rn, ro, (ks_, kid, kp, ls_, lid, lp)


def cells(rec, c="n_s"):
    for nm in rec["groups"][c]:
        for how in v6b.BASELINE:
            for st in p6.STAGES:
                for row in p6.ROWS_O:
                    yield nm, how, st, row, rec["groups"][c][nm][how][st][row]


def test_count_all_structure_no_per_person(run):
    rec, _, _, _ = run
    out = {"main_cells": p6.main_cells_out(rec), **rec, "predictions": p6.predictions(rec)}
    k6.v6d.no_per_person(out)
    k6.v6i.no_long_list(out)
    for c in p6.COVARIATES:
        assert {(nm, how, st, row) for nm, how, st, row, _ in cells(rec, c)} == {
            (nm, how, st, row) for nm in ("kupa_essay", "lwp_lwp") for how in v6b.BASELINE for st in p6.STAGES
            for row in p6.ROWS_O}


def test_fake_run_has_written_withheld_and_no_count_cells(run):
    rec, _, _, _ = run
    kinds = {"written": 0, "withheld": 0, "no_count": 0}
    for c in p6.COVARIATES:
        for nm, how, st, row, cell in cells(rec, c):
            if cell.get("fixed_by_construction"):
                continue
            kinds["no_count" if cell.get("no_count") else ("written" if cell["written"] else "withheld")] += 1
    assert all(v > 0 for v in kinds.values()), kinds


def test_count_all_fixed_cells_are_not_split(run):
    rec, _, _, _ = run
    for nm, how, st, row, cell in cells(rec):
        assert bool(cell.get("fixed_by_construction")) == p6.fixed(st, row, how)
        if p6.fixed(st, row, how):
            assert set(cell) == {"fixed_by_construction", "persons"}


def test_count_all_groups_give_v6n_rule_shares(run):
    rec, rn, _, _ = run
    n_written = 0
    for c in p6.COVARIATES:
        for nm, how, st, row, cell in cells(rec, c):
            if not cell.get("written"):
                continue
            n_written += 1
            old = rn["rule_persons"][nm][how][row][st]
            tot = sum(cell["persons"].values())
            assert tot == old["stage"]["persons"]
            w = sum(cell["groups"][g]["share_vs_own"] * cell["groups"][g]["persons"] for g in p6.GROUPS) / tot
            assert w == pytest.approx(old["stage"]["share_vs_own"], abs=1e-12)
            for ref in p6.REFS:
                w = sum(cell["groups"][g][ref]["share_vs_own"] * cell["groups"][g]["persons"] for g in p6.GROUPS) / tot
                assert w == pytest.approx(old[ref]["share_vs_own"], abs=1e-12)
            assert cell["cell_R_loc_class"] == o6.size_class(old["R_loc"]["share_vs_own_ci"])
    assert n_written > 0


def test_count_all_gate_equals_v6o_where_persons_agree(run):
    rec, rn, ro, _ = run
    n = 0
    for c in p6.COVARIATES:
        for nm, how, st, row, cell in cells(rec, c):
            if not cell.get("written"):
                continue
            g0 = rn["count_1"][nm][how][row]["R_loc"]["persons"]
            if sum(cell["persons"].values()) != g0:
                continue
            for ref in p6.REFS:
                oc = ro["groups"][c][nm][how]["cells"][ref][row]
                if not oc["written"]:
                    continue
                for g in p6.GROUPS:
                    n += 1
                    e, og = cell["groups"][g][ref], oc["groups"][g]
                    assert (e["share_vs_own"], e["share_vs_own_ci"], e["class"]) == (
                        og["share_vs_own"], og["share_vs_own_ci"], og["class"])
    assert n > 0


def test_count_all_states_follow_the_table(run):
    rec, _, _, _ = run
    for c in p6.COVARIATES:
        for nm, how, st, row, cell in cells(rec, c):
            if not cell.get("written"):
                continue
            for g in p6.GROUPS:
                e = cell["groups"][g]
                assert e["state"] == p6.gate_state(cell["cell_R_loc_class"], e["R_loc"]["class"], e["R_mean"]["class"])
                assert e["class"] == o6.size_class(e["share_vs_own_ci"])
                lo, hi = e["stage_minus_R_loc"]["ci"]
                assert lo <= e["stage_minus_R_loc"]["difference"] + 1e-12 and hi >= e["stage_minus_R_loc"]["difference"] - 1e-12
                assert e["stage_minus_R_loc"]["difference"] == pytest.approx(e["share_vs_own"] - e["R_loc"]["share_vs_own"])
                assert e["stage_minus_R_mean"]["difference"] == pytest.approx(
                    e["share_vs_own"] - e["R_mean"]["share_vs_own"])


def test_count_all_paired_uses_the_stages_resamples(run, monkeypatch):
    # recompute one written group's paired interval by hand from the stage's key
    rec, rn, ro, args = run
    found = None
    for nm, how, st, row, cell in cells(rec):
        if cell.get("written") and st == "I":
            found = (nm, how, st, row, cell)
            break
    assert found is not None
    nm, how, st, row, cell = found
    ks_, kid, kp, ls_, lid, lp = args
    ds, task = ("kupa", "essay") if nm == "kupa_essay" else ("lwp", "lwp")
    series, ids, place = (ks_, kid, kp) if ds == "kupa" else (ls_, lid, lp)
    b = n6.build(series, ids, ds, task, place, M, o6.BUILD_ROWS)
    hv = n6.hv_of(b, o6.BUILD_ROWS)
    rule = p6.rule_keep(hv, ds, task, how, row, st, rn)
    cov = [o6.covariates_of(*(series[pid][task] if ds == "kupa" else series[pid])) for pid in b["persons"]]
    keep = o6.set_keep(hv, cov, ds, task, rn)
    lab = o6.labels_of(keep)
    gate0 = o6.cell_persons(hv, how, o6.ROWS_O[0], "R")
    g = p6.group_of_stage(rule, gate0, lab[how], "n_s")
    for j, gn in enumerate(p6.GROUPS):
        sel = g == j
        idx = np.random.default_rng(p6.skey(ds, task, st, row, how, j)).integers(0, sel.sum(), (v6.N_BOOT, sel.sum()))
        d = rule["stage"][sel] - rule["R_loc"][sel]
        assert cell["groups"][gn]["stage_minus_R_loc"]["ci"] == v6.ci(d, idx)
        assert cell["groups"][gn]["share_vs_own_ci"] == v6.ci(rule["stage"][sel], idx)


def test_count_all_stops_on_a_changed_v6n_rule_cell(run):
    _, rn, ro, args = run
    for part, field, val in (("stage", "share_vs_own", 0.123), ("R_loc", "share_vs_own_ci", [0.0, 1.0]),
                             ("R_mean", "persons", 999)):
        bad = copy.deepcopy(rn)
        bad["rule_persons"]["lwp_lwp"]["halves"]["cross_75"]["I"][part][field] = val
        with pytest.raises(SystemExit, match=f"lwp_lwp halves cross_75 I: the rule cell's {part}"):
            p6.count_all(*args, M, bad, ro)
    bad = copy.deepcopy(rn)
    bad["rule_persons"]["kupa_essay"]["oddeven"]["inside_75"]["Bsd@2000"]["paired_minus_R_loc"]["difference"] = 0.5
    with pytest.raises(SystemExit, match="paired difference"):
        p6.count_all(*args, M, bad, ro)
    bad = copy.deepcopy(rn)
    del bad["rule_persons"]["kupa_essay"]["oddeven"]["inside_75"]["U_med"]
    with pytest.raises(SystemExit, match="not in v6n's rule cells"):
        p6.count_all(*args, M, bad, ro)


def test_count_all_stops_on_changed_v6o_groups(run):
    _, rn, ro, args = run
    bad = copy.deepcopy(ro)
    bad["groups"]["n_s"]["lwp_lwp"]["halves"]["cuts"][0] += 0.25
    with pytest.raises(SystemExit, match="cuts or groups differ"):
        p6.count_all(*args, M, rn, bad)
    bad = copy.deepcopy(ro)
    g = bad["groups"]["n_s_min"]["kupa_essay"]["oddeven"]["persons"]
    g["low"] += 1
    with pytest.raises(SystemExit, match="cuts or groups differ"):
        p6.count_all(*args, M, rn, bad)
    bad = copy.deepcopy(ro)
    bad["written"]["oddeven"]["kupa_essay"]["n_s"] = not bad["written"]["oddeven"]["kupa_essay"]["n_s"]
    with pytest.raises(SystemExit, match="written plan"):
        p6.count_all(*args, M, rn, bad)


def test_count_all_stops_on_a_changed_v6o_gate_share(run):
    rec, rn, ro, args = run
    for nm, how, st, row, cell in cells(rec):
        if cell.get("written") and ro["groups"]["n_s"][nm][how]["cells"]["R_mean"][row]["written"] and \
                sum(cell["persons"].values()) == rn["count_1"][nm][how][row]["R_loc"]["persons"]:
            for ref in p6.REFS:
                bad = copy.deepcopy(ro)
                bad["groups"]["n_s"][nm][how]["cells"][ref][row]["groups"]["mid"]["share_vs_own_ci"][0] += 1e-9
                with pytest.raises(SystemExit, match=f"{ref} differs from v6o's group"):
                    p6.count_all(*args, M, rn, bad)
            return
    pytest.fail("no written cell with v6o's persons")


def test_count_all_stops_on_count_persons(run):
    _, rn, ro, args = run
    bad = copy.deepcopy(rn)
    bad["tallies"]["kupa_essay"]["count_persons"] += 1
    with pytest.raises(SystemExit, match="count persons"):
        p6.count_all(*args, M, bad, ro)


def hand_rule():
    rng = np.random.default_rng(5)
    return {"persons": list(range(15)), "cell_R_loc_class": "above",
            "stage": np.array([1, 0, 0.5, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0.5, 1.0]),
            "R_loc": np.array([1.0, 1, 1, 1, 1, 1, 0, 1, 1, 0.5, 0, 0, 1, 0, 0]) * 0.9 + rng.random(15) * 0.1,
            "R_mean": rng.random(15)}  # continuous trial values, so that a wrong key moves the interval's ends


def test_group_stats_by_hand():
    rule = hand_rule()
    g = np.repeat([0, 1, 2], 5)
    out = p6.stage_groups(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", "written", False)
    assert out["plan"] == "written" and out["written"]
    assert out["groups"]["low"]["share_vs_own"] == pytest.approx(0.5)
    assert out["groups"]["mid"]["share_vs_own"] == pytest.approx(0.8)
    assert out["groups"]["high"]["share_vs_own"] == pytest.approx(0.3)
    assert out["high_minus_low"]["difference"] == pytest.approx(-0.2)
    assert out["rising"] is False
    for gn, sel in zip(p6.GROUPS, (slice(0, 5), slice(5, 10), slice(10, 15))):
        assert out["groups"][gn]["R_loc"]["share_vs_own"] == pytest.approx(rule["R_loc"][sel].mean())
    for c, off in (("n_s", 0), ("n_s_min", 10)):
        out = p6.stage_groups(rule, g, "kupa", "essay", "I", "cross_100", "halves", c, "written", False)
        for j, gn in enumerate(p6.GROUPS):
            sel = g == j
            idx = np.random.default_rng(p6.skey("kupa", "essay", "I", "cross_100", "halves", off + j)).integers(
                0, 5, (v6.N_BOOT, 5))
            e = out["groups"][gn]
            assert e["share_vs_own_ci"] == v6.ci(rule["stage"][sel], idx)
            assert e["stage_minus_R_loc"]["ci"] == v6.ci(rule["stage"][sel] - rule["R_loc"][sel], idx)
            assert e["stage_minus_R_mean"]["ci"] == v6.ci(rule["stage"][sel] - rule["R_mean"][sel], idx)
            for ref, last in (("R_loc", 9), ("R_mean", 13)):
                rk = p6.skey("kupa", "essay", "I", "cross_100", "halves", off + j, last)
                ridx = np.random.default_rng(rk).integers(0, 5, (v6.N_BOOT, 5))
                assert e[ref]["share_vs_own_ci"] == v6.ci(rule[ref][sel], ridx)
            assert e["state"] == p6.gate_state("above", e["R_loc"]["class"], e["R_mean"]["class"])
        m_hi = rule["stage"][10:][np.random.default_rng(p6.skey("kupa", "essay", "I", "cross_100", "halves", off + 2))
                                  .integers(0, 5, (v6.N_BOOT, 5))].mean(axis=1)
        m_lo = rule["stage"][:5][np.random.default_rng(p6.skey("kupa", "essay", "I", "cross_100", "halves", off))
                                 .integers(0, 5, (v6.N_BOOT, 5))].mean(axis=1)
        assert out["high_minus_low"]["ci"] == o6.pct(m_hi - m_lo)
    # with v6o's persons the references use v6o's keys
    out = p6.stage_groups(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", "written", True)
    for ref in p6.REFS:
        k = o6.gkey("kupa", "essay", ref, "cross_100", "halves", 1)
        ridx = np.random.default_rng(k).integers(0, 5, (v6.N_BOOT, 5))
        assert out["groups"]["mid"][ref]["share_vs_own_ci"] == v6.ci(rule[ref][5:10], ridx)


def test_group_stats_withheld_and_small():
    rule = hand_rule()
    g = np.repeat([0, 1, 2], 5)
    for why in ("derivable", "group under 5", "v6o withheld"):
        out = p6.stage_groups(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", why, False)
        assert out == {"persons": {"low": 5, "mid": 5, "high": 5}, "written": False, "plan": why}
    g4 = np.array([0] * 4 + [1] * 6 + [2] * 5)
    with pytest.raises(SystemExit, match="under 5"):
        p6.stage_groups(rule, g4, "kupa", "essay", "I", "cross_100", "halves", "n_s", "written", False)


def test_group_stats_weighted_guard():
    rule = hand_rule()
    g = np.repeat([0, 1, 2], 5)
    bad = dict(rule)
    bad["persons"] = list(range(16))
    with pytest.raises(SystemExit, match="do not give"):
        p6.stage_groups(bad, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", "written", False)


def test_check_gate_v6o_stops_and_passes():
    rule = hand_rule()
    g = np.repeat([0, 1, 2], 5)
    oc = {}
    for ref in p6.REFS:
        groups = {}
        for j, gn in enumerate(p6.GROUPS):
            r = rule[ref][g == j]
            ridx = np.random.default_rng(o6.gkey("kupa", "essay", ref, "cross_100", "halves", j)).integers(
                0, 5, (v6.N_BOOT, 5))
            ci = v6.ci(r, ridx)
            groups[gn] = {"persons": 5, "share_vs_own": float(r.mean()), "share_vs_own_ci": ci,
                          "class": o6.size_class(ci)}
        oc[ref] = {"written": True, "groups": groups}
    p6.check_gate_v6o(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", oc)
    for ref in p6.REFS:
        bad = copy.deepcopy(oc)
        bad[ref]["groups"]["high"]["share_vs_own"] += 1e-9
        with pytest.raises(SystemExit, match=f"{ref} differs"):
            p6.check_gate_v6o(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", bad)
    bad = copy.deepcopy(oc)
    bad["R_mean"]["written"] = False
    with pytest.raises(SystemExit, match="not written"):
        p6.check_gate_v6o(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", bad)


def test_group_of_stage_needs_subset():
    lab = {"n_s": (0.0, 1.0, np.array([0, 1, 2, 0, 1]))}
    assert list(p6.group_of_stage({"persons": [0, 2, 4]}, [0, 1, 2, 3, 4], lab, "n_s")) == [0, 2, 1]
    with pytest.raises(SystemExit, match="not among"):
        p6.group_of_stage({"persons": [0, 5]}, [0, 1, 2, 3, 4], lab, "n_s")


# ---------------------------------------------------------------- predictions and counts


def pred_rec(states, classes, upper, hl_side="includes 0", written=True):
    """A record where every primary group has the given state and class, stage - R_loc's upper end and high-low."""
    hl_ci = {"includes 0": [-0.1, 0.1], "above 0": [0.1, 0.3], "below 0": [-0.3, -0.1]}[hl_side]
    rec: dict = {"groups": {"n_s": {}}}
    for nm in ("kupa_essay", "lwp_lwp"):
        rec["groups"]["n_s"][nm] = {}
        for how in v6b.BASELINE:
            rec["groups"]["n_s"][nm][how] = {}
            for st in p6.PRIMARY_STAGES:
                rec["groups"]["n_s"][nm][how][st] = {}
                for row in p6.PRIMARY_ROWS:
                    if not written:
                        rec["groups"]["n_s"][nm][how][st][row] = {"written": False, "persons": {}}
                        continue
                    rec["groups"]["n_s"][nm][how][st][row] = {
                        "written": True, "high_minus_low": {"ci": hl_ci},
                        "groups": {g: {"state": states[g], "class": classes[g],
                                       "stage_minus_R_loc": {"ci": [-0.5, upper]}} for g in p6.GROUPS}}
    return rec


def test_predictions_held():
    st = {"low": "R_mean only", "mid": "gate", "high": "gate"}
    cl = {"low": "below", "mid": "below", "high": "undecided"}
    out = p6.predictions(pred_rec(st, cl, -0.01))
    assert out["P1"].startswith("held") and "oddeven gate undecided" in out["P1"]
    assert out["P2"].startswith("held") and "(48 of 48 groups read)" in out["P2"]
    assert out["P3"].startswith("held") and "(12 of 12)" in out["P3"]
    assert out["P4"].startswith("did not hold")  # KUPA's high group is 'gate', not 'gate in group only'


def test_predictions_did_not_hold_and_not_evaluable():
    st = {"low": "gate in group only", "mid": "gate", "high": "gate in group only"}
    cl = {"low": "below", "mid": "below", "high": "above"}
    out = p6.predictions(pred_rec(st, cl, 0.01, hl_side="above 0"))
    assert out["P1"].startswith("did not hold") and out["P4"].startswith("did not hold")
    assert out["P2"].startswith("did not hold") and "(0 of 72 groups read)" in out["P2"]
    assert out["P3"].startswith("did not hold") and "(0 of 12)" in out["P3"]
    cl = {"low": "below", "mid": "below", "high": "undecided"}
    out = p6.predictions(pred_rec(st, cl, -0.01))
    assert out["P4"].startswith("held") and out["P1"].startswith("did not hold")
    out = p6.predictions(pred_rec(st, cl, -0.01, written=False))
    assert out["P1"] == p6.NE and out["P4"] == p6.NE
    assert out["P2"] == p6.NE + " (no group read)" and out["P3"].startswith(p6.NE + " (12 of 12 cells withheld)")
    rec = pred_rec(st, cl, -0.01)
    rec["groups"]["n_s"]["kupa_essay"]["halves"]["I"]["cross_100"] = {"written": False, "persons": {}}
    out = p6.predictions(rec)
    assert out["P2"].startswith(p6.NE + " (1 of 24 cells withheld); over the written: held")
    assert out["P3"].startswith(p6.NE + " (1 of 12 cells withheld); over the written: 11 of 11")


def test_predictions_p2_upper_end_strict():
    st = {g: "gate" for g in p6.GROUPS}
    cl = {g: "below" for g in p6.GROUPS}
    assert p6.predictions(pred_rec(st, cl, 0.0))["P2"].startswith("did not hold")
    st = {g: "not read" for g in p6.GROUPS}
    assert p6.predictions(pred_rec(st, cl, 0.0))["P2"] == p6.NE + " (no group read)"


def test_predictions_p3_boundary():
    st = {g: "gate" for g in p6.GROUPS}
    cl = {g: "below" for g in p6.GROUPS}
    rec = pred_rec(st, cl, -0.01)
    rows = [(nm, how, row) for nm in ("kupa_essay", "lwp_lwp") for how in v6b.BASELINE for row in p6.PRIMARY_ROWS]
    for k, (nm, how, row) in enumerate(rows):
        if k >= 8:
            rec["groups"]["n_s"][nm][how]["I"][row]["high_minus_low"]["ci"] = [0.0, 0.2]  # end at 0: not above
    assert p6.predictions(rec)["P3"].startswith("held (12 of 12)")
    for k, (nm, how, row) in enumerate(rows):
        if k >= 7:
            rec["groups"]["n_s"][nm][how]["I"][row]["high_minus_low"]["ci"] = [0.01, 0.2]
    assert p6.predictions(rec)["P3"] == "did not hold (7 of 12)"


def test_counts_tiny_record():
    def cell(states, classes, hl):
        return {"written": True, "high_minus_low": {"ci": hl},
                "groups": {g: {"state": s_, "class": c_} for g, s_, c_ in zip(p6.GROUPS, states, classes)}}
    rec = {"groups": {"n_s": {"kupa_essay": {}}}}
    for how in v6b.BASELINE:
        rec["groups"]["n_s"]["kupa_essay"][how] = {st: {row: {"written": False, "plan": "derivable", "persons": {}}
                                                        for row in p6.ROWS_O} for st in p6.STAGES}
    t = rec["groups"]["n_s"]["kupa_essay"]["oddeven"]["I"]
    t["cross_100"] = cell(("gate", "gate", "not read"), ("below", "undecided", "above"), [0.1, 0.2])
    t["inside/before"] = cell(("gate",) * 3, ("above",) * 3, [0.1, 0.2])  # not counted
    t["cross_75"] = {"no_count": True, "persons": 1, "written": False}
    out = p6.counts(rec, "n_s")["I"]
    assert out["cells_written"] == 1 and out["no_count"] == 1 and out["families_withheld"] == 2 * 10 - 2
    assert out["states"]["gate"] == {"above": 0, "undecided": 1, "below": 1}
    assert out["states"]["not read"]["above"] == 1 and out["high_minus_low"]["above 0"] == 1


def test_rising_needs_mid_between():
    rule = hand_rule()
    rule["stage"] = np.array([0.0, 0, 0, 0.5, 1] + [1.0] * 4 + [0.5] + [0.5, 0.5, 0.5, 0, 1])  # .3 / .9 / .5
    g = np.repeat([0, 1, 2], 5)
    out = p6.stage_groups(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", "written", False)
    assert out["rising"] is False
    rule["stage"] = np.array([0.0, 0, 0, 0.5, 1] + [0.5] * 5 + [1.0] * 5)
    assert p6.stage_groups(rule, g, "kupa", "essay", "I", "cross_100", "halves", "n_s", "written", False)["rising"]


def test_plan_row_pooled_only():
    # a hand-made ledger: count 1 holds U_med over {0..6} and R_mean over {7..13}; no rule cell sets, no v6o family.
    # The stages on all 21 persons write their groups ({0..6} is I's low group too); B_complete misses 0, its low
    # group {1..6} makes {0} derivable against U_med's set or I's low group: only in the pooled ledger
    n = 21
    gate0 = list(range(n))
    g = np.repeat([0, 1, 2], 7)
    lab = {c: (0.0, 1.0, g) for c in p6.COVARIATES}
    rules = {st: {"persons": list(range(1, n)) if st == "B_complete@2000" else list(range(n))} for st in p6.STAGES}
    base = {"count_1": {"U_med": p6.mask_of(range(7), n), "R_mean": p6.mask_of(range(7, 14), n)}, "rule": {},
            "v6o": []}
    plan, tally = p6.plan_row(base, "oddeven", "cross_100", gate0, lab, {c: True for c in p6.COVARIATES}, rules, n)
    assert plan["I"] == W and tally["stage_before"]["B_complete@2000"] == 0
    assert plan["B_complete@2000"] == {"n_s": "derivable", "n_s_min": "derivable"}
    # with the pooled sets accepted but the earlier stages' families left out of the pool, U_med's set still splits
    base["count_1"].pop("U_med")
    plan, _ = p6.plan_row(base, "oddeven", "cross_100", gate0, lab, {c: True for c in p6.COVARIATES}, rules, n)
    assert plan["B_complete@2000"]["n_s"] == "derivable"  # by I's accepted low group in the pool


def test_plan_row_stage_only():
    # pooled, {0} and {7} are derivable before (U_med and Bsd_complete miss them); B_complete misses both: in its own
    # ledger only {0, 7} was derivable before, and v6o's groups would split it: withheld by the stage's ledger
    plan, tally = plan_case({"U_med": [0], "Bsd_complete@2000": [7], "B_complete@2000": [0, 7]})
    assert plan["B_complete@2000"]["n_s"] == "derivable"
    assert tally["stage_before"]["B_complete@2000"] == 1


def test_check_plan_stage_only_stops():
    missing = {"U_med": [0], "Bsd_complete@2000": [7], "B_complete@2000": [0, 7]}
    n, row, how = 21, "cross_100", "oddeven"
    hv = {how: {"text/before": [], row: []}}
    for i in range(n):
        keys = {k for k in ALL if i not in missing.get(k, [])}
        hv[how]["text/before"].append(vec(keys))
        hv[how][row].append(vec(keys))
    gate0 = list(range(n))
    g = np.repeat([0, 1, 2], 7)
    lab = {c: (0.0, 1.0, g) for c in p6.COVARIATES}
    rn_c1 = {row: {k: {"persons": len(p6.persons_with(hv, how, row, (k,))), "share_vs_own": 0.5} for k in ALL}}
    rules = {st: {"persons": p6.persons_with(hv, how, row, (st, "R"))} for st in p6.STAGES}
    o_fams = [p6.mask_of(np.array(gate0)[g == j], n) for c in p6.COVARIATES for j in range(3)]
    base = p6.base_sets(hv, how, row, gate0, o_fams, rn_c1, rules, n)
    plan, _ = p6.plan_row(base, how, row, gate0, lab, {c: True for c in p6.COVARIATES}, rules, n)
    bad = copy.deepcopy(plan)
    bad["B_complete@2000"]["n_s"] = "written"
    with pytest.raises(SystemExit, match="stage's ledger"):
        p6.check_plan(base, bad, gate0, lab, rules, n)


def test_counts_by_hand(run):
    rec, _, _, _ = run
    for c in p6.COVARIATES:
        got = p6.counts(rec, c)
        for st in p6.STAGES:
            want_states = {s: dict.fromkeys(p6.CLASSES, 0) for s in p6.STATES}
            withheld, written, none = 0, 0, 0
            for nm, how, s2, row, cell in cells(rec, c):
                if s2 != st or row == "inside/before" or cell.get("fixed_by_construction"):
                    continue
                if cell.get("no_count"):
                    none += 1
                    continue
                if not cell["written"]:
                    withheld += 1
                    continue
                written += 1
                for g in p6.GROUPS:
                    want_states[cell["groups"][g]["state"]][cell["groups"][g]["class"]] += 1
            assert got[st]["states"] == want_states
            assert got[st]["families_withheld"] == withheld and got[st]["cells_written"] == written
            assert got[st].get("no_count", 0) == none
            assert sum(got[st]["high_minus_low"].values()) == written


# ---------------------------------------------------------------- rule_keep's stops and the gate check on fixed cells


@pytest.fixture(scope="module")
def hv_small():
    from test_prototype_v6o import fake_hv
    return fake_hv(n=12, rows=("text/before", "cross_100"))


def rn_of(hv, st="I", row="cross_100", how="halves"):
    c = n6.rule_cell(hv, "kupa", "essay", row, how, st)
    return {"rule_persons": {"kupa_essay": {how: {row: {st: n6.jsonish(c)}}}}}


def test_rule_keep_passes_and_checks_the_domain(hv_small, monkeypatch):
    rn = rn_of(hv_small)
    r = p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", rn)
    assert len(r["persons"]) == 12 and set(np.unique(r["stage"])) <= {0.0, 0.5, 1.0}
    real = o6.contrib_base_change

    def off(first, sel, key):
        own, base, change = real(first, sel, key)
        own = own.copy()
        own[0] = 0.25
        return own, base, change

    monkeypatch.setattr(o6, "contrib_base_change", off)
    with pytest.raises(SystemExit, match="not 0, 1/2 or 1"):
        p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", rn)

    def nonfinite(first, sel, key):
        own, base, change = real(first, sel, key)
        base = base.copy()
        base[0] = np.nan
        return own, base, change

    monkeypatch.setattr(o6, "contrib_base_change", nonfinite)
    with pytest.raises(SystemExit, match="non-finite"):
        p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", rn)


def test_rule_keep_stops_when_a_references_persons_differ(hv_small):
    hv = copy.deepcopy(hv_small)
    a, b = hv["halves"]["cross_100"][3]
    a = dict(a)
    a["R_mean"] = None
    hv["halves"]["cross_100"][3] = (a, b)
    rn = {"rule_persons": {"kupa_essay": {"halves": {"cross_100": {"I": {}}}}}}
    with pytest.raises(SystemExit, match="persons of R_mean differ"):
        p6.rule_keep(hv, "kupa", "essay", "halves", "cross_100", "I", rn)


def test_rule_keep_missing_fields_do_not_pass(hv_small):
    rn = rn_of(hv_small)
    for part, field in (("stage", "share_vs_own_ci"), ("R_mean", "share_vs_own"), ("R_loc", "persons")):
        bad = copy.deepcopy(rn)
        del bad["rule_persons"]["kupa_essay"]["halves"]["cross_100"]["I"][part][field]
        with pytest.raises(SystemExit, match=f"rule cell's {part}"):
            p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", bad)


def test_rule_keep_no_count_checks_persons(hv_small, monkeypatch):
    rn = rn_of(hv_small)
    few = {part: {"persons": 12, "too_few_persons": True} for part in ("stage", "R", "R_loc", "R_lag", "R_mean")}
    monkeypatch.setattr(n6, "rule_cell", lambda *a: copy.deepcopy(few))
    old = {part: {"persons": 12} for part in ("stage", "R_loc", "R_mean")}
    ok = {"rule_persons": {"kupa_essay": {"halves": {"cross_100": {"I": old}}}}}
    assert p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", ok)["no_count"]
    bad = copy.deepcopy(ok)
    bad["rule_persons"]["kupa_essay"]["halves"]["cross_100"]["I"]["R_loc"]["persons"] = 11
    with pytest.raises(SystemExit, match="rule cell's R_loc"):
        p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", bad)
    bad = copy.deepcopy(ok)
    bad["rule_persons"]["kupa_essay"]["halves"]["cross_100"]["I"]["stage"]["share_vs_own_ci"] = [0.1, 0.2]
    with pytest.raises(SystemExit, match="rule cell's stage"):
        p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", bad)
    del rn


def test_count_all_checks_v6o_on_fixed_cells(run):
    rec, rn, ro, args = run
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            oc = ro["groups"]["n_s"][nm][how]["cells"]["R_loc"]["inside_100"]
            cell = rec["groups"]["n_s"][nm][how]["U@2000"]["inside_100"]
            if oc["written"] and cell.get("fixed_by_construction") and \
                    cell["persons"] == rn["count_1"][nm][how]["inside_100"]["R_loc"]["persons"]:
                bad = copy.deepcopy(ro)
                bad["groups"]["n_s"][nm][how]["cells"]["R_loc"]["inside_100"]["groups"]["low"]["share_vs_own"] += 1e-9
                with pytest.raises(SystemExit, match="inside_100 U@2000 low: R_loc differs"):
                    p6.count_all(*args, M, rn, bad)
                return
    pytest.fail("no fixed cell with v6o's persons and written family")


def test_count_all_checks_lwps_bands(run):
    _, rn, ro, args = run
    bad = copy.deepcopy(ro)
    bad["bands"]["n_s"]["oddeven"]["cuts_from_kupa_essay"][1] += 0.5
    with pytest.raises(SystemExit, match="KUPA essay's cuts differ from v6o's bands"):
        p6.count_all(*args, M, rn, bad)
    bad = copy.deepcopy(ro)
    bad["bands"]["n_s_min"]["halves"]["persons"]["lwp"]["mid"] += 1
    with pytest.raises(SystemExit, match="band persons"):
        p6.count_all(*args, M, rn, bad)



def test_rule_keep_paired_ci_and_missing_on_both_sides(hv_small, monkeypatch):
    rn = rn_of(hv_small)
    bad = copy.deepcopy(rn)
    bad["rule_persons"]["kupa_essay"]["halves"]["cross_100"]["I"]["paired_minus_R_loc"]["ci"] = [0.1, 0.2]
    with pytest.raises(SystemExit, match="paired difference"):
        p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", bad)
    real = n6.rule_cell

    def no_paired(*a):
        c = real(*a)
        c.pop("paired_minus_R_loc", None)
        return c

    monkeypatch.setattr(n6, "rule_cell", no_paired)
    bad = copy.deepcopy(rn)
    del bad["rule_persons"]["kupa_essay"]["halves"]["cross_100"]["I"]["paired_minus_R_loc"]
    with pytest.raises(SystemExit, match="paired difference"):
        p6.rule_keep(hv_small, "kupa", "essay", "halves", "cross_100", "I", bad)


def test_set_cells_runs_check_plan_and_puts_lwps_bands_in_the_ledger(run, monkeypatch):
    _, rn, ro0, args = run
    ro = copy.deepcopy(ro0)
    for how in v6b.BASELINE:
        for c in p6.COVARIATES:
            ro["written"][how]["bands"][c] = True  # as if v6o had written every band family
    seen = []
    real = p6.plan_row

    def spy(base, how, row, *a):
        seen.append((how, row, len(base["v6o"])))
        return real(base, how, row, *a)

    monkeypatch.setattr(p6, "plan_row", spy)
    p6.count_all(*args, M, rn, ro)
    n_rows = len(p6.ROWS_O)
    for k, nm in enumerate(("kupa_essay", "lwp_lwp")):
        for how in v6b.BASELINE:
            want = 3 * sum(bool(ro["written"][how][nm][c]) for c in p6.COVARIATES)
            if nm == "lwp_lwp":
                want += 3 * sum(bool(ro["written"][how]["bands"][c]) for c in p6.COVARIATES)
            got = {x for (h, r, x) in seen[k * 2 * n_rows:(k + 1) * 2 * n_rows] if h == how}
            assert got == {want}, (nm, how, got, want)
    assert any(ro["written"][how]["bands"][c] for how in v6b.BASELINE for c in p6.COVARIATES)
    monkeypatch.setattr(p6, "plan_row", real)

    def boom(*a):
        raise SystemExit("check_plan ran")

    monkeypatch.setattr(p6, "check_plan", boom)
    with pytest.raises(SystemExit, match="check_plan ran"):
        p6.count_all(*args, M, rn, ro)
