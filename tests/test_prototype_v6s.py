import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6s as s6  # noqa: E402
from test_prototype_v6o import fake_sets  # noqa: E402

r6, q6, p6, o6, n6, k6 = s6.r6, s6.q6, s6.p6, s6.o6, s6.n6, s6.k6
v6 = k6.v6
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
RL = json.loads((ROOT / "results/prototype_v6l/record.json").read_text())
COND = json.loads((ROOT / "works/conditions/prototype_v6s.json").read_text())
I2 = s6.I2


def test_constants():
    assert s6.TAG == 102 and s6.PROTECT == 4 and s6.MIN_SUMMARY == 5
    assert "[20261031, 102, dataset, task, stage, row index, 7, 9, 0]" in COND["cells"]["keys"]
    assert s6.skey("lwp", "lwp", "R_loc", "inside_75") == [20261031, 102, 2, 0, 8, 14, 7, 9, 0]
    assert s6.STAGES_R == r6.STAGES_R and s6.ROWS_R == r6.ROWS_R and s6.READ_FIRST == r6.READ_FIRST


PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def test_dset_equals_enumeration():
    reach = {(0, 0)}
    for d in range(0, 10):
        got = {(x, y) for x in range(-2, 2 * d + 3) for y in range(-2, 2 * d + 3) if s6.dset(d, x, y)}
        assert got == reach, d
        reach = {(x + a, y + b) for x, y in reach for a, b in PAIRS}


def joint_oracle(x2, y2, n_c, d):
    """All assignments of the 9 pairs (half units) to n_c persons, d of them off the diagonal."""
    from itertools import combinations_with_replacement
    cells = [(i, j) for i in range(3) for j in range(3)]
    for combo in combinations_with_replacement(cells, n_c):
        if sum(i != j for i, j in combo) == d and sum(i for i, _ in combo) == x2 and sum(j for _, j in combo) == y2:
            return True
    return False


def test_joint_equals_oracle():
    for n_c in range(0, 5):
        for d in range(0, n_c + 1):
            for x2 in range(0, 2 * n_c + 1):
                for y2 in range(0, 2 * n_c + 1):
                    assert s6.joint(x2, y2, n_c, d) == joint_oracle(x2, y2, n_c, d), (n_c, d, x2, y2)


def test_codex_counterexample_is_caught():
    # n_c = 5, odd/even sum 0 on everyone, halves full sum 1 over 6 persons (1 halves-only), d = 2: the two
    # disagreeing persons have halves >= 1/2 each, so Y >= 1 and the halves-only sum is 0
    h2, m = {"oddeven": 0, "halves": 2}, {"oddeven": 0, "halves": 1}
    assert s6.feasible(h2, m, 5, None, "halves") == {0, 1, 2}
    assert s6.feasible(h2, m, 5, 2, "halves") == {0}
    assert s6.narrows_d(h2, m, 5, 2) and s6.withhold(h2, m, 5, 2, {"oddeven": 0, "halves": 0})


def test_feasible_by_hand():
    # one person halves-only (sum 0, 1/2 or 1), none odd/even-only; common 10; full sums (twice) 8 and 8
    h2, m = {"oddeven": 8, "halves": 8}, {"oddeven": 0, "halves": 1}
    assert s6.feasible(h2, m, 10, None, "halves") == {0, 1, 2}
    # d = 0 forces X = Y: X2 = 8 and Y2 = 8 - s, so s = 0
    assert s6.feasible(h2, m, 10, 0, "halves") == {0}
    # d = 1: X2 - Y2 = s in {-2, -1, 1, 2}, so s in {1, 2}
    assert s6.feasible(h2, m, 10, 1, "halves") == {1, 2}
    assert s6.feasible(h2, m, 10, 2, "halves") == {0, 1, 2}
    # the range of Y: n_c = 3 means Y2 <= 6, so with H_h2 = 8, s >= 2
    assert s6.feasible({"oddeven": 6, "halves": 8}, {"oddeven": 0, "halves": 2}, 3, None, "halves") == {2, 3, 4}


def test_withhold_by_hand():
    h2, m = {"oddeven": 8, "halves": 8}, {"oddeven": 0, "halves": 1}
    # d = 0 narrows; the union over W = {0, 1} (d = 1 gives {1, 2}) is {0, 1, 2}: not narrower, so d in W decides
    assert s6.withhold(h2, m, 10, 0, {"oddeven": 0, "halves": 0})
    assert s6.withhold(h2, m, 10, 1, {"oddeven": 0, "halves": 2})
    assert not s6.withhold(h2, m, 10, 2, {"oddeven": 0, "halves": 2})
    # an odd/even-only set of 5 is not protected
    assert not s6.withhold(h2, {"oddeven": 5, "halves": 0}, 10, 0, {"oddeven": 0, "halves": 0})
    with pytest.raises(SystemExit, match="true one-split-only sum"):
        s6.withhold(h2, m, 10, 0, {"oddeven": 0, "halves": 2})  # s = 2 is not feasible with d = 0
    h2b, mb = {"oddeven": 8, "halves": 8}, {"oddeven": 1, "halves": 1}
    with pytest.raises(SystemExit, match="true one-split-only sum"):
        s6.withhold(h2b, mb, 10, 0, {"oddeven": 0, "halves": 7})


def test_withhold_union_rule_from_code_review():
    """code-reviewer H1: with W = {d that narrow}, if the union over W narrows, withholding itself would leak, so the
    cell is withheld for every d (also for a d that alone does not narrow)."""
    found = False
    for n_c in range(3, 9):
        for ho in range(0, 2 * n_c + 1):
            for hh in range(0, 2 * n_c + 9):
                h2, m = {"oddeven": ho, "halves": hh}, {"oddeven": 0, "halves": 4}
                without = s6.pairs(h2, m, n_c, None)
                if not without:
                    continue
                w = [d for d in range(n_c + 1) if s6.narrows_d(h2, m, n_c, d)]
                if not w:
                    continue
                union = set().union(*(s6.pairs(h2, m, n_c, d) for d in w))
                if s6.view(union, m) == s6.view(without, m):
                    continue
                for d in range(n_c + 1):
                    ps = s6.pairs(h2, m, n_c, d)
                    if ps and d not in w:
                        t = sorted(ps)[0]
                        assert s6.withhold(h2, m, n_c, d, {"oddeven": t[0], "halves": t[1]})
                        found = True
    assert found


def test_view_and_odd_even_side_and_m4():
    # odd/even-only set of 1 narrowed by d = 0
    h2, m = {"oddeven": 8, "halves": 8}, {"oddeven": 1, "halves": 0}
    assert s6.narrows_d(h2, m, 10, 0) and s6.feasible(h2, m, 10, 0, "oddeven") == {0}
    # a protected set of 4 counts, one of 5 does not
    assert s6.narrows_d(h2, {"oddeven": 0, "halves": 4}, 10, 0)
    assert not s6.narrows_d(h2, {"oddeven": 0, "halves": 5}, 10, 0)
    # both protected: compared as pairs
    assert s6.view({(0, 1), (1, 0)}, {"oddeven": 1, "halves": 1}) == (frozenset({(0, 1), (1, 0)}),)
    assert s6.view({(0, 1), (1, 0)}, {"oddeven": 1, "halves": 6}) == (frozenset({0, 1}),)
    # the range of X: n_c = 3 means X2 <= 6, so with H_oe2 = 8 the odd/even-only sum (twice) is at least 2
    assert s6.feasible({"oddeven": 8, "halves": 3}, {"oddeven": 2, "halves": 0}, 3, None, "oddeven") == {2, 3, 4}


def test_half2():
    assert s6.half2(3.5, "w") == 7 and s6.half2(0.0, "w") == 0
    with pytest.raises(SystemExit, match="multiple of 1/2"):
        s6.half2(1.3, "w")


A = np.array([1, 1, 1, 0, 0, 0.5, 1, 0, 1, 0, 1, 1, 0.5, 0, 0, 0.5], dtype=float)
B = np.array([1, 0, 1, 0, 1, 0.5, 1, 0, 0, 0, 1, 0.5, 1, 0, 0.5, 0], dtype=float)


def test_agreement_po_interval_depends_on_counts_only():
    rng = np.random.default_rng(7)
    e = s6.agreement_po(A, B, [3, 4])
    for _ in range(5):
        perm = rng.permutation(len(A))
        f = s6.agreement_po(A[perm], B[perm], [3, 4])
        assert f == e
    # a different pair with the same counts gives the same record
    a2 = np.array([0.5] * 9 + [1.0] * 7)
    b2 = np.array([0.5] * 9 + [0.0] * 7)
    assert s6.agreement_po(a2, b2, [3, 4]) == e


def test_agreement_po_by_hand():
    e = s6.agreement_po(A, B, [3, 4])
    n = len(A)
    agree = int(np.sum(A == B))
    assert e["persons_agree"] == agree == 9 and e["persons_disagree"] == n - agree
    assert e["p_o"] == agree / n and e["one_minus_p_o"] == 1 - agree / n
    idx = np.random.default_rng([3, 4]).integers(0, n, (v6.N_BOOT, n))
    canon = np.array([1.0] * agree + [0.0] * (n - agree))
    ci = v6.ci(canon, idx)
    assert e["p_o_ci"] == ci and e["one_minus_p_o_ci"] == [1 - ci[1], 1 - ci[0]]
    assert set(e) == {"persons_agree", "persons_disagree", "p_o", "p_o_ci", "one_minus_p_o", "one_minus_p_o_ci"}
    with pytest.raises(SystemExit, match="same persons"):
        s6.agreement_po(A, B[:-1], [1])
    with pytest.raises(SystemExit, match="not 0, 1/2 or 1"):
        s6.agreement_po(np.array([0.2, 1.0]), np.array([0.0, 1.0]), [1])


# ---------------------------------------------------------------- the fake run


@pytest.fixture(scope="module")
def run():
    ks_, ls_ = fake_sets()
    kid, lid = sorted(ks_), sorted(ls_)
    kp, lp = {p: i for i, p in enumerate(kid)}, {p: i for i, p in enumerate(lid)}
    rn = n6.jsonish(n6.count_all(ks_, kid, kp, ls_, lid, lp, M, RL, n_splits=3))
    ro = n6.jsonish(o6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn))
    rp = n6.jsonish(p6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro))
    rq = n6.jsonish(q6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro, rp))
    rr = n6.jsonish(r6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro, rp, rq))
    rec = n6.jsonish(s6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, rp, rq, rr))
    return rec, rn, rp, rq, rr, (ks_, kid, kp, ls_, lid, lp)


def rerun(run, **ch):
    rec, rn, rp, rq, rr, (ks_, kid, kp, ls_, lid, lp) = run
    a = {"rn": rn, "rp": rp, "rq": rq, "rr": rr, **ch}
    return s6.count_all(ks_, kid, kp, ls_, lid, lp, M, a["rn"], a["rp"], a["rq"], a["rr"])


def cells(rec):
    for nm in rec["cells"]:
        for row in s6.ROWS_R:
            for st in s6.STAGES_R:
                yield nm, row, st, rec["cells"][nm][row][st]


def test_structure_no_per_person(run):
    rec = run[0]
    out = {"main_cells": s6.main_cells_out(rec), **rec}
    k6.v6d.no_per_person(out)
    k6.v6i.no_long_list(out)
    base = {"persons_common", "persons_rule", "persons_one_split_only"}
    w = base | {"persons_agree", "persons_disagree", "p_o", "p_o_ci", "one_minus_p_o", "one_minus_p_o_ci",
                "class_own_rule_persons", "state"}
    for nm, row, st, e in cells(rec):
        if "p_o" in e:
            assert set(e) == w
        else:
            assert set(e) - base <= {"fixed_by_construction", "no_count", "withheld"} and len(set(e) - base) == 1


def test_written_cells_equal_an_independent_recount(run):
    rec, rn, rp, rq, rr, (ks_, kid, kp, ls_, lid, lp) = run
    seen = 0
    for nm in rec["cells"]:
        ds, task = nm.split("_")
        series, ids, place = (ks_, kid, kp) if ds == "kupa" else (ls_, lid, lp)
        hv = q6.with_i2(n6.hv_of(n6.build(series, ids, ds, task, place, M, o6.BUILD_ROWS), o6.BUILD_ROWS))
        for row in s6.ROWS_R:
            for st in s6.STAGES_R:
                e = rec["cells"][nm][row][st]
                if "p_o" not in e:
                    continue
                src = "I" if st in (I2, "R_loc", "R_mean") else st
                own = {}
                for how in s6.SPLITS:
                    ok = p6.persons_with(hv, how, row, (src, "R"))
                    first = [hv[how]["text/before"][i] for i in ok]
                    sel = [hv[how][row][i] for i in ok]
                    own[how] = dict(zip(ok, r6.m6.own_contrib(first, sel, st)))
                common = sorted(set(own["oddeven"]) & set(own["halves"]))
                a = np.array([own["oddeven"][p] for p in common])
                b = np.array([own["halves"][p] for p in common])
                assert e["persons_common"] == len(common) and e["persons_agree"] == int(np.sum(a == b))
                assert e == {**e, **s6.agreement_po(a, b, s6.skey(ds, task, st, row))}
                seen += 1
    assert seen > 40


def test_fake_run_writes_where_nobody_is_one_split_only(run):
    rec = run[0]
    for nm, row, st, e in cells(rec):
        if not (e.get("fixed_by_construction") or e.get("no_count")):
            assert sum(e["persons_one_split_only"].values()) == 0 and "p_o" in e


def test_withheld_when_d_narrows(run, monkeypatch):
    real = r6.verdicts

    def drop1(*a, **k):
        v = real(*a, **k)
        if a[4] == "oddeven":
            for st in s6.STAGES_R:
                s = v["stages"][st]
                if not s.get("no_count"):
                    v["stages"][st] = {**s, "persons": s["persons"][1:], "own": s["own"][1:]}
        return v

    monkeypatch.setattr(r6, "verdicts", drop1)
    rr = copy.deepcopy(run[4])
    for nm in rr["cells"]:
        for row in s6.ROWS_R:
            for st, e in rr["cells"][nm][row].items():
                e["persons_common"] -= 1
                e["persons_rule"]["oddeven"] -= 1
                e["persons_one_split_only"]["halves"] += 1
    monkeypatch.setattr(s6, "full_sum", lambda v, *a: float(v["own"].sum()))
    calls = []
    real_n = s6.withhold

    def spy(h2, m, n_c, d, true2):
        out = real_n(h2, m, n_c, d, true2)
        calls.append((m["halves"], d, out))
        return out

    monkeypatch.setattr(s6, "withhold", spy)
    rec = rerun(run, rr=rr)
    assert calls and all(mh == 1 for mh, _, _ in calls)
    for nm, row, st, e in cells(rec):
        if e.get("fixed_by_construction") or e.get("no_count"):
            continue
        if "p_o" not in e:
            assert e["withheld"] == "narrows a one-split-only sum"
            for k in ("p_o", "p_o_ci", "persons_agree", "persons_disagree", "one_minus_p_o", "one_minus_p_o_ci",
                      "class_own_rule_persons", "state"):
                assert k not in e
    n_w = sum(e.get("withheld") == "narrows a one-split-only sum" for _, _, _, e in cells(rec))
    n_narrow = sum(out for _, _, out in calls)
    assert n_w == n_narrow and n_w > 0


def test_stops_when_common_differs_from_v6r(run):
    rr = copy.deepcopy(run[4])
    rr["cells"]["lwp_lwp"]["cross_100"]["I"]["persons_common"] += 1
    with pytest.raises(SystemExit, match="v6r's record"):
        rerun(run, rr=rr)


def test_stops_when_full_sum_differs_from_v6n(run):
    rn = copy.deepcopy(run[1])
    # change a share by exactly 1/2 person: rule_keep's check is on the share too, so stop either way
    e = rn["rule_persons"]["kupa_essay"]["halves"]["cross_100"]["Bsd@2000"]["stage"]
    e["share_vs_own"] += 0.5 / e["persons"]
    with pytest.raises(SystemExit, match="differs from v6n"):
        rerun(run, rn=rn)


def test_full_sum_by_hand():
    v = {"own": np.array([1, 0, 0.5, 1.0])}
    rn = {"rule_persons": {"x": {"h": {"r": {"I": {"stage": {"share_vs_own": 0.625}, "R_loc": {"share_vs_own": 0.625}}}}}}}
    assert s6.full_sum(v, rn, {}, "x", "h", "r", "I") == 2.5
    assert s6.full_sum(v, rn, {}, "x", "h", "r", "R_loc") == 2.5
    rq = {"cells": {"x": {"h": {"r": {"rule": {I2: {"share_vs_own": 0.625}}}}}}}
    assert s6.full_sum(v, {}, rq, "x", "h", "r", I2) == 2.5
    rn["rule_persons"]["x"]["h"]["r"]["I"]["R_loc"]["share_vs_own"] = 0.5
    with pytest.raises(SystemExit, match="share x persons"):
        s6.full_sum(v, rn, {}, "x", "h", "r", "R_loc")


def test_same_row_is_called(run):
    rp = copy.deepcopy(run[2])
    rp["sets"]["lwp_lwp"]["written"]["oddeven"]["inside/before"]["I"] = {"n_s": "x"}
    with pytest.raises(SystemExit, match="v6p's plans"):
        rerun(run, rp=rp)


# ---------------------------------------------------------------- predictions and summaries


def cellw(agree, n=100):
    return {"persons_common": n, "persons_agree": agree, "persons_disagree": n - agree, "p_o": agree / n,
            "one_minus_p_o": 1 - agree / n}


def fake_rec(over=None):
    c = {nm: {row: {st: cellw(90) for st in s6.STAGES_R} for row in s6.ROWS_R} for nm in ("kupa_essay", "lwp_lwp")}
    for (nm, row, st), e in (over or {}).items():
        c[nm][row][st] = e
    return {"cells": c}


def test_predictions_boundaries():
    rec = fake_rec({("lwp_lwp", "inside_75", "R_loc"): cellw(3, 4), ("lwp_lwp", "inside_100", "R_loc"): cellw(75),
                    ("lwp_lwp", "inside_75", "I"): cellw(79), ("lwp_lwp", "inside_100", "I"): cellw(4, 5)})
    p = s6.predictions(rec)
    assert p["P1"].startswith("held") and p["P3"].startswith("did not hold") and p["P2"].startswith("did not hold")
    rec = fake_rec({("lwp_lwp", "inside_100", "R_loc"): cellw(74), ("lwp_lwp", "inside_75", "I"): cellw(79),
                    ("lwp_lwp", "inside_100", "I"): cellw(79)})
    p = s6.predictions(rec)
    assert p["P1"].startswith("did not hold") and p["P3"].startswith("held")
    over = {(nm, row, st): cellw(4, 5) for nm in ("kupa_essay", "lwp_lwp") for row in s6.PRIMARY_ROWS for st in s6.STAGES_R}
    assert s6.predictions(fake_rec(over))["P2"] == "held (42 of 42)"
    k = 0
    for key in over:
        over[key] = cellw(80 if k % 2 == 0 else 81)
        k += 1
    assert s6.predictions(fake_rec(over))["P2"] == "held (21 of 42)"
    over[("kupa_essay", "inside_75", "U@2000")] = {"persons_common": 9, "fixed_by_construction": True}
    assert s6.predictions(fake_rec(over))["P2"] == "did not hold (20 of 41)"


def test_predictions_not_evaluable():
    nw = {"persons_common": 50, "withheld": "narrows a one-split-only sum"}
    p = s6.predictions(fake_rec({("lwp_lwp", "inside_75", "R_loc"): nw, ("lwp_lwp", "inside_100", "I"): nw}))
    assert p["P1"] == p["P3"] == s6.NE and "P4" not in p
    over = {(nm, row, st): nw for nm in ("kupa_essay", "lwp_lwp") for row in s6.PRIMARY_ROWS for st in s6.STAGES_R}
    assert s6.predictions(fake_rec(over))["P2"] == s6.NE


def test_summary_and_short():
    rec = fake_rec({("kupa_essay", "inside_75", "U@2000"): {"persons_common": 9, "fixed_by_construction": True},
                    ("kupa_essay", "inside_75", "B@2000"): {"persons_common": 9, "withheld": "narrows a one-split-only sum"},
                    ("kupa_essay", "inside_100", "I"): cellw(60),
                    ("lwp_lwp", "shuffled/before", "I"): {"persons_common": 3, "withheld": "fewer than 5"}})
    s = s6.summary(rec)
    assert s["primary"]["written"] == 40 and s["primary"]["fixed"] == 1 and s["primary"]["withheld_narrows"] == 1
    assert s["primary"]["one_minus_p_o"] == {"cells": 40, "min": 0.1, "median": 0.1, "max": 0.4}
    assert s["shuffles"]["withheld_fewer_than_5"] == 1 and s["shuffles"]["written"] == 27
    assert s6.short({"persons_common": 9, "fixed_by_construction": True}) == "not written (fixed; common 9)"
