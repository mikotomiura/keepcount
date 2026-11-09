import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6r as r6  # noqa: E402
from test_prototype_v6o import fake_sets  # noqa: E402

q6, p6, o6, n6, k6 = r6.q6, r6.p6, r6.o6, r6.n6, r6.k6
v6, v6b = k6.v6, k6.v6b
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
RL = json.loads((ROOT / "results/prototype_v6l/record.json").read_text())
COND = json.loads((ROOT / "works/conditions/prototype_v6r.json").read_text())
I2 = r6.I2


# ---------------------------------------------------------------- constants and small functions


def test_constants_follow_the_conditions():
    assert r6.TAG == 101 and r6.PAIR == 7 and r6.CELL == 9
    assert "[20261031, 101, dataset, task, stage, row index, 7, 9, 0]" in COND["cells"]["keys"]
    assert "U 2, B 3, Bsd 4, I 5, R_loc 8, R_mean 12" in COND["cells"]["keys"] and "22 for I[2]" in COND["cells"]["keys"]
    assert r6.STAGE_IX == {"U@2000": 2, "B@2000": 3, "Bsd@2000": 4, "I": 5, "R_loc": 8, "R_mean": 12, I2: 22}
    for st in ("U@2000", "B@2000", "Bsd@2000", "I"):
        assert r6.STAGE_IX[st] == p6.stage_ix(st)
    assert r6.STAGE_IX["R_loc"] == n6.STAGE["R_loc"] and r6.STAGE_IX["R_mean"] == n6.STAGE["R_mean"]
    assert r6.STAGES_R == ("U@2000", "B@2000", "Bsd@2000", "I", I2, "R_loc", "R_mean")
    assert "order U, B, Bsd, I, I[2], R_loc, R_mean" in COND["written_plan"]["rule"]
    assert r6.ROWS_R == ("inside_75", "inside_100", "cross_100", "shuffled/before", "shuffled_in_halves/before")
    assert r6.PRIMARY_ROWS == ("inside_75", "inside_100", "cross_100")
    assert r6.SHUFFLES == ("shuffled/before", "shuffled_in_halves/before")
    assert r6.READ_FIRST == tuple(("lwp_lwp", row, st) for row in ("inside_75", "inside_100") for st in ("I", I2, "R_loc"))
    assert r6.rkey("lwp", "lwp", I2, "cross_100") == [20261031, 101, 2, 0, 22, n6.ROW_IX["cross_100"], 7, 9, 0]
    assert r6.rkey("kupa", "essay", "R_mean", "inside_75") == [20261031, 101, 0, 1, 12, 14, 7, 9, 0]
    assert r6.MIN_SUMMARY == 5 and r6.CATS == (0.0, 0.5, 1.0)


def test_is_fixed():
    for row in ("inside_75", "inside_100"):
        assert r6.is_fixed("U@2000", row) and r6.is_fixed("B@2000", row)
        assert not r6.is_fixed("Bsd@2000", row) and not r6.is_fixed("I", row)
    assert r6.is_fixed("B@2000", "shuffled_in_halves/before")  # fixed with halves only
    assert not r6.is_fixed("B@2000", "shuffled/before") and not r6.is_fixed("U@2000", "cross_100")
    for st in (I2, "R_loc", "R_mean"):
        assert not any(r6.is_fixed(st, row) for row in r6.ROWS_R)


# continuous trial values are not possible for categories: the contributions are a mix of 0, 1/2 and 1 with unequal
# counts, so that no two categories or marginals coincide by accident
A = np.array([1, 1, 1, 0, 0, 0.5, 1, 0, 1, 0, 1, 1, 0.5, 0, 0, 0.5], dtype=float)
B = np.array([1, 0, 1, 0, 1, 0.5, 1, 0, 0, 0, 1, 0.5, 1, 0, 0.5, 0], dtype=float)


def test_table_by_hand():
    t = r6.table_of(A, B)
    want = np.zeros((3, 3), dtype=int)
    for x, y in zip(A, B):
        want[r6.CATS.index(x), r6.CATS.index(y)] += 1
    assert np.array_equal(t, want) and t.sum() == len(A)
    with pytest.raises(SystemExit, match="not 0, 1/2 or 1"):
        r6.table_of(np.array([0.3, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(SystemExit, match="not 0, 1/2 or 1"):
        r6.table_of(np.array([0.0, 1.0]), np.array([0.0, 0.7]))


def test_agreement_by_hand():
    key = [5, 6, 7]
    e = r6.agreement(A, B, key)
    n = len(A)
    po = float(np.mean(A == B))
    pa = np.array([np.mean(A == x) for x in r6.CATS])
    pb = np.array([np.mean(B == x) for x in r6.CATS])
    pe = float(np.sum(pa * pb))
    assert e["p_o"] == pytest.approx(po, abs=1e-15) and e["one_minus_p_o"] == pytest.approx(1 - po, abs=1e-15)
    assert e["p_e"] == pytest.approx(pe, abs=1e-15)
    assert e["kappa"] == pytest.approx((po - pe) / (1 - pe), abs=1e-15)
    assert e["share"] == {"oddeven": pytest.approx(A.mean()), "halves": pytest.approx(B.mean())}
    assert e["proportions"]["oddeven"] == pytest.approx(pa.tolist()) and e["proportions"]["halves"] == pytest.approx(pb.tolist())
    assert e["1->0"] == int(np.sum((A == 1) & (B == 0))) and e["0->1"] == int(np.sum((A == 0) & (B == 1)))
    assert e["with_1/2"] == int(np.sum((A != B) & ((A == 0.5) | (B == 0.5))))
    assert sum(e["table"].values()) == n and e["table"]["1/2->1/2"] == 1 and e["table"]["1->1/2"] == 1
    idx = np.random.default_rng(key).integers(0, n, (v6.N_BOOT, n))
    assert e["p_o_ci"] == v6.ci((A == B).astype(float), idx)
    kb = []
    for row in idx:
        a, b = A[row], B[row]
        po_, pe_ = np.mean(a == b), sum(np.mean(a == x) * np.mean(b == x) for x in r6.CATS)
        kb.append((po_ - pe_) / (1 - pe_))
    assert e["kappa_ci"] == pytest.approx([np.percentile(kb, 2.5), np.percentile(kb, 97.5)], abs=1e-12)


def test_agreement_with_swapped_splits_flips_the_directions():
    e, f = r6.agreement(A, B, [1]), r6.agreement(B, A, [1])
    assert e["1->0"] == f["0->1"] and e["0->1"] == f["1->0"] and e["p_o"] == f["p_o"] and e["p_e"] == pytest.approx(f["p_e"])


def test_agreement_p_e_one_and_undefined_resamples():
    a = np.ones(8)
    e = r6.agreement(a, a.copy(), [1])
    assert e["p_e"] == 1.0 and e["kappa_not_written"] == "p_e = 1" and "kappa" not in e and e["p_o_ci"] == [1.0, 1.0]
    # one person differs: p_e < 1 in the sample, but some resamples leave everyone at 1 in both splits
    a = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    e = r6.agreement(a, b, [2])
    assert "kappa" in e and "kappa_ci" not in e and e["kappa_resamples_p_e_1"] > 0


def test_agreement_stops():
    with pytest.raises(SystemExit, match="same persons"):
        r6.agreement(A, B[:-1], [1])


def test_identity_is_checked(monkeypatch):
    monkeypatch.setattr(r6, "ID_TOL", -1.0)
    with pytest.raises(SystemExit, match="identity"):
        r6.agreement(A, B, [1])


def test_table_checks_stop(monkeypatch):
    real = r6.table_of

    def shifted(a, b):
        t = real(a, b).copy()
        t[2, 0] += 1
        t[2, 2] -= 1
        return t

    monkeypatch.setattr(r6, "table_of", shifted)
    with pytest.raises(SystemExit, match="differs from the table"):
        r6.agreement(A, B, [1])


def test_new_small_and_admit_common():
    n = 30
    m = p6.mask_of
    assert not r6.new_small([m(range(20), n)], [m(range(12), n)])
    assert r6.new_small([m(range(20), n)], [m(range(17), n)])  # 3 persons derivable
    assert not r6.new_small([], [m(range(12), n)])
    pool = [m(range(20), n)]
    leds = {h: {"pool": [x.copy() for x in pool], "stage": {"I": [m(range(20), n)]}, I2: [m(range(20), n)]}
            for h in r6.SPLITS}
    assert r6.admit_common(leds, "I", m(range(12), n))
    assert all(len(leds[h]["pool"]) == 2 and len(leds[h]["stage"]["I"]) == 2 for h in r6.SPLITS)
    assert all(len(leds[h][I2]) == 1 for h in r6.SPLITS)  # I[2]'s own ledger untouched
    # R_loc goes against the pooled ledger and I's stage ledger
    assert not r6.admit_common(leds, "R_loc", m(range(16), n))  # 16 vs 20 leaves 4
    assert all(len(leds[h]["pool"]) == 2 for h in r6.SPLITS)
    # refused under one split only: added to neither
    leds["halves"]["stage"]["I"].append(m(range(14), n))
    before = {h: (len(leds[h]["pool"]), len(leds[h]["stage"]["I"])) for h in r6.SPLITS}
    assert not r6.admit_common(leds, "I", m(range(11), n))
    assert {h: (len(leds[h]["pool"]), len(leds[h]["stage"]["I"])) for h in r6.SPLITS} == before
    # I[2] goes against its own ledger alone
    assert r6.admit_common(leds, I2, m(range(10), n))
    assert all(len(leds[h][I2]) == 2 for h in r6.SPLITS)
    assert not r6.admit_common(leds, I2, m(range(18), n))


def test_ledger_of():
    led = {"pool": ["p"], "stage": {"I": ["i"], "Bsd@2000": ["b"]}, I2: ["q"]}
    assert r6.ledger_of(led, I2) == [["q"]]
    assert r6.ledger_of(led, "R_loc") == [["p"], ["i"]] and r6.ledger_of(led, "R_mean") == [["p"], ["i"]]
    assert r6.ledger_of(led, "Bsd@2000") == [["p"], ["b"]]


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
    rec = n6.jsonish(r6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro, rp, rq))
    return rec, rn, ro, rp, rq, (ks_, kid, kp, ls_, lid, lp)


def all_cells(rec):
    for nm in rec["cells"]:
        for row in r6.ROWS_R:
            for st in r6.STAGES_R:
                yield nm, row, st, rec["cells"][nm][row][st]


def rerun(run, **change):
    rec, rn, ro, rp, rq, (ks_, kid, kp, ls_, lid, lp) = run
    args = {"rn": rn, "ro": ro, "rp": rp, "rq": rq, **change}
    return r6.count_all(ks_, kid, kp, ls_, lid, lp, M, args["rn"], args["ro"], args["rp"], args["rq"])


def test_structure_no_per_person(run):
    rec = run[0]
    out = {"main_cells": r6.main_cells_out(rec), **rec}
    k6.v6d.no_per_person(out)
    k6.v6i.no_long_list(out)
    assert set(rec["cells"]) == {"kupa_essay", "lwp_lwp"}
    assert set(out["main_cells"]["read_first"]) == {f"{a} {b} {c}" for a, b, c in r6.READ_FIRST}
    assert len(out["main_cells"]["primary"]) == 6 and len(out["main_cells"]["shuffles"]) == 4
    for nm, row, st, e in all_cells(rec):
        assert "_common" not in e
        assert e["persons_common"] + e["persons_one_split_only"]["oddeven"] == e["persons_rule"]["oddeven"]
        assert e["persons_common"] + e["persons_one_split_only"]["halves"] == e["persons_rule"]["halves"]


def test_some_cells_are_written_and_fixed_are_listed(run):
    rec = run[0]
    kinds = [("p_o" in e, bool(e.get("fixed_by_construction"))) for _, _, _, e in all_cells(rec)]
    assert sum(w for w, _ in kinds) > 10
    for nm, row, st, e in all_cells(rec):
        if r6.is_fixed(st, row):
            assert e.get("fixed_by_construction") and "p_o" not in e
        else:
            assert not e.get("fixed_by_construction")


_HV: dict = {}


def own_on(run, nm, how, row, st):
    """The split's verdicts by v6m's own_contrib on the stage's rule persons (I's for R and I[2]), with each split's
    cell scale."""
    rec, rn, ro, rp, rq, (ks_, kid, kp, ls_, lid, lp) = run
    ds, task = nm.split("_")
    series, ids, place = (ks_, kid, kp) if ds == "kupa" else (ls_, lid, lp)
    if nm not in _HV:
        b = n6.build(series, ids, ds, task, place, M, o6.BUILD_ROWS)
        _HV[nm] = q6.with_i2(n6.hv_of(b, o6.BUILD_ROWS))
    hv = _HV[nm]
    src = "I" if st in (I2, "R_loc", "R_mean") else st
    ok = p6.persons_with(hv, how, row, (src, "R"))
    first = [hv[how]["text/before"][i] for i in ok]
    sel = [hv[how][row][i] for i in ok]
    return ok, r6.m6.own_contrib(first, sel, st)


def test_written_cells_equal_an_independent_recount(run):
    rec = run[0]
    seen = 0
    for nm, row, st, e in all_cells(rec):
        if "p_o" not in e:
            continue
        oa, ca = own_on(run, nm, "oddeven", row, st)
        ob, cb = own_on(run, nm, "halves", row, st)
        common = sorted(set(oa) & set(ob))
        a = ca[[oa.index(p) for p in common]]
        b = cb[[ob.index(p) for p in common]]
        assert e["persons_common"] == len(common)
        assert e["p_o"] == pytest.approx(float(np.mean(a == b)), abs=1e-15)
        assert e["share"]["oddeven"] == pytest.approx(a.mean()) and e["share"]["halves"] == pytest.approx(b.mean())
        ds, task = nm.split("_")
        assert e["p_o_ci"] == r6.agreement(a, b, r6.rkey(ds, task, st, row))["p_o_ci"]
        tab = r6.table_of(a, b)
        assert e["table"] == {f"{r6.LAB[i]}->{r6.LAB[j]}": int(tab[i, j]) for i in range(3) for j in range(3)}
        assert e["persons_agree"] + e["persons_disagree"] == e["persons_common"]
        seen += 1
    assert seen == sum("p_o" in e for _, _, _, e in all_cells(rec)) and seen > 40


def test_beside_follows_v6n_and_v6q(run):
    rec, rn, ro, rp, rq, _ = run
    for nm, row, st, e in all_cells(rec):
        if "p_o" not in e:
            continue
        for how in r6.SPLITS:
            if st == I2:
                want = o6.size_class(rq["cells"][nm][how][row]["rule"][I2]["share_vs_own_ci"])
            elif st in ("R_loc", "R_mean"):
                want = o6.size_class(rn["rule_persons"][nm][how][row]["I"][st]["share_vs_own_ci"])
            else:
                want = o6.size_class(rn["rule_persons"][nm][how][row][st]["stage"]["share_vs_own_ci"])
            assert e["class_own_rule_persons"][how] == want
            src = "I" if st in (I2, "R_loc", "R_mean") else st
            oi = rn["rule_persons"][nm][how][row][src]
            assert e["state"][how] == q6.cell_state(o6.size_class(oi["R_loc"]["share_vs_own_ci"]),
                                                    o6.size_class(oi["R_mean"]["share_vs_own_ci"]))


def test_i2_and_i_share_common_persons(run):
    rec = run[0]
    for nm in rec["cells"]:
        for row in r6.ROWS_R:
            c = rec["cells"][nm][row]
            assert c[I2]["persons_common"] == c["I"]["persons_common"] == c["R_loc"]["persons_common"]


def test_stops_on_a_changed_v6n_rule_cell(run):
    rn = copy.deepcopy(run[1])
    rn["rule_persons"]["lwp_lwp"]["halves"]["cross_100"]["Bsd@2000"]["stage"]["share_vs_own"] += 0.01
    with pytest.raises(SystemExit, match="differs from v6n"):
        rerun(run, rn=rn)


def first_i2(rq, last=False):
    cands = [rq["cells"][nm][how][row]["rule"] for nm in ("kupa_essay", "lwp_lwp") for how in r6.SPLITS
             for row in r6.ROWS_R if I2 in rq["cells"][nm][how][row]["rule"]]
    assert cands, "v6q wrote no I[2] share in the fake run"
    return cands[-1] if last else cands[0]


def test_stops_on_a_changed_v6q_share(run):
    rq = copy.deepcopy(run[4])
    e = first_i2(rq)
    e[I2]["share_vs_own"] += 0.01
    with pytest.raises(SystemExit, match="v6q's share"):
        rerun(run, rq=rq)


def test_stops_on_a_changed_v6q_interval(run):
    rq = copy.deepcopy(run[4])
    e = first_i2(rq, last=True)
    e[I2]["share_vs_own_ci"] = [e[I2]["share_vs_own_ci"][0] - 0.01, e[I2]["share_vs_own_ci"][1]]
    with pytest.raises(SystemExit, match="v6q's share and interval"):
        rerun(run, rq=rq)


def test_stops_on_a_changed_v6p_plan(run):
    rp = copy.deepcopy(run[3])
    plan = rp["sets"]["lwp_lwp"]["written"]["oddeven"]["cross_100"]
    plan["I"] = {"n_s": "derivable", "n_s_min": "derivable"} if plan["I"].get("n_s") == "written" else \
        {"n_s": "written", "n_s_min": "written"}
    with pytest.raises(SystemExit, match="v6p's plan"):
        rerun(run, rp=rp)


def test_withheld_when_not_admitted(run, monkeypatch):
    monkeypatch.setattr(r6, "admit_common", lambda leds, st, mask: False)
    rec = rerun(run)
    for nm in rec["cells"]:
        for row in r6.ROWS_R:
            for st, e in rec["cells"][nm][row].items():
                assert "p_o" not in e
                if not (e.get("fixed_by_construction") or e.get("no_count") or e.get("withheld") == "fewer than 5"):
                    assert e["withheld"] == "derivable"


def test_admission_order_and_ledgers(run, monkeypatch):
    calls = []
    real = r6.admit_common

    def spy(leds, st, mask):
        calls.append(st)
        return real(leds, st, mask)

    monkeypatch.setattr(r6, "admit_common", spy)
    rerun(run)
    order = [s for s in calls[: len(r6.STAGES_R)]]
    want = [s for s in r6.STAGES_R if not r6.is_fixed(s, r6.ROWS_R[0])]
    assert order[: len(want)] == want


def test_final_check_catches_a_new_small_set(run, monkeypatch):
    # admit everything without checking: the final check must stop on a set that admit_common would refuse
    def admit_blind(leds, st, mask):
        for how in r6.SPLITS:
            for s in r6.ledger_of(leds[how], st):
                s.append(mask)
                s.append(mask.copy() & ~np.eye(len(mask), dtype=bool)[np.flatnonzero(mask)[0]])
        return True

    monkeypatch.setattr(r6, "admit_common", admit_blind)
    with pytest.raises(SystemExit, match="new small set"):
        rerun(run)


def test_i2_common_check(run, monkeypatch):
    real = r6.verdicts

    def shift(*a, **k):
        v = real(*a, **k)
        if a[4] == "halves":
            s = v["stages"][I2]
            if not s.get("no_count"):
                v["stages"][I2] = {**s, "persons": s["persons"][1:], "own": s["own"][1:]}
        return v

    monkeypatch.setattr(r6, "verdicts", shift)
    with pytest.raises(SystemExit, match="I\\[2\\]'s common persons differ"):
        rerun(run)


def test_fewer_than_5_common(run, monkeypatch):
    real = r6.verdicts

    def tiny(*a, **k):
        v = real(*a, **k)
        if a[4] == "halves":
            for st in r6.STAGES_R:
                s = v["stages"][st]
                if not s.get("no_count"):
                    v["stages"][st] = {**s, "persons": s["persons"][:4], "own": s["own"][:4]}
        return v

    monkeypatch.setattr(r6, "verdicts", tiny)
    rec = rerun(run)
    for nm, row, st, e in all_cells({"cells": rec["cells"]}):
        if not (e.get("fixed_by_construction") or e.get("no_count")):
            assert e["withheld"] == "fewer than 5" and e["persons_common"] <= 4


def test_stops_when_splits_index_differently(run, monkeypatch):
    real = n6.hv_of

    def drop(b, rows):
        hv = real(b, rows)
        hv["halves"] = {k: v[:-1] for k, v in hv["halves"].items()}
        return hv

    monkeypatch.setattr(n6, "hv_of", drop)
    with pytest.raises(SystemExit, match="same persons"):
        rerun(run)


# ---------------------------------------------------------------- predictions and summaries


def cellw(po, kappa=None, pe=0.5, n=100):
    agree = int(round(po * n))
    e = {"p_o": agree / n, "one_minus_p_o": 1 - agree / n, "p_e": pe, "persons_common": n, "persons_agree": agree,
         "persons_disagree": n - agree}
    if kappa is not None:
        e["kappa"] = kappa
    return e


def fake_rec(over=None):
    over = over or {}
    cells = {nm: {row: {st: cellw(0.9, 0.5) for st in r6.STAGES_R} for row in r6.ROWS_R}
             for nm in ("kupa_essay", "lwp_lwp")}
    for (nm, row, st), e in over.items():
        cells[nm][row][st] = e
    return {"cells": cells}


def test_predictions_by_hand():
    rec = fake_rec({("lwp_lwp", "inside_75", "R_loc"): cellw(0.8, 0.4), ("lwp_lwp", "inside_100", "R_loc"): cellw(0.76, 0.4),
                      ("lwp_lwp", "inside_75", "I"): cellw(0.7, 0.3), ("lwp_lwp", "inside_100", "I"): cellw(0.79, 0.35),
                      ("lwp_lwp", "inside_100", I2): cellw(0.7, 0.2)})
    p = r6.predictions(rec)
    assert p["P1"].startswith("held") and p["P3"].startswith("held") and p["P4"].startswith("held")
    assert p["P2"].startswith("did not hold")  # most cells have 1 - p_o = 0.1
    rec = fake_rec({("lwp_lwp", "inside_100", "R_loc"): cellw(0.74, 0.4), ("lwp_lwp", "inside_100", "I"): cellw(0.81, 0.1),
                      ("lwp_lwp", "inside_100", I2): cellw(0.7, 0.2)})
    p = r6.predictions(rec)
    assert p["P1"].startswith("did not hold") and p["P3"].startswith("did not hold") and p["P4"].startswith("did not hold")
    tie = fake_rec({("lwp_lwp", "inside_100", I2): cellw(0.7, 0.25), ("lwp_lwp", "inside_100", "I"): cellw(0.7, 0.25)})
    assert r6.predictions(tie)["P4"].startswith("did not hold")


def test_prediction_p2_counts_by_hand():
    over = {}
    k = 0
    for nm in ("kupa_essay", "lwp_lwp"):
        for row in r6.PRIMARY_ROWS:
            for st in r6.STAGES_R:
                over[(nm, row, st)] = cellw(0.79 if k % 2 == 0 else 0.81, 0.3)
                k += 1
    p = r6.predictions(fake_rec(over))
    assert p["P2"] == "held (21 of 42)"
    over[("kupa_essay", "inside_75", "U@2000")] = {"persons_common": 50, "fixed_by_construction": True}
    p = r6.predictions(fake_rec(over))
    assert p["P2"] == "did not hold (20 of 41)"


def test_prediction_boundaries_are_exact():
    # 1 - p_o = 1/5 exactly (0.8 in floating point gives 0.19999999999999996): counts as at least 0.20
    over = {(nm, row, st): cellw(0.8, 0.3, n=5) for nm in ("kupa_essay", "lwp_lwp") for row in r6.PRIMARY_ROWS
            for st in r6.STAGES_R}
    assert 1 - 0.8 < 0.2 and r6.predictions(fake_rec(over))["P2"] == "held (42 of 42)"
    # p_o = 3/4 exactly counts for P1; p_o = 4/5 exactly does not count as below 0.80 for P3
    rec = fake_rec({("lwp_lwp", "inside_75", "R_loc"): cellw(0.75, 0.4, n=4), ("lwp_lwp", "inside_100", "R_loc"): cellw(0.75, 0.4, n=8),
                    ("lwp_lwp", "inside_75", "I"): cellw(0.8, 0.3, n=5), ("lwp_lwp", "inside_100", "I"): cellw(0.6, 0.3, n=5)})
    p = r6.predictions(rec)
    assert p["P1"].startswith("held") and p["P3"].startswith("did not hold")


def test_predictions_not_evaluable():
    nw = {"persons_common": 50, "withheld": "derivable"}
    rec = fake_rec({("lwp_lwp", "inside_75", "R_loc"): nw, ("lwp_lwp", "inside_100", "I"): nw})
    p = r6.predictions(rec)
    assert p["P1"] == p["P3"] == p["P4"] == r6.NE
    rec = fake_rec({("lwp_lwp", "inside_100", I2): cellw(0.9)})  # kappa not written
    assert r6.predictions(rec)["P4"] == r6.NE
    over = {(nm, row, st): cellw(0.9) for nm in ("kupa_essay", "lwp_lwp") for row in r6.PRIMARY_ROWS for st in r6.STAGES_R}
    assert r6.predictions(fake_rec(over))["P2"] == r6.NE


def test_summary_by_hand():
    rec = fake_rec({("kupa_essay", "inside_75", "U@2000"): {"persons_common": 9, "fixed_by_construction": True},
                      ("kupa_essay", "inside_75", "B@2000"): {"persons_common": 9, "withheld": "derivable"},
                      ("kupa_essay", "inside_100", "I"): cellw(0.6, 0.1),
                      ("lwp_lwp", "shuffled/before", "I"): {"persons_common": 3, "withheld": "fewer than 5"}})
    s = r6.summary(rec)
    assert s["primary"]["written"] == 40 and s["primary"]["fixed"] == 1 and s["primary"]["withheld_derivable"] == 1
    assert s["primary"]["one_minus_p_o"] == {"cells": 40, "min": 0.1, "median": 0.1, "max": 0.4}
    assert s["primary"]["kappa"]["min"] == 0.1 and s["shuffles"]["withheld_fewer_than_5"] == 1
    assert s["shuffles"]["written"] == 27


def test_short_cell():
    rule = {"oddeven": 11, "halves": 10}
    e = {"persons_common": 9, "persons_rule": rule, "fixed_by_construction": True}
    assert r6.short(e) == "not written (fixed; common 9, rule 11/10)"
    assert r6.short({"persons_common": 9, "persons_rule": rule, "withheld": "derivable"}) == \
        "not written (derivable; common 9, rule 11/10)"


def test_dropping_two_persons_is_withheld_as_derivable(run, monkeypatch):
    """Without monkeypatching admit_common: a common set two persons short of odd/even's rule set makes those two
    persons' sum derivable from v6n's rule cell, so the real ledgers must refuse it."""
    real = r6.verdicts

    def drop2(*a, **k):
        v = real(*a, **k)
        if a[4] == "halves":
            for st in r6.STAGES_R:
                s = v["stages"][st]
                if not s.get("no_count"):
                    v["stages"][st] = {**s, "persons": s["persons"][2:], "own": s["own"][2:]}
        return v

    monkeypatch.setattr(r6, "verdicts", drop2)
    rec = rerun(run)
    seen = 0
    for nm, row, st, e in all_cells({"cells": rec["cells"]}):
        if e.get("fixed_by_construction") or e.get("no_count"):
            continue
        assert e["persons_one_split_only"]["oddeven"] == 2 and e["withheld"] == "derivable" and "p_o" not in e
        seen += 1
    assert seen > 10


def test_ledgers_hold_v6n_sets(run):
    """The rebuilt pooled ledger holds the masks of every rule cell and count 1 cell that v6n wrote, and I[2]'s ledger
    holds I's rule persons where v6q wrote them (a hand check of the rebuild, independent of v6p's plan)."""
    rec, rn, ro, rp, rq, (ks_, kid, kp, ls_, lid, lp) = run
    nm, ds, task, how, row = "lwp_lwp", "lwp", "lwp", "oddeven", "cross_100"
    b = n6.build(ls_, lid, ds, task, lp, M, o6.BUILD_ROWS)
    hv = n6.hv_of(b, o6.BUILD_ROWS)
    cov = [o6.covariates_of(*ls_[pid]) for pid in b["persons"]]
    keep = o6.set_keep(hv, cov, ds, task, rn)
    lab = o6.labels_of(keep)
    n = len(hv[how]["text/before"])
    gate0 = o6.cell_persons(hv, how, p6.ROWS_O[0], "R")
    rules = {st: p6.rule_keep(hv, ds, task, how, row, st, rn) for st in p6.STAGES}
    written_o = {c: bool(ro["written"][how][nm][c]) for c in p6.COVARIATES}
    led = r6.ledgers(hv, how, row, gate0, [], lab[how], written_o, rules, nm, rn, rp, rq, n)
    have = {m.tobytes() for m in led["pool"]}
    for st in p6.STAGES:
        if not rules[st].get("no_count"):
            assert p6.mask_of(rules[st]["persons"], n).tobytes() in have
        ix = p6.persons_with(hv, how, row, (st,))
        if "share_vs_own" in rn["count_1"][nm][how][row][st]:
            assert p6.mask_of(ix, n).tobytes() in have
    if "withheld" not in rq["cells"][nm][how][row]["rule"]:
        assert led[I2][0].tobytes() == p6.mask_of(rules["I"]["persons"], n).tobytes()


def test_dropping_two_persons_from_oddeven_is_withheld(run, monkeypatch):
    real = r6.verdicts

    def drop2(*a, **k):
        v = real(*a, **k)
        if a[4] == "oddeven":
            for st in r6.STAGES_R:
                s = v["stages"][st]
                if not s.get("no_count"):
                    v["stages"][st] = {**s, "persons": s["persons"][2:], "own": s["own"][2:]}
        return v

    monkeypatch.setattr(r6, "verdicts", drop2)
    rec = rerun(run)
    for nm, row, st, e in all_cells({"cells": rec["cells"]}):
        if not (e.get("fixed_by_construction") or e.get("no_count")):
            assert e["persons_one_split_only"]["halves"] == 2 and e["withheld"] == "derivable"


def test_i2_ledger_equals_v6q_own(run, monkeypatch):
    """The rebuilt I[2] ledger equals the ledger v6q checked (spied from v6q's own_ledger_ok), row by row."""
    rec, rn, ro, rp, rq, (ks_, kid, kp, ls_, lid, lp) = run
    seen_q = []
    monkeypatch.setattr(q6, "own_ledger_ok", lambda sets: seen_q.append([m.tobytes() for m in sets]))
    q6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn, ro, rp)
    order_q = [(nm, how, row) for nm in ("kupa_essay", "lwp_lwp") for how in v6b.BASELINE for row in q6.ROWS_O]
    assert len(seen_q) == len(order_q)
    q = dict(zip(order_q, seen_q))
    seen_r = {}
    real = r6.ledgers

    def spy(hv, how, row, *a):
        led = real(hv, how, row, *a)
        seen_r[(a[5], how, row)] = [m.tobytes() for m in led[I2]]
        return led

    monkeypatch.setattr(r6, "ledgers", spy)
    rerun(run)
    assert set(seen_r) == {(nm, how, row) for nm in ("kupa_essay", "lwp_lwp") for how in r6.SPLITS for row in r6.ROWS_R}
    for k, v in seen_r.items():
        assert sorted(v) == sorted(q[k])


def test_same_row_stops(run):
    rp, rq = copy.deepcopy(run[3]), copy.deepcopy(run[4])
    r6.same_row(rp, rq, "lwp_lwp", "oddeven")
    rp["sets"]["lwp_lwp"]["written"]["oddeven"]["inside/before"]["I"] = {"n_s": "x"}
    with pytest.raises(SystemExit, match="v6p's plans"):
        r6.same_row(rp, rq, "lwp_lwp", "oddeven")
    rp = copy.deepcopy(run[3])
    rq["groups"]["n_s"]["lwp_lwp"]["halves"]["inside/before"]["written"] = \
        not rq["groups"]["n_s"]["lwp_lwp"]["halves"]["inside_100"].get("written")
    with pytest.raises(SystemExit, match="families differ"):
        r6.same_row(rp, rq, "lwp_lwp", "halves")


def test_state_of_reads_r_loc_then_r_mean():
    def cell(loc, mean):
        return {"R_loc": {"share_vs_own_ci": loc}, "R_mean": {"share_vs_own_ci": mean}}
    assert r6.state_of(cell([0.7, 0.9], [0.1, 0.2])) == "gate"
    assert r6.state_of(cell([0.1, 0.2], [0.7, 0.9])) == "R_mean only"
    assert r6.state_of(cell([0.5, 0.7], [0.5, 0.7])) == "not read"


def test_i2_persons_check_stops(run, monkeypatch):
    real = p6.persons_with

    def fewer(hv, how, row, need):
        out = real(hv, how, row, need)
        return out[1:] if need == (I2, "R") else out

    monkeypatch.setattr(p6, "persons_with", fewer)
    with pytest.raises(SystemExit, match="persons with I\\[2\\] and R"):
        rerun(run)


def test_admit_checks_the_second_split_too():
    n, m = 30, p6.mask_of
    leds = {"oddeven": {"pool": [m(range(20), n)], "stage": {"I": [m(range(20), n)]}, I2: []},
            "halves": {"pool": [m(range(20), n)], "stage": {"I": [m(range(20), n), m(range(8), n)]}, I2: []}}
    assert not r6.admit_common(leds, "I", m(range(6), n))  # fine under odd/even, 2 left under halves
    assert len(leds["oddeven"]["pool"]) == 1 and len(leds["oddeven"]["stage"]["I"]) == 1


@pytest.mark.parametrize("which", [0, 1])
def test_final_checks_each_catch(run, monkeypatch, which):
    def admit_blind(leds, st, mask):
        for how in r6.SPLITS:
            ls = r6.ledger_of(leds[how], st)
            if len(ls) == 2:
                ls[which].append(mask)
                ls[which].append(mask.copy() & ~np.eye(len(mask), dtype=bool)[np.flatnonzero(mask)[0]])
        return True

    monkeypatch.setattr(r6, "admit_common", admit_blind)
    with pytest.raises(SystemExit, match="pooled ledger" if which == 0 else "stage's ledger"):
        rerun(run)


def test_same_row_is_called(run):
    rp = copy.deepcopy(run[3])
    rp["sets"]["kupa_essay"]["written"]["halves"]["inside/before"]["I"] = {"n_s": "x"}
    with pytest.raises(SystemExit, match="v6p's plans"):
        rerun(run, rp=rp)


def test_pool_holds_v6p_written_families(run, monkeypatch):
    rp = run[3]
    got = []
    real = r6.ledgers

    def spy(hv, how, row, gate0, o_fams, lab_how, written_o, rules, nm, *rest):
        led = real(hv, how, row, gate0, o_fams, lab_how, written_o, rules, nm, *rest)
        got.append((how, row, gate0, lab_how, rules, nm, len(hv[how]["text/before"]), {x.tobytes() for x in led["pool"]}))
        return led

    monkeypatch.setattr(r6, "ledgers", spy)
    rerun(run)
    seen = 0
    for how, row, gate0, lab_how, rules, nm, n, have in got:
        pos = {p: k for k, p in enumerate(gate0)}
        plan = rp["sets"][nm]["written"][how][row]
        for st in p6.STAGES:
            for c in p6.COVARIATES:
                if plan[st].get(c) == "written":
                    ok = rules[st]["persons"]
                    for f in p6.family(ok, lab_how[c][2][[pos[p] for p in ok]], n):
                        assert f.tobytes() in have
                        seen += 1
    assert seen > 0


def test_ledgers_adds_written_families_to_pool_and_stage(monkeypatch):
    n = 12
    fam = [p6.mask_of([0, 1, 2, 3, 4], n), p6.mask_of([5, 6, 7, 8], n), p6.mask_of([9, 10, 11], n)]
    plan = {st: ({"n_s": "written"} if st == "Bsd@2000" else {}) for st in p6.STAGES}
    monkeypatch.setattr(p6, "base_sets", lambda *a: {"count_1": {}, "rule": {}, "v6o": []})
    monkeypatch.setattr(p6, "plan_row", lambda *a: (plan, {}))
    monkeypatch.setattr(p6, "pooled", lambda base: [])
    monkeypatch.setattr(p6, "stage_ledger", lambda base, st: [])
    monkeypatch.setattr(p6, "family", lambda ok, g, n_: [f.copy() for f in fam])
    rp = {"sets": {"x": {"written": {"oddeven": {"r": plan}}}}}
    rq = {"cells": {"x": {"oddeven": {"r": {"rule": {"no_count": True}, "gate_0": {"no_count": True}}}}},
          "groups": {c: {"x": {"oddeven": {"r": {}}}} for c in p6.COVARIATES}}
    rules = {st: {"persons": list(range(n))} for st in p6.STAGES}
    rules["I"] = {"persons": list(range(n)), "no_count": True}
    lab = {"n_s": (0, 0, np.zeros(n, dtype=int)), "n_s_min": (0, 0, np.zeros(n, dtype=int))}
    led = r6.ledgers({}, "oddeven", "r", list(range(n)), [], lab, {}, rules, "x", {"count_1": {"x": {"oddeven": {}}}},
                     rp, rq, n)
    assert len(led["pool"]) == 3 and len(led["stage"]["Bsd@2000"]) == 3 and led["stage"]["I"] == []
    assert led[I2] == []
