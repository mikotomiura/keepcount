import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6o as o6  # noqa: E402
from test_prototype_v6k import fake_person  # noqa: E402

n6, m6, k6 = o6.n6, o6.m6, o6.k6
v6, v6b = k6.v6, k6.v6b
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
RL = json.loads((ROOT / "results/prototype_v6l/record.json").read_text())


# ---------------------------------------------------------------- constants


def test_constants_follow_the_conditions():
    c = json.loads((ROOT / "works/conditions/prototype_v6o.json").read_text())
    assert o6.TAG == 96 and "96" in c["count_1_by_group"]["interval"]
    assert o6.REFS == ("R_loc", "R_mean") and n6.STAGE["R_loc"] == 8 and n6.STAGE["R_mean"] == 12
    assert o6.ROWS_O == ("inside/before", "shuffled/before", "shuffled_in_halves/before", "inside_25", "inside_50",
                         "inside_75", "inside_100", "cross_25", "cross_50", "cross_75", "cross_100")
    assert o6.COVARIATES == {"n_s": 0, "n_s_min": 10} and "10, 11, 12" in c["covariate"]["n_s_min"]
    assert "13, 14, 15" in c["covariate"]["n_s_min"] and "group 3 + band" in c["matched_bands"]["how"]
    assert o6.gkey("lwp", "lwp", "R_mean", "cross_75", "halves", 2) == [20261031, 96, 2, 0, 12, 15, 1, 2, 0]
    assert o6.gkey("kupa", "essay", "R_loc", "inside/before", "oddeven", 0) == [20261031, 96, 0, 1, 8, 3, 0, 0, 0]
    assert o6.PRIMARY_ROWS == ("inside_75", "inside_100", "cross_100") and o6.MIN_SUMMARY == 5


# ---------------------------------------------------------------- the covariate


def test_sentence_counts_by_hand():
    s, kinds = fake_person(7, 9)
    lab = v6b.labels(s, kinds)
    ranges = s.ranges("before")
    for how in v6b.BASELINE:
        A, B = v6b.parts(len(ranges), how)
        want = [int(sum(np.sum(lab[np.arange(*ranges[i])] == "sentence") for i in idx)) for idx in (A, B)]
        assert list(o6.sentence_counts(s, kinds, "text/before", how)) == want
    cov = o6.covariates_of(s, kinds)
    for how in v6b.BASELINE:
        a, b = o6.sentence_counts(s, kinds, "text/before", how)
        assert cov[how] == {"n_s": (a + b) / 2, "n_s_min": float(min(a, b))}
    assert cov["oddeven"]["n_s"] > 0


def test_covariates_stop_if_a_row_changes_the_counts(monkeypatch):
    s, kinds = fake_person(8, 9)
    real = o6.sentence_counts

    def bad(s_, k_, row, how):
        a, b = real(s_, k_, row, how)
        return (a + 1, b) if row == "cross_50" else (a, b)
    monkeypatch.setattr(o6, "sentence_counts", bad)
    with pytest.raises(SystemExit, match="cross_50"):
        o6.covariates_of(s, kinds)


# ---------------------------------------------------------------- groups


def test_cuts_and_groups_ties_go_down():
    x = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4], dtype=float)
    c1, c2 = o6.cuts_of(x)
    assert (c1, c2) == tuple(np.quantile(x, [1 / 3, 2 / 3]))
    g = o6.group_of(x, c1, c2)
    assert all(g[x <= c1] == 0) and all(g[(x > c1) & (x <= c2)] == 1) and all(g[x > c2] == 2)
    g2 = o6.group_of(np.array([5.0, 5.0, 6.0]), 5.0, 5.0)
    assert list(g2) == [0, 0, 2]  # c1 = c2: mid empty


def test_by_group_partition_share_and_difference():
    rng = np.random.default_rng(3)
    own = rng.random(30)  # continuous (by_group only averages), so that a changed key changes the interval
    base, change = rng.random(30) + 0.1, rng.random(30)
    g = np.repeat([0, 1, 2], 10)
    out = o6.by_group(own, base, change, g, "kupa", "essay", "R_loc", "inside_100", "halves", 0)
    assert out["written"] and out["persons"] == {"low": 10, "mid": 10, "high": 10}
    w = sum(out["groups"][k]["share_vs_own"] * 10 for k in o6.GROUPS) / 30
    assert w == pytest.approx(own.mean(), abs=1e-12)
    hi = out["groups"]["high"]
    assert hi["share_vs_own"] == pytest.approx(own[20:].mean())
    idx = np.random.default_rng(o6.gkey("kupa", "essay", "R_loc", "inside_100", "halves", 2)).integers(0, 10, (2000, 10))
    assert hi["share_vs_own_ci"] == v6.ci(own[20:], idx)
    assert "median_base" not in hi and hi["median_change"] == pytest.approx(np.median(change[20:]))
    il = np.random.default_rng(o6.gkey("kupa", "essay", "R_loc", "inside_100", "halves", 0)).integers(0, 10, (2000, 10))
    d = own[20:][idx].mean(1) - own[:10][il].mean(1)
    assert out["high_minus_low"]["ci"] == pytest.approx([np.percentile(d, 2.5), np.percentile(d, 97.5)])
    sh = [out["groups"][k]["share_vs_own"] for k in o6.GROUPS]
    assert out["rising"] == (sh[0] <= sh[1] <= sh[2])


def test_by_group_offset_changes_the_key():
    own = np.tile([0.0, 1.0, 0.5], 10)
    g = np.repeat([0, 1, 2], 10)
    a = o6.by_group(own, own + 1, own, g, "lwp", "lwp", "R_mean", "cross_100", "oddeven", 0)
    b = o6.by_group(own, own + 1, own, g, "lwp", "lwp", "R_mean", "cross_100", "oddeven", 10)
    assert a["groups"]["low"]["share_vs_own"] == b["groups"]["low"]["share_vs_own"]
    assert a["high_minus_low"]["ci"] != b["high_minus_low"]["ci"]


def test_by_group_small_group_writes_nothing_but_persons():
    own = np.ones(14)
    g = np.array([0] * 5 + [1] * 4 + [2] * 5)
    out = o6.by_group(own, own, own, g, "kupa", "essay", "R_loc", "inside_75", "halves", 0)
    assert out == {"persons": {"low": 5, "mid": 4, "high": 5}, "written": False}
    g = np.array([0] * 5 + [1] * 5 + [2] * 5)
    assert o6.by_group(np.ones(15), np.ones(15), np.ones(15), g, "kupa", "essay", "R_loc", "inside_75", "halves",
                       0)["written"]


def test_by_group_stops_if_not_a_partition():
    with pytest.raises(SystemExit, match="partition"):
        o6.by_group(np.ones(15), np.ones(15), np.ones(15), np.array([0] * 5 + [1] * 5 + [3] * 5), "kupa", "essay",
                    "R_loc", "inside_75", "halves", 0)


def test_ratio_median_needs_five_positive_bases():
    own = np.ones(6)
    st, _ = o6.group_stats(own, np.array([0, 0, 1, 1, 1, 1.0]), np.ones(6), [1, 2, 3])
    assert st["median_change_over_base"] is None
    st, _ = o6.group_stats(own, np.array([0, 1, 1, 1, 1, 2.0]), np.ones(6) * 2, [1, 2, 3])
    assert st["median_change_over_base"] == pytest.approx(2.0)


def test_size_class_and_side():
    assert o6.size_class([0.61, 0.9]) == "above" and o6.size_class([0.1, 0.59]) == "below"
    assert o6.size_class([0.6, 0.7]) == "undecided"
    assert o6.interval_side([0.01, 0.2]) == "above 0" and o6.interval_side([-0.2, -0.01]) == "below 0"
    assert o6.interval_side([0.0, 0.2]) == "includes 0"


# ---------------------------------------------------------------- contributions


def fake_hv(n=20, rows=("text/before", "inside_100", "cross_100")):
    hv = {how: {row: [] for row in rows} for how in v6b.BASELINE}
    for i in range(n):
        s, kinds = fake_person(500 + i, 8 + i % 6)
        vec, _, _, _, _ = n6.person_rows(s, kinds, [0, 1, i], M, list(rows))
        for how in v6b.BASELINE:
            for row in rows:
                hv[how][row].append(vec[row][how])
    return hv


@pytest.fixture(scope="module")
def hv():
    return fake_hv()


def test_contrib_base_change_equals_v6b_one(hv):
    for ref in o6.REFS:
        ok = o6.cell_persons(hv, "halves", "inside_100", ref)
        f = [hv["halves"]["text/before"][i] for i in ok]
        s = [hv["halves"]["inside_100"][i] for i in ok]
        own, base, change = o6.contrib_base_change(f, s, ref)
        one = v6b.one(f, s, ref, [1, 2, 3])
        assert own.mean() == pytest.approx(one["share_vs_own"], abs=1e-12)
        assert np.median(base) == pytest.approx(one["median_base"])
        assert np.median(change) == pytest.approx(one["median_change"])


def test_contrib_stops_if_own_contrib_differs(hv, monkeypatch):
    ok = o6.cell_persons(hv, "halves", "inside_100", "R_loc")
    f = [hv["halves"]["text/before"][i] for i in ok]
    s = [hv["halves"]["inside_100"][i] for i in ok]
    monkeypatch.setattr(m6, "own_contrib", lambda a, b, c: np.zeros(len(a)))
    with pytest.raises(SystemExit, match="own_contrib"):
        o6.contrib_base_change(f, s, "R_loc")


# ---------------------------------------------------------------- the whole count on fake persons


def fake_sets(n=21):
    ks_, ls_ = {}, {}
    for i in range(n):
        ks_[f"k{i}"] = {"copy": fake_person(1000 + i, 10), "essay": fake_person(2000 + i, 6 + i % 7)}
        ls_[f"l{i}"] = fake_person(3000 + i, 7 + i % 9)
    return ks_, ls_


@pytest.fixture(scope="module")
def run():
    ks_, ls_ = fake_sets()
    kid, lid = sorted(ks_), sorted(ls_)
    kp, lp = {p: i for i, p in enumerate(kid)}, {p: i for i, p in enumerate(lid)}
    rn = n6.jsonish(n6.count_all(ks_, kid, kp, ls_, lid, lp, M, RL, n_splits=3))
    rec = o6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn)
    return rec, rn, (ks_, kid, kp, ls_, lid, lp)


def test_count_all_structure_no_per_person(run):
    rec, rn, _ = run
    out = {"main_cells": o6.main_cells_out(rec), **rec, "predictions": o6.predictions(rec)}
    v6b_no = k6.v6d.no_per_person
    v6b_no(out)
    k6.v6i.no_long_list(out)
    for c in o6.COVARIATES:
        for nm in ("kupa_essay", "lwp_lwp"):
            for how in v6b.BASELINE:
                o = rec["groups"][c][nm][how]
                assert set(o["cells"]) == set(o6.REFS)
                assert set(o["cells"]["R_loc"]) == set(o6.ROWS_O)
                assert sum(o["persons"].values()) == rn["count_1"][nm][how]["inside_100"]["R_loc"]["persons"]


def test_count_all_groups_give_v6n_shares(run):
    rec, rn, _ = run
    n_written = 0
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for ref in o6.REFS:
                for row in o6.ROWS_O:
                    cell = rec["groups"]["n_s"][nm][how]["cells"][ref][row]
                    if not cell["written"]:
                        continue
                    n_written += 1
                    w = sum(cell["groups"][g]["share_vs_own"] * cell["groups"][g]["persons"] for g in o6.GROUPS)
                    full = rn["count_1"][nm][how][row][ref]
                    assert w / full["persons"] == pytest.approx(full["share_vs_own"], abs=1e-12)
    assert n_written > 0


def test_count_all_stops_on_a_changed_v6n_cell(run):
    _, rn, args = run
    ks_, kid, kp, ls_, lid, lp = args
    for field, val in (("share_vs_own", 0.123), ("persons", 999), ("share_vs_own_ci", [0.0, 1.0])):
        bad = copy.deepcopy(rn)
        bad["count_1"]["lwp_lwp"]["halves"]["cross_75"]["R_mean"][field] = val
        with pytest.raises(SystemExit, match="lwp_lwp halves cross_75 R_mean"):
            o6.count_all(ks_, kid, kp, ls_, lid, lp, M, bad)
    bad = copy.deepcopy(rn)
    bad["tallies"]["kupa_essay"]["count_persons"] += 1
    with pytest.raises(SystemExit, match="count persons"):
        o6.count_all(ks_, kid, kp, ls_, lid, lp, M, bad)


def test_bands_written_rule_and_values(run):
    rec, _, _ = run
    for c in o6.COVARIATES:
        for how in v6b.BASELINE:
            o = rec["bands"][c][how]
            sizes = list(o["persons"]["kupa_essay"].values()) + list(o["persons"]["lwp"].values())
            assert o["written"] == (min(sizes) >= 5)
            for ref in o6.REFS:
                for row in o6.ROWS_O:
                    cell = o["cells"][ref][row]
                    assert ("bands" in cell) == o["written"]
                    if o["written"]:
                        for g in o6.GROUPS:
                            b = cell["bands"][g]
                            assert b["lwp_minus_kupa"] == pytest.approx(b["lwp"] - b["kupa_essay"])


def test_bands_cuts_are_kupas(run):
    rec, _, _ = run
    for how in v6b.BASELINE:
        assert rec["bands"]["n_s"][how]["cuts_from_kupa_essay"] == rec["groups"]["n_s"]["kupa_essay"][how]["cuts"]
        assert rec["bands"]["n_s"][how]["persons"]["kupa_essay"] == rec["groups"]["n_s"]["kupa_essay"][how]["persons"]


# ---------------------------------------------------------------- predictions


def pred_rec(side_k, side_l, bases, band_sides, written=True):
    def cell(ci):
        return {"written": True, "high_minus_low": {"difference": 0.0, "ci": ci}}
    ci_of = {"above 0": [0.1, 0.2], "includes 0": [-0.1, 0.2], "below 0": [-0.2, -0.1]}
    groups = {"n_s": {}}
    for nm, side, row in (("kupa_essay", side_k, "inside_100"), ("lwp_lwp", side_l, "inside_75")):
        groups["n_s"][nm] = {how: {"cells": {"R_loc": {row: cell(ci_of[side])}},
                                   "base_by_group": {ref: {"low": bases[0], "mid": 0, "high": bases[1]}
                                                     for ref in o6.REFS}} for how in v6b.BASELINE}
    bands = {"n_s": {how: {"written": written, "cells": {"R_loc": {"inside_100": {"bands": {
        g: {"ci": ci_of[band_sides[how]]} for g in o6.GROUPS}}}}} for how in v6b.BASELINE}}
    return {"groups": groups, "bands": bands}


def test_predictions_held_and_not():
    r = pred_rec("above 0", "above 0", (2.0, 1.0), {"oddeven": "includes 0", "halves": "above 0"})
    p = o6.predictions(r)
    assert p["P1"] == "held" and p["P2"] == "held" and p["P3"].startswith("held (8 of 8)") and p["P4"] == "held"
    r = pred_rec("includes 0", "above 0", (1.0, 2.0), {"oddeven": "above 0", "halves": "above 0"})
    p = o6.predictions(r)
    assert p["P1"] == "did not hold" and p["P3"].startswith("did not hold (0 of 8)") and p["P4"] == "did not hold"
    r = pred_rec("above 0", "above 0", (2.0, 1.0), {"oddeven": "above 0", "halves": "above 0"}, written=False)
    assert o6.predictions(r)["P4"] == o6.NE


def test_predictions_not_evaluable_when_unwritten():
    r = pred_rec("above 0", "above 0", (2.0, 1.0), {"oddeven": "includes 0", "halves": "includes 0"})
    r["groups"]["n_s"]["kupa_essay"]["halves"]["cells"]["R_loc"]["inside_100"] = {"written": False}
    r["groups"]["n_s"]["lwp_lwp"]["halves"]["base_by_group"]["R_mean"] = None
    p = o6.predictions(r)
    assert p["P1"] == o6.NE and p["P2"] == "held" and p["P3"] == o6.NE


def fake_keep(x, own, seed, x_min=None):
    rng = np.random.default_rng(seed)
    out = {}
    for how in v6b.BASELINE:
        out[how] = {"x": {"n_s": np.asarray(x, dtype=float),
                          "n_s_min": np.asarray(x if x_min is None else x_min, dtype=float)}}
        for ref in o6.REFS:
            out[how][ref] = {}
            for row in o6.ROWS_O:
                o = np.asarray(own, dtype=float)
                out[how][ref][row] = {"own": o, "base": np.ones(len(o)), "change": o * 2,
                                      "full_means": rng.random(2000)}
    return out


def test_bands_by_hand():
    xk = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9]
    xl = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9]  # LWP's own cuts equal KUPA's: bands = LWP's groups
    ok = np.random.default_rng(11).random(17)  # continuous, so that a changed key changes the interval
    ol = np.random.default_rng(12).random(17)
    kk, lk = fake_keep(xk, ok, 1), fake_keep(xl, ol, 2)
    _, _, blab, written = o6.plan_all(kk, lk)
    out = o6.bands(kk, lk, blab, written)
    o = out["n_s"]["halves"]
    c1, c2 = np.quantile(np.array(xk, dtype=float), [1 / 3, 2 / 3])
    assert o["cuts_from_kupa_essay"] == [c1, c2]
    gk, gl = o6.group_of(np.array(xk, float), c1, c2), o6.group_of(np.array(xl, float), c1, c2)
    assert o["persons"]["lwp"] == {g: int(np.sum(gl == j)) for j, g in enumerate(o6.GROUPS)}
    assert o["written"]
    cell = o["cells"]["R_mean"]["cross_100"]
    for j, g in enumerate(o6.GROUPS):
        b = cell["bands"][g]
        assert b["kupa_essay"] == pytest.approx(ok[gk == j].mean()) and b["lwp"] == pytest.approx(ol[gl == j].mean())
        ik = np.random.default_rng(o6.gkey("kupa", "essay", "R_mean", "cross_100", "halves", j)).integers(
            0, int(np.sum(gk == j)), (2000, int(np.sum(gk == j))))
        il = np.random.default_rng(o6.gkey("lwp", "lwp", "R_mean", "cross_100", "halves", 3 + j)).integers(
            0, int(np.sum(gl == j)), (2000, int(np.sum(gl == j))))
        d = ol[gl == j][il].mean(1) - ok[gk == j][ik].mean(1)
        assert b["ci"] == pytest.approx([np.percentile(d, 2.5), np.percentile(d, 97.5)])
        assert o["median_covariate"][g]["lwp"] == pytest.approx(np.median(np.array(xl, float)[gl == j]))
    whole = cell["whole"]
    assert whole["difference"] == pytest.approx(ol.mean() - ok.mean())
    d = lk["halves"]["R_mean"]["cross_100"]["full_means"] - kk["halves"]["R_mean"]["cross_100"]["full_means"]
    assert whole["ci"] == pytest.approx([np.percentile(d, 2.5), np.percentile(d, 97.5)])
    # n_s_min uses keys 10+ and 13+
    b2 = out["n_s_min"]["halves"]["cells"]["R_mean"]["cross_100"]["bands"]["low"]
    assert b2["kupa_essay"] == cell["bands"]["low"]["kupa_essay"] and b2["ci"] != cell["bands"]["low"]["ci"]


def test_bands_not_written_if_one_band_small():
    xk = list(range(15))
    xl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # high band (> c2 = 9.33) has 4
    kk, lk = fake_keep(xk, np.ones(15), 1), fake_keep(xl, np.ones(14), 2)
    _, _, blab, written = o6.plan_all(kk, lk)
    o = o6.bands(kk, lk, blab, written)["n_s"]["oddeven"]
    assert o["persons"]["lwp"]["high"] == 4 and not o["written"]
    assert "bands" not in o["cells"]["R_loc"]["inside_100"] and "median_covariate" not in o
    assert "whole" in o["cells"]["R_loc"]["inside_100"]


# ---------------------------------------------------------------- what may be written


def test_small_derivable_cases():
    n = 20
    full = np.ones(n, bool)
    a = np.zeros(n, bool); a[:10] = True
    assert not o6.small_derivable([full, a, ~a])
    b = np.zeros(n, bool); b[:12] = True  # nested in-between: a set of 2 persons is derivable (b minus a)
    assert o6.small_derivable([full, a, b])
    c = np.zeros(n, bool); c[:15] = True  # difference of 5: fine
    assert not o6.small_derivable([full, a, c])
    d = np.zeros(n, bool); d[16:] = True  # 4 persons directly
    assert o6.small_derivable([full, d])
    # not nested, but derivable by a combination: x = {0..9}, y = {5..14}, z = {0..4, 10..14} -> x + y - z = 2*{5..9}
    # (5 persons, fine); with y' = {7..16} a 3-person set {7,8,9} = x ∩ y' needs more: not derivable from x, y' alone
    x = np.zeros(n, bool); x[:10] = True
    y = np.zeros(n, bool); y[7:17] = True
    assert not o6.small_derivable([full, x, y])
    with pytest.raises(SystemExit):
        o6.small_derivable([])


def test_plan_written_withholds_nested_bands():
    # LWP's own groups and KUPA's bands differ by 2 persons at the low end
    gk = {"n_s": np.repeat([0, 1, 2], 6), "n_s_min": np.repeat([0, 1, 2], 6)}
    gl_own = np.repeat([0, 1, 2], 7)
    bl = gl_own.copy(); bl[5:7] = 1  # low band has 5, i.e. own low minus 2 persons
    w = o6.plan_written(gk, {"n_s": gl_own, "n_s_min": gl_own}, {"n_s": bl, "n_s_min": bl})
    assert w["kupa_essay"] == {"n_s": True, "n_s_min": True} and w["lwp_lwp"]["n_s"]
    assert w["bands"] == {"n_s": False, "n_s_min": False}
    # bands that are LWP's own groups add nothing: written
    w = o6.plan_written(gk, {"n_s": gl_own, "n_s_min": gl_own}, {"n_s": gl_own, "n_s_min": gl_own})
    assert w["bands"] == {"n_s": True, "n_s_min": True}


def test_plan_written_withholds_n_s_min_nested_in_n_s():
    g = np.repeat([0, 1, 2], 7)
    gm = g.copy(); gm[7:9] = 0  # n_s_min low = n_s low plus 2 persons
    w = o6.plan_written({"n_s": g, "n_s_min": g}, {"n_s": g, "n_s_min": gm}, {"n_s": g, "n_s_min": gm})
    assert w["lwp_lwp"] == {"n_s": True, "n_s_min": False}
    assert w["bands"]["n_s"] and not w["bands"]["n_s_min"]


def test_plan_written_small_group_and_kupa_needed():
    small = np.array([0] * 4 + [1] * 9 + [2] * 8)
    ok = np.repeat([0, 1, 2], 7)
    w = o6.plan_written({"n_s": small, "n_s_min": ok}, {"n_s": ok, "n_s_min": ok}, {"n_s": ok, "n_s_min": ok})
    assert not w["kupa_essay"]["n_s"] and not w["bands"]["n_s"]  # KUPA's bands are its groups
    assert w["kupa_essay"]["n_s_min"]


def test_bands_unwritten_by_plan_write_no_bands():
    xk = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9]
    xl = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9, 9]
    kk, lk = fake_keep(xk, np.ones(17), 1), fake_keep(xl, np.ones(18), 2)
    labk, labl, blab, written = o6.plan_all(kk, lk)
    for how in v6b.BASELINE:
        written[how]["bands"]["n_s"] = False
    out = o6.bands(kk, lk, blab, written)
    assert not out["n_s"]["halves"]["written"] and "bands" not in out["n_s"]["halves"]["cells"]["R_loc"]["inside_75"]
    for how in v6b.BASELINE:
        written[how]["bands"]["n_s"] = True
    lk2 = fake_keep(xl[:14], np.ones(14), 2)
    _, _, blab2, _ = o6.plan_all(kk, lk2)
    with pytest.raises(SystemExit, match="under 5"):
        o6.bands(kk, lk2, blab2, written)


def test_tied_at_cut_and_median_only():
    x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], float)
    kk = fake_keep(x, np.ones(18), 1)
    lab = o6.labels_of(kk)
    written = {how: {"kupa_essay": {"n_s": True, "n_s_min": True}} for how in v6b.BASELINE}
    out = o6.set_groups(kk, lab, written, "kupa", "essay")
    o = out["n_s"]["halves"]
    c1, c2 = np.quantile(x, [1 / 3, 2 / 3])
    assert o["tied_at_cut"] == [int(np.sum(x == c1)), int(np.sum(x == c2))]
    assert set(o["covariate_median_by_group"]) == set(o6.GROUPS)
    assert all(isinstance(v, float) for v in o["covariate_median_by_group"].values())
    assert o["base_by_group"]["R_loc"] == {"low": 1.0, "mid": 1.0, "high": 1.0}
    written = {how: {"kupa_essay": {"n_s": False, "n_s_min": True}} for how in v6b.BASELINE}
    o = o6.set_groups(kk, lab, written, "kupa", "essay")["n_s"]["halves"]
    assert o["covariate_median_by_group"] is None and o["base_by_group"]["R_loc"] is None
    assert not o["cells"]["R_loc"]["inside_100"]["written"]


def test_set_keep_stops_on_persons_and_base(run, monkeypatch):
    _, rn, args = run
    ks_, kid, kp, ls_, lid, lp = args
    real = o6.cell_persons
    monkeypatch.setattr(o6, "cell_persons", lambda hv, how, row, ref: real(hv, how, row, ref)[1:]
                        if row == "cross_25" else real(hv, how, row, ref))
    with pytest.raises(SystemExit, match="persons with R differ"):
        o6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn)
    monkeypatch.setattr(o6, "cell_persons", real)
    realc = o6.contrib_base_change
    calls = {"R_mean": 0}

    def shifted(f, s_, ref):
        own, base, change = realc(f, s_, ref)
        if ref == "R_mean":
            calls["R_mean"] += 1
            if calls["R_mean"] == 2:  # the second row of R_mean
                base = base.copy()
                base[0] += 1e-9
        return own, base, change
    monkeypatch.setattr(o6, "contrib_base_change", shifted)
    with pytest.raises(SystemExit, match="own distance depends on the row"):
        o6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn)


def test_set_keep_stops_on_a_missing_v6n_cell(run):
    _, rn, args = run
    ks_, kid, kp, ls_, lid, lp = args
    bad = copy.deepcopy(rn)
    del bad["count_1"]["kupa_essay"]["oddeven"]["cross_50"]
    with pytest.raises(SystemExit, match="not in v6n's record"):
        o6.count_all(ks_, kid, kp, ls_, lid, lp, M, bad)


def test_fake_run_writes_n_s_min_and_checks_derivability(run):
    rec, _, _ = run
    assert any(rec["groups"]["n_s"][nm][how]["written"] for nm in ("kupa_essay", "lwp_lwp")
               for how in v6b.BASELINE)
    for how in v6b.BASELINE:
        for nm in ("kupa_essay", "lwp_lwp"):
            sets = []
            for c in o6.COVARIATES:
                o = rec["groups"][c][nm][how]
                assert o["written"] == rec["written"][how][nm][c]


def test_predictions_p3_counts():
    def r(lower):
        rec = pred_rec("above 0", "above 0", (2.0, 1.0), {"oddeven": "includes 0", "halves": "includes 0"})
        k = 0
        for ref in o6.REFS:
            for nm in ("kupa_essay", "lwp_lwp"):
                for how in v6b.BASELINE:
                    b = rec["groups"]["n_s"][nm][how]["base_by_group"]
                    b[ref] = {"low": 2.0, "mid": 0, "high": 1.0} if k < lower else {"low": 1.0, "mid": 0, "high": 2.0}
                    k += 1
        return o6.predictions(rec)["P3"]
    assert r(6).startswith("held (6 of 8)") and r(5).startswith("did not hold (5 of 8)")



def lab_of(c1, c2, n=60):
    x = np.arange(n)
    return np.where(x < c1, 0, np.where(x < c2, 1, 2))


def test_plan_written_counts_the_bands_already_written():
    a, b, c = lab_of(20, 40), lab_of(10, 50), lab_of(12, 45)  # c is 2 persons from b, but far from a
    w = o6.plan_written({"n_s": a, "n_s_min": a}, {"n_s": a, "n_s_min": c}, {"n_s": b, "n_s_min": a})
    assert w["bands"]["n_s"] and not w["lwp_lwp"]["n_s_min"]
    w = o6.plan_written({"n_s": a, "n_s_min": a}, {"n_s": a, "n_s_min": c}, {"n_s": a, "n_s_min": a})
    assert w["lwp_lwp"]["n_s_min"]  # without b, c alone is fine


def test_plan_written_bands_need_kupas_written_groups():
    a, near = lab_of(20, 40), lab_of(22, 40)  # KUPA's n_s_min 2 persons from its n_s: withheld by derivability
    w = o6.plan_written({"n_s": a, "n_s_min": near}, {"n_s": a, "n_s_min": a}, {"n_s": a, "n_s_min": a})
    assert not w["kupa_essay"]["n_s_min"] and not w["bands"]["n_s_min"] and w["bands"]["n_s"]


def test_contrib_ties_count_half():
    v = (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))  # parts equal: base 0
    w = (np.array([2.0, 1.0, 0.0]), np.array([0.0, 1.0, 5.0]))
    first = [({"R_loc": v[0]}, {"R_loc": v[1]}), ({"R_loc": w[0]}, {"R_loc": w[1]})]
    sel = [({"R_loc": v[0]}, {"R_loc": v[1]}), ({"R_loc": w[0] + 9}, {"R_loc": w[1] + 9})]
    own, base, change = o6.contrib_base_change(first, sel, "R_loc")
    assert base[0] == 0 and change[0] == 0 and own[0] == 0.5 and own[1] == 1.0


def test_rising_needs_mid():
    own = np.concatenate([np.full(5, 0.2), np.full(5, 0.1), np.full(5, 0.9)])
    g = np.repeat([0, 1, 2], 5)
    out = o6.by_group(own, own + 1, own, g, "kupa", "essay", "R_loc", "inside_75", "halves", 0)
    assert out["rising"] is False
    own2 = np.concatenate([np.full(5, 0.1), np.full(5, 0.2), np.full(5, 0.9)])
    assert o6.by_group(own2, own2 + 1, own2, g, "kupa", "essay", "R_loc", "inside_75", "halves", 0)["rising"]


def test_ratio_median_uses_positive_bases_only():
    own = np.ones(7)
    base = np.array([0, 0, 1, 1, 1, 1, 1.0])
    change = np.array([5, 5, 1, 2, 3, 4, 5.0])
    st, _ = o6.group_stats(own, base, change, [1, 2, 3])
    assert st["median_change_over_base"] == pytest.approx(3.0)  # with the zero bases it would be 4


def test_set_keep_contribution_guard(run, monkeypatch):
    _, rn, args = run
    ks_, kid, kp, ls_, lid, lp = args
    realc = o6.contrib_base_change

    def flipped(f, s_, ref):
        own, base, change = realc(f, s_, ref)
        return 1.0 - own, base, change
    monkeypatch.setattr(o6, "contrib_base_change", flipped)
    with pytest.raises(SystemExit, match="do not give v6n's share"):
        o6.count_all(ks_, kid, kp, ls_, lid, lp, M, rn)


def test_whole_difference_uses_v6ns_keys(run):
    rec, rn, args = run
    ks_, kid, kp, ls_, lid, lp = args
    # recompute the full-cell means of both sets from v6n's keys
    hvs = {}
    for ds, task, series, ids, place in (("kupa", "essay", ks_, kid, kp), ("lwp", "lwp", ls_, lid, lp)):
        b = n6.build(series, ids, ds, task, place, M, o6.BUILD_ROWS)
        hvs[ds] = n6.hv_of(b, o6.BUILD_ROWS)
    how, ref, row = "oddeven", "R_loc", "cross_50"  # shares strictly between 0 and 1 in both fake sets
    means = {}
    for ds, task in o6.SETS:
        hv = hvs[ds]
        ok = o6.cell_persons(hv, how, row, ref)
        own, _, _ = o6.contrib_base_change([hv[how]["text/before"][i] for i in ok], [hv[how][row][i] for i in ok],
                                           ref)
        idx = np.random.default_rng(n6.key1(ds, task, ref, row, how, 0)).integers(0, len(own), (2000, len(own)))
        means[ds] = own[idx].mean(1)
    d = means["lwp"] - means["kupa"]
    assert rec["bands"]["n_s"][how]["cells"][ref][row]["whole"]["ci"] == pytest.approx(
        [np.percentile(d, 2.5), np.percentile(d, 97.5)])


def test_predictions_p1_needs_both_baselines():
    r = pred_rec("above 0", "above 0", (2.0, 1.0), {"oddeven": "includes 0", "halves": "includes 0"})
    r["groups"]["n_s"]["kupa_essay"]["halves"]["cells"]["R_loc"]["inside_100"]["high_minus_low"]["ci"] = [-0.1, 0.2]
    assert o6.predictions(r)["P1"] == "did not hold"
