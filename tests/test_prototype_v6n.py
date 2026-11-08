import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6n as n6  # noqa: E402
from test_prototype_v6k import fake_person  # noqa: E402
from test_prototype_v6m import many_sentences  # noqa: E402

m6, l6, k6, ks = n6.m6, n6.l6, n6.k6, n6.ks
v6, v6b = k6.v6, k6.v6b
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
RK = json.loads((ROOT / "results/prototype_v6k/record.json").read_text())
RL = json.loads((ROOT / "results/prototype_v6l/record.json").read_text())


def rec_of(s, kinds):
    return k6.Rec(s.times, s.ranges("before"), v6b.labels(s, kinds), M)


# ---------------------------------------------------------------- the conditions' constants


def test_constants_follow_the_conditions():
    c = json.loads((ROOT / "works/conditions/prototype_v6n.json").read_text())
    assert n6.STAGE == {**m6.STAGE, "R_mean": 12}
    assert [p for _, p in n6.STEPS] == [0.25, 0.5, 0.75, 1.0]
    assert n6.ROWS[:7] == k6.ROWS and n6.ROWS[7:] == ["inside_25", "inside_50", "inside_75", "inside_100",
                                                      "cross_25", "cross_50", "cross_75", "cross_100"]
    assert {r: n6.ROW_IX[r] for r in m6.ROWS} == m6.ROW_IX
    assert n6.ROW_IX["inside_75"] == 14 and n6.ROW_IX["cross_75"] == 15 and 13 not in n6.ROW_IX.values()
    assert n6.key1("lwp", "lwp", "R_mean", "cross_75", "halves", 0) == [20261031, 95, 2, 0, 12, 15, 1, 0, 0]
    assert n6.key1("kupa", "essay", "R_mean", "inside_75", "oddeven", n6.gate_of("I")) == [20261031, 95, 0, 1, 12,
                                                                                          14, 0, 0, 6]
    for k in m6.KEYS:  # v6m's cells keep v6m's keys
        for row in m6.ROWS[1:]:
            assert n6.key1("kupa", "essay", k, row, "halves", 3) == m6.key1("kupa", "essay", k, row, "halves", 3)
    assert n6.REL_KEY == 98 and n6.REL_DIMS["R_mean"] == 6 and n6.REL_STAGES[-1] == "R_mean"
    assert "98" in c["part_3_reliability"]["splits"] and "12" in c["stages"]["R_mean"]
    assert "14" in c["rows"]["row_index"] and "15" in c["rows"]["row_index"]
    assert n6.GATES == ("R", "R_loc", "R_lag", "R_mean")


# ---------------------------------------------------------------- R_mean


def test_r_mean_by_hand():
    gaps = np.array([3000.0, 100.0, 400.0, 200.0, 2500.0, 150.0, 500.0, 120.0])
    lab = np.array(["sentence", "within", "word", "within", "subsentence", "within", "word", "within"])
    held = np.arange(8)
    r = n6.r_mean_of(gaps, lab, held)
    lw = np.log([100.0, 200.0, 150.0, 120.0]).mean()
    assert r[0] == pytest.approx(np.log(3000.0))
    assert r[1] == pytest.approx(np.log(2500.0))
    assert r[2] == pytest.approx(np.log([400.0, 500.0]).mean())
    assert r[3] == pytest.approx(lw)
    assert r[4] == pytest.approx(np.log(3000.0) - lw)
    assert r[5] == pytest.approx(1.0)
    assert n6.r_mean_of(gaps, lab, np.arange(1, 4)) is None  # no sentence gap


def test_r_mean_is_a_mean_for_every_label():
    vals = {"sentence": [2100.0, 2400.0, 9000.0], "subsentence": [300.0, 350.0, 4000.0],
            "word": [400.0, 500.0, 5000.0], "within": [90.0, 100.0, 900.0]}
    gaps, lab = [], []
    for k, v in vals.items():
        gaps += v
        lab += [k] * len(v)
    r = n6.r_mean_of(np.array(gaps), np.array(lab), np.arange(len(gaps)))
    for i, k in enumerate(n6.LABS):
        assert r[i] == pytest.approx(np.log(vals[k]).mean())
        assert abs(r[i] - np.median(np.log(vals[k]))) > 0.1
    assert r[4] == pytest.approx(r[0] - r[3])
    long_ = np.array(gaps) >= 2000.0
    assert r[5] == pytest.approx(np.array(gaps)[long_ & np.isin(lab, ["sentence", "subsentence"])].sum()
                                 / np.array(gaps)[long_].sum())


def test_r_mean_differs_from_median_where_long_gaps_are_few():
    gaps = np.array([3000.0, 100.0, 300.0, 100.0, 310.0, 100.0, 320.0, 100.0, 9000.0, 100.0])
    lab = np.array(["sentence", "within", "sentence", "within", "sentence", "within", "subsentence", "word",
                    "word", "within"])
    r = n6.r_mean_of(gaps, lab, np.arange(10))
    assert r[0] == pytest.approx(np.log([3000.0, 300.0, 310.0]).mean())
    assert r[0] > np.median(np.log([3000.0, 300.0, 310.0]))
    assert r[2] == pytest.approx(np.log([100.0, 9000.0]).mean())


@pytest.mark.parametrize("seed", range(5))
def test_part_vectors_n(seed):
    s, kinds = fake_person(seed, 12)
    rec = rec_of(s, kinds)
    for idx in (np.arange(0, rec.n, 2), np.arange(rec.n // 2), np.array([1, 4, 5])):
        out, _ = n6.part_vectors_n(rec, idx)
        ref, _ = m6.part_vectors_m(rec, idx)
        for k, v in ref.items():
            assert (v is None and out[k] is None) or np.array_equal(v, out[k])
        held = np.concatenate([rec.held[i] for i in idx])
        assert np.array_equal(out["R_mean"], n6.r_mean_of(rec.gaps, rec.lab, np.sort(held)))
        assert out["R_mean"][5] == out["R_loc"][5]
        assert set(out) >= set(n6.KEYS)


def test_add_r_mean_stops_on_existence_and_sixth(monkeypatch):
    s, kinds = fake_person(1, 10)
    rec = rec_of(s, kinds)
    out, _ = m6.part_vectors_m(rec, np.arange(rec.n))
    bad = dict(out)
    bad["R_loc"] = None
    with pytest.raises(SystemExit, match="same part"):
        n6.add_r_mean(bad, rec, np.arange(rec.n))
    bad = dict(out)
    bad["R_loc"] = out["R_loc"].copy()
    bad["R_loc"][5] += 0.01
    with pytest.raises(SystemExit, match="sixth"):
        n6.add_r_mean(bad, rec, np.arange(rec.n))


# ---------------------------------------------------------------- the dose rows


@pytest.mark.parametrize("seed", range(10))
def test_dose_rows_reproduce_v6m_and_nest(seed):
    s, _ = fake_person(seed, 14)
    key = [0, 1, seed]
    t_n, tal_n = n6.dose_rows(s, key)
    t_m, tal_m = m6.dose_rows(s, key)
    assert set(t_n) == set(n6.DOSE_ROWS)
    for row in m6.DOSE_ROWS:
        assert np.array_equal(t_n[row], t_m[row])
        for f, v in tal_m[row].items():
            if f == "same_as_next_step" and row.endswith("_50"):
                continue
            assert tal_n[row][f] == v, (row, f)
    g = np.diff(s.times)
    for fam in n6.FAMILIES:
        made = [tal_n[f"{fam}_{st}"]["made"] for st, _ in n6.STEPS]
        assert all(made[i + 1][:len(made[i])] == made[i] for i in range(3))
        m = tal_n[f"{fam}_75"]["m"]
        assert len(made[2]) + tal_n[f"{fam}_75"]["left_without_partner"] == m6.n_at(0.75, m)
        for st, _ in n6.STEPS:
            row = f"{fam}_{st}"
            assert tal_n[row]["positions_changed"] == int(np.sum(np.abs(np.diff(t_n[row]) - g) > ks.TOL_MS))
    n6.check_rows(g, t_n, k6.row_times(s, "inside/before", key), tal_n)


def test_same_as_next_step_at_50_compares_with_75():
    s = many_sentences(n_sent=5, inner=3)  # 4 givers: 25 -> 1, 50 -> 2, 75 -> 3, 100 -> 4
    _, tal = n6.dose_rows(s, [0, 1, 3])
    assert [tal[f"cross_{st}"]["swaps"] for st, _ in n6.STEPS] == [1, 2, 3, 4]
    assert not tal["cross_50"]["same_as_next_step"]
    s = many_sentences(n_sent=3, inner=3)  # 2 givers: 25 -> 1, 50 -> 1, 75 -> 2, 100 -> 2
    _, tal = n6.dose_rows(s, [0, 1, 3])
    assert [tal[f"cross_{st}"]["swaps"] for st, _ in n6.STEPS] == [1, 1, 2, 2]
    assert tal["cross_25"]["same_as_next_step"] and not tal["cross_50"]["same_as_next_step"]
    assert tal["cross_75"]["same_as_next_step"] and not tal["cross_100"]["same_as_next_step"]
    _, tal_m = m6.dose_rows(s, [0, 1, 3])
    assert not tal_m["cross_50"]["same_as_next_step"]  # v6m's 50 against 100


def test_plan_overlap_stops(monkeypatch):
    s, _ = fake_person(2, 12)
    monkeypatch.setattr(m6, "inside_plan", lambda s, key: [(1, 5), (5, 9)])
    with pytest.raises(SystemExit, match="two swaps"):
        n6.dose_rows(s, [0, 1, 0])


def test_check_rows_stops_on_75_not_nested():
    s, _ = fake_person(3, 14)
    key = [0, 1, 3]
    t, tal = n6.dose_rows(s, key)
    g = np.diff(s.times)
    bad = dict(t)
    bad["cross_75"] = t["cross_25"]
    if not np.array_equal(t["cross_25"], t["cross_50"]):
        with pytest.raises(SystemExit, match="not nested"):
            n6.check_rows(g, bad, None)
    bad_tal = copy.deepcopy(tal)
    bad_tal["inside_75"]["made"] = list(reversed(bad_tal["inside_75"]["made"]))
    if len(bad_tal["inside_75"]["made"]) > 1:
        with pytest.raises(SystemExit, match="swaps of the doses"):
            n6.check_rows(g, t, None, bad_tal)
    bad = dict(t)
    bad["inside_100"] = t["inside_75"]
    if not np.array_equal(t["inside_75"], t["inside_100"]):
        with pytest.raises(SystemExit):
            n6.check_rows(g, bad, k6.row_times(s, "inside/before", key))


# ---------------------------------------------------------------- persons, rows and construction


@pytest.mark.parametrize("seed", range(4))
def test_person_rows_and_construction(seed):
    s, kinds = fake_person(10 + seed, 12)
    rows = n6.ROWS
    vec, whole, tallies, dose_tally, text_rec = n6.person_rows(s, kinds, [0, 1, seed], M, rows)
    n6.check_construction(vec, tallies, whole)
    assert set(vec) == set(rows) and "text/after" not in whole and "inside_75" in whole
    assert np.array_equal(text_rec.gaps, np.diff(s.times))
    vm, wm, _, _ = m6.person_rows(s, kinds, [0, 1, seed], M, m6.ROWS)
    for row in m6.ROWS:
        for how in v6b.BASELINE:
            for h in (0, 1):
                for k in m6.KEYS:
                    a, b = vm[row][how][h][k], vec[row][how][h][k]
                    assert (a is None and b is None) or np.array_equal(a, b)
    for k in ("U@2000", "B@2000", "I"):  # the whole-record identities
        for row in ("inside_75", "cross_75", "shuffled/before"):
            j = 0 if k != "U@2000" else 2
            if k == "U@2000" and not row.startswith("inside"):
                continue
            assert abs(whole[row][k][j] - whole["text/before"][k][j]) <= 1e-9


def test_check_construction_stops_on_whole_and_inside_75():
    s, kinds = fake_person(21, 12)
    vec, whole, tallies, _, _ = n6.person_rows(s, kinds, [0, 1, 0], M, n6.ROWS)
    bad = copy.deepcopy(whole)
    bad["cross_50"]["B@2000"] = bad["cross_50"]["B@2000"] + 0.5
    with pytest.raises(SystemExit, match="whole B@2000"):
        n6.check_construction(vec, tallies, bad)
    bad = copy.deepcopy(whole)
    bad["inside_75"]["U@2000"] = bad["inside_75"]["U@2000"] + np.array([0, 0, 0, 0.1])
    with pytest.raises(SystemExit, match="whole U@2000"):
        n6.check_construction(vec, tallies, bad)
    bad = copy.deepcopy(whole)
    bad["cross_75"]["U@2000"] = bad["cross_75"]["U@2000"] + 0.1  # not fixed under cross: no stop
    n6.check_construction(vec, tallies, bad)
    bad_vec = copy.deepcopy(vec)
    a, b = bad_vec["inside_75"]["halves"]
    a = dict(a)
    a["S1"] = a["S1"] + 0.1
    bad_vec["inside_75"]["halves"] = (a, b)
    with pytest.raises(SystemExit, match="inside_75 halves S1"):
        n6.check_construction(bad_vec, tallies, whole)


# ---------------------------------------------------------------- count 1 and rule persons


def fake_hv(n=30, rows=("text/before", "inside_75", "cross_75", "cross_100", "text/none"), seed=0):
    hv = {how: {row: [] for row in rows} for how in v6b.BASELINE}
    for i in range(n):
        s, kinds = fake_person(100 + i + seed, 10)
        vec, _, _, _, _ = n6.person_rows(s, kinds, [0, 1, i], M, list(dict.fromkeys(["text/before", *rows])))
        for how in v6b.BASELINE:
            for row in rows:
                hv[how][row].append(vec[row][how])
    return hv


@pytest.fixture(scope="module")
def hv():
    return fake_hv()


def test_one_cell_r_mean_text_none(hv):
    assert n6.one_cell(hv, "text/none", "halves", "R_mean", [1]) == {"defined": False, "persons": 0}
    assert n6.one_cell(hv, "text/none", "halves", "R_loc", [1]) == {"defined": False, "persons": 0}


def test_rule_cell_reproduces_v6m_and_pairs(hv):
    c = n6.rule_cell(hv, "kupa", "essay", "cross_100", "oddeven", "U@2000")
    cm_ = m6.rule_cell(hv, "kupa", "essay", "cross_100", "oddeven", "U@2000")
    for part, v in cm_.items():
        assert c[part] == v
    assert c["R_mean"]["persons"] == c["R_loc"]["persons"] == c["stage"]["persons"]
    gate = n6.gate_of("U@2000")
    assert c["R_mean"] == m6.one_cell(hv, "cross_100", "oddeven", "R_mean",
                                      n6.key1("kupa", "essay", "R_mean", "cross_100", "oddeven", gate), extra="U@2000")
    first, sel = hv["oddeven"]["text/before"], hv["oddeven"]["cross_100"]
    ok = [i for i in range(len(first)) if all(v6b.has(first[i], x) and v6b.has(sel[i], x) for x in ("U@2000", "R"))]
    f_, s_ = [first[i] for i in ok], [sel[i] for i in ok]
    idx = np.random.default_rng(n6.key1("kupa", "essay", "U@2000", "cross_100", "oddeven", gate)).integers(
        0, len(ok), (v6.N_BOOT, len(ok)))
    for pk, a, b in (("paired_minus_R_mean", "U@2000", "R_mean"), ("paired_R_mean_minus_R_loc", "R_mean", "R_loc")):
        d = m6.own_contrib(f_, s_, a) - m6.own_contrib(f_, s_, b)
        assert c[pk]["ci"] == v6.ci(d, idx) and c[pk]["persons"] == len(ok)
    for pk, a, b in (("paired_minus_R_mean", "stage", "R_mean"), ("paired_R_mean_minus_R_loc", "R_mean", "R_loc")):
        d = c[pk]
        assert d["difference"] == pytest.approx(c[a]["share_vs_own"] - c[b]["share_vs_own"])
        assert d["ci"][0] <= d["difference"] <= d["ci"][1]


def test_paired_uses_the_given_key(hv):
    c1 = {k: n6.one_cell(hv, "cross_75", "halves", k, n6.key1("lwp", "lwp", k, "cross_75", "halves", 0))
          for k in ("R_mean", "R_loc")}
    p1 = n6.paired(hv, "cross_75", "halves", "R_mean", "R_loc", ("R_mean",), c1["R_mean"], c1["R_loc"], [1, 2], "x")
    p2 = n6.paired(hv, "cross_75", "halves", "R_mean", "R_loc", ("R_mean",), c1["R_mean"], c1["R_loc"], [1, 3], "x")
    assert p1["difference"] == p2["difference"] and p1["persons"] == c1["R_mean"]["persons"]
    first, sel = hv["halves"]["text/before"], hv["halves"]["cross_75"]
    a = m6.own_contrib(first, sel, "R_mean") - m6.own_contrib(first, sel, "R_loc")
    idx = np.random.default_rng([1, 2]).integers(0, len(first), (v6.N_BOOT, len(first)))
    assert p1["ci"] == v6.ci(a, idx)


def test_paired_stops_on_wrong_share(hv):
    c1 = {k: n6.one_cell(hv, "cross_75", "halves", k, [5]) for k in ("R_mean", "R_loc")}
    bad = dict(c1["R_mean"])
    bad["share_vs_own"] += 0.01
    with pytest.raises(SystemExit, match="contributions"):
        n6.paired(hv, "cross_75", "halves", "R_mean", "R_loc", ("R_mean",), bad, c1["R_loc"], [1], "x")
    bad = dict(c1["R_mean"])
    bad["persons"] -= 1
    with pytest.raises(SystemExit, match="paired persons"):
        n6.paired(hv, "cross_75", "halves", "R_mean", "R_loc", ("R_mean",), bad, c1["R_loc"], [1], "x")


def test_gate0_refs(hv):
    c1 = {how: {row: {k: n6.one_cell(hv, row, how, k, n6.key1("lwp", "lwp", k, row, how, 0)) for k in n6.REFS}
                for row in hv[how] if row != "text/before"} for how in v6b.BASELINE}
    rows_backup = n6.ROWS
    try:
        n6.ROWS = ["text/before", "inside_75", "cross_75", "cross_100", "text/none"]
        g0 = n6.gate0_refs(hv, c1, "lwp", "lwp")
        assert "text/none" not in g0["halves"]
        d = g0["halves"]["cross_75"]
        first, sel = hv["halves"]["text/before"], hv["halves"]["cross_75"]
        ok = [i for i in range(len(first)) if v6b.has(first[i], "R_mean") and v6b.has(sel[i], "R_mean")]
        f_, s_ = [first[i] for i in ok], [sel[i] for i in ok]
        dd = m6.own_contrib(f_, s_, "R_mean") - m6.own_contrib(f_, s_, "R_loc")
        idx = np.random.default_rng(n6.key1("lwp", "lwp", "R_mean", "cross_75", "halves", 0)).integers(
            0, len(ok), (v6.N_BOOT, len(ok)))
        assert d["ci"] == v6.ci(dd, idx)
        bad = copy.deepcopy(c1)
        bad["halves"]["cross_75"]["R_loc"] = {"too_few_persons": True, "persons": 1}
        with pytest.raises(SystemExit, match="not both counted"):
            n6.gate0_refs(hv, bad, "lwp", "lwp")
        assert d["difference"] == pytest.approx(c1["halves"]["cross_75"]["R_mean"]["share_vs_own"]
                                                - c1["halves"]["cross_75"]["R_loc"]["share_vs_own"])
        bad = copy.deepcopy(c1)
        bad["halves"]["cross_75"]["R_loc"]["persons"] -= 1
        with pytest.raises(SystemExit, match="different persons"):
            n6.gate0_refs(hv, bad, "lwp", "lwp")
    finally:
        n6.ROWS = rows_backup


# ---------------------------------------------------------------- part 1 and part 2 helpers


def cell(lo, hi, share=None):
    return {"share_vs_own": (lo + hi) / 2 if share is None else share, "share_vs_own_ci": [lo, hi], "persons": 9}


def test_ladder():
    e = n6.ladder([cell(0.1, 0.2), cell(0.4, 0.62), cell(0.61, 0.7), cell(0.7, 0.9)])
    assert e["classes"] == {"25": "below", "50": "undecided", "75": "above", "100": "above"}
    assert e["first_step_above"] == "75" and e["rising"] is True
    e = n6.ladder([cell(0.1, 0.2), cell(0.3, 0.4, 0.35), cell(0.3, 0.4, 0.35), cell(0.2, 0.3)])
    assert e["first_step_above"] == "none" and e["rising"] is False
    e = n6.ladder([cell(0.1, 0.2), cell(0.1, 0.2), cell(0.1, 0.2), cell(0.1, 0.2)])
    assert e["rising"] is True  # equal shares rise
    assert n6.ladder([cell(0.7, 0.8)] * 4, fixed=True)["first_step_above"] == "not read"
    assert n6.ladder([{"too_few_persons": True}] * 4)["rising"] is None
    e = n6.ladder([cell(0.1, 0.2), {"too_few_persons": True}, cell(0.7, 0.8), cell(0.7, 0.8)])
    assert e["first_step_above"] == "not evaluable"
    e = n6.ladder([cell(0.7, 0.8), {"too_few_persons": True}, cell(0.7, 0.8), cell(0.7, 0.8)])
    assert e["first_step_above"] == "25"


def test_fixed_feature():
    assert n6.fixed_feature("cross_25", "B@2000", "0") and n6.fixed_feature("cross_25", "I", "0")
    assert not n6.fixed_feature("cross_25", "I", "1")
    assert n6.fixed_feature("inside_75", "U@2000", "3") and not n6.fixed_feature("cross_75", "U@2000", "3")
    assert not n6.fixed_feature("inside_75", "R_mean", "0")


def test_signed_set_keys():
    pairs = [({st: np.zeros(d) for st, d in (("U@2000", 4), ("B@2000", 1), ("Bsd@2000", 1), ("I", 3), ("R_loc", 6),
                                              ("R_lag", 1), ("B_complete@2000", 1), ("R_mean", 6))},
              {st: np.full(d, 0.1 * (i + 1)) for st, d in (("U@2000", 4), ("B@2000", 1), ("Bsd@2000", 1), ("I", 3),
                                                           ("R_loc", 6), ("R_lag", 1), ("B_complete@2000", 1),
                                                           ("R_mean", 6))}) for i in range(8)]
    out = n6.signed_set(pairs, 2, 0, 15)
    assert set(out) == set(n6.SIGNED_STAGES) and set(out["R_mean"]) == {str(i) for i in range(6)}
    d = np.array([0.1 * (i + 1) for i in range(8)])
    assert out["R_mean"]["2"] == m6.signed(d, [n6.SEED, 97, 2, 0, 12, 15, 2])


# ---------------------------------------------------------------- reliability


@pytest.mark.parametrize("seed", range(3))
def test_fast_n_equals_part_vectors(seed):
    s, kinds = fake_person(40 + seed, 13)
    rec = rec_of(s, kinds)
    f = n6.FastN(rec)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        perm = rng.permutation(rec.n)
        idx = np.sort(perm[:rec.n // 2])
        ref, _ = n6.part_vectors_n(rec, idx)
        assert l6.same_vectors(f.vectors(idx), ref, n6.REL_STAGES)


def texts_of(n=14, seed=0):
    out = []
    for i in range(n):
        s, kinds = fake_person(500 + i + seed, 11)
        vec, _, _, _, rec = n6.person_rows(s, kinds, [0, 1, i], M, ["text/before"])
        out.append((rec, i, vec["text/before"]))
    return out


@pytest.fixture(scope="module")
def texts():
    return texts_of()


def test_reliability_runs_and_uses_key_98(texts):
    rel = n6.reliability(texts, "lwp", "lwp", n_splits=6)
    assert rel["persons"] == 14 and rel["splits"] == 6
    assert set(rel["features"]) >= {f"R_mean[{j}]" for j in range(6)} | {"R[0]", "I[2]", "B@2000[0]"}
    # five splits by hand with key 98 (each person's generator draws its permutations in split order)
    rngs = [np.random.default_rng([n6.SEED, 98, 2, 0, place]) for _, place, _ in texts]
    rs = []
    for _ in range(5):
        perms = [g.permutation(rec.n) for g, (rec, _, _) in zip(rngs, texts)]
        A = np.array([n6.flat(n6.part_vectors_n(rec, np.sort(p[:rec.n // 2]))[0], "R_mean")[0]
                      for (rec, _, _), p in zip(texts, perms)])
        B = np.array([n6.flat(n6.part_vectors_n(rec, np.sort(p[rec.n // 2:]))[0], "R_mean")[0]
                      for (rec, _, _), p in zip(texts, perms)])
        ok = ~(np.isnan(A) | np.isnan(B))
        assert ok.sum() >= 10
        rs.append(float(np.corrcoef(A[ok], B[ok])[0, 1]))
    r5 = n6.reliability(texts, "lwp", "lwp", n_splits=5)["features"]["R_mean[0]"]
    assert r5["mean_r"] == pytest.approx(float(np.mean(rs)))
    assert r5["mean_rSB"] == pytest.approx(float(np.mean([2 * r / (1 + r) for r in rs])))
    other = n6.reliability(texts, "kupa", "essay", n_splits=5)["features"]["R_mean[0]"]
    assert other["mean_r"] != r5["mean_r"]  # the key carries the dataset and task
    r93 = l6.reliability(texts, "lwp", "lwp", n_splits=1)["features"]["R[0]"]
    r98 = n6.reliability(texts, "lwp", "lwp", n_splits=1)["features"]["R[0]"]
    assert r93["fixed_parts"] == r98["fixed_parts"]


def test_fixed_parts_equal_v6l(texts):
    rel = l6.reliability(texts, "kupa", "essay", n_splits=2)
    fp = n6.fixed_parts(texts, l6.REL_STAGES)
    for how in v6b.BASELINE:
        for f, v in fp[how].items():
            assert v == rel["features"][f]["fixed_parts"][how]


def test_check_v6l_fixed_stops(texts):
    rl = {"reliability": {"kupa_essay": {"features": {}}}}
    fp = n6.jsonish(n6.fixed_parts(texts, l6.REL_STAGES))
    for how in v6b.BASELINE:
        for f, v in fp[how].items():
            rl["reliability"]["kupa_essay"]["features"].setdefault(f, {"fixed_parts": {}})["fixed_parts"][how] = v
    n6.check_v6l_fixed(texts, "kupa_essay", rl)
    rl["reliability"]["kupa_essay"]["features"]["I[1]"]["fixed_parts"]["halves"]["r"] += 1e-9
    with pytest.raises(SystemExit, match="fixed-part"):
        n6.check_v6l_fixed(texts, "kupa_essay", rl)


def test_beside_v6l():
    rel = {"lwp_lwp": {"features": {"R[0]": {"mean_rSB": 0.8, "rSB_percentile_95": [0.7, 0.9]},
                                    "R_mean[0]": {"mean_rSB": 0.6, "rSB_percentile_95": [0.5, 0.7]}}}}
    rl = {"reliability": {"lwp_lwp": {"features": {"R[0]": {"mean_rSB": 0.75, "rSB_percentile_95": [0.6, 0.8]}}}}}
    b = n6.beside_v6l(rel, rl)
    assert b["lwp_lwp"]["R[0]"]["new_minus_v6l"] == pytest.approx(0.05)
    assert b["lwp_lwp"]["R_mean[0]"]["v6l_mean_rSB"] is None and b["lwp_lwp"]["R_mean[0]"]["new_minus_v6l"] is None


# ---------------------------------------------------------------- the whole run on fake persons, against v6m


def fake_kupa_lwp(n=12):
    ks_, ls_ = {}, {}
    for i in range(n):
        ks_[f"k{i}"] = {"copy": fake_person(1000 + i, 10), "essay": fake_person(2000 + i, 11)}
        ls_[f"l{i}"] = fake_person(3000 + i, 12)
    return ks_, ls_


@pytest.fixture(scope="module")
def both():
    ks_, ls_ = fake_kupa_lwp()
    kid, lid = sorted(ks_), sorted(ls_)
    kp, lp = {p: i for i, p in enumerate(kid)}, {p: i for i, p in enumerate(lid)}
    rec_n = n6.count_all(ks_, kid, kp, ls_, lid, lp, M, RL, n_splits=5)
    rec_m = n6.jsonish(m6.count_new(ks_, kid, kp, ls_, lid, lp, M, RK, RL))
    return rec_n, rec_m


def test_count_all_reproduces_v6m(both):
    rec_n, rec_m = both
    n6.check_v6m(rec_n, rec_m)


def test_check_v6m_stops_on_each_kind_of_difference(both):
    rec_n, rec_m = both
    for path in (("count_1", "lwp_lwp", "halves", "cross_50", "R_loc", "share_vs_own"),
                 ("rule_persons", "kupa_essay", "oddeven", "inside_25", "I", "paired_minus_R_loc", "difference"),
                 ("part_4", "lwp_lwp", "cross_100", "I", "1", "median"),
                 ("tallies", "kupa_essay", "doses", "cross_25", "swaps", "median"),
                 ("tallies", "lwp_lwp", "count_persons")):
        bad = copy.deepcopy(rec_m)
        x = bad
        for p in path[:-1]:
            x = x[p]
        if x[path[-1]] is None:
            continue
        x[path[-1]] = x[path[-1]] + 1
        with pytest.raises(SystemExit):
            n6.check_v6m(rec_n, bad)
    bad = copy.deepcopy(rec_m)
    first = next(iter(bad["part_2"]["I"]))
    bad["part_2"]["I"][first]["stage"]["rising"] = "x"
    with pytest.raises(SystemExit, match="part 2"):
        n6.check_v6m(rec_n, bad)
    bad = copy.deepcopy(rec_m)
    bad["tallies"]["kupa_copy"]["count_persons"] += 1
    with pytest.raises(SystemExit, match="kupa_copy"):
        n6.check_v6m(rec_n, bad)
    bad = copy.deepcopy(rec_m)
    bad["tallies"]["kupa_essay"]["doses"]["cross_25"]["persons_same_as_next_step"] += 1
    with pytest.raises(SystemExit, match="same_as_next_step"):
        n6.check_v6m(rec_n, bad)
    ok = copy.deepcopy(rec_m)  # same_as_next_step at 50 is not compared
    ok["tallies"]["kupa_essay"]["doses"]["cross_50"]["persons_same_as_next_step"] += 1
    n6.check_v6m(rec_n, ok)


def test_count_all_structure_and_no_per_person(both):
    rec_n, _ = both
    rec = dict(rec_n)
    p4 = rec.pop("part_4_all")
    rec["part_2_signed_dose_rows"] = {nm: {row: p4[nm][row] for row in n6.SIGNED_ROWS} for nm in p4}
    rec["predictions"] = n6.predictions(rec)
    out = {"main_cells": n6.main_cells_out(rec), **rec}
    k6.v6d.no_per_person(out)
    k6.v6i.no_long_list(out)
    json.dumps(out, allow_nan=False)
    assert set(rec["predictions"]) == {f"P{i}" for i in range(1, 8)}
    assert set(rec["part_3"]["reliability"]) == {"kupa_essay", "kupa_copy", "lwp_lwp"}
    assert len(rec["part_1"]["gate_0"]) == 8
    assert set(rec["part_1"]["rule_persons"]) == {*n6.RULE_STAGES, *n6.BESIDE}
    e = rec["part_1"]["rule_persons"]["U@2000"]["lwp_lwp | halves | inside"]
    assert e["fixed_by_construction"] and e["stage"]["first_step_above"] == "not read"
    assert rec["tallies"]["lwp_lwp"]["doses"]["cross_75"]["persons"] == 12
    assert "positions_changed" in rec["tallies"]["lwp_lwp"]["doses"]["inside_75"]
    k = "lwp_lwp | cross_75 | B@2000"
    assert rec["part_2"][k]["signs"]["0"]["fixed_by_construction"] and rec["part_2"][k]["signs"]["0"]["share_zero"] == 1
    assert set(rec["part_2"][k]["count_1"]["halves"]) == {"gate_0", "rule_persons"}
    assert set(rec["part_2"]["lwp_lwp | cross_75 | R_mean"]["count_1"]["halves"]) == {"gate_0"}
    assert set(rec["part_2"]["lwp_lwp | cross_75 | B_complete@2000"]["count_1"]["halves"]) == {"gate_0", "rule_persons"}


def test_count_all_r_mean_and_r_loc_same_persons(both):
    rec_n, _ = both
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for row in n6.DOSE_ROWS:
                a, b = rec_n["count_1"][nm][how][row]["R_mean"], rec_n["count_1"][nm][how][row]["R_loc"]
                assert a["persons"] == b["persons"]
                p = rec_n["gate_0_refs"][nm][how][row]
                if p is not None:
                    assert p["difference"] == pytest.approx(a["share_vs_own"] - b["share_vs_own"])


def test_count_all_inside_75_fixed_cells(both):
    rec_n, _ = both
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for k in n6.INSIDE_FIXED:
                c = rec_n["count_1"][nm][how]["inside_75"][k]
                if "share_vs_own" in c:
                    assert c["median_change"] < 1e-9


# ---------------------------------------------------------------- predictions


def pred_rec(r_mean_cross="above", lwp_first="75", kupa_in100="above", rule75="below", loc75="undecided",
             rsb=(0.5, 0.6), diff=0.1):
    c = {"above": cell(0.7, 0.8), "below": cell(0.1, 0.2), "undecided": cell(0.5, 0.7), "none": {"persons": 1}}
    c1, rp, g0 = {}, {}, {}
    for ds, task in n6.MAIN:
        nm = f"{ds}_{task}"
        c1[nm], rp[nm] = {}, {}
        for how in v6b.BASELINE:
            c1[nm][how] = {"cross_100": {"R_mean": c[r_mean_cross]},
                           "inside_100": {"R_mean": c[kupa_in100]},
                           "inside_75": {"R_loc": c[loc75]}}
            rp[nm][how] = {row: {k: {"stage": c[rule75]} for k in n6.RULE_STAGES} for row in n6.NEW_ROWS}
            for fam in n6.FAMILIES:
                g0[f"{nm} | {how} | {fam}"] = {"R_mean": {"classes": {"25": "below"}, "first_step_above": lwp_first}}
    feats = {f"{st}[{j}]": {"new_minus_v6l": diff} for st in l6.REL_STAGES for j in range(l6.REL_DIMS[st])}
    rel = {nm: {"features": {"R_mean[0]": {"mean_rSB": rsb[0]}, "R[0]": {"mean_rSB": rsb[1]}}}
           for nm in ("kupa_essay", "lwp_lwp")}
    return {"count_1": c1, "rule_persons": rp, "part_1": {"gate_0": g0},
            "part_3": {"reliability": rel, "beside_v6l": {nm: feats for nm in ("kupa_essay", "kupa_copy", "lwp_lwp")}}}


def test_predictions_true_false_and_not_evaluable():
    assert n6.predictions(pred_rec()) == {f"P{i}": True for i in range(1, 8)}
    p = n6.predictions(pred_rec(r_mean_cross="undecided", lwp_first="100", kupa_in100="undecided", rule75="above",
                                loc75="above", rsb=(0.7, 0.6), diff=0.2))
    assert p == {f"P{i}": False for i in range(1, 8)}
    p = n6.predictions(pred_rec(r_mean_cross="none", kupa_in100="none", rule75="none", loc75="none", rsb=(None, 0.6),
                                diff=None))
    assert all(p[f"P{i}"] == n6.NE for i in (1, 3, 4, 5, 6, 7))
    p = n6.predictions(pred_rec(lwp_first="none"))
    assert p["P2"] is False
    r = pred_rec()
    for ds, task in n6.MAIN:
        for how in v6b.BASELINE:
            r["count_1"][f"{ds}_{task}"][how]["cross_100"]["R_mean"] = cell(0.1, 0.2)
            r["count_1"][f"{ds}_{task}"][how]["inside_100"]["R_mean"] = cell(0.1, 0.2)
    r["count_1"]["lwp_lwp"]["halves"]["cross_100"]["R_mean"] = cell(0.7, 0.8)
    r["count_1"]["kupa_essay"]["halves"]["inside_100"]["R_mean"] = cell(0.7, 0.8)
    p = n6.predictions(r)
    assert p["P1"] is False and p["P3"] is False  # one cell above is not 2 of 4 / not both baselines
    r = pred_rec()
    r["part_3"]["reliability"]["lwp_lwp"]["features"]["R_mean[0]"]["mean_rSB"] = 0.9
    assert n6.predictions(r)["P6"] is False
    r = pred_rec(diff=-0.2)
    assert n6.predictions(r)["P7"] is False
    # P4 reads every rule stage's cell at 75, the cells fixed by construction included (as the conditions word it)
    r = pred_rec()
    r["rule_persons"]["lwp_lwp"]["halves"]["inside_75"]["U@2000"]["stage"] = cell(0.7, 0.8)
    assert n6.predictions(r)["P4"] is False
    r = pred_rec()
    r["rule_persons"]["kupa_essay"]["oddeven"]["cross_75"]["I"]["stage"] = cell(0.7, 0.8)
    assert n6.predictions(r)["P4"] is False


def test_fast_n_stops_when_r_mean_and_r_differ_in_a_split(monkeypatch, texts):
    rec = texts[0][0]
    f = n6.FastN(rec)
    real = n6.r_mean_of
    monkeypatch.setattr(n6, "r_mean_of", lambda g, lab, held: None)
    with pytest.raises(SystemExit, match="in a split"):
        f.vectors(np.arange(rec.n // 2))
    monkeypatch.setattr(n6, "r_mean_of", lambda g, lab, held: real(g, lab, held) + np.array([0, 0, 0, 0, 0, 0.1]))
    with pytest.raises(SystemExit, match="in a split"):
        f.vectors(np.arange(rec.n // 2))


def test_reliability_few_splits_not_summarised(texts):
    rel = n6.reliability(texts, "lwp", "lwp", n_splits=3)
    c = rel["features"]["R_mean[0]"]
    assert c["splits_with_a_value"] == 3 and "mean_rSB" not in c and c["summary_not_written"]
    rel = n6.reliability(texts, "lwp", "lwp", n_splits=5)
    assert "mean_rSB" in rel["features"]["R_mean[0]"]


def test_part1_and_part2_are_assembled_from_the_cells(both):
    rec_n, _ = both
    c1, rp = rec_n["count_1"], rec_n["rule_persons"]
    for where, e in rec_n["part_1"]["gate_0"].items():
        nm, how, fam = where.split(" | ")
        for r in n6.REFS:
            for st, _ in n6.STEPS:
                c = c1[nm][how][f"{fam}_{st}"][r]
                assert e[r]["classes"][st] == n6.size_class(c) and e[r]["shares"][st] == c.get("share_vs_own")
                assert e["medians"][r][st]["median_base"] == c.get("median_base")
            assert e["paired_R_mean_minus_R_loc"] == {st: rec_n["gate_0_refs"][nm][how][f"{fam}_{st}"]
                                                     for st, _ in n6.STEPS}
        for k in (*n6.RULE_STAGES, *n6.BESIDE):
            ent = rec_n["part_1"]["rule_persons"][k][where]
            for st, _ in n6.STEPS:
                cell_ = rp[nm][how][f"{fam}_{st}"][k]
                for part in ("stage", *n6.REFS):
                    assert ent[part]["classes"][st] == n6.size_class(cell_[part])
                    assert ent[part]["shares"][st] == cell_[part].get("share_vs_own")
                assert ent["paired_minus_R_mean"][st] == cell_.get("paired_minus_R_mean")
    for key, e in rec_n["part_2"].items():
        nm, row, st = key.split(" | ")
        for how in v6b.BASELINE:
            c = c1[nm][how][row][st]
            assert e["count_1"][how]["gate_0"] == [c.get("share_vs_own"), n6.size_class(c)]
            if "rule_persons" in e["count_1"][how]:
                c = rp[nm][how][row][st]["stage"]
                assert e["count_1"][how]["rule_persons"] == [c.get("share_vs_own"), n6.size_class(c)]
        for f, v in e["signs"].items():
            src = rec_n["part_4_all"][nm][row][st][f]
            assert v["share_positive"] == src["share_positive"] and v["persons"] == src["persons"]


def test_count_all_with_rm_runs_the_check_first(both, monkeypatch):
    rec_n, rec_m = both
    ks_, ls_ = fake_kupa_lwp()
    kid, lid = sorted(ks_), sorted(ls_)
    kp, lp = {p: i for i, p in enumerate(kid)}, {p: i for i, p in enumerate(lid)}
    bad = copy.deepcopy(rec_m)
    bad["tallies"]["lwp_lwp"]["count_persons"] += 1
    called = []
    monkeypatch.setattr(n6, "reliability", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit, match="count_persons"):
        n6.count_all(ks_, kid, kp, ls_, lid, lp, M, RL, n_splits=5, rm=bad)
    assert not called  # stopped before the splits


def test_check_v6m_key_presence(both):
    rec_n, rec_m = both
    bad = copy.deepcopy(rec_m)
    del bad["rule_persons"]["lwp_lwp"]["halves"]["cross_25"]["I"]["paired_minus_R_loc"]
    with pytest.raises(SystemExit, match="one side only"):
        n6.check_v6m(rec_n, bad)


def test_positions_changed_uses_the_tolerance():
    g, starts, ws = [], [0], []
    p = 0
    for k in range(6):
        if k > 0:
            g.append(300.5)
            p += 1
            starts.append(p)
        for w in range(3):
            g.append(110.0)
            p += 1
            g.append(300.0)
            p += 1
            ws.append(p)
    g.append(100.0)
    s = v6.Series(m6.times_of(np.array(g)), starts, ws)
    _, tal = n6.dose_rows(s, [0, 1, 0])
    assert tal["inside_100"]["swaps"] == 5 and tal["inside_100"]["positions_changed"] == 10


def test_check_rows_stops_on_the_last_pair():
    s, _ = fake_person(3, 14)
    key = [0, 1, 3]
    t, tal = n6.dose_rows(s, key)
    g = np.diff(s.times)
    assert not np.array_equal(t["cross_50"], t["cross_100"])
    bad = dict(t)
    bad["cross_100"] = t["cross_50"]
    with pytest.raises(SystemExit, match="not nested"):
        n6.check_rows(g, bad, None)
    bad_tal = copy.deepcopy(tal)
    assert len(bad_tal["cross_100"]["made"]) > len(bad_tal["cross_75"]["made"]) >= 2
    bad_tal["cross_100"]["made"] = [bad_tal["cross_100"]["made"][1], bad_tal["cross_100"]["made"][0],
                                    *bad_tal["cross_100"]["made"][2:]]
    with pytest.raises(SystemExit, match="swaps of the doses"):
        n6.check_rows(g, t, None, bad_tal)


def test_fixed_parts_small_counts_not_summarised(texts):
    fp = n6.fixed_parts(texts[:4], ("I",))
    c = fp["halves"]["I[0]"]
    assert c["persons"] == 4 and "sd_A" not in c and "mean_abs_A_minus_B" not in c
    fp = n6.fixed_parts(texts[:5], ("I",))
    assert "sd_A" in fp["halves"]["I[0]"]


def test_paired_keys_passed(hv, monkeypatch):
    seen = []
    real = n6.paired

    def spy(hv_, row, how, a, b, need, ca, cb, rng_key, where):
        seen.append((a, b, list(rng_key)))
        return real(hv_, row, how, a, b, need, ca, cb, rng_key, where)

    monkeypatch.setattr(n6, "paired", spy)
    n6.rule_cell(hv, "kupa", "essay", "cross_75", "halves", "I")
    kk = n6.key1("kupa", "essay", "I", "cross_75", "halves", n6.gate_of("I"))
    assert [x[:2] for x in seen] == [("I", "R_loc"), ("I", "R_mean"), ("R_mean", "R_loc")]
    assert all(x[2] == kk for x in seen)
    seen.clear()
    c1 = {how: {row: {k: n6.one_cell(hv, row, how, k, n6.key1("lwp", "lwp", k, row, how, 0)) for k in n6.REFS}
                for row in hv[how] if row != "text/before"} for how in v6b.BASELINE}
    monkeypatch.setattr(n6, "ROWS", ["text/before", "inside_75", "cross_75"])
    n6.gate0_refs(hv, c1, "lwp", "lwp")
    assert {tuple(x[2]) for x in seen} == {tuple(n6.key1("lwp", "lwp", "R_mean", row, how, 0))
                                           for how in v6b.BASELINE for row in ("inside_75", "cross_75")}
