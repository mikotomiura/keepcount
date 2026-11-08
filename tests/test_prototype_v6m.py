import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6m as m6  # noqa: E402
from test_prototype_v6k import fake_person  # noqa: E402

k6, l6, ks = m6.k6, m6.l6, m6.ks
v6, v6b = k6.v6, k6.v6b
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
KEY = [0, 1, 7]


def text_gaps(s):
    return np.diff(s.times)


# ---------------------------------------------------------------- the conditions' constants


def test_constants_follow_the_conditions():
    c = json.loads((ROOT / "works/conditions/prototype_v6m.json").read_text())
    assert m6.ROWS[:7] == k6.ROWS and m6.ROWS[7:] == ["inside_25", "inside_50", "inside_100", "cross_25", "cross_50",
                                                      "cross_100"]
    assert [m6.ROW_IX[r] for r in m6.ROWS] == list(range(13)) and m6.CONTRAST_ROW_IX == 13
    assert m6.STAGE == {**l6.STAGE, "U_med": 11} and l6.STAGE["B_complete"] == 10
    assert m6.SKIP == {"kupa": 656, "lwp": 411}
    assert "656" in c["participants"]["kupa"] and "411" in c["participants"]["lwp"]
    assert m6.key1("lwp", "lwp", "U_med", "cross_50", "halves", 0) == [20261031, 95, 2, 0, 11, 11, 1, 0, 0]
    assert m6.key1("kupa", "essay", "R_loc", "inside_25", "oddeven", m6.gate_of("I")) == [20261031, 95, 0, 1, 8, 7, 0,
                                                                                          0, 6]
    assert m6.key_c("B@2000", "halves", "essay", 4) == [20261031, 96, 3, 1, 1, 4]
    assert [p for _, p in m6.STEPS] == [0.25, 0.5, 1.0]


# ---------------------------------------------------------------- part vectors


@pytest.mark.parametrize("seed", range(4))
def test_part_vectors_m(seed):
    s, kinds = fake_person(seed, 12)
    rec = k6.Rec(s.times, s.ranges("before"), v6b.labels(s, kinds), M)
    idx = np.arange(0, rec.n, 2)
    out, _ = m6.part_vectors_m(rec, idx)
    ref, _ = k6.part_vectors(rec, idx, {"2000": 2000.0})
    for k, v in ref.items():
        assert (v is None and out[k] is None) or np.array_equal(v, out[k])
    assert np.array_equal(out["U_med"], ref["U@2000"][[0, 2]])
    assert np.array_equal(out["R_loc"], ref["R"][:6]) and np.array_equal(out["R_lag"], ref["R"][6:])
    assert set(out) >= set(m6.KEYS)


# ---------------------------------------------------------------- the dose plans


@pytest.mark.parametrize("seed", range(10))
def test_inside_100_equals_rests_inside_at(seed):
    s, _ = fake_person(seed, 10)
    rows, tally = m6.dose_rows(s, KEY)
    ref = ks.rests_inside_at(s.times, s.starts, s.word_starts, np.random.default_rng([m6.SEED, 6, 5, *KEY]))[0]
    assert np.array_equal(rows["inside_100"], ref)
    assert np.array_equal(rows["inside_100"], k6.row_times(s, "inside/before", KEY))
    assert tally["inside_100"]["swaps"] == tally["inside_100"]["m"] == len(s.starts) - 1


def changed(s, t):
    return set(np.flatnonzero(np.abs(np.diff(t) - text_gaps(s)) > 1e-6).tolist())


@pytest.mark.parametrize("seed", range(10))
def test_doses_nested_and_sized(seed):
    s, _ = fake_person(seed, 13)
    rows, tally = m6.dose_rows(s, KEY)
    for fam in m6.FAMILIES:
        a, b, c = (changed(s, rows[f"{fam}_{st}"]) for st in ("25", "50", "100"))
        assert a <= b <= c
        m = tally[f"{fam}_100"]["m"]
        for st, p in m6.STEPS:
            t = tally[f"{fam}_{st}"]
            assert t["swaps"] + t["left_without_partner"] == math.ceil(p * m)
    m6.check_rows(text_gaps(s), rows, k6.row_times(s, "inside/before", KEY))


def test_inside_plan_prefix_by_hand():
    s, _ = fake_person(3, 9)
    plan = m6.inside_plan(s, KEY)
    m = len(plan)
    order = np.random.default_rng([m6.SEED, 94, 1, *KEY]).permutation(m)
    # candidates in sentence order: the gap before each sentence's first press (but the first)
    starts = [a - 1 for a, _ in ks._sentence_bounds(s.starts, s.times.size)[1:]]
    assert [a for a, _ in plan] == [starts[int(i)] for i in order]
    rows, _ = m6.dose_rows(s, KEY)
    g = text_gaps(s)
    k = math.ceil(0.25 * m)
    assert np.array_equal(rows["inside_25"], m6.times_of(m6.apply_swaps(g, plan[:k])))


@pytest.mark.parametrize("seed", range(10))
def test_cross_plan_rules(seed):
    s, _ = fake_person(seed, 12)
    g = text_gaps(s)
    long = g >= 2000.0 - 1e-6
    sent = m6.sentence_of_gap(s.starts, s.times.size)
    bounds = ks._sentence_bounds(s.starts, s.times.size)
    givers = {a - 1 for a, _ in bounds[1:] if long[a - 1]}
    ws = set(s.word_starts)
    plan = m6.cross_plan(s, KEY)
    assert {a for a, _ in plan} == givers and len(plan) == len(givers)
    partners = [b for _, b in plan if b is not None]
    assert len(partners) == len(set(partners))
    for a, b in plan:
        if b is None:
            continue
        assert not long[b] and (b + 1) in ws and sent[b] != sent[a]
        lo, hi = bounds[sent[b]]
        assert lo < b + 1 < hi  # strictly after its sentence's first press


def test_cross_plan_by_hand():
    # three sentences; sentence starts at presses 0, 5, 10; word starts at 2, 7, 12; long gaps before 5 and 10
    g = np.array([100, 300, 100, 100, 2500, 100, 300, 100, 100, 3000, 100, 300, 100], dtype=float)
    s = v6.Series(m6.times_of(g), [0, 5, 10], [2, 7, 12])
    plan = m6.cross_plan(s, KEY)
    assert sorted(a for a, _ in plan) == [4, 9]
    order = np.random.default_rng([m6.SEED, 94, 2, *KEY]).permutation(2)
    givers = [4, 9]
    rng = np.random.default_rng([m6.SEED, 94, 3, *KEY])
    pool = [1, 6, 11]
    sent = {1: 0, 6: 1, 11: 2, 4: 1, 9: 2}
    taken, want = set(), []
    for i in order:
        gi = givers[int(i)]
        ok = [j for j in pool if j not in taken and sent[j] != sent[gi]]
        pick = ok[int(rng.integers(len(ok)))]
        taken.add(pick)
        want.append((gi, pick))
    assert plan == want


def test_cross_plan_no_partner_draws_nothing():
    # two sentences: the only pool gap lies in the giver's own sentence
    g = np.array([100, 100, 100, 2500, 100, 300, 100], dtype=float)
    s = v6.Series(m6.times_of(g), [0, 4], [6])
    assert m6.cross_plan(s, KEY) == [(3, None)]
    rows, tally = m6.dose_rows(s, KEY)
    assert np.array_equal(rows["cross_100"], s.times) and tally["cross_100"]["left_without_partner"] == 1


def test_cross_without_givers_equals_text():
    g = np.array([100, 300, 100, 100, 500, 100, 300, 100], dtype=float)
    s = v6.Series(m6.times_of(g), [0, 5], [2, 7])
    rows, tally = m6.dose_rows(s, KEY)
    for st, _ in m6.STEPS:
        assert np.array_equal(rows[f"cross_{st}"], s.times) and tally[f"cross_{st}"]["m_zero"]
        assert tally[f"cross_{st}"]["share_swapped"] is None


def test_dose_tallies_by_hand():
    g = np.array([100, 300, 100, 100, 2500, 100, 300, 100, 100, 3000, 100, 300, 100], dtype=float)
    s = v6.Series(m6.times_of(g), [0, 5, 10], [2, 7, 12])
    rows, tally = m6.dose_rows(s, KEY)
    t = tally["cross_100"]
    assert t["m"] == 2 and t["swaps"] == 2 and t["time_moved_share"] == pytest.approx(5500 / g.sum())
    assert tally["cross_25"]["swaps"] == 1 and tally["cross_50"]["swaps"] == 1
    assert tally["cross_25"]["same_as_next_step"] and not tally["cross_50"]["same_as_next_step"]
    # odd/even: sentence 0 and 2 in A, 1 in B; halves: sentence 0 in A, 1 and 2 in B
    plan = m6.cross_plan(s, KEY)
    sent = m6.sentence_of_gap(s.starts, s.times.size)
    for how, part in (("oddeven", {0: 0, 1: 1, 2: 0}), ("halves", {0: 0, 1: 1, 2: 1})):
        want = sum(part[int(sent[a])] != part[int(sent[b])] for a, b in plan)
        assert t["crossing_parts"][how] == want
    it = tally["inside_100"]
    swaps = m6.inside_plan(s, KEY)
    assert it["time_moved_share"] == pytest.approx(sum(abs(g[a] - g[b]) for a, b in swaps) / g.sum())


def test_sentence_of_gap():
    assert m6.sentence_of_gap([0, 3, 5], 8).tolist() == [0, 0, 1, 1, 2, 2, 2]


def test_check_rows_stops():
    s, _ = fake_person(1, 10)
    rows, _ = m6.dose_rows(s, KEY)
    ins = k6.row_times(s, "inside/before", KEY)
    bad = dict(rows)
    bad["inside_25"] = rows["inside_100"]
    bad["inside_100"] = rows["inside_25"]
    with pytest.raises(SystemExit, match="nested"):
        m6.check_rows(text_gaps(s), bad, ins)
    bad = dict(rows)
    g = np.diff(rows["cross_50"]).copy()
    g[0] += 5.0
    bad["cross_50"] = m6.times_of(g)
    with pytest.raises(SystemExit, match="permutation"):
        m6.check_rows(text_gaps(s), bad, ins)
    with pytest.raises(SystemExit, match="inside_100"):
        m6.check_rows(text_gaps(s), rows, rows["inside_50"] if changed(s, rows["inside_50"]) != changed(
            s, rows["inside_100"]) else s.times)


# ---------------------------------------------------------------- persons, rows and construction


@pytest.mark.parametrize("seed", range(4))
def test_person_rows_and_construction(seed):
    s, kinds = fake_person(seed, 12)
    vec, whole, tallies, dose = m6.person_rows(s, kinds, KEY, M, m6.ROWS)
    m6.check_construction(vec, tallies)
    assert set(vec) == set(m6.ROWS) and set(dose) == set(m6.DOSE_ROWS)
    assert set(whole) == set(m6.ROWS) - {"text/after", "text/none"}
    for how in v6b.BASELINE:
        for st, _ in m6.STEPS:
            for h in (0, 1):
                for k in m6.INSIDE_FIXED:
                    assert np.array_equal(vec[f"inside_{st}"][how][h][k], vec["text/before"][how][h][k]) or \
                        np.max(np.abs(vec[f"inside_{st}"][how][h][k] - vec["text/before"][how][h][k])) < 1e-9
    rec = k6.Rec(s.times, s.ranges("before"), v6b.labels(s, kinds), M)
    assert np.array_equal(whole["text/before"]["U@2000"], m6.part_vectors_m(rec, np.arange(rec.n))[0]["U@2000"])


def test_check_construction_stops_on_a_change():
    s, kinds = fake_person(2, 12)
    vec, _, tallies, _ = m6.person_rows(s, kinds, KEY, M, m6.ROWS)
    a, b = vec["inside_50"]["halves"]
    a = dict(a)
    a["U_med"] = a["U_med"] + 0.01
    vec["inside_50"]["halves"] = (a, b)
    with pytest.raises(SystemExit, match="U_med"):
        m6.check_construction(vec, tallies)


def test_check_construction_cross_not_checked():
    s, kinds = fake_person(2, 12)
    vec, _, tallies, _ = m6.person_rows(s, kinds, KEY, M, m6.ROWS)
    a, b = vec["cross_50"]["halves"]
    a = dict(a)
    a["U_med"] = a["U_med"] + 0.01
    vec["cross_50"]["halves"] = (a, b)
    m6.check_construction(vec, tallies)


# ---------------------------------------------------------------- count 1 pieces


def fake_hv(n=30, rows=("text/before", "cross_100"), seed=0):
    hv = {how: {row: [] for row in rows} for how in v6b.BASELINE}
    for i in range(n):
        s, kinds = fake_person(100 + i + seed, 10)
        vec, _, _, _ = m6.person_rows(s, kinds, [0, 1, i], M, list(dict.fromkeys(["text/before", *rows])))
        for how in v6b.BASELINE:
            for row in rows:
                hv[how][row].append(vec[row][how])
    return hv


@pytest.fixture(scope="module")
def hv():
    return fake_hv()


@pytest.mark.parametrize("st", ["U@2000", "U_med", "R_loc", "I", "B@2000"])
def test_own_contrib_gives_v6b_one(hv, st):
    first, sel = hv["halves"]["text/before"], hv["halves"]["cross_100"]
    ok = [i for i in range(len(first)) if v6b.has(first[i], st) and v6b.has(sel[i], st)]
    f, s = [first[i] for i in ok], [sel[i] for i in ok]
    res = v6b.one(f, s, st, [1, 2, 3])
    assert float(m6.own_contrib(f, s, st).mean()) == res["share_vs_own"]


def test_rule_cell_paired(hv):
    c = m6.rule_cell(hv, "kupa", "essay", "cross_100", "oddeven", "U@2000")
    assert c["stage"]["persons"] == c["R_loc"]["persons"] == c["R"]["persons"] == c["R_lag"]["persons"]
    d = c["paired_minus_R_loc"]
    assert d["difference"] == pytest.approx(c["stage"]["share_vs_own"] - c["R_loc"]["share_vs_own"])
    assert d["ci"][0] <= d["difference"] <= d["ci"][1]
    gate = m6.gate_of("U@2000")
    assert c["stage"] == l6.one_cell(hv, "cross_100", "oddeven", "U@2000",
                                     m6.key1("kupa", "essay", "U@2000", "cross_100", "oddeven", gate), extra="R")


# ---------------------------------------------------------------- part 1


def test_contrast_one_by_hand():
    # 3 persons, 1 feature; copy parts and essay parts
    cA = np.array([[0.0], [1.0], [2.0]])
    cB = np.array([[0.0], [1.5], [2.0]])
    eA = np.array([[1.0], [1.0], [2.0]])
    eB = np.array([[1.0], [1.5], [2.0]])
    idx = np.zeros((10, 3), dtype=int)
    sd = np.vstack([cA, cB, eA, eB]).std(axis=0)[0]
    res, own = m6.contrast_one(cA, cB, eA, eB, "copy", idx)
    # person 0: change 1/sd, base 0 -> 1; person 1: change 0, base 0.5/sd -> 0; person 2: change 0 = base 0 -> 1/2
    assert own.tolist() == [1.0, 0.0, 0.5] and res["base_zero"] == 2
    assert "median_change" not in res and res["medians_not_written"] == "fewer than 5 persons"
    res_e, own_e = m6.contrast_one(cA, cB, eA, eB, "essay", idx)
    assert own_e.tolist() == [1.0, 0.0, 0.5]
    del sd


def test_contrast_one_bases_differ():
    cA = np.array([[0.0], [0.0]])
    cB = np.array([[3.0], [0.0]])
    eA = np.array([[1.0], [1.0]])
    eB = np.array([[1.0], [2.0]])
    idx = np.zeros((4, 2), dtype=int)
    _, oc = m6.contrast_one(cA, cB, eA, eB, "copy", idx)
    _, oe = m6.contrast_one(cA, cB, eA, eB, "essay", idx)
    # person 0: change (1 + 2)/2 = 1.5 (unscaled), base copy 3, base essay 0
    assert oc[0] == 0.0 and oe[0] == 1.0


def test_contrast_scales_and_pairs():
    rng = np.random.default_rng(4)
    cA, cB = rng.normal(0, 1, (20, 3)), rng.normal(0, 1, (20, 3))
    eA, eB = rng.normal(2, 1, (20, 3)), rng.normal(2, 1, (20, 3))
    cA[:, 2] = cB[:, 2] = eA[:, 2] = eB[:, 2] = 5.0  # constant feature: dropped
    idx = np.random.default_rng(1).integers(0, 20, (50, 20))
    res, own = m6.contrast_one(cA, cB, eA, eB, "copy", idx)
    assert res["dropped_features"] == [2]
    sd = np.vstack([cA, cB, eA, eB])[:, :2].std(axis=0)
    ch = (np.linalg.norm((eA - cA)[:, :2] / sd, axis=1) + np.linalg.norm((eB - cB)[:, :2] / sd, axis=1)) / 2
    b = np.linalg.norm((cA - cB)[:, :2] / sd, axis=1)
    assert own.tolist() == np.where(ch > b, 1.0, np.where(ch == b, 0.5, 0.0)).tolist()
    _, ow = m6.contrast_one(cA, cB, eA, eB, "copy", idx, scale="within")
    sdw = np.sqrt((np.vstack([cA, cB])[:, :2].var(axis=0) + np.vstack([eA, eB])[:, :2].var(axis=0)) / 2)
    chw = (np.linalg.norm((eA - cA)[:, :2] / sdw, axis=1) + np.linalg.norm((eB - cB)[:, :2] / sdw, axis=1)) / 2
    bw = np.linalg.norm((cA - cB)[:, :2] / sdw, axis=1)
    assert ow.tolist() == np.where(chw > bw, 1.0, np.where(chw == bw, 0.5, 0.0)).tolist()
    _, oa = m6.contrast_one(cA, cB, eA, eB, "essay", idx, all_pairs=True)
    z = lambda x: x[:, :2] / sd  # noqa: E731
    cha = sum(np.linalg.norm(z(e) - z(c), axis=1) for e in (eA, eB) for c in (cA, cB)) / 4
    ba = np.linalg.norm(z(eA) - z(eB), axis=1)
    assert oa.tolist() == np.where(cha > ba, 1.0, np.where(cha == ba, 0.5, 0.0)).tolist()


def test_contrast_one_stops():
    x = np.ones((3, 2))
    with pytest.raises(SystemExit, match="constant"):
        m6.contrast_one(x, x, x, x, "copy", np.zeros((2, 3), dtype=int))
    y = x.copy()
    y[0, 0] = np.nan
    with pytest.raises(SystemExit, match="non-finite"):
        m6.contrast_one(y, x, x, x, "copy", np.zeros((2, 3), dtype=int))


@pytest.fixture(scope="module")
def pairs():
    out = []
    for i in range(25):
        sc, kc = fake_person(300 + i, 10)
        se, ke = fake_person(600 + i, 11)
        vc, _, _, _ = m6.person_rows(sc, kc, [0, 0, i], M, ["text/before"])
        ve, _, _, _ = m6.person_rows(se, ke, [0, 1, i], M, ["text/before"])
        out.append((vc["text/before"], ve["text/before"]))
    return out


def test_contrast_structure(pairs):
    c = m6.contrast(pairs)
    assert set(c["labelled"]) == set(m6.CONTRAST_LABELLED) and set(c["beside"]) == set(m6.CONTRAST_BESIDE)
    for k, cells in {**c["labelled"], **c["beside"]}.items():
        assert set(cells) == {f"{h} | {b}" for h in v6b.BASELINE for b in m6.BASES}
        for cell in cells.values():
            assert "share_vs_own" in cell["own_persons"] or cell["own_persons"].get("too_few_persons")
            if k in m6.CONTRAST_LABELLED:
                assert "share_vs_own" in cell["own_persons"]
            if k == "R_loc":
                assert set(cell) == {"own_persons"}
                continue
            assert cell["stage"]["persons"] == cell["R_loc"]["persons"] == cell["R"]["persons"]
            if cell["stage"].get("too_few_persons"):
                assert k not in m6.CONTRAST_LABELLED and "paired_minus_R_loc" not in cell
                continue
            assert "within_scale" in cell["stage"] and "all_pairs" in cell["stage"]
            d = cell["paired_minus_R_loc"]["difference"]
            assert d == pytest.approx(cell["stage"]["share_vs_own"] - cell["R_loc"]["share_vs_own"])
    r = m6.contrast_rule(c)
    for k in m6.CONTRAST_LABELLED:
        assert r[k]["cells_read"] + sum(v in ("control not met", "fewer than 2 rule persons")
                                        for v in r[k]["status"].values()) == 4


def test_contrast_cell_persons_and_keys(pairs):
    res, own, idx = m6.contrast_cell(pairs, "I", "oddeven", "copy", 0)
    ok = [p for p in pairs if m6.has4(p, "oddeven", ("I",))]
    assert res["persons"] == len(ok) and res["left_out_no_vector"] == len(pairs) - len(ok)
    want = np.random.default_rng([m6.SEED, 96, 5, 0, 0, 0]).integers(0, len(ok), (v6.N_BOOT, len(ok)))
    assert np.array_equal(idx, want) and res["share_vs_own"] == float(own.mean())


def cell(lo, hi):
    return {"share_vs_own": (lo + hi) / 2, "share_vs_own_ci": [lo, hi]}


def test_contrast_rule_labels():
    above, below, und = cell(0.7, 0.9), cell(0.2, 0.4), cell(0.5, 0.7)

    def c_of(stages, gate):
        return {"labelled": {k: {f"{h} | {b}": {"stage": st, "R_loc": g, "R": g}
                                 for (h, b), st, g in zip([(h, b) for h in v6b.BASELINE for b in m6.BASES], stages,
                                                          gate)} for k in m6.CONTRAST_LABELLED}}
    r = m6.contrast_rule(c_of([above] * 4, [above] * 4))
    assert r["U@2000"]["label"] == "shows these operations beyond the smallest difference"
    r = m6.contrast_rule(c_of([above, above, below, und], [above] * 4))
    assert r["I"]["label"] == "mixed directions"
    r = m6.contrast_rule(c_of([below] * 4, [above, above, above, und]))
    assert r["B@2000"]["label"] == "not evaluable" and r["B@2000"]["cells_read"] == 3
    r = m6.contrast_rule(c_of([below] * 4, [above] * 4))
    assert r["Bsd@2000"]["label"] == "does not show them beyond it"


# ---------------------------------------------------------------- part 2


def test_doses_summary():
    above, below, und = cell(0.7, 0.9), cell(0.2, 0.4), cell(0.5, 0.7)
    rp = {}
    for ds, task in m6.MAIN:
        rp[f"{ds}_{task}"] = {how: {} for how in v6b.BASELINE}
        for how in v6b.BASELINE:
            for fam in m6.FAMILIES:
                for st, cl in zip(("25", "50", "100"), (below, und, above)):
                    rp[f"{ds}_{task}"][how][f"{fam}_{st}"] = {
                        k: {"stage": cl, "R": above, "R_loc": above if st != "25" else und, "R_lag": below}
                        for k in (*m6.RULE_STAGES, *m6.BESIDE)}
    out = m6.doses(rp)
    e = out["I"]["kupa_essay | oddeven | cross"]
    assert e["stage"]["first_step_above"] == "100" and e["stage"]["rising"]
    assert e["R_loc"]["first_step_above"] == "50" and e["R_lag"]["first_step_above"] == "none"
    f = out["U@2000"]["lwp_lwp | halves | inside"]
    assert f["fixed_by_construction"] and f["stage"]["first_step_above"] == "not read"
    assert not out["I"]["lwp_lwp | halves | inside"]["fixed_by_construction"]
    rp["lwp_lwp"]["halves"]["cross_50"]["B@2000"]["stage"] = cell(0.0, 0.1)
    assert m6.doses(rp)["B@2000"]["lwp_lwp | halves | cross"]["stage"]["rising"] is False
    rp["lwp_lwp"]["halves"]["cross_50"]["Bsd@2000"] = dict(rp["lwp_lwp"]["halves"]["cross_50"]["Bsd@2000"])
    rp["lwp_lwp"]["halves"]["cross_50"]["Bsd@2000"]["stage"] = cell(0.9, 0.99)
    assert m6.doses(rp)["Bsd@2000"]["lwp_lwp | halves | cross"]["stage"]["rising"] is False


# ---------------------------------------------------------------- part 4


def test_signed_shares_and_median():
    d = np.array([1.0, 2.0, -1.0, 0.0, 5e-10, 3.0])
    r = m6.signed(d, [1, 2])
    assert r["share_positive"] == pytest.approx(3 / 6) and r["share_negative"] == pytest.approx(1 / 6)
    assert r["share_zero"] == pytest.approx(2 / 6) and r["share_of_the_more_frequent_sign"] == pytest.approx(0.5)
    assert r["median"] == float(np.median(d))
    idx = np.random.default_rng([1, 2]).integers(0, 6, (v6.N_BOOT, 6))
    assert r["median_ci"] == m6.boot_median(d, idx)
    assert m6.signed(d[:4], [1]) is None
    with pytest.raises(SystemExit, match="non-finite"):
        m6.signed(np.array([1.0, np.inf, 0, 0, 0]), [1])


def test_signed_set_direction():
    a = {st: None for st in m6.SIGNED_STAGES}
    pairs = []
    for i in range(6):
        x = dict(a, **{"B@2000": np.array([1.0 + i])})
        y = dict(a, **{"B@2000": np.array([3.0 + i])})
        pairs.append((x, y))
    out = m6.signed_set(pairs, 0, 1, 13)
    assert out["B@2000"]["0"]["share_positive"] == 1.0 and out["B@2000"]["0"]["median"] == 2.0
    assert out["U@2000"] == {}


# ---------------------------------------------------------------- the record holds no per-person value


def test_no_per_person_lists(pairs, hv):
    c = m6.contrast(pairs)
    v6d, v6i = k6.v6d, k6.v6i
    v6d.no_per_person(c)
    v6i.no_long_list(c)
    rc = m6.rule_cell(hv, "kupa", "essay", "cross_100", "halves", "I")
    v6i.no_long_list(rc)


# ---------------------------------------------------------------- the whole count on fake persons


def fake_kupa_lwp(n=12):
    ks_, ls_ = {}, {}
    for i in range(n):
        ks_[f"k{i}"] = {"copy": fake_person(1000 + i, 10), "essay": fake_person(2000 + i, 11)}
        ls_[f"l{i}"] = fake_person(3000 + i, 12)
    return ks_, ls_


@pytest.fixture(scope="module")
def whole():
    ks_, ls_ = fake_kupa_lwp()
    rk = json.loads((ROOT / "results/prototype_v6k/record.json").read_text())
    rl = json.loads((ROOT / "results/prototype_v6l/record.json").read_text())
    kid, lid = sorted(ks_), sorted(ls_)
    return m6.count_new(ks_, kid, {p: i for i, p in enumerate(kid)}, ls_, lid, {p: i for i, p in enumerate(lid)}, M,
                        rk, rl)


def test_count_new_runs_and_writes_no_per_person_value(whole):
    rec = dict(whole)
    out = {"main_cells": m6.main_cells_out(rec), **rec}
    k6.v6d.no_per_person(out)
    k6.v6i.no_long_list(out)
    json.dumps(out, allow_nan=False)
    assert set(rec["predictions"]) == {f"P{i}" for i in range(1, 9)}
    assert rec["part_1"]["contrast_persons"] == 12
    assert set(rec["part_0"]["by_gate"]) == {"R", "R_loc"}
    assert set(rec["part_2"]) == {*m6.RULE_STAGES, *m6.BESIDE}
    assert "kupa_essay_minus_copy" in rec["part_4"] and set(rec["part_4"]["lwp_lwp"]) == set(m6.SIGNED_ROWS)
    assert rec["tallies"]["kupa_copy"]["doses"] == {row: {} for row in m6.DOSE_ROWS}
    assert rec["tallies"]["lwp_lwp"]["doses"]["cross_100"]["persons"] == 12


def test_count_new_inside_fixed_cells_are_zero(whole):
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for st, _ in m6.STEPS:
                for k in m6.INSIDE_FIXED:
                    c = whole["count_1"][nm][how][f"inside_{st}"][k]
                    if "share_vs_own" in c:
                        assert c["median_change"] < 1e-9


def test_count_new_inside_100_equals_inside(whole):
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in v6b.BASELINE:
            for k in ("I", "R_loc", "Bsd@2000"):
                a = whole["count_1"][nm][how]["inside_100"][k]
                b = whole["count_1"][nm][how]["inside/before"][k]
                assert a["median_change"] == b["median_change"] and a["persons"] == b["persons"]


# ---------------------------------------------------------------- after the reviews


def test_same_as_next_step_counts_applied_swaps():
    # four long sentence-initial gaps, one pool gap: only the first giver in the plan gets a partner
    g = np.array([100, 300, 100, 2500, 100, 2500, 100, 2500, 100, 2500, 100], dtype=float)
    s = v6.Series(m6.times_of(g), [0, 4, 6, 8, 10], [2])
    plan = m6.cross_plan(s, KEY)
    assert sum(b is not None for _, b in plan) == 1 and plan[0][1] == 1
    _, tally = m6.dose_rows(s, KEY)
    # ceil(0.25*4)=1, ceil(0.5*4)=2, 4: the swaps applied are the same at every step
    assert tally["cross_25"]["same_as_next_step"] and tally["cross_50"]["same_as_next_step"]
    assert tally["cross_50"]["left_without_partner"] == 1


def test_check_rows_swaps_nested():
    s, _ = fake_person(5, 12)
    rows, tally = m6.dose_rows(s, KEY)
    m6.check_rows(text_gaps(s), rows, k6.row_times(s, "inside/before", KEY), tally)
    bad = {r: dict(t) for r, t in tally.items()}
    bad["cross_50"]["made"] = list(reversed(bad["cross_50"]["made"]))
    if len(bad["cross_50"]["made"]) >= 2:
        with pytest.raises(SystemExit, match="swaps"):
            m6.check_rows(text_gaps(s), rows, k6.row_times(s, "inside/before", KEY), bad)


def test_permutation_checked_for_every_row(monkeypatch):
    s, kinds = fake_person(6, 12)
    real = k6.row_times

    def broken(s_, row, key):
        t = real(s_, row, key)
        if row == "shuffled/before":
            t = t.copy()
            t[1:] += 3.0
        return t
    monkeypatch.setattr(k6, "row_times", broken)
    with pytest.raises(SystemExit, match="shuffled/before: not a permutation"):
        m6.person_rows(s, kinds, KEY, M, m6.ROWS)


def test_inside_sentence_without_a_later_word_start():
    # sentence 2 (presses 5-6) has no word start after its first press: it is skipped without a draw
    g = np.array([100, 300, 100, 100, 2500, 100, 2600, 100, 300, 100], dtype=float)
    s = v6.Series(m6.times_of(g), [0, 5, 7], [2, 9])
    plan = m6.inside_plan(s, KEY)
    assert sorted(a for a, _ in plan) == [6]
    rows, _ = m6.dose_rows(s, KEY)
    ref = ks.rests_inside_at(s.times, s.starts, s.word_starts, np.random.default_rng([m6.SEED, 6, 5, *KEY]))
    assert ref[1] == 1 and np.array_equal(rows["inside_100"], ref[0])


def test_construction_nan_stops():
    s, kinds = fake_person(2, 12)
    vec, _, tallies, _ = m6.person_rows(s, kinds, KEY, M, m6.ROWS)
    a, b = vec["inside_25"]["oddeven"]
    a = dict(a)
    a["B@2000"] = np.array([np.nan])
    vec["inside_25"]["oddeven"] = (a, b)
    with pytest.raises(SystemExit, match="B@2000"):
        m6.check_construction(vec, tallies)


def test_guard_small_counts():
    r = m6.guard({"persons": 4, "share_vs_own": 0.5, "median_change": 1.0, "median_base": 2.0,
                  "median_change_over_base": 0.5, "base_zero": 0})
    assert "median_change" not in r and "median_change_over_base" not in r and r["share_vs_own"] == 0.5
    r = m6.guard({"persons": 9, "median_change": 1.0, "median_base": 0.0, "median_change_over_base": 3.0,
                  "base_zero": 5})
    assert r["median_change_over_base"] is None and r["median_change"] == 1.0
    r = m6.guard({"persons": 9, "median_change": 1.0, "median_change_over_base": 3.0, "base_zero": 4})
    assert r["median_change_over_base"] == 3.0
    x = np.array([[0.0], [1.0], [2.0]])
    res, _ = m6.contrast_one(x, x + 1, x + 2, x * 2, "copy", np.zeros((2, 3), dtype=int))
    assert "median_change" not in res


def test_signed_nan_with_few_persons_stops():
    with pytest.raises(SystemExit, match="non-finite"):
        m6.signed(np.array([1.0, np.nan]), [1])


def test_count1_keys_and_gate(hv):
    hv2 = {how: {row: hv[how][row] for row in ("text/before", "cross_100")} for how in v6b.BASELINE}
    c = m6.one_cell(hv2, "cross_100", "halves", "I", m6.key1("kupa", "essay", "I", "cross_100", "halves", 0))
    ref = l6.one_cell(hv2, "cross_100", "halves", "I", [20261031, 95, 0, 1, 5, 12, 1, 0, 0])
    assert c["share_vs_own_ci"] == ref["share_vs_own_ci"]


def test_rule_cell_gate_keys(hv):
    for k in ("U@2000", "I", "U_med"):
        c = m6.rule_cell(hv, "kupa", "essay", "cross_100", "halves", k)
        gate = m6.gate_of(k)
        for g in m6.GATES:
            ref = l6.one_cell(hv, "cross_100", "halves", g, m6.key1("kupa", "essay", g, "cross_100", "halves", gate),
                              extra=k)
            assert c[g]["share_vs_own_ci"] == ref["share_vs_own_ci"] and c[g]["persons"] == ref["persons"]
    assert m6.gate_of("U_med") == 12 and m6.gate_of("I") == 6


def test_contrast_gate_keys(pairs):
    cell = m6.contrast_stage(pairs, "I")["halves | essay"]
    gate = m6.gate_of("I")
    for g in m6.GATES:
        ok = [p for p in pairs if m6.has4(p, "halves", (g, "I"))]
        arr = m6.contrast_arrays(ok, "halves", g)
        idx = np.random.default_rng([m6.SEED, 96, m6.STAGE[g], 1, 1, gate]).integers(0, len(ok), (v6.N_BOOT, len(ok)))
        assert cell[g]["share_vs_own_ci"] == m6.contrast_one(*arr, "essay", idx)[0]["share_vs_own_ci"]
    own = m6.contrast_stage(pairs, "I")["oddeven | copy"]["own_persons"]
    ok = [p for p in pairs if m6.has4(p, "oddeven", ("I",))]
    idx = np.random.default_rng([m6.SEED, 96, 5, 0, 0, 0]).integers(0, len(ok), (v6.N_BOOT, len(ok)))
    assert own["share_vs_own_ci"] == m6.contrast_one(*m6.contrast_arrays(ok, "oddeven", "I"), "copy",
                                                     idx)[0]["share_vs_own_ci"]


def test_signed_key():
    a = {st: None for st in m6.SIGNED_STAGES}
    pairs = [(dict(a, I=np.array([1.0, 2.0, 3.0 + i])), dict(a, I=np.array([1.5, 2.0, 2.0 * i]))) for i in range(7)]
    out = m6.signed_set(pairs, 2, 0, 9)
    d = np.array([2.0 * i - 3.0 - i for i in range(7)])
    idx = np.random.default_rng([20261031, 97, 2, 0, 5, 9, 2]).integers(0, 7, (v6.N_BOOT, 7))
    assert out["I"]["2"]["median_ci"] == m6.boot_median(d, idx)
    assert out["I"]["1"]["share_zero"] == 1.0


def test_predictions_true_false_and_not_evaluable(whole):
    import copy
    base = copy.deepcopy(whole)
    shows, above, below = "shows these operations beyond the smallest difference", cell(0.7, 0.9), cell(0.1, 0.3)
    rec = copy.deepcopy(base)
    rec["part_4"]["kupa_essay_minus_copy"]["U@2000"]["0"] = {"median": 0.2, "median_ci": [0.1, 0.3]}
    assert m6.predictions(rec)["P8"] is True
    rec["part_4"]["kupa_essay_minus_copy"]["U@2000"]["0"] = {"median": 0.2, "median_ci": [-0.1, 0.3]}
    assert m6.predictions(rec)["P8"] is False
    rec["part_4"]["kupa_essay_minus_copy"]["U@2000"]["0"] = None
    assert m6.predictions(rec)["P8"] == m6.NE
    # P6: R_loc and I above in all 4 cross_100 cells, U above in exactly 2 -> True; in 1 -> False
    rec = copy.deepcopy(base)
    cells = [(f"{ds}_{task}", how) for ds, task in m6.MAIN for how in v6b.BASELINE]
    for i, (nm, how) in enumerate(cells):
        c = rec["rule_persons"][nm][how]["cross_100"]
        c["U@2000"]["R_loc"] = above
        c["I"]["stage"] = above
        c["U@2000"]["stage"] = above if i < 2 else below
    assert m6.predictions(rec)["P6"] is True
    rec["rule_persons"][cells[1][0]][cells[1][1]]["cross_100"]["U@2000"]["stage"] = below
    assert m6.predictions(rec)["P6"] is False
    # P5: LWP inside_100 above in both, inside_25 in neither
    rec = copy.deepcopy(base)
    for how in v6b.BASELINE:
        rec["rule_persons"]["lwp_lwp"][how]["inside_100"]["U@2000"]["R_loc"] = above
        rec["rule_persons"]["lwp_lwp"][how]["inside_25"]["U@2000"]["R_loc"] = below
        rec["rule_persons"]["lwp_lwp"][how]["inside_50"]["U@2000"]["R_loc"] = above
    assert m6.predictions(rec)["P5"] is True
    rec["rule_persons"]["lwp_lwp"]["halves"]["inside_25"]["U@2000"]["R_loc"] = above
    assert m6.predictions(rec)["P5"] is False
    # P3 and P1 not evaluable when a label is
    rec = copy.deepcopy(base)
    rec["part_1"]["rule"]["I"]["label"] = m6.NE
    rec["part_1"]["rule"]["U@2000"]["label"] = shows
    assert m6.predictions(rec)["P3"] == m6.NE
    rec["part_1"]["rule"]["I"]["label"] = shows
    assert m6.predictions(rec)["P3"] is True
    rec["part_0"]["by_gate"]["R"]["B@2000"]["label"] = m6.NE
    assert m6.predictions(rec)["P1"] == m6.NE
    # P4 at least 2 of 4
    rec = copy.deepcopy(base)
    bc = list(rec["part_1"]["contrast"]["labelled"]["B@2000"].values())
    for i, c in enumerate(bc):
        c["stage"] = above if i < 2 else below
    assert m6.predictions(rec)["P4"] is True
    bc[1]["stage"] = below
    assert m6.predictions(rec)["P4"] is False
    # P7: any shuffle cell of U_med above -> False
    rec = copy.deepcopy(base)
    for ds, task in m6.MAIN:
        for how in v6b.BASELINE:
            for row in ("shuffled/before", "shuffled_in_halves/before"):
                rec["count_1"][f"{ds}_{task}"][how][row]["U_med"] = below
    assert m6.predictions(rec)["P7"] is True
    rec["count_1"]["lwp_lwp"]["halves"]["shuffled/before"]["U_med"] = above
    assert m6.predictions(rec)["P7"] is False


def test_part_0_and_3_and_main_cells(whole):
    assert whole["part_0"]["by_gate"]["R"]["U_med"]["label"] == "beside the rule: no label"
    st = whole["part_0"]["by_gate"]["R_loc"]["U_med"]["status"]
    assert st["kupa_essay | oddeven | inside/before"] == "fixed by construction"
    assert len(st) == 12 and set(whole["part_0"]["gate_cells"]["U_med"]) == set(st)
    p3 = whole["part_3"]["U_med_vs_U_main_cells"]
    assert p3["lwp_lwp | halves | inside/before"]["fixed_by_construction"]
    assert not p3["lwp_lwp | halves | shuffled/before"]["fixed_by_construction"]
    lb = whole["part_0"]["labels_beside_earlier"]["I"]
    assert lb["v6k_gate_R"] == "mixed directions" and set(lb) == {"v6k_gate_R", "v6l_gate_R_loc", "new_gate_R",
                                                                  "new_gate_R_loc"}
    mc = m6.main_cells_out(whole)
    v = mc["part 2 | I | lwp_lwp | oddeven | cross"]
    assert set(v) == {"25", "50", "100"} and v["25"][0] in ("above", "below", "undecided", "no count")
    r = whole["part_1"]["rule"]["I"]
    assert set(r["classes_by_base"]) == {"copy", "essay"}
    assert sum(sum(x.values()) for x in r["classes_by_base"].values()) == r["cells_read"]


def test_check_v6k_persons_stops_on_a_difference():
    ks_, ls_ = fake_kupa_lwp(10)
    ids = sorted(ks_)
    place = {p: i for i, p in enumerate(ids)}
    rows = ["text/before", *m6.MAIN_ROWS]
    hv = {how: {row: [] for row in rows} for how in v6b.BASELINE}
    for pid in ids:
        s, kinds = ks_[pid]["essay"]
        vec, _, _, _ = m6.person_rows(s, kinds, [0, 1, place[pid]], M, rows)
        for how in v6b.BASELINE:
            for row in rows:
                hv[how][row].append(vec[row][how])
    rp = l6.rule_persons(hv, "kupa", "essay")
    rl = {"rule_persons": {"kupa_essay": rp}}
    m6.check_v6k_persons(ks_, ids, "kupa", "essay", place, M, rl)
    import copy
    bad = copy.deepcopy(rl)
    bad["rule_persons"]["kupa_essay"]["halves"]["shuffled/before"]["I"]["R_loc"]["share_vs_own"] += 0.01
    with pytest.raises(SystemExit, match="differs from v6l"):
        m6.check_v6k_persons(ks_, ids, "kupa", "essay", place, M, bad)


def test_draw_checked_stops():
    order = [f"p{i}" for i in range(300)]
    ids, _ = m6.draw_checked(order, lambda p: {}, 200, 0, k6.ids_sha(order[:200]), "x")
    assert ids == order[:200]
    with pytest.raises(SystemExit, match="not x"):
        m6.draw_checked(order, lambda p: {}, 199, 0, k6.ids_sha(order[:200]), "x")
    with pytest.raises(SystemExit, match="not x"):
        m6.draw_checked(order, lambda p: {}, 200, 1, k6.ids_sha(order[:200]), "x")


# ---------------------------------------------------------------- after the mutations


def many_sentences(n_sent=8, inner=4, long_inner=()):
    """n_sent sentences of inner word starts each; every sentence starts with a long gap; long_inner: (sentence, word)
    whose word-start gap is long."""
    g, starts, ws = [], [0], []
    p = 0
    for k in range(n_sent):
        if k > 0:
            g.append(2500.0 + k)
            p += 1
            starts.append(p)
        for w in range(inner):
            g.append(110.0)
            p += 1
            g.append(3000.0 if (k, w) in long_inner else 300.0 + w)
            p += 1
            ws.append(p)
    g.append(100.0)
    return v6.Series(m6.times_of(np.array(g)), starts, ws)


def test_cross_plan_draws_from_its_generator():
    s = many_sentences()
    plan = m6.cross_plan(s, KEY)
    g = text_gaps(s)
    sent = m6.sentence_of_gap(s.starts, s.times.size)
    givers = [a - 1 for a in s.starts[1:]]
    pool = sorted(w - 1 for w in s.word_starts)
    order = np.random.default_rng([m6.SEED, 94, 2, *KEY]).permutation(len(givers))
    rng = np.random.default_rng([m6.SEED, 94, 3, *KEY])
    taken, want = set(), []
    for i in order:
        gi = givers[int(i)]
        ok = [j for j in pool if j not in taken and sent[j] != sent[gi]]
        pick = ok[int(rng.integers(len(ok)))]
        taken.add(pick)
        want.append((gi, pick))
    assert plan == want
    del g


def test_cross_pool_leaves_out_long_word_gaps():
    s = many_sentences(long_inner={(k, w) for k in range(8) for w in range(3)})
    g = text_gaps(s)
    for _, b in m6.cross_plan(s, KEY):
        assert b is not None and g[b] < 2000.0


@pytest.mark.parametrize("seed", range(6))
def test_crossing_parts_by_hand(seed):
    s, _ = fake_person(seed, 14)
    _, tally = m6.dose_rows(s, KEY)
    n = len(s.ranges("before"))
    sent = m6.sentence_of_gap(s.starts, s.times.size)
    made = tally["cross_100"]["made"]
    for how in v6b.BASELINE:
        A, _ = v6b.parts(n, how)
        inA = set(A.tolist())
        want = sum((int(sent[a]) in inA) != (int(sent[b]) in inA) for a, b in made)
        assert tally["cross_100"]["crossing_parts"][how] == want
    assert any(tally["cross_100"]["crossing_parts"][h] not in (0, len(made) / 2) for h in v6b.BASELINE) or seed > 0


def test_crossing_parts_asymmetric():
    s = many_sentences(n_sent=6)
    _, tally = m6.dose_rows(s, KEY)
    made = tally["cross_100"]["made"]
    sent = m6.sentence_of_gap(s.starts, s.times.size)
    for how in v6b.BASELINE:
        A, _ = v6b.parts(6, how)
        want = sum((int(sent[a]) in set(A.tolist())) != (int(sent[b]) in set(A.tolist())) for a, b in made)
        assert tally["cross_100"]["crossing_parts"][how] == want
    assert tally["cross_100"]["crossing_parts"]["oddeven"] * 2 != len(made) or \
        tally["cross_100"]["crossing_parts"]["halves"] * 2 != len(made)


def test_within_scale_differs_from_pooled_variance_of_one_task():
    rng = np.random.default_rng(9)
    cA, cB = rng.normal(0, 1, (30, 2)), rng.normal(0, 1, (30, 2))
    eA, eB = rng.normal(0, [8.0, 0.2], (30, 2)), rng.normal(0, [8.0, 0.2], (30, 2))
    idx = np.zeros((3, 30), dtype=int)
    _, ow = m6.contrast_one(cA, cB, eA, eB, "copy", idx, scale="within")
    sdw = np.sqrt((np.vstack([cA, cB]).var(axis=0) + np.vstack([eA, eB]).var(axis=0)) / 2)
    ch = (np.linalg.norm((eA - cA) / sdw, axis=1) + np.linalg.norm((eB - cB) / sdw, axis=1)) / 2
    b = np.linalg.norm((cA - cB) / sdw, axis=1)
    assert ow.tolist() == np.where(ch > b, 1.0, np.where(ch == b, 0.5, 0.0)).tolist()
    sdc = np.vstack([cA, cB]).std(axis=0)
    ch2 = (np.linalg.norm((eA - cA) / sdc, axis=1) + np.linalg.norm((eB - cB) / sdc, axis=1)) / 2
    b2 = np.linalg.norm((cA - cB) / sdc, axis=1)
    assert ow.tolist() != np.where(ch2 > b2, 1.0, np.where(ch2 == b2, 0.5, 0.0)).tolist()


def test_signed_key_continuous():
    a = {st: None for st in m6.SIGNED_STAGES}
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (9, 3))
    y = rng.normal(0, 1, (9, 3))
    pairs = [(dict(a, I=x[i]), dict(a, I=y[i])) for i in range(9)]
    out = m6.signed_set(pairs, 2, 0, 9)
    for f in range(3):
        idx = np.random.default_rng([20261031, 97, 2, 0, 5, 9, f]).integers(0, 9, (v6.N_BOOT, 9))
        assert out["I"][str(f)]["median_ci"] == m6.boot_median(y[:, f] - x[:, f], idx)


def test_rising_allows_equal_shares():
    above = cell(0.7, 0.9)
    rp = {f"{ds}_{task}": {how: {f"{fam}_{st}": {k: {"stage": above, "R": above, "R_loc": above, "R_lag": above}
                                               for k in (*m6.RULE_STAGES, *m6.BESIDE)}
                                for fam in m6.FAMILIES for st, _ in m6.STEPS} for how in v6b.BASELINE}
          for ds, task in m6.MAIN}
    assert m6.doses(rp)["I"]["lwp_lwp | oddeven | cross"]["stage"]["rising"] is True


def test_own_contrib_ties():
    v = {"X": np.array([1.0, 2.0])}
    w = {"X": np.array([1.0, 2.0])}
    first = [(v, w), ({"X": np.array([0.0, 0.0])}, {"X": np.array([3.0, 1.0])})]
    sel = [(v, w), ({"X": np.array([5.0, 5.0])}, {"X": np.array([3.0, 1.0])})]
    own = m6.own_contrib(first, sel, "X")
    assert own[0] == 0.5  # change 0 = base 0
    assert float(own.mean()) == v6b.one(first, sel, "X", [1])["share_vs_own"]


def test_contrast_arrays_order():
    c = ({"halves": ({"X": np.array([1.0])}, {"X": np.array([2.0])})},
         {"halves": ({"X": np.array([3.0])}, {"X": np.array([4.0])})})
    cA, cB, eA, eB = m6.contrast_arrays([c], "halves", "X")
    assert (cA[0, 0], cB[0, 0], eA[0, 0], eB[0, 0]) == (1.0, 2.0, 3.0, 4.0)


def test_contrast_rule_classes_by_base():
    above, below = cell(0.7, 0.9), cell(0.2, 0.4)
    keys = [f"{h} | {b}" for h in v6b.BASELINE for b in m6.BASES]
    stages = dict(zip(keys, [above, below, above, above]))
    c = {"labelled": {k: {w: {"stage": stages[w], "R_loc": above, "R": above} for w in keys}
                      for k in m6.CONTRAST_LABELLED}}
    r = m6.contrast_rule(c)["I"]
    # keys: oddeven|copy above, oddeven|essay below, halves|copy above, halves|essay above
    assert r["classes_by_base"] == {"copy": {"above": 2, "below": 0, "undecided": 0},
                                    "essay": {"above": 1, "below": 1, "undecided": 0}}
