import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6l as l6  # noqa: E402
from test_prototype_v6k import fake_person  # noqa: E402

k6 = l6.k6
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]
T = 120000.0


# ---------------------------------------------------------------- the trim


def gaps_with(n: int, at: dict) -> np.ndarray:
    g = np.full(n, 150.0)
    for i, v in at.items():
        g[i] = v
    return g


def test_trim_untouched_record():
    tr = l6.trim_range(gaps_with(40, {20: 200000.0}), T)
    assert (tr["a"], tr["b"], tr["consumed"]) == (0, 40, False) and tr["steps"] == {"start": 0, "end": 0}


@pytest.mark.parametrize("v,cut", [(T - 1.0, False), (T, True), (T - 1e-7, True), (T + 1.0, True)])
def test_trim_boundary_of_the_length(v, cut):
    tr = l6.trim_range(gaps_with(40, {3: v}), T)
    assert tr["a"] == (4 if cut else 0)


def test_trim_window_is_the_first_10_gaps():
    assert l6.trim_range(gaps_with(40, {9: T}), T)["a"] == 10
    assert l6.trim_range(gaps_with(40, {10: T}), T)["a"] == 0


def test_trim_takes_the_last_long_gap_and_repeats():
    tr = l6.trim_range(gaps_with(60, {2: T, 5: T, 14: T}), T)
    # first window gaps 0-9: last long 5 -> a = 6; window 6-15 holds 14 -> a = 15; window 15-24 none
    assert tr["a"] == 15 and tr["steps"]["start"] == 2


def test_trim_end_takes_the_first_long_gap_and_repeats():
    tr = l6.trim_range(gaps_with(60, {57: T, 52: T, 45: T}), T)
    # window 50-59: first long 52 -> b = 52; window 42-51 holds 45 -> b = 45; window 35-44 none
    assert tr["b"] == 45 and tr["steps"]["end"] == 2 and tr["a"] == 0


def test_trim_consumed():
    g = np.full(15, T)
    assert l6.trim_range(g, T)["consumed"]
    assert l6.trim_range(np.full(9, 100.0), T)["consumed"]  # 10 presses
    assert not l6.trim_range(np.full(10, 100.0), T)["consumed"]  # 11 presses


def test_untrimmed_fit_equals_v6k():
    rng = np.random.default_rng(1)
    g = rng.lognormal(6, 1, 300)
    word = rng.random(300) < 0.3
    key = [2, 0, 5, 0]
    fit, tr = l6.trimmed_fit(g, word, T, key)
    assert tr["presses_dropped"] == 0 and tr["time_dropped_share"] == 0.0
    assert fit == k6.fit_writer(np.log(g[word]), key)
    g2 = g.copy()
    g2[1] = T
    fit2, tr2 = l6.trimmed_fit(g2, word, T, key)
    assert tr2["a"] == 2 and fit2 == k6.fit_writer(np.log(g2[2:][word[2:]]), key)


# ---------------------------------------------------------------- the splits' code


@pytest.mark.parametrize("seed", range(8))
def test_fast_equals_v6k_part_vectors(seed):
    s, kinds = fake_person(seed, 14)
    lab = k6.v6b.labels(s, kinds)
    rec = k6.Rec(s.times, s.ranges("before"), lab, M)
    f = l6.Fast(rec)
    rng = np.random.default_rng(seed)
    for _ in range(10):
        idx = np.sort(rng.choice(rec.n, int(rng.integers(1, rec.n)), replace=False))
        ref, _ = l6.part_vectors_l(rec, idx, {"2000": 2000.0})
        assert l6.same_vectors(f.vectors(idx), ref, l6.REL_STAGES)


def test_same_vectors_detects_differences():
    a = {"x": np.array([1.0, 2.0]), "y": None}
    assert l6.same_vectors(a, {"x": np.array([1.0, 2.0]), "y": None}, ("x", "y"))
    assert not l6.same_vectors(a, {"x": np.array([1.0, 2.0 + 1e-15]), "y": None}, ("x", "y"))
    assert not l6.same_vectors(a, {"x": np.array([1.0, 2.0]), "y": np.array([0.0])}, ("x", "y"))


def test_new_stages_of_a_part():
    s, kinds = fake_person(3, 12)
    rec = k6.Rec(s.times, s.ranges("before"), k6.v6b.labels(s, kinds), M)
    out, tally = l6.part_vectors_l(rec, np.arange(0, rec.n, 2), {"2000": 2000.0})
    assert np.array_equal(out["R_loc"], out["R"][:6]) and np.array_equal(out["R_lag"], out["R"][6:])
    _, complete, _ = k6.bursts(rec.gaps, rec.held_of(np.arange(0, rec.n, 2)), 2000.0)
    if complete.size:
        assert out["B_complete@2000"][0] == pytest.approx(complete.mean())
    assert tally["complete_bursts"] == complete.size


# ---------------------------------------------------------------- reliability


def test_corr_split_cases():
    a = np.arange(12.0)
    assert l6.corr_split(a, a)[0] == pytest.approx(1.0)
    assert l6.corr_split(a[:9], a[:9])[1] == "fewer than 10 persons"
    assert l6.corr_split(np.ones(12), a)[1] == "constant"
    assert l6.corr_split(a, -a)[1] == "r = -1"
    b = a.copy()
    b[0] = np.nan
    assert l6.corr_split(b, a)[2] == 11
    assert l6.sb(0.5) == pytest.approx(2 / 3)


def fake_texts(n=14, sentences=12):
    texts = []
    for i in range(n):
        s, kinds = fake_person(40 + i, sentences)
        vec, _, _ = l6.person_rows(s, kinds, [2, 0, i], M, ["text/before"])
        rec = k6.Rec(s.times, s.ranges("before"), k6.v6b.labels(s, kinds), M)
        texts.append((rec, i, vec["text/before"]))
    return texts


def test_reliability_runs_and_is_deterministic():
    texts = fake_texts()
    r1 = l6.reliability(texts, "lwp", "lwp", n_splits=4)
    r2 = l6.reliability(texts, "lwp", "lwp", n_splits=4)
    assert r1 == r2
    assert set(r1["features"]) == {f"{st}[{j}]" for st in l6.REL_STAGES for j in range(l6.REL_DIMS[st])}
    c = r1["features"]["S0[3]"]
    assert c["splits_with_a_value"] + sum(c["splits_without"].values()) == 4
    assert set(c["fixed_parts"]) == {"oddeven", "halves"}


def test_reliability_split_uses_the_person_generator(monkeypatch):
    texts = fake_texts(n=12)
    seen, real = [], np.random.default_rng

    def spy(key=None):
        if isinstance(key, (list, tuple)) and len(key) == 5 and key[1] == 93:
            seen.append(list(key))
        return real(key)

    monkeypatch.setattr(l6.np.random, "default_rng", spy)
    l6.reliability(texts, "kupa", "essay", n_splits=2)
    assert seen == [[l6.SEED, 93, 0, 1, i] for i in range(12)]


def test_reliability_check_bites_when_the_splits_code_differs(monkeypatch):
    texts = fake_texts(n=12)
    real = l6.Fast.vectors

    def broken(self, idx):
        out = real(self, idx)
        out["S0"] = out["S0"] + 1e-9
        return out

    monkeypatch.setattr(l6.Fast, "vectors", broken)
    with pytest.raises(SystemExit):
        l6.reliability(texts, "lwp", "lwp", n_splits=1)


def test_reliability_check_bites_on_the_random_splits_only(monkeypatch):
    texts = fake_texts(n=12)
    real = l6.Fast.vectors
    calls = {"n": 0}

    def broken_late(self, idx):
        calls["n"] += 1
        out = real(self, idx)
        if calls["n"] > 2 * 2 * len(texts):  # the fixed parts pass; the random splits are broken
            out["I"] = None
        return out

    monkeypatch.setattr(l6.Fast, "vectors", broken_late)
    with pytest.raises(SystemExit):
        l6.reliability(texts, "lwp", "lwp", n_splits=2)


# ---------------------------------------------------------------- gates and labels


@pytest.mark.parametrize("r,loc,lag,kind", [("above", "above", "above", "both"),
                                            ("above", "above", "below", "location only"),
                                            ("above", "undecided", "above", "order only"),
                                            ("above", "below", "undecided", "composite only"),
                                            ("undecided", "above", "above", "not seen")])
def test_cell_kind(r, loc, lag, kind):
    assert l6.cell_kind(r, loc, lag) == kind


def cell(lo, hi):
    return {"share_vs_own": (lo + hi) / 2, "share_vs_own_ci": [lo, hi], "left_out_no_vector": 0, "persons": 50}


def fake_rp(gates: dict, stage):
    rp = {}
    for ds, task in l6.MAIN:
        nm = f"{ds}_{task}"
        rp[nm] = {}
        for how in k6.v6b.BASELINE:
            rp[nm][how] = {}
            for row in l6.MAIN_ROWS:
                rp[nm][how][row] = {k: {"stage": stage(k, row), **{g: gates[g](row) for g in l6.GATES}}
                                    for k in (*l6.RULE_STAGES, *l6.BESIDE)}
    return rp


def test_gate_changes_the_readable_cells():
    above, below = cell(0.8, 0.9), cell(0.1, 0.2)
    gates = {"R": lambda row: above, "R_loc": lambda row: above if row == "shuffled/before" else below,
             "R_lag": lambda row: above}
    rp = fake_rp(gates, lambda k, row: below)
    parts = {g: l6.evaluate_rule(rp, g) for g in l6.GATES}
    assert parts["R"]["I"]["cells_read"] == 12 and parts["R_loc"]["I"]["cells_read"] == 4
    lc = l6.label_changes(parts)
    assert lc["I"]["labels"]["R"] == "does not show them beyond it"
    assert lc["I"]["labels"]["R_loc"] == "does not show them beyond it" and not lc["I"]["changed_R_to_R_loc"]
    assert len(lc["I"]["cells_whose_status_differs_R_to_R_loc"]) == 8
    kinds = l6.gate_cells(rp)["I"]
    assert sum(v["kind"] == "both" for v in kinds.values()) == 4
    assert sum(v["kind"] == "order only" for v in kinds.values()) == 8
    gates["R_loc"] = lambda row: below
    parts = {g: l6.evaluate_rule(fake_rp(gates, lambda k, row: below), g) for g in l6.GATES}
    assert parts["R_loc"]["I"]["label"] == "not evaluable"
    assert l6.label_changes(parts)["I"]["changed_R_to_R_loc"]


def test_beside_gets_no_label():
    above = cell(0.8, 0.9)
    rp = fake_rp({g: (lambda row: above) for g in l6.GATES}, lambda k, row: above)
    out = l6.evaluate_rule(rp, "R_loc")
    assert out["B_complete@2000"]["label"] == "beside the rule: no label"
    assert out["B@2000"]["cells_read"] == 6  # 6 fixed by construction


# ---------------------------------------------------------------- end to end on fake persons


@pytest.fixture(scope="module")
def built():
    ids = [f"p{i}" for i in range(12)]
    series = {pid: fake_person(10 + i) for i, pid in enumerate(ids)}
    place = {pid: i for i, pid in enumerate(ids)}
    hv, diag, texts = l6.build(series, ids, "lwp", "lwp", place, M)
    return ids, series, place, hv, diag, texts


def test_build_and_counts(built):
    _, _, _, hv, diag, texts = built
    assert diag["count_persons"] == 12 == len(texts)
    t = diag["trim"]["120"]
    assert t["untrimmed"] + t["consumed"] <= 12
    assert t["difference_exactly_0"] == t["untrimmed"]  # untrimmed records give v6k's threshold exactly
    c1 = l6.count1(hv, "lwp", "lwp")
    for g in l6.GATES:
        assert c1["oddeven"]["text/none"][g] == {"defined": False, "persons": 0}
    b = c1["halves"]["shuffled_in_halves/before"]["B@trim120_fixed"]
    assert b["median_change"] == 0
    rp = l6.rule_persons(hv, "lwp", "lwp")
    assert set(rp["oddeven"]["shuffled/before"]["I"]) == {"stage", *l6.GATES}


def test_reuse_of_v6k_counts_and_keys(built):
    ids, series, place, hv, diag, _ = built
    hv_k, diag_k = k6.build(series, ids, "lwp", "lwp", place, M)
    c_k = k6.count1(hv_k, "lwp", "lwp")
    c_l = l6.count1(hv, "lwp", "lwp")
    for how in k6.v6b.BASELINE:
        for row in l6.ROWS[1:]:
            for k in l6.REUSE_GATE0:
                assert c_l[how][row][k] == c_k[how][row][k]
    rp_k = k6.rule_persons(hv_k, "lwp", "lwp")
    rp_l = l6.rule_persons(hv, "lwp", "lwp")
    for how in k6.v6b.BASELINE:
        for row in l6.MAIN_ROWS:
            for k in (*l6.RULE_STAGES, "Bsd_complete@2000"):
                assert rp_l[how][row][k]["stage"] == rp_k[how][row][k]["stage"]
                assert rp_l[how][row][k]["R"] == rp_k[how][row][k]["R"]
    assert diag["theta_w_text_untrimmed"] == diag_k["theta_w_text"]
    rk = {"writer": {"lwp_lwp": diag_k}, "count_1": {"lwp_lwp": c_k}, "rule_persons": {"lwp_lwp": rp_k}}
    l6.check_against_v6k(rk, "lwp_lwp", c_l, rp_l, diag)
    bad = copy.deepcopy(rk)
    bad["count_1"]["lwp_lwp"]["halves"]["inside/before"]["I"]["share_vs_own"] += 1e-12
    with pytest.raises(SystemExit):
        l6.check_against_v6k(bad, "lwp_lwp", c_l, rp_l, diag)
    bad = copy.deepcopy(rk)
    bad["rule_persons"]["lwp_lwp"]["oddeven"]["shuffled/before"]["U@2000"]["R"]["persons"] -= 1
    with pytest.raises(SystemExit):
        l6.check_against_v6k(bad, "lwp_lwp", c_l, rp_l, diag)
    bad = copy.deepcopy(rk)
    bad["writer"]["lwp_lwp"]["theta_w_text"]["k"]["2"] += 1
    with pytest.raises(SystemExit):
        l6.check_against_v6k(bad, "lwp_lwp", c_l, rp_l, diag)
    part_k = k6.evaluate_rule({"kupa_essay": rp_k, "lwp_lwp": rp_k})
    part_l = l6.evaluate_rule({"kupa_essay": rp_l, "lwp_lwp": rp_l}, "R")
    l6.check_rule_against_v6k(part_l, {"rule": {"part_1": part_k}})
    part_k["I"]["label"] = "x"
    with pytest.raises(SystemExit):
        l6.check_rule_against_v6k(part_l, {"rule": {"part_1": part_k}})


def test_keys_of_v6k_stages_match_v6k():
    for k in ("S0", "R", "I", "U@2000", "B@2000", "Bsd_complete@2000"):
        for gate in (0, 3):
            assert l6.key1("kupa", "essay", k, "shuffled/before", "halves", gate) == \
                k6.key1("kupa", "essay", k, "shuffled/before", "halves", gate)
    new = {tuple(l6.key1("kupa", "essay", k, "inside/before", "oddeven", 0))
           for k in ("R_loc", "R_lag", "B_complete@2000", "U@trim120", "U@trim120_fixed", "U@trim30", "U@trim30_fixed")}
    assert len(new) == 7


def test_construction_check_bites(built):
    ids, series, place, *_ = built
    s, kinds = series[ids[0]]
    vec, _, tallies = l6.person_rows(s, kinds, [2, 0, 0], M, l6.ROWS)
    l6.check_construction(vec, tallies)
    v = copy.deepcopy(vec)
    a, b = v["inside/before"]["oddeven"]
    a["U@trim30_fixed"] = None if a["U@trim30_fixed"] is None else a["U@trim30_fixed"] + 1e-6
    v["inside/before"]["oddeven"] = (a, b)
    if vec["inside/before"]["oddeven"][0]["U@trim30_fixed"] is not None:
        with pytest.raises(SystemExit):
            l6.check_construction(v, tallies)
    v = copy.deepcopy(vec)
    v["shuffled/before"]["halves"][1]["R_lag"] = np.array([0.123])
    with pytest.raises(SystemExit):
        l6.check_construction(v, tallies)


def test_trim_fixed_keeps_the_text_threshold(built):
    ids, series, place, *_ = built
    s, kinds = series[ids[1]]
    _, fits, _ = l6.person_rows(s, kinds, [2, 0, 1], M, l6.ROWS)
    text = fits["trim"]["30"]["text/before"][0]
    assert fits["trim"]["30"]["text/after"] is fits["trim"]["30"]["text/before"]
    assert fits["trim"]["30"]["shuffled/before"][0] is not text


def test_whole_record_on_fake_persons(built):
    _, _, _, hv, diag, texts = built
    c1 = l6.count1(hv, "lwp", "lwp")
    rp = l6.rule_persons(hv, "lwp", "lwp")
    rel = l6.reliability(texts, "lwp", "lwp", n_splits=3)
    rp2 = {"kupa_essay": rp, "lwp_lwp": rp}
    parts = {g: l6.evaluate_rule(rp2, g) for g in l6.GATES}
    rec = {"count_1": {"kupa_essay": c1, "lwp_lwp": c1}, "rule_persons": rp2,
           "reliability": {"kupa_essay": rel, "lwp_lwp": rel}, "writer": {"kupa_essay": diag, "lwp_lwp": diag},
           "rule": {"part_1_by_gate": parts, "label_changes": l6.label_changes(parts), "gate_cells": l6.gate_cells(rp2),
                    "part_2_trim": l6.trims_part2({"kupa_essay": c1, "lwp_lwp": c1})}}
    rec["predictions"] = l6.predictions(rec)
    assert set(rec["predictions"]) == {f"P{i}" for i in range(1, 9)}
    rec = {"main_cells": l6.main_cells_out(rec), **rec}
    k6.v6d.no_per_person(rec)
    k6.v6i.no_long_list(rec)
    json.dumps(rec, allow_nan=False)


def pred_rec(rp, rel, w):
    parts = {g: l6.evaluate_rule(rp, g) for g in l6.GATES}
    return {"rule_persons": rp, "rule": {"label_changes": l6.label_changes(parts), "part_1_by_gate": parts},
            "reliability": {"kupa_essay": rel, "lwp_lwp": rel}, "writer": w}


def test_predictions_true_and_false():
    above, below = cell(0.8, 0.9), cell(0.1, 0.2)
    feat = lambda v: {"mean_rSB": v}  # noqa: E731
    rel = {"features": {**{f"S0[{j}]": feat(0.9) for j in range(7)}, **{f"U@2000[{j}]": feat(0.6) for j in range(4)}}}
    w = {"x": {"trim": {"120": {"ln_theta_t_minus_ln_theta_w_text": {"median": 0.05}},
                        "30": {"theta_t_text": {"theta_ms": {"median": 5000.0}}}},
               "theta_w_text_untrimmed": {"theta_ms": {"median": 9000.0}}}}
    rp = fake_rp({g: (lambda row: above) for g in l6.GATES}, lambda k, row: below)
    assert all(v is True for v in l6.predictions(pred_rec(rp, rel, w)).values())
    rp2 = fake_rp({g: (lambda row: above) for g in l6.GATES}, lambda k, row: below)
    for nm in ("kupa_essay", "lwp_lwp"):
        for how in ("oddeven", "halves"):
            rp2[nm][how]["shuffled/before"]["U@2000"]["R_lag"] = below
    rp2["lwp_lwp"]["halves"]["shuffled/before"]["U@2000"]["R_loc"] = below
    rp2["lwp_lwp"]["halves"]["shuffled/before"]["B_complete@2000"]["stage"] = above
    rp2["kupa_essay"]["halves"]["shuffled/before"]["I"]["R_loc"] = below  # a label change under R_loc
    rel2 = copy.deepcopy(rel)
    rel2["features"]["S0[6]"] = feat(0.79)
    rel2["features"]["U@2000[0]"] = feat(0.49)
    w2 = copy.deepcopy(w)
    w2["x"]["trim"]["120"]["ln_theta_t_minus_ln_theta_w_text"]["median"] = -0.2
    w2["x"]["trim"]["30"]["theta_t_text"]["theta_ms"]["median"] = 1999.0
    p = l6.predictions(pred_rec(rp2, rel2, w2))
    assert all(v is False for k, v in p.items() if k != "P3"), p
    # P3: a label that changes; here I's cells stay below under R_loc, so its label does not change
    rp3 = fake_rp({"R": lambda row: above, "R_loc": lambda row: below, "R_lag": lambda row: above},
                  lambda k, row: below)
    assert l6.predictions(pred_rec(rp3, rel, w))["P3"] is False


def test_p6_ignores_cells_the_gate_does_not_read():
    above, below = cell(0.8, 0.9), cell(0.1, 0.2)
    feat = lambda v: {"mean_rSB": v}  # noqa: E731
    rel = {"features": {**{f"S0[{j}]": feat(0.9) for j in range(7)}, **{f"U@2000[{j}]": feat(0.6) for j in range(4)}}}
    w = {"x": {"trim": {"120": {"ln_theta_t_minus_ln_theta_w_text": None},
                        "30": {"theta_t_text": None}}, "theta_w_text_untrimmed": None}}
    rp = fake_rp({"R": lambda row: below, "R_loc": lambda row: above, "R_lag": lambda row: above},
                 lambda k, row: above)
    p = l6.predictions(pred_rec(rp, rel, w))
    assert p["P6"] == p["P7"] == p["P8"] == l6.NE


def test_key_numbers_are_the_conditions():
    assert (l6.STAGE["R_loc"], l6.STAGE["R_lag"], l6.STAGE["B_complete"]) == (8, 9, 10)
    assert l6.TH_IX == {"2000": 0, "trim120": 5, "trim120_fixed": 6, "trim30": 7, "trim30_fixed": 8}
    assert l6.key1("lwp", "lwp", "U@trim30_fixed", "inside/before", "halves", 4) == \
        [l6.SEED, 90, 2, 0, 2, k6.ROW_IX["inside/before"], 1, 8, 4]
    assert l6.key1("kupa", "copy", "R_lag", "shuffled/before", "oddeven", 1 + 10) == \
        [l6.SEED, 90, 0, 0, 9, k6.ROW_IX["shuffled/before"], 0, 0, 11]


def with_edges(seed, long_ms, sentences=30):
    """A fake writer with long gaps among its first and last presses."""
    s, kinds = fake_person(seed, sentences)
    g = np.diff(s.times)
    g[2] = long_ms
    g[-3] = long_ms
    t = np.concatenate([[0.0], np.cumsum(g)])
    return s.with_times(t), kinds


def test_trimmed_fits_use_v6k_keys_and_drop_the_edges(monkeypatch):
    s, kinds = with_edges(5, 130000.0)
    seen, real = [], k6.fit_writer

    def spy(x, key):
        seen.append((x.size, list(key)))
        return real(x, key)

    monkeypatch.setattr(l6.k6, "fit_writer", spy)
    _, fits, _ = l6.person_rows(s, kinds, [2, 0, 7], M, ["text/before", "shuffled/before"])
    for T in ("120", "30"):
        tr = fits["trim"][T]["text/before"][1]
        assert tr["steps"]["start"] >= 1 and tr["steps"]["end"] >= 1 and tr["a"] == 3
        assert tr["presses_dropped"] >= 6 and tr["time_dropped_share"] > 0
    keys = [k for _, k in seen]
    assert keys.count([2, 0, 7, 0]) == 3  # text: trim 120, trim 30, untrimmed
    assert keys.count([2, 0, 7, k6.ROW_IX["shuffled/before"]]) == 2
    n_text = [n for n, k in seen if k == [2, 0, 7, 0]]  # trim 120, trim 30, untrimmed (in this order)
    for i, T in enumerate(("120", "30")):
        assert n_text[i] == n_text[2] - fits["trim"][T]["text/before"][1]["word_dropped"]


def test_build_tallies_on_trimmed_writers():
    ids = [f"q{i}" for i in range(8)]
    series = {pid: (with_edges(60 + i, 40000.0) if i < 5 else fake_person(60 + i, 30)) for i, pid in enumerate(ids)}
    place = {pid: i for i, pid in enumerate(ids)}
    _, diag, _ = l6.build(series, ids, "lwp", "lwp", place, M)
    t30, t120 = diag["trim"]["30"], diag["trim"]["120"]
    assert t30["trimmed_start"] == 5 and t30["trimmed_end"] == 5 and t30["untrimmed"] == 3
    assert t120["untrimmed"] == 8 and t120["difference_exactly_0"] == 8
    assert t30["difference_exactly_0"] == 3 or t30["difference_exactly_0"] >= 3
    assert t30["presses_dropped"]["max"] > 0 and t120["presses_dropped"]["max"] == 0


def test_thresholds_passed_per_row(monkeypatch):
    s, kinds = with_edges(9, 40000.0)
    seen, real = [], l6.part_vectors_l

    def spy(rec, idx, th):
        seen.append(dict(th))
        return real(rec, idx, th)

    monkeypatch.setattr(l6, "part_vectors_l", spy)
    rows = ["text/before", "text/after", "shuffled/before", "inside/before"]
    _, fits, _ = l6.person_rows(s, kinds, [2, 0, 3], M, rows)
    text30 = fits["trim"]["30"]["text/before"][0]["theta"]
    for i, row in enumerate(rows):
        for th in seen[4 * i:4 * i + 4]:
            assert th["2000"] == 2000.0 and th["trim30_fixed"] == text30
            assert th["trim30"] == fits["trim"]["30"][row][0]["theta"]
    assert seen[8]["trim30"] != text30


def test_untrimmed_check_bites(monkeypatch):
    ids = [f"p{i}" for i in range(3)]
    series = {pid: fake_person(10 + i) for i, pid in enumerate(ids)}
    real = l6.trimmed_fit

    def off(gaps, word, T, key):
        return real(gaps, word, T, [*key[:-1], key[-1] + 50])

    monkeypatch.setattr(l6, "trimmed_fit", off)
    with pytest.raises(SystemExit):
        l6.build(series, ids, "lwp", "lwp", {pid: i for i, pid in enumerate(ids)}, M)


def test_reliability_values_replay():
    texts = fake_texts(n=12)
    out = l6.reliability(texts, "lwp", "lwp", n_splits=3)
    rngs = [np.random.default_rng([l6.SEED, 93, 2, 0, place]) for _, place, _ in texts]
    rs = []
    for _ in range(3):
        a, b = [], []
        for (rec, _, _), rng in zip(texts, rngs):
            perm = rng.permutation(rec.n)
            h = rec.n // 2
            a.append(l6.part_vectors_l(rec, np.sort(perm[:h]), {"2000": 2000.0})[0]["S0"][3])
            b.append(l6.part_vectors_l(rec, np.sort(perm[h:]), {"2000": 2000.0})[0]["S0"][3])
        rs.append(np.corrcoef(a, b)[0, 1])
    rs = np.array(rs)
    c = out["features"]["S0[3]"]
    assert c["mean_r"] == pytest.approx(rs.mean()) and c["mean_rSB"] == pytest.approx((2 * rs / (1 + rs)).mean())
    assert c["rSB_percentile_95"][0] == pytest.approx(np.percentile(2 * rs / (1 + rs), 2.5))
    f = c["fixed_parts"]["oddeven"]
    a = np.array([v["oddeven"][0]["S0"][3] for _, _, v in texts])
    b = np.array([v["oddeven"][1]["S0"][3] for _, _, v in texts])
    assert f["r"] == pytest.approx(np.corrcoef(a, b)[0, 1]) and f["mean_abs_A_minus_B"] == pytest.approx(
        np.mean(np.abs(a - b)))
    assert c["low"] == (c["mean_rSB"] < 0.5)


def test_corr_split_boundary_of_persons():
    a = np.arange(10.0)
    assert l6.corr_split(a, a)[0] == pytest.approx(1.0)


def test_five_or_none():
    assert l6.five_or_none([1, 2, 3, 4]) is None and l6.five_or_none([1, 2, 3, 4, 5])["median"] == 3


def test_trim_exactly_11_presses_left_is_kept():
    tr = l6.trim_range(gaps_with(20, {9: T}), T)
    assert tr["a"] == 10 and not tr["consumed"]  # 10 gaps, 11 presses


def test_reliability_odd_number_of_sentences():
    texts = []
    for i in range(12):
        s, kinds = fake_person(70 + i, 13)
        vec, _, _ = l6.person_rows(s, kinds, [2, 0, i], M, ["text/before"])
        texts.append((k6.Rec(s.times, s.ranges("before"), k6.v6b.labels(s, kinds), M), i, vec["text/before"]))
    out = l6.reliability(texts, "lwp", "lwp", n_splits=2)
    rngs = [np.random.default_rng([l6.SEED, 93, 2, 0, place]) for _, place, _ in texts]
    rs = []
    for _ in range(2):
        a, b = [], []
        for (rec, _, _), rng in zip(texts, rngs):
            assert rec.n % 2 == 1
            perm = rng.permutation(rec.n)
            a.append(l6.part_vectors_l(rec, np.sort(perm[:rec.n // 2]), {"2000": 2000.0})[0]["S0"][2])
            b.append(l6.part_vectors_l(rec, np.sort(perm[rec.n // 2:]), {"2000": 2000.0})[0]["S0"][2])
        rs.append(np.corrcoef(a, b)[0, 1])
    assert out["features"]["S0[2]"]["mean_r"] == pytest.approx(np.mean(rs))


def test_theta_difference_has_its_sign():
    ids = [f"q{i}" for i in range(6)]
    series = {pid: with_edges(80 + i, 40000.0) for i, pid in enumerate(ids)}
    place = {pid: i for i, pid in enumerate(ids)}
    _, diag, _ = l6.build(series, ids, "lwp", "lwp", place, M)
    d = []
    for pid in ids:
        s, kinds = series[pid]
        _, fits, _ = l6.person_rows(s, kinds, [2, 0, place[pid]], M, ["text/before"])
        d.append(math.log(fits["trim"]["30"]["text/before"][0]["theta"]) - math.log(fits["text_untrimmed"]["theta"]))
    got = diag["trim"]["30"]["ln_theta_t_minus_ln_theta_w_text"]
    assert [got[k] for k in ("min", "median", "max")] == pytest.approx([min(d), np.median(d), max(d)])
    assert min(d) != max(d)  # the sign shows: some records' thresholds moved


def test_p2_needs_6_of_8():
    above, below = cell(0.8, 0.9), cell(0.1, 0.2)
    feat = lambda v: {"mean_rSB": v}  # noqa: E731
    rel = {"features": {**{f"S0[{j}]": feat(0.9) for j in range(7)}, **{f"U@2000[{j}]": feat(0.6) for j in range(4)}}}
    w = {"x": {"trim": {"120": {"ln_theta_t_minus_ln_theta_w_text": {"median": 0.0}},
                        "30": {"theta_t_text": {"theta_ms": {"median": 5000.0}}}},
               "theta_w_text_untrimmed": {"theta_ms": {"median": 9000.0}}}}
    for n_below, expect in ((2, True), (3, False)):
        rp = fake_rp({g: (lambda row: above) for g in l6.GATES}, lambda k, row: below)
        cells_ = [(nm, how, row) for nm in ("kupa_essay", "lwp_lwp") for how in ("oddeven", "halves")
                  for row in ("shuffled/before", "shuffled_in_halves/before")]
        for nm, how, row in cells_[:n_below]:
            rp[nm][how][row]["U@2000"]["R_lag"] = below
        assert l6.predictions(pred_rec(rp, rel, w))["P2"] is expect
