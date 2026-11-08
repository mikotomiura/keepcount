import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))
import prototype_v6k as k6  # noqa: E402

C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]


# ---------------------------------------------------------------- a fake writer


def fake_person(seed: int, sentences: int = 12):
    """Kinds and times of a fake typed text: words of letters, spaces, commas, sentence ends."""
    rng = np.random.default_rng(seed)
    kinds = []
    for _ in range(sentences):
        words = int(rng.integers(5, 10))
        for w in range(words):
            kinds += ["alnum"] * int(rng.integers(2, 7))
            if w == words - 1:
                kinds.append("end")
            elif rng.random() < 0.15:
                kinds.append("sub")
            kinds.append("space")
    gaps = []
    for i in range(1, len(kinds)):
        base = rng.lognormal(math.log(150), 0.35)
        if kinds[i] == "alnum" and kinds[i - 1] == "space":
            base = rng.lognormal(math.log(320), 0.5)
            if i > 1 and kinds[i - 2] == "end":
                base = rng.lognormal(math.log(2600), 0.5)
        gaps.append(base)
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    k6v = k6.v6b.v6_kinds(kinds)
    s = k6.v6.Series(times, k6.v6.kupa_starts(k6v), k6.v6.kupa_word_starts(k6v))
    return s, kinds


# ---------------------------------------------------------------- bursts


def test_bursts_by_hand():
    g = np.array([100, 3000, 100, 100, 2500, 2500, 100.0])
    lengths, complete, cut = k6.bursts(g, np.arange(7), 2000)
    assert lengths.tolist() == [2, 3, 1, 2] and complete.tolist() == [3, 1] and cut == 2
    lengths, complete, cut = k6.bursts(g, np.array([0, 1, 2, 5, 6]), 2000)
    assert lengths.tolist() == [2, 2, 1, 2] and complete.size == 0 and cut == 4
    lengths, _, cut = k6.bursts(np.array([100.0, 200.0]), np.arange(2), 2000)
    assert lengths.tolist() == [3] and cut == 1


def test_burst_threshold_is_at_or_above():
    g = np.array([100.0, 2000.0, 100.0])
    assert k6.bursts(g, np.arange(3), 2000)[0].tolist() == [2, 2]
    assert k6.bursts(g, np.arange(3), 2000.5)[0].tolist() == [4]


@pytest.mark.parametrize("seed", range(20))
def test_mean_burst_is_order_free_within_runs(seed):
    rng = np.random.default_rng(seed)
    g = np.where(rng.random(60) < 0.2, rng.uniform(2100, 5000, 60), rng.uniform(80, 400, 60))
    held = np.sort(rng.choice(60, 40, replace=False))
    lengths, _, _ = k6.bursts(g, held, 2000)
    assert lengths.sum() == held.size + len(k6.runs(held))
    assert lengths.mean() == pytest.approx(k6.b_identity(g, held, 2000))
    g2 = g.copy()
    for r in k6.runs(held):
        g2[r] = rng.permutation(g[r])
    assert k6.bursts(g2, held, 2000)[0].mean() == pytest.approx(lengths.mean())


# ---------------------------------------------------------------- the writer's threshold


def test_fit_recovers_three_components_and_is_deterministic():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(5.6, 0.25, 600), rng.normal(6.6, 0.4, 250), rng.normal(8, 0.6, 60)])
    a = k6.fit_writer(x, [0, 1, 2, 0])
    b = k6.fit_writer(x, [0, 1, 2, 0])
    assert a == b and a["k"] == 3
    assert math.log(a["theta"]) == pytest.approx(6.6 + 3 * 0.4, abs=0.25)


def test_fit_k1_closed_form_with_few_gaps():
    x = np.log(np.array([200.0, 300, 250, 400, 350]))
    f = k6.fit_writer(x, [0, 0, 0, 0])
    assert f["k"] == 1 and f["theta"] == pytest.approx(math.exp(x.mean() + 3 * x.std()))
    assert k6.fit_writer(np.array([5.0]), [0, 0, 0, 0]) is None
    same = k6.fit_writer(np.full(10, 5.0), [0, 0, 0, 0])
    assert same["theta"] == pytest.approx(math.exp(5.0 + 3 * 0.01))


def test_fit_k2_needs_40_gaps():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(5, 0.1, 20), rng.normal(8, 0.1, 19)])
    assert k6.fit_writer(x, [0, 0, 0, 0])["k"] == 1
    x = np.concatenate([rng.normal(5, 0.1, 20), rng.normal(8, 0.1, 20)])
    f = k6.fit_writer(x, [0, 0, 0, 0])
    assert f["k"] == 2 and math.log(f["theta"]) == pytest.approx(8 + 0.3, abs=0.2)


def test_fit_uses_the_second_component_of_two():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(5, 0.2, 400), rng.normal(6.5, 0.3, 200)])
    f = k6.fit_writer(x, [0, 0, 0, 0])
    assert f["k"] == 2 and math.log(f["theta"]) == pytest.approx(6.5 + 0.9, abs=0.15)


def test_fit_discards_and_falls_back():
    x = np.random.default_rng(5).normal(5, 0.3, 400)
    x = np.concatenate([x, [9.0]])
    f = k6.fit_writer(x, [0, 0, 0, 0])
    assert f["discarded"] > 0 and f["k"] == 1


def test_bic_tie_takes_the_smaller_k(monkeypatch):
    real = k6.em

    def em_tie(x, mu, sd, w):
        f = real(x, mu, sd, w)
        if f is not None and len(mu) > 1:
            k = len(mu)
            n = x.size
            ll1 = k6.normal_ll(x, float(x.mean()), float(x.std()))
            f["ll"] = ll1 + (3 * k - 3) * math.log(n) / 2  # the BIC of k equals that of k = 1
        return f

    monkeypatch.setattr(k6, "em", em_tie)
    rng = np.random.default_rng(6)
    x = np.concatenate([rng.normal(5, 0.2, 100), rng.normal(7, 0.2, 100)])
    assert k6.fit_writer(x, [0, 0, 0, 0])["k"] == 1
    monkeypatch.setattr(k6, "em", real)
    assert k6.fit_writer(x, [0, 0, 0, 0])["k"] == 2


def test_em_tiny_component_is_discarded():
    x = np.concatenate([np.random.default_rng(5).normal(5, 0.3, 300)])
    f = k6.em(x, np.array([5.0, 9.0]), np.array([0.3, 0.3]), np.array([0.5, 0.5]))
    assert f is None or np.any(f["w"] < k6.W_MIN_WEIGHT)


# ---------------------------------------------------------------- part vectors


def test_part_vectors_of_a_record():
    s, kinds = fake_person(0)
    lab = k6.v6b.labels(s, kinds)
    rec = k6.Rec(s.times, s.ranges("before"), lab, M)
    v, t = k6.part_vectors(rec, np.arange(rec.n), {"2000": 2000.0, "writer": None})
    assert v["S0"].size == 7 and v["S1"].size == 4 and v["U@2000"].size == 4 and v["I"].size == 3
    assert v["U@writer"] is None and v["B@writer"] is None
    g = rec.gaps
    assert v["I"][0] == pytest.approx(g[g > 200].mean() / 1000)
    assert v["I"][2] == pytest.approx(g[(g > 200) & (lab == "word")].mean() / 1000)
    sp = [math.log(np.median(g[lo:hi][g[lo:hi] < 2000])) for lo, hi in s.ranges("before")]
    assert v["U@2000"][0] == pytest.approx(np.median(sp))
    assert 0 <= t["edge_share_2000"] <= 1
    lengths, complete, _ = k6.bursts(g, rec.held_of(np.arange(rec.n)), 2000.0)
    assert v["Bsd@2000"][0] == pytest.approx(np.sqrt(np.mean((lengths - lengths.mean()) ** 2)))
    assert v["Bsd_complete@2000"][0] == pytest.approx(complete.std())


def test_tiles_match_v6():
    s, kinds = fake_person(1)
    rec = k6.Rec(s.times, s.ranges("before"), k6.v6b.labels(s, kinds), M)
    tt = k6.v6.Tiles(s, "sentence", "before", M, C4["relief"], 2000.0)
    idx = np.arange(0, rec.n, 2)
    v, _ = k6.part_vectors(rec, idx, {"2000": 2000.0})
    ref = tt.vectors(idx)
    assert np.array_equal(v["S0"], ref["S0"]) and np.array_equal(v["S1"], ref["S1"])


def copy_rows(d):
    return {r: {h: tuple(dict(x) for x in p) for h, p in dd.items()} for r, dd in d.items()}


@pytest.mark.parametrize("where", [("vec", "inside/before", "halves", "B@2000"),
                                   ("vec", "shuffled_in_halves/before", "halves", "B@500"),
                                   ("vec", "inside/before", "oddeven", "U@writer_fixed"),
                                   ("vec", "inside/before", "oddeven", "U@1000"),
                                   ("vec", "inside/before", "halves", "S1"),
                                   ("tally", "shuffled_in_halves/before", "halves", "I0")])
def test_construction_holds_and_its_check_bites(where):
    s, kinds = fake_person(2)
    vec, fits, tallies = k6.person_rows(s, kinds, [0, 1, 5], M, k6.ROWS)
    k6.check_construction(vec, tallies)
    assert fits["text/after"] is fits["text/before"]
    what, row, how, nm = where
    vec2, tal2 = copy_rows(vec), copy_rows(tallies)
    target = vec2 if what == "vec" else tal2
    target[row][how][0][nm] = target[row][how][0][nm] + 0.5
    with pytest.raises(SystemExit):
        k6.check_construction(vec2, tal2)


def test_construction_compares_i0_where_i_is_missing():
    s, kinds = fake_person(2)
    vec, _, tallies = k6.person_rows(s, kinds, [0, 1, 5], M, k6.ROWS)
    vec2 = copy_rows(vec)
    vec2["shuffled_in_halves/before"]["halves"][0]["I"] = None
    k6.check_construction(vec2, tallies)


def test_writer_and_writer_fixed_wiring():
    s, kinds = fake_person(4, 20)
    vec, fits, _ = k6.person_rows(s, kinds, [0, 1, 6], M, k6.ROWS)
    lab = k6.v6b.labels(s, kinds)
    row = "shuffled/before"
    rec = k6.Rec(k6.row_times(s, row, [0, 1, 6]), s.ranges("before"), lab, M)
    A, _ = k6.v6b.parts(rec.n, "oddeven")
    tw, rw = fits["text/before"]["theta"], fits[row]["theta"]
    assert tw != rw
    ref_fixed, _ = k6.part_vectors(rec, A, {"x": tw})
    ref_own, _ = k6.part_vectors(rec, A, {"x": rw})
    got = vec[row]["oddeven"][0]
    assert np.array_equal(got["U@writer_fixed"], ref_fixed["U@x"])
    assert np.array_equal(got["U@writer"], ref_own["U@x"])
    assert np.array_equal(got["B@writer"], ref_own["B@x"])


def test_row_keys(monkeypatch):
    s, kinds = fake_person(3)
    seen, real = [], np.random.default_rng

    def spy(key):
        seen.append(list(key))
        return real(key)

    monkeypatch.setattr(k6.np.random, "default_rng", spy)
    for row in k6.ROWS[1:]:
        k6.row_times(s, row, [2, 0, 9])
    for key in ([k6.SEED, 6, 4, 2, 0, 9], [k6.SEED, 6, 5, 2, 0, 9], [k6.SEED, 6, 7, 2, 0, 0, 9],
                [k6.SEED, 6, 32, 2, 0, 9, 2], [k6.SEED, 6, 33, 2, 0, 9, 2]):
        assert key in seen


# ---------------------------------------------------------------- rule


def cell(lo, hi):
    return {"share_vs_own": (lo + hi) / 2, "share_vs_own_ci": [lo, hi], "left_out_no_vector": 0}


def test_labels():
    assert k6.label(["above"] * 4) == "shows these operations beyond the smallest difference"
    assert k6.label(["below"] * 6) == "does not show them beyond it"
    assert k6.label(["below", "undecided", "below", "below"]) == "not beyond, partly undecided"
    assert k6.label(["above", "undecided", "above", "above"]) == "beyond, partly undecided"
    assert k6.label(["undecided"] * 4) == "undecided"
    assert k6.label(["above", "below", "below", "below"]) == "mixed directions"
    assert k6.label(["below"] * 3) == "not evaluable"


def test_evaluate_rule_skips_construction_and_failed_controls():
    rp = {}
    for ds, task in k6.MAIN:
        rp[f"{ds}_{task}"] = {how: {row: {k: {"stage": cell(0.1, 0.3), "R": cell(0.8, 0.9)}
                                          for k in (*k6.RULE_STAGES, *k6.BESIDE_RULE)}
                                    for row in k6.MAIN_ROWS} for how in k6.v6b.BASELINE}
    rp["lwp_lwp"]["oddeven"]["shuffled/before"]["I"]["R"] = cell(0.5, 0.7)
    out = k6.evaluate_rule(rp)
    assert out["B@2000"]["cells_read"] == 6 and out["B@2000"]["label"] == "does not show them beyond it"
    assert out["U@2000"]["cells_read"] == 8
    assert out["I"]["cells_read"] == 11
    assert out["I"]["status"]["lwp_lwp | oddeven | shuffled/before"] == "control not met"
    rp["kupa_essay"]["halves"]["shuffled/before"]["Bsd@2000"]["stage"] = cell(0.7, 0.9)
    assert k6.evaluate_rule(rp)["Bsd@2000"]["label"] == "mixed directions"
    k6.v6i.no_long_list(out)


# ---------------------------------------------------------------- end to end on fake persons


def test_end_to_end_on_fake_persons():
    ids = [f"p{i}" for i in range(12)]
    series = {pid: fake_person(10 + i) for i, pid in enumerate(ids)}
    place = {pid: i for i, pid in enumerate(ids)}
    hv, diag = k6.build(series, ids, "lwp", "lwp", place, M)
    assert diag["count_persons"] == 12
    c1 = k6.count1(hv, "lwp", "lwp")
    assert c1["oddeven"]["text/none"]["R"] == {"defined": False, "persons": 0}
    b = c1["halves"]["shuffled_in_halves/before"]["B@2000"]
    assert b["median_change"] == 0 and b["share_vs_own"] == 0
    rp = {"kupa_essay": k6.rule_persons(hv, "lwp", "lwp"), "lwp_lwp": k6.rule_persons(hv, "lwp", "lwp")}
    out = k6.evaluate_rule(rp)
    assert set(out) == {*k6.RULE_STAGES, *k6.BESIDE_RULE}
    part2 = k6.thresholds_part2({"kupa_essay": c1, "lwp_lwp": c1})
    assert part2["U"]["500"]["cells"] == 12
    kser = {pid: {"copy": fake_person(100 + i, 8), "essay": fake_person(200 + i)} for i, pid in enumerate(ids)}
    vec2, w = k6.vectors2(kser, ids, place, M)
    c2 = k6.count2(vec2)
    assert c2["kupa_copy_to_essay"]["S0"]["persons"] == 12
    assert set(c2["kupa_copy_to_essay"]["paired"]) == {f"{x} - {y}" for x, y in k6.PAIRS}
    rec = {"count_1": {"kupa_essay": c1, "lwp_lwp": c1}, "rule": {"part_1": out, "part_2": part2},
           "rule_persons": rp, "count_2": c2, "writer": {"kupa_essay": diag, "kupa_count_2_records": w}}
    assert out["Bsd_complete@2000"]["label"] == "beside the rule: no label"
    assert "Bsd_complete@2000" not in c2["kupa_copy_to_essay"]
    rec["predictions"] = k6.predictions(rec)
    rec = {"main_cells": k6.main_cells_out(rec), **rec}
    k6.v6d.no_per_person(rec)
    k6.v6i.no_long_list(rec)
    json.dumps(rec, allow_nan=False)


def test_check_count1_detects_a_difference():
    ids = [f"p{i}" for i in range(6)]
    series = {pid: fake_person(30 + i) for i, pid in enumerate(ids)}
    place = {pid: i for i, pid in enumerate(ids)}
    rows = k6.v6.ROWS["kupa"]["sentence"]
    hv = k6.hv_for_check(series, ids, None, "lwp", place, M, rows)
    ref = {}
    for how in k6.v6b.BASELINE:
        ref[how] = {}
        for row in rows[1:]:
            first, sel = hv[how]["text/before"], hv[how][row]
            res = k6.v6b.one(first, sel, "S1", [1, 2, 1, k6.ROW_IX[row], k6.v6b.BASELINE[how]])
            ref[how][row] = {"S1": res}
    k6.check_count1(hv, ref, rows[1:], {"S1": ("S1", 1)}, [1, 2], "fake")
    ref["halves"]["shuffled/before"]["S1"] = dict(ref["halves"]["shuffled/before"]["S1"], share_vs_own=-1)
    with pytest.raises(SystemExit):
        k6.check_count1(hv, ref, rows[1:], {"S1": ("S1", 1)}, [1, 2], "fake")


# ---------------------------------------------------------------- keys of the counts


def spy_keys(monkeypatch):
    seen, real = [], np.random.default_rng

    def spy(key=None):
        if isinstance(key, (list, tuple)):
            seen.append(list(key))
        return real(key)

    monkeypatch.setattr(k6.np.random, "default_rng", spy)
    monkeypatch.setattr(k6.v6b.np.random, "default_rng", spy)
    return seen


def test_count_keys(monkeypatch):
    ids = [f"p{i}" for i in range(6)]
    series = {pid: fake_person(40 + i) for i, pid in enumerate(ids)}
    place = {pid: i for i, pid in enumerate(ids)}
    hv, _ = k6.build(series, ids, "lwp", "lwp", place, M)
    seen = spy_keys(monkeypatch)
    k6.count1(hv, "kupa", "essay")
    assert [k6.SEED, 90, 0, 1, 2, 4, 1, 1, 0] in seen      # U@1000, shuffled, halves, gate 0
    assert [k6.SEED, 90, 0, 1, 3, 6, 0, 4, 0] in seen      # B@writer_fixed, random_half, odd/even
    seen.clear()
    k6.rule_persons(hv, "kupa", "essay")
    assert [k6.SEED, 90, 0, 1, 5, 3, 0, 0, 6] in seen      # I on the rule persons, inside
    assert [k6.SEED, 90, 0, 1, 6, 3, 0, 0, 6] in seen      # R with I's gate
    assert [k6.SEED, 90, 0, 1, 6, 5, 1, 0, 5] in seen      # R with Bsd's gate
    kser = {pid: {"copy": fake_person(300 + i, 8), "essay": fake_person(400 + i)} for i, pid in enumerate(ids)}
    vec2, _ = k6.vectors2(kser, ids, place, M)
    seen.clear()
    k6.count2(vec2)
    assert [k6.SEED, 91, 0, 2, 1, 2, 0] in seen            # U@500, copy -> essay, registration scale
    assert [k6.SEED, 91, 1, 2, 0, 0, 7] in seen            # paired U@writer - U@2000: y's stage and threshold
    assert [k6.SEED, 91, 0, 1, 0, 0, 1] in seen            # paired U@2000 - S1


def test_paired_on_common_persons_and_too_few():
    ids = [f"p{i}" for i in range(6)]
    place = {pid: i for i, pid in enumerate(ids)}
    kser = {pid: {"copy": fake_person(500 + i, 8), "essay": fake_person(600 + i)} for i, pid in enumerate(ids)}
    vec2, _ = k6.vectors2(kser, ids, place, M)
    vec2["essay"][0]["U@500"] = None
    c2 = k6.count2(vec2)
    assert c2["kupa_copy_to_essay"]["paired"]["U@500 - U@2000"]["persons"] == 5
    assert c2["kupa_copy_to_essay"]["U@2000"]["persons"] == 6
    vec2["essay"][0]["U@500"] = vec2["copy"][0]["U@500"]
    vec2["copy"][1]["U@2000"] = None
    c2 = k6.count2(vec2)
    assert c2["kupa_copy_to_essay"]["paired"]["U@500 - U@2000"]["persons"] == 5
    for i in range(5):
        vec2["copy"][i]["U@500"] = None
    c2 = k6.count2(vec2)
    assert c2["kupa_copy_to_essay"]["paired"]["U@500 - U@2000"]["main"] == {"too_few_persons": True}


# ---------------------------------------------------------------- predictions


def fake_rec(stage_cls, r_cls="above", u_pair_lo=0.01, theta=1000.0, same=8):
    vals = {"above": (0.7, 0.9), "below": (0.1, 0.3), "undecided": (0.5, 0.7)}
    cnt, rp = {}, {}
    for ds, task in k6.MAIN:
        nm = f"{ds}_{task}"
        cnt[nm] = {how: {row: {"R": cell(*vals[r_cls])} for row in k6.MAIN_ROWS} for how in k6.v6b.BASELINE}
        rp[nm] = {how: {row: {k: {"stage": cell(*vals[stage_cls.get((k, row), "below")]), "R": cell(*vals[r_cls])}
                              for k in (*k6.RULE_STAGES, *k6.BESIDE_RULE)} for row in k6.MAIN_ROWS}
                  for how in k6.v6b.BASELINE}
    part2 = {st: {th: {"same_as_2000": same} for th in ("1000", "500")} for st in k6.TH_STAGES}
    c2 = {"kupa_copy_to_essay": {"paired": {"U@2000 - S1": {"main": {"ci": [u_pair_lo, 0.1]}}}}}
    return {"count_1": cnt, "rule_persons": rp, "rule": {"part_2": part2}, "count_2": c2,
            "writer": {"kupa_count_2_records": {"essay": {"theta_ms": {"median": theta}}}}}


def test_predictions_true_and_false():
    sh = ("shuffled/before", "shuffled_in_halves/before")
    up = {**{("Bsd@2000", r): "above" for r in sh}, **{("I@x", r): "above" for r in sh},
          **{("I", r): "above" for r in (*sh, "inside/before")}}
    p = k6.predictions(fake_rec(up))
    assert all(p.values()), p
    assert not k6.predictions(fake_rec(up, r_cls="undecided"))["P1"]
    assert not k6.predictions(fake_rec({**up, ("U@2000", "shuffled/before"): "above"}))["P2"]
    assert not k6.predictions(fake_rec({**up, ("B@2000", "shuffled/before"): "undecided"}))["P3"]
    assert k6.predictions(fake_rec({**up, ("B@2000", "inside/before"): "above"}))["P3"]  # fixed cells not read
    assert not k6.predictions(fake_rec({k: v for k, v in up.items() if k[0] != "Bsd@2000"}))["P4"]
    rec3 = fake_rec(up)
    shuffled_cells = [(nm, how, row) for nm in rec3["rule_persons"] for how in k6.v6b.BASELINE for row in sh]
    for nm, how, row in shuffled_cells[3:]:
        rec3["rule_persons"][nm][how][row]["Bsd@2000"]["stage"] = cell(0.1, 0.3)
    assert not k6.predictions(rec3)["P4"]
    for nm, how, row in shuffled_cells[3:4]:
        rec3["rule_persons"][nm][how][row]["Bsd@2000"]["stage"] = cell(0.7, 0.9)
    assert k6.predictions(rec3)["P4"]
    assert not k6.predictions(fake_rec({**up, ("I", "inside/before"): "below"}))["P5"]
    assert not k6.predictions(fake_rec(up, u_pair_lo=0.0))["P6"]
    assert k6.predictions(fake_rec(up, theta=500.0))["P7"] and not k6.predictions(fake_rec(up, theta=2001.0))["P7"]
    assert not k6.predictions(fake_rec(up, same=7))["P8"]


def test_part2_does_not_count_missing_as_agreement():
    c = {"share_vs_own": 0.2, "share_vs_own_ci": [0.1, 0.3]}
    cnt = {}
    for ds, task in k6.MAIN:
        cnt[f"{ds}_{task}"] = {how: {row: {k6.sk(st, th): ({"too_few_persons": True} if th in ("2000", "500") else c)
                                           for st in k6.TH_STAGES for th in k6.THRESH}
                                     for row in k6.MAIN_ROWS} for how in k6.v6b.BASELINE}
    out = k6.thresholds_part2(cnt)
    assert out["U"]["500"]["same_as_2000"] == 0 and out["U"]["500"]["cells_without_a_count"] == 12


def test_check_v6_count2_bites():
    ids = [f"p{i}" for i in range(6)]
    series = {pid: {"copy": fake_person(700 + i, 8), "essay": fake_person(800 + i)} for i, pid in enumerate(ids)}
    ref = {"count_2_person": {}}
    for name, (a, b, _) in k6.DIRECTIONS.items():
        ref["count_2_person"][name] = {"sentence": {"main": {}}}
        for st in ("S0", "S1"):
            vs = {t: [k6.part_vectors(k6.Rec(series[p][t][0].times, series[p][t][0].ranges("before"),
                                             k6.v6b.labels(*series[p][t]), M),
                                      np.arange(len(series[p][t][0].ranges("before"))), {"2000": 2000.0})[0][st]
                      for p in ids] for t in (a, b)}
            d, _ = k6.distances(np.array(vs[a]), np.array(vs[b]), "main")
            ref["count_2_person"][name]["sentence"]["main"][st] = {"rank_auc": float(k6.v6.per_person(d)[1].mean())}
    k6.check_v6_count2(series, ids, M, ref)
    ref["count_2_person"]["kupa_essay_to_copy"]["sentence"]["main"]["S1"]["rank_auc"] += 1e-9
    with pytest.raises(SystemExit):
        k6.check_v6_count2(series, ids, M, ref)


def test_a_gap_of_200_ms_is_not_over_after_the_rebuild():
    s, kinds = fake_person(9, 14)
    lab = k6.v6b.labels(s, kinds)
    g = np.diff(s.times)
    w = np.flatnonzero(lab == "within")[:3]
    for j, v in zip(w, (200.0, 200.0 + 7e-12, 200.0 - 7e-12)):
        g[j] = v
    t = np.concatenate([[0.0], np.cumsum(g)])
    rec = k6.Rec(t, s.ranges("before"), lab, M)
    v, tal = k6.part_vectors(rec, np.arange(rec.n), {"2000": 2000.0})
    gg = rec.gaps
    mask = gg > 200.0 + 1e-6
    assert tal["I0"] == pytest.approx(gg[mask].mean() / 1000, rel=1e-12)
    assert not np.any(mask[w])
