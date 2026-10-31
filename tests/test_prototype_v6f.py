import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6 as v6  # noqa: E402
import prototype_v6b as v6b  # noqa: E402
import prototype_v6f as v6f  # noqa: E402

C4 = json.loads((ROOT / "works" / "conditions" / "prototype_v4.json").read_text())
M, R = C4["mapping"], C4["relief"]


def series(gaps):
    return np.concatenate([[0.0], np.cumsum(np.asarray(gaps, dtype=float))])


def test_gathered_moves_rests_from_giving_to_receiving_tiles_and_keeps_the_gaps():
    # 8 tiles of 3 gaps; one rest in each giving tile (2, 3, 6, 7), none in receiving tiles (0, 1, 4, 5)
    gaps = [100.0] * 24
    for k in (2, 3, 6, 7):
        gaps[3 * k + 1] = 3000.0 + k
    held = [np.arange(3 * k, 3 * k + 3) for k in range(8)]
    t, moved, left = v6f.gathered(series(gaps), held, 2000.0, np.random.default_rng(0))
    g = np.diff(t)
    assert (moved, left) == (4, 0)
    assert sorted(g.tolist()) == pytest.approx(sorted(gaps))
    rest_tiles = {k for k in range(8) if (g[held[k]] >= 2000.0).any()}
    assert rest_tiles <= {0, 1, 4, 5} and len([j for j in range(24) if g[j] >= 2000]) == 4


def test_gathered_leaves_rests_when_receiving_tiles_have_no_short_gap():
    gaps = [3000.0] * 6 + [100.0] * 6  # tiles 0, 1 all rests; tiles 2, 3 short with one rest each
    gaps[7], gaps[10] = 2500.0, 2600.0
    held = [np.arange(3 * k, 3 * k + 3) for k in range(4)]
    t, moved, left = v6f.gathered(series(gaps), held, 2000.0, np.random.default_rng(1))
    assert (moved, left) == (0, 2)
    assert np.allclose(np.diff(t), gaps)


def test_scaled_multiplies_only_rests():
    t, n = v6f.scaled(series([100.0, 2000.0, 150.0, 5000.0]), 1.5, 2000.0)
    assert n == 2 and np.diff(t).tolist() == pytest.approx([100.0, 3000.0, 150.0, 7500.0])


def test_rms_and_pairing():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    y = np.zeros((2, 2))
    assert v6f.rms(x, y) == pytest.approx(np.sqrt(25 / 2))
    assert v6f.paired(x, np.zeros((5, 2))) == pytest.approx(np.sqrt(25 / 2))
    with pytest.raises(SystemExit):
        v6f.rms(x, np.zeros((3, 2)))


def test_quarters_partition_the_tiles_and_halves_join_to_v6b_parts():
    for n in (16, 17, 18, 19, 23):
        for how in ("oddeven", "halves"):
            q = v6f.quarters(n, how)
            assert sorted(np.concatenate(q).tolist()) == list(range(n)) and min(len(x) for x in q) >= 4
        A, B = v6b.parts(n, "halves")
        q = v6f.quarters(n, "halves")
        assert np.concatenate(q[:2]).tolist() == A.tolist() and np.concatenate(q[2:]).tolist() == B.tolist()
        A, B = v6b.parts(n, "oddeven")
        q = v6f.quarters(n, "oddeven")
        assert sorted(np.concatenate([q[0], q[2]]).tolist()) == A.tolist()


def test_img_terms_on_a_small_example():
    # person 0: tiles alternate (0,0),(1,1); the row moves every tile by (1,0)
    t0 = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
    t1 = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)
    first, sel = [t0, t1], [t0 + [1.0, 0.0], t1]
    own, other, base, change, between, dropped = v6f.img_terms(first, sel, "oddeven", "test")
    sd = np.concatenate(first).std(axis=0)
    z0 = t0 / sd
    assert base[0] == pytest.approx(v6f.rms(z0[[0, 2]], z0[[1, 3]]))
    assert change[0] == pytest.approx(1.0 / sd[0])  # every tile moved by 1 in the first number
    assert change[1] == 0.0 and base[1] == 0.0 and own[1] == 0.5 and dropped == []  # odd and even tiles equal: a tie
    assert own[0] == (1.0 if change[0] > base[0] else 0.0)
    with pytest.raises(SystemExit):
        v6f.img_terms(first, [t0[:3], t1], "oddeven", "test")


def test_img_terms_drop_a_constant_number():
    t = [np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]), np.array([[1.0, 1.0], [0.0, 1.0], [2.0, 1.0],
                                                                                 [0.5, 1.0]])]
    *_, dropped = v6f.img_terms(t, t, "halves", "test")
    assert dropped == [1]


def test_own_summary_matches_v6b_one():
    rng = np.random.default_rng(3)

    def vec():
        return {"S1": rng.normal(size=4)}

    first = [(vec(), vec()) for _ in range(30)]
    sel = [(vec(), vec()) for _ in range(30)]
    one = v6b.one(first, sel, "S1", [1, 2, 3])
    assert float(v6f.own_summary(first, sel, "S1").mean()) == one["share_vs_own"]


def test_quarter_terms_identical_quarters_tie():
    v = np.array([1.0, 2.0])
    w = np.array([2.0, 0.0])
    vq = [[v, w, v, w], [w, v, w, v]]  # base0 = d(Q0, Q1) > 0 = change0 under oddeven
    assert v6f.quarter_terms(vq, "oddeven", False, "t")[0].tolist() == [0.0, 0.0]
    vq = [[v, v, v, v], [w, v, w, v]]
    own, _ = v6f.quarter_terms(vq, "oddeven", False, "t")
    assert own[0] == 0.5  # all distances 0: a tie


# ---------------------------------------------------------------- rule


def fake(gather_lo=0.7, op_hi=0.55, x15_lo=0.5, x2_lo=0.7, share0=0.5):
    def cell(lo, hi):
        return {"share_vs_own_ci": [lo, hi]}
    cnt, quarters = {}, {}
    for name in ("kupa_essay", "lwp_lwp"):
        cnt[name] = {"sentence": {}}
        quarters[name] = {}
        for how in ("oddeven", "halves"):
            d = {v6f.GATHERED["sentence"]: {st: cell(gather_lo, 0.99) for st in ("S1", "S2", "SEQ", "IMG_pq", "IMG_df")},
                 "rests_x1.5/before": {st: cell(x15_lo, 0.9) for st in ("S1", "S2", "SEQ", "IMG_pq", "IMG_df")},
                 "rests_x2/before": {st: cell(x2_lo, 0.9) for st in ("S1", "S2", "SEQ", "IMG_pq", "IMG_df")},
                 "rests_x3/before": {st: cell(x2_lo, 0.9) for st in ("S1", "S2", "SEQ", "IMG_pq", "IMG_df")}}
            for row in v6f.OPERATIONS["sentence"]:
                d[row] = {st: cell(0.2, op_hi) for st in ("IMG_pq", "IMG_df")}
            cnt[name]["sentence"][how] = d
            quarters[name][how] = {st: {"share0": share0} for st in ("S1", "S2", "IMG_pq", "IMG_df")}
    return cnt, quarters


def test_rule_met_and_extended():
    r = v6f.evaluate(*fake())
    assert (r["positive_control"], r["finding_1_tile_sequence"]) == ("met", "extended")
    assert r["predictions"]["P1"] and r["predictions"]["P2"] and r["predictions"]["P4"]
    assert not r["predictions"]["P3"]  # every oddeven cell below 0.60


def test_rule_boundaries():
    cnt, q = fake()
    gat = v6f.GATHERED["sentence"]
    cells = [(nm, how, st) for nm in ("kupa_essay", "lwp_lwp") for how, sts in
             (("oddeven", ("S1", "S2", "IMG_pq", "IMG_df")), ("halves", ("S1", "S2", "SEQ", "IMG_pq", "IMG_df")))
             for st in sts]
    for nm, how, st in cells[:3]:  # 15 of 18 above
        cnt[nm]["sentence"][how][gat][st]["share_vs_own_ci"] = [0.55, 0.9]
    r = v6f.evaluate(cnt, q)
    assert r["positive_control"] == "partly met" and r["finding_1_tile_sequence"] == "partly extended"
    nm, how, st = cells[3]  # 14 of 18
    cnt[nm]["sentence"][how][gat][st]["share_vs_own_ci"] = [0.55, 0.9]
    assert v6f.evaluate(cnt, q)["finding_1_tile_sequence"] == "not evaluable"


def test_rule_b_boundary_and_not_sensitive_names():
    cnt, q = fake()
    b = [(nm, how, row, st) for nm in ("kupa_essay", "lwp_lwp") for how in ("oddeven", "halves")
         for row in v6f.OPERATIONS["sentence"] for st in ("IMG_pq", "IMG_df")]
    for nm, how, row, st in b[:4]:  # 20 of 24
        cnt[nm]["sentence"][how][row][st]["share_vs_own_ci"] = [0.5, 0.65]
    r = v6f.evaluate(cnt, q)
    assert (r["positive_control"], r["finding_1_tile_sequence"]) == ("met", "partly extended")
    nm, how, row, st = b[4]  # 19 of 24
    cnt[nm]["sentence"][how][row][st]["share_vs_own_ci"] = [0.5, 0.65]
    assert v6f.evaluate(cnt, q)["finding_1_tile_sequence"] == "not extended"
    cnt, q = fake()
    cnt["lwp_lwp"]["sentence"]["halves"][v6f.GATHERED["sentence"]]["IMG_df"]["share_vs_own_ci"] = [0.5, 0.9]
    r = v6f.evaluate(cnt, q)
    assert len(r["b_cells_not_sensitive"]) == 3 and r["positive_control"] == "partly met"


def test_rule_p2_and_p4():
    assert not v6f.evaluate(*fake(x15_lo=0.7))["predictions"]["P2"]  # x1.5 above everywhere
    assert not v6f.evaluate(*fake(x2_lo=0.55))["predictions"]["P2"]
    assert not v6f.evaluate(*fake(share0=0.61))["predictions"]["P4"]


def test_tiles_of_gathered_rows_keep_their_number_in_sentences():
    gaps = [150.0, 120.0, 2600.0, 130.0, 110.0, 140.0, 3100.0, 100.0] * 3
    t = series(gaps)
    starts = list(range(0, len(t), 3))[:-1] if (len(t) - 1) % 3 == 0 else list(range(0, len(t) - 1, 3))
    s = v6.Series(t, starts, starts)
    tt = v6.Tiles(s, "sentence", "before", M, R, 2000.0)
    tg, moved, left = v6f.gathered(t, tt.held, 2000.0, np.random.default_rng(5))
    tg_tiles = v6.Tiles(s.with_times(tg), "sentence", "before", M, R, 2000.0)
    assert tg_tiles.n == tt.n
    assert np.allclose(np.sort(np.diff(tg)), np.sort(np.diff(t)))


# ---------------------------------------------------------------- keys, rows and the reference (after the code review)


def test_keys_and_rows_as_written_in_the_conditions():
    assert v6f.key_of("lwp", "lwp", "sentence", "IMG_df", 9, "halves") == [20261031, 70, 2, 0, 0, 8, 9, 1]
    assert v6f.key_of("kupa", "essay", "window_10s", "SEQ", 4, "oddeven") == [20261031, 70, 0, 1, 1, 3, 4, 0]
    assert v6f.old_key("kupa", "copy", "sentence", "S1", 3, "halves") == [20261031, 66, 0, 0, 1, 3, 1]
    assert v6f.old_key("lwp", "lwp", "window_10s", "S2", 2, "oddeven") == [20261031, 69, 1, 2, 2, 0]
    assert v6f.rows_of("sentence") == ["text/before", "text/after", "text/none", "inside/before", "shuffled/before",
                                       "shuffled_in_halves/before", "gathered/before", "rests_x1.5/before",
                                       "rests_x2/before", "rests_x3/before"]
    assert v6f.rows_of("window_10s") == ["text", "inside", "shuffled", "shuffled_in_halves", "gathered"]
    assert v6f.QUARTER_ROW == 90 and v6f.PREFIX == 70


def test_quarter_terms_halves_img_by_hand():
    # quarters of one tile each: Q0=(0,0), Q1=(1,0), Q2=(0,3), Q3=(0,4); halves base0 = d(Q0,Q2), change0 = (d(Q1,Q0)+d(Q3,Q2))/2
    q = [np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([[0.0, 3.0]]), np.array([[0.0, 4.0]])]
    other = [np.array([[0.0, 0.0]]), np.array([[5.0, 0.0]]), np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]])]
    own, dropped = v6f.quarter_terms([q, other], "halves", True, "t")
    sd = np.concatenate([np.concatenate(q), np.concatenate(other)]).std(axis=0)
    base0 = 3.0 / sd[1]
    change0 = (1.0 / sd[0] + 1.0 / sd[1]) / 2
    assert own[0] == (1.0 if change0 > base0 else 0.0) and dropped == []


def test_quarter_pairs_differ_by_baseline():
    assert v6f.QUARTER_PAIRS == {"oddeven": (0, 1, 2, 3), "halves": (0, 2, 1, 3)}
    q = [np.array([0.0]), np.array([10.0]), np.array([1.0]), np.array([11.0])]
    # halves: base0 = |0-1| = 1, change0 = (|10-0| + |11-1|)/2 = 10 -> 1; oddeven: base0 = 10, change0 = (1 + 1)/2 -> 0
    two = [q, [np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([3.0])]]
    assert v6f.quarter_terms(two, "halves", False, "t")[0][0] == 1.0
    assert v6f.quarter_terms(two, "oddeven", False, "t")[0][0] == 0.0


def test_non_finite_tiles_stop():
    t = [np.array([[0.0, np.nan], [1.0, 1.0]]), np.array([[1.0, 2.0], [0.0, 1.0]])]
    with pytest.raises(SystemExit):
        v6f.img_terms(t, t, "halves", "t")
    good = [np.array([[0.0, 1.0], [1.0, 2.0]]), np.array([[1.0, 2.0], [0.0, 1.0]])]
    with pytest.raises(SystemExit):
        v6f.img_terms(good, t, "halves", "t")


def test_p3_true_and_exact_thresholds():
    cnt, q = fake()
    cnt["kupa_essay"]["sentence"]["oddeven"]["shuffled/before"]["IMG_pq"]["share_vs_own_ci"] = [0.5, 0.65]
    assert v6f.evaluate(cnt, q)["predictions"]["P3"]
    cnt["lwp_lwp"]["sentence"]["halves"]["inside/before"]["IMG_df"]["share_vs_own_ci"] = [0.5, 0.65]
    assert not v6f.evaluate(cnt, q)["predictions"]["P3"]
    cnt, q = fake()
    cnt["kupa_essay"]["sentence"]["oddeven"]["inside/before"]["IMG_pq"]["share_vs_own_ci"] = [0.5, 0.60]
    cnt["kupa_essay"]["sentence"]["oddeven"][v6f.GATHERED["sentence"]]["S1"]["share_vs_own_ci"] = [0.60, 0.9]
    r = v6f.evaluate(cnt, q)
    assert (r["a_held"], r["b_held"]) == (17, 23)


def fake_b(n=20, tiles=16, rng=None):
    """A fake build() result for the sentence unit with random tiles (for reference_quarters)."""
    rng = rng or np.random.default_rng(0)

    class T:
        def __init__(self, n_):
            self.n = n_
            self.p, self.q = rng.random(n_), rng.random(n_)

        def vectors(self, idx):
            return {"S0": rng.random(7), "S1": np.array([np.median(self.p[idx]), np.median(self.q[idx]), 0.1, 0.2]),
                    "S2": np.array([np.median(self.p[idx]), 0.3, np.median(self.q[idx]), 0.4])}

    rows = v6f.rows_of("sentence")
    tt = [T(tiles + (i % 3)) for i in range(n)]
    img = {st: {row: [] for row in rows} for st in v6f.IMG}
    hv = {how: {row: [] for row in rows} for how in ("oddeven", "halves")}
    for t in tt:
        for row in rows:
            x = np.column_stack([t.p, t.q]) + (0 if row.startswith("text") else rng.normal(0, 0.3, (t.n, 2)))
            for st in v6f.IMG:
                img[st][row].append(x)
            for how in hv:
                A, B = v6b.parts(t.n, how)
                hv[how][row].append(tuple({"S1": np.r_[np.median(x[i, 0]), np.median(x[i, 1]), 0.1, 0.2],
                                           "S2": np.r_[np.median(x[i, 0]), 0.3, np.median(x[i, 1]), 0.4]}
                                          for i in (A, B)))
    return {"hv": hv, "img": img, "text_tiles": tt}


def test_reference_quarters_paired_difference_is_the_mean_of_per_person_differences():
    b = fake_b()
    out = v6f.reference_quarters(b, "kupa", "essay", "sentence")
    assert out["persons"] == 20
    for how in ("oddeven", "halves"):
        for st in ("S1", "IMG_pq"):
            x = out[how][st]
            for row, y in x["paired"].items():
                assert y["share_R_minus_share0"] == pytest.approx(y["share_R"] - x["share0"])
                assert y["ci"][0] <= y["share_R_minus_share0"] + 1e-12 and y["ci"][1] >= y["share_R_minus_share0"] - 1e-12
            assert "text/after" not in x["paired"] and "gathered/before" in x["paired"]


def test_reproduce_stops_on_a_changed_or_missing_record_entry(monkeypatch):
    kept = v6.ROWS["kupa"]["sentence"]

    def fake_count(hv, rows, task, unit, how):
        out = {row: {st: {"persons": 10, "share_vs_own": 0.25, "share_vs_own_ci": [0.1, 0.4], "reading": {},
                          "median_base": 1.0, "median_change": 0.5, "median_change_over_base": 0.5}
                     for st in v6f.expected_stages("kupa", how, row)} for row in rows[1:]}
        out["persons"] = 10
        return out

    monkeypatch.setattr(v6b, "count", fake_count)
    ref = {how: fake_count(None, kept, None, None, how) for how in ("oddeven", "halves")}
    b = {"hv": {how: {row: [] for row in kept} for how in ("oddeven", "halves")}}
    n, out = v6f.reproduce(b, "kupa", "essay", "sentence", ref)
    assert n == sum(len(v6f.expected_stages("kupa", h, r)) for h in ("oddeven", "halves") for r in kept[1:])
    assert out["halves"]["shuffled/before"]["SEQ"]["share_vs_own"] == 0.25
    ref["oddeven"]["inside/before"]["S1"] = dict(ref["oddeven"]["inside/before"]["S1"], share_vs_own=0.3)
    with pytest.raises(SystemExit):
        v6f.reproduce(b, "kupa", "essay", "sentence", ref)
    ref = {how: fake_count(None, kept, None, None, how) for how in ("oddeven", "halves")}
    del ref["halves"]["shuffled/before"]["SEQ"]
    with pytest.raises(SystemExit):
        v6f.reproduce(b, "kupa", "essay", "sentence", ref)
