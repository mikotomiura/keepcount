import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6i as v6i  # noqa: E402
import prototype_v6h as v6h  # noqa: E402

REST = 2000.0
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M, R = C4["mapping"], C4["relief"]


def series(gaps, starts):
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    return v6i.v6.Series(times, sentence_ranges=starts)


def ranges_of(sizes):
    out, at = [], 0
    for k in sizes:
        out.append((at, at + k))
        at += k
    return out


RNG = np.random.default_rng(0)
GAPS = np.where(RNG.random(60) < 0.15, RNG.uniform(2100, 6000, 60), RNG.uniform(80, 400, 60))
RANGES = ranges_of([6, 7, 5, 8, 6, 7, 5, 9, 7])  # 9 tiles, 60 gaps


# ---------------------------------------------------------------- part (a)


def test_dispersion_known_values():
    d = v6i.dispersion(np.array([0.0, 1.0, 2.0, 3.0]))
    assert d["raw"] == pytest.approx(np.std([0, 1, 2, 3]))
    assert d["detrended"] == pytest.approx(0.0, abs=1e-12)
    assert d["mad"] == pytest.approx(1.0)
    c = v6i.dispersion(np.full(5, 0.7))
    assert c == {"raw": 0.0, "detrended": 0.0, "mad": 0.0}
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    assert v6i.dispersion(x)["detrended"] > 0


def test_mid_rank_ties_and_ends():
    null = np.array([1.0, 2.0, 2.0, 3.0])
    assert v6i.mid_rank(2.0, null) == pytest.approx((1 + 0.5 * 2) / 4)
    assert v6i.mid_rank(0.0, null) == 0.0 and v6i.mid_rank(9.0, null) == 1.0
    assert v6i.mid_rank(1.0, np.full(10, 1.0)) == 0.5


def test_circular_is_a_nonzero_rotation():
    g = np.arange(1.0, 11.0)
    seen = set()
    for k in range(40):
        c = v6i.circular(g, np.random.default_rng(k))
        off = int(np.flatnonzero(c == 1.0)[0])
        assert off != 0 and np.array_equal(np.roll(g, off), c)
        seen.add(off)
    assert len(seen) > 3
    with pytest.raises(SystemExit):
        v6i.circular(np.array([5.0]), np.random.default_rng(0))


def test_pq_without_relief_equals_tiles():
    s = series(GAPS, RANGES)
    t = v6i.v6.Tiles(s, "sentence", "before", M, R, REST)
    p, q = v6i.pq_of_gaps(np.diff(s.times), s.ranges("before"), M, REST)
    assert np.array_equal(p, t.p) and np.array_equal(q, t.q)


def test_h0_is_v6h_h0_and_keys():
    s = series(GAPS, RANGES)
    key = [0, 1, 7]
    seen = []

    real = np.random.default_rng

    def spy(k):
        seen.append(list(k))
        return real(k)

    v6i.np.random.default_rng = spy
    try:
        pa = v6i.person_a(s, key, M, REST)
    finally:
        v6i.np.random.default_rng = real
    h0 = v6h.shuffle_all(s.times, real([v6i.SEED, 6, v6h.KEY_H, *key, 0]))
    p0, q0 = v6i.pq_of_gaps(h0, s.ranges("before"), M, REST)
    assert pa["H"]["q"]["raw"][0] == v6i.dispersion(q0)["raw"]
    assert [v6i.SEED, 6, 16, *key, 999] in seen and [v6i.SEED, 6, 30, *key, 999] in seen
    assert [v6i.SEED, 6, 8, *key] in seen
    assert sum(1 for k in seen if k[:3] == [v6i.SEED, 6, 16]) == v6i.K
    assert pa["H"]["p"]["raw"].size == v6i.K and pa["C"]["q"]["mad"].size == v6i.K


def test_null_holds_gaps_per_tile_and_degenerate_person():
    flat = series(np.full(40, 150.0) + np.arange(40), ranges_of([5] * 8))
    v6i.K, k0 = 50, v6i.K
    try:
        pp = v6i.per_person_a(v6i.person_a(flat, [0, 1, 2], M, REST))
    finally:
        v6i.K = k0
    x = pp[("q", "raw", "H")]
    assert x["degenerate"] and x["r"] == 0.5 and x["R"] is None and x["T"] == 0.0
    assert not pp[("p", "raw", "H")]["degenerate"]


def test_eligible_a():
    assert v6i.eligible_a(ranges_of([3] * 7)) == "fewer_than_8"
    assert v6i.eligible_a(ranges_of([3] * 8)) is None
    assert v6i.eligible_a([(0, 2), (2, 2), (2, 4), (4, 4), (4, 6), (6, 6), (6, 8), (8, 8)]) == "part_without_gap"


def test_r_cell_and_tails():
    rs = np.array([0.99, 0.5, 0.01, 0.975, 0.025, 0.6])
    Rs = [3.0, None, 0.8, 1.0, 1.1, 0.9]
    deg = np.array([False, True, False, False, False, False])
    idx = np.random.default_rng(1).integers(0, rs.size, (200, rs.size))
    c = v6i.r_cell(rs, Rs, deg, idx)
    assert c["mean_r"] == pytest.approx(rs.mean())
    assert c["share_r_at_or_above_0.975"] == pytest.approx(2 / 6) and c["share_r_at_or_below_0.025"] == pytest.approx(2 / 6)
    assert c["persons_R_undefined"] == 1 and c["median_R"] == pytest.approx(1.0)
    assert c["non_degenerate"]["persons"] == 5 and c["non_degenerate"]["mean_r"] == pytest.approx(rs[~deg].mean())
    assert c["mean_r_ci"][0] <= c["mean_r"] <= c["mean_r_ci"][1]


def fake(q_ci, ctl_lo, p_ci=(0.6, 0.7), d_ci=(0.1, 0.2), med_r=1.0):
    cell = {"q": {"mean_r_ci": list(q_ci), "median_R": med_r}, "p": {"mean_r_ci": list(p_ci)},
            "r_p_minus_r_q": {"ci": list(d_ci)}}
    return {"H": {"raw": cell}, "gathered": {"q": {"mean_r_ci": [ctl_lo, 0.9]}}}


def test_evaluate_a_rule_and_gate():
    e = v6i.evaluate_a({"kupa_essay": fake((0.55, 0.6), 0.7), "lwp_lwp": fake((0.51, 0.6), 0.7)})
    assert e["A1"].startswith("the dispersion of q") and e["predictions"]["P9"]
    e = v6i.evaluate_a({"kupa_essay": fake((0.45, 0.6), 0.7), "lwp_lwp": fake((0.5, 0.6), 0.7)})
    assert e["A1"] == "not confirmed"
    e = v6i.evaluate_a({"kupa_essay": fake((0.55, 0.6), 0.5), "lwp_lwp": fake((0.55, 0.6), 0.7)})
    assert e["A1"] == "mixed" and e["A1_cells"]["kupa_essay"] == "not evaluable" and not e["predictions"]["P9"]
    e = v6i.evaluate_a({"kupa_essay": fake((0.3, 0.45), 0.7, d_ci=(-0.2, -0.1)), "lwp_lwp": fake((0.4, 0.6), 0.7,
                                                                                                    d_ci=(-0.1, 0.1))})
    assert e["A1_cells"]["kupa_essay"] == "below 0.5" and e["A1"] == "mixed"
    assert e["A3"] == {"kupa_essay": "q above p", "lwp_lwp": "no difference confirmed"}
    e = v6i.evaluate_a({"kupa_essay": fake((0.45, 0.6), 0.7), "lwp_lwp": fake((0.55, 0.6), 0.7)})
    assert e["predictions"]["P1"]


# ---------------------------------------------------------------- part (b)


class T:
    def __init__(self, p, q):
        self.p, self.q = np.asarray(p, float), np.asarray(q, float)
        self.depth, self.fineness = self.p * 2, self.q * 3
        self.n = self.p.size


def test_groups_and_binned():
    assert [g.size for g in v6i.groups(7, 5)] == [2, 2, 1, 1, 1]
    with pytest.raises(SystemExit):
        v6i.groups(4, 5)
    b = v6i.binned(np.arange(7.0), np.arange(7.0) * 10, 5)
    assert b.tolist() == [0.5, 5.0, 2.5, 25.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0]
    sk = np.array([0.0, 0.0, 9.0] * 5)
    assert v6i.binned(sk, sk, 5).tolist() == [3.0] * 10  # means, not medians (0.0)


def test_degenerate_only_when_every_null_ties():
    nv = np.array([0.5, 0.5, 0.7])
    pa = {"text": {x: {st: 0.5 for st in v6i.STAT} for x in ("p", "q")},
          "gathered": {x: {st: 0.5 for st in v6i.STAT} for x in ("p", "q")},
          "H": {x: {st: nv for st in v6i.STAT} for x in ("p", "q")},
          "C": {x: {st: np.full(3, 0.5) for st in v6i.STAT} for x in ("p", "q")}}
    pp = v6i.per_person_a(pa)
    assert not pp[("q", "raw", "H")]["degenerate"] and pp[("q", "raw", "C")]["degenerate"]
    assert pp[("q", "raw", "H")]["r"] == pytest.approx(1 / 3) and pp[("q", "raw", "H")]["R"] == 1.0


def test_new_vectors():
    t = T(np.linspace(0, 1, 9), np.linspace(1, 0, 9) ** 2)
    v = v6i.new_vectors(t, np.random.default_rng(3))
    assert v["IMG5_pq"].size == 10 and v["IMG3_pq"].size == 6 and v["NT"].tolist() == [9.0]
    lvl = np.tile([t.p.mean(), t.q.mean()], 5)
    assert np.allclose(v["IMG5_centred"], v["IMG5_pq"] - lvl)
    assert np.allclose(v["IMG5_df"], v6i.binned(t.depth, t.fineness, 5))
    assert np.array_equal(v["IMG5_pq_NT"][:10], v["IMG5_pq"]) and v["IMG5_pq_NT"][10] == 9.0
    assert v["SEQ"].size == 4
    s = v["IMG5_pq_scrambled"]
    assert not np.allclose(s, v["IMG5_pq"])
    with pytest.raises(SystemExit):
        v6i.new_vectors(T([0.1] * 4, [0.2] * 4), np.random.default_rng(0))


def dtw_ref(a, b):
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = np.linalg.norm(a[i - 1] - b[j - 1])
            d[i, j] = c + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return d[n, m] / (n + m)


def test_dtw_against_reference_and_padding():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(6, 2))
    gs = [rng.normal(size=(k, 2)) for k in (3, 9, 6, 1)]
    got = v6i.dtw(a, gs)
    for g, x in zip(gs, got):
        assert x == pytest.approx(dtw_ref(a, g), rel=1e-12)
    assert v6i.dtw(a, [a])[0] == 0.0


def test_full_d_matches_v6c_full():
    rng = np.random.default_rng(2)
    g, e = rng.normal(size=(30, 4)), rng.normal(size=(30, 4))
    key = [1, 2, 3]
    ref = v6i.v6c.full(g, e, key)
    d, drop = v6i.stage_vectors(g, e)
    mine, rauc, idx = v6i.full_d(d, key)
    assert drop == [] and mine["rank_n"] == ref["rank_n"] and mine["eer"] == ref["eer"]
    assert mine["top1"] == ref["top1"] and mine["rank_auc"] == ref["rank_auc"]
    assert mine["rank_auc_ci"] == v6i.v6.ci(rauc, idx)


def test_constant_stage():
    g = np.full((10, 1), 7.0)
    d, drop = v6i.stage_vectors(g, g + 1)
    assert d is None and drop == [0]


def test_predict_b_constant_nt():
    cell = {"rank_auc": 0.7, "rank_auc_ci": [0.65, 0.75], "rank_n": {"1": {"rate": 0.01}}}
    st = {k: dict(cell) for k in ("S1", "S2", "IMG5_pq", "IMG5_df", "IMG5_pq_scrambled", "DTW_pq")}
    st["DTW_pq"]["rank_auc"] = 0.6
    st["SEQ"] = {"rank_auc": 0.55, "rank_auc_ci": [0.5, 0.59]}
    st["IMG5_centred"] = {"rank_auc": 0.55, "rank_auc_ci": [0.5, 0.59]}
    st["NT"] = {"constant": True}
    win = {"NT": {"rank_auc_ci": [0.61, 0.7]}}
    cnt = {"kupa_copy_to_essay": {"sentence": st, "window_10s": win},
           "kupa_essay_to_copy": {"sentence": {"NT": {"rank_auc_ci": [0.5, 0.58]}}, "window_10s": win}}
    p = v6i.predict_b(cnt)
    assert p == {"P5": True, "P6": True, "P7": True, "P8": True, "P10": True, "P11": True}
    cnt["kupa_essay_to_copy"]["window_10s"] = {"NT": {"constant": True}}
    assert not v6i.predict_b(cnt)["P7"]


def test_no_long_list():
    v6i.no_long_list({"a": [1, 2, 3, 4, 5], "dropped_features": list(range(9))})
    with pytest.raises(SystemExit):
        v6i.no_long_list({"a": list(range(6))})
