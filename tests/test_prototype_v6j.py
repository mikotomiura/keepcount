import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6j as v6j  # noqa: E402

REST = 2000.0
C4 = json.loads((ROOT / "works/conditions/prototype_v4.json").read_text())
M = C4["mapping"]


def ranges_of(sizes):
    out, at = [], 0
    for k in sizes:
        out.append((at, at + k))
        at += k
    return out


RNG = np.random.default_rng(1)
GAPS = np.where(RNG.random(72) < 0.2, RNG.uniform(2100, 6000, 72), RNG.uniform(80, 400, 72))
RANGES = ranges_of([8] * 9)
HELD = [np.arange(lo, hi) for lo, hi in RANGES]


def series():
    return v6j.v6.Series(np.concatenate([[0.0], np.cumsum(GAPS)]), sentence_ranges=RANGES)


def rest_tiles(g):
    return [int(v6j.is_rest(g[h], REST).sum()) for h in HELD]


def test_move_rests_moves_into_receiving_only():
    recv = np.array([0, 3, 5])
    new, moved, left = v6j.move_rests(GAPS, HELD, recv, None, REST, np.random.default_rng(0))
    assert sorted(new) == sorted(GAPS)
    before, after = rest_tiles(GAPS), rest_tiles(new)
    giving = [k for k in range(9) if k not in recv]
    assert all(after[k] == 0 for k in giving) or left > 0
    assert moved == sum(before[k] for k in giving) - left
    assert sum(after[k] for k in recv) == sum(before[k] for k in recv) + moved
    new2, moved2, _ = v6j.move_rests(GAPS, HELD, recv, 2, REST, np.random.default_rng(0))
    assert moved2 == 2 and sum(np.abs(new2 - GAPS) > 0) == 4


def test_weak_is_the_first_half_of_the_same_draw(monkeypatch):
    s = series()
    seen, real = [], np.random.default_rng

    def spy(k):
        seen.append(list(k))
        return real(k)

    monkeypatch.setattr(v6j.np.random, "default_rng", spy)
    out = v6j.controls(s, [0, 1, 7], REST)
    key = [v6j.SEED, 6, 32, 0, 1, 7, 2]
    assert key in seen and [v6j.SEED, 6, 33, 0, 1, 7, 2] in seen and [v6j.SEED, 6, 33, 0, 1, 7, 4] in seen
    assert [v6j.SEED, 6, 8, 0, 1, 7] in seen
    half = real(key).choice(9, 4, replace=False)
    m = sum(rest_tiles(GAPS)[k] for k in range(9) if k not in half)
    full, weak = out["random_half"], out["random_half_weak"]
    assert weak[1] + weak[2] == math.ceil(m / 2) and full[1] + full[2] == m
    changed_w = np.flatnonzero(weak[0] != GAPS)
    assert set(changed_w) <= set(np.flatnonzero(full[0] != GAPS))
    blk = out["middle_block"][0]
    after = rest_tiles(blk)
    assert all(after[k] == 0 for k in range(9) if not 3 <= k < 6) or out["middle_block"][2] > 0


def test_person_a_keys_and_nulls(monkeypatch):
    s = series()
    seen, real = [], np.random.default_rng

    def spy(k):
        seen.append(list(k))
        return real(k)

    monkeypatch.setattr(v6j, "K", 7)
    monkeypatch.setattr(v6j.np.random, "default_rng", spy)
    pa = v6j.person_a(s, [2, 0, 5], M, REST)
    assert [v6j.SEED, 6, 16, 2, 0, 5, 6] in seen
    for name, ci in v6j.CONTROL.items():
        assert [v6j.SEED, 6, 31, 2, 0, 5, ci, 6] in seen
        t, nv = pa[name]["H_raw"]
        assert nv.size == 7 and pa[name]["H_detrended"][1].size == 7
    # H is shared by the controls (text's H); C is each control's own
    assert np.array_equal(pa["gathered"]["H_raw"][1], pa["random_half"]["H_raw"][1])
    assert not np.array_equal(pa["gathered"]["C_raw"][1], pa["random_half"]["C_raw"][1])
    g0 = real([v6j.SEED, 6, 16, 2, 0, 5, 0]).permutation(np.diff(s.times))
    assert pa["gathered"]["H_raw"][1][0] == v6j.q_stats(g0, RANGES, M, REST)[0]
    assert pa["gathered"]["H_detrended"][0] == v6j.q_stats(
        np.diff(v6j.v6f.gathered(s.times, HELD, REST, real([v6j.SEED, 6, 8, 2, 0, 5]))[0]), RANGES, M, REST)[1]


def test_non_finite_stops(monkeypatch):
    monkeypatch.setattr(v6j, "K", 3)
    monkeypatch.setattr(v6j, "q_stats", lambda *a: (float("nan"), 0.0))
    with pytest.raises(SystemExit):
        v6j.person_a(series(), [0, 1, 2], M, REST)


def cnt_of(lo):
    """lo[name][null] -> lower end of mean r."""
    return {name: {null: {"mean_r_ci": [lo[name][null], 0.99]} for null in v6j.NULL} for name in v6j.CONTROL}


def test_evaluate_a_rule():
    ok = {name: dict.fromkeys(v6j.NULL, 0.6) for name in v6j.CONTROL}
    e = v6j.evaluate_a({"kupa_essay": cnt_of(ok), "lwp_lwp": cnt_of(ok)})
    assert all(v.startswith("sensitive") for c in e["reading_of_rest"].values() for v in c.values())
    bad = {name: dict(d) for name, d in ok.items()}
    bad["random_half"]["C_raw"] = 0.45
    bad["gathered"]["H_raw"] = 0.4  # gathered does not gate the reading
    e = v6j.evaluate_a({"kupa_essay": cnt_of(bad), "lwp_lwp": cnt_of(ok)})
    assert e["reading_of_rest"]["kupa_essay"]["C_raw"] == "not evaluable"
    assert e["reading_of_rest"]["kupa_essay"]["H_raw"].startswith("sensitive")
    assert not e["predictions"]["P1"]
    blk = {name: dict(d) for name, d in ok.items()}
    blk["middle_block"]["H_detrended"] = 0.4
    assert v6j.evaluate_a({"kupa_essay": cnt_of(ok), "lwp_lwp": cnt_of(blk)})["predictions"]["P3"]
    assert not v6j.evaluate_a({"kupa_essay": cnt_of(ok), "lwp_lwp": cnt_of(ok)})["predictions"]["P3"]


def test_scrambled_keeps_pairs():
    class T:
        p = np.arange(6.0)
        q = np.arange(6.0) * 10
        n = 6
    x = v6j.scrambled_seq(T(), np.random.default_rng(2))
    assert np.array_equal(x[:, 1], x[:, 0] * 10) and sorted(x[:, 0]) == list(range(6))
    assert not np.array_equal(x[:, 0], T.p)


def test_dtw_registration_uses_registration_stats():
    rng = np.random.default_rng(4)
    reg = [rng.normal(size=(k, 2)) for k in (5, 6, 7)]
    ev = [rng.normal(loc=3, size=(k, 2)) for k in (6, 5, 8)]
    d, drop = v6j.dtw_registration(reg, ev)
    allr = np.concatenate(reg)
    mu, sd = allr.mean(axis=0), allr.std(axis=0)
    assert drop == [] and d[1, 2] == pytest.approx(v6j.v6i.dtw((ev[1] - mu) / sd, [(reg[2] - mu) / sd])[0], rel=1e-12)
    main, _ = v6j.v6i.dtw_matrix(reg, ev)
    assert not np.allclose(main, d)
    reg_c = [np.column_stack([r[:, 0], np.full(r.shape[0], 0.2)]) for r in reg]
    d2, drop2 = v6j.dtw_registration(reg_c, ev)
    assert drop2 == [1] and np.isfinite(d2).all()


def test_vec_distance_registration():
    rng = np.random.default_rng(5)
    g, e = rng.normal(size=(20, 3)), rng.normal(loc=1, size=(20, 3))
    d, _ = v6j.vec_distance(g, e, "registration")
    gs, es, _ = v6j.v6.standardise(g, e, "registration_scale")
    assert np.array_equal(d, v6j.v6.distances(es, gs))
    assert v6j.vec_distance(np.ones((5, 1)), np.ones((5, 1)), "main")[0] is None


def test_paired_uses_resamples_of_the_later_named():
    rng = np.random.default_rng(6)
    rauc = {(sc, st): rng.random(10) for sc in ("main", "registration") for st in ("S1", "IMG5_pq", "DTW_pq",
                                                                                     "DTW_pq_scrambled")}
    idx = {k: np.random.default_rng(hash(k) % 1000).integers(0, 10, (50, 10)) for k in rauc}
    out = v6j.paired(rauc, idx, "sentence")
    k = "main DTW_pq - main DTW_pq_scrambled"
    diff = rauc[("main", "DTW_pq")] - rauc[("main", "DTW_pq_scrambled")]
    assert out[k]["ci"] == v6j.v6.ci(diff, idx[("main", "DTW_pq_scrambled")])
    k = "main IMG5_pq - registration IMG5_pq"
    diff = rauc[("main", "IMG5_pq")] - rauc[("registration", "IMG5_pq")]
    assert out[k]["ci"] == v6j.v6.ci(diff, idx[("registration", "IMG5_pq")]) and len(out) == 6


def test_predict_b():
    def cell(dtw, scr, fall):
        return {"main": {"DTW_pq": {"rank_auc": dtw}, "DTW_pq_scrambled": {"rank_auc_mean_over_draws": scr}},
                "paired": {f"main {st} - registration {st}": {"rank_auc_difference": fall} for st in ("IMG5_pq",
                                                                                                       "DTW_pq")}}
    cnt = {"kupa_copy_to_essay": {"sentence": cell(0.79, 0.77, 0.07)},
           "kupa_essay_to_copy": {"sentence": cell(0.79, 0.70, 0.06)}}
    assert v6j.predict_b(cnt) == {"P4": True, "P5": True}
    cnt["kupa_essay_to_copy"]["sentence"] = cell(0.79, 0.70, 0.12)
    assert v6j.predict_b(cnt)["P5"] is False
    cnt["kupa_copy_to_essay"]["sentence"] = cell(0.79, 0.75, 0.07)
    assert v6j.predict_b(cnt)["P4"] is False


def fake_pp(n, shift=0.0):
    out = []
    for i in range(n):
        d = {}
        for name in v6j.CONTROL:
            for null in v6j.NULL:
                d[(name, null)] = {"r": ((i * 7) % 10) / 10 + shift, "R": 1.0 + i / 20, "degenerate": False}
        out.append(d)
    return out


def test_check_gathered_stops_on_a_difference():
    pp = fake_pp(12)
    n = 12
    idx = np.random.default_rng([v6j.SEED, 80, 0, 1, 0]).integers(0, n, (v6j.v6.N_BOOT, n))
    rs = np.array([p[("gathered", "H_raw")]["r"] for p in pp])
    ref = v6j.v6i.r_cell(rs, [p[("gathered", "H_raw")]["R"] for p in pp], np.zeros(n, dtype=bool), idx)
    rec = {"a": {"count": {"kupa_essay": {"gathered": {"q": ref}}}}}
    v6j.check_gathered(pp, rec, "kupa", "essay")
    bad = json.loads(json.dumps(rec))
    bad["a"]["count"]["kupa_essay"]["gathered"]["q"]["mean_r_ci"][0] += 1e-9
    with pytest.raises(SystemExit):
        v6j.check_gathered(pp, bad, "kupa", "essay")


def test_count_a_keys(monkeypatch):
    seen, real = [], np.random.default_rng

    def spy(k):
        seen.append(list(k))
        return real(k)

    monkeypatch.setattr(v6j.np.random, "default_rng", spy)
    tallies = [{name: (2, 0) for name in v6j.CONTROL} for _ in range(8)]
    out = v6j.count_a(fake_pp(8), tallies, "lwp", "lwp")
    assert [k for k in seen if k[:2] == [v6j.SEED, 81]] == [[v6j.SEED, 81, 2, 0, i] for i in range(4)]
    assert out["random_half"]["C_raw"]["mean_r_ci"] == out["gathered"]["C_raw"]["mean_r_ci"]


def test_weak_moves_the_first_rests_with_the_same_partners():
    G = np.diff(series().times)  # the gaps the controls start from (rebuilt times, so not GAPS bit for bit)
    found_odd = False
    for place in range(30):
        key = [0, 1, place]
        out = v6j.controls(series(), key, REST)
        half = np.random.default_rng([v6j.SEED, 6, 32, *key, 2]).choice(9, 4, replace=False)
        giving_rests = [int(j) for k, h in enumerate(HELD) if k not in half for j in h if G[j] >= REST - v6j.ks.TOL_MS]
        m = len(giving_rests)
        found_odd |= m % 2 == 1
        k = math.ceil(m / 2)
        full, weak = out["random_half"][0], out["random_half_weak"][0]
        moved_w = [j for j in giving_rests if weak[j] != G[j]]
        assert moved_w == giving_rests[:len(moved_w)] and len(moved_w) == out["random_half_weak"][1]
        assert out["random_half_weak"][1] + out["random_half_weak"][2] == k
        for j in giving_rests[:out["random_half_weak"][1]]:
            assert weak[j] == full[j]  # the same partner as in the full move
    assert found_odd


def test_c_is_the_controls_own_and_h_detrended_column(monkeypatch):
    s = series()
    monkeypatch.setattr(v6j, "K", 3)
    key = [2, 0, 5]
    pa = v6j.person_a(s, key, M, REST)
    gc = v6j.controls(s, key, REST)["random_half"][0]
    c1 = v6j.v6i.circular(gc, np.random.default_rng([v6j.SEED, 6, 31, *key, 2, 1]))
    assert pa["random_half"]["C_raw"][1][1] == v6j.q_stats(c1, RANGES, M, REST)[0]
    assert pa["random_half"]["C_detrended"][1][1] == v6j.q_stats(c1, RANGES, M, REST)[1]
    g0 = np.random.default_rng([v6j.SEED, 6, 16, *key, 0]).permutation(np.diff(s.times))
    assert pa["random_half"]["H_detrended"][1][0] == v6j.q_stats(g0, RANGES, M, REST)[1]


def test_non_finite_h_stops(monkeypatch):
    real = v6j.q_stats
    calls = {"n": 0}

    def bad(*a):
        calls["n"] += 1
        x = real(*a)
        return (float("nan"), x[1]) if calls["n"] == 1 else x

    monkeypatch.setattr(v6j, "K", 3)
    monkeypatch.setattr(v6j, "q_stats", bad)
    with pytest.raises(SystemExit):
        v6j.person_a(series(), [0, 1, 2], M, REST)
