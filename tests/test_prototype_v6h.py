import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6h as v6h  # noqa: E402

REST = 2000.0
GAPS = np.array([100.0, 2500.0, 120.0, 130.0, 3000.0, 140.0, 150.0, 4000.0, 160.0, 170.0])
TIMES = np.concatenate([[0.0], np.cumsum(GAPS)])
HELD = [np.arange(0, 3), np.arange(3, 7), np.arange(7, 10)]


def g(t):
    return np.diff(t)


def test_rests_moved_keeps_both_orders_and_moves_positions():
    seen = set()
    for k in range(30):
        new = g(v6h.rests_moved(TIMES, REST, np.random.default_rng(k)))
        rest = new >= REST
        assert np.array_equal(new[rest], GAPS[GAPS >= REST]) and np.array_equal(new[~rest], GAPS[GAPS < REST])
        seen.add(tuple(np.flatnonzero(rest)))
    assert len(seen) > 1
    no_rest = np.concatenate([[0.0], np.cumsum(np.full(6, 100.0))])
    assert np.array_equal(v6h.rests_moved(no_rest, REST, np.random.default_rng(0)), no_rest)


def test_rests_swapped_and_short_shuffled_keep_what_they_say():
    for k in range(20):
        a = g(v6h.rests_swapped(TIMES, REST, np.random.default_rng(k)))
        assert np.array_equal(a < REST, GAPS < REST) and np.array_equal(a[a < REST], GAPS[GAPS < REST])
        assert sorted(a[a >= REST]) == sorted(GAPS[GAPS >= REST])
        b = g(v6h.short_shuffled(TIMES, REST, np.random.default_rng(k)))
        assert np.array_equal(b[GAPS >= REST], GAPS[GAPS >= REST]) and sorted(b) == sorted(GAPS)
    assert any(not np.array_equal(g(v6h.rests_swapped(TIMES, REST, np.random.default_rng(k))), GAPS) for k in range(20))
    assert any(not np.array_equal(g(v6h.short_shuffled(TIMES, REST, np.random.default_rng(k))), GAPS) for k in range(20))


def test_tiles_permuted_moves_blocks_whole():
    for k in range(10):
        t, ranges = v6h.tiles_permuted(TIMES, HELD, np.random.default_rng(k))
        new = g(t)
        blocks = sorted(tuple(new[a:b]) for a, b in ranges)
        assert blocks == sorted(tuple(GAPS[h]) for h in HELD)
        assert ranges[0][0] == 0 and ranges[-1][1] == GAPS.size
        assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    with pytest.raises(SystemExit):
        v6h.tiles_permuted(TIMES, [np.arange(0, 3), np.arange(4, 10)], np.random.default_rng(0))


def test_shuffle_all_is_a_permutation():
    new = g(v6h.shuffle_all(TIMES, np.random.default_rng(3)))
    assert sorted(new) == sorted(GAPS) and not np.array_equal(new, GAPS)


def ident(x):
    return np.asarray(x, dtype=float)


def test_own_of_ties_and_zero_base():
    text = np.array([[0.0], [1.0]])
    o, ind, c, mb, z0 = v6h.own_of(ident, text, [text.copy() for _ in range(v6h.B)], text + 1.0)
    assert o == 1.0 and z0 and mb == 0.0
    o, *_ = v6h.own_of(ident, text, [text.copy() for _ in range(v6h.B)], text.copy())
    assert o == 0.5


def fake_build(n=30, seed=0, h_far=0.0):
    """Persons whose H changes are smaller (h_far < 0) or equal to text changes."""
    rng = np.random.default_rng(seed)
    persons = []
    for _ in range(n):
        txt = {st: rng.normal(size=(6, len(v6h.NUMBERS[st]))) for st in v6h.STAGE}
        base = [{st: txt[st] + rng.normal(0, 0.3, txt[st].shape) for st in v6h.STAGE} for _ in range(v6h.B)]
        rows = {row: {st: txt[st] + rng.normal(0, 0.6, txt[st].shape) for st in v6h.STAGE} for row in v6h.ROWS}
        hs = []
        for _ in range(v6h.M):
            ht = {st: rng.normal(size=(6, len(v6h.NUMBERS[st]))) for st in v6h.STAGE}
            hb = [{st: ht[st] + rng.normal(0, 0.3, ht[st].shape) for st in v6h.STAGE} for _ in range(v6h.B)]
            hr = {row: {st: ht[st] + rng.normal(0, 0.6 + h_far, ht[st].shape) for st in v6h.STAGE}
                  for row in v6h.KEY_ON_H}
            hs.append({"text": ht, "base": hb, "rows": hr})
        persons.append({"text": txt, "base": base, "rows": rows, "H": hs})
    return {"persons": persons, "unchanged": dict.fromkeys(v6h.ROWS, 0), "unchanged_H": dict.fromkeys(v6h.ROWS, 0)}


def test_count_cell_pairs_the_bootstrap_and_skips_H_for_gathered():
    b = fake_build()
    c = v6h.count_cell(b, "shuffled", "IMG_pq", [1, 2, 3], "t")
    assert set(c) >= {"share_H", "mean_delta", "mean_delta_ci", "share_H_by_h_min", "median_change_H"}
    assert c["mean_delta"] == pytest.approx(c["share_text"] - c["share_H"])
    gg = v6h.count_cell(b, "gathered", "IMG_pq", [1, 2, 3], "t")
    assert "share_H" not in gg and "mean_delta" not in gg
    b2 = fake_build(h_far=-0.55)  # H changes about as small as its redraws: delta > 0
    c2 = v6h.count_cell(b2, "shuffled", "IMG_q", [1, 2, 4], "t")
    assert c2["mean_delta_ci"][0] > 0


def test_count_cell_interval_uses_the_cell_key():
    b = fake_build()
    a = v6h.count_cell(b, "rests_moved", "IMG_q", [5, 6, 7], "t")
    c = v6h.count_cell(b, "rests_moved", "IMG_q", [5, 6, 8], "t")
    assert a["share_text"] == c["share_text"] and a["share_text_ci"] != c["share_text_ci"]


def fake_cnt(d_lo=0.1, d_hi=0.3, q_lo=0.7, qd_lo=0.1, qd_hi=0.3, h_lo=0.6, short_lo=0.3, swap=0.5, moved=0.8):
    cnt = {}
    for nm in ("kupa_essay", "lwp_lwp"):
        cnt[nm] = {row: {} for row in v6h.ROWS}
        for row in v6h.ROWS:
            for st in v6h.STAGE:
                cnt[nm][row][st] = {"share_text": 0.8, "share_text_ci": [0.7, 0.9], "share_H": 0.7,
                                    "share_H_ci": [h_lo, 0.8], "mean_delta": 0.1, "mean_delta_ci": [d_lo, d_hi]}
        cnt[nm]["rests_moved"]["IMG_q"].update(share_text=moved, share_text_ci=[q_lo, 0.9], mean_delta_ci=[qd_lo, qd_hi])
        cnt[nm]["short_shuffled"]["IMG_q"].update(share_text_ci=[short_lo, 0.5])
        cnt[nm]["rests_swapped"]["IMG_q"].update(share_text=swap)
    return cnt


def test_rule():
    r = v6h.evaluate(fake_cnt())
    assert r["q1"].startswith("shuffling shows") and r["q2"].startswith("moving the rests")
    assert all(r["predictions"].values())
    assert v6h.evaluate(fake_cnt(d_lo=-0.1, d_hi=0.1))["q1"] == "a difference from exchangeable gaps is not confirmed"
    r = v6h.evaluate(fake_cnt(d_lo=-0.3, d_hi=-0.1))
    assert r["q1"] == "mixed" and len(r["q1_cells_below"]) == 4
    c = fake_cnt()
    c["lwp_lwp"]["shuffled"]["IMG_df"]["mean_delta_ci"] = [0.0, 0.2]
    assert v6h.evaluate(c)["q1"] == "mixed"
    assert v6h.evaluate(fake_cnt(q_lo=0.60))["q2"] == "mixed"
    assert v6h.evaluate(fake_cnt(qd_lo=-0.1, qd_hi=0.1))["q2"] == "not confirmed"
    p = v6h.evaluate(fake_cnt(h_lo=0.5, short_lo=0.61, swap=0.9))["predictions"]
    assert not p["P1"] and not p["P4"] and not p["P5"]


def test_keys_follow_the_conditions():
    assert v6h.key_of("lwp", "lwp", "IMG_q", "rests_swapped") == [20261031, 72, 2, 0, 10, 6]
    assert v6h.key_of("kupa", "essay", "IMG_pq", "shuffled") == [20261031, 72, 0, 1, 7, 1]
    assert v6h.KEY_TEXT == {"tiles_permuted": 13, "rests_moved": 14, "short_shuffled": 15, "rests_swapped": 22}
    assert v6h.KEY_ON_H == {"shuffled": 17, "tiles_permuted": 18, "rests_moved": 19, "short_shuffled": 20,
                            "rests_swapped": 23}
    assert (v6h.KEY_H, v6h.KEY_BASE_H, v6h.M, v6h.B) == (16, 21, 5, 20)


def test_main_cells_order():
    cell = {"share_text": 0.5, "share_text_ci": [0.4, 0.6], "share_H": 0.5, "share_H_ci": [0.4, 0.6],
            "mean_delta": 0.0, "mean_delta_ci": [-0.1, 0.1]}
    rec = {"count": {nm: {row: dict.fromkeys(v6h.STAGE, cell) for row in v6h.ROWS} for nm in ("kupa_essay", "lwp_lwp")}}
    keys = list(v6h.main_cells(rec))
    assert keys[0] == "kupa_essay | shuffled | IMG_pq" and keys[4] == "kupa_essay | rests_moved | IMG_q"
    assert keys[14] == "lwp_lwp | shuffled | IMG_pq" and len(keys) == 28
