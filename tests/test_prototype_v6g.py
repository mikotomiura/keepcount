import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6g as v6g  # noqa: E402


def series(gaps):
    return np.concatenate([[0.0], np.cumsum(np.asarray(gaps, dtype=float))])


HELD = [np.arange(0, 3), np.arange(3, 7), np.arange(7, 10)]
GAPS = [100.0, 200.0, 300.0, 5000.0, 110.0, 120.0, 130.0, 3000.0, 140.0, 150.0]


def test_resampled_draws_each_tile_from_its_own_gaps():
    g = np.diff(v6g.resampled(series(GAPS), HELD, np.random.default_rng(0)))
    for h in HELD:
        assert set(g[h].tolist()) <= set(np.asarray(GAPS)[h].tolist())
    gs = [np.diff(v6g.resampled(series(GAPS), HELD, np.random.default_rng(k))) for k in range(30)]
    assert any(not np.allclose(x, GAPS) for x in gs)


def test_resampled_keeps_the_boundary_gap_when_asked():
    for k in range(20):
        g = np.diff(v6g.resampled(series(GAPS), HELD, np.random.default_rng(k), keep_boundary=True))
        assert g[3] == 5000.0 and g[7] == 3000.0  # first gaps of tiles 1 and 2 stay
        assert set(g[4:7].tolist()) <= {110.0, 120.0, 130.0} and set(g[0:3].tolist()) <= {100.0, 200.0, 300.0}


def test_own_terms_redrawn_is_symmetric_and_operations_are_compared_per_draw():
    rng = np.random.default_rng(1)
    n = 400
    text = [rng.normal(size=(6, 2)) for _ in range(n)]
    base = [[t + rng.normal(0, 0.3, t.shape) for _ in range(v6g.B)] for t in text]
    redraw = [[t + rng.normal(0, 0.3, t.shape) for _ in range(v6g.B)] for t in text]
    ind, *_ = v6g.own_terms(text, base, redraw, "IMG_pq", "t")
    assert abs(ind.mean() - 0.5) < 0.03
    big = [t + 3.0 for t in text]
    ind2, mb, mc, zero, dropped = v6g.own_terms(text, base, big, "IMG_pq", "t")
    assert ind2.mean() == 1.0 and zero == 0 and dropped == []
    same = list(text)
    ind3, *_ = v6g.own_terms(text, base, same, "IMG_pq", "t")
    assert ind3.mean() == 0.0


def test_zero_base_ties():
    text = [np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0, 1.0], [0.0, 0.0]])]
    base = [[t.copy() for _ in range(v6g.B)] for t in text]  # resampled to itself
    ind, _, _, zero, _ = v6g.own_terms(text, base, [t.copy() for t in text], "IMG_pq", "t")
    assert zero == 2 and ind.mean() == 0.5


def test_summary_and_keys():
    ind = np.zeros((10, v6g.B))
    ind[:5] = 1.0
    ind[:, 0] = 1.0
    s = v6g.summary(ind, np.ones(10), np.ones(10) * 2, 0, [], [1, 2])
    assert s["share_vs_own"] == pytest.approx(0.5 + 0.5 / v6g.B) and s["share_by_single_draw_max"] == 1.0
    assert s["share_by_single_draw_min"] == 0.5
    assert v6g.key_of("lwp", "lwp", "IMG_df", "redrawn", False) == [20261031, 71, 2, 0, 8, 8]
    assert v6g.key_of("kupa", "essay", "S1", "inside/before", True) == [20261031, 71, 0, 1, 1, 21]
    assert v6g.KEY_BASE == {False: 9, True: 11} and v6g.KEY_REDRAW == {False: 10, True: 12} and v6g.B == 20


def fake(g_lo=0.7, red=0.5, red_ci=(0.44, 0.56), shuf=(0.7, 0.9)):
    cnt = {}
    for nm in ("kupa_essay", "lwp_lwp"):
        cnt[nm] = {}
        for row in v6g.ROWS:
            cnt[nm][row] = {}
            for st in ("IMG_pq", "IMG_df", "S1", "S2"):
                if row == "redrawn":
                    x = (red, list(red_ci))
                elif row == "gathered/before":
                    x = (0.8, [g_lo, 0.9])
                elif row in v6g.SHUFFLES:
                    x = (0.8, list(shuf))
                elif row == "inside/before":
                    x = (0.0, [0.0, 0.0])
                else:
                    x = (0.8, [0.7, 0.9])
                cnt[nm][row][st] = {"share_vs_own": x[0], "share_vs_own_ci": x[1]}
    return cnt


def test_rule():
    r = v6g.evaluate(fake())
    assert (r["positive_control"], r["symmetry_check"], r["tile_sequence"]) == \
        ("met", "passed", "shows the shuffles beyond its sampling variation")
    assert v6g.evaluate(fake(shuf=(0.3, 0.55)))["tile_sequence"].startswith("does not show")
    assert v6g.evaluate(fake(shuf=(0.55, 0.7)))["tile_sequence"] == "mixed"
    assert v6g.evaluate(fake(g_lo=0.60))["tile_sequence"] == "not evaluable"
    assert v6g.evaluate(fake(red=0.56))["symmetry_check"] == "failed"
    r = v6g.evaluate(fake(red_ci=(0.39, 0.6)))
    assert r["symmetry_check"] == "undetermined" and r["tile_sequence"] == "not evaluable"
    assert v6g.evaluate(fake(red_ci=(0.40, 0.60)))["symmetry_check"] == "passed"


def test_rule_one_cell_and_predictions():
    c = fake()
    c["lwp_lwp"]["shuffled/before"]["IMG_df"]["share_vs_own_ci"] = [0.60, 0.8]
    r = v6g.evaluate(c)
    assert r["shuffles_above"] == 7 and r["tile_sequence"] == "mixed"
    assert r["predictions"]["P1"] and r["predictions"]["P3"] and not r["predictions"]["P4"] and r["predictions"]["P5"]
    assert not r["predictions"]["P2"]  # x1.5 above everywhere in the fake
    c = fake()
    c["kupa_essay"]["rests_x1.5/before"]["IMG_pq"]["share_vs_own_ci"] = [0.5, 0.7]
    assert v6g.evaluate(c)["predictions"]["P2"]


def test_resampling_is_with_replacement():
    multisets = [tuple(sorted(np.diff(v6g.resampled(series(GAPS), HELD, np.random.default_rng(k)))[3:7]))
                 for k in range(30)]
    assert any(m != tuple(sorted(GAPS[3:7])) for m in multisets)  # a permutation would keep the multiset


def test_positive_control_needs_all_four_cells():
    c = fake()
    c["lwp_lwp"]["gathered/before"]["IMG_pq"]["share_vs_own_ci"] = [0.55, 0.9]
    r = v6g.evaluate(c)
    assert r["a_held"] == 3 and r["positive_control"] == "not met" and r["tile_sequence"] == "not evaluable"


def test_change_and_base_values():
    text = [np.array([[0.0, 0.0], [2.0, 2.0]]), np.array([[2.0, 0.0], [0.0, 2.0]])]
    sd = np.concatenate(text).std(axis=0)  # 1, 1
    base = [[t + [0.5, 0.0] for _ in range(v6g.B)] for t in text]
    change = [t + [0.0, 1.5] for t in text]
    ind, mb, mc, zero, _ = v6g.own_terms(text, base, change, "IMG_pq", "t")
    assert sd.tolist() == [1.0, 1.0]
    assert mb.tolist() == pytest.approx([0.5, 0.5]) and mc.tolist() == pytest.approx([1.5, 1.5])
    assert ind.mean() == 1.0
