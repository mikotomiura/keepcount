import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6 as v6  # noqa: E402
import prototype_v6c as v6c  # noqa: E402


def test_rank_one_equals_v6_top1_with_ties():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 4, (30, 30)).astype(float)  # many ties
    top1, _ = v6.per_person(d)
    assert np.allclose(v6c.rank_rates(d, (1,))[1], top1)


def test_rank_rates_count_closer_impostors_and_share_ties():
    # person 0: genuine 2, impostors 1 (closer), 2 (tied), 3 -> r = 2, t = 1
    d = np.array([[2.0, 1.0, 2.0, 3.0],
                  [9.0, 0.0, 9.0, 9.0],
                  [9.0, 9.0, 0.0, 9.0],
                  [9.0, 9.0, 9.0, 0.0]])
    rr = v6c.rank_rates(d, (1, 2, 3))
    assert rr[1][0] == 0.0
    assert rr[2][0] == 0.5
    assert rr[3][0] == 1.0
    assert all(rr[1][1:] == 1.0)


def test_rank_rate_reaches_one_at_the_number_of_persons():
    d = np.random.default_rng(1).random((12, 12))
    assert np.all(v6c.rank_rates(d, (12,))[12] == 1.0)


def test_eer_separated_and_reversed():
    n = 6
    sep = np.full((n, n), 5.0)
    np.fill_diagonal(sep, 1.0)
    assert v6c.eer(sep)["eer"] == 0.0
    rev = np.full((n, n), 1.0)
    np.fill_diagonal(rev, 5.0)
    assert v6c.eer(rev)["eer"] == 1.0


def test_eer_of_random_distances_is_near_one_half():
    d = np.random.default_rng(2).random((200, 200))
    assert abs(v6c.eer(d)["eer"] - 0.5) < 0.06


def test_population_subsets_and_full_set():
    rng = np.random.default_rng(3)
    g = {u: {st: rng.random((200, 4)) for st in v6.STAGE} for u in v6.UNIT}
    e = {u: {st: g[u][st] + 0.05 * rng.random((200, 4)) for st in v6.STAGE} for u in v6.UNIT}
    out = v6c.population(g, e, "kupa", 0)
    cell = out["sentence"]["S1"]
    assert set(cell) == {"25", "50", "100", "200"}
    assert "p2.5" not in cell["200"]["top1"] and "p2.5" in cell["25"]["top1"]
    assert cell["25"]["chance"] == {"top1": 1 / 25, "rank5": 5 / 25}
    top1, rauc = v6.per_person(v6.distances(*reversed(v6.standardise(g["sentence"]["S1"], e["sentence"]["S1"],
                                                                     "main")[:2])))
    assert cell["200"]["top1"]["mean"] == float(top1.mean())
    assert cell["200"]["rank_auc"]["mean"] == float(rauc.mean())
    again = v6c.population(g, e, "kupa", 0)
    assert again["window_10s"]["S2"]["50"] == out["window_10s"]["S2"]["50"]
