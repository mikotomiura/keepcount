import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "works"))
import prototype_v5 as v5  # noqa: E402


def wave(n, kx, ky=0):
    x = np.arange(n) / n
    return 0.5 + 0.2 * np.sin(2 * np.pi * (ky * x[:, None] + kx * x[None, :]))


@pytest.mark.parametrize("n,kx,ky", [(120, 5, 0), (120, 0, 7), (80, 3, 4), (80, 12, 0)])
def test_fineness_of_a_plane_wave_is_its_cycles_per_tile(n, kx, ky):
    assert v5.fineness(wave(n, kx, ky)) == pytest.approx(np.hypot(kx, ky))


def test_fineness_ignores_the_phase_and_the_mean():
    n = 120
    x = np.arange(n) / n
    a = 0.3 + 0.1 * np.sin(2 * np.pi * 6 * x[None, :] + 0.0) * np.ones((n, 1))
    b = 0.7 + 0.1 * np.sin(2 * np.pi * 6 * x[None, :] + 2.1) * np.ones((n, 1))
    assert v5.fineness(a) == pytest.approx(v5.fineness(b))


def test_fineness_of_a_constant_tile_stops():
    with pytest.raises(SystemExit):
        v5.fineness(np.full((80, 80), 0.5))


def test_depth_is_the_population_standard_deviation():
    t = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert v5.depth(t) == pytest.approx(0.5)


def test_diff_pairs_tiles_up_to_the_smaller_number():
    assert v5.diff([1.0, 2.0, 3.0], [1.5, 2.0]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        v5.diff([], [1.0])


def test_counts_directions_and_lengths():
    # v[row][field][k][number]: row a = 0, row b = k + field / 100 (field 42 or 43), depth only; fineness the same
    fields, seeds = [42, 43], 3
    def vals(x, f, k):
        d = [0.0, 0.0] if x == "a" else [k + (f - 42) / 100] * 2
        return {"depth": d, "fineness": d}
    v = {x: {f: [vals(x, f, k) for k in range(seeds)] for f in fields} for x in ("a", "b")}
    c = v5.counts(v, ["a", "b"], fields, seeds)["depth"]
    assert list(c["between_rows"]) == ["b"]
    assert c["between_rows"]["b"] == pytest.approx([0.005, 1.005, 2.005])  # mean over both fields
    assert c["between_fields"]["b"] == pytest.approx([0.01] * 3)  # field 43 against 42 only
    assert c["between_fields"]["a"] == pytest.approx([0.0] * 3)
    assert c["between_seeds"]["b"] == pytest.approx([1.0, 1.0])  # seeds - 1 pairs
    assert len(c["between_seeds"]["a"]) == seeds - 1


def test_spearman_stops_on_a_constant_input():
    with pytest.raises(SystemExit):
        v5.spearman([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], "x")
    assert v5.spearman([1.0, 2.0, 3.0], [2.0, 5.0, 9.0], "x") == pytest.approx(1.0)


def test_grid_rows_are_q_and_columns_p():
    r = {"octaves": 4, "normalize": "ampsum", "s_max": 0.3}
    g = v5.grid_check(r, [42], {"u": 40})["u"]["42"]
    assert g["depth"][0][0] < g["depth"][-1][0]  # depth rises down the rows (q)
    assert g["fineness"][0][0] < g["fineness"][0][-1]  # fineness rises along a row (p)
