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
