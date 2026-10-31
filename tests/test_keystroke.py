import numpy as np
import pytest

import keystroke as ks


def test_positions_labels_every_press():
    text = "Ab cd. Ef\n\nGh"
    labels = ks.positions(text)
    # A b _ c d . _ E f <Enter> G h
    assert labels == ["start", "within_word", "within_word", "word_start", "within_word", "within_word", "within_word",
                      "sentence_end", "within_word", "within_word", "paragraph_end", "within_word"]


def test_positions_joins_lines_of_a_paragraph():
    assert ks.positions("ab\ncd") == ks.positions("ab cd")


def test_from_text_uses_the_timing_of_each_label():
    one = ks.Lognormal(100.0, 0.0)
    timing = ks.TextTiming(one, ks.Lognormal(200.0, 0.0), ks.Lognormal(3000.0, 0.0), ks.Lognormal(5000.0, 0.0))
    t = ks.from_text("ab. c\n\nd", timing, np.random.default_rng(0))
    assert np.diff(t).tolist() == [100.0, 100.0, 100.0, 3000.0, 100.0, 5000.0]


def test_shuffled_keeps_the_gaps():
    t = np.array([0.0, 100.0, 2600.0, 2700.0, 2950.0])
    s = ks.shuffled(t, np.random.default_rng(1))
    assert s[0] == 0.0 and sorted(np.diff(s)) == sorted(np.diff(t))


def test_regimes_rests_are_at_least_the_threshold():
    g = ks.Regimes(ks.Lognormal(100, 0.1), ks.Lognormal(300, 0.1), 5, 5, ks.Lognormal(500, 0.1), 0.2)
    t = ks.from_regimes(500, g, 2000.0, np.random.default_rng(2))
    gaps = np.diff(t)
    assert t.size == 500 and np.all(gaps > 0)
    assert np.all((gaps < 1000) | (gaps >= 2000 - ks.TOL_MS))
    assert all(w.rest_share > 0 for w in ks.windows(t, 30000.0, 2000.0)[:-1])
    assert 0.1 < np.mean(gaps >= 2000 - ks.TOL_MS) < 0.3


def test_windows_speed_and_rest_split_across_the_edge():
    # presses every 100 ms to 1000 ms, a 4000 ms rest across the edge at 3000 ms, then every 100 ms
    t = np.concatenate([np.arange(0, 1001, 100.0), 5000.0 + np.arange(0, 1001, 100.0)])
    w = ks.windows(t, 3000.0, 2000.0)
    assert len(w) == 2
    assert w[0].speed_ms == 100.0 and w[1].speed_ms == 100.0
    assert w[0].rest_share == pytest.approx(2000.0 / 3000.0)  # 1000..3000 of the 1000..5000 rest
    assert w[1].rest_share == pytest.approx(2000.0 / 3000.0)  # 3000..5000 of a window cut at 6000
    assert w[0].presses + w[1].presses == t.size


def test_window_without_short_gap_has_no_speed():
    w = ks.windows(np.array([0.0, 2500.0, 5000.0]), 3000.0, 2000.0)
    assert w[0].speed_ms is None and w[0].rest_share == pytest.approx(1.0)


def test_windows_reject_bad_times():
    with pytest.raises(ValueError):
        ks.windows(np.array([1.0, 2.0]), 1000.0, 2000.0)
    with pytest.raises(ValueError):
        ks.windows(np.array([0.0, 5.0, 3.0]), 1000.0, 2000.0)
