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


# ---------------------------------------------------------------- v2


def _timing():
    return ks.TextTiming(ks.Lognormal(240, 0.4), ks.Lognormal(735, 0.5), ks.Lognormal(2697, 0.5),
                         ks.Lognormal(5394, 0.5))


def test_full_windows_are_half_open_and_drop_the_tail():
    t = np.array([0.0, 100.0, 3000.0, 3100.0, 6000.0, 6100.0])
    w, dropped = ks.full_windows(t, 3000.0, 2000.0)
    assert [x.presses for x in w] == [2, 2]  # the press at 6000 is not in [3000, 6000)
    assert dropped == pytest.approx(100.0)
    assert w[0].rest_share == pytest.approx(2900.0 / 3000.0) and w[1].rest_share == pytest.approx(2900.0 / 3000.0)


def test_full_windows_check_inputs():
    for bad in ([0.0], [0.0, 0.0, 5.0], [1.0, 2.0], [0.0, np.nan]):
        with pytest.raises(ValueError):
            ks.full_windows(np.array(bad), 1000.0, 2000.0)
    with pytest.raises(ValueError):
        ks.full_windows(np.array([0.0, 5.0]), 0.0, 2000.0)


def test_rest_time_is_conserved_over_full_windows():
    rng = np.random.default_rng(3)
    g = ks.matched_regimes(open("works/texts/rule.en.txt").read(), _timing(), 2000.0, 30.0)
    t = ks.from_matched_regimes(600, g, 2000.0, rng)
    w, _ = ks.full_windows(t, 10000.0, 2000.0)
    gaps = np.diff(t)
    end = len(w) * 10000.0
    starts, ends = t[:-1][gaps >= 2000], t[1:][gaps >= 2000]
    total = np.sum(np.clip(np.minimum(ends, end) - starts, 0, None))
    assert sum(x.rest_share * 10000.0 for x in w) == pytest.approx(total)


def test_matched_regimes_match_the_text_in_distribution():
    text = open("works/texts/seasons.en.txt").read()
    timing = _timing()
    g = ks.matched_regimes(text, timing, 2000.0, 30.0)
    n = len(ks.positions(text))
    a = np.concatenate([np.diff(ks.from_text(text, timing, np.random.default_rng(i))) for i in range(200)])
    b = np.concatenate([np.diff(ks.from_matched_regimes(n, g, 2000.0, np.random.default_rng(1000 + i)))
                        for i in range(200)])
    assert np.mean(a >= 2000) == pytest.approx(np.mean(b >= 2000), abs=0.003)
    assert np.median(a[a < 2000]) == pytest.approx(np.median(b[b < 2000]), rel=0.03)
    assert np.median(a[a >= 2000]) == pytest.approx(np.median(b[b >= 2000]), rel=0.05)
    assert a.sum() / 200 == pytest.approx(b.sum() / 200, rel=0.02)


def test_mixture_draws_respect_the_bounds():
    m = ks.Mixture(((1.0, ks.Lognormal(1000, 0.5)),))
    rng = np.random.default_rng(4)
    assert all(m.draw(rng, below=800) < 800 for _ in range(200))
    assert all(m.draw(rng, at_least=2000) >= 2000 for _ in range(200))
