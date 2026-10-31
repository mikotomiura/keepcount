import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import keystroke as ks  # noqa: E402
import prototype_v6 as v6  # noqa: E402

C4 = json.loads((ROOT / "works" / "conditions" / "prototype_v4.json").read_text())
M, R = C4["mapping"], C4["relief"]
TEXT = (ROOT / "works" / "texts" / "seasons.en.txt").read_text()


def series_from_text(seed=0):
    timing = ks.TextTiming(ks.Lognormal(240.0, 0.4), ks.Lognormal(735.0, 0.5), ks.Lognormal(2697.0, 0.5),
                           ks.Lognormal(5394.0, 0.5))
    return ks.from_text(TEXT, timing, np.random.default_rng(seed))


@pytest.mark.parametrize("counting", ["before", "after", "none"])
def test_segments_of_gaps_on_counted_ranges_equal_segments_counted(counting):
    t = series_from_text()
    starts = ks.sentence_starts(TEXT)
    a = ks.segments_counted(t, starts, 2000.0, counting)
    b = ks.segments_of_gaps(np.diff(t), ks.counted_ranges(starts, t.size, counting), 2000.0)
    assert [(x.gaps, x.speed_ms, x.rest_share) for x in a] == [(x.gaps, x.speed_ms, x.rest_share) for x in b]


def test_rests_inside_at_equals_rests_inside_with_the_text_labels():
    t = series_from_text(1)
    labels = ks.positions(TEXT)
    ws = [i for i, lab in enumerate(labels) if lab == "word_start"]
    a = ks.rests_inside(t, TEXT, np.random.default_rng(7))
    b, left = ks.rests_inside_at(t, ks.sentence_starts(TEXT), ws, np.random.default_rng(7))
    assert left == 0
    assert np.array_equal(a, b)


def test_rests_inside_at_leaves_a_sentence_without_a_word_start():
    t = np.arange(8, dtype=float) * 100
    out, left = ks.rests_inside_at(t, [0, 3, 6], [4], np.random.default_rng(0))
    assert left == 1  # the sentence [6, 8) has no word start
    assert np.array_equal(out, t)  # every gap is 100: the swap in [3, 6) changes nothing


def test_shuffled_in_two_keeps_each_side():
    t = np.concatenate([[0.0], np.cumsum(np.arange(1, 11, dtype=float))])
    out = ks.shuffled_in_two(t, 4, np.random.default_rng(3))
    g = np.diff(out)
    assert sorted(np.round(g[:4], 9)) == [1, 2, 3, 4]
    assert sorted(np.round(g[4:], 9)) == [5, 6, 7, 8, 9, 10]
    with pytest.raises(ValueError):
        ks.shuffled_in_two(t, 0, np.random.default_rng(0))


def test_kupa_kinds_starts_and_word_starts():
    keys = ["H", "i", ".", " ", "Y", "o", " ", "u", "Enter", "a", "!", "?", " ", "1", "Shift", "ArrowLeft", "Backspace"]
    kinds = [v6.kupa_kind(k) for k in keys]
    assert kinds == ["alnum", "alnum", "end", "space", "alnum", "alnum", "space", "alnum", "space", "alnum", "end",
                     "end", "space", "alnum", None, None, "other"]
    counted = [k for k in kinds if k is not None]
    assert v6.kupa_starts(counted) == [0, 4, 13]
    assert v6.kupa_word_starts(counted) == [4, 7, 9, 13]


def test_clean_sorts_and_drops_equal_times():
    t, k, dropped = v6.clean([10.0, 5.0, 10.0, 20.0], ["a", "b", "c", "d"])
    assert t.tolist() == [0.0, 5.0, 15.0]
    assert k == ["b", "a", "d"]
    assert dropped == 1


def test_per_person_top1_and_rank_auc_with_ties():
    d = np.array([[1.0, 2.0, 3.0],
                  [1.0, 1.0, 5.0],
                  [0.5, 4.0, 2.0]])
    top1, rauc = v6.per_person(d)
    assert top1.tolist() == [1.0, 0.5, 0.0]
    assert rauc.tolist() == [1.0, 0.75, 0.5]


def test_pooled_auc_matches_brute_force():
    rng = np.random.default_rng(0)
    d = rng.random((6, 6)).round(1)
    gen = np.diag(d)
    imp = d[~np.eye(6, dtype=bool)]
    brute = np.mean([(i > g) + 0.5 * (i == g) for g in gen for i in imp])
    assert v6.pooled(d)[0] == pytest.approx(brute)


def test_standardise_drops_a_constant_feature_and_scales():
    g = np.array([[1.0, 5.0, 0.0], [2.0, 5.0, 1.0], [3.0, 5.0, 2.0]])
    e = g * 2 + 1
    gs, es, drop = v6.standardise(g, e, "main")
    assert drop == [1]
    assert np.allclose(gs, es)  # a common shift and scale is removed per set
    gr, er, _ = v6.standardise(g, e, "registration_scale")
    assert not np.allclose(gr, er)


def test_reading_has_two_axes():
    assert v6.reading(0.53, 0.58) == {"detection": "detectable", "size": "below 0.60"}
    assert v6.reading(0.45, 0.58) == {"detection": "not detected", "size": "below 0.60"}
    assert v6.reading(0.61, 0.70) == {"detection": "detectable", "size": "above 0.60"}
    assert v6.reading(0.55, 0.65) == {"detection": "detectable", "size": "undecided"}


def aalto_zip(lines):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Keystrokes/files/7_keystrokes.txt", "\n".join(lines) + "\n")
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_read_aalto_reads_fields_from_both_ends_and_needs_15_sentences():
    head = "PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tUSER_INPUT\tKEYSTROKE_ID\tPRESS_TIME\tRELEASE_TIME\tLETTER\tKEYCODE"
    lines = [head]
    for s in range(15):
        sent = "a\tb" if s == 0 else "ab"  # a tab inside the sentence of the first section
        for j, (letter, t) in enumerate([("SHIFT", 0), ("a", 10), ("b", 30), ("b", 30), ("BKSP", 55)]):
            lines.append(f"7\t{100 + s}\t{sent}\tab\t{j}\t{1000 * s + t}\t{1000 * s + t + 5}\t{letter}\t0")
    lines.insert(20, head)  # a repeated header
    tally = {"missing_files": 0, "short_lines": 0, "lines_read_from_ends": 0, "not_counted": 0,
             "dropped_equal_times": 0}
    out = v6.read_aalto(aalto_zip(lines), 7, tally)
    assert len(out) == 15
    assert out[0].tolist() == [0.0, 20.0, 45.0]
    assert tally["lines_read_from_ends"] == 5 and tally["not_counted"] == 15 and tally["dropped_equal_times"] == 15
    assert v6.read_aalto(aalto_zip(lines[:-5]), 7, dict(tally)) is None  # 14 sentences


def test_aalto_series_drops_the_gap_between_sentences():
    s = v6.aalto_series([np.array([0.0, 100.0, 300.0]), np.array([0.0, 50.0])])
    assert np.diff(s.times).tolist() == [100.0, 200.0, 50.0]
    assert s.ranges("none") == [(0, 2), (2, 3)]


def kupa_like(seed=0):
    t = series_from_text(seed)
    labels = ks.positions(TEXT)
    return v6.Series(t, ks.sentence_starts(TEXT), [i for i, lab in enumerate(labels) if lab == "word_start"])


@pytest.mark.parametrize("unit", ["sentence", "window_10s"])
def test_rows_pass_the_s0_checks_and_shuffle_in_halves_stays_in_its_half(unit):
    s = kupa_like()
    tt = v6.Tiles(s, unit, "before", M, R, 2000.0)
    rows, _ = v6.make_rows("kupa", s, unit, tt, [0, 1, 5], M, R, 2000.0)
    v6.check_s0(rows, unit)
    h = tt.n // 2
    k = v6.split_index(tt, s, unit, h)
    g0 = np.diff(s.times)
    row = "shuffled_in_halves" if unit == "window_10s" else "shuffled_in_halves/before"
    g1 = rows[row].gaps
    assert np.allclose(np.sort(g0[:k]), np.sort(g1[:k]))
    assert not np.allclose(g0[:k], g1[:k])  # it did shuffle
    if unit == "window_10s":
        assert np.all(s.times[:k] < tt.window_starts[h]) and s.times[k] >= tt.window_starts[h]


def test_check_s0_stops_on_a_changed_half():
    s = kupa_like()
    tt = v6.Tiles(s, "sentence", "before", M, R, 2000.0)
    bad = v6.Tiles(s.with_times(ks.shuffled(s.times, np.random.default_rng(1))), "sentence", "before", M, R, 2000.0)
    with pytest.raises(SystemExit):
        v6.check_s0({"text/before": tt, "shuffled_in_halves/before": bad}, "sentence")


def test_count1_of_an_unchanged_row_is_zero_change():
    rng = np.random.default_rng(0)
    halves = []
    for _ in range(10):
        a = {st: rng.random(4) for st in v6.STAGE}
        b = {st: rng.random(4) for st in v6.STAGE}
        halves.append((a, b))
    out = v6.count1({"text": halves, "shuffled": halves, "shuffled_in_halves": halves}, "aalto", "session", "sentence")
    assert out["persons"] == 10
    assert out["shuffled"]["S1"]["share_vs_own"] == 0.0
    assert out["shuffled"]["S1"]["median_change"] == 0.0
    assert out["shuffled"]["S1"]["reading"] == {"detection": "not detected", "size": "below 0.60"}
