import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6 as v6  # noqa: E402
import prototype_v6b as v6b  # noqa: E402

C4 = json.loads((ROOT / "works" / "conditions" / "prototype_v4.json").read_text())
M, R = C4["mapping"], C4["relief"]


def test_kind_b_marks_comma_like_keys_and_keeps_v6_otherwise():
    assert [v6b.kind_b(k) for k in [",", ";", ":", "-", "a", ".", " ", "Shift"]] == \
        ["sub", "sub", "sub", "other", "alnum", "end", "space", None]
    assert v6b.v6_kinds(["sub", "alnum"]) == ["other", "alnum"]


def series_of(keys, gap=100.0):
    kinds = [v6b.kind_b(k) for k in keys]
    k6 = v6b.v6_kinds(kinds)
    t = np.arange(len(keys), dtype=float) * gap
    return v6.Series(t, v6.kupa_starts(k6), v6.kupa_word_starts(k6)), kinds


def test_labels_follow_the_press_after_the_gap():
    keys = list("Ab cd, ef. Gh")
    s, kinds = series_of(keys)
    lab = v6b.labels(s, kinds)
    # gaps before: b c d , _ e f . _ G h
    assert lab.tolist() == ["within", "other", "word", "within", "other", "other", "subsentence", "within", "other",
                            "other", "sentence", "within"]


def test_lag1_is_zero_for_short_or_constant_input():
    assert v6b.lag1(np.array([1.0, 2.0])) == 0.0
    assert v6b.lag1(np.array([1.0, 1.0, 1.0, 1.0])) == 0.0
    assert v6b.lag1(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)


def test_parts():
    a, b = v6b.parts(7, "oddeven")
    assert a.tolist() == [0, 2, 4, 6] and b.tolist() == [1, 3, 5]
    a, b = v6b.parts(7, "halves")
    assert a.tolist() == [0, 1, 2] and b.tolist() == [3, 4, 5, 6]


def test_loc_values_and_missing_label():
    keys = list("Ab cd, ef. Gh ij, kl. Mn")
    s, kinds = series_of(keys)
    gaps = np.full(len(keys) - 1, 100.0)
    lab = v6b.labels(s, kinds)
    gaps[lab == "sentence"] = 3000.0
    t = np.concatenate([[0.0], np.cumsum(gaps)])
    s2 = s.with_times(t)
    tt = v6.Tiles(s2, "sentence", "before", M, R, 2000.0)
    out = v6b.loc(tt, np.arange(tt.n), lab)
    assert out[0] == pytest.approx(np.log(3000.0)) and out[3] == pytest.approx(np.log(100.0))
    assert out[4] == pytest.approx(np.log(30.0))
    assert out[5] == pytest.approx(1.0)  # all rest time sits at sentence starts
    assert v6b.loc(tt, np.array([0]), lab) is None  # the first sentence has no sentence-start gap


def test_shuffle_moves_loc_but_not_s0():
    rng = np.random.default_rng(0)
    keys = list(("The cat sat, on a mat. " * 12).strip())
    s, kinds = series_of(keys)
    lab = v6b.labels(s, kinds)
    gaps = np.where(lab == "sentence", 2500.0, np.where(lab == "within", 120.0, 300.0)) * np.exp(
        0.1 * rng.standard_normal(lab.size))
    s2 = s.with_times(np.concatenate([[0.0], np.cumsum(gaps)]))
    tt = v6.Tiles(s2, "sentence", "before", M, R, 2000.0)
    rows, _ = v6.make_rows("kupa", s2, "sentence", tt, [0, 1, 3], M, R, 2000.0)
    sh = rows["shuffled/before"]
    idx = np.arange(tt.n)
    assert np.allclose(sh.vectors(idx)["S0"], tt.vectors(idx)["S0"])
    a, b = v6b.loc(tt, idx, lab), v6b.loc(sh, idx, lab)
    assert a[4] > 2.5 and abs(b[4]) < a[4] / 2  # sentence minus within collapses after a shuffle
