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


def test_labels_with_a_leading_comma():
    s, kinds = series_of(list(",ab"))
    assert v6b.labels(s, kinds).tolist() == ["subsentence", "within"]


def test_oddeven_parts_hold_their_own_gaps_and_lag1_pairs_stay_inside():
    keys = list(("The cat sat, on a mat. " * 12).strip())
    s, kinds = series_of(keys)
    tt = v6.Tiles(s, "sentence", "before", M, R, 2000.0)
    a, b = v6b.parts(tt.n, "oddeven")
    ha = set(np.concatenate([tt.held[i] for i in a]).tolist())
    hb = set(np.concatenate([tt.held[i] for i in b]).tolist())
    assert not ha & hb and ha | hb == set(range(s.times.size - 1))
    # a pair (j, j + 1) with j in A and j + 1 in B is not used by A's lag-1: change B's gaps only
    gaps = np.diff(s.times).copy()
    rng = np.random.default_rng(5)
    gaps[sorted(ha)] = rng.uniform(50, 5000, len(ha))  # A varied, so its lag-1 is not a correlation of rounding noise
    gaps[sorted(hb)] = rng.uniform(50, 5000, len(hb))
    lab = v6b.labels(s, kinds)
    t2 = v6.Tiles(s.with_times(np.concatenate([[0.0], np.cumsum(gaps)])), "sentence", "before", M, R, 2000.0)
    gaps[sorted(hb)] = rng.uniform(50, 5000, len(hb))
    t3 = v6.Tiles(s.with_times(np.concatenate([[0.0], np.cumsum(gaps)])), "sentence", "before", M, R, 2000.0)
    assert np.allclose(v6b.loc(t2, a, lab), v6b.loc(t3, a, lab))


def test_count_leaves_loc_undefined_for_a_row_without_it():
    rng = np.random.default_rng(0)
    def vec(loc_ok=True):
        return {"S0": rng.random(7), "S1": rng.random(4), "S2": rng.random(4), "LOC": rng.random(7) if loc_ok else None}
    text = [(vec(), vec()) for _ in range(5)]
    none = [(vec(False), vec(False)) for _ in range(5)]
    shuf = [(vec(), vec()) for _ in range(5)]
    out = v6b.count({"text": text, "text/none": none, "shuffled": shuf}, ["text", "text/none", "shuffled"], "essay",
                    "sentence", "oddeven")
    assert out["text/none"]["LOC"] == {"defined": False, "persons": 0}
    assert out["shuffled"]["LOC"]["persons"] == 5 and "same_persons_as_LOC" in out["shuffled"]


def test_count_stops_on_a_nan_and_on_partial_loc_in_text_none():
    rng = np.random.default_rng(1)
    def vec(loc=True):
        return {"S0": rng.random(7), "S1": rng.random(4), "S2": rng.random(4), "LOC": rng.random(7) if loc else None}
    text = [(vec(), vec()) for _ in range(5)]
    bad = [(vec(), vec()) for _ in range(5)]
    bad[2][0]["S1"][1] = np.nan
    with pytest.raises(SystemExit):
        v6b.count({"text": text, "shuffled": bad}, ["text", "shuffled"], "essay", "sentence", "oddeven")
    none = [(vec(False), vec(False)) for _ in range(4)] + [(vec(), vec())]
    with pytest.raises(SystemExit):
        v6b.count({"text": text, "text/none": none}, ["text", "text/none"], "essay", "sentence", "oddeven")


def test_read_kupa_b_matches_v6_after_mapping_back(tmp_path):
    rows = ["id,time,type,key,key_code,alt_key,ctrl_key,meta_key,shift_key,is_repeat,range_start,range_end,text"]
    keys = ["H", "i", ",", " ", "y", ";", ":", "Backspace", ".", "Enter", "Shift", "a"]
    for i, k in enumerate(keys):
        rows.append(f"P1,{100 * i},down,{k if k != ',' else '\",\"'},,,,,,,0,0,")
    rows.append("P2,5,click,,,,,,,,0,0,")
    f = tmp_path / "t.csv"
    f.write_text("\n".join(rows) + "\n")
    b, seen_b = v6b.read_kupa_b(f)
    a, seen_a = v6.read_kupa(f)
    assert seen_a == seen_b == {"P1", "P2"}
    assert v6b.v6_kinds(b["P1"]["kinds"]) == a["P1"]["kinds"]
    assert b["P1"]["kinds"].count("sub") == 3
