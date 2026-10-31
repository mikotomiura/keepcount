import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6b as v6b  # noqa: E402
import prototype_v6d as v6d  # noqa: E402


def test_split_takes_six_position_numbers_and_the_lag1():
    v = np.arange(7, dtype=float)
    s = v6d.split(v)
    assert np.array_equal(s["R_pos"], v[:6]) and np.array_equal(s["R_lag"], v[6:])
    assert v6d.split(None) == {"LOC": None, "R_pos": None, "R_lag": None}
    with pytest.raises(SystemExit):
        v6d.split(np.arange(6, dtype=float))


def halves(rng, n=30, shift=0.0):
    return [(v6d.split(rng.random(7) + shift), v6d.split(rng.random(7) + shift)) for _ in range(n)]


def test_count_checks_loc_against_v6b_and_counts_the_parts():
    rng = np.random.default_rng(0)
    rows = ["text/before", "shuffled/before"]
    hv = {rows[0]: halves(rng), rows[1]: halves(rng, shift=0.3)}
    key = [20261031, 66, 1, 0, v6b.STAGE["LOC"], 1, v6b.BASELINE["oddeven"]]
    ref = v6b.one(hv[rows[0]], hv[rows[1]], "LOC", key)
    out = v6d.count(hv, rows, "essay", "sentence", "oddeven", {rows[1]: {"LOC": ref}})
    cell = out[rows[1]]
    assert cell["LOC"]["share_vs_own"] == ref["share_vs_own"]
    assert set(cell) == {"LOC", "R_pos", "R_lag"}
    # the parts are counted on their own numbers
    lag = v6b.one(hv[rows[0]], hv[rows[1]], "R_lag", [20261031, 67, 1, 0, 1, 1, v6b.BASELINE["oddeven"]])
    assert cell["R_lag"]["share_vs_own"] == lag["share_vs_own"]
    bad = dict(ref, share_vs_own=ref["share_vs_own"] + 0.01)
    with pytest.raises(SystemExit):
        v6d.count(hv, rows, "essay", "sentence", "oddeven", {rows[1]: {"LOC": bad}})


def test_count_text_none_must_have_no_loc():
    rng = np.random.default_rng(1)
    rows = ["text/before", "text/none"]
    hv = {rows[0]: halves(rng, 5), rows[1]: [(v6d.split(None), v6d.split(None))] * 5}
    assert v6d.count(hv, rows, "essay", "sentence", "halves", {})["text/none"] == {"defined": False, "persons": 0}


def test_five_and_record_guard():
    f = v6d.five([1, 2, 3, 4, 5])
    assert (f["min"], f["median"], f["max"], f["persons"]) == (1.0, 3.0, 5.0, 5)
    with pytest.raises(SystemExit):
        v6d.five([])
    with pytest.raises(SystemExit):
        v6d.no_per_person({"a": list(range(25))})
