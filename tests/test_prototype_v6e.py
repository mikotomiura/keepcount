import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
import prototype_v6e as v6e  # noqa: E402

HEAD = "id,event_id,down_time,up_time,action_time,activity,down_event,up_event,text_change,cursor_position,word_count\n"


def write(tmp_path, rows):
    p = tmp_path / "logs.csv"
    p.write_text(HEAD + "".join(f"{i},{e},{t},{t},0,{a},{k},{k},x,0,0\n" for i, e, t, a, k in rows))
    return p


def test_key_mapping():
    assert [v6e.lwp_key(k) for k in ["q", "Space", "Enter", "Backspace", "Delete", ".", ",", "Shift", "ArrowLeft",
                                     "Leftclick", "Unidentified"]] == \
        ["q", " ", "Enter", "Backspace", "Delete", ".", ",", None, None, None, None]


def test_read_counts_kinds_and_tallies(tmp_path):
    rows = [("a", 1, 100, "Input", "q"), ("a", 2, 200, "Input", "Space"), ("a", 3, 150, "Nonproduction", "Shift"),
            ("a", 4, 300, "Input", "."), ("a", 5, 300, "Input", "q"), ("a", 6, 400, "Paste", "v"),
            ("b", 1, 10, "Input", "q"), ("b", 2, 20, "Remove/Cut", "Backspace")]
    p = write(tmp_path, rows)
    out, ids, tally = v6e.read_lwp(p, None)
    assert ids == ["a", "b"]
    assert out["a"]["kinds"] == ["alnum", "space", "end", "alnum"]  # the press at 300 after '.' is dropped
    assert list(out["a"]["times"]) == [0, 100, 200, 300]
    assert tally["decreasing_down_time"] == 1 and tally["not_counted"] == 1 and tally["dropped_equal_times"] == 1
    out2, _, tally2 = v6e.read_lwp(p, v6e.EDIT_ACTIVITIES)
    assert out2["a"]["kinds"] == ["alnum", "space", "end"]  # Shift and the pasted 'v' are left out
    assert out2["b"]["kinds"] == ["alnum", "other"]
    assert tally2["other_activity"] == 2


def test_read_stops_on_disordered_rows(tmp_path):
    with pytest.raises(SystemExit):
        v6e.read_lwp(write(tmp_path, [("a", 1, 1, "Input", "q"), ("b", 1, 1, "Input", "q"),
                                      ("a", 2, 2, "Input", "q")]), None)
    with pytest.raises(SystemExit):
        v6e.read_lwp(write(tmp_path, [("a", 2, 1, "Input", "q"), ("a", 1, 2, "Input", "q")]), None)


def fake_counts(img_hi, loc_lo, shuf_img_hi=0.4):
    """Counts with every image upper end img_hi, every LOC-like lower end loc_lo."""
    def cell(lo, hi):
        return {"share_vs_own_ci": [lo, hi]}
    cnt = {}
    rows = {"sentence": ["text/after", "text/none", "inside/before", "shuffled/before", "shuffled_in_halves/before"],
            "window_10s": ["inside", "shuffled", "shuffled_in_halves"]}
    for u, rs in rows.items():
        cnt[u] = {}
        for how in ("oddeven", "halves"):
            cnt[u][how] = {}
            for row in rs:
                hi = shuf_img_hi if "shuffled" in row else img_hi
                d = {st: cell(0.1, hi) for st in ("S0", "S1", "S2", "SEQ")}
                d.update({st: cell(loc_lo, 0.99) for st in ("LOC", "R_pos", "R_lag")})
                d["same_persons_as_LOC"] = {st: cell(0.1, hi) for st in ("S1", "S2", "SEQ")}
                cnt[u][how][row] = d
    return cnt


def test_rule():
    assert v6e.evaluate(fake_counts(0.55, 0.7))["finding_1"] == "confirmed"
    assert v6e.evaluate(fake_counts(0.55, 0.7))["stronger_form"] == "confirmed"
    assert v6e.evaluate(fake_counts(0.55, 0.7, shuf_img_hi=0.55))["stronger_form"] == "not confirmed"
    assert v6e.evaluate(fake_counts(0.61, 0.7))["finding_1"] == "not confirmed"
    c = fake_counts(0.55, 0.7)
    c["sentence"]["oddeven"]["shuffled/before"]["LOC"]["share_vs_own_ci"] = [0.55, 0.9]
    c["window_10s"]["halves"]["shuffled"]["LOC"]["share_vs_own_ci"] = [0.55, 0.9]
    assert v6e.evaluate(c)["finding_1"] == "partly confirmed"
    c["window_10s"]["oddeven"]["shuffled"]["LOC"]["share_vs_own_ci"] = [0.55, 0.9]
    assert v6e.evaluate(c)["finding_1"] == "not confirmed"
