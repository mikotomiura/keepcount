"""Prototype v4: the first version of the atlas of what the transform cannot tell apart (claim E of
.steering/20261031-keystroke-claim; pairs and counts fixed in decisions 1-3 of .steering/20261031-atlas-v4).

Usage:  uv run python works/prototype_v4.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v4.json (committed before this script is first run)
Input:  works/texts/*.en.txt
Output: results/prototype_v4/record.json and, per text and unit, results/prototype_v4/{text}_{unit}_{scale}_f{seed}.png
        (written to a temporary directory beside it and renamed when complete)

Steps: src/keystroke (from_text, shuffled, from_matched_regimes as v2 and v3; rests_inside and reordered_by_runs from
the series from the text) -> segments_counted at the text's sentence starts (counting before / after / none) and
full_windows of 10 s and 30 s -> speed and rest -> p, q (external scale; min-max per series as reference) -> f, s,
gamma with the coefficients of atact.mapping -> H = s T_gamma(F) on the fields of four seeds -> grey
H / (2 s_max) + 1/2. Counts: spreads, the correlation of sentence length with q, and the differences of grey between
rows and between fields.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
import keystroke as ks  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.mapping import F0, F1, G0, G1, S0, S1, WP, WQ  # noqa: E402
from atact.transform import t_gamma  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v4.json"
OUT_DIR = ROOT / "results" / "prototype_v4"
CODE = ["works/prototype_v4.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
SERIES = ("from_text", "shuffled", "regimes", "inside", "reordered")


def lognormal(d: dict) -> ks.Lognormal:
    return ks.Lognormal(cm.need(d, "median_ms", float), cm.need(d, "sigma", float))


def text_seed(name: str) -> int:
    """A stable number for a text from its name (not its place in the conditions); as v2 and v3."""
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


def rng_key(c: dict, name: str, gen_id: int, k: int) -> list[int]:
    return [c["keystrokes"]["seed_base"], text_seed(name), gen_id, k]


def regimes_of(text: str, c: dict) -> tuple[ks.TextTiming, ks.MatchedRegimes]:
    kc = c["keystrokes"]
    tt = kc["text_timing"]
    timing = ks.TextTiming(*(lognormal(tt[n]) for n in ("within_word", "word_start", "sentence_end", "paragraph_end")))
    g = ks.matched_regimes(text, timing, c["units"]["rest_ms"], float(kc["regimes"]["mean_run_fast"]))
    return timing, g


def series(name: str, text: str, c: dict, k: int) -> dict[str, np.ndarray]:
    """Press times of the five series for a text and seed index k (the first three as v2 and v3)."""
    timing, g = regimes_of(text, c)
    rest_ms = c["units"]["rest_ms"]
    s = c["series"]
    t_text = ks.from_text(text, timing, np.random.default_rng(rng_key(c, name, 0, k)))
    return {"from_text": t_text,
            "shuffled": ks.shuffled(t_text, np.random.default_rng(rng_key(c, name, 1, k))),
            "regimes": ks.from_matched_regimes(len(ks.positions(text)), g, rest_ms,
                                               np.random.default_rng(rng_key(c, name, 2, k))),
            "inside": ks.rests_inside(t_text, text, np.random.default_rng(rng_key(c, name, s["inside"]["id"], k))),
            "reordered": ks.reordered_by_runs(t_text, text, rest_ms, g.mean_run_fast, g.mean_run_slow,
                                              np.random.default_rng(rng_key(c, name, s["reordered"]["id"], k)))}


def pq(ws: list, m: dict, scale: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """p and q of each tile (sentence or window); tiles without a speed get p = 0 and are listed (the min-max of speed
    uses only the tiles with a speed). As v3."""
    ext = m["external"]
    lo, hi = math.log(ext["fast_ms"]), math.log(ext["slow_ms"])
    no_speed = [i for i, x in enumerate(ws) if x.speed_ms is None]
    has = np.array([x.speed_ms is not None for x in ws])
    logm = np.array([math.log(x.speed_ms) if x.speed_ms is not None else np.nan for x in ws])
    rest = np.array([x.rest_share for x in ws])

    def mm(v):
        span = np.nanmax(v) - np.nanmin(v)
        return (v - np.nanmin(v)) / span if span > 0 else np.where(np.isnan(v), np.nan, 0.5)

    if scale == "external":
        p = np.clip((hi - logm) / (hi - lo), 0.0, 1.0)
        q = 1.0 - rest
    else:
        p = 1.0 - mm(logm) if has.any() else np.zeros(len(ws))
        q = 1.0 - mm(rest)
    p = np.where(has, p, 0.0)
    return p, q, no_speed


def tile(p: float, q: float, r: dict, seed: int, px: int) -> np.ndarray:
    f, s, g = F0 + F1 * p, S0 + S1 * q, G0 + G1 * (WP * p + WQ * q)
    F = procedural_field_separable(f, px, seed, octaves=r["octaves"], normalize=r["normalize"])
    return np.flipud(np.clip(s * t_gamma(F, g) / (2.0 * r["s_max"]) + 0.5, 0.0, 1.0))  # origin lower left


def figure(rows: list[list[np.ndarray]], px: int, gap: int) -> Image.Image:
    cols = max(len(r) for r in rows)
    canvas = np.ones((len(rows) * px + (len(rows) - 1) * gap, cols * px + (cols - 1) * gap))
    for i, row in enumerate(rows):
        for j, t in enumerate(row):
            canvas[i * (px + gap): i * (px + gap) + px, j * (px + gap): j * (px + gap) + px] = t
    return Image.fromarray(np.round(canvas * 255).astype(np.uint8), mode="L")


def middle_half(v: np.ndarray) -> float:
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 - q1)


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson; None when x or y does not vary (all values equal, compared exactly)."""
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summary(v: list) -> dict:
    """Median, min and max over seed indices (None values left out and counted)."""
    a = np.array([x for x in v if x is not None], dtype=float)
    return {"median": float(np.median(a)) if a.size else None, "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None, "none": len(v) - int(a.size)}


def tile_diff(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """Mean over tiles (paired by index up to the smaller number) of the mean absolute difference of grey, on the
    clipped floats before the 8-bit figures."""
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("no tiles to compare")
    return float(np.mean([np.mean(np.abs(a[j] - b[j])) for j in range(n)]))


def row_name(row) -> str:
    return row if isinstance(row, str) else f"{row[0]}/{row[1]}"


def run(c: dict, texts: dict[str, str], out_dir: Path) -> dict:
    u, m, r = c["units"], c["mapping"], c["relief"]
    rest_ms, fields = u["rest_ms"], r["seeds"]
    if not fields or fields[0] != 42:
        raise SystemExit("the first field seed must be 42 (the field of v1 to v3, the base of the differences between fields)")
    sent_rows = [tuple(x) for x in u["sentence"]["rows"]]
    win_rows = list(u["windows"]["rows"])
    fig_w = u["windows"]["figure_width_ms"]
    units = {"sentence": (sent_rows, u["sentence"]["tile_px"], u["sentence"]["gap_px"]),
             f"window_{int(fig_w / 1000)}s": (win_rows, u["windows"]["tile_px"], u["windows"]["gap_px"])}
    rec: dict = {"texts": {}}
    for name, text in texts.items():
        starts = ks.sentence_starts(text)
        presses = np.array([b - a for a, b in zip(starts, [*starts[1:], len(ks.positions(text))])], dtype=float)
        sent: dict = {row_name(x): [] for x in sent_rows}
        win: dict = {f"{int(w)}": {s: [] for s in win_rows} for w in u["windows"]["widths_ms"]}
        diffs: dict = {un: {"between_rows": {row_name(x): [] for x in rows[1:]},
                            "between_fields": {row_name(x): [] for x in rows}} for un, (rows, _, _) in units.items()}
        figs: dict = {}
        for k in range(c["keystrokes"]["seeds"]):
            ser = series(name, text, c, k)
            tiles_pq: dict = {un: {} for un in units}  # unit -> row -> list of (p, q) under both scales
            for row in sent_rows:
                ss = ks.segments_counted(ser[row[0]], starts, rest_ms, row[1])
                p, q, no_speed = pq(ss, m, "external")
                tiles_pq["sentence"][row] = {"external": (p, q), "minmax": pq(ss, m, "minmax")[:2]}
                sent[row_name(row)].append({
                    "seed_index": k, "no_speed": no_speed, "gaps_per_sentence": [x.gaps for x in ss],
                    "speed_ms": [x.speed_ms for x in ss], "rest_share": [x.rest_share for x in ss],
                    "p": p.tolist(), "q": q.tolist(), "middle_half_p": middle_half(p), "middle_half_q": middle_half(q),
                    "correlation_ln_presses_q": correlation(np.log(presses), q),
                    "correlation_without_first": correlation(np.log(presses[1:]), q[1:])})
            for w in u["windows"]["widths_ms"]:
                for s in win_rows:
                    ws, tail = ks.full_windows(ser[s], w, rest_ms)
                    if not ws:
                        raise SystemExit(f"{name} {s} seed {k}: no full window of {w} ms")
                    p, q, no_speed = pq(ws, m, "external")
                    if w == fig_w:
                        tiles_pq[f"window_{int(fig_w / 1000)}s"][s] = {"external": (p, q),
                                                                        "minmax": pq(ws, m, "minmax")[:2]}
                    win[f"{int(w)}"][s].append({
                        "seed_index": k, "presses": int(ser[s].size), "duration_ms": float(ser[s][-1]),
                        "windows": len(ws), "tail_ms": tail, "no_speed": no_speed, "p": p.tolist(), "q": q.tolist(),
                        "middle_half_p": middle_half(p), "middle_half_q": middle_half(q)})
            for un, (rows, px, gap) in units.items():
                grey = {row: {f: [tile(a, b, r, f, px) for a, b in zip(*tiles_pq[un][row]["external"])]
                              for f in fields} for row in rows}
                for row in rows[1:]:
                    diffs[un]["between_rows"][row_name(row)].append(
                        float(np.mean([tile_diff(grey[row][f], grey[rows[0]][f]) for f in fields])))
                for row in rows:
                    diffs[un]["between_fields"][row_name(row)].append(
                        float(np.mean([tile_diff(grey[row][f], grey[row][fields[0]]) for f in fields[1:]])))
                if k == 0:
                    for f in fields:
                        figs[f"{un}_external_f{f}"] = figure([grey[row][f] for row in rows], px, gap)
                    figs[f"{un}_minmax_f{fields[0]}"] = figure(
                        [[tile(a, b, r, fields[0], px) for a, b in zip(*tiles_pq[un][row]["minmax"])] for row in rows],
                        px, gap)
        fig_rec = {}
        for key, im in figs.items():
            path = out_dir / f"{name}_{key}.png"
            im.save(path)
            fig_rec[key] = {"file": f"results/prototype_v4/{path.name}", "sha256": cm.sha256(path)}
        rec["texts"][name] = {
            "sentence_starts": starts, "presses_per_sentence": presses.astype(int).tolist(),
            "mean_run_slow": regimes_of(text, c)[1].mean_run_slow,
            "sentence": sent, "windows": win, "image_difference": diffs, "figures": fig_rec,
            "summary": {
                "sentence": {n: {"middle_half_p": summary([x["middle_half_p"] for x in v]),
                                 "middle_half_q": summary([x["middle_half_q"] for x in v]),
                                 "correlation_ln_presses_q": summary([x["correlation_ln_presses_q"] for x in v]),
                                 "correlation_without_first": summary([x["correlation_without_first"] for x in v])}
                             for n, v in sent.items()},
                "windows": {w: {s: {"middle_half_p": summary([x["middle_half_p"] for x in v]),
                                    "middle_half_q": summary([x["middle_half_q"] for x in v])} for s, v in d.items()}
                            for w, d in win.items()},
                "image_difference": {un: {kind: {n: summary(v) for n, v in d.items()} for kind, d in dd.items()}
                                     for un, dd in diffs.items()}}}
    return rec


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    cm.check_generator(c["generator"])
    from importlib import metadata
    for pkg in ("numpy", "pillow", "scipy"):
        if metadata.version(pkg) != c[pkg]:
            raise SystemExit(f"{pkg} is {metadata.version(pkg)}, not {c[pkg]}")
    paths = [COND.relative_to(ROOT).as_posix(), *CODE, *(t["file"] for t in c["texts"].values())]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    texts = {n: cm.check_file(ROOT / t["file"], t["sha256"], f"text {n}").decode("utf-8") for n, t in c["texts"].items()}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v4-", dir=OUT_DIR.parent))
    try:
        rec = run(c, texts, tmp)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "texts_sha256": {n: t["sha256"] for n, t in c["texts"].items()},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "atact_library_sha256": cm.library_sha256(), "versions": cm.versions("numpy", "pillow", "scipy"),
                         "rng": "numpy.random.default_rng([seed_base, sha256(text name)[:8], generator id, seed index])",
                         "bit_generator": type(np.random.default_rng(0).bit_generator).__name__,
                         "text_seeds": {n: text_seed(n) for n in texts}}
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({n: v["summary"]["image_difference"] for n, v in rec["texts"].items()})[:2000])


if __name__ == "__main__":
    main()
