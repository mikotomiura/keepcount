"""Prototype v3: one tile per sentence instead of a time window, the rest before a sentence counted in it; the same
press indices cut the shuffled series and the regimes (decisions 9 and 10 of .steering/20261031-synthetic-prototype).

Usage:  uv run python works/prototype_v3.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v3.json (committed before this script is first run)
Input:  works/texts/*.en.txt
Output: results/prototype_v3/record.json and, per text, results/prototype_v3/{text}_{scale}.png (written to a
        temporary directory beside it and renamed when complete)

Steps: src/keystroke (from_text, shuffled, from_matched_regimes -> segments at the text's sentence starts -> speed
and rest) -> p, q (external scale, and min-max per series as reference) -> f, s, gamma with the coefficients of
atact.mapping -> H = s T_gamma(F) on the generator's field -> grey H / (2 s_max) + 1/2.
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

COND = ROOT / "works" / "conditions" / "prototype_v3.json"
OUT_DIR = ROOT / "results" / "prototype_v3"
CODE = ["works/prototype_v3.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
GENERATORS = ("from_text", "shuffled", "regimes")
SCALES = ("external", "minmax")


def lognormal(d: dict) -> ks.Lognormal:
    return ks.Lognormal(cm.need(d, "median_ms", float), cm.need(d, "sigma", float))


def text_seed(name: str) -> int:
    """A stable number for a text from its name (not its place in the conditions)."""
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


GEN_ID = {"from_text": 0, "shuffled": 1, "regimes": 2}


def series(name: str, text: str, c: dict, k: int) -> dict[str, np.ndarray]:
    """Press times of the three generators for a text and seed index k."""
    kc = c["keystrokes"]
    tt = kc["text_timing"]
    timing = ks.TextTiming(*(lognormal(tt[n]) for n in ("within_word", "word_start", "sentence_end", "paragraph_end")))
    rest_ms = c["windows"]["rest_ms"]
    g = ks.matched_regimes(text, timing, rest_ms, float(kc["regimes"]["mean_run_fast"]))
    base, ts = kc["seed_base"], text_seed(name)
    t_text = ks.from_text(text, timing, np.random.default_rng([base, ts, GEN_ID["from_text"], k]))
    return {"from_text": t_text,
            "shuffled": ks.shuffled(t_text, np.random.default_rng([base, ts, GEN_ID["shuffled"], k])),
            "regimes": ks.from_matched_regimes(len(ks.positions(text)), g, rest_ms,
                                               np.random.default_rng([base, ts, GEN_ID["regimes"], k]))}


def pq(ws: list, m: dict, scale: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """p and q of each window; windows without a speed get p = 0 and are listed (the min-max of speed uses only the
    windows with a speed)."""
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


def tile(p: float, q: float, r: dict, px: int) -> np.ndarray:
    f, s, g = F0 + F1 * p, S0 + S1 * q, G0 + G1 * (WP * p + WQ * q)
    F = procedural_field_separable(f, px, r["seed"], octaves=r["octaves"], normalize=r["normalize"])
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


def series_record(times: np.ndarray, starts: list[int], w: dict, m: dict, k: int) -> tuple[dict, list]:
    ss = ks.segments(times, starts, w["rest_ms"])
    p, q, no_speed = pq(ss, m, "external")
    return ({"seed_index": k, "presses": int(times.size), "duration_ms": float(times[-1]), "sentences": len(ss),
             "no_speed": no_speed, "presses_per_sentence": [x.presses for x in ss],
             "time_per_sentence_ms": [float(np.sum(np.diff(times)[max(x.first_press - 1, 0): x.first_press + x.presses - 1]))
                                      for x in ss],
             "speed_ms": [x.speed_ms for x in ss], "rest_share": [x.rest_share for x in ss],
             "p": p.tolist(), "q": q.tolist(), "middle_half_p": middle_half(p), "middle_half_q": middle_half(q)}, ss)


def run(c: dict, texts: dict[str, str], root: Path, out_dir: Path) -> dict:
    w, m, r, fig = c["windows"], c["mapping"], c["relief"], c["figure"]
    rec: dict = {"texts": {}}
    for name, text in texts.items():
        starts = ks.sentence_starts(text)
        per: dict = {g: [] for g in GENERATORS}
        rows: dict = {s: [] for s in SCALES}
        for k in range(c["keystrokes"]["seeds"]):
            for g, times in series(name, text, c, k).items():
                one, ss = series_record(times, starts, w, m, k)
                per[g].append(one)
                if k == 0:
                    for s in SCALES:
                        ps, qs, _ = pq(ss, m, s)
                        rows[s].append([tile(a, b, r, fig["tile_px"]) for a, b in zip(ps, qs)])
        figs = {}
        for s in SCALES:
            path = out_dir / f"{name}_{s}.png"
            figure(rows[s], fig["tile_px"], fig["gap_px"]).save(path)
            figs[s] = {"file": f"results/prototype_v3/{path.name}", "sha256": cm.sha256(path)}
        rec["texts"][name] = {"sentence_starts": starts, "series": per, "figures": figs}
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
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v3-", dir=OUT_DIR.parent))
    try:
        rec = run(c, texts, ROOT, tmp)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "texts_sha256": {n: t["sha256"] for n, t in c["texts"].items()},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "atact_library_sha256": cm.library_sha256(), "versions": cm.versions("numpy", "pillow", "scipy"),
                         "rng": "numpy.random.default_rng([seed_base, sha256(text name)[:8], generator id, seed index])"}
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    summary = {n: {g: [round(float(np.median([s[k] for s in v["series"][g]])), 3)
                       for k in ("middle_half_p", "middle_half_q")] for g in GENERATORS} for n, v in rec["texts"].items()}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
