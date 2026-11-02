"""Prototype v1: synthetic English keystrokes of four texts, one tile per 30 s window, through the form of the rule.

Usage:  uv run python works/prototype_v1.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v1.json (committed before this script is first run)
Input:  works/texts/*.en.txt (written for this purpose, in the repository)
Output: results/prototype_v1/record.json and, per text, results/prototype_v1/{text}_{scale}.png

Steps: src/keystroke (from_text, shuffled, from_regimes -> windows -> speed and rest) -> p, q (external scale, and
min-max per series as reference) -> f, s, gamma with the coefficients of atact.mapping -> H = s T_gamma(F) on the
generator's field -> grey H / (2 s_max) + 1/2.
"""

from __future__ import annotations

import atexit
import datetime as dt
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
atexit.register(shutil.rmtree, sys.pycache_prefix, ignore_errors=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
import keystroke as ks  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.mapping import F0, F1, G0, G1, S0, S1, WP, WQ  # noqa: E402
from atact.transform import t_gamma  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v1.json"
OUT_DIR = ROOT / "results" / "prototype_v1"
CODE = ["works/prototype_v1.py", "works/_common.py", "src/keystroke.py"]
GENERATORS = ("from_text", "shuffled", "regimes")
SCALES = ("external", "minmax")


def lognormal(d: dict) -> ks.Lognormal:
    return ks.Lognormal(cm.need(d, "median_ms", float), cm.need(d, "sigma", float))


def series(text: str, c: dict, ti: int, k: int) -> dict[str, np.ndarray]:
    """Press times of the three generators for text index ti and seed index k."""
    kc = c["keystrokes"]
    tt = kc["text_timing"]
    timing = ks.TextTiming(*(lognormal(tt[n]) for n in ("within_word", "word_start", "sentence_end", "paragraph_end")))
    labels = ks.positions(text)
    rests = sum(lab in ("sentence_end", "paragraph_end") for lab in labels)
    rg = kc["regimes"]
    regimes = ks.Regimes(timing.within_word, timing.word_start, float(rg["mean_run_fast"]), float(rg["mean_run_slow"]),
                         timing.sentence_end, rests / (len(labels) - 1))
    base = kc["seed_base"]
    t_text = ks.from_text(text, timing, np.random.default_rng([base, ti, 0, k]))
    return {"from_text": t_text,
            "shuffled": ks.shuffled(t_text, np.random.default_rng([base, ti, 1, k])),
            "regimes": ks.from_regimes(len(labels), regimes, c["windows"]["rest_ms"],
                                       np.random.default_rng([base, ti, 2, k]))}


def full_windows(times: np.ndarray, w: dict) -> tuple[list[ks.Window], int]:
    """The windows of full width; the last partial window is dropped (counted)."""
    ws = ks.windows(times, w["width_ms"], w["rest_ms"])
    full = [x for x in ws if x.start_ms + w["width_ms"] <= times[-1]]
    return full, len(ws) - len(full)


def pq(ws: list[ks.Window], m: dict, scale: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """p and q of each window; windows without a speed get p = 0 and are listed."""
    ext = m["external"]
    lo, hi = math.log(ext["fast_ms"]), math.log(ext["slow_ms"])
    no_speed = [i for i, x in enumerate(ws) if x.speed_ms is None]
    logm = np.array([math.log(x.speed_ms) if x.speed_ms is not None else hi for x in ws])
    rest = np.array([x.rest_share for x in ws])
    if scale == "external":
        p = np.clip((hi - logm) / (hi - lo), 0.0, 1.0)
        q = 1.0 - rest
    else:
        def mm(v):
            span = v.max() - v.min()
            return (v - v.min()) / span if span > 0 else np.full_like(v, 0.5)
        p = 1.0 - mm(logm)
        q = 1.0 - mm(rest)
    p[no_speed] = 0.0
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


def run(c: dict, root: Path, out_dir: Path) -> dict:
    w, m, r, fig = c["windows"], c["mapping"], c["relief"], c["figure"]
    out_dir.mkdir(parents=True)
    rec: dict = {"texts": {}}
    for ti, (name, t) in enumerate(c["texts"].items()):
        text = cm.check_file(root / t["file"], t["sha256"], f"text {name}").decode("utf-8")
        per: dict = {g: [] for g in GENERATORS}
        rows: dict = {s: [] for s in SCALES}
        for k in range(c["keystrokes"]["seeds"]):
            for g, times in series(text, c, ti, k).items():
                ws, dropped = full_windows(times, w)
                p, q, no_speed = pq(ws, m, "external")
                per[g].append({"seed_index": k, "presses": int(times.size), "duration_ms": float(times[-1]),
                               "windows": len(ws), "dropped_partial": dropped, "no_speed": no_speed,
                               "speed_ms": [x.speed_ms for x in ws], "rest_share": [x.rest_share for x in ws],
                               "p": p.tolist(), "q": q.tolist(),
                               "middle_half_p": middle_half(p), "middle_half_q": middle_half(q)})
                if k == 0:
                    for s in SCALES:
                        ps, qs, _ = pq(ws, m, s)
                        rows[s].append([tile(a, b, r, fig["tile_px"]) for a, b in zip(ps, qs)])
        figs = {}
        for s in SCALES:
            path = out_dir / f"{name}_{s}.png"
            figure(rows[s], fig["tile_px"], fig["gap_px"]).save(path)
            figs[s] = {"file": path.relative_to(root).as_posix(), "sha256": cm.sha256(path)}
        rec["texts"][name] = {"series": per, "figures": figs}
    return rec


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    cm.check_generator(c["generator"])
    for pkg in ("numpy", "pillow", "scipy"):
        from importlib import metadata
        if metadata.version(pkg) != c[pkg]:
            raise SystemExit(f"{pkg} is {metadata.version(pkg)}, not {c[pkg]}")
    paths = [COND.relative_to(ROOT).as_posix(), *CODE, *(t["file"] for t in c["texts"].values())]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    rec = run(c, ROOT, OUT_DIR)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": cm.hashlib.sha256(raw).hexdigest(),
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"),
                         "atact_library_sha256": cm.library_sha256(), "versions": cm.versions("numpy", "pillow", "scipy")}
    (OUT_DIR / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    summary = {n: {g: {"windows": [s["windows"] for s in v["series"][g]],
                       "middle_half_p": round(float(np.median([s["middle_half_p"] for s in v["series"][g]])), 3),
                       "middle_half_q": round(float(np.median([s["middle_half_q"] for s in v["series"][g]])), 3)}
                   for g in GENERATORS} for n, v in rec["texts"].items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
