"""Prototype v5: the difference between the images of v4 measured with numbers that do not depend on the phase of the
field's lattice (decision 5 of .steering/20261031-atlas-v4; numbers and baselines fixed in decisions 1-3 of
.steering/20261031-atlas-v5).

Usage:  uv run python works/prototype_v5.py        (refuses to overwrite)
Conditions: works/conditions/prototype_v5.json (committed before this script is first run), and v4's conditions for
        every key of the tiles
Input:  works/texts/*.en.txt, results/prototype_v4/record.json (the SHA-256 of v4's figures)
Output: results/prototype_v5/record.json (written to a temporary directory beside it and renamed when complete); no
        figure

Steps: the tiles of v4 (series, pq and tile of works/prototype_v4.py, the same rows, units and fields) -> for seed
index 0, v4's figures rebuilt in memory and checked against the SHA-256 of v4's record -> depth (standard deviation of
the grey) and fineness (radial centroid of the power spectrum) of every tile -> differences between rows, between
fields and between seed indices; the map check on a grid of p and q and by Spearman correlation over the tiles.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
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
from scipy.stats import spearmanr  # noqa: E402

import _common as cm  # noqa: E402
import keystroke as ks  # noqa: E402
import prototype_v4 as v4  # noqa: E402

COND = ROOT / "works" / "conditions" / "prototype_v5.json"
OUT_DIR = ROOT / "results" / "prototype_v5"
CODE = ["works/prototype_v5.py", "works/prototype_v4.py", "works/_common.py", "src/keystroke.py", "uv.lock"]
GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def depth(t: np.ndarray) -> float:
    """Standard deviation of the grey over the pixels (ddof 0)."""
    return float(np.std(t))


def fineness(t: np.ndarray) -> float:
    """Radial centroid of the power spectrum of the grey minus its mean, in cycles per tile."""
    g = t - t.mean()
    power = np.abs(np.fft.fft2(g)) ** 2
    k = np.fft.fftfreq(t.shape[0]) * t.shape[0], np.fft.fftfreq(t.shape[1]) * t.shape[1]
    r = np.sqrt(k[0][:, None] ** 2 + k[1][None, :] ** 2)
    total = power.sum()
    if total == 0:
        raise SystemExit("a tile has no power (constant grey); fineness is not defined")
    return float((r * power).sum() / total)


NUMBERS = {"depth": depth, "fineness": fineness}


def diff(a: list[float], b: list[float]) -> float:
    """Mean over tiles paired by index up to the smaller number of |a_j - b_j|."""
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("no tiles to compare")
    return float(np.mean(np.abs(np.asarray(a[:n]) - np.asarray(b[:n]))))


def counts(v: dict, names: list[str], fields: list[int], seeds: int) -> dict:
    """Differences of the conditions for one unit: v[row][field][seed index][number] = values per tile. Between rows
    against names[0] (mean over all fields), between fields against fields[0] (mean over the others), between seeds k
    and k + 1 (mean over all fields)."""
    return {n: {
        "between_rows": {x: [float(np.mean([diff(v[x][f][k][n], v[names[0]][f][k][n]) for f in fields]))
                             for k in range(seeds)] for x in names[1:]},
        "between_fields": {x: [float(np.mean([diff(v[x][f][k][n], v[x][fields[0]][k][n]) for f in fields[1:]]))
                               for k in range(seeds)] for x in names},
        "between_seeds": {x: [float(np.mean([diff(v[x][f][k + 1][n], v[x][f][k][n]) for f in fields]))
                              for k in range(seeds - 1)] for x in names}} for n in NUMBERS}


def spearman(a: list[float], b: list[float], what: str) -> float:
    rho = float(spearmanr(a, b).statistic)
    if not math.isfinite(rho):
        raise SystemExit(f"Spearman correlation of {what} is not defined (a constant input)")
    return rho


def png_sha256(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def grid_check(r: dict, fields: list[int], sizes: dict[str, int]) -> dict:
    """Depth and fineness on the grid of p and q, per tile size and field: rows q, columns p."""
    return {un: {str(f): {n: [[fn(v4.tile(p, q, r, f, px)) for p in GRID] for q in GRID] for n, fn in NUMBERS.items()}
                 for f in fields} for un, px in sizes.items()}


def run(c4: dict, texts: dict[str, str], figs4: dict, want_units: list[str] | None = None,
        want_seeds: int | None = None) -> dict:
    u, m, r = c4["units"], c4["mapping"], c4["relief"]
    rest_ms, fields = u["rest_ms"], r["seeds"]
    if not fields or fields[0] != 42:
        raise SystemExit("the first field seed must be 42 (as v4)")
    sent_rows = [tuple(x) for x in u["sentence"]["rows"]]
    fig_w = u["windows"]["figure_width_ms"]
    win_key = f"window_{int(fig_w / 1000)}s"
    units = {"sentence": (sent_rows, u["sentence"]["tile_px"], u["sentence"]["gap_px"]),
             win_key: (list(u["windows"]["rows"]), u["windows"]["tile_px"], u["windows"]["gap_px"])}
    seeds = c4["keystrokes"]["seeds"]
    if want_units is not None and list(units) != want_units:
        raise SystemExit(f"units {list(units)} of v4's conditions differ from v5's {want_units}")
    if want_seeds is not None and seeds != want_seeds:
        raise SystemExit(f"{seeds} seed indices, not the {want_seeds} of v5's conditions")
    rec: dict = {"grid": {"p": GRID, "q": GRID,
                          "values": grid_check(r, fields, {un: px for un, (_, px, _) in units.items()})},
                 "texts": {}}
    for name, text in texts.items():
        starts = ks.sentence_starts(text)
        # unit -> row -> field -> seed index -> number -> values per tile; and the p, q of the tiles
        vals: dict = {un: {v4.row_name(x): {f: [] for f in fields} for x in rows} for un, (rows, _, _) in units.items()}
        pqs: dict = {un: {v4.row_name(x): [] for x in rows} for un, (rows, _, _) in units.items()}
        checked = 0
        for k in range(seeds):
            ser = v4.series(name, text, c4, k)
            tiles_pq: dict = {un: {} for un in units}
            for row in sent_rows:
                ss = ks.segments_counted(ser[row[0]], starts, rest_ms, row[1])
                ext = v4.pq(ss, m, "external")
                tiles_pq["sentence"][row] = {"external": ext[:2], "no_speed": ext[2],
                                             "minmax": v4.pq(ss, m, "minmax")[:2]}
            for s in units[win_key][0]:
                ws, _ = ks.full_windows(ser[s], fig_w, rest_ms)
                if not ws:
                    raise SystemExit(f"{name} {s} seed {k}: no full window of {fig_w} ms")
                ext = v4.pq(ws, m, "external")
                tiles_pq[win_key][s] = {"external": ext[:2], "no_speed": ext[2], "minmax": v4.pq(ws, m, "minmax")[:2]}
            for un, (rows, px, gap) in units.items():
                grey = {row: {f: [v4.tile(a, b, r, f, px) for a, b in zip(*tiles_pq[un][row]["external"])]
                              for f in fields} for row in rows}
                if k == 0:
                    built = {f"{un}_external_f{f}": v4.figure([grey[row][f] for row in rows], px, gap) for f in fields}
                    built[f"{un}_minmax_f{fields[0]}"] = v4.figure(
                        [[v4.tile(a, b, r, fields[0], px) for a, b in zip(*tiles_pq[un][row]["minmax"])] for row in rows],
                        px, gap)
                    for key, im in built.items():
                        if png_sha256(im) != figs4[name][key]["sha256"]:
                            raise SystemExit(f"{name} {key}: the tiles differ from v4's figure")
                        checked += 1
                for row in rows:
                    p, q = tiles_pq[un][row]["external"]
                    pqs[un][v4.row_name(row)].append({"p": p.tolist(), "q": q.tolist(),
                                                      "no_speed": tiles_pq[un][row]["no_speed"]})
                    for f in fields:
                        vals[un][v4.row_name(row)][f].append({n: [fn(t) for t in grey[row][f]] for n, fn in NUMBERS.items()})
        if checked != len(figs4[name]):
            raise SystemExit(f"{name}: {checked} of v4's {len(figs4[name])} figures checked")
        diffs: dict = {}
        spear: dict = {}
        for un, (rows, _, _) in units.items():
            names = [v4.row_name(x) for x in rows]
            v = vals[un]
            diffs[un] = counts(v, names, fields, seeds)
            num = {n: [] for n in NUMBERS}
            pv, qv = [], []
            for x in names:
                for f in fields:
                    for k in range(seeds):
                        for n in NUMBERS:
                            num[n].extend(v[x][f][k][n])
                        pv.extend(pqs[un][x][k]["p"])
                        qv.extend(pqs[un][x][k]["q"])
            spear[un] = {f"{n}_{w}": spearman(num[n], vv, f"{name} {un} {n} with {w}")
                         for n in NUMBERS for w, vv in (("p", pv), ("q", qv))}
            spear[un]["pairs"] = len(pv)  # tiles x fields x rows x seed indices: not independent tiles
        rec["texts"][name] = {
            "scale": "external", "figures_checked": checked, "pq": pqs,
            "numbers": {un: {x: {str(f): ks_ for f, ks_ in fv.items()} for x, fv in uv.items()} for un, uv in vals.items()},
            "differences": diffs, "spearman": spear,
            "summary": {un: {n: {kind: {x: v4.summary(vv) for x, vv in dd.items()} for kind, dd in d.items()}
                             for n, d in ud.items()} for un, ud in diffs.items()}}
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
    t = c["tiles"]
    raw4 = cm.check_file(ROOT / t["v4_conditions"]["file"], t["v4_conditions"]["sha256"], "v4's conditions")
    c4 = json.loads(raw4)
    if c4["generator"] != c["generator"] or any(c4[p] != c[p] for p in ("numpy", "pillow", "scipy")):
        raise SystemExit("the generator or the packages of v4's conditions differ from v5's")
    rec4 = json.loads(cm.check_file(ROOT / t["v4_record"]["file"], t["v4_record"]["sha256"], "v4's record"))
    for path, h in rec4["provenance"]["code_sha256"].items():  # the code that made v4's record, not only HEAD
        if cm.sha256(ROOT / path) != h:
            raise SystemExit(f"{path} differs from the version that made v4's record")
    paths = [COND.relative_to(ROOT).as_posix(), t["v4_conditions"]["file"], t["v4_record"]["file"], *CODE,
             *(x["file"] for x in c4["texts"].values())]
    cm.same_as_head([ROOT / p for p in paths])
    state = cm.git_state(paths)
    texts = {n: cm.check_file(ROOT / x["file"], x["sha256"], f"text {n}").decode("utf-8") for n, x in c4["texts"].items()}
    rec = run(c4, texts, {n: rec4["texts"][n]["figures"] for n in texts}, want_units=t["units"], want_seeds=10)
    rec["provenance"] = {"run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **state,
                         "conditions_sha256": hashlib.sha256(raw).hexdigest(),
                         "v4_conditions_sha256": hashlib.sha256(raw4).hexdigest(),
                         "v4_record_sha256": t["v4_record"]["sha256"],
                         "texts_sha256": {n: x["sha256"] for n, x in c4["texts"].items()},
                         "code_sha256": {p: cm.sha256(ROOT / p) for p in CODE},
                         "atact_library_sha256": cm.library_sha256(), "versions": cm.versions("numpy", "pillow", "scipy"),
                         "rng": rec4["provenance"]["rng"],
                         "bit_generator": type(np.random.default_rng(0).bit_generator).__name__,
                         "text_seeds": {n: v4.text_seed(n) for n in texts}}
    tmp = Path(tempfile.mkdtemp(prefix="prototype_v5-", dir=OUT_DIR.parent))
    (tmp / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the result is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({n: v["figures_checked"] for n, v in rec["texts"].items()}))


if __name__ == "__main__":
    main()
