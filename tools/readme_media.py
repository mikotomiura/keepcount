"""Images of the README, made from synthetic keystrokes only: the series of run v4 for one text written by the authors
(works/texts/seasons.en.txt, seed index 0), re-made with v4's own functions and checked against v4's record and figure.
No person's record enters these images.

- atlas.png: the sentence tiles of that text under the countings before, after and none, the operation inside and the
  shuffle (the rows of Fig. 3 of the article; read from v4's figure after checking its SHA-256 and that the re-made
  tiles equal it pixel for pixel).
- process.gif: for each sentence, its key presses on the time line, its gaps against the 2 s threshold, the speed p
  and rest q, the parameters f, s and gamma of the rule, and its tile (the tile of v4's figure, enlarged).
- pipeline.svg: the stages, the reference observable and the two counts of the audit (drawn from Sections 3 and 4 of
  the article; no number).

It is an illustration: no number of the article comes from it.

Usage:  uv run --frozen python tools/readme_media.py      (writes docs/media/; overwrites)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import keystroke as ks  # noqa: E402
import prototype_v4 as v4  # noqa: E402
from atact.mapping import F0, F1, G0, G1, S0, S1, WP, WQ  # noqa: E402

OUT = ROOT / "docs" / "media"
TEXT = "seasons"
FIELD = 42  # the first field of v4, the one of Fig. 3
ATLAS_ROWS = (("from_text", "before"), ("from_text", "after"), ("from_text", "none"), ("inside", "before"),
              ("shuffled", "before"))
ATLAS_LABELS = ("before", "after", "none", "inside", "shuffled")
FONT = "/System/Library/Fonts/Helvetica.ttc"
INK, MUTED, FAINT, PAPER = (31, 31, 30), (107, 107, 102), (215, 215, 210), (255, 255, 255)
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)


def press_chars(text: str) -> list[str]:
    """The character of each press, in the order of keystroke.positions (spaces as typed, Enter as a line break)."""
    paras = [p.split() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    out: list[str] = []
    for pi, words in enumerate(paras):
        if pi > 0:
            out.append("\n")
        for wi, word in enumerate(words):
            if wi > 0:
                out.append(" ")
            out.extend(word)
    if len(out) != len(ks.positions(text)):
        raise SystemExit("the characters do not match the presses of keystroke.positions")
    return out


def remake(c: dict, text: str, rec: dict) -> dict:
    """v4's series of the text at seed index 0, its sentence tiles under the atlas rows on FIELD, checked against the
    record (p and q exactly) and against v4's figure (every pixel)."""
    u, m, r = c["units"], c["mapping"], c["relief"]
    starts = ks.sentence_starts(text)
    if starts != rec["sentence_starts"]:
        raise SystemExit("the sentence starts differ from the record")
    ser = v4.series(TEXT, text, c, 0)
    rows = [tuple(x) for x in u["sentence"]["rows"]]
    tiles, pqs = {}, {}
    for row in rows:
        ss = ks.segments_counted(ser[row[0]], starts, u["rest_ms"], row[1])
        p, q, _ = v4.pq(ss, m, "external")
        got = rec["sentence"][v4.row_name(row)][0]
        if (got["seed_index"] != 0 or got["p"] != p.tolist() or got["q"] != q.tolist()
                or got["speed_ms"] != [x.speed_ms for x in ss] or got["rest_share"] != [x.rest_share for x in ss]):
            raise SystemExit(f"p, q, speed or rest of {v4.row_name(row)} differ from the record")
        pqs[row] = (ss, p, q)
        tiles[row] = [v4.tile(a, b, r, FIELD, u["sentence"]["tile_px"]) for a, b in zip(p, q)]
    fig = rec["figures"][f"sentence_external_f{FIELD}"]
    path = ROOT / fig["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != fig["sha256"]:
        raise SystemExit("v4's figure does not match the SHA-256 of v4's record")
    remade = v4.figure([tiles[row] for row in rows], u["sentence"]["tile_px"], u["sentence"]["gap_px"])
    if not np.array_equal(np.asarray(remade), np.asarray(Image.open(path))):
        raise SystemExit("the re-made tiles differ from v4's figure")
    return {"times": ser["from_text"], "starts": starts, "pqs": pqs, "tiles": tiles, "figure": path,
            "px": u["sentence"]["tile_px"], "gap": u["sentence"]["gap_px"], "rows": rows}


def atlas(mk: dict, fonts: dict, dst: Path) -> None:
    """The five rows of Fig. 3, cut from v4's figure, with their labels."""
    im = Image.open(mk["figure"]).convert("RGB")
    px, gap = mk["px"], mk["gap"]
    label_w, pad = 96, 12
    out = Image.new("RGB", (label_w + im.width + pad, len(ATLAS_ROWS) * (px + gap) - gap + 2 * pad), PAPER)
    dr = ImageDraw.Draw(out)
    for i, (row, label) in enumerate(zip(ATLAS_ROWS, ATLAS_LABELS)):
        k = mk["rows"].index(row)
        band = im.crop((0, k * (px + gap), im.width, k * (px + gap) + px))
        y = pad + i * (px + gap)
        out.paste(band, (label_w, y))
        dr.text((label_w - 12, y + px / 2), label, font=fonts["text"], fill=INK, anchor="rm")
    out.save(dst, optimize=True)


def bar(dr: ImageDraw.ImageDraw, x: float, y: float, w: float, v: float, lo: float, hi: float, col) -> None:
    dr.rectangle((x, y, x + w, y + 6), fill=(236, 236, 232))
    dr.rectangle((x, y, x + w * (v - lo) / (hi - lo), y + 6), fill=col)


def frame(k: int, mk: dict, chars: list[str], c: dict, fonts: dict) -> Image.Image:
    W, H = 1000, 470
    im = Image.new("RGB", (W, H), PAPER)
    dr = ImageDraw.Draw(im)
    t, starts = mk["times"] / 1000.0, mk["starts"]
    ss, p, q = mk["pqs"][("from_text", "before")]
    n = len(starts)
    rest_s = c["units"]["rest_ms"] / 1000.0
    a, b = starts[k], (starts[k + 1] if k + 1 < n else t.size)
    dr.text((24, 16), f"Synthetic keystrokes of a text written for the purpose (run v4), sentence {k + 1} of {n}",
            font=fonts["small"], fill=MUTED)
    sentence = "".join(chars[a:b]).replace("\n", " ").strip()
    for j, ln in enumerate(textwrap.wrap(sentence, 120)[:2]):
        dr.text((24, 38 + 20 * j), ln, font=fonts["text"], fill=INK)

    # 1. the whole series on the time line: one tick per press, rests (gaps of 2 s or more) in orange
    x0, x1, y = 24, W - 24, 104
    span = float(t[-1])

    def at(s: float) -> float:
        return x0 + (x1 - x0) * s / span

    lo_s = float(t[max(a - 1, 0)])  # the gap before the sentence is counted in it ('before')
    dr.rectangle((at(lo_s), y - 14, at(float(t[b - 1])), y + 14), fill=(232, 240, 251))
    gaps = np.diff(t)
    for j, g in enumerate(gaps):
        if g >= rest_s - ks.TOL_MS / 1000.0:
            dr.line((at(float(t[j])), y, at(float(t[j + 1])), y), fill=ORANGE, width=3)
    for s in t:
        dr.line((at(float(s)), y - 6, at(float(s)), y + 6), fill=INK, width=1)
    dr.text((x0, y + 20), f"key presses over time (0 to {span:.0f} s); orange: rests of {rest_s:.0f} s or more; "
            "blue: this sentence and the gap before it", font=fonts["small"], fill=MUTED)

    # 2. the gaps of the sentence on a log scale, against the threshold
    gx0, gy0, gw, gh = 24, 170, 300, 150
    g = gaps[max(a - 1, 0): b - 1]
    lg_lo, lg_hi = math.log10(0.05), math.log10(20.0)

    def gx(v: float) -> float:
        return gx0 + gw * (math.log10(min(max(v, 0.05), 20.0)) - lg_lo) / (lg_hi - lg_lo)

    dr.text((gx0, gy0 - 4), "gaps of the sentence (s, log scale)", font=fonts["small"], fill=MUTED)
    rng = np.random.default_rng(0)  # vertical jitter of the dots only
    for v, jy in zip(g, rng.uniform(0, 1, g.size)):
        col = ORANGE if v >= rest_s - ks.TOL_MS / 1000.0 else INK
        x = gx(float(v))
        yy = gy0 + 26 + jy * (gh - 60)
        dr.ellipse((x - 3, yy - 3, x + 3, yy + 3), fill=col)
    base = gy0 + gh - 24
    dr.line((gx0, base, gx0 + gw, base), fill=FAINT)
    for v in (0.1, 0.5, 2.0, 10.0):
        dr.line((gx(v), base, gx(v), base + 4), fill=MUTED)
        dr.text((gx(v), base + 6), f"{v:g}", font=fonts["small"], fill=MUTED, anchor="ma")
    dr.line((gx(rest_s), gy0 + 18, gx(rest_s), base), fill=ORANGE, width=1)
    seg = ss[k]
    if seg.speed_ms is not None:
        dr.line((gx(seg.speed_ms / 1000.0), gy0 + 18, gx(seg.speed_ms / 1000.0), base), fill=BLUE, width=2)
    dr.text((gx0, base + 24), f"blue: median of the gaps under {rest_s:g} s (speed)", font=fonts["small"], fill=MUTED)

    # 3. p and q, and the rule
    px0 = 370
    dr.text((px0, gy0 - 4), "two quantities", font=fonts["small"], fill=MUTED)
    speed = "none" if seg.speed_ms is None else f"{seg.speed_ms:.0f} ms"
    rows = [(f"p  speed  (median gap {speed})", p[k], 0.0, 1.0),
            (f"q  1 - rest  (share of rest {seg.rest_share:.2f})", q[k], 0.0, 1.0)]
    for j, (name, v, lo, hi) in enumerate(rows):
        yy = gy0 + 20 + 44 * j
        dr.text((px0, yy), f"{name}", font=fonts["small"], fill=INK)
        dr.text((px0 + 250, yy), f"{v:.2f}", font=fonts["small"], fill=INK, anchor="ra")
        bar(dr, px0, yy + 20, 250, float(v), lo, hi, BLUE)
    f, s, gm = F0 + F1 * p[k], S0 + S1 * q[k], G0 + G1 * (WP * p[k] + WQ * q[k])
    dr.text((px0, gy0 + 112), "the rule", font=fonts["small"], fill=MUTED)
    for j, (name, v, lo, hi) in enumerate((("f  fineness", f, F0, F0 + F1), ("s  depth", s, S0, S0 + S1),
                                           ("γ  roughness", gm, G0, G0 + G1))):
        yy = gy0 + 134 + 36 * j
        dr.text((px0, yy), name, font=fonts["small"], fill=INK)
        dr.text((px0 + 250, yy), f"{v:.2f}", font=fonts["small"], fill=INK, anchor="ra")
        bar(dr, px0, yy + 18, 250, float(v), lo, hi, MUTED)

    # 4. the tile, as in v4's figure, enlarged
    tile = mk["tiles"][("from_text", "before")][k]
    big = Image.fromarray(np.round(tile * 255).astype(np.uint8), mode="L").resize((240, 240), Image.NEAREST)
    tx = W - 24 - 240
    im.paste(big.convert("RGB"), (tx, gy0 + 16))
    dr.text((tx, gy0 - 4), "the tile (grey = height)", font=fonts["small"], fill=MUTED)
    for ax in (334, 636):
        ym = gy0 + 110
        dr.line((ax, ym, ax + 18, ym), fill=MUTED, width=2)
        dr.polygon([(ax + 24, ym), (ax + 16, ym - 5), (ax + 16, ym + 5)], fill=MUTED)

    # 5. the sequence of tiles so far
    sx, sy, sp = 24, H - 50, 34
    dr.text((sx, sy - 22), "the image: one tile per sentence", font=fonts["small"], fill=MUTED)
    for j, tj in enumerate(mk["tiles"][("from_text", "before")]):
        small = Image.fromarray(np.round(tj * 255).astype(np.uint8), mode="L").resize((sp - 4, sp - 4), Image.LANCZOS)
        x = sx + j * sp
        if j <= k:
            im.paste(small.convert("RGB"), (x, sy))
        else:
            dr.rectangle((x, sy, x + sp - 5, sy + sp - 5), outline=FAINT)
        if j == k:
            dr.rectangle((x - 2, sy - 2, x + sp - 3, sy + sp - 3), outline=BLUE, width=2)
    return im


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 470" width="960" height="470" font-family="Helvetica, Arial, sans-serif">
  <title>The stages audited by KeepCount, the reference observable and the two counts</title>
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#6b6b66"/>
    </marker>
  </defs>
  <rect width="960" height="470" fill="#ffffff"/>
  <g font-size="13" fill="#1f1f1e" text-anchor="middle">
    <rect x="16" y="150" width="150" height="54" rx="4" fill="#ffffff" stroke="#8a8a85"/>
    <text x="91" y="173">key presses</text><text x="91" y="191" font-size="11" fill="#6b6b66">times, kind of key</text>

    <rect x="226" y="60" width="170" height="54" rx="4" fill="#ffffff" stroke="#8a8a85"/>
    <text x="311" y="83">S0  times</text><text x="311" y="101" font-size="11" fill="#6b6b66">quantiles of the log gap</text>

    <rect x="226" y="150" width="170" height="54" rx="4" fill="#ffffff" stroke="#2a78d6" stroke-width="1.6"/>
    <text x="311" y="173">S1  two quantities</text><text x="311" y="191" font-size="11" fill="#6b6b66">speed p, rest q per tile</text>

    <rect x="456" y="150" width="170" height="54" rx="4" fill="#ffffff" stroke="#2a78d6" stroke-width="1.6"/>
    <text x="541" y="173">S2  form</text><text x="541" y="191" font-size="11" fill="#6b6b66">depth, fineness of the grey</text>

    <rect x="456" y="240" width="170" height="54" rx="4" fill="#ffffff" stroke="#2a78d6" stroke-width="1.6"/>
    <text x="541" y="263">SEQ</text><text x="541" y="281" font-size="11" fill="#6b6b66">order of the tiles</text>

    <rect x="686" y="150" width="170" height="54" rx="4" fill="#ffffff" stroke="#8a8a85" stroke-dasharray="4 3"/>
    <text x="771" y="173">image</text><text x="771" y="191" font-size="11" fill="#6b6b66">(not measured)</text>

    <rect x="686" y="240" width="170" height="54" rx="4" fill="#ffffff" stroke="#2a78d6" stroke-width="1.6"/>
    <text x="771" y="263">IMG  tile sequence</text><text x="771" y="281" font-size="11" fill="#6b6b66">each tile in its place</text>

    <rect x="16" y="330" width="380" height="62" rx="4" fill="#fdf1eb" stroke="#eb6834" stroke-width="1.6"/>
    <text x="206" y="352">R  reference observable</text>
    <text x="206" y="370" font-size="11" fill="#6b6b66">median log gap at each kind of boundary (sentence, subsentence,</text>
    <text x="206" y="384" font-size="11" fill="#6b6b66">word, within word), rest at them, lag-1 correlation of the gaps</text>
  </g>
  <g stroke="#6b6b66" stroke-width="1.3" fill="none" marker-end="url(#a)">
    <path d="M166,170 L226,95"/>
    <path d="M166,177 L226,177"/>
    <path d="M396,177 L456,177"/>
    <path d="M396,195 L456,255"/>
    <path d="M626,177 L686,177" stroke-dasharray="4 3"/>
    <path d="M626,195 L686,255"/>
    <path d="M91,204 L91,330"/>
    <path d="M330,204 L330,308 L760,308 L760,296"/>
  </g>
  <text x="541" y="140" font-size="11" fill="#2a78d6" text-anchor="middle">the image stages (blue)</text>
  <g font-size="12" fill="#1f1f1e">
    <rect x="456" y="320" width="488" height="134" rx="4" fill="#f6f6f4" stroke="#d7d7d2"/>
    <text x="472" y="343" font-weight="bold">At every stage, two counts</text>
    <text x="472" y="364">(a) the person: register on one task, match on another; rank AUC</text>
    <text x="472" y="384">(b) an operation (shuffling the gaps, moving a rest inside a sentence):</text>
    <text x="490" y="400">does it move the stage more than the person's own variation?</text>
    <text x="472" y="420" fill="#6b6b66">An operation counts only where it moves R, and an observable only where</text>
    <text x="472" y="436" fill="#6b6b66">a change made to show (gathering the rests) shows; narrower yardsticks</text>
    <text x="472" y="452" fill="#6b6b66">come with a null of exchangeable gaps (the same gaps in random order).</text>
  </g>
  <text x="16" y="30" font-size="15" fill="#1f1f1e">The stages are summaries of the raw series, not a chain: S1 cannot be computed from S0.</text>
</svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = v4.COND.read_bytes()
    full = json.loads((v4.OUT_DIR / "record.json").read_text())
    if hashlib.sha256(raw).hexdigest() != full["provenance"]["conditions_sha256"]:
        raise SystemExit("v4's conditions differ from those of v4's record")
    if v4.cm.library_sha256() != full["provenance"]["atact_library_sha256"]:
        raise SystemExit("the installed generator differs from the one of v4's record")
    c = json.loads(raw)
    tc = c["texts"][TEXT]
    text = v4.cm.check_file(ROOT / tc["file"], tc["sha256"], f"text {TEXT}").decode("utf-8")
    rec = full["texts"][TEXT]
    mk = remake(c, text, rec)
    fonts = {"text": ImageFont.truetype(FONT, 15), "small": ImageFont.truetype(FONT, 13)}
    atlas(mk, fonts, OUT / "atlas.png")
    chars = press_chars(text)
    frames = [frame(k, mk, chars, c, fonts) for k in range(len(mk["starts"]))]
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(OUT / "process.gif", save_all=True, append_images=pal[1:], duration=1800, loop=0, optimize=True)
    (OUT / "pipeline.svg").write_text(SVG)
    for f in sorted(OUT.iterdir()):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
