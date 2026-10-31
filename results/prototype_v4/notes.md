# Notes on prototype v4 (read after the run, 2026-10-31)

Record: results/prototype_v4/record.json (git commit 32a8315, not dirty; conditions 4cc995d, committed before the
run; code fixes from the code review and the Codex review in 32a8315, also before the run). Seeds 10, field seeds 42-45.
All counts below were fixed in the conditions before running (decision 3 of .steering/20261031-atlas-v4); medians over
the ten seed indices, external scale.

Check: the rows from_text/before, shuffled/before and regimes/before give the same p and q as v3, and the 10 s and
30 s windows of from_text, shuffled and regimes the same as v2 (smoke run of two seeds before the run; the same
random keys). v1 to v3 re-run at b821a79 into a scratch directory give their records and figures byte for byte.

## Per sentence: spread of p / q and correlation of ln(presses of a sentence) with q

| text | before | after | none | inside | shuffled | reordered | regimes |
|---|---|---|---|---|---|---|---|
| seasons, spread | .056/.256 | .054/.209 | .054/.000 | .056/.256 | .071/.237 | .130/.208 | .146/.247 |
| seasons, r | +0.72 | +0.52 | -0.20 | +0.72 | -0.21 | -0.30 | -0.25 |
| rule, spread | .062/.155 | .058/.165 | .062/.000 | .062/.155 | .058/.175 | .082/.182 | .125/.189 |
| rule, r | +0.25 | +0.21 | -0.22 (3 None) | +0.25 | +0.02 | +0.02 | -0.07 |
| elements, spread | .063/.235 | .061/.147 | .063/.000 | .063/.235 | .077/.212 | .110/.209 | .098/.204 |
| elements, r | +0.79 | +0.59 | -0.13 (1 None) | +0.79 | -0.28 | -0.08 | -0.19 |
| features, spread | .062/.105 | .063/.111 | .064/.000 | .062/.105 | .063/.186 | .112/.189 | .120/.163 |
| features, r | +0.19 | +0.39 | -0.04 | +0.19 | +0.01 | +0.01 | -0.09 |

before / after / none: from_text with the gap before a sentence counted in it, in the previous one, in none.
inside: the gap before a sentence swapped with one of its own word-start gaps. Without the first sentence: before
+0.74, +0.30, +0.93, +0.26; after +0.54, +0.17, +0.71, +0.42 (record, summary.correlation_without_first). Ranges over
seeds are wide (after: elements -0.31 to +0.82, features -0.55 to +0.66).

## 10 s windows: spread of p / q

| text | from_text | inside | shuffled | reordered | regimes |
|---|---|---|---|---|---|
| seasons | .107/.273 | .092/.298 | .116/.294 | .229/.290 | .226/.293 |
| rule | .113/.247 | .103/.257 | .120/.237 | .206/.234 | .214/.284 |
| elements | .096/.302 | .098/.269 | .110/.288 | .228/.282 | .209/.369 |
| features | .114/.220 | .106/.253 | .132/.216 | .241/.232 | .250/.279 |

30 s: p .04-.06 for from_text, inside, shuffled and .08-.13 for reordered and regimes (record).

## Difference of grey (mean absolute, floats before the 8-bit figures)

Between rows on the same field (against from_text/before), and between fields (same row, 43-45 against 42):

| text | after | none | inside | shuffled | reordered | regimes | fields (range over rows) |
|---|---|---|---|---|---|---|---|
| seasons, sentence | .050 | .047 | .000 | .181 | .183 | .195 | .242-.293 |
| rule, sentence | .038 | .035 | .000 | .158 | .181 | .201 | .251-.295 |
| elements, sentence | .035 | .044 | .000 | .165 | .179 | .180 | .243-.294 |
| features, sentence | .036 | .035 | .000 | .165 | .181 | .174 | .259-.292 |
| seasons, 10 s | | | .087 | .195 | .204 | .203 | .244-.253 |
| rule, 10 s | | | .093 | .197 | .213 | .208 | .246-.264 |
| elements, 10 s | | | .093 | .193 | .207 | .203 | .238-.254 |
| features, 10 s | | | .095 | .201 | .207 | .218 | .251-.262 |

## Readings (Claude, after seeing seasons_sentence_external_f42.png, seasons_sentence_external_f43.png and
## seasons_window_10s_external_f42.png; the other figures not looked at)

1. Where inside a sentence its rest falls does not show per sentence: inside equals before exactly (difference .000),
   as it must, since a sentence's speed and rest share do not depend on the order of its gaps. In 10 s windows the
   move shows a little (.09 against .19-.22 for the other generators): the rest moves by a few words only.
2. The correlation of sentence length with q comes from counting the gap before a sentence in some sentence. Counted
   in none, q is 1 almost everywhere (spread .000; the tiles of that row are all deep) and the correlation is near
   zero or negative. Counted in the previous sentence it stays positive in every text (+0.21 to +0.59): any sentence
   that takes the rest dilutes it by its length. This is the denominator effect of the Codex review of the claim
   (.steering/20261031-keystroke-claim, candidates.md); the counting, not the timing, decides which sentences are pale.
   Each rest moves to the sentence on the left, and how pale that tile becomes depends on that sentence's length
   (seasons, seed index 0: q below 0.8 at sentences 3, 6, 9 for "before" and at 1, 5, 8, 9 for "after"; the rest
   before sentence 3 makes sentence 2 only 0.88). So which tiles are pale is a choice of counting.
3. The wider p of the regimes (v2's reading 3) comes from the order of the runs, not from the variation of counts and
   lengths within a series: the gaps of the series from the text put in the regimes' order (reordered) spread p as
   wide as the regimes (10 s .21-.24 against .21-.25; text and shuffled .10-.13). Limit (Codex review of v4): the
   reordered series has the text's counts fixed, and after one class is used up the rest of the places take the
   other in one block, so the runs it realises are not exactly the regimes'.
4. Changing the field seed changes the grey more than any change of series (.24-.29 against at most .22). But this
   count is pixel by pixel and the fields differ mainly in the phase of the lattice; to the eye (seasons, 42 against
   43) the pale and coarse tiles stand in the same places on both fields and only the lattice's look changes. The
   count fixed before running thus measures the phase, which the eye discounts; it does not show that the series
   cannot be told apart. The difference is on the floats before the 8-bit figures, and differences in saturated
   pixels are lost (Codex review of v4).
5. What the atlas of v4 lists, for this transform and these synthetic series: lost per sentence, the order of gaps
   inside a sentence; lost in windows and per sentence alike, the text against its shuffled gaps (v1-v3); kept, the
   counting of the gap before a sentence (which sentence takes the rest) and the clustering of slow gaps into runs;
   kept by the eye across fields, the depth and fineness of each tile.

Limits: synthetic English keystrokes only; one seed index per figure; the row "after" keeps an edge at the last
sentence (it holds only its inner gaps), which "without the first sentence" does not remove (code review of v4);
the row "none" has q = 1 in most sentences, so its correlation rests on few sentences (None in 3 seeds of rule and 1
of elements); the readings of the figures are from three figures of one text.
