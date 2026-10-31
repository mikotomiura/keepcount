# Notes on prototype v5 (read after the run, 2026-10-31)

Record: results/prototype_v5/record.json (SHA-256 356b10efff7e63d2df751697fbdbedc09c278539d559143ba9bbd8007ac5f9ad;
git commit f7a659f, not dirty; conditions 8fe045c, committed before the run; code fixes from the code review and the
Codex review in f7a659f, also before the run). The tiles are v4's: for seed index 0 the 40 figures of v4 were rebuilt
in memory and matched the SHA-256 of v4's record (figures_checked 10 per text), and the code that made v4's record
matched its recorded SHA-256. All counts below were fixed in the conditions before running (decisions 1-3 of
.steering/20261031-atlas-v5); medians over seed indices (between rows) and over the nine pairs k, k + 1 (between
seeds); external scale; all tiles, those without a speed included.

Depth = standard deviation of the grey of a tile; fineness = radial centroid of its power spectrum (cycles per tile).
Each cell: difference from the text's row (same seed index, mean over fields) / difference between seed indices k and
k + 1 of that row. "text" shows only its own seed difference. Last column: difference between fields (43-45 against
42), range over the rows.

### depth, per sentence: between rows / between seeds of the row (fields: range over rows)

| text | text | after | none | inside | shuffled | reordered | regimes | fields |
|---|---|---|---|---|---|---|---|---|
| seasons | -  / 0.026 | 0.039 / 0.025 | 0.043 / 0.013 | 0.000 / 0.026 | 0.045 / 0.035 | 0.047 / 0.034 | 0.047 / 0.034 | 0.002-0.003 |
| rule | -  / 0.022 | 0.026 / 0.024 | 0.027 / 0.012 | 0.000 / 0.022 | 0.026 / 0.029 | 0.025 / 0.032 | 0.038 / 0.040 | 0.002-0.003 |
| elements | -  / 0.021 | 0.031 / 0.027 | 0.040 / 0.009 | 0.000 / 0.021 | 0.051 / 0.039 | 0.048 / 0.042 | 0.047 / 0.046 | 0.002-0.002 |
| features | -  / 0.020 | 0.023 / 0.023 | 0.026 / 0.014 | 0.000 / 0.020 | 0.028 / 0.036 | 0.031 / 0.035 | 0.030 / 0.032 | 0.002-0.003 |

### fineness, per sentence: between rows / between seeds of the row (fields: range over rows)

| text | text | after | none | inside | shuffled | reordered | regimes | fields |
|---|---|---|---|---|---|---|---|---|
| seasons | -  / 0.51 | 0.10 / 0.46 | 0.13 / 0.50 | 0.00 / 0.51 | 0.63 / 0.72 | 0.92 / 1.25 | 0.89 / 1.26 | 0.161-0.268 |
| rule | -  / 0.45 | 0.07 / 0.44 | 0.08 / 0.44 | 0.00 / 0.45 | 0.39 / 0.39 | 0.78 / 0.87 | 0.79 / 1.19 | 0.181-0.255 |
| elements | -  / 0.29 | 0.09 / 0.28 | 0.11 / 0.27 | 0.00 / 0.29 | 0.53 / 0.82 | 0.94 / 1.30 | 0.71 / 0.92 | 0.185-0.239 |
| features | -  / 0.63 | 0.07 / 0.59 | 0.08 / 0.61 | 0.00 / 0.63 | 0.54 / 0.55 | 0.87 / 1.02 | 0.88 / 1.19 | 0.260-0.311 |

### depth, 10 s windows: between rows / between seeds of the row (fields: range over rows)

| text | text | inside | shuffled | reordered | regimes | fields |
|---|---|---|---|---|---|---|
| seasons | -  / 0.050 | 0.028 / 0.043 | 0.054 / 0.052 | 0.047 / 0.053 | 0.055 / 0.049 | 0.002-0.002 |
| rule | -  / 0.035 | 0.031 / 0.045 | 0.045 / 0.043 | 0.051 / 0.049 | 0.047 / 0.062 | 0.002-0.002 |
| elements | -  / 0.053 | 0.033 / 0.052 | 0.049 / 0.057 | 0.057 / 0.059 | 0.063 / 0.066 | 0.002-0.002 |
| features | -  / 0.042 | 0.040 / 0.044 | 0.046 / 0.049 | 0.043 / 0.048 | 0.047 / 0.046 | 0.002-0.002 |

### fineness, 10 s windows: between rows / between seeds of the row (fields: range over rows)

| text | text | inside | shuffled | reordered | regimes | fields |
|---|---|---|---|---|---|---|
| seasons | -  / 0.78 | 0.31 / 0.86 | 0.91 / 0.91 | 1.41 / 1.81 | 1.31 / 1.69 | 0.212-0.262 |
| rule | -  / 0.77 | 0.29 / 0.75 | 0.83 / 0.87 | 1.36 / 1.77 | 1.44 / 1.83 | 0.254-0.290 |
| elements | -  / 0.69 | 0.29 / 0.65 | 0.68 / 0.86 | 1.45 / 2.05 | 1.30 / 1.66 | 0.228-0.286 |
| features | -  / 0.85 | 0.33 / 0.88 | 0.95 / 0.82 | 1.54 / 1.85 | 1.61 / 1.92 | 0.201-0.299 |

Map check (record, grid and spearman). On the grid of p and q, the four fields change depth by at most 0.006 and
fineness by at most 1.1 cycles (sentence tiles; 10 s tiles 0.006 and 1.0). Over the tiles of the run, Spearman of depth
with q is 0.94-0.96 per sentence and 0.87-0.91 in windows; of fineness with p 0.67-0.81 per sentence and 0.89-0.91 in
windows; of fineness with q 0.20-0.31 per sentence and -0.02-0.06 in windows (the pairs repeat each tile on four fields
and are not independent tiles).

## Readings (Claude)

1. The field changes depth hardly at all and fineness a little. Between fields, depth differs by 0.002-0.003, a third to
   a thirtieth of the difference between two seed indices of the same row; the prediction holds for depth. Fineness
   differs by 0.16-0.31 cycles between fields against 0.27-2.05 between seed indices: smaller in every cell, but not
   negligible (elements per sentence, 0.19 against 0.29 for the text). The field is not whole periods within a tile,
   so the spectrum keeps some of the phase (Codex review of v5); the numbers depend little on the phase, though
   not zero. This corrects reading 4 of v4: measured without the lattice's phase, changing the field is the smallest change
   of the atlas.
2. Both numbers are nearly monotone images of p and q: depth follows q, fineness follows p (and per sentence q a
   little, through gamma). They bring no information about a tile beyond its p and q; what v5 adds is the size of the
   differences in the words the eye uses (pale, coarse) and against a baseline.
3. Apart from none in depth, no change of series moves the tiles much more than a new draw of the same series does.
   Per sentence, the differences in depth of none, shuffled, reordered and regimes are 0.025-0.051, 1.1 to 2.4 times
   the text's own seed difference (0.020-0.026); shuffled, reordered and regimes are about the size of their own seed
   difference, while none is 1.9-4.4 times its own (0.009-0.014: with q = 1 in most sentences its draws hardly differ);
   after is 1.2-1.5 times the text's. In fineness reordered and
   regimes differ from the text by 0.71-0.94 per sentence and 1.30-1.61 in windows, 1.4-3.2 and 1.7-2.1 times the
   text's seed difference, but less than their own seed difference (0.87-1.30 and 1.66-2.05). The prediction ("clearly
   more than between seeds") holds for none in depth, and for reordered and regimes in fineness against the text's
   baseline only. Note the pairing: after, none, inside, shuffled
   and reordered are made from the text's gaps of the same seed index, so their difference from the text is paired,
   while the seed difference is between two independent draws; a paired change that moves the tiles as much as an
   independent draw is a large change, not a small one. Regimes is not paired.
4. Inside equals the text per sentence (0 in both numbers, as predicted) and differs in 10 s windows by a little less than the seed
   difference in depth (0.028-0.040 against 0.035-0.053) and by a third to a half of it in fineness.
5. What v5 adds to the atlas: tile by tile (the j-th tile against the j-th), the depth and fineness of this transform
   change under every change of series but the counting "none" (in depth) by about as much as under a new
   draw of the series; which tile is pale or
   coarse is not a stable mark of the series at this length. What told the series apart in v4 was the distribution
   over tiles (the spread of p and q), a different count, which v5 did not fix as an image number.

Limits: synthetic English keystrokes only; tiles are paired by index, so windows of rows of different duration compare
different parts of the text at the end (Codex review of v5, as v4); the nine seed pairs are neighbours and share
seeds; the Spearman pools four fields and weights the rows with more windows; no figure was looked at for v5 (the
tiles are v4's; v4's readings of the figures rest on three figures of one text); the prediction was Claude's, not the
researcher's.
