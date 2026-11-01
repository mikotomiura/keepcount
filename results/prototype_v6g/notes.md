# Notes on prototype v6g (read after the run, 2026-11-01)

Record: results/prototype_v6g/record.json (SHA-256 090f455b56055ddf5eb0eab0bfc7c5675292baa1e4246ba8d4cd193915a3e37a;
git commit 7866dd3, not dirty). Conditions: works/conditions/prototype_v6g.json, committed in 652ad2e before any code
(after codex-consult and the researcher's check). Code 0a2d05d, fixes 7866dd3 (after the code review and the Codex
review), both before the run. Run time about 11 minutes.

Persons: v6f's count persons in the sentence unit (KUPA essay 178, KUPA copy 200, LWP 196); the persons left out
matched v6f's (KUPA essay 22, LWP 4 with fewer than 8 tiles). Yardstick: each sentence tile re-made from its own gaps
drawn with replacement, 20 draws per person; own = mean over the 20 draws of the comparison change > base.

How much a draw moves q (draw 0, median over persons of the median |change of q| per tile): KUPA essay 0.07, LWP 0.06,
KUPA copy 0.00 (copying has few rests).

## The rule fixed before running (main cells: KUPA essay and LWP, sentence unit, IMG_pq and IMG_df)

| | cells | held | |
|---|---|---|---|
| (a) positive control: gathered above 0.60 | 4 | 4 | met (0.91-0.99, lower ends 0.89-0.98) |
| (b) symmetry: redrawn within 0.45-0.55, interval within 0.40-0.60 | 4 | 4 | passed (0.49-0.50, intervals 0.48-0.52) |
| (c) shuffles above 0.60 | 8 | 8 | **shows the shuffles beyond its sampling variation** |

Shuffled 0.94-0.98 (lower ends 0.92-0.97), shuffled in halves 0.93-0.98 (0.91-0.97); median change over base 1.66-1.95.
Each single draw gives 0.89-0.99, so the result does not rest on the average over draws.

Predictions: P1 held; P2 not held (rests x2 below 0.60 everywhere at IMG: 0.12-0.20; x3 0.55-0.67, one cell above, two
undecided, one below); P3 held; P4 held; P5 not held (below).

P5 (inside reads 0 at every stage) holds in KUPA essay and LWP (0 at every stage) but not in KUPA copy (IMG 0.001, S1
0.016, S2 0.013). In copy, 1 person (IMG) and 10 and 8 persons (S1, S2) have a base of exactly 0 in some draws (every
tile's gaps equal or its summary unmoved by a redraw), and a tie counts 1/2 by the conditions: the share of tied
(person, draw) pairs, 0.03 at S1, times 1/2 is the 0.016. This is the case the Codex review foresaw. Separately,
inside_identical_to_text_all_persons is false in KUPA: a check outside the record (scratchpad, 40 persons) found the
largest difference between the inside row and text at 3e-16 (IMG, copy) and 0 in the essay, the rounding of the
non-integer KUPA times rebuilt by cumsum, not a change of a tile. No base fell between 0 and 1e-9 (the near-zero count
equals the exact-zero count in every cell), so the rounding did not touch the tie rule.

## Main cells (share of persons with change > own, CI)

| row | KUPA essay IMG_pq | IMG_df | LWP IMG_pq | IMG_df |
|---|---|---|---|---|
| redrawn | 0.49 (0.48-0.51) | 0.50 (0.48-0.51) | 0.50 (0.49-0.52) | 0.50 (0.49-0.51) |
| gathered | 0.92 (0.89-0.94) | 0.91 (0.89-0.94) | 0.99 (0.98-1.00) | 0.99 (0.97-1.00) |
| rests x1.5 | 0.01 | 0.01 | 0.01 | 0.00 |
| rests x2 | 0.20 (0.17-0.23) | 0.16 (0.14-0.19) | 0.17 (0.15-0.20) | 0.12 (0.10-0.14) |
| rests x3 | 0.62 (0.58-0.66) | 0.55 (0.51-0.60) | 0.67 (0.63-0.71) | 0.60 (0.56-0.63) |
| inside | 0.00 | 0.00 | 0.00 | 0.00 |
| shuffled | 0.95 (0.93-0.97) | 0.94 (0.92-0.96) | 0.98 (0.97-0.99) | 0.98 (0.96-0.99) |
| shuffled in halves | 0.93 (0.91-0.95) | 0.93 (0.91-0.95) | 0.98 (0.97-0.99) | 0.98 (0.97-0.99) |

S1 and S2 against the same yardstick (not in the rule): gathered 0.75-0.90 (above), rests x3 0.62-0.81, the shuffles
0.35-0.48 (upper ends 0.39-0.53: below 0.60), redrawn 0.48-0.51.

## Secondary cells (not in the rule)

- KUPA copy (sentences): the shuffles read 0.94-0.97 at IMG (above); gathered only 0.52-0.55 (undecided or below),
  as in v6f: copying has few rests to gather.
- Sensitivity, the gap before each sentence kept in place (main cells, IMG): the shuffles 0.97-0.99, gathered
  0.95-0.99, rests x3 0.68-0.78, redrawn 0.49-0.51: the same reading, with a narrower base.

## Readings (Claude)

1. Against the sampling variation of a tile at its place, the tile sequence shows both the positive control and the
   two shuffles in almost every person, in both datasets. A shuffled tile takes its gaps from the whole record, so the
   distribution of speed and rest within the tile changes beyond what drawing the tile's own gaps again produces. So
   the image (as a sequence of tiles) carries which sentences hold the long and short gaps.
2. This is not the question v6f asked. v6f compared with the person's variation across two parts of the record, which
   also differ in content; there the positive control failed at IMG and the shuffles were not evaluable. v6g's base is
   narrower by design (limits_written_before). Both are true together: the tile sequence moves under shuffling more
   than a redraw of each tile moves it, and less than, or about as much as, two parts of the same record differ.
3. The summaries S1 and S2 do not show the shuffles even against this narrow base (0.35-0.48), while they show
   gathered. The loss of the place of the pauses at S1 and S2 is therefore not only a matter of a coarse yardstick.
4. Within a sentence nothing changes (inside 0 by construction); below the sentence the image has no place.
5. Tripling every rest is near 0.60 at IMG and doubling is not seen: the tile sequence is sensitive to where the
   rests are, less to how long they all are.

Limits: limits_written_before (the gaps of a tile treated as exchangeable; a narrower base than the person's own
variation; chosen after v6f's failed control; IMG is a distance between numbers, not a viewer). Many correlated cells
read descriptively. Five-number summaries hold the minimum and maximum, which are single persons' values (as v6e).
Readings are Claude's; the researcher has not reviewed them.
