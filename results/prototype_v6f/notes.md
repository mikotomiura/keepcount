# Notes on prototype v6f (read after the run, 2026-11-01)

Record: results/prototype_v6f/record.json (SHA-256 cbca3ad1c579ea1495a9b6faf5c78553747ad9560324ba51333800cf93d7dbfb;
git commit 9c95af7, not dirty). Conditions: works/conditions/prototype_v6f.json, committed in 3ef9868 before any code
(after codex-consult and the researcher's check). Code a553c20, fixes 9c95af7 (after the code review and the Codex
review), both before the run. Run time about 6 minutes.

Persons: v6b's 200 KUPA persons and v6e's 200 LWP persons, checked by the SHA-256 of their ids. Count persons as v6b and
v6e (KUPA essay sentences 178, LWP sentences 196, every other cell 200). Every kept count of v6b and v6e was reproduced
exactly before the new counts (43 + 27 + 43 + 27 + 59 + 39 = 238 counts compared). The reference quarters use the count
persons with at least 16 sentences: 64 in KUPA essay, 161 in LWP.

How much the positive controls move q (sentence unit, median over persons of the median |change of q| per tile, and the
tiles changed): gathered 0.19 (KUPA essay) and 0.23 (LWP) in a median of 11 and 20 tiles; rests x1.5 0.08, x2 0.14,
x3 0.22 in the same tiles. No rest was left unmoved by gathered (left 0 in every person).

## The rule fixed before running (main cells: KUPA essay and LWP, sentence unit)

| | cells | held | |
|---|---|---|---|
| (a) gathered above 0.60 at the image observables | 18 | 8 | not met (needs 15 for partly) |
| (b) inside and the shuffles below 0.60 at IMG_pq and IMG_df | 24 | 22 | (not read: (a) not met) |

Finding 1 extended to the tile sequence: **not evaluable**. The 8 cells of (a) that hold are S1 and S2 under both
baselines in both datasets; the 10 that do not are IMG_pq and IMG_df in every cell and SEQ under halves. So the positive
control is met by the image summaries S1 and S2 and not by the tile sequence or the order of the tiles. The two cells of
(b) that do not hold are LWP, odd/even, shuffled at IMG_pq and IMG_df (undecided, 0.56).

Predictions: P1 not held (above); P2 not held (rests x2 and x3 above 0.60 in 1 of 32 cells); P3 held (halves 12 of 12
below; odd/even 10 of 12 below); P4 held (share0 within 0.40-0.60 in 16 of 16).

## Main cells (share of persons with change > own, CI)

KUPA essay, sentences, odd/even:

| row | S1 | S2 | IMG_pq | IMG_df |
|---|---|---|---|---|
| gathered | 0.71 (0.65-0.77) | 0.72 (0.65-0.78) | 0.46 (0.38-0.52) | 0.47 (0.39-0.54) |
| rests x1.5 | 0.07 | 0.07 | 0.00 | 0.01 |
| rests x2 | 0.17 (0.12-0.22) | 0.20 (0.14-0.26) | 0.01 | 0.01 |
| rests x3 | 0.46 (0.38-0.53) | 0.48 (0.41-0.56) | 0.15 (0.10-0.20) | 0.11 (0.07-0.16) |
| inside | | | 0.00 | 0.00 |
| shuffled | | | 0.40 (0.33-0.48) | 0.43 (0.35-0.49) |
| shuffled in halves | | | 0.35 (0.28-0.42) | 0.36 (0.29-0.43) |

LWP, sentences, odd/even:

| row | S1 | S2 | IMG_pq | IMG_df |
|---|---|---|---|---|
| gathered | 0.82 (0.77-0.87) | 0.85 (0.80-0.89) | 0.63 (0.57-0.70) | 0.64 (0.57-0.70) |
| rests x1.5 | 0.12 | 0.13 | 0.00 | 0.00 |
| rests x2 | 0.35 (0.28-0.42) | 0.33 (0.26-0.39) | 0.00 | 0.00 |
| rests x3 | 0.68 (0.62-0.74) | 0.62 (0.55-0.69) | 0.04 (0.02-0.07) | 0.03 (0.01-0.06) |
| inside | | | 0.00 | 0.00 |
| shuffled | | | 0.56 (0.49-0.62) | 0.56 (0.48-0.63) |
| shuffled in halves | | | 0.49 (0.43-0.56) | 0.49 (0.43-0.56) |

Under halves (sentences): gathered S1 0.71-0.72, S2 0.73-0.75, SEQ 0.27-0.29, IMG 0.44-0.51; shuffles at IMG 0.29-0.38.
(Gathered's lower ends: LWP odd/even IMG 0.57, below 0.60, so not 'above'.)

## Secondary cells (not in the rule)

- 10 s windows, odd/even: the shuffles read **above 0.60 at IMG** in KUPA essay (0.79-0.80, lower ends 0.72-0.74), in LWP
  (0.98-0.99, lower ends 0.96-0.98); in KUPA copy they are undecided (0.58-0.64, lower ends 0.52-0.57). Gathered, which is not a
  clean control in windows (the labels are not kept after the cut), reads 0.94 at IMG in LWP and 0.61-0.64 in KUPA essay.
  Under halves the shuffles read 0.15-0.28 at IMG. At S1 the shuffles stay 0.41-0.52 under odd/even in windows.
- KUPA copy, sentences: gathered reads only 0.17-0.26 at S1 and S2 (copying has few rests: the control moves little).
- Gathered at the stage of times: S0 0.01-0.24 (the whole-series gaps are the same by construction), LOC 0.06-0.50.

## Reference quarters (outside the rule; main cells, persons with at least 16 sentences)

share0 (no operation, quarter against quarter by the same formula) is 0.45-0.58 at every stage and baseline (CIs about
+-0.12 in KUPA, 64 persons; +-0.08 in LWP). Against it, the shuffles at S1 and IMG lie below share0 in KUPA (differences
-0.11 to -0.23). In LWP they lie below it under halves (-0.21 to -0.29), within 0.04 of it at S1 under odd/even, and above it at IMG_pq under odd/even (shuffled +0.12, 0.03-0.21; shuffled in halves +0.09, -0.01 to 0.18). The positive
controls at S1: gathered +0.13 to +0.31 above share0; rests x3 -0.06 to +0.24.

## Readings (Claude)

1. The positive control of the procedure fails at the tile sequence and at the order of the tiles. Moving q by about
   0.2 in most tiles (gathered, rests x3) is not seen beyond the person's own variation at IMG in sentences (gathered 0.44-0.64, rests x3 0.03-0.15),
   because neighbouring sentence tiles (or tiles half a record apart) already differ by more than that. By the rule
   fixed before running, whether the image itself loses the place of the pauses is **not evaluable** in sentences: the
   absence of the shuffles at IMG (0.29-0.56) cannot be read as a loss when a change made to show is also absent.
2. The image summaries S1 and S2 pass the strong control in all eight main cells (0.71-0.85) but not the graded one:
   tripling every rest is seen in 0.41-0.68 of persons, doubling in 0.13-0.35. The shuffles move S1 about as often as
   doubling or tripling the rests (0.24-0.51 in the sentence unit of KUPA essay and LWP, v6b and v6e). So the own-variation yardstick is coarse: at S1 and S2 it
   detects a strong move of the rests and misses moderate ones. Finding 1 holds for S1 and S2 in cells where they are
   sensitive, but "does not show beyond the smallest difference" must be read with that sensitivity beside it.
3. In 10 s windows with the odd/even baseline, the tile sequence shows the shuffles beyond the smallest difference
   (KUPA essay 0.80, LWP 0.99): neighbouring windows are alike, and shuffling breaks that. This is secondary (windows are
   not in the rule and their control is not clean), but it contradicts a general statement that the image does not show
   shuffling. What the image shows depends on the unit, the baseline and the observable.
4. The quarter reference is near 0.5 (0.45-0.58), so the share of 0.5 is close to what the same formula gives without
   any operation; it puts the shuffles below or near it except at LWP odd/even IMG_pq, where shuffled is 0.12 above it.

Limits: limits_written_before (the controls are operations on the times made to move q; IMG pairs tiles by position and
is not a model of a viewer; the quarters do not calibrate an operation's null; many correlated cells read
descriptively). Five-number summaries of the tallies hold the minimum and maximum, which are single persons' values
(as v6e). Readings are Claude's; the researcher has not reviewed them.
