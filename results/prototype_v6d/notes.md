# Notes on prototype v6d (read after the run, 2026-11-01)

Record: results/prototype_v6d/record.json (SHA-256 9695438b24392e766fe7bb8e2d50d6a3cd939f24c8903359ebf8c263bd2a9b1d; git commit a29ba2e, not dirty). Conditions:
works/conditions/prototype_v6d.json, committed in 1b70dbe and amended once before the first run in a26a10b (after the code
review and the Codex review of the code). Code 3770a18, fixes a29ba2e, both before the run.

Persons, parts, rows and baselines are v6b's (ids matched v6b's and v6's SHA-256; the code matched the SHA-256 recorded
in both records). Check: LOC counted again equals v6b's record (persons, share_vs_own and median_change_over_base) in every
cell, baseline and row. Status: a decomposition of v6b's result, chosen after seeing v6b; not a new confirmation.

## LOC split: share of persons with change > own (CI; v6's two-axis reading)

Main cell, KUPA essay, sentence unit, odd/even baseline:

| row | LOC (v6b) | R_pos (six numbers of position) | R_lag (lag-1 alone) |
|---|---|---|---|
| inside | 0.60 (0.53-0.68) undecided | 0.66 (0.58-0.73) undecided | 0.29 (0.22-0.36) below 0.60 |
| shuffled | 0.93 (0.88-0.97) above | 0.90 (0.85-0.94) above | 0.81 (0.74-0.87) above |
| shuffled in halves | 0.95 (0.91-0.98) above | 0.93 (0.88-0.97) above | 0.85 (0.79-0.90) above |

Over all cells (essay and copy, sentences and 10 s windows, both baselines):
- shuffled and shuffled in halves: R_pos 0.90-0.96, above 0.60 in every cell; R_lag 0.73-0.85, above 0.60 in every cell.
- inside: R_pos 0.60-0.70 (above 0.60 in one cell, copy sentences odd/even 0.70 (0.64-0.76); detectable in all eight;
  undecided in seven); R_lag 0.24-0.31, below 0.60 in every cell.
- after: R_pos 0.01-0.15, R_lag 0.01-0.24, below 0.60.

## Lengths of the records (min / 25th / median / 75th / max over persons)

| persons, part | presses | sentences | full 10 s windows |
|---|---|---|---|
| v6 KUPA copy | 1356 / 1690 / 1738 / 1794 / 2182 | 17 / 22 / 23 / 24 / 26 | 21 / 35 / 47 / 61 / 152 |
| v6 KUPA essay | 851 / 1116 / 1423 / 1751 / 3326 | 5 / 9 / 12 / 17 / 39 | 13 / 42 / 60 / 84 / 378 |
| v6 Aalto sentences 1-7 | 195 / 286 / 319 / 350 / 433 | 7 | 3 / 4 / 6 / 9 / 36 |
| v6 Aalto sentences 8-15 | 237 / 325 / 364 / 398 / 612 | 8 | 3 / 5 / 7 / 10 / 39 |
| v6b KUPA copy | 1478 / 1682 / 1733 / 1796 / 2091 | 18 / 22 / 23 / 23 / 31 | 18 / 37 / 48 / 62 / 132 |
| v6b KUPA essay | 843 / 1100 / 1444 / 1830 / 4090 | 5 / 9 / 13 / 17 / 37 | 16 / 46 / 60 / 91 / 332 |

In v6b's count-1 persons each half holds 4-19 sentence tiles (median 6.5-7) or 8-166 window tiles (median 30) and at
least 247 adjacent pairs of held gaps; no person has fewer than 3 pairs in a half (R_lag is never 0 by the rule of
fewer than 3 pairs).

## Readings (Claude)

1. Shuffling exceeds the person's own variation at the reference stage without the lag-1: the six numbers of position
   alone show it in every cell (0.90-0.96). So v6b's reading 2 does not rest on the order statistic: what the image
   lacks includes which kind of gap (sentence, subsentence, word, within word) carries the long and short gaps.
2. The lag-1 alone also shows shuffling (0.73-0.85), less clearly than position. Its own difference between halves is
   small enough against the drop that the prediction "below or undecided" did not hold: with 247-2,471 adjacent pairs
   per half, the lag-1 of a half is estimated well.
3. Moving the rest inside its sentence is seen weakly by position (0.60-0.70) and not by the lag-1 (0.24-0.31): it moves
   one gap per sentence between labels, and changes the order little.
4. The records are long in KUPA (the essay median 1,423-1,444 presses, 12-13 sentences; the copy about 1,735 presses,
   23 sentences) and short in Aalto (319 and 364 presses per part, 6-7 windows). The counts of v6 and v6b are for
   records of these lengths; the dependence on length was not counted.
5. Predictions written before running: (a) R_pos above 0.60 for the shuffles in every cell held; inside about as LOC,
   undecided in most cells, held (seven of eight). (b) R_lag below or undecided for the shuffles did not hold (above in
   every cell); R_lag inside below 0.60 held. (c) KUPA essay median 10-15 sentences held (12-13), copy about 22 held
   (23), Aalto 7 and 8 by design held.

Limits: as v6b and limits_written_before (one dataset, LOC chosen after the exploration, the split chosen after seeing
v6b, the six numbers not independent). Readings are Claude's; the researcher has not reviewed them.

Licences: counts derived from KUPA-KEYS (Velentzas et al., LREC-COLING 2024) are under CC BY-NC-SA 4.0; those from the
136M Keystrokes dataset (Dhakal, Feit, Kristensson, Oulasvirta, CHI 2018) are for non-commercial use with attribution
to the authors.
