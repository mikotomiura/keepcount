# Notes on prototype v6e (read after the run, 2026-11-01)

Record: results/prototype_v6e/record.json (SHA-256 aca8d16a950b365c77f2f8f82eaeecec10c07e781a9bdba14bd7ff3dbbf7ee29; git commit 6c5ba84, not dirty). Conditions:
works/conditions/prototype_v6e.json, committed in cbffc5d and amended once before the first run in 6d8c1a6 (after the code
review and the Codex review of the code). Code 5e138e5, fixes 6c5ba84, both before the run.

Data: the essays of the Kaggle competition Linking Writing Processes to Writing Quality (LWP; train_logs.csv, 2,471
essays, one per person; US crowd workers, 30-minute argumentative essays; letters and digits masked as q). 200 persons
(204 ids looked at); count persons 196 in sentences (4 with fewer than 8 sentences), 200 in windows. The persons' essays
are long: 582 / 1,947 / 2,915 / 3,860 / 9,309 counted presses and 5 / 17 / 23 / 29 / 59 sentences (min / quartiles / max).
Events read 8,405,898; not counted 697,908; one decreasing down_time; 180 presses dropped for an equal time.

## The rule fixed before running

| prediction | cells | held | |
|---|---|---|---|
| P1 image stages (S1, S2, SEQ) below 0.60 | 40 | 40 | held |
| P2 shuffles at the image stages: upper end below 0.5 | 20 | 12 | not held (needs 14) |
| P3 LOC, shuffles above 0.60 | 8 | 8 | held |
| P4 R_pos / R_lag, shuffles above 0.60 | 8 / 8 | 8 / 8 | held |
| P5 inside not above 0.60 on LOC | 4 | 1 | not held |
| P5 after below 0.60 on every stage | 13 | 13 | held |

Finding 1 (the image stages do not show the operations beyond the person's own variation, the reference observable
does): **confirmed** on LWP. Its stronger form (most persons less than their own variation): **not confirmed**. The eight
cells of P2 that do not hold are all under the odd/even baseline (sentences and windows, S1 and S2, both shuffles).

## Main cells (essay, sentence unit, odd/even; share of persons with change > own, CI)

| row | S1 | S2 | LOC | R_pos | R_lag |
|---|---|---|---|---|---|
| inside | 0.00 | 0.00 | 0.75 (0.69-0.82) | 0.81 (0.75-0.87) | 0.23 (0.17-0.30) |
| shuffled | 0.51 (0.43-0.58) | 0.46 (0.39-0.53) | 0.96 (0.93-0.98) | 0.94 (0.90-0.98) | 0.91 (0.87-0.95) |
| shuffled in halves | 0.45 (0.38-0.52) | 0.46 (0.39-0.53) | 0.96 (0.93-0.98) | 0.93 (0.90-0.97) | 0.90 (0.85-0.94) |

Over all cells: shuffles at the image stages 0.14-0.52 (odd/even 0.45-0.52, halves 0.14-0.43); LOC 0.93-0.98, median
change / base 1.90-2.43; R_pos 0.91-0.95; R_lag 0.84-0.91. Inside: image 0 in sentences (by construction) and 0.04-0.14
in windows; LOC 0.64-0.75 (above 0.60 in three of four cells; windows under halves undecided, 0.56-0.72); R_pos 0.71-0.81;
R_lag 0.14-0.28.

Sensitivity (only Input and Remove/Cut events; 200 persons, 199 shared with the main persons; main cells): every share
within 0.04 of the main counts (shuffled S1 0.51 / 0.31, LOC 0.95-0.96, inside LOC 0.68-0.75).

## Readings (Claude)

1. Finding 1 holds on a second dataset under the rule fixed before running: the image stages stay below the smallest
   difference in all 40 cells, and the reference observable shows shuffling above it in all 8 cells, by the place of the
   pauses alone (R_pos) as well as by the lag-1 alone (R_lag). The persons, the logger, the task and the writers differ
   from KUPA (US crowd workers, not English learners; essays about twice as long).
2. The stronger form does not hold under the odd/even baseline: there, about half of the persons (0.45-0.52) have a
   change under shuffling larger than the difference between their odd and even tiles. Under halves it holds (0.14-0.43).
   On KUPA the odd/even baseline also gave the higher shares (v6b reading 1). So "less than the person's own variation in
   most persons" depends on how the person's own variation is cut; "not beyond the smallest difference" does not.
3. Unlike KUPA, moving the rest inside its sentence is seen by the reference observable (above 0.60 in three of four
   cells, by position rather than lag-1). On LWP inside therefore passes the positive control of the procedure, and the
   image stages show nothing of it (0 per sentence, 0.04-0.14 in windows): a case where the reference sees an operation
   that the image loses by construction. The prediction P5 (inside not above, from KUPA) did not hold.
4. Leaving out edit, paste and move events changes nothing that matters (within 0.04).

Limits: limits_written_before (one essay per person; a different logger; letters masked; key repeats not flagged; consent
inferred from the KLiCKe paper by an author of the competition, with the same procedure; LOC and its split chosen on
KUPA, fixed here before running). Readings are Claude's; the researcher has not reviewed them.

Terms: derived from the Kaggle competition data under its rules (non-commercial, academic research; no redistribution of
the data); only counts over persons are published.
