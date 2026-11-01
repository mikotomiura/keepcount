# Notes on prototype v6i (read after the run, 2026-11-01)

Record: results/prototype_v6i/record.json (SHA-256 65d4d8efbe779207279c38b08846788b4b29f7a28a8a69d6f2baf124fbd91cdb;
git commit 3cf818d, not dirty). Conditions: works/conditions/prototype_v6i.json, committed in 94ceb8c before any code
(after codex-consult and the researcher's check). Code b148ab3, fixes 3cf818d (after the Codex review
and the code review), all before the run. Run time about 8 minutes.

Persons: (a) v6h's count persons, KUPA essay 178 and LWP 196 (left out as v6h: 22 and 4); p and q without the relief
equalled v6.Tiles' exactly. (b) v6's 200 KUPA and 200 Aalto persons; S0, S1 and S2 equalled v6c's record (top1 and
rank_auc within 1e-12, Rank-n with intervals and EER exactly). No degenerate person and no undefined R in (a); in (b) only
NT in Aalto sentences was constant (7 and 8 sentences), as stated in the conditions.

## (a) The dispersion over sentences against the person's own null (raw statistic, null H, 1000 series per person)

| cell | mean r_q (CI) | median R_q | mean r_p (CI) | median R_p | r_p - r_q (CI) | gathered r_q |
|---|---|---|---|---|---|---|
| KUPA essay | 0.53 (0.49-0.58) | 1.01 | 0.69 (0.65-0.74) | 1.26 | 0.16 (0.09-0.22) | 0.96 (0.94-0.97) |
| LWP | 0.60 (0.55-0.65) | 1.05 | 0.86 (0.83-0.89) | 1.63 | 0.26 (0.20-0.32) | 0.96 (0.94-0.98) |

Rule fixed before running: positive control met in both cells; A1 (q) **mixed** (KUPA contains 0.5, LWP above 0.5);
A2 (p) above exchangeable gaps in both cells; A3 p above q in both cells. Predictions P1-P4 and P9 held.

Beside the rule: detrended (a least-squares line on the tile index removed) r_q 0.49 (0.45-0.54) and 0.51 (0.46-0.55),
r_p 0.67 and 0.85. Under circular shifts C (local runs and drift kept) r_q 0.50 (0.45-0.54) and 0.48 (0.44-0.53), r_p
0.60 (0.56-0.65) and 0.60 (0.56-0.65). M_raw as raw (q 0.53 and 0.58, p 0.69 and 0.88). Tail shares of r_q >= 0.975:
0.10 and 0.13 (0.025 expected under exchangeability), of r_p: 0.17 and 0.48.

## (b) Identification at the stages nearest the image (KUPA copy -> essay, sentence unit; N = 200)

| stage | rank AUC (CI) | rank-1 | rank-20 | EER |
|---|---|---|---|---|
| S1 | 0.71 (0.68-0.75) | 0.010 | 0.30 | 0.36 |
| S2 | 0.69 (0.65-0.73) | 0.020 | 0.29 | 0.36 |
| SEQ | 0.55 (0.51-0.59) | 0.010 | 0.15 | 0.46 |
| IMG5_pq | 0.75 (0.71-0.78) | 0.025 | 0.32 | 0.34 |
| IMG5_df | 0.74 (0.71-0.77) | 0.020 | 0.28 | 0.33 |
| IMG5_centred | 0.51 (0.47-0.55) | 0.000 | 0.08 | 0.50 |
| IMG5_pq_scrambled | 0.75 (0.72-0.78) | 0.030 | 0.34 | 0.33 |
| DTW_pq | 0.79 (0.76-0.81) | 0.025 | 0.36 | 0.31 |
| NT | 0.49 (0.45-0.53) | 0.005 | 0.10 | 0.51 |
| IMG5_pq_NT | 0.72 (0.69-0.76) | 0.010 | 0.27 | 0.35 |

Paired (resamples of the stage with the larger index): IMG5_pq - S1 0.035 (-0.004 to 0.071); IMG5_df - S2 0.052 (0.014
to 0.090); SEQ - S1 -0.160; IMG5_pq - scrambled 0.001 (-0.019 to 0.021); DTW_pq - S1 0.074 (0.038 to 0.108).
Other cells: essay -> copy sentences the same picture (IMG5_pq 0.75, DTW 0.79, scrambled 0.75, centred 0.51, NT 0.48).
Windows: IMG5_pq 0.69-0.70, about S1 (0.68-0.71); NT 0.58 (0.54-0.62) and 0.60 (0.56-0.64). Aalto sentences (within a
session): IMG5_pq 0.80, DTW 0.85 (S1 0.77); Aalto windows NT 0.76 (0.74-0.79).
Predictions: P6, P8 and P10 held; P5 (IMG5 within 0.03 of S1 and S2) did not (0.035 and 0.052 above); P7 did not (NT
in KUPA windows 0.58-0.60, lower ends 0.54-0.56, not above 0.60); P11 did not (DTW above S1, not below).

## Readings (Claude)

1. The ceiling of v6h is removed: against each person's own null, where the room is the same for p and q (mean 0.5
   under exchangeability), the rests are not dispersed over sentences beyond exchangeable gaps in KUPA, and in LWP they
   are (0.60), but that excess disappears when a linear drift along the record is removed (0.51) and under circular
   shifts that keep local runs and drift (0.48). So what LWP adds at rest is a slow change along the essay, not rests
   placed in particular sentences. The speed differs between sentences beyond exchangeable gaps in both datasets, after
   detrending, and (less) under circular shifts (0.60): the contrast of speed and rest is not an artefact of the room of
   the yardstick.
2. Identification at the stages nearest the image is a little higher than at the summaries (rank AUC 0.75 for the tile
   sequence aligned in fifths, 0.79 with DTW, against 0.71 at S1), with rank-1 still 2-3% among 200 (chance 0.5%). The
   gain comes from the distribution of the tiles, not their order: discarding the order changes nothing (0.001), the
   sequence with the level removed and SEQ are near chance (0.51, 0.55).
3. The number of sentence tiles leaks nothing across tasks in KUPA (copying is of a fixed text; 0.49), the number of
   windows a little (0.58-0.60), and within a session in 136M Keystrokes the number of windows identifies as well as S1
   (0.76): the length of a record is part of what an image of windows gives away.
4. P11 failed in the direction that matters: DTW, a stronger attack on the sequence, identifies more than the summaries.
   The section 6 statement that the rates are achieved values of one attack, not bounds, is borne out.

Limits: limits_written_before in the conditions (H removes local runs, drift and concentration together; the linear
detrend removes only a linear drift; status d for (a); one alignment by fifths, DTW in sentences only, nearest
neighbour, no attack on pixels; exploratory, many correlated cells). DTW's order was not separated from its distribution
(no scrambled DTW was prespecified; not added after the run). Readings are Claude's; the researcher has not reviewed
them.
