# Notes on prototype v6j (read after the run, 2026-11-01)

Record: results/prototype_v6j/record.json (SHA-256 1ba6b70a2466d8e75b273e9d9c9b6026ff9df70f67dd6f8461b84b4a32fc4356;
git commit f3abbac). Conditions: works/conditions/prototype_v6j.json, committed in d46e4b4 before any code (after
codex-consult and the researcher's check). Code 6b740fa and 30143c9, fixes f3abbac (after the Codex review and the code
review), all before the run. Run time about 16 minutes.

Checks passed: gathered against H_raw equalled v6i's record exactly (mean r, interval, tails, R) in both cells; SEQ,
IMG5_pq, IMG5_df, IMG5_pq_scrambled and DTW_pq under the main standardisation equalled v6i's record with v6i's keys in
every cell; S1 and S2 under the registration scale equalled v6's record in all six direction-units. The rests meant to
move were all moved (left: 0 for every person and control).

## (a) Positive controls under the four nulls (mean r_q, 95% interval)

| control | null | KUPA essay | LWP |
|---|---|---|---|
| random_half (rule) | H raw | 0.95 (0.93-0.97) | 0.96 (0.93-0.98) |
| | H detrended | 0.95 (0.92-0.97) | 0.95 (0.92-0.97) |
| | C raw | 0.87 (0.84-0.90) | 0.91 (0.88-0.94) |
| | C detrended | 0.84 (0.81-0.88) | 0.88 (0.84-0.91) |
| random_half_weak | H raw / H detr / C raw / C detr | 0.90 / 0.84 / 0.75 / 0.63 (0.58-0.68) | 0.93 / 0.79 / 0.79 / 0.54 (0.49-0.59) |
| gathered | H raw / H detr / C raw / C detr | 0.96 / 0.95 / 0.89 / 0.88 | 0.96 / 0.97 / 0.92 / 0.94 |
| middle_block | H raw / H detr / C raw / C detr | 0.97 / 0.98 / 0.65 / 0.85 | 0.98 / 0.98 / 0.65 / 0.89 |

Rule fixed before running: the rule's control (random_half) is seen under all four nulls in both cells, so v6i's
reading of rest under every null is "sensitive to a concentration in particular sentences of this strength".
Predictions: P1 and P2 held; P3 did not (middle_block was seen under H detrended too, since a block in the middle is
not a straight line).

## (b) DTW with scrambled tiles; the registration scale (KUPA, sentence tiles)

| | copy -> essay | essay -> copy |
|---|---|---|
| DTW (main) | 0.79 | 0.79 |
| DTW scrambled (mean of 5; range) | 0.79 (0.784-0.788) | 0.79 (0.788-0.794) |
| DTW - scrambled | 0.001 (-0.008 to 0.012) | 0.000 (-0.010 to 0.010) |
| S1 / IMG5 / DTW, registration scale | 0.63 / 0.66 / 0.75 | 0.64 / 0.67 / 0.74 |
| IMG5 - S1, registration | 0.030 (0.003-0.056) | 0.033 (0.004-0.065) |
| DTW - S1, registration | 0.119 (0.083-0.153) | 0.095 (0.063-0.128) |
| main - registration: S1 / IMG5 / DTW | 0.080 / 0.085 / 0.036 | 0.070 / 0.073 / 0.056 |

136M sentences (within a session): DTW 0.85, scrambled 0.84, the registration scale changes nothing (as v6 for S1).
Predictions: P4 held; P5 did not (DTW falls by only 0.04-0.06 under the registration scale, IMG5 by 0.07-0.09).

## Readings (Claude)

1. The reading of rest in v6i stands under the nulls that allow for drift: those comparisons see rests gathered into a
   random half of the sentences (0.84-0.96), so v6i's "not confirmed" at rest after drift is allowed for is not a
   blindness of those nulls to such a concentration. They see a half-strength concentration less well (0.54-0.93), and
   in LWP under circular shifts with the line removed not at all; weaker structure is not ruled out.
2. The gain of DTW over the summaries comes from the distribution of the tiles, not their order: scrambling the tiles
   changes DTW by 0.001 or less. Under the registration scale, which assumes nothing of the evaluated set, DTW still
   ranks above S1 by 0.10-0.12 (0.74-0.75 against 0.63-0.64), while IMG5's gain is small (0.03).

Limits: limits_written_before in the conditions (the controls are strong; seeing them shows sensitivity to these
concentrations at these strengths; a scrambled DTW is not a distance without order; a change of scale also changes
weights; status d). Readings are Claude's; the researcher has not reviewed them.
