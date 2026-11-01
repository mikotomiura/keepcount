# Notes on prototype v6h (read after the run, 2026-11-01)

Record: results/prototype_v6h/record.json (SHA-256 7f13b6730cae88789b5028bec27bccefbef9d4dc6025edda5e5b0fba5839ed1f;
git commit d12df8b, not dirty). Conditions: works/conditions/prototype_v6h.json, committed in 713cdf7 before any code
(after codex-consult and the researcher's check). Code bec9df0, fixes d12df8b (after the code review and the Codex
review), both before the run. Run time about 14 minutes.

Persons: v6g's count persons, KUPA essay 178 and LWP 196 (left out as v6g: 22 and 4). The counts against text of
shuffled and gathered at IMG_pq and IMG_df equal v6g's (checked in the run), and the redraws of text are v6g's. No
operated series equalled the original (0 persons on text, 0 of 890 and 980 person-H pairs). No base was 0 except at
IMG_p (4 and 5 persons on text, 13 and 20 on H: tiles whose short gaps are all equal), and none lay between 0 and
1e-9 apart from those.

share_text / share_H are means over persons of per-person win rates (20 redraws); delta = own_text - own_H per person,
with paired bootstrap intervals.

## The rule fixed before running

(Q1) shuffled, delta at IMG_pq and IMG_df: KUPA essay 0.020 (0.001 to 0.039) and 0.021 (-0.001 to 0.041); LWP 0.016
(0.001 to 0.030) and 0.018 (0.003 to 0.032). Three cells above 0, one contains 0: **mixed**.
(Q2) rests_moved at IMG_q: share_text KUPA 0.87 (0.84-0.91), LWP 0.96 (0.94-0.97); delta KUPA -0.020 (-0.055 to
0.013), LWP 0.025 (0.006 to 0.044): **mixed** (held in LWP, not in KUPA).

Predictions: P1 held (share_H above 0.5: 0.92-0.96 at IMG_pq and IMG_df); P2 not held; P3 not held; P4 held
(short_shuffled at IMG_q 0.00); P5 held (rests_swapped below rests_moved at IMG_q).

## Main cells

| row | stage | KUPA essay: text / H / delta | LWP: text / H / delta |
|---|---|---|---|
| shuffled | IMG_pq | 0.95 / 0.93 / 0.020 (0.001, 0.039) | 0.98 / 0.96 / 0.016 (0.001, 0.030) |
| shuffled | IMG_df | 0.94 / 0.92 / 0.021 (-0.001, 0.041) | 0.98 / 0.96 / 0.018 (0.003, 0.032) |
| shuffled | IMG_p | 0.79 / 0.71 / 0.085 (0.045, 0.124) | 0.85 / 0.73 / 0.116 (0.078, 0.151) |
| shuffled | IMG_q | 0.94 / 0.93 / 0.003 (-0.019, 0.025) | 0.98 / 0.96 / 0.011 (-0.005, 0.025) |
| rests_moved | IMG_q | 0.87 / 0.89 / -0.020 (-0.055, 0.013) | 0.96 / 0.93 / 0.025 (0.006, 0.044) |
| rests_swapped | IMG_q | 0.37 / 0.38 / -0.015 (-0.049, 0.018) | 0.62 / 0.59 / 0.030 (-0.010, 0.066) |
| short_shuffled | IMG_q | 0.00 / 0.00 / 0.001 | 0.00 / 0.00 / 0.001 |
| short_shuffled | IMG_p | 0.79 / 0.70 / 0.091 (0.052, 0.130) | 0.87 / 0.73 / 0.148 (0.119, 0.177) |
| tiles_permuted | IMG_pq | 0.93 / 0.92 / 0.009 (-0.015, 0.033) | 0.98 / 0.96 / 0.020 (0.007, 0.033) |
| tiles_permuted | IMG_p | 0.78 / 0.67 / 0.109 (0.062, 0.154) | 0.88 / 0.69 / 0.189 (0.152, 0.225) |
| tiles_permuted | IMG_q | 0.92 / 0.93 / -0.016 (-0.043, 0.011) | 0.96 / 0.96 / 0.007 (-0.009, 0.023) |
| gathered | IMG_q | 0.96 (text only) | 0.99 (text only) |

Spread over the 5 homogeneous series (share_H by h): within about 0.02-0.05 of each other in every cell. Medians at
shuffled IMG_pq: change 1.26 (text) and 1.16 (H) against a base of 0.70 and 0.68 (KUPA); 1.30 and 1.17 against 0.66
and 0.67 (LWP): the difference comes mainly from the change, not the base.

## Readings (Claude)

1. Most of v6g's reading is the yardstick's own: with exchangeable gaps, shuffling already moves the tile sequence
   beyond each tile's redraw in 0.92-0.96 (H). The real series add 0.02 (three of four cells above 0).
2. At q (the rests), the real series add nothing that can be confirmed under shuffling (delta 0.003 and 0.011,
   intervals containing 0), or under reordering the sentence tiles. Moving the rests to other sentences adds 0.025 in
   LWP and nothing confirmed in KUPA. So which sentences hold the long gaps is carried by the tile sequence no more
   than exchangeable gaps would carry it in KUPA, and only slightly more in LWP. The earlier statement that the tile
   sequence "carries which sentences hold the long gaps" is not supported beyond the construction.
3. At p (the speed), the real series add 0.09-0.19 under shuffling, shuffling the short gaps and reordering the
   tiles: the sentences of a record differ in speed more than exchangeable gaps would make them (or the speed drifts
   along the record; H removes both, limits_written_before).
4. The operations behave as built: short_shuffled leaves q's sequence (0.00), rests_swapped moves it less than
   rests_moved (0.37 and 0.62 against 0.87 and 0.96), gathered moves q (0.96-0.99) and hardly p.

Limits: limits_written_before (the yardstick treats a tile's gaps as exchangeable; H removes drift as well as
differences between sentences; rests_moved also moves short gaps between tiles; share and delta are win rates, not
effect sizes, with a ceiling near 1; status d on the same persons). Readings are Claude's; the researcher has not
reviewed them.
