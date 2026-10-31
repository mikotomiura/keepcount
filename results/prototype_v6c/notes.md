# Notes on prototype v6c (read after the run, 2026-11-01)

Record: results/prototype_v6c/record.json (SHA-256 85e82a489fa6da959ca57a338deae96fb50ab092ef43b039b2c69b4dfc3326d5;
git commit f6ff61f, not dirty). Conditions: works/conditions/prototype_v6c.json, committed in c0927ce and amended once
before the first run in 6070855 (after the code review and the Codex review of the code). Code 1a33f24, fixes f6ff61f,
both before the run.

Persons, stages, units and directions are v6's (the ids matched v6's SHA-256; the code matched the SHA-256 recorded in
v6's record). Check: at N = 200, top1 and rank_auc of every direction, unit and stage equal v6's record within 1e-12.

## Rank-n rates and EER (N = 200; chance n / 200: rank-1 0.005, rank-5 0.025, rank-10 0.05, rank-20 0.10)

Main cell, KUPA copy -> essay, sentence unit:

| stage | rank-1 | rank-5 | rank-10 | rank-20 (CI) | EER |
|---|---|---|---|---|---|
| S0 times | 0.160 | 0.355 | 0.495 | 0.670 (0.60-0.73) | 0.205 |
| S1 two quantities | 0.010 | 0.080 | 0.154 | 0.298 (0.24-0.36) | 0.355 |
| S2 form | 0.020 | 0.080 | 0.125 | 0.287 (0.23-0.35) | 0.355 |

Over all cells (rank-1 / rank-20 / EER):

| cell | S0 | S1 | S2 |
|---|---|---|---|
| KUPA copy -> essay, windows | 0.165 / 0.665 / 0.206 | 0.015 / 0.299 / 0.389 | 0.021 / 0.275 / 0.385 |
| KUPA essay -> copy, sentences | 0.185 / 0.695 / 0.205 | 0.010 / 0.265 / 0.355 | 0.030 / 0.270 / 0.355 |
| KUPA essay -> copy, windows | 0.185 / 0.695 / 0.206 | 0.025 / 0.295 / 0.389 | 0.035 / 0.260 / 0.385 |
| Aalto 1-7 -> 8-15, sentences | 0.270 / 0.875 / 0.105 | 0.035 / 0.406 / 0.345 | 0.050 / 0.443 / 0.300 |
| Aalto 1-7 -> 8-15, windows | 0.260 / 0.840 / 0.116 | 0.085 / 0.429 / 0.335 | 0.068 / 0.390 / 0.291 |

## The number of persons (mean over 200 subsets; top1 with its ratio to chance 1 / N)

KUPA copy -> essay, sentence unit:

| stage | N = 25 | N = 50 | N = 100 | N = 200 |
|---|---|---|---|---|
| S0 top1 | 0.398 (x9.9) | 0.300 (x15.0) | 0.219 (x21.9) | 0.160 (x32.0) |
| S0 rank_auc | 0.896 | 0.896 | 0.894 | 0.893 |
| S1 top1 | 0.118 (x3.0) | 0.068 (x3.4) | 0.035 (x3.5) | 0.010 (x2.0) |
| S1 rank-5 (chance 5 / N) | 0.469 (0.20) | 0.293 (0.10) | 0.156 (0.05) | 0.080 (0.025) |
| S1 rank_auc | 0.712 | 0.712 | 0.713 | 0.713 |
| S2 top1 | 0.112 (x2.8) | 0.070 (x3.5) | 0.037 (x3.7) | 0.020 (x4.0) |

Over all cells, the mean rank_auc of a stage moves by at most 0.014 over N = 25-200 (KUPA essay -> copy, windows, S2
0.690 to 0.676); top1 is 2.0-5.0 times chance at S1 and 2.8-7.0 at S2 in KUPA, 5.1-17.0 and 5.0-13.5 in Aalto, and
9.9-54.0 at S0.

Dropped features in subsets: in Aalto S1 a feature is constant within 88-89 of the 200 subsets of 25 persons and 23-37
of those of 50 (q = 1 for most Aalto persons, v6 reading 5); none elsewhere and none at N = 200.

## Readings (Claude)

1. With the field's measures the picture of v6 holds: from times to the two quantities the person is lost most (KUPA
   rank-1 0.16-0.19 to 0.01-0.03, rank-20 0.67-0.70 to 0.27-0.30, EER 0.21 to 0.36-0.39), and what is left after the
   two quantities is above chance but small: at N = 200 the true person is the nearest in 1-3% of cases (chance 0.5%)
   and among the 20 nearest in 27-30% (chance 10%). The form (S2) is about the same as S1 (EER equal in KUPA
   sentences).
2. rank_auc does not depend on the number of persons (within 0.014), while top1 falls with N; at S1 its ratio to
   chance stays between 2 and 5 in KUPA. So "above chance" in v6 is not an effect of having 200 persons: in a group of
   25, an attacker with a registration record of another task finds the true person first in about 12% of cases with
   the two quantities (chance 4%) and in about 40% with the times.
3. These are rates of one attack (nearest neighbour on standardised summaries, with a registration record of the same
   person); they bound from below what such an attacker could do with these observables, not what the observables
   carry. Impersonation is not counted.
4. Predictions written before running: (a) rank_auc flat within 0.03 held; (b) S1 top1 about 2 times chance at N = 200
   (2.0 in the main cell) and 1-5 times for N = 25-100 held in KUPA, S0 above 20 times held; (c) KUPA S1 rank-20 below
   0.4 held (0.27-0.30); (d) EER at S1 in KUPA 0.25-0.40 held (0.355-0.389), at S0 below 0.20 did not hold (0.205-0.206),
   Aalto below KUPA at every stage held.

Limits: as v6 and limits_written_before in the conditions (browser recordings, mostly English learners in KUPA, one
sitting, S0 a summary, Aalto within one session; the subsets are drawn from the same 200 persons and overlap; one
attack). The EER is the discrete one of the conditions. Readings are Claude's; the researcher has not reviewed them.

Licences of what is derived here: counts from KUPA-KEYS (Velentzas et al., LREC-COLING 2024) are under CC BY-NC-SA
4.0; those from the 136M Keystrokes dataset (Dhakal, Feit, Kristensson, Oulasvirta, CHI 2018) are for non-commercial
use with attribution to the authors.
