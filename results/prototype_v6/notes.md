# Notes on prototype v6 (read after the run, 2026-10-31)

Record: results/prototype_v6/record.json (SHA-256 4672ad73eb24d0716ff52ec31f9e000b0027c4663949668e0cfe598568dc11d3;
git commit f07a394, not dirty). Conditions: works/conditions/prototype_v6.json, committed in 06b46c2 and amended twice
before the first run (dadc78d: KUPA ids are strings; 618ce57: eligibility, draw order and count 1 persons after the code
reviews). Code: 1fde6db, fixes from the code review and the Codex review in f07a394, both before the run. All counts and
readings below were fixed in the conditions before running; "below 0.60" etc. are the two-axis readings of decision 4
and 8 of .steering/20261031-compression-audit-v6.

Persons: 200 per dataset (KUPA 217 ids looked at, Aalto 227). Count 1 persons: KUPA essay sentences 177 (at least 8
sentences), Aalto windows 192, every other cell 200.

## Count 2: identification (rank_auc with 95% CI; main standardisation unless stated)

Main cell, KUPA copy -> essay, sentence unit:

| stage | rank_auc | CI | top-1 (chance 0.005) | reading |
|---|---|---|---|---|
| S0 times (7 quantiles of ln gap) | 0.893 | 0.873-0.913 | 0.160 | detectable, above 0.60 |
| S1 two quantities (p, q) | 0.713 | 0.675-0.746 | 0.010 | detectable, above 0.60 |
| S2 form (depth, fineness) = image proxy | 0.690 | 0.652-0.726 | 0.020 | detectable, above 0.60 |
| S0 - S1 | 0.181 | 0.146-0.216 | | |
| S1 - S2 | 0.023 | -0.007-0.055 | | |

Other cells (S0 / S1 / S2 rank_auc):

| cell | unit | main | registration scale | ranks |
|---|---|---|---|---|
| KUPA copy -> essay | window 10 s | 0.895 / 0.711 / 0.685 | 0.882 / 0.653 / 0.617 | 0.893 / 0.711 / 0.699 |
| KUPA copy -> essay | sentence | (above) | 0.882 / 0.632 / 0.611 | 0.893 / 0.712 / 0.692 |
| KUPA essay -> copy | sentence | 0.895 / 0.710 / 0.712 | 0.878 / 0.640 / 0.651 | 0.892 / 0.708 / 0.711 |
| KUPA essay -> copy | window 10 s | 0.895 / 0.680 / 0.676 | 0.877 / 0.579 / 0.577 | 0.891 / 0.708 / 0.714 |
| Aalto 1-7 -> 8-15 (one session) | sentence | 0.958 / 0.766 / 0.799 | 0.957 / 0.767 / 0.798 | 0.956 / 0.752 / 0.818 |
| Aalto 1-7 -> 8-15 (one session) | window 10 s | 0.952 / 0.776 / 0.786 | 0.952 / 0.775 / 0.790 | 0.952 / 0.748 / 0.798 |

Every main cell is "detectable, above 0.60" at every stage; every permutation p is 1/1001. The only cell whose CI
reaches below 0.60 is KUPA essay -> copy, windows, registration scale, S1 and S2 (0.537-0.621 and 0.535-0.621:
detectable, undecided).

## Count 1: operations against the person's own halves (share of persons with change > base)

Every row in every cell and stage reads "below 0.60" with the lower end at or below 0.5 (not detected): no operation
moves a stage more than the person's own difference between the two halves of the record in most persons. Share vs
another person is at most 0.105. Selected rows (S1; share vs own with CI, median change / base):

| cell | row | S1 share vs own | change / base |
|---|---|---|---|
| KUPA essay sentence | after | 0.161 (0.110-0.220) | 0.43 |
| KUPA essay sentence | none | 0.127 (0.082-0.178) | 0.39 |
| KUPA essay sentence | inside | 0.003 | 0 |
| KUPA essay sentence | shuffled | 0.412 (0.339-0.486) | 0.88 |
| KUPA essay sentence | shuffled in halves | 0.410 (0.339-0.480) | 0.86 |
| KUPA copy sentence | none | 0.070 (0.040-0.102) | 0.19 |
| KUPA copy sentence | shuffled | 0.263 (0.203-0.320) | 0.67 |
| KUPA essay window 10 s | shuffled | 0.297 (0.235-0.365) | 0.75 |
| Aalto sentence | shuffled | 0.475 (0.410-0.543) | 0.97 |
| Aalto window 10 s | shuffled | 0.385 (0.320-0.451) | 0.87 |

## Readings (Claude)

1. The step from times to the two quantities is where the person is lost most: rank_auc falls by 0.18-0.21 in every
   cell (CI clear of 0), top-1 from 16-27% to 1-9%. What survives the two quantities is still above the smallest
   difference fixed before running (0.60) in every main cell: the image carries the person, less than the gaps do.
   This is "with this observable and this attack", not a statement of anonymity.
2. From the two quantities to the form nothing systematic is lost in KUPA (S1 - S2 within -0.01 to 0.03), as v5's
   reading 2 predicts (depth and fineness are nearly monotone images of p and q). In Aalto the form tells the persons
   apart a little better than p and q (sentence 0.799 against 0.766, difference -0.033, CI -0.059 to -0.008; with ranks
   -0.066). A per-tile function of p and q cannot add information about a tile; the gain is in how the nearest-neighbour
   distance weighs the medians and spreads after the nonlinear map, so S1 - S2 mixes loss with a change of metric
   (Codex's point 5, written before running).
3. Part of S1's identification across tasks depends on correcting each task as a group. With the registration task's
   scale applied to both, KUPA S1 falls from 0.68-0.71 to 0.58-0.65; Aalto (same session) does not move. Persons keep
   their rank within a task better than their absolute speed and rest between copying and composing.
4. The operations on a series move the image less than the person's own variation between two halves of one record,
   and much less than another person. Shuffling all gaps comes closest (change / base 0.67-0.97); in the KUPA essay,
   shuffling only within each half moves S1 as much as shuffling across the whole essay (0.410 against 0.412), so
   there the order inside a half matters as much as moving gaps between halves. Moving the rest inside its sentence
   ("inside") changes nothing per sentence, as by construction under the counting "before" (v5 reading 4). The
   prediction that "none" and "shuffled" would move S1 more than own variation in most persons did not hold.
5. The scatter figures show where the external scale of v4 compresses human data: in Aalto almost every person has
   median q = 1 (within the presented sentences there is hardly a rest of 2 s, the gaps between sentences being dropped
   by design), and in KUPA copy many persons sit at p = 1 (median gap faster than Dhakal's fast typists' 121.7 ms). For
   those persons one of the two quantities is flat before the image is made. 21% of KUPA essay windows have no gap
   below 2 s (no speed, p = 0).
6. Predictions written before running: (2) held (S0 above 0.8, S1 detectable but lower, S2 within a few hundredths of
   S1, Aalto above KUPA, the two directions similar); (1) did not hold for "none" and "shuffled" (see 4); "every row
   moves the stages less than switching to another person" held.

Post-run diagnostic (not in the conditions): the shares of "inside" and "shuffled in halves" above 0 where the median
change is 0 come from persons whose two halves give identical S1 or S2 vectors (base 0: KUPA copy sentence S1 8, S2 3;
essay 1), counted as ties 1/2; these are persons whose p and q saturate (reading 5).

Limits: see limits_written_before in the conditions (browser recordings in both datasets; mostly English learners in
KUPA; one sitting, not across days; S0 is a summary, not the raw series; many cells read descriptively). The CIs are
conditional on the 200 registration persons. Count 1 in KUPA essay sentences leaves out 23 persons with short essays.
The image itself (what a viewer sees) is not audited; S2 stands for it. Readings are Claude's; the researcher has not
reviewed them.

Licences of what is derived here: figures and counts from KUPA-KEYS (Velentzas et al., LREC-COLING 2024) are under CC
BY-NC-SA 4.0; those from the 136M Keystrokes dataset (Dhakal, Feit, Kristensson, Oulasvirta, CHI 2018) are for
non-commercial use with attribution to the authors.
