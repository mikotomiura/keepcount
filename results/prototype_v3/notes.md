# Notes on prototype v3 (read after the run, 2026-10-31)

Record: results/prototype_v3/record.json (git commit acd4f6c, not dirty). One tile per sentence (10-11 per text).

Counts fixed before running (median over 10 seeds of the share of [0, 1] spanned by the middle half of p / q,
external scale):

| text | from_text | shuffled | regimes (matched) |
|---|---|---|---|
| seasons | 0.056 / 0.256 | 0.071 / 0.237 | 0.146 / 0.247 |
| rule | 0.062 / 0.155 | 0.058 / 0.175 | 0.125 / 0.189 |
| elements | 0.063 / 0.235 | 0.077 / 0.212 | 0.098 / 0.204 |
| features | 0.062 / 0.105 | 0.063 / 0.186 | 0.120 / 0.163 |

Exploratory count, chosen after seeing seasons_external.png (not fixed before running): Pearson correlation of
ln(presses of a sentence) with its q, median over seeds [min, max]:

| text | from_text | shuffled | regimes |
|---|---|---|---|
| seasons | +0.72 [+0.55, +0.87] | -0.21 [-0.49, +0.22] | -0.25 [-0.51, -0.00] |
| rule | +0.25 [+0.01, +0.65] | +0.02 [-0.70, +0.32] | -0.07 [-0.35, +0.72] |
| elements | +0.79 [+0.67, +0.90] | -0.28 [-0.59, +0.94] | -0.19 [-0.36, +0.36] |
| features | +0.19 [-0.05, +0.52] | +0.01 [-0.64, +0.71] | -0.09 [-0.48, +0.62] |

Readings (Claude, after seeing seasons_external.png and elements_external.png; the others not looked at):
1. The spreads fixed before running do not separate the series from the text from its shuffled gaps (as in v1, v2).
2. The tiles do: in the series from the text, short sentences and sentences that open a paragraph are pale (seasons:
   "Summer is loud.", "Winter is quiet."), and q rises with sentence length in every text, strongly where sentence
   lengths vary (seasons, elements). The shuffled series and the regimes show no positive relation; their medians
   are mostly weakly negative (-0.28 to +0.02), as expected when rests fall at random and a longer stretch is more
   likely to hold one.
3. What shows is mechanical. q = 1 - (sum of the rests in the sentence) / (the sentence's time). The gap before a
   sentence is a rest (2 s or more) with chance 0.725 at a sentence start and 0.976 at a paragraph start (lognormal
   medians 2697 and 5394 ms, sigma 0.5; the first taken from a mixture, the second assumed), while a gap at a word
   start inside a sentence is a rest with chance 0.023. So most sentences carry one rest, before them, and its weight
   is larger in a short sentence. The tiles show sentence length and paragraph starts as the synthetic timing
   places them; whether people's rests before sentences behave so is not tested here.
   (Corrected after the code review of v3: the first version said every sentence but the first carries one rest.)
4. p (speed) stays narrow in the series from the text (middle half about 6 %): the regular word structure does not
   vary speed between sentences.
Limits: synthetic English keystrokes only; one seed per figure row; the exploratory count was chosen after seeing.

Checked after the Codex review of v3 (2026-10-31):
- The first sentence has no rest before it, so its q is always 1 (the first tile of each row is deep). Without the
  first sentence the exploratory correlation of the series from the text holds: seasons +0.74, rule +0.30,
  elements +0.93, features +0.26 (shuffled -0.38 to +0.05, regimes -0.26 to -0.03; medians over seeds).
- In every series the times per sentence add up to the duration and the presses per sentence to the presses.
