# Notes on prototype v2 (read after the run, 2026-10-31)

Record: results/prototype_v2/record.json (git commit 5502d13, not dirty). Median over 10 seeds of the share of
[0, 1] spanned by the middle half of p / q, external scale:

| text | window | from_text | shuffled | regimes (matched) |
|---|---|---|---|---|
| seasons | 10 s | 0.107 / 0.273 | 0.116 / 0.294 | 0.226 / 0.293 |
| rule | 10 s | 0.113 / 0.247 | 0.120 / 0.237 | 0.214 / 0.284 |
| elements | 10 s | 0.096 / 0.302 | 0.110 / 0.288 | 0.209 / 0.369 |
| features | 10 s | 0.114 / 0.220 | 0.132 / 0.216 | 0.250 / 0.279 |
| seasons | 30 s | 0.041 / 0.125 | 0.055 / 0.144 | 0.085 / 0.127 |
| rule | 30 s | 0.059 / 0.084 | 0.058 / 0.172 | 0.104 / 0.124 |
| elements | 30 s | 0.048 / 0.128 | 0.042 / 0.157 | 0.100 / 0.142 |
| features | 30 s | 0.064 / 0.115 | 0.062 / 0.125 | 0.128 / 0.163 |

Matching of the regimes (10 s windows, mean over seeds): rest share per window within 0.04 of the series from the
text (0.133-0.200 against 0.134-0.179), duration 2-13 s longer, median speed within 4 ms. Windows without a speed:
0-2 of about 250 per text and generator (p = 0, flagged).

Readings (Claude, after seeing seasons_external.png and rule_external.png; the others not looked at):
1. 10 s windows double the spread of p and of q against 30 s, and pale tiles (a rest in the window) come often.
2. The series from a text and its shuffled gaps still hardly differ, in the counts and to the eye. The text's
   structure is regular (a slower gap at the start of every word of about five letters), so its speed hardly varies
   between windows, and its rests at sentence and paragraph starts are no more spread over the windows than shuffled
   rests are. The texts do not show through the windows; the order the text gives the gaps is not visible here.
3. The matched regimes spread p about twice as wide as the text (0.21-0.25 against 0.10-0.11 at 10 s); coarse
   tiles appear only in their row. With the distribution of gaps the same, what shows is the regimes' runs (mean
   fast run 30 short gaps, assumed): clustering of slow gaps, a setting of the generator.
4. The 30 s counts repeat v1 (p 4-6 %), with the matched regimes now narrower than v1's.
Limits: synthetic keystrokes only; the text timing at boundaries and the run length of the regimes are assumptions.
