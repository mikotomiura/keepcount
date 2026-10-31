# Notes on prototype v1 (read after the run, 2026-10-31)

Record: results/prototype_v1/record.json (git commit c22cd74, not dirty). Counts fixed before running: the share of
[0, 1] spanned by the middle half of p and of q over the windows (median over 10 seeds).

| text | generator | middle half p | middle half q |
|---|---|---|---|
| seasons | from_text / shuffled / regimes | 0.054 / 0.066 / 0.242 | 0.151 / 0.168 / 0.109 |
| rule | from_text / shuffled / regimes | 0.055 / 0.053 / 0.131 | 0.150 / 0.147 / 0.086 |
| elements | from_text / shuffled / regimes | 0.059 / 0.057 / 0.148 | 0.130 / 0.177 / 0.123 |
| features | from_text / shuffled / regimes | 0.052 / 0.068 / 0.203 | 0.089 / 0.180 / 0.128 |

Readings (Claude, after seeing seasons_external.png, seasons_minmax.png and features_external.png; rule and elements not looked at):
1. Under the external scale, the series from a text and its shuffled gaps hardly differ in p (middle half 5-7 %
   of [0, 1]) and to the eye: with about 70 presses per 30 s window (median over seeds, 64-74), the structure of the
   text averages out, and the tiles differ mainly where a long rest (a paragraph start) falls. In q the shuffled
   gaps spread wider in two texts of four (features 0.089 against 0.180, elements 0.130 against 0.177): the text
   keeps its rests at sentence and paragraph starts, while shuffling scatters them over the windows.
   (Corrected after the code review: the first reading said "about 100 presses" and "hardly differ in the counts".)
2. The regimes widen p (slow runs reach p = 0): what shows is the generator's setting (slow 735 ms, runs of 10).
3. The min-max reference widens small differences, as expected.
Limits: synthetic keystrokes only; the boundary values of the text timing are not grounded in a source read in full.

Checked after the reviews (Codex via run-codex-consult.sh, 2026-10-31):
- No window lacked a speed, and the series from a text and its shuffled gaps had the same number of full windows in
  every seed (the edge cases Codex raised did not occur in v1).
- The regimes were not matched to the series from a text: mean rest share per window 0.125-0.158 against
  0.141-0.179, and durations 7-30 s longer (they use no paragraph rests and their rests are always 2 s or more).
  So reading 2 compares generators whose distributions differ; decision 5 (compare with the distributions aligned)
  is not met by v1.
