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
1. Under the external scale, the series from a text and its shuffled gaps hardly differ, in the counts and to the
   eye: with about 100 presses per 30 s window, the structure of the text averages out, and the tiles differ mainly
   where a long rest (a paragraph start) falls. The texts' structure does not show at this window width.
2. The regimes widen p (slow runs reach p = 0): what shows is the generator's setting (slow 735 ms, runs of 10).
3. The min-max reference widens small differences, as expected.
Limits: synthetic keystrokes only; the boundary values of the text timing are not grounded in a source read in full.
