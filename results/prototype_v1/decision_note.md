# Decision note: prototype v1 (form of Table 1 of the works of words)

Columns: quantity | value | reason | alternatives tried by then | who decided | set before or after seeing what it produced | images that used it.
"Before" = before any synthetic keystroke series of these texts was generated (conditions committed at c22cd74).
Steering records: ~/Affective-Tactility-Keystroke/.steering/20261031-keystroke-design and 20261031-synthetic-prototype (not tracked).

| Quantity | Value | Reason | Alternatives by then | Who | Before / after | Images |
|---|---|---|---|---|---|---|
| Form of the work | one tile per 30 s window, in time order | one typist's flow (decision 1, design) | comparing typists or scenes; both | researcher | before | all |
| How values enter the rule | form of the rule only: p, q from keystrokes, not called V, A | avoid implying affect is read from typing (decision 2, design) | as V, A; f, s, gamma directly; open quantities | researcher (option proposed by Codex) | before | all |
| Two keystroke quantities | speed (median gap < 2 s) and rest (share of time in gaps >= 2 s) | timing only, no key kinds (decision 3, design) | speed and deletions; bursts and rest | researcher | before | all |
| Direction | faster -> finer (p); more rest -> shallower (q = 1 - rest share) | rest reads as a still surface | rest -> deeper; swapped | researcher | after seeing four corner tiles without keystroke values; before any keystroke value | all |
| Scale of speed | log, 481.03 ms -> p = 0, 121.70 ms -> p = 1, saturated | slow and fast typists of Dhakal et al. 2018 | min-max within series (kept as reference) | researcher | before | *_external.png |
| Reference scale | min-max per series | to see how a within-work scale widens differences | none | researcher | before | *_minmax.png |
| Texts | four English texts written for the prototype (seasons, rule, elements, features) | researcher's topics; drafted by Claude | none | researcher accepted the draft | before | all |
| Text timing | lognormal medians 240 / 735 / 2697 / 5394 ms, sigma 0.4 / 0.5 / 0.5 / 0.5 | Dhakal 2018; Baaijen et al. 2012 via Van Waes et al. 2021; paragraph and sigmas assumed | none | Claude, shown to researcher | before | rows 1-2 |
| Regimes | fast 240 ms, slow 735 ms, runs 30 / 10 presses, rest chance from the text's structure | assumed | none | Claude | before | row 3 |
| Window | 30 s, rest threshold 2 s, last partial window dropped | GSE features.rs | none | Claude | before | all |
| Field | seed 42, 4 octaves, ampsum, s_max 0.3 | as the works of words | none | Claude | before | all |
| Seeds of keystrokes | 10 per text and generator; figures show seed index 0 | all recorded, no selection | none | Claude | before | all |
