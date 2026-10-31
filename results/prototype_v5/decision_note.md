# Decision note: prototype v5 (rows added to the notes of v1, v3 and v4, results/prototype_v1/decision_note.md,
# results/prototype_v3/decision_note.md and results/prototype_v4/decision_note.md)

Columns as in v1. "After" = after seeing the results of v1 to v4; every row was committed in the conditions before
v5's first run (8fe045c), the code fixes before it too (f7a659f).

| Quantity | Value | Reason | Alternatives by then | Who | Before / after | Images |
|---|---|---|---|---|---|---|
| Image numbers (v5) | depth (standard deviation of the grey) and fineness (radial centroid of the power spectrum) of each tile, compared separately | v4's pixel-by-pixel difference measured the phase of the lattice (v4 reading 4) | the radial spectrum as a whole as well; the difference of p and q, the image numbers only to check the fields | researcher | after v4 | none (v4's tiles) |
| Baselines (v5) | between fields (as v4) and between seed indices k and k + 1 of the same row | without the phase the field difference was expected near zero, and any row difference would then read as telling apart | fields only; seeds only | researcher | after v4 | none |
| Rows, seeds, units (v5) | as v4; image numbers for the sentence and 10 s tiles only; no figure, v4's figures checked by SHA-256 | only the counts change, so they compare directly with v4 | as v4 with figures; image numbers for 30 s windows too | researcher | after v4 | none |
| Prediction (v5) | fields small against seeds; reordered and regimes clearly above seeds in fineness, none in depth | v4's spreads | none | Claude | after v4 | none |
