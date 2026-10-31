# Notes on prototype v6b (read after the run, 2026-10-31)

Record: results/prototype_v6b/record.json (SHA-256 94d3feda2b4a499738e2711821f930a88443ddca05e02c31782ad564c9cacb6f;
git commit e74c880, not dirty). Conditions: works/conditions/prototype_v6b.json, committed in 8bec59a and amended before
the first run in b9bd087 (after the Codex review of the code) and efcce01 (after the code review; that commit also holds
code fixes). Code 8892943, fixes d3b55dd, efcce01 and e74c880, all before the run.

Persons: 200 KUPA persons not used in v6 (437 ids of v6's permutation looked at, v6's first 217 skipped and checked
against v6's ids; overlap 0). Count persons: KUPA essay sentences 178 (22 with fewer than 8 sentences), every other cell
200; LOC on 156-200 per row (persons whose parts have every pause label).

The question (from the researcher): with a fairer baseline and observables chosen to show it, does an operation exceed
the person's own variation? The features and the odd/even baseline were explored on v6's persons first
(.steering/20261031-v6b-own-variation, record 1); v6b confirms them on new persons.

## Main cells: KUPA essay, sentence unit, baseline odd/even (share of persons with change > own, CI, reading)

| row | S1 (image) | S2 (image) | LOC (stage of times) |
|---|---|---|---|
| inside | 0.00 (0.00-0.00) below 0.60 | 0.00 below 0.60 | 0.60 (0.53-0.68) undecided |
| shuffled | 0.41 (0.34-0.48) below 0.60 | 0.48 (0.40-0.55) below 0.60 | 0.93 (0.88-0.97) above 0.60 |
| shuffled in halves | 0.38 (0.31-0.45) below 0.60 | 0.39 (0.31-0.46) below 0.60 | 0.95 (0.91-0.98) above 0.60 |

SEQ (the order of the tiles, baseline halves): 0.00 / 0.46 (0.38-0.53) / 0.47 (0.40-0.54), all below 0.60.

## Over all cells (essay and copy, sentence and 10 s windows, both baselines)

- Image (S1, S2, SEQ): every row in every cell reads "below 0.60". Highest: essay windows, odd/even, shuffled, S1 0.49
  (0.42-0.56) and S2 0.49 (0.43-0.56).
- LOC, shuffled and shuffled in halves: every cell "above 0.60", shares 0.92-0.97, median change / base 1.85-2.67.
- LOC, inside: 0.52-0.66, "undecided" in 6 cells and "below 0.60" in 2 (copy, halves); the lower end is above 0.5 in four
  cells (copy sentences odd/even 0.66, 0.59-0.72; essay sentences odd/even 0.60, 0.53-0.68; essay windows odd/even 0.59,
  0.52-0.66; essay windows halves 0.58, 0.51-0.65): detectable in most persons there, but not above 0.60.
- LOC, after: 0.01-0.10, below 0.60 (moving the counting of a gap does not move the gaps).
- Same persons: S1, S2 and SEQ on the persons of LOC give the same shares within 0.02.

## Readings (Claude)

1. With the odd/even baseline and the order of the tiles, the image still does not show the operations above the
   person's own variation: the answer to the question is no in the image, for every operation and cell. The odd/even
   baseline raises the shares a little (essay windows, shuffled, S1 from 0.35 with halves to 0.49), as in the
   exploration, but not past 0.5.
2. At the stage of times, the place of the pauses (Conijn et al. 2019's locations: sentence, subsentence, word, within
   word, with the lag-1 correlation of Condrey 2026) shows shuffling clearly above the person's own variation in every
   cell (0.92-0.97; the change is about twice the own difference). The same persons, the same parts: what differs is
   the observable. Shuffling leaves a mark that the compression to the two quantities removes.
3. Moving the rest inside its sentence ("inside") is invisible in the image by construction (0 per sentence) and only
   weakly visible at the stage of times (0.52-0.66, mostly undecided). The prediction "above 0.60 for inside in the
   sentence unit" did not hold; it rested on the exploration, where the share was 0.69-0.82 with a LOC of five
   numbers (without the subsentence label and the lag-1). Inside moves one gap per sentence, so its mark is small
   against the person's own variation even before the image.
4. Predictions: LOC above 0.60 for the shuffles held in every cell; the image below 0.60 or undecided held (all below);
   inside above 0.60 did not hold.

Limits: one dataset (KUPA), mostly English learners, one sitting; the features and the baseline were chosen after
seeing v6 and the exploration on v6's persons (confirmed here on other persons of the same dataset only); LOC reads the
kind of key and follows Conijn et al.'s locations roughly (the gap before the first letter of a sentence); many cells
read descriptively. Readings are Claude's; the researcher has not reviewed them.

Licence: counts derived from KUPA-KEYS (Velentzas et al., LREC-COLING 2024) are under CC BY-NC-SA 4.0.
