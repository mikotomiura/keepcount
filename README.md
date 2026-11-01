# KeepCount: conditions, code, records and history

This is the archive of the article "KeepCount: A Stage-by-Stage Audit of What Images of Behavioral Data Keep, with
Keystroke Tiles as a Worked Example" (Mikoto Miura and Yuusuke Harada; preprint https://doi.org/10.5281/zenodo.23858734,
archive https://doi.org/10.5281/zenodo.23858736).

The audited transform borrows the form of the rule from affect to tactile relief (Harada, *Affective Tactility*,
Research Square, 2026, https://doi.org/10.21203/rs.3.rs-8696852/v1) and feeds it two quantities of keystroke timing.
The rule's generator is `atact`, pinned to a commit of affective-relief-audit (https://doi.org/10.5281/zenodo.23681773).

## How it was made

Every computation ran a script on conditions committed to git before the script was first run, and each record holds
the SHA-256 of its conditions, script and inputs and the git commit.

- `works/conditions/prototype_*.json`: the conditions of each run, committed before its code.
- `works/prototype_*.py`, `works/_common.py`, `src/keystroke.py`: the code of each run.
- `results/prototype_*/record.json` and `notes.md`: the record of each run and the notes on it.
- `tests/`: the tests of the code.
- `works/texts/`: the texts, written by the authors, from which runs v1 to v5 made synthetic keystrokes.

Runs v1 to v5 use synthetic keystrokes only. Runs v6 to v6j count three public keystroke datasets (KUPA-KEYS,
136M Keystrokes and the essays of the Kaggle competition Linking Writing Processes to Writing Quality), which are not
redistributed here; the records hold counts and summaries over persons (the minimum and maximum of a summary are
single persons' values), and the persons' ids only as a count and a SHA-256. References to `.steering/` in the
conditions, notes and commit messages point to working notes that are not archived.

## What differs from the working repository

Six figures of run v6 (the tiles and scatter plots of individual persons' records,
`results/prototype_v6/{kupa_copy,kupa_essay,aalto_session}_{tiles,scatter}.png`) were removed from every commit, since
the article publishes only counts and summaries over persons; `results/prototype_v6/record.json` still gives their SHA-256. The
commits after them therefore differ from the commits named in the article and in the records; `commit-map.tsv` maps
each original commit to the archived one. The last commit adds this README, the licences and the map.

## Licences

The code is under the MIT License (LICENSE); the conditions, records, notes and other documents under CC BY-NC-SA 4.0
(LICENSE-docs), which also states the terms of the datasets.
